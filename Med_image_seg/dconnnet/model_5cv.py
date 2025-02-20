import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from matplotlib import pyplot as plt

# from apex import amp
from torch.optim import lr_scheduler
from torch.autograd import Variable

from libs.metric import metric
from libs.base_model import base_model
from Med_image_seg.dconnnet.util.lr_update import get_lr
from Med_image_seg.dconnnet.util.connect_loss import connect_loss, Bilateral_voting
from Med_image_seg.dconnnet.util.cldice import clDice

from libs.data_utils.dataset_chase import MyDataset_CHASE

from Med_image_seg.dconnnet.network import DconnNet
from torch.utils.tensorboard import SummaryWriter



def arguments():
    args = {
    '--num_class': 1, 
    '--use_SDL': False,
    '--lr_update': 'step',
    '--pretrained': None,
    '--img_size_H':960,
    '--img_size_W':960,
    '--lr_step': 12

}  
    return args


class dconnnet(base_model):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())

        print('#----------Creating logger----------#')
        ## make work_dir ,log_dir, res_dir, checkpoint_dir
        self.make_dir(parser)

        self.args = parser.get_args()

        global writer
        self.writer = SummaryWriter(self.work_dir + 'summary')

        global logger
        self.logger = self.get_logger('train', self.args.log_dir)

        
        print('#----------GPU init----------#')
        # self.set_seed(self.args.seed)
        self.set_cuda()
        torch.cuda.empty_cache()


        ######################################################################################
        """ Trainer """ 
        print('#----------Prepareing Model----------#')

        self.network = DconnNet(self.args.num_class).to('cuda')
        # self.step = 0
        self.save_args()


        ######################################################################################


        self.hori_translation = torch.zeros([1,self.args.num_class, self.args.img_size_W, self.args.img_size_W])

        for i in range(self.args.img_size_W-1):
            self.hori_translation[:,:,i,i+1] = torch.tensor(1.0)

        self.verti_translation = torch.zeros([1,self.args.num_class, self.args.img_size_H, self.args.img_size_H])

        for j in range(self.args.img_size_H-1):
            self.verti_translation[:,:,j,j+1] = torch.tensor(1.0)

        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()


        ######################################################################################

        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        """define loss"""
        # self.BceDiceLoss = BceDiceLoss().cuda()

        """define optimizer"""
        # self.optimizer = self.set_optimizer()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.lr)
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.lr, momentum=self.args.momentum,nesterov=self.args.nesterov, weight_decay=self.args.weight_decay)

        """define lr_scheduler"""
        # self.lr_scheduler = self.set_lr_scheduler()
        # self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_lr)


    def create_exp_directory(self, exp_id):
        # if not os.path.exists('models/' + str(exp_id)):
        #     os.makedirs('models/' + str(exp_id))

        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(self.args.log_dir, csv), 'w') as f:
            f.write('epoch, dice, Jac, clDice \n')

    def get_density(self, pos_cnt, bins = 50):
        ### only used for Retouch in this code
        val_in_bin_ = [[],[],[]]
        density_ = [[],[],[]]
        bin_wide_ = []

        ### check
        for n in range(3):
            density = []
            val_in_bin = []
            c1 = [i for i in pos_cnt[n] if i != 0]
            c1_t = torch.tensor(c1)
            bin_wide = (c1_t.max()+50)/bins
            bin_wide_.append(bin_wide)

            edges = torch.arange(bins + 1).float()*bin_wide
            for i in range(bins):
                val = [c1[j] for j in range(len(c1)) if ((c1[j] >= edges[i]) & (c1[j] < edges[i + 1]))]
                # print(val)
                val_in_bin.append(val)
                inds = (c1_t >= edges[i]) & (c1_t < edges[i + 1]) #& valid
                num_in_bin = inds.sum().item()
                # print(num_in_bin)
                density.append(num_in_bin)

            denominator = torch.tensor(density).sum()
            # print(val_in_bin)

            #### get density ####
            density = torch.tensor(density)/denominator
            density_[n]=density
            val_in_bin_[n] = val_in_bin
        # print(density_)

        return density_, val_in_bin_, bin_wide_


    def train_one_fold(self, train_loader, test_loader, exp_id, num_epochs=10):
        print('#----------Training----------#')

        ######################################################################################

        self.create_exp_directory(exp_id)

        if self.args.use_SDL:
            assert 'retouch' in self.args.dataset, 'Please input the calculated distribution data of your own dataset, if you are now using Retouch'
            device_name = self.args.dataset.split('retouch-')[1]
            pos_cnt = np.load(self.args.weights+device_name+'/training_positive_pixel_'+str(exp_id)+'.npy', allow_pickle=True)
            density, val_in_bin, bin_wide = self.get_density(pos_cnt)
            self.loss_func = connect_loss(self.args, self.hori_translation, self.verti_translation, density, bin_wide)
        else:
            self.loss_func = connect_loss(self.args, self.hori_translation, self.verti_translation)

        # net, optimizer = amp.initialize(self.network, self.optimizer, opt_level='O2')
        # net = self.network
        optimizer = self.optimizer

        best_p = 0
        best_epo = 0
        start_epoch = 1
        
        scheduled = ['CosineAnnealingWarmRestarts']
        if self.args.lr_update in scheduled:
            scheduled = True
            if self.args.lr_update == 'CosineAnnealingWarmRestarts':
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min = 0.00001)
        else:
            scheduled = False

        for epoch in range(start_epoch, self.args.epochs + 1):
        # for epoch in range(self.args.epochs):
                self.network.train()

                if scheduled:
                    scheduler.step()
                else:
                    curr_lr = get_lr(self.args.lr, self.args.lr_update, epoch, num_epochs, gamma=self.args.gamma, step=self.args.lr_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = curr_lr

                
                for i_batch, sample_batched in enumerate(train_loader):
                    X = Variable(sample_batched[0])
                    y = Variable(sample_batched[1])   ## y是gt ？？？

                    X = X.cuda()
                    y = y.float().cuda()
                    # print(X.shape,y.shape)

                    optimizer.zero_grad()
                    output, aux_out = self.network(X)

                    ## output/aux_out: c_map 八通道, y: target 一通道
                    loss_main = self.loss_func(output, y)    ## Lmain
                    loss_aux = self.loss_func(aux_out, y)    ## Lprior SDE的辅助输出

                    loss = loss_main + 0.3*loss_aux

                    # with amp.scale_loss(loss, optimizer) as scale_loss:
                    #     scale_loss.backward()
                    loss.backward()
                    optimizer.step()

                    log_info = f'Epoch: {str(epoch)}, Iteration: {str(i_batch)} / {str(len(train_loader))}, Total: {loss.item():.3f}'
                    
                    # print('[epoch:'+str(epoch)+'][Iteration : ' + str(i_batch) + '/' + str(len(train_loader)) + '] Total:%.3f' %(
                    #     loss.item()))
                    print(log_info)
                    self.logger.info(log_info)
                
                ## 自己加的
                # self.lr_scheduler.step()
                  
                dice_p = self.val_epoch(self.network, test_loader, epoch, exp_id)
                if best_p<dice_p:
                    best_p = dice_p
                    best_epo = epoch
                    check_path = self.args.checkpoint_dir + '/' + str(exp_id)
                    if not os.path.exists(check_path):
                        os.makedirs(check_path)
                    torch.save(self.network.state_dict(), check_path + '/best_model.pth')
                    print('save best!')

                # if (epoch+1) % self.args.save_per_epochs == 0:
                #     torch.save(self.network.state_dict(), 'models/' + str(exp_id) + '/'+str(epoch+1)+'_model.pth')

                # if (epoch) % self.args.save_per_epochs == 0:
                #     torch.save(self.network.state_dict(), self.args.checkpoint_dir + str(exp_id) + '/'+str(epoch)+'_model.pth')

                # print('[Epoch :%d] total loss:%.3f ' %(epoch, loss.item()))
                log_info1 = f'Epoch: {epoch}, total loss: {loss.item():.3f}'
                print(log_info1)
                self.logger.info(log_info1)
                torch.cuda.empty_cache()

        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%03d, %0.6f \n' % (
                    best_epo,
                    best_p
                ))
            # writer.close()
        print('FINISH.')
        torch.cuda.empty_cache()




    def test(self):
        print('#----------Testing----------#')

        for exp_id in range(self.args.folds):
            if self.args.dataset == 'CHASE_DB1':

                overall_id = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
                test_id = overall_id[3*exp_id:3*(exp_id+1)]
                train_id = list(set(overall_id)-set(test_id))

                self.test_dataset = MyDataset_CHASE(self.args, train_root = self.args.data_path, pat_ls=test_id, mode='test')
            else:
                pass

            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)

            # best = 0.0
            # if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best_model.pth')):
            best_weight = torch.load(self.args.checkpoint_dir +'/'+str(exp_id) + '/best_model.pth', map_location=torch.device('cpu'))
            self.network.load_state_dict(best_weight)
            dice_test = self.test_epoch(self.network, self.test_loader, exp_id)

        torch.cuda.empty_cache()
            


    def train(self):
        for exp_id in range(self.args.folds):
            if self.args.dataset == 'CHASE_DB1':

                overall_id = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
                test_id = overall_id[3*exp_id:3*(exp_id+1)]
                train_id = list(set(overall_id)-set(test_id))

                self.train_dataset = MyDataset_CHASE(self.args, train_root = self.args.data_path, pat_ls=train_id, mode='train')
                self.test_dataset = MyDataset_CHASE(self.args, train_root = self.args.data_path, pat_ls=test_id, mode='test')

            # elif self.args.dataset == 'isic':
            #     trainset = ISIC2018_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='train', 
            #                                with_name=False)
            #     validset = ISIC2018_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='test',
            #                                    with_name=False)
            else:
                ####  define how you get the data on your own dataset ######
                pass

            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=6)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
        
            print("Train batch number: %i" % len(self.train_loader))
            print("Test batch number: %i" % len(self.test_loader))

            if self.args.pretrained:
                self.network.load_state_dict(torch.load(self.args.pretrained, map_location = torch.device('cpu')))
                self.network = self.network.cuda()

            self.train_one_fold(self.train_loader, self.test_loader, exp_id, self.args.epochs)


    def val_epoch(self, model, loader, epoch, exp_id):
        model.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):

                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                # name = test_data[2]

                X_test= X_test.cuda()
                y_test = y_test.long().cuda()

                output_test,_ = model(X_test)
                batch, channel, H, W = X_test.shape


                hori_translation = self.hori_translation.repeat(batch,1,1,1).cuda()
                verti_translation = self.verti_translation.repeat(batch,1,1,1).cuda()

                
                if self.args.num_class == 1:  
                    output_test = F.sigmoid(output_test)
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    pred = torch.where(class_pred>0.5,1,0)
                    pred,_ = Bilateral_voting(pred.float(),hori_translation,verti_translation)

                else:
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    final_pred,_ = Bilateral_voting(class_pred,hori_translation,verti_translation)
                    pred = get_mask(final_pred)
                    pred = self.one_hot(pred, X_test.shape)
                

                dice, Jac = self.per_class_dice(pred,y_test)
                
                if self.args.num_class == 1:
                    pred_np = pred.squeeze().cpu().numpy()
                    target_np = y_test.squeeze().cpu().numpy()
                    cldc = clDice(pred_np, target_np)
                    self.cldice_ls.append(cldc)

                ###### notice: for multi-class segmentation, the self.dice_ls calculated following exclude the background (BG) class

                if self.args.num_class>1:
                    self.dice_ls += torch.mean(dice[:,1:],1).tolist() ## use self.dice_ls += torch.mean(dice,1).tolist() if you want to include BG
                    self.Jac_ls += torch.mean(Jac[:,1:],1).tolist() ## same as above
                else:
                    self.dice_ls += dice[:,0].tolist()
                    self.Jac_ls += Jac[:,0].tolist()

                if j_batch%(max(1,int(len(loader)/5)))==0:
                    # print('[Iteration : ' + str(j_batch) + '/' + str(len(loader)) + '] Total DSC:%.3f ' %(
                    #     np.mean(self.dice_ls)))
                    
                    log_info = f'Iteration: {str(j_batch)} / {str(len(loader))}, Total DSC: {np.mean(self.dice_ls):.3f}'
                    print(log_info)
                    self.logger.info(log_info)
                    

            # print(len(self.Jac_ls))
            Jac_ls = np.array(self.Jac_ls)
            dice_ls = np.array(self.dice_ls)
            total_dice = np.mean(dice_ls)
            csv = 'results_'+str(exp_id)+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f \n' % (
                    (epoch),
                    total_dice,
                    np.mean(Jac_ls),
                    np.mean(self.cldice_ls)
                ))

            return np.mean(self.dice_ls)


    def test_epoch(self, model, test_loader, exp_id):
        model.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []
        # self.dsc_ls = []
        # self.iou_ls = []
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(test_loader):
                curr_dice = []
                img_test = Variable(test_data[0])   ## img
                gt_test = Variable(test_data[1])   ## gt
                # name = test_data[2]

                img_test= img_test.cuda()
                gt_test = gt_test.long().cuda()
                # print('img_test, gt_test')
                # print(img_test.size())   ## torch.Size([1, 3, 960, 960])
                # print(gt_test.size())    ## torch.Size([1, 1, 960, 960])
                # print(type(img_test))    ## <class 'torch.Tensor'>
                # print(type(gt_test))     ## <class 'torch.Tensor'>

                ## output_test: (B, 8, H, W)
                output_test, _ = model(img_test)
                batch, channel, H, W = img_test.shape
                # print(output_test.size())   ## torch.Size([1, 8, 960, 960])
                # print(type(output_test))    ## <class 'torch.Tensor'>
                # print('***********')

                hori_translation = self.hori_translation.repeat(batch,1,1,1).cuda()
                verti_translation = self.verti_translation.repeat(batch,1,1,1).cuda()

                if self.args.num_class == 1:  
                    output_test = F.sigmoid(output_test)
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    # print(class_pred.size())   ## torch.Size([1, 1, 8, 960, 960])
                    pred = torch.where(class_pred>0.5,1,0)
                    # print(pred.size())         ## torch.Size([1, 1, 8, 960, 960])
                    pred,_ = Bilateral_voting(pred.float(),hori_translation,verti_translation)
                    # print(pred.size())         ## torch.Size([1, 1, 960, 960])
                    # print(type(pred))          ## <class 'torch.Tensor'>

                else:
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    final_pred,_ = Bilateral_voting(class_pred,hori_translation,verti_translation)
                    pred = get_mask(final_pred)
                    pred = self.one_hot(pred, img_test.shape)

                ## save img
                # self.save_img(img_test, gt_test, pred, j_batch)
                self.save_img(gt_test, pred, j_batch, exp_id)
                
                dice, Jac = self.per_class_dice(pred,gt_test)
                
                if self.args.num_class == 1:
                    pred_np = pred.squeeze().cpu().numpy()
                    target_np = gt_test.squeeze().cpu().numpy()
                    cldc = clDice(pred_np, target_np)
                    self.cldice_ls.append(cldc)

                ###### notice: for multi-class segmentation, the self.dice_ls calculated following exclude the background (BG) class
                if self.args.num_class>1:
                    self.dice_ls += torch.mean(dice[:,1:],1).tolist() ## use self.dice_ls += torch.mean(dice,1).tolist() if you want to include BG
                    self.Jac_ls += torch.mean(Jac[:,1:],1).tolist() ## same as above
                else:
                    self.dice_ls += dice[:,0].tolist()
                    self.Jac_ls += Jac[:,0].tolist()
                    # self.dsc_ls += DSC[:,0].tolist()
                    # self.iou_ls += iou[:,0].tolist()


                if j_batch%(max(1,int(len(test_loader)/5)))==0:
                    # print('[Iteration : ' + str(j_batch) + '/' + str(len(loader)) + '] Total DSC:%.3f ' %(
                    #     np.mean(self.dice_ls)))
                    
                    log_info = f'Iteration: {str(j_batch)} / {str(len(test_loader))}, Total DSC: {np.mean(self.dice_ls):.3f}'
                    print(log_info)
                    self.logger.info(log_info)

            # print(len(self.Jac_ls))
            Jac_ls = np.array(self.Jac_ls)
            dice_ls = np.array(self.dice_ls)
            # dsc_ls = np.array(self.dsc_ls)
            # iou_ls = np.array(self.iou_ls)
            total_dice = np.mean(dice_ls)
            ## 由结果来看，dice==dsc,jac==iou

            csv = 'test_results_'+str(exp_id)+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%0.6f,%0.6f,%0.6f \n' % (
                    total_dice,
                    np.mean(Jac_ls),
                    # np.mean(dsc_ls),
                    # np.mean(iou_ls),
                    np.mean(self.cldice_ls)
                ))

            return np.mean(self.dice_ls)


    def per_class_dice(self, y_pred, y_true):
        smooth = 0.0001
        y_pred = y_pred
        y_true = y_true

        FN = torch.sum((1-y_pred)*y_true,dim=(2,3)) 
        FP = torch.sum((1-y_true)*y_pred,dim=(2,3)) 
        TN = torch.sum((1 - y_pred) * (1 - y_true), dim=(2, 3))
        TP = torch.sum(y_pred * y_true, dim=(2, 3))

        Pred = y_pred
        GT = y_true
        inter = torch.sum(GT* Pred,dim=(2,3))    ## 等于TP？？？


        union = torch.sum(GT,dim=(2,3)) + torch.sum(Pred,dim=(2,3)) 
        dice = (2*inter+smooth)/(union+smooth)
        Jac = (inter+smooth)/(inter+FP+FN+smooth)
        ## 下面是自己加的指标
        # DSC = (2*TP + smooth)/(2*TP + FP + FN + smooth)
        # iou = (TP+smooth)/(TP+FP+FN+smooth)


        return dice, Jac #, DSC, iou
    

    def one_hot(self,target,shape):

        one_hot_mat = torch.zeros([shape[0],self.args.num_class,shape[2],shape[3]]).cuda()
        target = target.cuda()
        one_hot_mat.scatter_(1, target, 1)
        return one_hot_mat
    
    def save_imgg(self, img, gt, pred, i):
        # img = img.float()
        # gt = gt.float()
        # pred = pred.float()

        ## 使用matplotlib显示彩色图像需要数据的维度为 【width, height, channel】,使用permute函数交换维度
        img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img = img / 255. if img.max() > 1.1 else img
        # img = img / 255. 

        # pred = pred.squeeze(1).cpu().detach().numpy()
        pred = pred.squeeze().cpu().numpy()

        # gt = gt.squeeze(1).cpu().detach().numpy()
        gt = gt.squeeze().cpu().numpy()


        # gt = np.where(np.squeeze(gt, axis=0) > 0.5, 1, 0)
        # pred = np.where(np.squeeze(pred, axis=0) > 0.5, 1, 0) 

        save_path = self.args.res_dir

        plt.figure(figsize=(15,15))

        plt.subplot(3,1,1)
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(3,1,2)
        plt.imshow(gt, cmap= 'gray')
        plt.axis('off')

        plt.subplot(3,1,3)
        plt.imshow(pred, cmap = 'gray')
        plt.axis('off')

        plt.savefig(save_path + '/' + str(i) +'.png')
        plt.close()

    def save_img(self,gt,pred,iter,exp_id):

        save_path = os.path.join(self.args.res_dir, str(exp_id))
        self.makedirs(save_path)

        pred_array = pred.squeeze(0).squeeze(0).cpu().numpy()
        plt.figure(figsize=(9.6,9.6),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(pred_array, cmap='gray')  # 如果是灰度图可以指定 cmap='gray'，如果是彩色图无需指定 cmap
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(save_path +'/'+ 'result'+ str(iter) +'.png')
        plt.close()


        gt_array = gt.squeeze(0).squeeze(0).cpu().numpy()
        plt.figure(figsize=(9.6,9.6),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(gt_array, cmap='gray')  
        plt.axis('off') 
        plt.savefig(save_path +'/'+ 'gt'+ str(iter) +'.png')
        plt.close()



def get_mask(output):
    output = F.softmax(output,dim=1)
    _,pred = output.topk(1, dim=1)
    #pred = pred.squeeze()
    
    return pred






