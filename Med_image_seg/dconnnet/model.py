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


from Med_image_seg.dconnnet.network import DconnNet


def arguments():
    args = {
    '--num_class': 1, 
    '--use_SDL': False,
    '--lr_update': 'step',
    '--pretrained': None,
    '--lr_step': 12,
    '--img_size_W': 512,
    '--img_size_H': 512

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
        global logger
        self.logger = self.get_logger('train', self.args.log_dir)


        
        print('#----------GPU init----------#')
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_id
        # self.set_seed(self.args.seed)
        self.set_cuda()
        torch.cuda.empty_cache()


        ######################################################################################
        """ Trainer """ 
        print('#----------Prepareing Model----------#')

        self.network = DconnNet(self.args.num_class).to('cuda')
        self.step = 0
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
        self.loss_func = connect_loss(self.args, self.hori_translation, self.verti_translation).cuda()

        """define optimizer"""
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.lr)

        """define lr_scheduler"""
        # self.lr_scheduler = self.set_lr_scheduler()
    

    def create_exp_directory(self):
        csv = 'val_results'+'.csv'
        with open(os.path.join(self.args.log_dir, csv), 'w') as f:
            f.write('epoch, dice, Jac, clDice \n')
        csv1 = 'test_results'+'.csv'
        with open(os.path.join(self.args.log_dir, csv1), 'w') as f:
            f.write('dice, Jac, clDice, acc, sen, spe, pre, recall, f1 \n')


    
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
    
    def train(self, train_loader, test_loader):
        print('#----------Training----------#')
        best_dice = 0.0
        
        self.create_exp_directory()

        if not self.args.load_model is None:
            print('#----------Resume Model and Set Other params----------#')
            checkpoint_epoch, self.network, self.optimizer, self.lr_scheduler = self.load_model(self.network, self.opt, self.lr_scheduler)
            start_epoch = checkpoint_epoch
        else:
            print('#----------Set other params----------#')
            checkpoint_epoch = 0
            start_epoch = 1

        for epoch in range(start_epoch, self.args.epochs + 1):
            torch.cuda.empty_cache()
            print('---------------------------------------')

            self.step = self.train_epoch(train_loader, epoch, self.step)

            dice, Jac, cldice = self.val_epoch(test_loader, epoch)

            ##########################################

            if best_dice < dice:
                best_dice = dice
                torch.save(self.network.state_dict(), os.path.join(self.args.checkpoint_dir, 'best.pth'))
                log_info = f'Epoch: {epoch}, Total DSC: {dice:.4f}, IOU: {Jac:.4f}, clDice: {cldice:.4f}'
                print(log_info)
                self.logger.info(log_info)
                print('save best!')
            torch.cuda.empty_cache()


    def test(self, train_loader, test_loader):
        if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')

            best_weight = torch.load(self.args.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
            self.network.load_state_dict(best_weight)

            dice_ls = self.test_epoch(test_loader)

            torch.cuda.empty_cache()


    def train_epoch(self, train_loader, epoch, step):
        
        self.network.train()
        loss_list = [] 
        num_epochs = 10
        if self.args.lr_update == 'step':
            now_lr = get_lr(self.args.lr, self.args.lr_update, epoch, num_epochs, gamma=self.args.gamma, step=self.args.lr_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = now_lr


        for iter, data in enumerate(train_loader):
            step += iter

            X = Variable(data[0])
            y = Variable(data[1])  

            X = X.float().cuda()
            y = y.float().cuda()

            self.optimizer.zero_grad()
            output, aux_out = self.network(X)

            ## output/aux_out: c_map 八通道, y: target 一通道
            loss_main = self.loss_func(output, y)    ## Lmain
            loss_aux = self.loss_func(aux_out, y)    ## Lprior SDE的辅助输出
            loss = loss_main + 0.3*loss_aux
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            loss_list.append(loss.item())
            

            if iter % self.args.print_interval == 0:
                self.save_print_loss_lr(iter, epoch, loss_list, now_lr)

        return step
    

    def val_epoch(self, loader, epoch):
        self.network.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []

        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):

                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])

                X_test= X_test.float().cuda()
                y_test = y_test.long().cuda()

                output_test,_ = self.network(X_test)
                batch, channel, H, W = X_test.shape


                hori_translation = self.hori_translation.repeat(batch,1,1,1).cuda()
                verti_translation = self.verti_translation.repeat(batch,1,1,1).cuda()

                
                if self.args.num_class == 1:  
                    output_test = F.sigmoid(output_test)
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    pred = torch.where(class_pred>0.5,1,0)
                    pred,_ = Bilateral_voting(pred.float(),hori_translation,verti_translation)
                

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
                    
                    log_info = f'Iteration: {str(j_batch)} / {str(len(loader))}, Total DSC: {np.mean(self.dice_ls):.3f}'
                    print(log_info)
                    self.logger.info(log_info)
                    

            Jac_ls = np.array(self.Jac_ls)
            dice_ls = np.array(self.dice_ls)
            total_dice = np.mean(dice_ls)
            csv = 'val_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f \n' % (
                    (epoch),
                    total_dice,
                    np.mean(Jac_ls),
                    np.mean(self.cldice_ls)
                ))

            return np.mean(self.dice_ls), np.mean(Jac_ls), np.mean(self.cldice_ls)
        

        
    def test_epoch(self, loader):
        self.network.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []
        self.acc_ls = []
        self.sen_ls = []
        self.spe_ls = []
        self.pre_ls = []
        self.recall_ls = []
        self.f1_ls = []

        pred_ls = []
        gt_ls = []

        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):

                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                # name = test_data[2]

                X_test = X_test.float().cuda()
                y_test = y_test.long().cuda()
                

                output_test,_ = self.network(X_test)
                batch, channel, H, W = X_test.shape


                hori_translation = self.hori_translation.repeat(batch,1,1,1).cuda()
                verti_translation = self.verti_translation.repeat(batch,1,1,1).cuda()

                
                if self.args.num_class == 1:  
                    output_test = F.sigmoid(output_test)
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    pred = torch.where(class_pred>0.5,1,0)
                    pred,_ = Bilateral_voting(pred.float(),hori_translation,verti_translation)

                if j_batch % self.args.save_interval == 0:
                    save_path = self.args.res_dir
                    self.save_img(y_test, pred, j_batch)
                    # self.save_img(X_test, y_test, pred, j_batch, save_path)
             
                # dice, Jac = self.per_class_dice(pred,y_test)
                dice, Jac, acc, sen, spe, pre, recall, f1_score = self.per_class_metric(pred,y_test)

                if self.args.num_class == 1:
                    pred_np = pred.squeeze().cpu().numpy()
                    target_np = y_test.squeeze().cpu().numpy()
                    cldc = clDice(pred_np, target_np)
                    self.cldice_ls.append(cldc)
                
                if self.args.num_class>1:
                    self.dice_ls += torch.mean(dice[:,1:],1).tolist() ## use self.dice_ls += torch.mean(dice,1).tolist() if you want to include BG
                    self.Jac_ls += torch.mean(Jac[:,1:],1).tolist() ## same as above
                else:
                    self.dice_ls += dice[:,0].tolist()
                    self.Jac_ls += Jac[:,0].tolist()
                    self.acc_ls += acc[:,0].tolist()
                    self.sen_ls += sen[:,0].tolist()
                    self.spe_ls += spe[:,0].tolist()
                    self.pre_ls += pre[:,0].tolist()
                    self.recall_ls += recall[:,0].tolist()
                    self.f1_ls += f1_score[:,0].tolist()


                if j_batch%(max(1,int(len(loader)/5)))==0:
                    # print('[Iteration : ' + str(j_batch) + '/' + str(len(loader)) + '] Total DSC:%.3f ' %(
                    #     np.mean(self.dice_ls)))
                    
                    log_info = f'Iteration: {str(j_batch)} / {str(len(loader))}, Total DSC: {np.mean(self.dice_ls):.3f}'
                    print(log_info)
                    self.logger.info(log_info)
                
            Jac_ls = np.array(self.Jac_ls)
            dice_ls = np.array(self.dice_ls)
            acc_ls = np.array(self.acc_ls)
            sen_ls = np.array(self.sen_ls)
            spe_ls = np.array(self.spe_ls)
            pre_ls = np.array(self.pre_ls)
            recall_ls = np.array(self.recall_ls)
            f1_ls = np.array(self.f1_ls)


            total_dice = np.mean(dice_ls)
            csv = 'test_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f \n' % (
                    total_dice,
                    np.mean(Jac_ls),
                    np.mean(self.cldice_ls),
                    np.mean(acc_ls),
                    np.mean(sen_ls),
                    np.mean(spe_ls),
                    np.mean(pre_ls),
                    np.mean(recall_ls),
                    np.mean(f1_ls)
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
        

    def per_class_metric(self, y_pred, y_true):
        smooth = 0.0001
        y_pred = y_pred
        y_true = y_true

        FN = torch.sum((1-y_pred)*y_true,dim=(2,3)) 
        FP = torch.sum((1-y_true)*y_pred,dim=(2,3)) 
        TN = torch.sum((1 - y_pred) * (1 - y_true), dim=(2, 3))
        TP = torch.sum(y_pred * y_true, dim=(2, 3))

        Pred = y_pred
        GT = y_true
        inter = torch.sum(GT* Pred,dim=(2,3)) 

        union = torch.sum(GT,dim=(2,3)) + torch.sum(Pred,dim=(2,3)) 

        dice = (2*inter + smooth)/(union + smooth)
        Jac = (inter + smooth)/(inter+FP+FN+smooth)
        acc = (TN + TP + smooth)/(FN+FP+TN+TP+smooth)
        sen = (TP + smooth)/(TP + FN + smooth)
        spe = (TN + smooth)/(TN + FP + smooth)
        pre = (TP + smooth)/(TP + FP + smooth)
        recall = (TP + smooth)/(TP + FN + smooth)
        f1_score = (2*pre*recall + smooth)/(pre + recall + smooth)

        return dice, Jac, acc, sen, spe, pre, recall, f1_score


    
    
    def save_img(self,gt,pred,iter):

        save_path = os.path.join(self.args.res_dir, str(0))
        self.makedirs(save_path)

        size = self.args.img_size / 100

        pred_array = pred.squeeze(0).squeeze(0).cpu().numpy()
        plt.figure(figsize=(size,size),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(pred_array, cmap='gray')  # 如果是灰度图可以指定 cmap='gray'，如果是彩色图无需指定 cmap
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(save_path +'/'+ 'result'+ str(iter) +'.png')
        plt.close()


        gt_array = gt.squeeze(0).squeeze(0).cpu().numpy()
        plt.figure(figsize=(size,size),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(gt_array, cmap='gray')  
        plt.axis('off') 
        plt.savefig(save_path +'/'+ 'gt'+ str(iter) +'.png')
        plt.close()
         
         
         




