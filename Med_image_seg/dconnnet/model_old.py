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
        # self.BceDiceLoss = BceDiceLoss().cuda()
        self.loss_func = connect_loss(self.args, self.hori_translation, self.verti_translation)

        """define optimizer"""
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.lr)

        """define lr_scheduler"""
        # self.lr_scheduler = self.set_lr_scheduler()
    

    
    def train(self, train_loader, test_loader):
        print('#----------Training----------#')
        best = 0.0
        
        # self.create_exp_directory()

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

            pred, gt= self.val_epoch(test_loader)

            metric_cluster = metric(pred, gt, self.args.metric_list)
            best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best, self.indicator_for_best)
            self.save_print_metric(epoch, metric_cluster, best)
            self.save_model(epoch)
           


    def test(self, train_loader, test_loader):
        best = 0.0
        if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')

            best_weight = torch.load(self.args.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
            self.network.load_state_dict(best_weight)
            self.network = self.network.cuda()

            pred, gt, cldice_ls = self.test_epoch(test_loader)
            # pred, gt = self.test_epoch(test_loader)
            cldice_mean =np.mean(cldice_ls) 
            print('cldice: %.4f'%cldice_mean)
            csv = 'test_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%0.6f \n' % (cldice_mean))

            metric_cluster = metric(pred, gt, self.args.metric_list)
            best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best, self.indicator_for_best)
            self.save_print_metric("test of best model", metric_cluster, best)
    
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
            y = Variable(data[1])   ## y是gt ？？？

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
    

    def val_epoch(self, loader):
        self.network.eval()
        pred_ls = []
        gt_ls = []
        loss_list = []

        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):

                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                # name = test_data[2]

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
                    pred,_ = Bilateral_voting(class_pred.float(),hori_translation,verti_translation)
                
                gt_ls.append(y_test.squeeze(1).cpu().detach().numpy())
                pred = pred.squeeze(1).cpu().detach().numpy()
                pred_ls.append(pred)

            return pred_ls, gt_ls


    def test_epoch(self, loader):
        self.network.eval()
        pred_ls = []
        gt_ls = []
        self.cldice_ls = []
        self.dice_ls = []
        self.Jac_ls=[]
        self.acc_ls = []
        self.sen_ls = []
        self.spe_ls = []
        self.pre_ls = []
        self.recall_ls = []
        self.f1_ls = []

        with torch.no_grad(): 
            for iter, test_data in enumerate(tqdm(loader)):

                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                # name = test_data[2]

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
                # gt__ = torch.where(y_test>0.5,1,0)

                dice, Jac, acc, sen, spe, pre, recall, f1_score = self.per_class_metric(pred,y_test)
                self.dice_ls += dice[:,0].tolist()
                self.Jac_ls += Jac[:,0].tolist()
                self.acc_ls += acc[:,0].tolist()
                self.sen_ls += sen[:,0].tolist()
                self.spe_ls += spe[:,0].tolist()
                self.pre_ls += pre[:,0].tolist()
                self.recall_ls += recall[:,0].tolist()
                self.f1_ls += f1_score[:,0].tolist()


                if self.args.num_class == 1:
                    pred_np = pred.squeeze().cpu().numpy()
                    target_np = y_test.squeeze().cpu().numpy()
                    cldc = clDice(pred_np, target_np)
                    self.cldice_ls.append(cldc)

                targets = y_test.squeeze(1).cpu().detach().numpy()
                gt_ls.append(targets)
              
                preds = pred.squeeze(1).cpu().detach().numpy()
                pred_ls.append(preds) 

                
                if iter % self.args.save_interval == 0:
                    save_path = self.args.res_dir
                    self.save_imgs(X_test, targets, preds, iter, save_path)
            
            Jac_ls = np.array(self.Jac_ls)
            dice_ls = np.array(self.dice_ls)
            acc_ls = np.array(self.acc_ls)
            sen_ls = np.array(self.sen_ls)
            spe_ls = np.array(self.spe_ls)
            pre_ls = np.array(self.pre_ls)
            recall_ls = np.array(self.recall_ls)
            f1_ls = np.array(self.f1_ls)
            print('dice:', np.mean(dice_ls))
            print('iou:', np.mean(Jac_ls))
            print('acc:', np.mean(acc_ls))
            print('sen:', np.mean(sen_ls))
            print('spe:', np.mean(spe_ls))
            print('pre:', np.mean(pre_ls))
            print('recall:', np.mean(recall_ls))
            print('f1:', np.mean(f1_ls))

            

            return pred_ls, gt_ls, np.mean(self.cldice_ls)



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


        




