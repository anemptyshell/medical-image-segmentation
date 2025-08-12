import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
from torch.nn import functional as F
import torch.optim
import torch.utils.data
# import torch.nn as nn
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# from sklearn.metrics import roc_auc_score, confusion_matrix

from libs.base_model import base_model
from libs.metric import metric

# from Med_image_seg.fang.utils.lr_update import get_lr
from Med_image_seg.fang1.utils.cldice import clDice
from Med_image_seg.fang1.utils.loss import BceDiceLoss, soft_iou_loss #, bce_ssim_loss, bceioussim_loss

from Med_image_seg.fang1.network import UNet_3de_emi_hidi#,UNet_3de_emi#UNet_3decoder_hidi#UNet_Multi_decoder#UNet_2de#UNet_3de_sfd# UNet_3de_emi_hidi #UNet_3decoder_hidi  UNet_3decoder_hidi #UNet_Multi_decoder_fusion#Rolling_Unet_L #PUnet

import numpy as np
# from hausdorff import hausdorff_distance




def arguments():
    args = {
    '--lr_update': 'step',
    '--lr_step': 12

}  
    return args


class fang1(base_model):
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
   
        self.set_seed(self.args.seed)
        self.set_cuda()
        torch.cuda.empty_cache()

        ######################################################################################
        """ Trainer """ 
        print('#----------Prepareing Model----------#')

        # self.network = UNet(n_channels=3).to('cuda')
        # self.network = UNet().to('cuda')
        self.network = UNet_3de_emi_hidi().to('cuda')
        self.step = 0
        self.save_args()
  
        ######################################################################################

        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        """define loss"""
        self.BceDiceLoss = BceDiceLoss().cuda()
        self.loss_iou = soft_iou_loss().cuda()
        # self.loss_bce = nn.BCELoss()  ## cuda报错 all elements of input should be between 0 and 1
        # self.loss_hybrid = bce_ssim_loss().cuda()
        # self.loss_ssim = bceioussim_loss().cuda()

        """define optimizer"""
        self.optimizer = self.set_optimizer()

        """define lr_scheduler"""
        self.lr_scheduler = self.set_lr_scheduler()

   
    def create_exp_directory(self):
        csv = 'val_results'+'.csv'
        with open(os.path.join(self.args.log_dir, csv), 'w') as f:
            f.write('epoch, dice, Jac, clDice \n')
        csv1 = 'test_results'+'.csv'
        with open(os.path.join(self.args.log_dir, csv1), 'w') as f:
            f.write('dice, Jac, clDice, acc, sen, spe, pre, recall, f1 \n')


    def train(self, train_loader, test_loader):
        print('#----------Training----------#')
        torch.cuda.empty_cache()

        best = 0.0
        self.create_exp_directory()

        ## Resume Model
        if not self.args.load_model is None:
            print('#----------Resume Model and Set Other params----------#')
            checkpoint_epoch, self.network, self.optimizer, self.lr_scheduler = self.load_model(self.network, self.opt, self.lr_scheduler)
            start_epoch = checkpoint_epoch
        else:
            print('#----------Set other params----------#')
            checkpoint_epoch = 0
            start_epoch = 1

        ## range 的范围:start_epoch ~ self.args.epochs
        for epoch in range(start_epoch, self.args.epochs + 1):
            torch.cuda.empty_cache()
            print('---------------------------------------')

            self.step = self.train_epoch(train_loader, epoch, self.step)
            dice, Jac, cldice, pred, gt= self.val_epoch(test_loader)
            
            ##########################################
            dice_ls = np.array(dice)
            Jac_ls = np.array(Jac)
            total_dice = np.mean(dice_ls)
            csv = 'val_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f \n' % (
                    (epoch),
                    total_dice,
                    np.mean(Jac_ls),
                    np.mean(cldice),
               
                ))
            ##########################################

            # if best_dice < np.mean(dice):
            #     best_dice = np.mean(dice)
            #     torch.save(self.network.state_dict(), os.path.join(self.args.checkpoint_dir, 'best.pth'))
            #     log_info = f'Epoch: {epoch}, Total DSC: {np.mean(dice):.4f}, IOU: {np.mean(Jac):.4f}, clDice: {np.mean(cldice):.4f}'
            #     print(log_info)
            #     self.logger.info(log_info)
            #     print('save best!')
           
            ## 使用egeunet的指标计算来保存最佳权重
            metric_cluster = metric(pred, gt, self.args.metric_list)
            best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best, self.indicator_for_best)
            self.save_print_metric(epoch, metric_cluster, best)
            self.save_model(epoch)

            torch.cuda.empty_cache()


    def test(self, train_loader, test_loader):
        best = 0.0
        if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')

            best_weight = torch.load(self.args.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
            self.network.load_state_dict(best_weight)

            # dice,Jac,cldice_ls,acc,sen,spe,pre,recall,f1,pred, gt = self.test_epoch(test_loader)
            pred, gt, cldice_ls = self.test_epoch(test_loader)
        

            dice_ls = np.array(self.dice_ls)
            Jac_ls = np.array(self.Jac_ls)
            acc_ls = np.array(self.acc_ls)
            sen_ls = np.array(self.sen_ls)
            spe_ls = np.array(self.spe_ls)
            pre_ls = np.array(self.pre_ls)
            recall_ls = np.array(self.recall_ls)
            f1_ls = np.array(self.f1_ls)

            dice_mean = np.mean(dice_ls)
            Jac_mean = np.mean(Jac_ls)
            acc_mean = np.mean(acc_ls)
            sen_mean = np.mean(sen_ls)
            spe_mean = np.mean(spe_ls)
            pre_mean = np.mean(pre_ls)
            recall_mean = np.mean(recall_ls)
            f1_mean = np.mean(f1_ls)
            # cldice_mean =np.mean(cldice_ls) 

            csv = 'test_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f \n' % (
                    dice_mean,
                    Jac_mean,
                    cldice_ls,
                    acc_mean,
                    sen_mean,
                    spe_mean,
                    pre_mean,
                    recall_mean,
                    f1_mean
                ))
           
            ###########################################################
            metric_cluster = metric(pred, gt, self.args.metric_list)
            best, self.best_trigger, self.indicator_for_best = metric_cluster.best_value_indicator(best, self.indicator_for_best)
            self.save_print_metric("test of best model", metric_cluster, best)
            torch.cuda.empty_cache()


    def train_epoch(self, train_loader, epoch, step):
        
        self.network.train()
        loss_list = []

        for iter, data in enumerate(train_loader):
            step += iter
            
            images, targets, edge, skeleton = data
            # images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
            edge, skeleton = edge.cuda(non_blocking=True).float(), skeleton.cuda(non_blocking=True).float()

            # preds, pred_edge, pred_skeleton = self.network(images)
            pred_raw, pred_edge, pred_skeleton, preds = self.network(images)

            """for edge or skeleton"""
            # preds, pred_edge = self.network(images)
            # preds, pred_skeleton = self.network(images)

            # loss1 = self.BceDiceLoss(preds, targets)
            # loss2 = self.BceDiceLoss(pred_edge, skeleton)
            # loss = 0.5*loss1 + 0.5*loss2
         
        
            loss1 = self.loss_iou(preds, targets)
            # loss1 = self.loss_iou(preds, targets)
            # loss1 = self.loss_ssim(preds, targets)
    
            loss2 = self.BceDiceLoss(pred_edge, edge)
            loss3 = self.BceDiceLoss(pred_skeleton, skeleton)
            # loss4 = self.BceDiceLoss(pred_raw, targets)
            loss4 = self.BceDiceLoss(pred_raw, targets)

            # loss = loss1 + 0.5 * loss2 + 0.5 * loss3
            # loss = 0.33 * loss1 + 0.33 * loss2 + 0.33 * loss3

            loss = loss1 + 0.33*loss2 + 0.33*loss3 + 0.33*loss4        
           
            ## compute gradient and do optimizing step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()

            loss_list.append(loss.item())
            now_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            
            if iter % self.args.print_interval == 0:
                self.save_print_loss_lr(iter, epoch, loss_list, now_lr)

        self.lr_scheduler.step() 
        return step
    

    def val_epoch(self, test_loader):
        self.network.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []

        pred_ls = []
        gt_ls = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
                
                # preds = self.network(images)
                # preds, pred_thin, pred_thick = self.network(images)
                # preds, pred_edge, pred_skeleton = self.network(images)
                pred_raw, pred_edge, pred_skeleton, preds = self.network(images)
                # preds, pred_edge, pred_skeleton, loss_mi = self.network(images)

                """for edge or skeleton"""
                # preds, pred_edge = self.network(images)
                # preds, pred_skeleton = self.network(images)
                
                dice, Jac = self.per_class_dice(preds, targets)

                preds_ = preds.squeeze().cpu()
                targets_ = targets.squeeze().cpu()

                y_scores = np.zeros_like(preds_)
                y_true = np.zeros_like(targets_)
                y_true[targets_ > 0.01] = 1
                y_scores[preds_ > 0.3] = 1    

                cldc = clDice(y_scores, y_true)
                self.cldice_ls.append(cldc)   

                self.dice_ls += dice[:,0].tolist()
                self.Jac_ls += Jac[:,0].tolist()

                gt_ls.append(targets.squeeze(1).cpu().detach().numpy())
                if type(preds) is tuple:
                    preds = preds[0]
                preds = preds.squeeze(1).cpu().detach().numpy()
                pred_ls.append(preds)

         
        return self.dice_ls, self.Jac_ls, self.cldice_ls, pred_ls, gt_ls



    def test_epoch(self, test_loader):
        self.network.eval()
        pred_ls = []
        gt_ls = []
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []
        self.acc_ls = []
        self.sen_ls = []
        self.spe_ls = []
        self.pre_ls = []
        self.recall_ls = []
        self.f1_ls = []
        # self.hd = []

        with torch.no_grad():
            for iter, data in enumerate(tqdm(test_loader)):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
                
                # preds, pred_thin, pred_thick = self.network(images)
                # preds, pred_edge, pred_skeleton = self.network(images)
                pred_raw, pred_edge, pred_skeleton, preds = self.network(images)
                # preds, pred_edge, pred_skeleton, loss_mi = self.network(images)
                # preds, pred_edge = self.network(images)

                preds_ = preds.squeeze().cpu()
                targets_ = targets.squeeze().cpu()
                   
                y_scores = np.zeros_like(preds_)
                y_true = np.zeros_like(targets_)
                y_true[targets_ > 0.01] = 1
                y_scores[preds_ > 0.3] = 1    

                cldc = clDice(y_scores, y_true)
                self.cldice_ls.append(cldc)            
               
                ##################################################
                # pred_np = output_.squeeze().cpu().numpy()
                # target_np = targets.squeeze().cpu().numpy()
                # cldc = clDice(pred_np, target_np)
                # self.cldice_ls.append(cldc)

                ##################################################
                dice, Jac, acc, sen, spe, pre, recall, f1_score = self.per_class_metric(preds, targets)

                self.dice_ls += dice[:,0].tolist()
                self.Jac_ls += Jac[:,0].tolist()
                self.acc_ls += acc[:,0].tolist()
                self.sen_ls += sen[:,0].tolist()
                self.spe_ls += spe[:,0].tolist()
                self.pre_ls += pre[:,0].tolist()
                self.recall_ls += recall[:,0].tolist()
                self.f1_ls += f1_score[:,0].tolist()

                # if iter % self.args.save_interval == 0:
                #     save_path = self.args.res_dir
                #     self.save_img(images, targets, output_, iter, save_path)

                ##################################################
              
                targets_np = targets.squeeze(1).cpu().detach().numpy()
                gt_ls.append(targets_np)
                if type(preds) is tuple:
                    preds = preds[0]
                preds_np = preds.squeeze(1).cpu().detach().numpy()
                pred_ls.append(preds_np) 

                save_path = self.args.res_dir
                if iter % self.args.save_interval == 0:
                    self.save_imgs(images, targets_np, preds_np, iter, save_path)
        
        epoch_clDice = np.mean(self.cldice_ls)

        return pred_ls, gt_ls, epoch_clDice #, self.cldice_ls

        
    



    # def test_epoch(self, test_loader):
    #     self.network.eval()
    #     pred_ls = []
    #     gt_ls = []
    #     cldice_ls = []
    #     with torch.no_grad():
    #         for iter, data in enumerate(tqdm(test_loader)):
    #             images, targets = data
    #             images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

    #             preds = self.network(images)
    #             loss = self.BceDiceLoss(preds, targets) 

    #             targets = targets.squeeze(1).cpu().detach().numpy()
    #             gt_ls.append(targets)
    #             if type(preds) is tuple:
    #                 preds = preds[0]
    #             preds = preds.squeeze(1).cpu().detach().numpy()
    #             pred_ls.append(preds) 

    #             cldc = clDice(preds, targets)
    #             cldice_ls.append(cldc)


    #             save_path = self.args.res_dir
    #             if iter % self.args.save_interval == 0:
    #                 # self.save_imgs(images, targets, preds, iter, save_path, self.args.dataset)
    #                 self.save_img(images, targets, preds, iter, save_path)
    #         cldice_mean =np.mean(cldice_ls)
    #         print('cldice:', cldice_mean)
    #         return pred_ls, gt_ls    


