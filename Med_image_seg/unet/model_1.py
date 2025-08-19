import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.utils.data


from libs.metric import metric
from libs.base_model import base_model

from Med_image_seg.unet.loss import BceDiceLoss
from Med_image_seg.unet.network import U_Net
from Med_image_seg.fang.utils.cldice import clDice

# from matplotlib import pyplot as plt
import numpy as np



def arguments():
    args = {
}  
    return args


class unet(base_model):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())


        print('#----------Creating logger----------#')
        ## make work_dir ,log_dir, res_dir, checkpoint_dir
        self.make_dir(parser)
        self.args = parser.get_args()
      
        print('#----------GPU init----------#')
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_id
        # self.set_seed(self.args.seed)
        self.set_cuda()
        torch.cuda.empty_cache()


        ######################################################################################
        """ Trainer """ 
        print('#----------Prepareing Model----------#')

        self.network = U_Net().to('cuda')
        self.step = 0
        self.save_args()
  
        ######################################################################################


        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        """define loss"""
        self.BceDiceLoss = BceDiceLoss().cuda()

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

            pred, gt, cldice_ls = self.test_epoch(test_loader)

            ###########################################################
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
            
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

            preds = self.network(images)
            loss = self.BceDiceLoss(preds, targets)
            
            ## compute gradient and do optimizing step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()

            loss_list.append(loss.item())
            now_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            

            ## % 取模 - 返回除法的余数
            ## 如果当前迭代次数达到了要求，log
            if iter % self.args.print_interval == 0:
                self.save_print_loss_lr(iter, epoch, loss_list, now_lr)

        self.lr_scheduler.step() 
        return step
    


    def val_epoch(self, test_loader):
        self.network.eval()
        pred_ls = []
        gt_ls = []
    
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images)

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
            for iter, data in enumerate(tqdm(test_loader)):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images)



                preds_ = preds.squeeze().cpu()
                targets_ = targets.squeeze().cpu()

                y_scores = np.zeros_like(preds_)
                y_true = np.zeros_like(targets_)
                y_true[targets_ > 0.01] = 1
                y_scores[preds_ > 0.3] = 1    
                cldc = clDice(y_scores, y_true)
                self.cldice_ls.append(cldc)  

                #########################################################
                dice, Jac, acc, sen, spe, pre, recall, f1_score = self.per_class_metric(preds, targets)

                self.dice_ls += dice[:,0].tolist()
                self.Jac_ls += Jac[:,0].tolist()
                self.acc_ls += acc[:,0].tolist()
                self.sen_ls += sen[:,0].tolist()
                self.spe_ls += spe[:,0].tolist()
                self.pre_ls += pre[:,0].tolist()
                self.recall_ls += recall[:,0].tolist()
                self.f1_ls += f1_score[:,0].tolist()
                #########################################################

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
        
        return pred_ls, gt_ls, epoch_clDice
    
    
    

    # def per_class_metric(self, y_pred, y_true):
    #     smooth = 0.0001
    #     y_pred = y_pred
    #     y_true = y_true

    #     FN = torch.sum((1-y_pred)*y_true,dim=(2,3)) 
    #     FP = torch.sum((1-y_true)*y_pred,dim=(2,3)) 
    #     TN = torch.sum((1 - y_pred) * (1 - y_true), dim=(2, 3))
    #     TP = torch.sum(y_pred * y_true, dim=(2, 3))

    #     Pred = y_pred
    #     GT = y_true
    #     inter = torch.sum(GT* Pred,dim=(2,3)) 

    #     union = torch.sum(GT,dim=(2,3)) + torch.sum(Pred,dim=(2,3)) 

    #     dice = (2*inter + smooth)/(union + smooth)
    #     Jac = (inter + smooth)/(inter+FP+FN+smooth)
    #     acc = (TN + TP + smooth)/(FN+FP+TN+TP+smooth)
    #     sen = (TP + smooth)/(TP + FN + smooth)
    #     spe = (TN + smooth)/(TN + FP + smooth)
    #     pre = (TP + smooth)/(TP + FP + smooth)
    #     recall = (TP + smooth)/(TP + FN + smooth)
    #     f1_score = (2*pre*recall + smooth)/(pre + recall + smooth)

    #     return dice, Jac, acc, sen, spe, pre, recall, f1_score   

    
























    # def test_one_epoch(self, test_loader):
    #     self.network.eval()
    #     loss_sum = 0
    #     dice_sum, iou_sum = 0.0, 0.0
    #     pred = []
    #     gts = []
    #     # loss_list = []
    #     threshold = 0.5


    #     with torch.no_grad():
    #         for iter, data in enumerate(tqdm(test_loader)):
    #             images, targets = data
    #             images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
    #             # print(images.size())  ## torch.Size([1, 3, 256, 256])
    #             # print(targets.size())  ## torch.Size([1, 1, 256, 256])
    #             # print(images.dtype)   ## torch.float32

    #             # images, targets = images.to(self.device), targets.to(self.device)

    #             preds = self.network(images)
    #             loss = self.BceDiceLoss(preds, targets) 

    #             iou,dice = iou_score(preds, targets)  
    #             loss_sum += len(images) * loss
    #             iou_sum += len(images) * iou
    #             dice_sum += len(images) * dice

    #             if iter == len(test_loader):
    #                 loss_avg = loss_sum / (self.args.batch_size*(iter-1) + len(images))
                
    #                 iou_avg = iou_sum / (self.args.batch_size*(iter-1) + len(images))
    #                 dice_avg = dice_sum / (self.args.batch_size*(iter-1) + len(images))
    #             else:
    #                 loss_avg = loss_sum / (iter * self.args.batch_size)
    #                 iou_avg = iou_sum / (iter * self.args.batch_size)                
    #                 dice_avg = dice_sum / (iter * self.args.batch_size)            

    #             # loss_list.append(loss.item())

    #             targets = targets.squeeze(1).cpu().detach().numpy()
    #             # print(targets.shape)   # (1, 256, 256)
    #             # print('*'*10)
    
    #             gts.append(targets)
    #             if type(preds) is tuple:
    #                 preds = preds[0]

    #             preds = preds.squeeze(1).cpu().detach().numpy()
    #             # print(preds.shape)  # (1, 256, 256)
    #             # print('*'*10)

    #             images = images.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    #             images = images / 255. if images.max() > 1.1 else images
    #             # print(images.shape)  # (256, 256, 3)
    #             # print('*'*10)

    #             pred.append(preds) 

    #             save_path = self.args.res_dir
    #             if iter % self.args.save_interval == 0:
    #                 save_imgs(images, targets, preds, iter, save_path, self.args.dataset)


    #         pred = np.array(pred).reshape(-1)
    #         gts = np.array(gts).reshape(-1)

    #         y_pre = np.where(pred >= threshold, 1, 0)
    #         y_true = np.where(gts>=0.5, 1, 0)

    #         confusion, accuracy, sensitivity, specificity, dsc, miou, precision, recall, f1_score = get_metric(y_pre, y_true)

    #         # 添加loss: {np.mean(loss_list):.4f},
    #         log_info_1 = f'test of best model, mIoU: {miou}, DSC: {dsc}, acc: {accuracy}, spe: {specificity}, sen_or_recall: {sensitivity}, precision: {precision}, \
    #                        F1_score: {f1_score}, confusion_matrix: {confusion}'
    #         log_info = f'test of best model, IoU: {iou}, IoU_avg: {iou_avg}, Dice: {dice}, Dice_avg: {dice_avg}'
    #         print(log_info_1)
    #         print(log_info)
    #         self.logger.info(log_info_1)
    #         self.logger.info(log_info)

    #     # return np.mean(loss_list)
    #     return loss_avg

 
    # def val_one_epoch(self, test_loader, epoch):
    #     # switch to evaluate mode
    #     self.network.eval()
    #     loss_sum = 0
    #     dice_sum, iou_sum = 0.0, 0.0
    #     pred = []
    #     gts = []
    #     loss_list = []
    #     threshold = 0.5

    #     with torch.no_grad():
    #         # for data in tqdm(test_loader):
    #         for iter, data in enumerate(test_loader):
    #             images, targets = data
    #             images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
    #             # img, msk = img.to(self.device), msk.to(self.device)

    #             preds = self.network(images)
    #             loss = self.BceDiceLoss(preds, targets)

    #             iou, dice = iou_score(preds, targets)
    #             loss_sum += len(images) * loss
    #             iou_sum += len(images) * iou
    #             dice_sum += len(images) * dice

    #             if iter == len(test_loader):
    #                 loss_avg = loss_sum / (self.args.batch_size*(iter-1) + len(images))
                
    #                 iou_avg = iou_sum / (self.args.batch_size*(iter-1) + len(images))
    #                 dice_avg = dice_sum / (self.args.batch_size*(iter-1) + len(images))
    #             else:
    #                 loss_avg = loss_sum / (iter * self.args.batch_size)
    #                 iou_avg = iou_sum / (iter * self.args.batch_size)                
    #                 dice_avg = dice_sum / (iter * self.args.batch_size)


    #             loss_list.append(loss.item())
    #             gts.append(targets.squeeze(1).cpu().detach().numpy())

    #             if type(preds) is tuple:
    #                 preds = preds[0]
    #             preds = preds.squeeze(1).cpu().detach().numpy()

    #             pred.append(preds) 

    #     if epoch % self.args.val_interval == 0:
    #         pred = np.array(pred).reshape(-1)   ## 展成一维数组
    #         gts = np.array(gts).reshape(-1)

    #         y_pre = np.where(pred >= threshold, 1, 0)
    #         y_true = np.where(gts>=0.5, 1, 0)

    #         confusion, accuracy, sensitivity, specificity, dsc, miou, precision, recall, f1_score = get_metric(y_pre, y_true)

    #         log_info_1 = f'val epoch: {epoch}, mIoU: {miou:.4f}, DSC: {dsc:.4f}, acc: {accuracy:.4f}, \
    #                     spe: {specificity:.4f}, sen_or_recall: {sensitivity:.4f}, precision: {precision:.4f}, F1_score: {f1_score:.4f}, confusion_matrix: {confusion}'
            
    #         # log_info = f'val epoch: {epoch}, loss: {loss:.4f}, loss_avg: {loss_avg:.4f}, IoU: {iou}, IoU_avg: {iou_avg}, Dice: {dice}, Dice_avg: {dice_avg}'
    #         log_info = f'val epoch: {epoch}, IoU: {iou}, IoU_avg: {iou_avg}, Dice: {dice}, Dice_avg: {dice_avg}'
    #         print(log_info_1)
    #         print(log_info)
    #         self.logger.info(log_info_1)
    #         self.logger.info(log_info)

    #     else:
    #         pred = np.array(pred).reshape(-1)   ## 展成一维数组
    #         gts = np.array(gts).reshape(-1)

    #         y_pre = np.where(pred >= threshold, 1, 0)
    #         y_true = np.where(gts>=0.5, 1, 0)

    #         dsc = get_dsc(y_pre, y_true)

    #         log_info = f'val epoch: {epoch}, loss_avg: {loss_avg:.4f}, loss: {np.mean(loss_list):.4f}'
    #         print(log_info)
    #         self.logger.info(log_info)
   
    #     return np.mean(loss_list), dsc




