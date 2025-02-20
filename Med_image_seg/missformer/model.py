import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.nn import functional as F
import cv2


from libs.metric import metric
from libs.utils import AverageMeter
from libs.base_model import base_model
from libs.metric_utils.cldice import clDice
from libs.metric_utils.BettiMatching import BettiMatching

from sklearn.metrics import roc_auc_score, confusion_matrix
from hausdorff import hausdorff_distance
from multiprocessing import Pool
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision


from Med_image_seg.missformer.loss import BceDiceLoss
from Med_image_seg.missformer.network import MISSFormer
# # from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter




def arguments():
    args = {
    '--num_classes': 1, 
}  
    return args



class missformer(base_model):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())

        print('#----------Creating logger----------#')
        ## make work_dir ,log_dir, res_dir, checkpoint_dir
        self.make_dir(parser)
        self.args = parser.get_args()
        global logger
        self.logger = self.get_logger('train', self.args.log_dir)

        # global writer
        # self.writer = SummaryWriter(self.work_dir + 'summary')

        print('#----------GPU init----------#')
        # self.set_seed(self.args.seed)
        self.set_cuda()
        torch.cuda.empty_cache()


        ######################################################################################
        """ Trainer """ 
        print('#----------Prepareing Model----------#')

        self.network = MISSFormer(self.args.num_classes).to('cuda')
        self.step = 0
        self.save_args()
  
        ######################################################################################

        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        """define loss"""
        self.BceDiceLoss = BceDiceLoss().cuda()

        """define optimizer"""
        self.optimizer = self.set_optimizer()
        # self.optimizer = optim.SGD(self.network.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0001)

        """define lr_scheduler"""
        self.lr_scheduler = self.set_lr_scheduler()




    def train(self, train_loader, test_loader):
        print('#----------Training----------#')
        best = 0.0

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

            pred, gt, val_loss= self.val_epoch(test_loader)

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
            cldice_mean =np.mean(cldice_ls) 

            csv = 'test_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f \n' % (
                    dice_mean,
                    Jac_mean,
                    cldice_mean,
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
        iter_num = 0
        max_iterations = self.args.epochs * len(train_loader)

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

            ## base_lr 0.05 missformer
            # now_lr = self.args.lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = now_lr

            loss_list.append(loss.item())
            now_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            
            if iter % self.args.print_interval == 0:
                self.save_print_loss_lr(iter, epoch, loss_list, now_lr)

        self.lr_scheduler.step() 
        return step
    

    def val_epoch(self, test_loader):
        self.network.eval()
        pred_ls = []
        gt_ls = []
        loss_list = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images)

                loss = self.BceDiceLoss(preds, targets)

                loss_list.append(loss.item())
                gt_ls.append(targets.squeeze(1).cpu().detach().numpy())

                if type(preds) is tuple:
                    preds = preds[0]
                preds = preds.squeeze(1).cpu().detach().numpy()
                pred_ls.append(preds)

        return pred_ls, gt_ls, np.mean(loss_list)
    

    """dconnet + ege-unet"""
    def test_epoch(self, test_loader):
        self.network.eval()
        pred_ls = []
        gt_ls = []
        self.cldice_ls = []

        self.dice_ls = []
        self.Jac_ls=[]
        # self.cldice_ls = []
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
                # loss = self.BceDiceLoss(preds, targets) 


                output = F.sigmoid(preds)
                output_ = torch.where(output>0.3,1,0)
                # gt_ = F.sigmoid(targets)
                gt__ = torch.where(targets>0.3,1,0)
                pred_np = output_.squeeze().cpu().numpy()
                target_np = gt__.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                self.cldice_ls.append(cldc)

                #########################################################
                dice, Jac, acc, sen, spe, pre, recall, f1_score = self.per_class_metric(output_, gt__)
                # self.cldice_ls.append(cldc)

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

                # if iter % self.args.save_interval == 0:
                #     save_path = self.args.res_dir
                #     self.save_img(images, gt__, output_, iter, save_path)

            return pred_ls, gt_ls, self.cldice_ls
                
    """"ffm"""
    # def test_epoch(self, test_loader):
    #     self.network.eval()
    #     total_img = 0
    #     total_iou = 0.0
    #     total_f1 = 0.0
    #     total_acc = 0.0
    #     total_sen = 0.0
    #     # total_auc = 0.0
    #     total_spec = 0.0

    #     cldices = []
    #     hds = []
    #     betti_losses = []
    #     pool = Pool(8)

    #     with torch.no_grad():
    #         for iter, data in enumerate(tqdm(test_loader)):
    #             images, targets = data
    #             images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

    #             preds = self.network(images)

    #             preds_ = preds.squeeze().cpu()
    #             targets_ = targets.squeeze().cpu()

    #             y_scores = np.zeros_like(preds_)
    #             y_true = np.zeros_like(targets_)
    #             y_true[preds_ > 0.01] = 1
    #             y_scores[targets_ > 0.3] = 1
    #             hd = hausdorff_distance(y_scores, y_true)

    #             cldc = clDice(y_scores, y_true)
    #             cldices.append(cldc)
    #             betti_losses.append(pool.apply_async(compute_metrics, args=(y_scores, y_true,)))
               
    #             y_scores1 = y_scores.flatten()
    #             y_true1 = y_true.flatten()

    #             hds.append(hd)

    #             confusion = confusion_matrix(y_true1, y_scores1)
    #             tp = float(confusion[1, 1])
    #             fn = float(confusion[1, 0])
    #             fp = float(confusion[0, 1])
    #             tn = float(confusion[0, 0])
    
    #             val_acc = (tp + tn) / (tp + fn + fp + tn)
    #             sensitivity = tp / (tp + fn)
    #             specificity = tn / (tn + fp)
    #             precision = tp / (tp + fp)
    #             f1 = 2 * sensitivity * precision / (sensitivity + precision + 1e-9)
    #             iou = tp / (tp + fn + fp + 1e-9)
    #             # auc = calc_auc(now_img, gt_arr)
    #             total_iou += iou
    #             total_acc += val_acc
    #             total_f1 += f1
    #             # total_auc += auc
    #             total_sen += sensitivity
    #             total_spec += specificity
    #             total_img += 1


    #             preds_img = F.sigmoid(preds)
    #             # preds_img = torch.where(preds>0.5,1,0)

    #             if iter % self.args.save_interval == 0:
    #                 save_path = self.args.res_dir
    #                 self.save_img(images, targets, preds_img, iter, save_path)
   

    #     epoch_iou = (total_iou) / total_img
    #     epoch_f1 = total_f1 / total_img
    #     epoch_acc = total_acc / total_img
    #     # epoch_auc = total_auc / total_img
    #     epoch_sen = total_sen / total_img
    #     epoch_spec = total_spec / total_img
    #     epoch_clDice = np.mean(cldices)
    #     epoch_hd = np.mean(hds)
    #     message = "inference  =====> Evaluation  ACC: {:.4f}; IOU: {:.4f}; F1_score: {:.4f}; Sen: {:.4f}; Spec: {:.4f}; clDice: {:.4f}; hausdorff_distance: {:.4f};".format(
    #         epoch_acc,
    #         epoch_iou,
    #         epoch_f1, epoch_sen, epoch_spec, epoch_clDice, epoch_hd)

    #     print("==> %s" % (message))

    #     pool.close()
    #     pool.join()

        # betti_results = []
        # for if_index in range(total_img):
        #     betti_result_now = betti_losses[if_index].get()
        #     betti_results.append(betti_result_now)
        # betti_losses_array2 = np.array(betti_results)
        # betti_mean = np.mean(betti_losses_array2, axis=0)
        # Betti_error = betti_mean[3]
        # Betti_0_error = betti_mean[4]
        # Betti_1_error = betti_mean[5]

        # print("Betti number error: ", Betti_error)
        # print("Betti number error dim 0: ", Betti_0_error)
        # print("Betti number error dim 1: ", Betti_1_error)






    # def test(self, train_loader, test_loader):

    #     if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best.pth')):
    #         print('#----------Testing----------#')

    #         best_weight = torch.load(self.args.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
    #         self.network.load_state_dict(best_weight)

    #         self.test_epoch(test_loader)

    #         torch.cuda.empty_cache()

    """rolling"""
    # def test_epoch(self, test_loader):
    #     self.network.eval()
    #     iou_avg_meter = AverageMeter()
    #     dice_avg_meter = AverageMeter()
    #     hd_avg_meter = AverageMeter()
    #     hd95_avg_meter = AverageMeter()
    #     recall_avg_meter = AverageMeter()
    #     specificity_avg_meter = AverageMeter()
    #     precision_avg_meter = AverageMeter()

    #     with torch.no_grad():
    #         for iter, data in enumerate(tqdm(test_loader)):
    #             images, targets = data
    #             images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

    #             preds = self.network(images)

    #             iou, dice, hd, hd95, recall, specificity, precision = indicators(preds, targets)
    #             iou_avg_meter.update(iou, images.size(0))
    #             dice_avg_meter.update(dice, images.size(0))
    #             hd_avg_meter.update(hd, images.size(0))
    #             hd95_avg_meter.update(hd95, images.size(0))
    #             recall_avg_meter.update(recall, images.size(0))
    #             specificity_avg_meter.update(specificity, images.size(0))
    #             precision_avg_meter.update(precision, images.size(0))

    #             output = torch.sigmoid(preds).cpu().numpy()
    #             output[output >= 0.5] = 1
    #             output[output < 0.5] = 0

    #     print('IoU: %.4f' % iou_avg_meter.avg)
    #     print('Dice: %.4f' % dice_avg_meter.avg)
    #     print('Hd: %.4f' % hd_avg_meter.avg)
    #     print('Hd95: %.4f' % hd95_avg_meter.avg)
    #     print('Recall: %.4f' % recall_avg_meter.avg)
    #     print('Specificity: %.4f' % specificity_avg_meter.avg)
    #     print('Precision: %.4f' % precision_avg_meter.avg)
    
    #     torch.cuda.empty_cache()

    










    # def inference(self, test_loader, epoch):
    #     self.network.eval()
    #     pred = []
    #     gts = []
    #     loss_list = []
    #     threshold = 0.5

    #     with torch.no_grad():
    #         for iter, data in enumerate(test_loader):
    #             images, targets = data
    #             images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

    #             preds = self.network(images)
    #             loss = self.BceDiceLoss(preds, targets)

    #             loss_list.append(loss.item())
    #             gts.append(targets.squeeze(1).cpu().detach().numpy())

    #             if type(preds) is tuple:
    #                 preds = preds[0]
    #             preds = preds.squeeze(1).cpu().detach().numpy()
    #             pred.append(preds) 

    #             if self.args.phase == 'test':
    #                 save_path = self.args.res_dir
    #                 if iter % self.args.save_interval == 0:
    #                     self.save_imgs(images, targets, preds, iter, save_path, self.args.dataset)

    #     pred = np.array(pred).reshape(-1)   ## 展成一维数组
    #     gts = np.array(gts).reshape(-1)

    #     y_pre = np.where(pred >= threshold, 1, 0)
    #     y_true = np.where(gts>=0.5, 1, 0)

    #     if self.args.phase == 'train':

    #         if epoch % self.args.val_interval == 0:

    #             confusion, accuracy, sensitivity, specificity, dsc, miou, precision, recall, f1_score = get_metric(y_pre, y_true)

    #             log_info = f'val epoch: {epoch}, mIoU: {miou:.4f}, DSC: {dsc:.4f}, acc: {accuracy:.4f}, spe: {specificity:.4f}, \
    #             sen_or_recall: {sensitivity:.4f}, precision: {precision:.4f}, F1_score: {f1_score:.4f}, confusion_matrix: {confusion}'         
    #             print(log_info)         
    #             self.logger.info(log_info)
            
    #         else:
    #             dsc = get_dsc(y_pre, y_true)

    #             log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
    #             print(log_info)
    #             self.logger.info(log_info)


    #     if self.args.phase == 'test':     

    #         confusion, accuracy, sensitivity, specificity, dsc, miou, precision, recall, f1_score = get_metric(y_pre, y_true)

    #         log_info = f'test of best model, mIoU: {miou:.4f}, DSC: {dsc:.4f}, acc: {accuracy:.4f}, spe: {specificity:.4f}, sen_or_recall: {sensitivity:.4f}, \
    #              precision: {precision:.4f}, F1_score: {f1_score:.4f}, confusion_matrix: {confusion}'    
    #         print(log_info)
    #         self.logger.info(log_info)

    #     return np.mean(loss_list), dsc     

def compute_metrics(y_scores, y_true, relative=True, comparison='union', filtration='superlevel', construction='V'):
    BM = BettiMatching(y_scores, y_true, relative=relative, comparison=comparison, filtration=filtration,
                       construction=construction)

    return [BM.loss(dimensions=[0, 1]), BM.loss(dimensions=[0]), BM.loss(dimensions=[1]), BM.Betti_number_error(
        threshold=0.5, dimensions=[0, 1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(
        threshold=0.5, dimensions=[1])]

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_