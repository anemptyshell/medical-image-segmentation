import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.utils.data
import numpy as np
import random
from torch.nn import functional as F
from collections import OrderedDict


from libs.metric import metric
from libs.base_model import base_model
from libs.utils import AverageMeter
from Med_image_seg.rollingunet.loss import BceDiceLoss
from Med_image_seg.rollingunet.network import Rolling_Unet_L
from libs.metric_utils.cldice import clDice
from libs.metric_utils.BettiMatching import BettiMatching


# from sklearn.metrics import roc_auc_score, confusion_matrix
# from hausdorff import hausdorff_distance
# from multiprocessing import Pool
# from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision


def arguments():
    args = {
    '--num_classes': 1, 
    '--deep_supervision': False
}  
    return args


class rollingunet(base_model):
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
        # self.set_seed(self.args.seed)
        self.set_cuda()
        torch.cuda.empty_cache()

        ######################################################################################
        """ Trainer """ 
        print('#----------Prepareing Model----------#')

        self.network = Rolling_Unet_L(num_classes=1, img_size=self.args.img_size).to('cuda')
        self.step = 0
        self.save_args()
  
        ######################################################################################

        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        """define loss"""
        self.BceDiceLoss = BceDiceLoss().cuda()

        """define optimizer"""
        # self.optimizer = self.set_optimizer()
        params = filter(lambda p: p.requires_grad, self.network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        """define lr_scheduler"""
        # self.lr_scheduler = self.set_lr_scheduler()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_lr)


    def create_exp_directory(self):
        csv = 'val_results'+'.csv'
        with open(os.path.join(self.args.log_dir, csv), 'w') as f:
            f.write('epoch, dice, Jac, clDice \n')
        csv1 = 'test_results'+'.csv'
        with open(os.path.join(self.args.log_dir, csv1), 'w') as f:
            f.write('dice, Jac, clDice, acc, sen, spe, pre, recall, f1 \n')


    def train(self, train_loader, test_loader):
        print('#----------Training----------#')
        seed_torch()
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

        for epoch in range(start_epoch, self.args.epochs + 1):           
            torch.cuda.empty_cache()
            print('---------------------------------------')
 
            self.step = self.train_epoch(train_loader, epoch, self.step)
            dice, Jac, cldice, pred, gt = self.val_epoch(test_loader)

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



    def train_epoch(self, train_loader, epoch, step):
        self.network.train()
        loss_list = []

        for iter, data in enumerate(train_loader):
            step += iter
            
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

            # preds = self.network(images)

            if self.args.deep_supervision:
                outputs = self.network(images)
                loss = 0
                for preds in outputs:
                    loss += self.BceDiceLoss(preds, targets)
                loss /= len(outputs)
            else:
                preds = self.network(images)
                loss = self.BceDiceLoss(preds, targets)

            ## compute gradient and do optimizing step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            now_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            if iter % self.args.print_interval == 0:
                self.save_print_loss_lr(iter, epoch, loss_list, now_lr)
            torch.cuda.empty_cache()

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
    


    """ege-unet / dconnnet 指标"""
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
                dice,Jac,acc,sen,spe,pre,recall,f1_score=self.per_class_metric(preds, targets)

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
        epoch_clDice = np.mean(self.cldice_ls)

        return pred_ls, gt_ls, epoch_clDice


    """ffm"""
    # def test(self, train_loader, test_loader):
    #     if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best.pth')):
    #         print('#----------Testing----------#')

    #         best_weight = torch.load(self.args.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
    #         self.network.load_state_dict(best_weight)

    #         self.test_epoch(test_loader)
    #         torch.cuda.empty_cache()

    
    # def test_epoch(self, test_loader):
    #     self.network.eval()
    #     total_img = 0
    #     total_iou = 0.0
    #     total_f1 = 0.0
    #     total_acc = 0.0
    #     total_sen = 0.0
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

    #     betti_results = []
    #     for if_index in range(total_img):
    #         betti_result_now = betti_losses[if_index].get()
    #         betti_results.append(betti_result_now)
    #     betti_losses_array2 = np.array(betti_results)
    #     betti_mean = np.mean(betti_losses_array2, axis=0)
    #     Betti_error = betti_mean[3]
    #     Betti_0_error = betti_mean[4]
    #     Betti_1_error = betti_mean[5]

    #     print("Betti number error: ", Betti_error)
    #     print("Betti number error dim 0: ", Betti_0_error)
    #     print("Betti number error dim 1: ", Betti_1_error)


    """rolling"""
    # def test(self, train_loader, test_loader):

    #     if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best.pth')):
    #         print('#----------Testing----------#')

    #         best_weight = torch.load(self.args.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
    #         self.network.load_state_dict(best_weight)
    #         self.test_epoch(test_loader)
    #         torch.cuda.empty_cache()


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




    

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice

def compute_metrics(y_scores, y_true, relative=True, comparison='union', filtration='superlevel', construction='V'):
    BM = BettiMatching(y_scores, y_true, relative=relative, comparison=comparison, filtration=filtration,
                       construction=construction)

    return [BM.loss(dimensions=[0, 1]), BM.loss(dimensions=[0]), BM.loss(dimensions=[1]), BM.Betti_number_error(
        threshold=0.5, dimensions=[0, 1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(
        threshold=0.5, dimensions=[1])]


# def indicators(output, target):
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5

#     iou_ = jc(output_, target_)
#     dice_ = dc(output_, target_)
#     hd_ = hd(output_, target_)
#     hd95_ = hd95(output_, target_)
#     recall_ = recall(output_, target_)
#     specificity_ = specificity(output_, target_)
#     precision_ = precision(output_, target_)

#     return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_

