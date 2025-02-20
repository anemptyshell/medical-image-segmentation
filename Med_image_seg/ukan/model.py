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
from torch.nn import CrossEntropyLoss, Softmax

from libs.metric import metric
from libs.base_model import base_model
from libs.utils import AverageMeter
from Med_image_seg.ukan.loss import BceDiceLoss # , GDiceLossV2
from Med_image_seg.ukan.network import UKAN
from libs.metric_utils.cldice import clDice
from libs.metric_utils.BettiMatching import BettiMatching


# from sklearn.metrics import roc_auc_score, confusion_matrix
# from hausdorff import hausdorff_distance
# from multiprocessing import Pool
# from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision



def arguments():
    args = {
    '--num_classes': 1, 
}  
    return args


class ukan(base_model):
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

        self.network = UKAN(num_classes=1, img_size=self.args.img_size).to('cuda')
        self.step = 0
        self.save_args()
  
        ######################################################################################

        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        """define loss"""
        self.BceDiceLoss = BceDiceLoss().cuda()
        # self.loss = CrossEntropyLoss().cuda()

        """define optimizer"""
        # self.optimizer = self.set_optimizer()
        params = filter(lambda p: p.requires_grad, self.network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        """define lr_scheduler"""
        # self.lr_scheduler = self.set_lr_scheduler()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_lr)


    def train(self, train_loader, test_loader):
        print('#----------Training----------#')
        seed_torch()
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

        for epoch in range(start_epoch, self.args.epochs + 1):
            
            torch.cuda.empty_cache()
            print('---------------------------------------')
 
            self.step = self.train_epoch(train_loader, epoch, self.step)

            pred, gt = self.val_epoch(test_loader)

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
          
            preds = self.network(images)
            loss = self.BceDiceLoss(preds, targets)
            # loss = self.loss(preds, targets)

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
        pred_ls = []
        gt_ls = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images)

                gt_ls.append(targets.squeeze(1).cpu().detach().numpy())

                if type(preds) is tuple:
                    preds = preds[0]
                preds = preds.squeeze(1).cpu().detach().numpy()
                pred_ls.append(preds)

        return pred_ls, gt_ls 
    


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
                dice,Jac,acc,sen,spe,pre,recall,f1_score=self.per_class_metric(output_,gt__)

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


