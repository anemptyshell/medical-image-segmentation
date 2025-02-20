from tqdm import tqdm
import os
import torch
import torch.nn
import torch.optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from libs.metric import metric
from libs.base_model import base_model

from Med_image_seg.cgmanet.loss import structure_loss
from Med_image_seg.cgmanet.network import CGMA
from Med_image_seg.cgmanet.utils import clip_gradient, adjust_lr
from Med_image_seg.fang.utils.cldice import clDice


def arguments():
    args = {
    '--num_classes': 1, 
    '--clip': 0.5,
    '--trainsize': 28
}  
    return args



class cgmanet(base_model):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())

        print('#----------Creating logger----------#')
        ## make work_dir ,log_dir, res_dir, checkpoint_dir
        self.make_dir(parser)
        self.args = parser.get_args()

        print('#----------GPU init----------#')
        # self.set_seed(self.args.seed)
        self.set_cuda()
        torch.cuda.empty_cache()

        ######################################################################################
        """ Trainer """ 
        print('#----------Prepareing Model----------#')

        self.network = CGMA().to('cuda')
        self.step = 0
        self.save_args()

        total = sum([param.nelement() for param in self.network.parameters()])
        print('Number of parameter: %.2fM' % (total / 1e6))
  
        ######################################################################################

        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        # self.optimizer = self.set_optimizer()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.lr, weight_decay=1e-4, momentum=0.9)
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
        size_rates = [1]

        # now_lr = adjust_lr(self.optimizer, self.args.lr, epoch, 0.1, 200)

        for iter, data in enumerate(train_loader):
            for rate in size_rates:
                self.optimizer.zero_grad()
                step += iter
            
                images, targets = data

                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                ## rescale
                trainsize = int(round(self.args.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    targets = F.upsample(targets, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                preds = self.network(images)
                total_loss = structure_loss(preds[0], targets)

                for _, _pred in enumerate(preds):
                    if _ == 0:
                        continue
                    loss = structure_loss(_pred, targets)
                    total_loss += loss

                ## compute gradient and do optimizing step
                self.optimizer.zero_grad()
                total_loss.backward()
                # clip_gradient(self.optimizer, self.args.clip)
                self.optimizer.step()
                torch.cuda.empty_cache()

                loss_list.append(total_loss.item())
                now_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            
                if iter % self.args.print_interval == 0:
                    self.save_print_loss_lr(iter, epoch, loss_list, now_lr)
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
                # targets = np.asarray(targets, np.float32)
                # targets /= (targets.max() + 1e-8)

                preds = self.network(images)
                ## preds是一个列表

                gt_ls.append(targets.squeeze(1).cpu().detach().numpy())
                preds = preds[0].squeeze(1).cpu().detach().numpy()
                pred_ls.append(preds)

        return pred_ls, gt_ls
    
    
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


                output = F.sigmoid(preds[0])
                output_ = torch.where(output>0.3,1,0)
                # gt_ = F.sigmoid(targets)
                gt__ = torch.where(targets>0.3,1,0)
                pred_np = output_.squeeze().cpu().numpy()
                target_np = gt__.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                self.cldice_ls.append(cldc)

                #########################################################
                dice, Jac, acc, sen, spe, pre, recall, f1_score = self.per_class_metric(output_, gt__)

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
                # if type(preds) is tuple:
                #     preds = preds[0]
                preds_np = preds[0].squeeze(1).cpu().detach().numpy()
                pred_ls.append(preds_np) 

                save_path = self.args.res_dir
                if iter % self.args.save_interval == 0:
                    self.save_imgs(images, targets_np, preds_np, iter, save_path)

            return pred_ls, gt_ls, self.cldice_ls


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
