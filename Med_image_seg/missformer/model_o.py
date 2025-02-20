import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.utils.data
import numpy as np
import random
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss

from libs.metric import metric
from libs.base_model import base_model

from Med_image_seg.missformer.loss import BceDiceLoss
from Med_image_seg.missformer.network import MISSFormer
from libs.metric_utils.cldice import clDice

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

        print('#----------GPU init----------#')
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

        self.BceDiceLoss = BceDiceLoss().cuda()

        # self.optimizer = optim.SGD(self.network.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0001)
        self.optimizer = self.set_optimizer()
        self.lr_scheduler = self.set_lr_scheduler()


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
            dice, Jac, cldice= self.val_epoch(test_loader)

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
                    np.mean(cldice)
                ))
            ##########################################

            if best_dice < np.mean(dice):
                best_dice = np.mean(dice)
                torch.save(self.network.state_dict(), os.path.join(self.args.checkpoint_dir, 'best.pth'))
                log_info = f'Epoch: {epoch}, Total DSC: {np.mean(dice):.4f}, IOU: {np.mean(Jac):.4f}, clDice: {np.mean(cldice):.4f}'
                print(log_info)
                self.logger.info(log_info)
                print('save best!')


    def test(self, train_loader, test_loader):
        if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')

            best_weight = torch.load(self.args.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
            self.network.load_state_dict(best_weight)

            cldice_ls = self.test_epoch(test_loader)

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
    
            torch.cuda.empty_cache()


    def train_epoch(self, train_loader, epoch, step):
        
        self.network.train()
        loss_list = [] 
        iter_num = 0
        # max_iterations = self.args.epochs * len(train_loader)

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

            # torch.cuda.empty_cache()
            iter_num = iter_num + 1        
            
            if iter % self.args.print_interval == 0:
                self.save_print_loss_lr(iter, epoch, loss_list, now_lr)
       
        self.lr_scheduler.step()
        return step


    def val_epoch(self, test_loader):
        self.network.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images)                                             
                preds_ = F.sigmoid(preds)
                preds__ = torch.where(preds_>0.5,1,0)

                dice, Jac = self.per_class_dice(preds, targets)

                pred_np = preds__.squeeze().cpu().numpy()
                target_np = targets.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                self.cldice_ls.append(cldc)

                self.dice_ls += dice[:,0].tolist()
                self.Jac_ls += Jac[:,0].tolist()

        return self.dice_ls, self.Jac_ls, self.cldice_ls


    def test_epoch(self, test_loader):
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

        with torch.no_grad():
            for iter, data in enumerate(tqdm(test_loader)):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
                

                preds = self.network(images)
                preds = F.sigmoid(preds)
                preds = torch.where(preds>0.5,1,0)
                # targets = torch.where(targets>0.5,1,0)


                dice, Jac, acc, sen, spe, pre, recall, f1_score = self.per_class_metric(preds, targets)

                pred_np = preds.squeeze().cpu().numpy()
                target_np = targets.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                # print('cldc:',cldc)
                self.cldice_ls.append(cldc)

                self.dice_ls += dice[:,0].tolist()
                self.Jac_ls += Jac[:,0].tolist()
                self.acc_ls += acc[:,0].tolist()
                self.sen_ls += sen[:,0].tolist()
                self.spe_ls += spe[:,0].tolist()
                self.pre_ls += pre[:,0].tolist()
                self.recall_ls += recall[:,0].tolist()
                self.f1_ls += f1_score[:,0].tolist()


                if iter % self.args.save_interval == 0:
                    save_path = self.args.res_dir
                    self.save_img(images, targets, preds, iter, save_path)


            return self.cldice_ls





