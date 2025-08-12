import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
from torch.nn import functional as F
import torch.optim
import torch.utils.data
from torch.optim import lr_scheduler

from libs.metric import metric
from libs.utils import AverageMeter
from libs.base_model import base_model

from Med_image_seg.fang.utils.loss import BceDiceLoss
from Med_image_seg.fang.network_old import UNet
from Med_image_seg.fang.utils.cldice import clDice

# from matplotlib import pyplot as plt
import numpy as np



def arguments():
    args = {
    '--lr_update': 'step',
    '--lr_step': 12
    }
    return args


class fang(base_model):
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

        self.network = UNet(n_channels=3).to('cuda')
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
    


    def test_epoch(self, test_loader):
        self.network.eval()
        pred_ls = []
        gt_ls = []
        self.cldice_ls = []

        with torch.no_grad():
            for iter, data in enumerate(tqdm(test_loader)):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images)
                # loss = self.BceDiceLoss(preds, targets) 


                output = F.sigmoid(preds)
                output_ = torch.where(output>0.5,1,0)
                # gt_ = F.sigmoid(targets)
                # gt__ = torch.where(gt_>0.5,1,0)
                pred_np = output_.squeeze().cpu().numpy()
                target_np = targets.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                self.cldice_ls.append(cldc)

                targets = targets.squeeze(1).cpu().detach().numpy()
                gt_ls.append(targets)
                if type(preds) is tuple:
                    preds = preds[0]
                preds = preds.squeeze(1).cpu().detach().numpy()
                pred_ls.append(preds) 


                save_path = self.args.res_dir
                if iter % self.args.save_interval == 0:
                    self.save_imgs(images, targets, preds, iter, save_path)

            return pred_ls, gt_ls, self.cldice_ls
    