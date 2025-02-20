
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim 
import torch.utils.data


# from libs.metric import metric
from libs.base_model import base_model
from libs.metric import metric

from Med_image_seg.mdnet.util import warmup_learning_rate
from Med_image_seg.fang.utils.cldice import clDice
from Med_image_seg.mdnet.loss import soft_iou_loss
from Med_image_seg.mdnet.network import Multi_decoder_Net


def arguments():
    args = {
    '--num_classes': 1,
    '--num_channels': 3,
    '--warmup_step': 1000,
    '--warmup_method': 'exp'
    ### lr 0.05
}  
    return args

class mdnet(base_model):
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

        self.network = Multi_decoder_Net(self.args.num_channels, self.args.num_classes).to('cuda')
        self.step = 0
        self.save_args()
  
        ######################################################################################
        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        """define loss"""
        self.bceloss = nn.BCELoss().cuda()
        self.iouloss = soft_iou_loss.cuda()

        """define optimizer"""
        self.optimizer = self.set_optimizer()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

        """define lr_scheduler"""
        self.lr_scheduler = self.set_lr_scheduler()



    def train(self, train_loader, test_loader):
        print('#----------Training----------#')
        best_dice = 0.0

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
            dice, Jac, cldice= self.val_epoch(test_loader)



    def train_epoch(self, train_loader, epoch, step):
        
        self.network.train()
        loss_list = [] 
        smooth_loss = 0.0
        current_step = 0

        for iter, data in enumerate(train_loader):

            # step += iter
            
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
            # skeletons = inputs['skeleton'].to(device)
            # edges = inputs['edge'].to(device)

            preds, pred_skeleton, pred_edge = self.network(images)
            loss = self.BceDiceLoss(preds, targets)

            loss1 = self.iouloss(preds, targets)
            loss2 = self.bceloss(pred_skeleton, skeletons)
            loss3 = self.bceloss(pred_edge, edges)
            loss = loss1 + 0.5 * loss2 + 0.5 * loss3

            step += 1
            current_step += 1
            smooth_loss += loss.item()

            ## compute gradient and do optimizing step
            self.optimizer.zero_grad()
            lr_update = warmup_learning_rate(self.optimizer, step, self.args.warmup_step, self.args.lr, self.args.warmup_method)
            loss.backward()
            self.optimizer.step()
            
            if step % self.args.print_interval == 0:    ## 10
                smooth_loss = smooth_loss / current_step
                self.save_print_loss_lr(iter, epoch, smooth_loss, lr_update)
                current_step = 0
                smooth_loss = 0.0
            
        return step
    

    def val_epoch(self, test_loader):
        self.network.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldice_ls = []
        # pred = []
        # gt = []
        # loss_list = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images)
                # loss = self.BceDiceLoss(preds, targets)
                dice, Jac = self.per_class_dice(preds, targets)

                pred_np = preds.squeeze().cpu().numpy()
                target_np = targets.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                self.cldice_ls.append(cldc)

                self.dice_ls += dice[:,0].tolist()
                self.Jac_ls += Jac[:,0].tolist()


        return self.dice_ls, self.Jac_ls, self.cldice_ls

