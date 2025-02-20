import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.utils.data
import numpy as np

from libs.metric import metric
from libs.base_model import base_model

from Med_image_seg.selfregunet.loss import KDloss, DiceLoss, BceDiceLoss
from Med_image_seg.selfregunet.network import SelfRegUNet
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss


def arguments():
    args = {
    '--num_classes': 1, 
    '--lambda_x': 0.015
}  
    return args

class selfregunet(base_model):
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

        self.network = SelfRegUNet(n_channels=3, n_classes=self.args.num_classes, bilinear=True).to('cuda')
        self.step = 0
        self.save_args()
  
        ######################################################################################

        print('#----------Prepareing loss, opt, lr_sch and amp----------#')

        """define loss"""
        # self.BceDiceLoss = BceDiceLoss().cuda()
        self.ce_loss = CrossEntropyLoss().cuda()
        self.dice_loss = DiceLoss(self.args.num_classes).cuda()
        self.kd_loss = KDloss(lambda_x=self.args.lambda_x).cuda()

        """define optimizer"""
        self.optimizer = self.set_optimizer()
        # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

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

            pred, gt = self.test_epoch(test_loader)

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

            
            preds, kd_encorder, kd_decorder, final_up = self.network(images)
            # print(len(kd_encorder))       ## 4
            # print(type(kd_encorder[0]))   ## <class 'torch.Tensor'>
            # print(kd_encorder[0].shape)   ## torch.Size([2, 128, 112, 112])
            # print(len(kd_decorder))       ## 3
            # print(kd_decorder[0].shape)   ## torch.Size([2, 256, 28, 28])
            # print(final_up.size())        ## torch.Size([2, 64, 224, 224])

            loss_ce = self.ce_loss(preds, targets.float()) ## ???
            # print(preds.size(), targets.size())   ## torch.Size([2, 1, 224, 224]) torch.Size([2, 1, 224, 224])
            loss_dice = self.dice_loss(preds, targets, softmax=True)
            loss_kd = self.kd_loss(kd_encorder,kd_decorder,final_up)
            loss = 0.4 * loss_ce + 0.6 * loss_dice + loss_kd

            # loss = self.BceDiceLoss(preds, targets)

            ## compute gradient and do optimizing step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()

            now_lr = self.args.lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = now_lr
            # now_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            iter_num = iter_num + 1

            loss_list.append(loss.item())
 
            
            if iter % self.args.print_interval == 0:
                self.save_print_loss_lr(iter, epoch, loss_list, now_lr)

        self.lr_scheduler.step() 
        return step
    

    def val_epoch(self, test_loader):
        self.network.eval()
        pred = []
        gt = []
        # loss_list = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
                
                preds = self.network(images)
                
                # loss = None

                # loss_list.append(loss.item())
                gt.append(targets.squeeze(1).cpu().detach().numpy())

                if type(preds) is tuple:
                    preds = preds[0]
                preds = preds.squeeze(1).cpu().detach().numpy()
                pred.append(preds)

        return pred, gt, #loss #np.mean(loss_list)
    


    def test_epoch(self, test_loader):
        self.network.eval()
        pred = []
        gt = []
        with torch.no_grad():
            for iter, data in enumerate(tqdm(test_loader)):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images)
                loss = self.BceDiceLoss(preds, targets) 

                targets = targets.squeeze(1).cpu().detach().numpy()
                gt.append(targets)
                if type(preds) is tuple:
                    preds = preds[0]
                preds = preds.squeeze(1).cpu().detach().numpy()
                pred.append(preds) 

                images = images.squeeze(0).permute(1,2,0).detach().cpu().numpy()
                images = images / 255. if images.max() > 1.1 else images

                save_path = self.args.res_dir
                if iter % self.args.save_interval == 0:
                    self.save_imgs(images, targets, preds, iter, save_path, self.args.dataset)

            return pred, gt