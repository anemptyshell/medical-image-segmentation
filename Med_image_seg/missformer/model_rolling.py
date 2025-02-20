import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.utils.data
import numpy as np
import cv2
# import random
# import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import pandas as pd

from libs.utils import AverageMeter, str2bool

# from libs.metric import metric
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
        log = OrderedDict([
        ('epoch', []),
        ('dice', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ])

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

            # self.step = self.train_epoch(train_loader, epoch, self.step)
            train_log = self.train_epoch(train_loader, epoch, self.step)
            # dice, Jac, cldice= self.val_epoch(test_loader)
            val_log, cldice= self.val_epoch(test_loader)

            ##########################################
            print('loss %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], val_log['loss'], val_log['iou'], val_log['dice']))
  
            log['epoch'].append(epoch)
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['dice'].append(train_log['dice'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])

            pd.DataFrame(log).to_csv('res_log.csv', index=False)

            ##########################################
            # dice_ls = np.array(dice)
            # Jac_ls = np.array(Jac)
            # total_dice = np.mean(dice_ls)
            csv = 'val_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%03d,%0.6f \n' % (
                    (epoch),
                    #total_dice,
                    #np.mean(Jac_ls),
                    np.mean(cldice)
                ))
            ##########################################

            # if best_dice < np.mean(dice):
            #     best_dice = np.mean(dice)
            #     torch.save(self.network.state_dict(), os.path.join(self.args.checkpoint_dir, 'best.pth'))
            #     log_info = f'Epoch: {epoch}, Total DSC: {np.mean(dice):.4f}, IOU: {np.mean(Jac):.4f}, clDice: {np.mean(cldice):.4f}'
            #     print(log_info)
            #     self.logger.info(log_info)
            #     print('save best!')
            if best_dice < val_log['dice']:
                best_dice = val_log['dice']
                torch.save(self.network.state_dict(), os.path.join(self.args.checkpoint_dir, 'best.pth'))
                log_info = f'Epoch: {epoch}, Total DSC: {best_dice:.4f}, clDice: {np.mean(cldice):.4f}'
                print(log_info)
                self.logger.info(log_info)
                print('save best!')
            
            torch.cuda.empty_cache()


    def test(self, train_loader, test_loader):
        if os.path.exists(os.path.join(self.args.checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')

            best_weight = torch.load(self.args.work_dir + '/checkpoints/best.pth', map_location=torch.device('cpu'))
            self.network.load_state_dict(best_weight)

            cldice_ls = self.test_epoch(test_loader)
            cldice_mean =np.mean(cldice_ls) 
            print('cldice: %.4f'%cldice_mean)


            csv = 'test_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%0.6f \n' % (
                    cldice_mean,
                ))
    
            torch.cuda.empty_cache()


    def train_epoch(self, train_loader, epoch, step):

        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dice': AverageMeter()}
        
        self.network.train()
        loss_list = [] 
        # iter_num = 0
        # max_iterations = self.args.epochs * len(train_loader)
        pbar = tqdm(total=len(train_loader))

        for iter, data in enumerate(train_loader):
            step += iter
            
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

            preds = self.network(images)

            loss = self.BceDiceLoss(preds, targets)
            iou, dice = iou_score(preds, targets)
            
            ## compute gradient and do optimizing step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()

            ## base_lr 0.05 missformer
            # now_lr = self.args.lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = now_lr

            # loss_list.append(loss.item())
            avg_meters['loss'].update(loss.item(), images.size(0))
            avg_meters['iou'].update(iou, images.size(0))
            avg_meters['dice'].update(dice, images.size(0))
            now_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)  
            ])

            # torch.cuda.empty_cache()
            # iter_num = iter_num + 1     
            pbar.set_postfix(postfix)
            pbar.update(1)
           
            # if iter % self.args.print_interval == 0:
            #     self.save_print_loss_lr(iter, epoch, loss_list, now_lr)

        pbar.close()
        self.lr_scheduler.step()
        # return step
        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg)])


    def val_epoch(self, test_loader):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dice': AverageMeter()}
        self.network.eval()
        # self.dice_ls = []
        # self.Jac_ls=[]
        self.cldice_ls = []

        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            # for data in tqdm(test_loader):
            for data in test_loader:
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

                preds = self.network(images) 
                
                loss = self.BceDiceLoss(preds, targets)
                iou, dice = iou_score(preds, targets)

                # dice, Jac = self.per_class_dice(preds, targets)
                avg_meters['loss'].update(loss.item(), images.size(0))
                avg_meters['iou'].update(iou, images.size(0))
                avg_meters['dice'].update(dice, images.size(0))

                ########################                                          
                output = F.sigmoid(preds)
                output_ = torch.where(output>0.5,1,0)
               
                pred_np = output_.squeeze().cpu().numpy()
                target_np = targets.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                self.cldice_ls.append(cldc)
                ########################

                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                    ('dice', avg_meters['dice'].avg)
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()

                # self.dice_ls += dice[:,0].tolist()
                # self.Jac_ls += Jac[:,0].tolist()
        # return self.dice_ls, self.Jac_ls, self.cldice_ls
        return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)]), self.cldice_ls


    def test_epoch(self, test_loader):
        self.network.eval()
        self.cldice_ls = []

        iou_avg_meter = AverageMeter()
        dice_avg_meter = AverageMeter()
        hd_avg_meter = AverageMeter()
        hd95_avg_meter = AverageMeter()
        recall_avg_meter = AverageMeter()
        specificity_avg_meter = AverageMeter()
        precision_avg_meter = AverageMeter()
        sensitivity_avg_meter = AverageMeter()

        with torch.no_grad():
            for iter, data in enumerate(tqdm(test_loader)):
                images, targets = data
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
                

                preds = self.network(images)

                iou, dice, hd, hd95, recall, specificity, precision, sensitivity = indicators(preds, targets)
                iou_avg_meter.update(iou, images.size(0))
                dice_avg_meter.update(dice, images.size(0))
                hd_avg_meter.update(hd, images.size(0))
                hd95_avg_meter.update(hd95, images.size(0))
                recall_avg_meter.update(recall, images.size(0))
                specificity_avg_meter.update(specificity, images.size(0))
                precision_avg_meter.update(precision, images.size(0))
                sensitivity_avg_meter.update(sensitivity, images.size(0))


                # dice, Jac, acc, sen, spe, pre, recall, f1_score = self.per_class_metric(preds, targets)
                ################################################ 
                output = F.sigmoid(preds)
                output_ = torch.where(output>0.5,1,0)
                # gt = F.sigmoid(targets)
                # gt_ = torch.where(gt>0.5,1,0)
                pred_np = output_.squeeze().cpu().numpy()
                target_np = targets.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                # print('cldc:',cldc)
                self.cldice_ls.append(cldc)
                ################################################ 

                # preds = torch.sigmoid(preds).cpu().numpy()
                # preds[preds >= 0.5] = 1
                # preds[preds < 0.5] = 0

                # for i in range(len(output)):
                #     cv2.imwrite(os.path.join('outputs', args.name, str(c), meta['img_id'][i] + '.png'),
                #                 (preds[i, c] * 255).astype('uint8'))


                if iter % self.args.save_interval == 0:
                    save_path = self.args.res_dir
                    self.save_img(images, targets, output_, iter, save_path)

        print('IoU: %.4f' % iou_avg_meter.avg)
        print('Dice: %.4f' % dice_avg_meter.avg)
        print('Hd: %.4f' % hd_avg_meter.avg)
        print('Hd95: %.4f' % hd95_avg_meter.avg)
        print('Recall: %.4f' % recall_avg_meter.avg)
        print('Specificity: %.4f' % specificity_avg_meter.avg)
        print('Precision: %.4f' % precision_avg_meter.avg)
        print('Sensitivity: %.4f' % sensitivity_avg_meter.avg)
    
        torch.cuda.empty_cache()

        return self.cldice_ls


from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision, sensitivity

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    # target_ = target > 0.5
    # output_ = output
    target_ = target
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    # target_ = target

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)
    sensitivity_ = sensitivity(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, sensitivity_
