import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from libs.metric import metric
from libs.utils import AverageMeter
from libs.base_model import base_model
from collections import OrderedDict

from Med_image_seg.unet.loss import BceDiceLoss
from Med_image_seg.unet0113.network import Multi_decoder_Net
from Med_image_seg.fang.utils.cldice import clDice

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt



def arguments():
    args = {
}  
    return args


class unet0113(base_model):
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

        self.network = Multi_decoder_Net(3).to('cuda')
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

    def generate_custom_skeleton_alternative(binary_image, a=1):
        """
        根据骨架点处的半径值来调整血管宽度
        - 半径 < a: 保留原血管 .
        - 半径 > a: 将血管宽度削减到半径a
        """

        edt = distance_transform_edt(binary_image)
        skeleton = skeletonize(binary_image)
        skeleton_radii = np.where(skeleton, edt, 0)

        # 创建自定义骨架区域
        custom_skeleton = np.zeros_like(binary_image, dtype=bool)

        # 获取所有骨架点坐标和半径
        y_coords, x_coords = np.where(skeleton)
        radii = skeleton_radii[skeleton]

        # 创建坐标网格
        y_grid, x_grid = np.indices(binary_image.shape)

        for y, x, radius in zip(y_coords, x_coords, radii):
            # 计算图像中每个像素到当前骨架点 (y, x) 的距离。
            dist_to_center = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)

            if radius >= a:
                # 半径大于a：创建半径为a的圆形区域
                circle_region = dist_to_center <= a
                custom_skeleton = np.logical_or(custom_skeleton, circle_region)

        # 对于半径<=a的区域，我们直接使用原始血管图像
        # 但需要确保这些区域不会被过度削减
        mask_radius_leq_a = np.zeros_like(binary_image, dtype=bool)

        for y, x, radius in zip(y_coords, x_coords, radii):
            if radius <= a:
                dist_to_center = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
                circle_region = dist_to_center <= radius  # 使用原始半径
                mask_radius_leq_a = np.logical_or(mask_radius_leq_a, circle_region)

        # 合并结果：半径<=a的区域 + 半径>a的削减区域
        custom_skeleton = np.logical_or(custom_skeleton, mask_radius_leq_a) 
        # 确保不超过原始血管边界
        custom_skeleton = np.logical_and(custom_skeleton, binary_image.astype(bool))

        return custom_skeleton


    def generate_custom_skeleton_alternative_tensor(self, binary_tensor, a=1):

        binary_image = binary_tensor.bool()
        
        edt_np = distance_transform_edt(binary_image.cpu().numpy())
        edt = torch.from_numpy(edt_np).to(binary_tensor.device)
        
        skeleton_np = skeletonize(binary_image.cpu().numpy())
        skeleton = torch.from_numpy(skeleton_np).to(binary_tensor.device)
        
        skeleton_radii = torch.where(skeleton, edt, torch.tensor(0, device=binary_tensor.device))

        y_grid, x_grid = torch.meshgrid(
            torch.arange(binary_image.size(0), device=binary_tensor.device),
            torch.arange(binary_image.size(1), device=binary_tensor.device)
        )
        
        # 获取所有骨架点坐标和半径
        y_coords, x_coords = torch.where(skeleton)
        radii = skeleton_radii[skeleton]
        
        custom_skeleton = torch.zeros_like(binary_image, dtype=torch.bool)
        
        # 处理半径>a的点
        mask_radius_gt_a = radii > a
        if mask_radius_gt_a.any():
            y_gt = y_coords[mask_radius_gt_a]
            x_gt = x_coords[mask_radius_gt_a]
            r_gt = radii[mask_radius_gt_a]
            
            dist_sq = (y_grid.unsqueeze(-1) - y_gt)**2 + (x_grid.unsqueeze(-1) - x_gt)**2
            dist = torch.sqrt(dist_sq)

            circles = dist <= a
            custom_skeleton = circles.any(dim=-1)
        
        mask_radius_le_a = radii <= a
        if mask_radius_le_a.any():
            y_le = y_coords[mask_radius_le_a]
            x_le = x_coords[mask_radius_le_a]
            r_le = radii[mask_radius_le_a]
            
            dist_sq = (y_grid.unsqueeze(-1) - y_le)**2 + (x_grid.unsqueeze(-1) - x_le)**2
            dist = torch.sqrt(dist_sq)

            circles = dist <= r_le
            custom_skeleton = torch.logical_or(custom_skeleton, circles.any(dim=-1))
        
        custom_skeleton = torch.logical_and(custom_skeleton, binary_image)
        
        return custom_skeleton



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
              ('val_hd', []),
              ('val_hd95', []),
              ('val_recall', []),
              ('val_spe', []),
              ('val_pre', []),
              ('val_sen', [])
              ])
      
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

            # self.step = self.train_epoch(train_loader, epoch, self.step)
            train_log = self.train_epoch(train_loader, epoch, self.step)
            val_log, cldice= self.val_epoch(test_loader, epoch)

            ##########################################
            print('epoch %d - loss %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (epoch, train_log['loss'], val_log['loss'], val_log['iou'], val_log['dice']))
  
            log['epoch'].append(epoch)
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['dice'].append(train_log['dice'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])
            log['val_hd'].append(val_log['hd'])
            log['val_hd95'].append(val_log['hd95'])
            log['val_recall'].append(val_log['recall'])
            log['val_spe'].append(val_log['spe'])
            log['val_pre'].append(val_log['pre'])
            log['val_sen'].append(val_log['sen'])

            log_file = os.path.join(self.args.log_dir, "res_log.csv")
            pd.DataFrame(log).to_csv(log_file, index=False)

            ##########################################

            csv = 'val_results'+'.csv'
            with open(os.path.join(self.args.log_dir, csv), 'a') as f:
                f.write('%03d,%0.6f \n' % (
                    (epoch),
                    np.mean(cldice)
                ))
            ##########################################
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


    def difference_loss(self, pred1, pred2, pred3):
        ideal_difference = pred1 - pred2
        return F.l1_loss(pred3, torch.clamp(ideal_difference, 0, 1))

    def train_epoch(self, train_loader, epoch, step):
        
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dice': AverageMeter()}
        
        self.network.train()

        pbar = tqdm(total=len(train_loader))

        for iter, data in enumerate(train_loader):
            step += iter
            
            images, targets, ske_strong, ske_alter, edge = data   
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
            ske_strong, ske_alter = ske_strong.cuda(non_blocking=True).float(), ske_alter.cuda(non_blocking=True).float()
            edge = edge.cuda(non_blocking=True).float()
            preds, pred_strong, pred_alter, w, out, loss_mi = self.network(images)

            loss1 = self.BceDiceLoss(preds, targets)
            loss2 = self.BceDiceLoss(pred_strong, ske_strong)
            loss3 = self.BceDiceLoss(pred_alter, ske_alter)
            loss_complement = self.BceDiceLoss(out, targets)

            loss = loss1 + 0.5 * loss2 + 0.5 * loss3 + loss_complement + 0.5*loss_mi
           
            iou, dice = iou_score(out, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()

            # loss_list.append(loss.item()
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


        pbar.close()
        self.lr_scheduler.step()
        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg)])
    


    def val_epoch(self, test_loader, epoch):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dice': AverageMeter(),
                      'hd':AverageMeter(),
                      'hd95':AverageMeter(),
                      'recall':AverageMeter(),
                      'spe':AverageMeter(),
                      'pre':AverageMeter(),
                      'sen':AverageMeter()}
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

                # preds, pred_strong, pred_alter, pred_edge = self.network(images)
                preds, pred_strong, pred_alter, w, out, loss_mi = self.network(images)
                
                loss = self.BceDiceLoss(out, targets)
                # iou, dice = iou_score(preds, targets)
                iou, dice, hd, hd95, recall, specificity, precision, sensitivity = indicators(out, targets, epoch)

                avg_meters['loss'].update(loss.item(), images.size(0))
                avg_meters['iou'].update(iou, images.size(0))
                avg_meters['dice'].update(dice, images.size(0))
                avg_meters['hd'].update(hd, images.size(0))
                avg_meters['hd95'].update(hd95, images.size(0))
                avg_meters['recall'].update(recall, images.size(0))
                avg_meters['spe'].update(specificity, images.size(0))
                avg_meters['pre'].update(precision, images.size(0))
                avg_meters['sen'].update(sensitivity, images.size(0))

                ########################                                          
                output = F.sigmoid(out)
                output_ = torch.where(output>0.5,1,0)
                gt_ = torch.where(targets>0.5,1,0)
                pred_np = output_.squeeze().cpu().numpy()
                target_np = gt_.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                self.cldice_ls.append(cldc)
                ########################

                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                    ('dice', avg_meters['dice'].avg),
                    ('hd', avg_meters['hd'].avg),
                    ('hd95', avg_meters['hd95'].avg),
                    ('recall', avg_meters['recall'].avg),
                    ('spe', avg_meters['spe'].avg),
                    ('pre', avg_meters['pre'].avg),
                    ('sen', avg_meters['sen'].avg)
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('hd', avg_meters['hd'].avg),
                        ('hd95', avg_meters['hd95'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('spe', avg_meters['spe'].avg),
                        ('pre', avg_meters['pre'].avg),
                        ('sen', avg_meters['sen'].avg)]), self.cldice_ls
    


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
                
                # preds, pred_strong, pred_alter, pred_edge = self.network(images)
                preds, pred_strong, pred_alter, w, out, loss_mi = self.network(images)

                iou, dice, hd, hd95, recall, specificity, precision, sensitivity = indicators_1(out, targets)
                iou_avg_meter.update(iou, images.size(0))
                dice_avg_meter.update(dice, images.size(0))
                hd_avg_meter.update(hd, images.size(0))
                hd95_avg_meter.update(hd95, images.size(0))
                recall_avg_meter.update(recall, images.size(0))
                specificity_avg_meter.update(specificity, images.size(0))
                precision_avg_meter.update(precision, images.size(0))
                sensitivity_avg_meter.update(sensitivity, images.size(0))

                ################################################ 
                output = F.sigmoid(out)
                output_ = torch.where(output>0.5,1,0)
                gt_ = torch.where(targets>0.5,1,0)
                pred_np = output_.squeeze().cpu().numpy()
                target_np = gt_.squeeze().cpu().numpy()
                cldc = clDice(pred_np, target_np)
                # print('cldc:',cldc)
                self.cldice_ls.append(cldc)
                ################################################ 


                size = self.args.img_size / 100
                if iter % self.args.save_interval == 0:
                    preds_com = torch.sigmoid(out).cpu().numpy()
                    preds_com[preds_com >= 0.5] = 1
                    preds_com[preds_com < 0.5] = 0
                    preds_com = np.squeeze(preds_com, axis=0)
                    preds_com = np.squeeze(preds_com, axis=0)


                    w = torch.sigmoid(w).cpu().numpy()
                    # w = w.cpu().numpy()
                    # w[w >= 0.5] = 1
                    # w[w < 0.5] = 0
                    w = np.squeeze(w, axis=0)
                    w = np.squeeze(w, axis=0)

                    plt.figure(figsize=(size,size),dpi=100)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                    plt.margins(0,0)
       
                    plt.imshow(preds_com, cmap='gray')  
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig(self.args.res_dir +'/'+ str(iter) +'.png')
                    plt.close()

                    plt.figure(figsize=(size,size),dpi=100)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                    plt.margins(0,0)

                    plt.imshow(w)  
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig(self.args.res_dir +'/'+ str(iter) +'w.png')
                    plt.close()


                # if iter % self.args.save_interval == 0:
                #     save_path = self.args.res_dir
                #     self.save_img(images, targets, output_, iter, save_path)

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

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision, sensitivity


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
    
def indicators(output, target, epoch):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    # target_ = target

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    # hd_ = hd(output_, target_)
    hd_ = safe_hd(output_, target_)
    hd95_ = safe_hd95(output_, target_)
    # hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)
    sensitivity_ = sensitivity(output_, target_)
    
    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, sensitivity_


def indicators_1(output, target):
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


def safe_hd(result, reference, voxelspacing=None, connectivity=1):
    """安全计算Hausdorff距离，当输入无效时返回nan"""
    # 检查result和reference是否包含至少一个非零像素
    has_result = np.any(result != 0)
    has_reference = np.any(reference != 0)
    
    # 如果任何一个数组为空，返回nan
    if not has_result or not has_reference:
        return np.nan
    
    try:
        return hd(result, reference)
    except RuntimeError:
        # 捕获其他可能的运行时错误
        return np.nan


def safe_hd95(result, reference, voxelspacing=None, connectivity=1):
    """安全计算Hausdorff距离，当输入无效时返回nan"""
    # 检查result和reference是否包含至少一个非零像素
    has_result = np.any(result != 0)
    has_reference = np.any(reference != 0)
    
    # 如果任何一个数组为空，返回nan
    if not has_result or not has_reference:
        return np.nan
    
    try:
        return hd95(result, reference)
    except RuntimeError:
        # 捕获其他可能的运行时错误
        return np.nan

















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




