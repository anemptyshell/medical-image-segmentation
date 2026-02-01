import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler

from libs.metric import metric
from libs.utils import AverageMeter
from libs.base_model import base_model
from collections import OrderedDict

from Med_image_seg.unet.loss import BceDiceLoss
from Med_image_seg.unet.network import U_Net
from Med_image_seg.fang.utils.cldice import clDice
# from Med_image_seg.unet.cam import U_Net_Attention_Visualizer

# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def arguments():
    args = {

    # '--betas': (0.9, 0.999), # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square 用于计算梯度及其平方的运行平均值的系数
    # '--T_max': 50, # – Maximum number of iterations. Cosine function period.
}  
    return args


class unet(base_model):
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

        self.network = U_Net().to('cuda')
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
           
            # if iter % self.args.print_interval == 0:
            #     self.save_print_loss_lr(iter, epoch, loss_list, now_lr)

        pbar.close()
        self.lr_scheduler.step()
        # return step
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

                preds = self.network(images) 
                
                loss = self.BceDiceLoss(preds, targets)
                # iou, dice = iou_score(preds, targets)
                iou, dice, hd, hd95, recall, specificity, precision, sensitivity = indicators(preds, targets, epoch)

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
                output = F.sigmoid(preds)
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
        # visualizer = U_Net_Attention_Visualizer(self.network)
        target_layers = [self.network.Conv_1x1]
        cam = GradCAM(model=self.network, target_layers=target_layers)

        # with torch.no_grad():
        for iter, data in enumerate(tqdm(test_loader)):
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
            with torch.no_grad():
                preds = self.network(images)

                # save_dir = "./attention_visualization"
                # os.makedirs(save_dir, exist_ok=True)
                
                # # 提供完整的保存路径
                # save_path = f"{save_dir}/attention_batch{iter}.png"
                # print("使用均值方法可视化注意力热图...")
                # attention_map = visualizer.visualize(images, method='mean', 
                #                                      save_path=save_path)

                # # 方法2：使用多种方法比较
                # print("\n使用多种方法比较注意力热图...")
                # visualizer.visualize_multiple_methods(images, 
                #                                      methods=['mean', 'max', 'std', 'l2_norm'],
                #                                      save_path=save_path)


                iou, dice, hd, hd95, recall, specificity, precision, sensitivity = indicators_1(preds, targets)
                iou_avg_meter.update(iou, images.size(0))
                dice_avg_meter.update(dice, images.size(0))
                hd_avg_meter.update(hd, images.size(0))
                hd95_avg_meter.update(hd95, images.size(0))
                recall_avg_meter.update(recall, images.size(0))
                specificity_avg_meter.update(specificity, images.size(0))
                precision_avg_meter.update(precision, images.size(0))
                sensitivity_avg_meter.update(sensitivity, images.size(0))

                ################################################ 
                output = F.sigmoid(preds)
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
                    preds = torch.sigmoid(preds).cpu().numpy()
                    preds[preds >= 0.5] = 1
                    preds[preds < 0.5] = 0
                    preds = np.squeeze(preds, axis=0)
                    preds = np.squeeze(preds, axis=0)

                    plt.figure(figsize=(size,size),dpi=100)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                    plt.margins(0,0)
       
                    plt.imshow(preds, cmap='gray')  
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig(self.args.res_dir +'/'+ str(iter) +'.png')
                    plt.close()
            
            if iter % 1 == 0: 
                # 开启梯度计算
                with torch.set_grad_enabled(True):
                    # 选择输入：取 Batch 中的第一张图并增加 batch 维度 [1, 3, H, W]
                    input_tensor = images[0:1] 

                    # 指定目标：针对输出通道 0 (如果是多类分割，可以更换 index)
                    # ClassifierOutputTarget 对于分割模型，默认会聚合空间像素的梯度
                    cam_targets = [ClassifierOutputTarget(0)]

                    # 计算 CAM (grayscale_cam 的维度是 [1, H, W])
                    grayscale_cam = cam(input_tensor=input_tensor, targets=cam_targets)
                    grayscale_cam = grayscale_cam[0, :]

                    # 转换原图用于叠加 (从 Tensor 转为 Numpy RGB)
                    img_to_show = input_tensor[0].permute(1, 2, 0).cpu().numpy()
                    # 归一化到 [0, 1] 方便显示
                    img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min() + 1e-8)

                    # 叠加生成热图
                    visualization = show_cam_on_image(img_to_show, grayscale_cam, use_rgb=True)

                    # 保存或处理热图
                    cv2.imwrite(f"{self.args.res_dir}/cam_{iter}.png", visualization[:, :, ::-1]) # RGB 转 BGR 保存

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




