import warnings
# import visdom
from libs.Visualizer import Visualizer
import torch.backends.cudnn as cudnn
from datetime import datetime
import torch
import os
import torch.nn as nn
import torch.optim
import torch.utils.data
# from ptflops import get_model_complexity_info
import math
import os
import logging
import logging.handlers
from matplotlib import pyplot as plt
import numpy as np
import random
from torch.optim import lr_scheduler as lr_sch




def arguments():
    args = {
        # "--epochs": 200,
        # "--batch_size": 32,
        # "--lr": 0.0001,
        # "--test_batch": 10,
        # "--test_weight_choose": "final",
        # "--weight_path": "None",
        # "--test_interval": 5,
    }
    return args


class base_model(nn.Module):
    def __init__(self, parser):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True if not self.device == "cpu" else False
        parser.add_args(arguments())
        self.args = parser.get_args()
        self.Tensor = torch.cuda.FloatTensor if self.args.gpu else torch.Tensor


        self.metric = self.args.metric
        self.indicator_for_best = None
        
        self.vis = Visualizer(parser)
        # if self.args.control_monitor:  ###开关控制是否监控
        #     self.viz = visdom.Visdom(port=self.args.visdom_port)  ####visdom监控
        #     if not self.viz.check_connection():
        #         warnings.warn("visdom服务器尚未启动,请打开visdom")  # 测试一下链接，链接错误的话会警告
        # if self.args.control_save_img_type is not None and "metricepoch" in self.args.control_save_img_type:
        #     self.metricdic = dict()

    def set_cuda(self):
        # Cuda Set-up
        if self.args.gpu is not None:
            ## (default) args.gpu = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu


    def make_dir(self, parser):

        self.work_dir = os.path.join(self.args.output, "Med_image_seg", self.args.model_name, self.args.dataset)
        # self.work_dir = os.path.join(self.args.output, self.args.model_name, self.args.dataset)
        self.log_dir = os.path.join(self.work_dir, "logs")
        self.res_dir = os.path.join(self.work_dir, "results")
        self.checkpoint_dir = os.path.join(self.work_dir, 'checkpoints')
        
        # self.resume_model = os.path.join(self.checkpoint_dir, 'latest.pth')

        parser.add_args({"--work_dir": self.work_dir, "--log_dir": self.log_dir, "--res_dir": self.res_dir, "--checkpoint_dir": self.checkpoint_dir})

        self.makedirs(self.log_dir)
        self.makedirs(self.res_dir)
        self.makedirs(self.checkpoint_dir)
        ## log_dir = "../output/Med_image_seg/unet/isic2017/logs"
        ## work_dir = "../output/Med_image_seg/unet/isic2017"
        ## res_dir = "../output/Med_image_seg/unet/isic2017/results"
        ## checkpoint_dir = "../output/Med_image_seg/unet/isic2017/checkpoints"


    def load_model(self, parser, network, optimizer, lr_scheduler):

        self.args = parser.get_args()
        ## resume_model .pth文件需要先放置好
        resume_model = os.path.join(self.args.checkpoint_dir, 'latest.pth')

        if os.path.exists(resume_model):

            checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

            network.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = 1
            saved_epoch = checkpoint['epoch']
            start_epoch += saved_epoch
            min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

            # log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
            # logger.info(log_info)

            print("=> resuming model from '{}'. resume_epoch: '{}', min_loss: '{}', min_epoch: '{}', loss: '{}'".format(
                resume_model, saved_epoch, min_loss, min_epoch, loss
                )
            )

            return checkpoint["epoch"], network, optimizer, lr_scheduler
        

    def save_print_metric(self, current_epoch, metric_cluster, best):
        self.final_epoch_trigger = True if current_epoch == self.args.epochs else False

        self.vis.loggin_metric(metric_cluster, current_epoch, best, self.indicator_for_best)

        if self.final_epoch_trigger:
            self.vis.plot_menu(best)


    def save_print_loss_lr(self, iter, current_epoch, loss_list, now_lr):

        self.vis.loggin_loss_lr(iter, current_epoch, loss_list, now_lr)


    def save_model(self, epoch):

        if self.best_trigger:
            print('save best!')
            torch.save(self.network.state_dict(), os.path.join(self.args.checkpoint_dir, 'best.pth'))

        self.final_epoch_trigger = True if epoch == self.args.epochs else False

        if self.final_epoch_trigger:
            torch.save(self.network.state_dict(), os.path.join(self.args.checkpoint_dir, 'latest.pth'))


    def set_optimizer(self):
        if self.args.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        elif self.args.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                              nesterov=self.args.nesterov, weight_decay=self.args.weight_decay)
        elif self.args.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.args.lr, weight_decay=1e-4)
        else:
            raise NotImplementedError
        
        return optimizer


    def set_lr_scheduler(self):
        if self.args.scheduler == 'CosineAnnealingLR':
            lr_scheduler = lr_sch.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_lr)

        elif self.args.scheduler == 'ReduceLROnPlateau':
            lr_scheduler = lr_sch.ReduceLROnPlateau(self.optimizer, factor=self.args.factor, 
                                                                patience=self.args.patience, verbose=1, min_lr=self.args.min_lr)
        elif self.args.scheduler == 'MultiStepLR':
            lr_scheduler = lr_sch.MultiStepLR(self.optimizer, milestones=[int(e) for e in self.args.milestones.split(',')], 
                                                            gamma=self.args.gamma)
        elif self.args.scheduler == 'ConstantLR':
            lr_scheduler = None
        else:
            raise NotImplementedError
        
        return lr_scheduler

        

    def set_seed(self, seed):
        # # for hash
        # os.environ['PYTHONHASHSEED'] = str(seed)
        # # for python and numpy
        # random.seed(seed)
        # np.random.seed(seed)
        # # for cpu gpu
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # # for cudnn
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)     
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True



    def save_imgs(self, img, gt, pred, iter, save_path, threshold=0.5):
        self.vis.save_imgs(img, gt, pred, iter, save_path, threshold=0.5)

    def save_img(self, img, gt, pred, iter, save_path):
        self.vis.save_img(img, gt, pred, iter,save_path)


    def makedirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


    def save_args(self):  ###########保存opt参数
        args_dir = self.args.log_dir
        if not os.path.exists(args_dir):
            os.makedirs(args_dir)
        with open(os.path.join(args_dir, "options.txt"), "w", newline="\n") as file:
            file.seek(0)
            file.truncate()
            for arg, content in self.args.__dict__.items():
                file.write("{}:{},\n".format(arg, content))


    # def compute_macs_params(self, net, net_name, size):
    #     macs, params = get_model_complexity_info(net, (size[0], size[1], size[2]))
    #     self.vis.plot_macs_params(macs, params, net_name)
    #     return macs, params
    
    def per_class_dice(self, y_pred, y_true):
        smooth = 0.0001
        y_pred = y_pred
        y_true = y_true

        FN = torch.sum((1-y_pred)*y_true,dim=(2,3)) 
        FP = torch.sum((1-y_true)*y_pred,dim=(2,3)) 
        TN = torch.sum((1 - y_pred) * (1 - y_true), dim=(2, 3))
        TP = torch.sum(y_pred * y_true, dim=(2, 3))

        Pred = y_pred
        GT = y_true
        inter = torch.sum(GT* Pred,dim=(2,3))    ## 等于TP？？？


        union = torch.sum(GT,dim=(2,3)) + torch.sum(Pred,dim=(2,3)) 

        dice = (2*inter+smooth)/(union+smooth)
        Jac = (inter+smooth)/(inter+FP+FN+smooth)

        ## 下面是自己加的指标
        # DSC = (2*TP + smooth)/(2*TP + FP + FN + smooth)
        # iou = (TP+smooth)/(TP+FP+FN+smooth)

        return dice, Jac
    
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
    
    def create_exp_directory(self):
        csv = 'val_results'+'.csv'
        with open(os.path.join(self.args.log_dir, csv), 'w') as f:
            f.write('epoch, dice, Jac, clDice \n')
        csv1 = 'test_results'+'.csv'
        with open(os.path.join(self.args.log_dir, csv1), 'w') as f:
            f.write('dice, Jac, clDice, acc, sen, spe, pre, recall, f1 \n')




    # def load_model(self, networks, opts):
    #     if self.args.weight_path == "None":
    #         weight_dir = (os.path.join(self.args.log_dir, "model_" + self.args.test_weight_choose + ".pth")if self.args.test_weight_choose == "best" or self.args.test_weight_choose == "final" else os.path.join(self.args.log_dir, "model_current.pth"))
    #     else:
    #         weight_dir = self.args.weight_path
    #     checkpoint = torch.load(weight_dir, map_location="cpu")
    #     ## 加载网络的状态字典
    #     for name, net in networks.items():
    #         if net != None:
    #             net.load_state_dict(checkpoint[name + "_state_dict"])
    #         networks[name] = net
    #     ## 加载优化器的状态字典
    #     for name, opt in opts.items():
    #         if opt != None:
    #             opt.load_state_dict(checkpoint[name.lower() + "_optimizer"]) 
    #         opts[name] = opt
    #     print(
    #         "=> loaded checkpoint '{}' (epoch {})".format(
    #             weight_dir, checkpoint["epoch"]
    #         )
    #     )
    #     return checkpoint["epoch"], networks, opts



    # def save_result_img(self, current_epoch, networks, opts, metric_cluster,embedding, real,lossdict, model_layer_list=None):  #多两个参数
    #     self.final_epoch_trigger = True if current_epoch == self.epochs else False
    #     if self.args.control_save_img_type is not None:
    #         #########记录每个epoch的非曲线metric值###########
    #         def get_metric_log():
    #             metric_name = self.args.metric
    #             for i in range(len(metric_name)):
    #                 if current_epoch == 1:
    #                     self.metricdic[metric_name[i]] = []
    #                 self.metricdic[metric_name[i]].append(metric_cluster.get_metric([metric_name[i]])[0])
    #         if metric_cluster is not None:
    #             if current_epoch != "test" and "metricepoch" in self.args.control_save_img_type:
    #                 if (current_epoch-1) % self.args.test_interval == 0:
    #                     get_metric_log()
            
    #             ############################################
    #             if "t-SNE" in self.args.control_save_img_type:  #######最优，多一个指标
    #                 self.vis.save_embedding(embedding, real, self.res_dir, self.args.pic_name)
    #             if self.final_epoch_trigger:  ####最后
    #                 if "lossepoch" in self.args.control_save_img_type:
    #                     self.vis.save_lossepochimg(lossdict, self.res_dir)
    #                 if "metricepoch" in self.args.control_save_img_type:
    #                     self.vis.save_metricepochimg(self.metricdic, self.res_dir)
    #             if self.final_epoch_trigger or current_epoch == "test":  #####最后或test
    #                 if self.final_epoch_trigger:
    #                     _, self.networks, self.opts = self.load_model(networks, opts)
    #                 if "attentionmap" in self.args.control_save_img_type:
    #                     self.vis.save_attentionmap(model_layer_list, self.res_dir, image_size=self.args.img_size)
    #                 if "featuremap" in self.args.control_save_img_type:
    #                     self.vis.save_featuremap(model_layer_list, self.res_dir, image_size=self.args.img_size)
    #                 if "filter" in self.args.control_save_img_type:
    #                     self.vis.save_filter(model_layer_list, self.res_dir)
    #     if self.args.control_monitor and current_epoch != "test":  ######实时,不要更改顺序
    #         self.vis.monitor(self.viz, current_epoch, lossdict, metric_cluster)



    ## 在test2.py中测试
    def get_logger(self, name, log_dir):
        '''
        Args:
        name(str): name of logger
        log_dir(str): path of log
        '''
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        info_name = os.path.join(log_dir, '{}.info.log'.format(name))
        info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
        info_handler.setLevel(logging.INFO)

        ## datefmt 指定日期时间格式
        formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        return logger

