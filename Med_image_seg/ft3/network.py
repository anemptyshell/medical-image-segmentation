import torch
import torch.nn as nn
import torch.nn.functional as F
# from Med_image_seg.fang1.model_util.resnet import resnet34
# from model_util.resnet import resnet34
from torch.nn import init
import logging
BatchNorm2d = nn.BatchNorm2d
relu_inplace = True

BN_MOMENTUM = 0.1


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        """(convolution => [BN] => ReLU) * 2"""
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        cat_x = torch.cat((x1, x2), 1)
        output = self.conv(cat_x)
        return output


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


"""Decouple Layer"""
class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        return f_fg, f_bg 


"""Auxiliary Head"""
class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()

        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        return mask_fg, mask_bg 


class EnhancedFusionWithSqueeze(nn.Module):
    """使用squeeze处理单通道特征的融合模块"""
    def __init__(self, input_channels=1, hidden_channels=32):
        super().__init__()
        
        # 融合卷积
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)
        )
        # 相似度计算网络（处理2D特征图）
        self.similarity_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # 输入两个2D特征图
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _compute_spatial_similarity(self, feat1, feat2, method='conv'):
        """
        计算两个2D特征图的空间相似度
        
        Args:
            feat1: [bs, h, w] (squeezed)
            feat2: [bs, h, w] (squeezed)
            method: 计算方法
            
        Returns:
            sim_map: [bs, 1, h, w] (重新添加通道维度)
        """
        bs, h, w = feat1.shape
        
        if method == 'conv':
            # 使用卷积网络学习相似度关系
            # 需要添加通道维度 [bs, h, w] -> [bs, 1, h, w]
            feat1_unsqueeze = feat1.unsqueeze(1)  # [bs, 1, h, w]
            feat2_unsqueeze = feat2.unsqueeze(1)  # [bs, 1, h, w]
            
            combined = torch.cat([feat1_unsqueeze, feat2_unsqueeze], dim=1)  # [bs, 2, h, w]
            sim_map = self.similarity_net(combined)  # [bs, 1, h, w]
            
        elif method == 'local_correlation':
            """
                计算两个特征图的局部相关性（皮尔逊相关系数）

                正确的局部相关计算步骤：
                1. 对每个位置，考虑其邻域窗口
                2. 计算窗口内的均值、标准差、协方差
                3. 计算相关系数：cov(X,Y) / (std(X) * std(Y))
            """
            bs, h, w = feat1.shape
    
            # 首先，需要将2D特征图添加通道维度
            feat1_unsq = feat1.unsqueeze(1)  # [bs, 1, h, w]
            feat2_unsq = feat2.unsqueeze(1)  # [bs, 1, h, w]

            # 用这个核做卷积 = 计算3×3窗口内的平均值
            kernel = torch.ones(1, 1, 3, 3, device=feat1.device) / 9.0

            # 计算局部均值
            local_mean1 = F.conv2d(feat1_unsq, kernel, padding=1, groups=1)
            local_mean2 = F.conv2d(feat2_unsq, kernel, padding=1, groups=1)

            # 计算局部二阶矩
            local_mean1_sq = F.conv2d(feat1_unsq**2, kernel, padding=1, groups=1)
            local_mean2_sq = F.conv2d(feat2_unsq**2, kernel, padding=1, groups=1)
            local_mean12 = F.conv2d(feat1_unsq * feat2_unsq, kernel, padding=1, groups=1)

            # 计算局部方差和协方差
            local_var1 = local_mean1_sq - local_mean1**2
            local_var2 = local_mean2_sq - local_mean2**2
            local_cov = local_mean12 - local_mean1 * local_mean2

            eps = 1e-8
            # 计算局部相关系数
            local_corr = local_cov / (torch.sqrt(local_var1 + eps) * torch.sqrt(local_var2 + eps))
            # 相关系数范围是[-1, 1]，映射到[0, 1]表示相似度
            sim_map = (local_corr + 1.0) / 2.0  # [bs, 1, h, w]
            
        elif method == 'inverse_diff':
            # 基于差异的相似度
            diff = torch.abs(feat1 - feat2)  # [bs, h, w]
            sim_map = 1.0 / (1.0 + diff)     # [bs, h, w]
            sim_map = sim_map.unsqueeze(1)   # [bs, 1, h, w]
            
        return sim_map
    
    def forward(self, preds, preds_strong, preds_alter):
        """
            preds: [bs, 1, h, w]
            preds_strong: [bs, 1, h, w]
            preds_alter: [bs, 1, h, w]
        """
        bs, c, h, w = preds.shape
        
        # 1. 融合
        fused_strong = self.fuse_conv1(torch.cat([preds, preds_strong], dim=1))
        fused_alter = self.fuse_conv2(torch.cat([preds, preds_alter], dim=1))
        
        # 2. 使用squeeze去掉通道维度，计算空间相似度
        preds_squeezed = preds.squeeze(1)                # [bs, h, w]
        fused_strong_squeezed = fused_strong.squeeze(1)  # [bs, h, w]
        fused_alter_squeezed = fused_alter.squeeze(1)    # [bs, h, w]
        
        # 计算相似度图
        sim_map1 = self._compute_spatial_similarity(
            fused_strong_squeezed, preds_squeezed, method='local_correlation'
        )  # [bs, 1, h, w]
        
        sim_map2 = self._compute_spatial_similarity(
            fused_alter_squeezed, preds_squeezed, method='local_correlation'
        )  # [bs, 1, h, w]
        
        # 3. 选择策略得到权重
        # 方法A：取最大值
        w = torch.max(sim_map1, sim_map2)  # [bs, 1, h, w]
        # 方法B：加权平均
        # w = 0.6 * sim_map1 + 0.4 * sim_map2
        out = w * preds
        
        return w, out


class Mine(nn.Module):
    """
    definition of MINE
    """

    def __init__(self, input_size, hidden_size=200):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        return output


class FeatureEncoder(nn.Module):
    """
    共享的特征提取Encoder
    将[bs, 1, 512, 512]的pred/label映射到高级语义特征
    """
    def __init__(self, in_channels=1, base_channels=32, out_dim=128):
        super().__init__()
        # 特征提取网络
        self.encoder = nn.Sequential(
            # 下采样块1: 512 -> 256
            CBR(in_channels, base_channels, kernel_size=3, padding=1),
            CBR(base_channels, base_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            
            # 下采样块2: 256 -> 128
            CBR(base_channels, base_channels*2, kernel_size=3, padding=1),
            CBR(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            
            # 下采样块3: 128 -> 64
            CBR(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            CBR(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            
            # 全局特征提取
            CBR(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            CBR(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            
            # 全局平均池化 + 全连接
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, out_dim)
        )
        self.out_dim = out_dim
        
    def forward(self, x):
        """
        x: [bs, 1, 512, 512]
        return: [bs, out_dim]
        """
        return self.encoder(x)


class PredLabelMutualInfoModule(nn.Module):
    """
    预测图与标签图的互信息最大化模块
    """
    def __init__(self, in_channels=1, feature_dim=128, 
                 hidden_size=200, use_shared_encoder=True):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # 特征提取器
        if use_shared_encoder:
            # 共享编码器（更合理，确保特征空间一致）
            self.pred_encoder = FeatureEncoder(in_channels, out_dim=feature_dim)
            self.label_encoder = self.pred_encoder
        else:
            # 独立编码器
            self.pred_encoder = FeatureEncoder(in_channels, out_dim=feature_dim)
            self.label_encoder = FeatureEncoder(in_channels, out_dim=feature_dim)
        
        # MINE网络 - 输入是拼接后的特征维度
        self.mine_net = Mine(input_size=feature_dim * 2, hidden_size=hidden_size)
        
    def mutual_information(self, joint, marginal):
        """
        计算互信息下界
        joint: 联合样本 [bs, 2*feature_dim]
        marginal: 边缘样本 [bs, 2*feature_dim]
        """
        # 确保浮点类型
        joint = joint.float()
        marginal = marginal.float()
        
        # MINE估计
        t_joint = self.mine_net(joint)            # E_p(x,y)[T(x,y)]
        t_marginal = self.mine_net(marginal)      # E_p(x)p(y)[e^{T(x,y)}]
        
        # 互信息下界估计
        mi_lb = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)) + 1e-8)
        
        return mi_lb
    
    def forward(self, pred, label):
        """
        pred: 预测图 [bs, 1, 512, 512]，应该已经过sigmoid/softmax
        label: 标签图 [bs, 1, 512, 512]，应该是0/1二值图
        return: (pred_feat, loss_mi, label_feat)
        """
        # 1. 提取高级特征
        pred_feat = self.pred_encoder(pred)      # [bs, feature_dim]
        label_feat = self.label_encoder(label)   # [bs, feature_dim]
        print(label_feat.size())  
        print('+++++++')
        
        # 2. 构建联合样本（保持配对关系）
        joint = torch.cat([pred_feat, label_feat], dim=1)  # [bs, 2*feature_dim]
        
        # 3. 构建边缘样本（破坏配对关系）
        # 方法1：批次滚动（简单有效）
        pred_feat_shuffled = torch.roll(pred_feat, shifts=1, dims=0)
        
        # 方法2：完全随机打乱（更严格）
        # idx = torch.randperm(pred_feat.size(0))
        # pred_feat_shuffled = pred_feat[idx]
        
        marginal = torch.cat([pred_feat_shuffled, label_feat], dim=1)
        
        mi = self.mutual_information(joint, marginal)
        
        # 5. 损失计算：最大化MI = 最小化 -MI
        loss_mi = -mi
        
        # 可选：对loss进行缩放，避免太大或太小
        # loss_mi = torch.sigmoid(loss_mi) * 10  # 缩放到0-10之间
        return pred_feat, loss_mi, label_feat



class selective_feature_decoupler(nn.Module):
    """
    definition of SFD
    """

    def __init__(self, in_c, out_c, in_h, in_w):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w

        # 3*3CBR 2
        self.c1 = nn.Sequential(
            CBR(in_c, in_c, kernel_size=3, padding=1),
            CBR(in_c, out_c, kernel_size=3, padding=1)
        )
        # 3*3CBR 2
        self.c2 = nn.Sequential(
            CBR(in_c, in_c, kernel_size=3, padding=1),
            CBR(in_c, out_c, kernel_size=3, padding=1)
        )

        # before MINE, reduce channel and size
        self.reduce_c = nn.Sequential(
            CBR(out_c, 16, kernel_size=1, padding=0),
            nn.MaxPool2d(2, stride=2)
        )

        # init MINE
        self.mine_net = Mine(input_size=16 * (in_h // 2) * (in_w // 2) * 2)
        # self.mine_net = Mine(input_size=524288)

    def mutual_information(self, joint, marginal):
        # joint = joint.float().cuda() if torch.cuda.is_available() else joint.float()
        # marginal = marginal.float().cuda() if torch.cuda.is_available() else marginal.float()
        joint = joint.float() 
        marginal = marginal.float() 
        # print('******************')
        # print(joint.size())     

        t = self.mine_net(joint)
        et = torch.exp(self.mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(1 + torch.mean(et))

        return mi_lb

    def forward(self, x):
        # x:[B,C,H,W]
        s = self.c1(x)  # significant feature
        u = self.c2(x)  # unimportant feature
        # print(s.size())  ## 2, 64, 256, 256

        # reduce channel
        s_16 = self.reduce_c(s)
        u_16 = self.reduce_c(u)
        # print(s_16.size())  ## 2, 16, 128, 128

        # flatten s and u
        s_flat = torch.flatten(s_16, start_dim=1)
        u_flat = torch.flatten(u_16, start_dim=1)
        # print(s_flat.size())  ## 2, 262144

        # create joint and marginal
        joint = torch.cat([s_flat, u_flat], dim=1)
        marginal = torch.cat([s_flat, torch.roll(u_flat, shifts=1, dims=0)], dim=1)
        # print('---------------------')
        # print(joint.size())    ## [2, 524288]

        # calculate mi loss
        mi_lb = self.mutual_information(joint, marginal)
        loss_mi = mi_lb

        # sigmoid
        loss_mi = torch.sigmoid(loss_mi)

        return s, loss_mi, u


"""concat+卷积 + cos相似度 +  /"""
class Multi_decoder_Net(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(Multi_decoder_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)   ## CBR * 2
        self.down1 = Down(64, 128)              ## maxpool + CBR*2
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1_1 = Up(1024, 512, bilinear)
        self.up1_2 = Up(1024, 512, bilinear)
        self.up1_3 = Up(1024, 512, bilinear)

        self.up2_1 = Up(512, 256, bilinear)
        self.up2_2 = Up(512, 256, bilinear)
        self.up2_3 = Up(512, 256, bilinear)

        self.up3_1 = Up(256, 128, bilinear)
        self.up3_2 = Up(256, 128, bilinear)
        self.up3_3 = Up(256, 128, bilinear)

        self.up4_1 = Up(128, 64, bilinear)
        self.up4_2 = Up(128, 64, bilinear)
        self.up4_3 = Up(128, 64, bilinear)

        self.out_1 = OutConv(64, n_classes)
        self.out_2 = OutConv(64, n_classes)
        self.out_3 = OutConv(64, n_classes)

        self.decouple_layer = DecoupleLayer(1024, 128)
        self.aux_head = AuxiliaryHead(128)

        self.fusion = EnhancedFusionWithSqueeze()
        self.mi_module = PredLabelMutualInfoModule(
                                in_channels=1,     
                                feature_dim=128,
                                use_shared_encoder=True)

    def forward(self, x, label=None):
        ## x: [bs, 3, 256, 256]
        # encoder
        x1 = self.inc(x)                                  ## [bs, 64, 256, 256]
        x2 = self.down1(x1)                               ## [bs, 128, 128, 128]
        x3 = self.down2(x2)                               ## [bs, 256, 64, 64]
        x4 = self.down3(x3)                               ## [bs, 512, 32, 32]
        x5 = self.down4(x4)                               ## [bs, 1024, 16, 16]
    
        # decoder
        o_4_1 = self.up1_1(x5, x4)                        ## [bs, 512, 32, 32]   
        o_3_1 = self.up2_1(o_4_1, x3)                     ## [bs, 256, 64, 64]
        o_2_1 = self.up3_1(o_3_1, x2)                     ## [bs, 128, 128, 128]
        o_1_1 = self.up4_1(o_2_1, x1)                     ## [bs, 64, 256, 256]
        o_seg1 = self.out_1(o_1_1)                        ## [bs, 1, 256, 256]

        ske_strong, ske_alter = self.decouple_layer(x5)                    ## [bs, 128, 16, 16]
        mask_strong, mask_alter = self.aux_head(ske_strong, ske_alter)     ## [bs, 1, 256, 256]
        w, out = self.fusion(o_seg1, mask_strong, mask_alter)
        # pred_sigmoid = torch.sigmoid(out)
        # _, loss_mi, _ = self.mi_module(pred_sigmoid, label)

        return o_seg1, mask_strong, mask_alter, w, out




# unet = Multi_decoder_Net(3)
# a = torch.rand(1, 3, 512, 512)
# b = torch.rand(1, 1, 512, 512)
# o_seg1, mask_strong, mask_alter, w, out = unet.forward(a, b)
# print(o_seg1.size())            # torch.Size([1, 1, 256, 256])
# print(f_fg.size())      
# print(f_bg.size())      
# print(x1.size())                # torch.Size([1, 1, 256, 256])
# print(x2.size())                # torch.Size([1, 1, 128, 128])
# print(x3.size())                # torch.Size([1, 1, 64, 64])
# print(complement_weighted.size())    # torch.Size([1, 1, 256, 256])
# print(loss.size())              # []