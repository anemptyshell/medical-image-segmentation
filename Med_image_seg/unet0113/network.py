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
            fused_strong_squeezed, preds_squeezed, method='conv'
        )  # [bs, 1, h, w]
        
        sim_map2 = self._compute_spatial_similarity(
            fused_alter_squeezed, preds_squeezed, method='conv'
        )  # [bs, 1, h, w]
        
        # 3. 选择策略得到权重
        # 方法A：取最大值
        w = torch.max(sim_map1, sim_map2)  # [bs, 1, h, w]
        # 方法B：加权平均
        # w = 0.6 * sim_map1 + 0.4 * sim_map2
        
        # 4. 计算补集并加权
        # complement = 1.0 - torch.sigmoid(preds)  # [bs, 1, h, w]   preds是不是应该归一化 结果图稍好，但指标不好 （结果在sigmoid）
        complement = 1.0 - preds
        complement_weighted = complement * w  # [bs, 1, h, w]
        
        return w, complement_weighted


class EnhancedFusionWithEntropyWeighting(nn.Module):
    """使用信息熵评估置信度并分配权重的融合模块"""
    def __init__(self, input_channels=1, hidden_channels=32):
        super().__init__()
        
        # 交叉注意力模块（保持原有结构）
        # self.cross_attn1 = CrossAttentionModule(
        #     in_channels=input_channels,
        #     hidden_channels=hidden_channels,
        #     num_heads=4
        # )
        # self.cross_attn2 = CrossAttentionModule(
        #     in_channels=input_channels,
        #     hidden_channels=hidden_channels,
        #     num_heads=4
        # )
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
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1))
        
        self.out_proj1 = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
        self.out_proj2 = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
        
        # 相似度计算网络（可选，可以保留或移除）
        self.similarity_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 熵权重调整网络（可选）
        self.entropy_refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _compute_entropy(self, prob1, prob2):
        #  """
        # 计算两个预测之间的分歧熵
        
        # 关键思想：
        # 1. 如果两个预测很接近（差异小）-> 低分歧熵 -> 高权重
        # 2. 如果两个预测差异大 -> 高分歧熵 -> 低权重
        
        # Args:
        #     prob1: [bs, 1, h, w] 预测1的概率图
        #     prob2: [bs, 1, h, w] 预测2的概率图
            
        # Returns:
        #     disagreement_entropy: [bs, 1, h, w] 分歧熵
        # """
        # 计算绝对差异
        diff = torch.abs(prob1 - prob2)  # [bs, 1, h, w]
        
        # 方法1：将差异视为二元分布的不确定性
        # 差异越小，不确定性越低
        epsilon = 1e-8
        
        # 将差异d转化为概率p：p = (1-d)，q = d
        # 当d=0（完全一致）时，p=1，q=0，熵为0
        # 当d=0.5（最大分歧）时，p=q=0.5，熵最大
        p = 1.0 - diff  # 一致性概率
        q = diff        # 分歧概率
        
        # 计算二元熵
        entropy = -p * torch.log(p + epsilon) - q * torch.log(q + epsilon)
        
        # 归一化
        entropy_normalized = entropy / torch.log(torch.tensor(2.0, device=prob1.device))
        
        return entropy_normalized
    
    def _compute_confidence_from_entropy(self, entropy_map, method='inverse'):
        """
        从熵值计算置信度（熵值越低，置信度越高）
        Args:
            entropy_map: [bs, 1, h, w] 熵值图
            method: 转换方法  
        Returns:
            confidence_map: [bs, 1, h, w] 置信度图
        """
        if method == 'inverse':
            # 简单逆变换：置信度 = 1 - 熵
            confidence = 1.0 - entropy_map
        
        elif method == 'exponential':
            # 指数衰减：exp(-λ*entropy)
            lambda_val = 5.0  # 衰减系数
            confidence = torch.exp(-lambda_val * entropy_map)
        
        elif method == 'sigmoid':
            # Sigmoid变换：将熵值映射到置信度
            # 当熵值高时快速降低置信度
            confidence = 1.0 / (1.0 + torch.exp(10.0 * (entropy_map - 0.5)))
        
        elif method == 'linear_threshold':
            # 带阈值的线性变换
            threshold = 0.3
            confidence = torch.where(
                entropy_map < threshold,
                1.0 - entropy_map / threshold,  # 低于阈值，置信度线性下降
                torch.zeros_like(entropy_map)   # 高于阈值，置信度为0
            )
        
        return confidence
    
    def _compute_spatial_similarity(self, feat1, feat2, method='conv'):
        """计算两个2D特征图的空间相似度"""
        bs, h, w = feat1.shape
        
        if method == 'conv':
            feat1_unsqueeze = feat1.unsqueeze(1)  # [bs, 1, h, w]
            feat2_unsqueeze = feat2.unsqueeze(1)  # [bs, 1, h, w]
            
            combined = torch.cat([feat1_unsqueeze, feat2_unsqueeze], dim=1)  # [bs, 2, h, w]
            sim_map = self.similarity_net(combined)  # [bs, 1, h, w]
            
        elif method == 'entropy_weighted':
            # 基于熵的相似度：相似度 = 低熵区域的交集
            feat1_prob = torch.sigmoid(feat1.unsqueeze(1))  # [bs, 1, h, w]
            feat2_prob = torch.sigmoid(feat2.unsqueeze(1))  # [bs, 1, h, w]
            
            # 计算两个特征的熵
            entropy1 = self._compute_entropy(feat1_prob)
            entropy2 = self._compute_entropy(feat2_prob)
            
            # 计算低熵区域（高置信度区域）
            confidence1 = self._compute_confidence_from_entropy(entropy1)
            confidence2 = self._compute_confidence_from_entropy(entropy2)
            
            # 相似度 = 两个高置信度区域的交集
            sim_map = confidence1 * confidence2
            
        return sim_map
    
    def _fuse_with_entropy_weighting(self, fused_strong, fused_alter):
        """
        Args:
            fused_strong: [bs, 1, h, w] 与强预测融合的结果
            fused_alter: [bs, 1, h, w] 与替代预测融合的结果      
        Returns:
            w: [bs, 1, h, w] 最终权重图
        """
        bs, c, h, w = fused_strong.shape
        
        # 1. 计算每个预测的熵和置信度
        with torch.no_grad():
            # 转换为概率
            fused_strong_prob = torch.sigmoid(fused_strong)
            fused_alter_prob = torch.sigmoid(fused_alter)
            
            # 计算熵
            entropy_strong = self._compute_entropy(fused_strong_prob)
            entropy_alter = self._compute_entropy(fused_alter_prob)
            
            # 计算置信度（熵的逆）
            confidence_strong = self._compute_confidence_from_entropy(entropy_strong, method='inverse')
            confidence_alter = self._compute_confidence_from_entropy(entropy_alter, method='inverse')
             
        # 4. 方法3：混合策略（基于置信度的加权平均）
        # 使用softmax进行温度缩放
        temperature = 0.1  # 温度参数，越小越接近hard selection
        confidence_stack = torch.cat([confidence_strong, confidence_alter], dim=1)
        confidence_stack_scaled = confidence_stack / temperature
        softmax_weights = F.softmax(confidence_stack_scaled, dim=1)

        # 加权平均
        w = softmax_weights[:, 0:1, :, :] * confidence_strong + \
                softmax_weights[:, 1:2, :, :] * confidence_alter
        
        # 可选：通过卷积网络进一步细化权重
        # w_refined = self.entropy_refine(w)
        
        return w
    
    def forward(self, preds, preds_strong, preds_alter):
        """
            preds: [bs, 1, h, w] - 主预测
            preds_strong: [bs, 1, h, w] - 强增强预测
            preds_alter: [bs, 1, h, w] - 替代预测
        """
        bs, c, h, w = preds.shape
        
        # 1. 使用交叉注意力融合
        # attn_output1 = self.cross_attn1(preds, preds_strong)
        # fused_strong = self.out_proj1(attn_output1)
        
        # attn_output2 = self.cross_attn2(preds, preds_alter)
        # fused_alter = self.out_proj2(attn_output2)
        fused_strong = self.fuse_conv1(torch.cat([preds, preds_strong], dim=1))
        fused_alter = self.fuse_conv2(torch.cat([preds, preds_alter], dim=1))

        # 2. 转换为概率
        fused_strong_prob = torch.sigmoid(fused_strong)
        fused_alter_prob = torch.sigmoid(fused_alter)

        with torch.no_grad():
            disagreement_entropy = self._compute_entropy(
                fused_strong_prob, fused_alter_prob)  # [bs, 1, h, w]
            # 计算权重：分歧熵越低，权重越高
            # w = 1 - disagreement_entropy  # 简单逆变换
            w = torch.exp(-5.0 * disagreement_entropy)  # 指数衰减

        # 3. 计算补集并加权
        complement = 1.0 - preds
        complement_weighted = complement * w
        
        return w, complement_weighted



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
        # print(joint.size())     ## [2, 524288]

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

        # self.end_conv1 = nn.Conv2d(64, 1, 1).cuda()
        # self.end_conv2 = nn.Conv2d(128, 1, 1).cuda() 
        # self.end_conv3 = nn.Conv2d(256, 1, 1).cuda()
        # self.end_conv1 = nn.Conv2d(64, 1, 1)
        # self.end_conv2 = nn.Conv2d(128, 1, 1) 
        # self.end_conv3 = nn.Conv2d(256, 1, 1)

        # self.fusion = EnhancedFusionWithSqueeze()
        self.fusion = EnhancedFusionWithEntropyWeighting()
        self.sfd = selective_feature_decoupler(1, 1, 256, 256)

    # def _calculate_layer_norm(self, layer):
    #     """实时计算层的参数范数"""
    #     total_norm = 0.0
    #     for param in layer.parameters():
    #         if param.requires_grad:
    #             # total_norm += torch.norm(param).item()
    #             total_norm = total_norm + torch.norm(param)
    #     return total_norm

    def forward(self, x):
        ## x: [bs, 3, 256, 256]
        # encoder
        x1 = self.inc(x)                                  ## [bs, 64, 256, 256]
        x2 = self.down1(x1)                               ## [bs, 128, 128, 128]
        x3 = self.down2(x2)                               ## [bs, 256, 64, 64]
        x4 = self.down3(x3)                               ## [bs, 512, 32, 32]
        x5 = self.down4(x4)                               ## [bs, 1024, 16, 16]
        # norm1 = self._calculate_layer_norm(self.inc)      # x1对应的层
        # norm2 = self._calculate_layer_norm(self.down1)    # x2对应的层  
        # norm3 = self._calculate_layer_norm(self.down2)    # x3对应的层

        # norms_tensor = torch.tensor([norm1, norm2, norm3], device=x.device, dtype=x.dtype)
        # norm_weights = norms_tensor / norms_tensor.sum()  
    
        # decoder
        o_4_1 = self.up1_1(x5, x4)                        ## [bs, 512, 32, 32]   
        o_3_1 = self.up2_1(o_4_1, x3)                     ## [bs, 256, 64, 64]
        o_2_1 = self.up3_1(o_3_1, x2)                     ## [bs, 128, 128, 128]
        o_1_1 = self.up4_1(o_2_1, x1)                     ## [bs, 64, 256, 256]
        o_seg1 = self.out_1(o_1_1)                        ## [bs, 1, 256, 256]

        ske_strong, ske_alter = self.decouple_layer(x5)                    ## [bs, 128, 16, 16]
        mask_strong, mask_alter = self.aux_head(ske_strong, ske_alter)     ## [bs, 1, 256, 256]
        w, complement_weighted = self.fusion(o_seg1, mask_strong, mask_alter)

        # complement_out, loss_mi, unimportant = self.sfd(complement_weighted)

        # x1 = self.end_conv1(x1)        ## [bs, 1, 256, 256]
        # x2 = self.end_conv2(x2)        ## [bs, 1, 128, 128]
        # x3 = self.end_conv3(x3)        ## [bs, 1, 64, 64]

        # 返回浅层特征用于尾部loss计算
        return o_seg1, mask_strong, mask_alter, w, complement_weighted #, complement_out, loss_mi




# unet = Multi_decoder_Net(3)
# a = torch.rand(1, 3, 256, 256)
# o_seg1, mask_strong, mask_alter, w, complement_weighted = unet.forward(a)
# print(o_seg1.size())            # torch.Size([1, 1, 256, 256])
# print(f_fg.size())      
# print(f_bg.size())      
# print(x1.size())                # torch.Size([1, 1, 256, 256])
# print(x2.size())                # torch.Size([1, 1, 128, 128])
# print(x3.size())                # torch.Size([1, 1, 64, 64])
# print(complement_weighted.size())    # torch.Size([1, 1, 256, 256])
# print(loss.size())              # []