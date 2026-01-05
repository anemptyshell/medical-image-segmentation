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



class UncertaintyAwareFusion(nn.Module):
    """不确定性感知融合模块"""
    def __init__(self, input_channels=1, hidden_channels=32):
        super().__init__()
        
        # 1. 特征融合网络
        self.fusion_net = nn.Sequential(
            nn.Conv2d(input_channels * 3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, input_channels, kernel_size=1)
        )
        
        # 2. 不确定性估计网络
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(input_channels * 3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()  # 输出0-1的不确定性权重
        )
        
        # 3. 差异计算卷积（用于计算预测间的不一致性）
        self.diff_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
    
    def _compute_predictions_disagreement(self, preds, preds_strong, preds_alter):
        """计算三个预测之间的不一致性"""
        # 计算两两之间的绝对差异
        diff_strong = torch.abs(preds - preds_strong)  # [bs, 1, h, w]
        diff_alter = torch.abs(preds - preds_alter)    # [bs, 1, h, w]
        diff_cross = torch.abs(preds_strong - preds_alter)  # [bs, 1, h, w]
        
        # 拼接差异图
        diff_concat = torch.cat([diff_strong, diff_alter, diff_cross], dim=1)  # [bs, 3, h, w]
        
        # 通过卷积网络计算综合不一致性
        disagreement = self.diff_conv(diff_concat)  # [bs, 1, h, w]
        
        return disagreement
    
    def _compute_confidence_based_uncertainty(self, preds, preds_strong, preds_alter):
        """基于置信度的不确定性估计"""
        # 对于二分类，预测值接近0.5时最不确定
        uncertainty_preds = 4.0 * preds * (1.0 - preds)  # 二次函数，0.5时最大为1.0
        uncertainty_strong = 4.0 * preds_strong * (1.0 - preds_strong)
        uncertainty_alter = 4.0 * preds_alter * (1.0 - preds_alter)
        
        # 平均不确定性
        avg_uncertainty = (uncertainty_preds + uncertainty_strong + uncertainty_alter) / 3.0
        
        return avg_uncertainty
    
    def forward(self, preds, preds_strong, preds_alter):
        """
        识别预测中的不确定区域
        
        Args:
            preds: 主分割预测 [bs, 1, h, w]
            preds_strong: 强化骨架预测 [bs, 1, h, w]
            preds_alter: 延伸骨架预测 [bs, 1, h, w]
            
        Returns:
            uncertainty_weights: 不确定性权重图 [bs, 1, h, w] (值越大越不确定)
            fused_features: 融合后的特征 [bs, 1, h, w]
        """
        bs, c, h, w = preds.shape
        
        # 1. 计算预测间的不一致性（差异越大，越不确定）
        disagreement = self._compute_predictions_disagreement(preds, preds_strong, preds_alter)
        
        # 2. 计算置信度不确定性（预测值接近0.5时最不确定）
        confidence_uncertainty = self._compute_confidence_based_uncertainty(
            preds, preds_strong, preds_alter
        )
        
        # 3. 拼接所有信息
        combined_input = torch.cat([preds, preds_strong, preds_alter], dim=1)  # [bs, 3, h, w]
        
        # 4. 通过神经网络学习不确定性（结合多种线索）
        learned_uncertainty = self.uncertainty_estimator(combined_input)  # [bs, 1, h, w]
        
        # 5. 综合不确定性（加权组合）
        # 不一致性 + 低置信度 + 学习的不确定性
        uncertainty_weights = 0.4 * disagreement + 0.3 * confidence_uncertainty + 0.3 * learned_uncertainty
        
        # 6. 特征融合（可选）
        fused_features = self.fusion_net(combined_input)  # [bs, 1, h, w]
        
        # 7. 应用非线性增强（使高不确定性区域更突出）
        uncertainty_weights = torch.sigmoid((uncertainty_weights - 0.5) * 6.0)
        
        return uncertainty_weights, fused_features



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
        self.end_conv1 = nn.Conv2d(64, 1, 1)
        self.end_conv2 = nn.Conv2d(128, 1, 1) 
        self.end_conv3 = nn.Conv2d(256, 1, 1)

        self.uncertainty_fusion = UncertaintyAwareFusion(input_channels=1)

        # 最终细化网络
        self.final_refiner = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 修正模块
        self.correction_module = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 输入：原始预测+不确定性权重
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),  # 修正量
            nn.Tanh()  # 输出[-1, 1]的修正值
        )

    def _calculate_layer_norm(self, layer):
        """实时计算层的参数范数"""
        total_norm = 0.0
        for param in layer.parameters():
            if param.requires_grad:
                # total_norm += torch.norm(param).item()
                total_norm = total_norm + torch.norm(param)
        return total_norm

    def forward(self, x):
        ## x: [bs, 3, 256, 256]
        # encoder
        x1 = self.inc(x)                                  ## [bs, 64, 256, 256]
        x2 = self.down1(x1)                               ## [bs, 128, 128, 128]
        x3 = self.down2(x2)                               ## [bs, 256, 64, 64]
        x4 = self.down3(x3)                               ## [bs, 512, 32, 32]
        x5 = self.down4(x4)                               ## [bs, 1024, 16, 16]
        norm1 = self._calculate_layer_norm(self.inc)      # x1对应的层
        norm2 = self._calculate_layer_norm(self.down1)    # x2对应的层  
        norm3 = self._calculate_layer_norm(self.down2)    # x3对应的层

        norms_tensor = torch.tensor([norm1, norm2, norm3], device=x.device, dtype=x.dtype)
        norm_weights = norms_tensor / norms_tensor.sum()  
    
        # decoder
        o_4_1 = self.up1_1(x5, x4)                        ## [bs, 512, 32, 32]   
        o_3_1 = self.up2_1(o_4_1, x3)                     ## [bs, 256, 64, 64]
        o_2_1 = self.up3_1(o_3_1, x2)                     ## [bs, 128, 128, 128]
        o_1_1 = self.up4_1(o_2_1, x1)                     ## [bs, 64, 256, 256]
        o_seg1 = self.out_1(o_1_1)                        ## [bs, 1, 256, 256]

        ske_strong, ske_alter = self.decouple_layer(x5)                    ## [bs, 128, 16, 16]
        mask_strong, mask_alter = self.aux_head(ske_strong, ske_alter)     ## [bs, 1, 256, 256]
        # 1. 识别不确定区域
        uncertainty_weights, _ = self.uncertainty_fusion(o_seg1, mask_strong, mask_alter)
        # uncertainty_weights: [bs, 1, 256, 256]，值越大表示越不确定

        # 2. 计算修正量
        correction_input = torch.cat([o_seg1, uncertainty_weights], dim=1)  # [bs, 2, 256, 256]
        correction = self.correction_module(correction_input)  # [bs, 1, 256, 256]，修正量
        
        # 3. 应用修正（不确定区域修正大，确定区域修正小）
        # 修正系数：0.5控制修正幅度
        corrected_preds = o_seg1 + uncertainty_weights * correction * 0.5
        corrected_preds = torch.clamp(corrected_preds, 0, 1)  # 限制在[0,1]

        # 4. 可选：最终细化
        final_output = self.final_refiner(corrected_preds)

        x1 = self.end_conv1(x1)        ## [bs, 1, 256, 256]
        x2 = self.end_conv2(x2)        ## [bs, 1, 128, 128]
        x3 = self.end_conv3(x3)        ## [bs, 1, 64, 64]

        # 返回浅层特征用于尾部loss计算
        return o_seg1, mask_strong, mask_alter, x1, x2, x3, norm_weights, uncertainty_weights, final_output




# unet = Multi_decoder_Net(3)
# a = torch.rand(1, 3, 256, 256)
# o_seg1, f_fg, f_bg, x1, x2, x3, norm_weights, w, final_output= unet.forward(a)
# print(o_seg1.size())        # torch.Size([1, 1, 256, 256])
# print(f_fg.size())  
# print(f_bg.size())  
# print(x1.size())            # torch.Size([1, 1, 256, 256])
# print(x2.size())            # torch.Size([1, 1, 128, 128])
# print(x3.size())            # torch.Size([1, 1, 64, 64])
# print(final_output.size())    # torch.Size([1, 1, 256, 256])