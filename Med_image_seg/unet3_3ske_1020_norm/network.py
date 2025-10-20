import torch
import torch.nn as nn
import torch.nn.functional as F
# from Med_image_seg.fang1.model_util.resnet import resnet34
# from model_util.resnet import resnet34
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
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc


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
        self.branch_uc = nn.Sequential(
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

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc



# class Multi_decoder_Net(nn.Module):
#     def __init__(self, n_channels, n_classes=1, bilinear=False):
#         super(Multi_decoder_Net, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         # encoder
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)

#         # decoder
#         self.up1_1 = Up(1024, 512, bilinear)
#         self.up1_2 = Up(1024, 512, bilinear)
#         self.up1_3 = Up(1024, 512, bilinear)

#         self.up2_1 = Up(512, 256, bilinear)
#         self.up2_2 = Up(512, 256, bilinear)
#         self.up2_3 = Up(512, 256, bilinear)

#         self.up3_1 = Up(256, 128, bilinear)
#         self.up3_2 = Up(256, 128, bilinear)
#         self.up3_3 = Up(256, 128, bilinear)

#         self.up4_1 = Up(128, 64, bilinear)
#         self.up4_2 = Up(128, 64, bilinear)
#         self.up4_3 = Up(128, 64, bilinear)

#         self.out_1 = OutConv(64, n_classes)
#         self.out_2 = OutConv(64, n_classes)
#         self.out_3 = OutConv(64, n_classes)

#         self.decouple_layer = DecoupleLayer(1024, 128)
#         self.aux_head = AuxiliaryHead(128)

#         self.edge_conv1 = nn.Conv2d(64, 1, 1).cuda()
#         self.edge_conv2 = nn.Conv2d(128, 1, 1).cuda() 
#         self.edge_conv3 = nn.Conv2d(256, 1, 1).cuda()


#     def forward(self, x):
#         # encoder
#         x1 = self.inc(x)     ## [1, 64, 256, 256]
#         x2 = self.down1(x1)  ## [1, 128, 128, 128]
#         x3 = self.down2(x2)  ## [1, 256, 64, 64]
#         x4 = self.down3(x3)  ## [1, 512, 32, 32]
#         x5 = self.down4(x4)  ## [1, 1024, 16, 16]
    
#         # decoder
#         o_4_1 = self.up1_1(x5, x4)
#         o_3_1 = self.up2_1(o_4_1, x3)
#         o_2_1 = self.up3_1(o_3_1, x2)
#         o_1_1 = self.up4_1(o_2_1, x1)
#         o_seg1 = self.out_1(o_1_1)
    
#         ske_strong, ske_alter, ske_edge = self.decouple_layer(x5)                    ## [1, 128, 16, 16]
#         mask_strong, mask_alter = self.aux_head(ske_strong, ske_alter, ske_edge)   ## [1, 1, 256, 256]

#         x1 = self.edge_conv1(x1)
#         x2 = self.edge_conv2(x2)
#         x3 = self.edge_conv3(x3)
    
#         # 返回浅层特征用于边界loss计算
#         return o_seg1, mask_strong, mask_alter, x1, x2, x3


class Multi_decoder_Net(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(Multi_decoder_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
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

        self.edge_conv1 = nn.Conv2d(64, 1, 1)
        self.edge_conv2 = nn.Conv2d(128, 1, 1)
        self.edge_conv3 = nn.Conv2d(256, 1, 1)
        
        # 初始化权重范数
        self._init_norm_weights()

    def _init_norm_weights(self):
        """计算并存储各层的参数范数"""
        # 编码器各层的范数
        self.inc_norm = self._calculate_layer_norm(self.inc)
        self.down1_norm = self._calculate_layer_norm(self.down1)
        self.down2_norm = self._calculate_layer_norm(self.down2)
        self.down3_norm = self._calculate_layer_norm(self.down3)
        self.down4_norm = self._calculate_layer_norm(self.down4)
        
        # 解码器各层的范数
        self.up1_1_norm = self._calculate_layer_norm(self.up1_1)
        self.up2_1_norm = self._calculate_layer_norm(self.up2_1)
        self.up3_1_norm = self._calculate_layer_norm(self.up3_1)
        self.up4_1_norm = self._calculate_layer_norm(self.up4_1)
        
        # 对范数进行归一化，使其和为1
        encoder_norms = [self.inc_norm, self.down1_norm, self.down2_norm, self.down3_norm, self.down4_norm]
        decoder_norms = [self.up1_1_norm, self.up2_1_norm, self.up3_1_norm, self.up4_1_norm]
        
        total_encoder_norm = sum(encoder_norms)
        total_decoder_norm = sum(decoder_norms)
        
        # 归一化的权重
        self.inc_weight = self.inc_norm / total_encoder_norm
        self.down1_weight = self.down1_norm / total_encoder_norm
        self.down2_weight = self.down2_norm / total_encoder_norm
        self.down3_weight = self.down3_norm / total_encoder_norm
        self.down4_weight = self.down4_norm / total_encoder_norm
        
        self.up1_1_weight = self.up1_1_norm / total_decoder_norm
        self.up2_1_weight = self.up2_1_norm / total_decoder_norm
        self.up3_1_weight = self.up3_1_norm / total_decoder_norm
        self.up4_1_weight = self.up4_1_norm / total_decoder_norm

    def _calculate_layer_norm(self, layer):
        """计算层的参数范数"""
        total_norm = 0.0
        for param in layer.parameters():
            if param.requires_grad:
                total_norm += torch.norm(param).item()
        return total_norm

    def _apply_norm_weight(self, x, weight):
        """对特征图应用范数权重"""
        # 将权重转换为与x相同的设备和数据类型
        weight_tensor = torch.tensor(weight, device=x.device, dtype=x.dtype)
        # 对特征图进行加权
        return x * weight_tensor

    def forward(self, x):
        # encoder - 对每个特征图应用范数权重
        x1 = self.inc(x)     ## [1, 64, 256, 256]
        x1_weighted = self._apply_norm_weight(x1, self.inc_weight)
        
        x2 = self.down1(x1_weighted)  ## [1, 128, 128, 128]
        x2_weighted = self._apply_norm_weight(x2, self.down1_weight)
        
        x3 = self.down2(x2_weighted)  ## [1, 256, 64, 64]
        x3_weighted = self._apply_norm_weight(x3, self.down2_weight)
        
        x4 = self.down3(x3_weighted)  ## [1, 512, 32, 32]
        x4_weighted = self._apply_norm_weight(x4, self.down3_weight)
        
        x5 = self.down4(x4_weighted)  ## [1, 1024, 16, 16]
        x5_weighted = self._apply_norm_weight(x5, self.down4_weight)
    
        # decoder - 对skip connection的特征图应用范数权重
        # 上采样时使用加权的skip connection特征
        o_4_1 = self.up1_1(x5_weighted, self._apply_norm_weight(x4, self.up1_1_weight))
        o_3_1 = self.up2_1(o_4_1, self._apply_norm_weight(x3, self.up2_1_weight))
        o_2_1 = self.up3_1(o_3_1, self._apply_norm_weight(x2, self.up3_1_weight))
        o_1_1 = self.up4_1(o_2_1, self._apply_norm_weight(x1, self.up4_1_weight))
        o_seg1 = self.out_1(o_1_1)
    
        ske_strong, ske_alter, ske_edge = self.decouple_layer(x5_weighted)  ## [1, 128, 16, 16]
        mask_strong, mask_alter, edge = self.aux_head(ske_strong, ske_alter, ske_edge)   ## [1, 1, 256, 256]

        # 用于边界loss的原始特征图（不加权）
        # x1_edge = self.edge_conv1(x1)
        # x2_edge = self.edge_conv2(x2)
        # x3_edge = self.edge_conv3(x3)
    
        # 返回浅层特征用于边界loss计算
        return o_seg1, mask_strong, mask_alter, edge#, x1_edge, x2_edge, x3_edge

    def update_norm_weights(self):
        """在训练过程中更新范数权重（可选）"""
        self._init_norm_weights()



# unet = Multi_decoder_Net(3)
# a = torch.rand(1, 3, 256, 256)
# o_seg1, f_fg, f_bg, edge= unet.forward(a)
# print(o_seg1.size())   # torch.Size([1, 1, 256, 256])
# print(f_fg.size()) 
# print(f_bg.size()) 
# print(x1.size())    ## torch.Size([1, 64, 256, 256])
# print(x2.size())    ## torch.Size([1, 128, 128, 128])
# print(x3.size())   ## torch.Size([1, 256, 64, 64]) 