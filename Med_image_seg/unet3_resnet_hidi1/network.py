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

class Fusion(nn.Module):
    def __init__(self, channel, ratio):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channel // 2, channel // 2 // ratio),
            nn.ReLU(),
            nn.Linear(channel // 2 // ratio, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(nn.Conv2d(channel // 2, channel // 2, kernel_size=3, padding=1), nn.BatchNorm2d(channel // 2),
                                  nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        # self.conv3 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=1))
        self.conv3 = nn.Sequential(nn.Conv2d(channel // 2, channel // 2, kernel_size=1))

    def forward(self, x, y, z):
        # y 作为 object map

        xy = self.conv1(torch.cat((x, y), 1))
        yz = self.conv1(torch.cat((y, z), 1))
        xz = self.conv1(torch.cat((x, z), 1))
        # print("xy,yz:")
        # print(xy.size())
        # print(yz.size())

        xy1 = xy
        xy2 = yz
        xy3 = xz

        result1 = torch.where(xy1 > xy2, xy1, xy2)
        resultk = torch.where(result1 > xy3, result1, xy3)
        # print(resultk.size())   ## [1, 1, 256, 256]

        # xx = xy1.view(xy1.shape[0], xy1.shape[1], -1)
        # yy = xy2.view(xy2.shape[0], xy2.shape[1], -1)
     
        y_view = y.view(y.shape[0], y.shape[1], -1)
        resultk_view = resultk.view(resultk.shape[0], resultk.shape[1], -1)
        # print(resultk_view.size())    ## [1, 1, 65536]
        # print('+++++++++++++')
        
        sim = F.cosine_similarity(y_view, resultk_view, dim=2)
        # sim2 = F.cosine_similarity(yy, result, dim=2)
        # print("sim1,sim2:")
        # print(sim.size())   ## [1, 1]
        # print(sim2.size())

        w = self.mlp(sim)
        # print(w.size())   ## [1, 1]

        w = F.softmax(w, dim=-1)
        # print(w.size())   ## [1, 1]


        # a = self.mlp(sim1)
        # b = self.mlp(sim2)
        # wei = torch.cat((a, b), -1)
        # w = F.softmax(wei, dim=-1)
        # w1, w2 = torch.split(w, 1, -1)
        # print(w1.size())

        ##  [-2, 1], but got 2)    0 1 2 3    0 1 -2 -1

        # z = self.conv3(torch.cat(
        #     (xy1 * w1.unsqueeze(-2).unsqueeze(-1).expand_as(xy1), xy2 * w2.unsqueeze(-2).unsqueeze(-1).expand_as(xy2)), 1))


        output = self.conv3(y * w.unsqueeze(-2).unsqueeze(-1).expand_as(y))   ## [1, 1, 256, 256]
        # print("output")
        # print(output.size())

        out = self.conv(output)
        # print("out:")
        # print(out.size())
        return out

logging.basicConfig(
    filename='/home/my/mis-ft/medical-image-segmentation/fusion_max_indices.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='a'
)


class Fusion1(nn.Module):  
    def __init__(self, channel, ratio):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channel // 2, channel // 2 // ratio),
            nn.ReLU(),
            nn.Linear(channel // 2 // ratio, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(nn.Conv2d(channel // 2, channel // 2, kernel_size=3, padding=1), nn.BatchNorm2d(channel // 2),
                                  nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        # self.conv3 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=1))
        self.conv3 = nn.Sequential(nn.Conv2d(channel // 2, channel // 2, kernel_size=1))
        self.forward_count = 0


    def forward(self, x, y, z):
        self.forward_count += 1

        xy = self.conv1(torch.cat((x, y), 1))
        yz = self.conv1(torch.cat((y, z), 1))
        xz = self.conv1(torch.cat((x, z), 1))
        
        xy1 = xy
        xy2 = yz
        xy3 = xz
    
        # 创建堆叠的张量以便找到最大值索引
        stacked = torch.stack([xy1, xy2, xy3], dim=0)  # [3, batch, channel, H, W]
        
        # 找到最大值和对应的索引
        max_vals, max_indices = torch.max(stacked, dim=0)

        # 创建掩码：当 max_indices != 2 时为 1，否则为 0
        mask = (max_indices != 2).float()  # [batch, channel, H, W]

        # 0: 来自 xy1
        # 1: 来自 xy2  
        # 2: 来自 xy3
        
        total_pixels = max_indices.numel()
        count_xy1 = (max_indices == 0).sum().item()
        count_xy2 = (max_indices == 1).sum().item()
        count_xy3 = (max_indices == 2).sum().item()
            
        # 计算百分比
        perc_xy1 = count_xy1 / total_pixels * 100
        perc_xy2 = count_xy2 / total_pixels * 100
        perc_xy3 = count_xy3 / total_pixels * 100
        
        # 保存到日志
        logging.info(f"Forward pass {self.forward_count}:")
        logging.info(f"  来自 xy1 的像素数: {count_xy1} ({perc_xy1:.2f}%)")
        logging.info(f"  来自 xy2 的像素数: {count_xy2} ({perc_xy2:.2f}%)")
        logging.info(f"  来自 xy3 的像素数: {count_xy3} ({perc_xy3:.2f}%)")
        logging.info(f"  总像素数: {total_pixels}")
        
        # 同时在控制台输出（可选）
        # print(f"Forward {self.forward_count}: xy1={perc_xy1:.1f}%, xy2={perc_xy2:.1f}%, xy3={perc_xy3:.1f}%")
        
        resultk = max_vals  

        y_view = y.view(y.shape[0], y.shape[1], -1)
        resultk_view = resultk.view(resultk.shape[0], resultk.shape[1], -1)
        sim = F.cosine_similarity(y_view, resultk_view, dim=2)
        w = self.mlp(sim)
        w = F.softmax(w, dim=-1)
        output = self.conv3(y * w.unsqueeze(-2).unsqueeze(-1).expand_as(y))
        out = self.conv(output)

        return out, mask  # 返回输出和掩码



class Fusion2(nn.Module):
    def __init__(self, channel, ratio):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 8),  # 输入3个统计特征
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(nn.Conv2d(channel // 2, channel // 2, kernel_size=3, padding=1), nn.BatchNorm2d(channel // 2),
                                  nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        # self.conv3 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=1))
        self.conv3 = nn.Sequential(nn.Conv2d(channel // 2, channel // 2, kernel_size=1))
        self.forward_count = 0


    def forward(self, x, y, z):
        self.forward_count += 1

        # 打印三个特征图的统计信息
        print(f"x - shape: {x.shape}")
        print(f"x - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        print(f"y - min: {y.min().item():.4f}, max: {y.max().item():.4f}, mean: {y.mean().item():.4f}")
        print(f"z - min: {z.min().item():.4f}, max: {z.max().item():.4f}, mean: {z.mean().item():.4f}")
        
        # 还可以打印前几个值看看
        print(f"x前10个值: {x.flatten()[:10]}")
        print(f"y前10个值: {y.flatten()[:10]}")
        print(f"z前10个值: {z.flatten()[:10]}")

        # 融合特征
        xy = self.conv1(torch.cat((x, y), 1))
        yz = self.conv1(torch.cat((y, z), 1))
        xz = self.conv1(torch.cat((x, z), 1))
        
        xy1 = xy
        xy2 = yz
        xy3 = xz

        # 打印三个特征图的统计信息
        print(f"xy1 - shape: {xy1.shape}")
        print(f"xy1 - min: {xy1.min().item():.4f}, max: {xy1.max().item():.4f}, mean: {xy1.mean().item():.4f}")
        print(f"xy2 - min: {xy2.min().item():.4f}, max: {xy2.max().item():.4f}, mean: {xy2.mean().item():.4f}")
        print(f"xy3 - min: {xy3.min().item():.4f}, max: {xy3.max().item():.4f}, mean: {xy3.mean().item():.4f}")
        
        # 还可以打印前几个值看看
        print(f"xy1前10个值: {xy1.flatten()[:10]}")
        print(f"xy2前10个值: {xy2.flatten()[:10]}")
        print(f"xy3前10个值: {xy3.flatten()[:10]}")

        # 最大值选择并记录来源
        result1 = torch.where(xy1 > xy2, xy1, xy2)
        source1 = torch.where(xy1 > xy2, torch.zeros_like(xy1), torch.ones_like(xy2))
        
        resultk = torch.where(result1 > xy3, result1, xy3)
        source_final = torch.where(result1 > xy3, source1, 2 * torch.ones_like(xy3))
        
        # 统计来源信息
        count_xy1 = (source_final == 0).sum().item()
        count_xy2 = (source_final == 1).sum().item()
        count_xy3 = (source_final == 2).sum().item()
        total_pixels = source_final.numel()
        
        # 记录到日志
        logging.info(f"Forward pass {self.forward_count}:")
        logging.info(f"  来自 xy1 的像素数: {count_xy1} ({count_xy1/total_pixels*100:.1f}%)")
        logging.info(f"  来自 xy2 的像素数: {count_xy2} ({count_xy2/total_pixels*100:.1f}%)")
        logging.info(f"  来自 xy3 的像素数: {count_xy3} ({count_xy3/total_pixels*100:.1f}%)")
        
        # 计算多个统计特征作为MLP输入
        y_view = y.view(y.shape[0], y.shape[1], -1)
        resultk_view = resultk.view(resultk.shape[0], resultk.shape[1], -1)
        
        # 计算3个不同的相似度统计量
        cosine_sim = F.cosine_similarity(y_view, resultk_view, dim=2).mean(dim=1)  # 平均余弦相似度
        l2_distance = torch.norm(y_view - resultk_view, p=2, dim=2).mean(dim=1)   # L2距离
        correlation = self._compute_correlation(y_view, resultk_view)              # 相关系数
        
        # 组合特征
        sim_features = torch.stack([cosine_sim, l2_distance, correlation], dim=1)  # [batch_size, 3]
        
        # 计算权重
        w = self.mlp(sim_features)  # [batch_size, 1]
        
        # 应用权重
        weighted_y = y * w.view(-1, 1, 1, 1)  # 广播权重到整个特征图
        
        # 最终输出
        output = self.conv3(weighted_y)
        out = self.conv(output)

        return out

    def _compute_correlation(self, x, y):
        """计算两个特征图之间的相关系数"""
        batch_size, channels, spatial = x.shape
        
        # 重塑为 [batch_size, channels, spatial] -> [batch_size, spatial, channels]
        x_reshaped = x.transpose(1, 2)
        y_reshaped = y.transpose(1, 2)
        
        # 计算相关系数
        x_centered = x_reshaped - x_reshaped.mean(dim=1, keepdim=True)
        y_centered = y_reshaped - y_reshaped.mean(dim=1, keepdim=True)
        
        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = torch.sqrt((x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1) + 1e-8)
        
        correlation = numerator / denominator
        return correlation.mean(dim=1)  # 返回批次中每个样本的平均相关系数


class Fusion2(nn.Module):
    def __init__(self, channel, ratio):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 8),  # 输入3个统计特征
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(nn.Conv2d(channel // 2, channel // 2, kernel_size=3, padding=1), nn.BatchNorm2d(channel // 2),
                                  nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(channel // 2), nn.ReLU(inplace=True))
        # self.conv3 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=1))
        self.conv3 = nn.Sequential(nn.Conv2d(channel // 2, channel // 2, kernel_size=1))


    def forward(self, x, y, z):
        # [1, 128, 16, 16]
        # 融合特征
        xy = self.conv1(torch.cat((x, y), 1))
        yz = self.conv1(torch.cat((y, z), 1))
        xz = self.conv1(torch.cat((x, z), 1))
        
        xy1 = xy      # [1, 64, 16, 16]
        xy2 = yz
        xy3 = xz

        result1 = torch.where(xy1 > xy2, xy1, xy2)
        resultk = torch.where(result1 > xy3, result1, xy3)   # # [1, 64, 16, 16]
        
        y_view = y.view(y.shape[0], y.shape[1], -1)   
        resultk_view = resultk.view(resultk.shape[0], resultk.shape[1], -1)    ## # [1, 64, 256]
        
        cosine_sim = F.cosine_similarity(y_view, resultk_view, dim=2).mean(dim=1)   # [1, 64]  [1, 1]
        sim_features = cosine_sim
        
        w = self.mlp(sim_features)    # w: [1, 1] 
        weighted_y = y * w.view(-1, 1, 1, 1)  # 广播权重到整个特征图 [1, 128, 16, 16]

        output = self.conv3(weighted_y)
        out = self.conv(output)    # [1, 64, 16, 16]
        return out



"""unet + 3de + hidi"""
class UNet3_resnet_hidi(nn.Module):
    def __init__(self, input_ch=3, output_ch=1):
        super(UNet3_resnet_hidi, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=input_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


        self.fusion = Fusion2(2, 1)
        self.backbone =resnet34(pretrained=True)


    def forward(self, x):

        ## x torch.Size([2, 3, 512, 512])
        """encoder"""

        ##--------------resnet----------------
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1) 
        x3 = self.backbone.layer3(x2) 
        x4 = self.backbone.layer4(x3)

        x5 = self.backbone.maxpool(x4)
        x5 = self.Conv5(x5) 
        ##------------------------------------
       
        """decoder"""
        d5_1 = self.Up5(x5)                 ## torch.Size([2, 512, 64, 64])
        d5_1 = torch.cat((x4, d5_1), dim=1)
        d5_1 = self.Up_conv5(d5_1)          ## torch.Size([2, 512, 64, 64])
        
        d5_2 = self.Up5(x5)
        d5_2 = torch.cat((x4, d5_2), dim=1)
        d5_2 = self.Up_conv5(d5_2)

        d5_3 = self.Up5(x5)
        d5_3 = torch.cat((x4, d5_3), dim=1)
        d5_3 = self.Up_conv5(d5_3) 

        # d5_1_1 = self.emi1(d5_1, d5_2, d5_3)
        # d5_1_1 = self.CDFA1(d5_1, d5_2, d5_3)
        # print(d5_1_1.size())        ## torch.Size([2, 512, 32, 32])

        #####################################
        d4_1 = self.Up4(d5_1)
        d4_1 = torch.cat((x3, d4_1), dim=1)  
        d4_1 = self.Up_conv4(d4_1)          ## torch.Size([2, 256, 128, 128])
        
        d4_2 = self.Up4(d5_2)
        d4_2 = torch.cat((x3, d4_2), dim=1)  
        d4_2 = self.Up_conv4(d4_2)  
        
        d4_3 = self.Up4(d5_3)
        d4_3 = torch.cat((x3, d4_3), dim=1)  
        d4_3 = self.Up_conv4(d4_3) 

        # d4_1_1 = self.emi2(d4_1, d4_2, d4_3)
        # d4_1_1 = self.CDFA2(d4_1, d4_2, d4_3)

        #####################################
        d3_1 = self.Up3(d4_1)
        d3_1 = torch.cat((x2, d3_1), dim=1)
        d3_1 = self.Up_conv3(d3_1)          ## torch.Size([2, 128, 256, 256])
        
        d3_2 = self.Up3(d4_2)
        d3_2 = torch.cat((x2, d3_2), dim=1)
        d3_2 = self.Up_conv3(d3_2)  

        d3_3 = self.Up3(d4_3)
        d3_3 = torch.cat((x2, d3_3), dim=1)
        d3_3 = self.Up_conv3(d3_3)  

        # d3_1_1 = self.emi3(d3_1, d3_2, d3_3)
        # d3_1_1 = self.CDFA3(d3_1, d3_2, d3_3)

        #####################################
        d2_1 = self.Up2(d3_1)
        d2_1 = torch.cat((x1, d2_1), dim=1)
        d2_1 = self.Up_conv2(d2_1)          ## torch.Size([2, 64, 512, 512]) 
        
        d2_2 = self.Up2(d3_2)
        d2_2 = torch.cat((x1, d2_2), dim=1)
        d2_2 = self.Up_conv2(d2_2) 

        d2_3 = self.Up2(d3_3)
        d2_3 = torch.cat((x1, d2_3), dim=1)
        d2_3 = self.Up_conv2(d2_3) 

        # d2_1_1 = self.emi4(d2_1, d2_2, d2_3)
        # d2_1_1 = self.CDFA4(d2_1, d2_2, d2_3)
        # print("-------")
        # print(d2_1.size())

        #####################################
        d1_1 = self.Conv_1x1(d2_1)          ## torch.Size([2, 1, 512, 512]) 
        d1_2 = self.Conv_1x1(d2_2) 
        d1_3 = self.Conv_1x1(d2_3)   

        out = self.fusion(d1_2, d1_1, d1_3)

        return d1_1, d1_2, d1_3, out   


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)  



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

        self.fusion = Fusion1(2, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        o_4_1 = self.up1_1(x5, x4)
        o_4_2 = self.up1_2(x5, x4)
        o_4_3 = self.up1_3(x5, x4)

        o_3_1 = self.up2_1(o_4_1, x3)
        o_3_2 = self.up2_2(o_4_2, x3)
        o_3_3 = self.up2_3(o_4_3, x3)

        o_2_1 = self.up3_1(o_3_1, x2)
        o_2_2 = self.up3_2(o_3_2, x2)
        o_2_3 = self.up3_3(o_3_3, x2)

        o_1_1 = self.up4_1(o_2_1, x1)
        o_1_2 = self.up4_2(o_2_2, x1)
        o_1_3 = self.up4_3(o_2_3, x1)

        o_seg1 = self.out_1(o_1_1)
        o_seg2 = self.out_2(o_1_2)
        o_seg3 = self.out_3(o_1_3)

        # # 打印三个特征图的统计信息
        # print(f"o_seg1 - shape: {x.shape}")
        # print(f"o_seg1 - min: {o_seg1.min().item():.4f}, max: {o_seg1.max().item():.4f}, mean: {o_seg1.mean().item():.4f}")
        # print(f"o_seg2 - min: {o_seg2.min().item():.4f}, max: {o_seg2.max().item():.4f}, mean: {o_seg2.mean().item():.4f}")
        # print(f"o_seg3 - min: {o_seg3.min().item():.4f}, max: {o_seg3.max().item():.4f}, mean: {o_seg3.mean().item():.4f}")
        
        # # 还可以打印前几个值看看
        # print(f"x前10个值: {o_seg1.flatten()[:10]}")
        # print(f"y前10个值: {o_seg2.flatten()[:10]}")
        # print(f"z前10个值: {o_seg3.flatten()[:10]}")

        out, mask = self.fusion(o_seg2, o_seg1, o_seg3)

        # if self.n_classes > 1:
        #     seg1 = F.softmax(o_seg1, dim=1)
        #     seg2 = F.softmax(o_seg2, dim=1)
        #     seg3 = F.softmax(o_seg3, dim=1)
        #     return seg1, seg2, seg3
        # elif self.n_classes == 1:
        #     seg1 = torch.sigmoid(o_seg1)
        #     seg2 = torch.sigmoid(o_seg2)
        #     seg3 = torch.sigmoid(o_seg3)
        #     return seg1, seg2, seg3
        return o_seg1, o_seg2, o_seg3, out, mask


# unet = Multi_decoder_Net(3)
# a = torch.rand(1, 3, 256, 256)
# output1, output2, output3 = unet.forward(a)
# # print(out.size())
# print(output1.size())   # torch.Size([2, 1, 512, 512])
# print(output2.size()) 
# print(output3.size()) 

    













