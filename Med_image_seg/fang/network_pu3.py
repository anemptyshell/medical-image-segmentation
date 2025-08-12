import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange, repeat
from functools import partial

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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        # Assuming the input from each block is concatenated along the channel dimension
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate along channel dimension
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiPathResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dilation=2, use_dilate_conv=True):
        super(BiPathResBlock, self).__init__()

        # Define two ResBlocks and two DilatedResBlocks in sequence for each path
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            ResBlock(mid_channels, mid_channels)
        )
        self.dilated_resblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            DilatedResBlock(mid_channels, mid_channels, dilation=dilation)
        )

        # Define the Fusion Block
        self.fusionblock = FusionBlock(2 * mid_channels, out_channels)
        self.use_dilate_conv = use_dilate_conv

    def forward(self, x):
        res_out = self.resblock(x)
        dilated_res_out = self.dilated_resblock(x)
        if self.use_dilate_conv:
            x = self.fusionblock(res_out, dilated_res_out)
        else:
            x = self.fusionblock(res_out, res_out)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, use_dilate_conv=True):
        super(CNNEncoder, self).__init__()

        # Define channel transitions from the input to the deepest layer
        channels = [3, 64, 128, 256, 512, 1024]
        self.layers = nn.ModuleList()

        for idx in range(1, len(channels)):  ## idx 1 2 3 4 5 
            self.layers.append(BiPathResBlock(channels[idx - 1], channels[idx], channels[idx], use_dilate_conv=use_dilate_conv))
            if idx != len(channels) - 1:
                self.layers.append(nn.MaxPool2d(2))

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, BiPathResBlock):  # Conditionally append feature maps following DoubleResBlock layers
                features.append(x)
        # # Include the final feature map post application of MaxPool2d layer for completeness of the hierarchical representations
        # features.append(x)

        return features


class PUnet(nn.Module):
    def __init__(self, use_dilate_conv=True, output_ch=1):
        super().__init__()
        self.Encoder1 = CNNEncoder(use_dilate_conv=use_dilate_conv)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):

        features = self.Encoder1(x)

        x5, x4, x3, x2, x1 = features[4], features[3], features[2], features[1], features[0]


        # decoder
        d5_1 = self.Up5(x5)      ## torch.Size([2, 512, 64, 64])
        d5_1 = torch.cat((x4, d5_1), dim=1)
        d5_1 = self.Up_conv5(d5_1)          ## torch.Size([2, 512, 64, 64])

        d5_2 = self.Up5(x5)
        d5_2 = torch.cat((x4, d5_2), dim=1)
        d5_2 = self.Up_conv5(d5_2)  

        d5_3 = self.Up5(x5)
        d5_3 = torch.cat((x4, d5_3), dim=1)
        d5_3 = self.Up_conv5(d5_3)  

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
        #####################################

        d1_1 = self.Conv_1x1(d2_1)          ## torch.Size([2, 1, 512, 512]) 
        d1_2 = self.Conv_1x1(d2_2) 
        d1_3 = self.Conv_1x1(d2_3) 

        return d1_1, d1_2, d1_3  
    
    



# if __name__ == '__main__':
#     input = torch.rand(1, 3, 256, 256)
#     punet = PUnet()
#     out1, out2, out3 = punet(input)
   
#     print(out1.size())
#     print(out2.size())
#     print(out3.size())


## encoder后 feature[0 1 2 3 4]分别为:
# torch.Size([1, 64, 512, 512])
# torch.Size([1, 128, 256, 256])
# torch.Size([1, 256, 128, 128])
# torch.Size([1, 512, 64, 64])
# torch.Size([1, 1024, 32, 32])