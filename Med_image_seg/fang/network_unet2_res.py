import torch
import torch.nn as nn
import torch.nn.functional as F
from Med_image_seg.fang.model_util.resnet import resnet34
# from model_util.resnet import resnet34
from torch.nn import init

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



class Fusion_Attention_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion_Attention_Module, self).__init__()

        self.Fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

        self.Attention1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.Sigmoid()
        )
        self.Attention2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):

        x = torch.cat((x1, x2), dim=1)
        x = self.Fusion(x)

        x_attention_1 = self.Attention1(x)
        x_attention_2 = self.Attention2(x)

        x_output_1 = x1 * x_attention_1
        x_output_2 = x2 * x_attention_2

        x_output_1 = F.relu(x_output_1)
        x_output_2 = F.relu(x_output_2)

        return x_output_1, x_output_2




"""unet + 3de + hidi"""
class UNet_2de(nn.Module):
    def __init__(self, input_ch=3, output_ch=1):
        super(UNet_2de, self).__init__()

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

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1

      
        self.backbone =resnet34(pretrained=True)

    def forward(self, x):

        ## x torch.Size([2, 3, 512, 512])
        """encoder"""
        # x1 = self.Conv1(x)     ## torch.Size([2, 64, 512, 512])

        # x2 = self.Maxpool(x1)
        # x2 = self.Conv2(x2)    ## torch.Size([2, 128, 256, 256])

        # x3 = self.Maxpool(x2)
        # x3 = self.Conv3(x3)    ## torch.Size([2, 256, 128, 128])

        # x4 = self.Maxpool(x3)
        # x4 = self.Conv4(x4)    ## torch.Size([2, 512, 64, 64])

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)    ## torch.Size([2, 1024, 32, 32])

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


        #####################################
        d4_1 = self.Up4(d5_1)
        d4_1 = torch.cat((x3, d4_1), dim=1)  
        d4_1 = self.Up_conv4(d4_1)          ## torch.Size([2, 256, 128, 128])
        
        d4_2 = self.Up4(d5_2)
        d4_2 = torch.cat((x3, d4_2), dim=1)  
        d4_2 = self.Up_conv4(d4_2)  
        

        #####################################
        d3_1 = self.Up3(d4_1)
        d3_1 = torch.cat((x2, d3_1), dim=1)
        d3_1 = self.Up_conv3(d3_1)          ## torch.Size([2, 128, 256, 256])
        
        d3_2 = self.Up3(d4_2)
        d3_2 = torch.cat((x2, d3_2), dim=1)
        d3_2 = self.Up_conv3(d3_2)  

        #####################################
        d2_1 = self.Up2(d3_1)
        d2_1 = torch.cat((x1, d2_1), dim=1)
        d2_1 = self.Up_conv2(d2_1)          ## torch.Size([2, 64, 512, 512]) 
        
        d2_2 = self.Up2(d3_2)
        d2_2 = torch.cat((x1, d2_2), dim=1)
        d2_2 = self.Up_conv2(d2_2) 


        #####################################
        d1_1 = self.Conv_1x1(d2_1)          ## torch.Size([2, 1, 512, 512]) 
        d1_2 = self.Conv_1x1(d2_2) 


        return d1_1, d1_2   


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)  


# unet = UNet_3decoder_hidi()
# a = torch.rand(2, 3, 256, 256)
# output1, output2 = unet.forward(a)

# print(output1.size())   # torch.Size([2, 1, 256, 256])
# print(output2.size()) 
