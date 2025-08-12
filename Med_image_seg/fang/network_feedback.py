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
        self.conv3 = nn.Sequential(nn.Conv2d(channel, channel // 2, kernel_size=1))

    def forward(self, x, y, z):
        # B, C, W, H = x.size()

        xy = self.conv1(torch.cat((x, y), 1))
        yz = self.conv1(torch.cat((y, z), 1))

        xy1 = xy
        xy2 = yz

        result = torch.where(xy1 > xy2, xy1, xy2)

        xx = xy1.view(xy1.shape[0], xy1.shape[1], -1)
        yy = xy2.view(xy2.shape[0], xy2.shape[1], -1)
        result = result.view(result.shape[0], result.shape[1], -1)
        #
        #
        sim1 = F.cosine_similarity(xx, result, dim=2)
        sim2 = F.cosine_similarity(yy, result, dim=2)

        a = self.mlp(sim1)
        b = self.mlp(sim2)
        # print(a.size())   ## torch.Size([2, 1])
        # print(b.size())

        wei = torch.cat((a, b), -1)
        w = F.softmax(wei, dim=-1)
        w1, w2 = torch.split(w, 1, -1)

        ##  [-2, 1], but got 2)    0 1 2 3    0 1 -2 -1
        
        z = self.conv3(torch.cat(
            (xy1 * w1.unsqueeze(-2).unsqueeze(-1).expand_as(xy1), xy2 * w2.unsqueeze(-2).unsqueeze(-1).expand_as(xy2)), 1))

        out = self.conv(z)
        return out



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Feedback(nn.Module): 
    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x, y, z):
        y= F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)  ##插值
        # print(y.size())
        z= F.interpolate(z, size=x.size()[2:], mode='bilinear', align_corners=True)
        # print(z.size())
        # print('**********')

        # out = self.conv2(self.conv1(x*y+x))
        out = self.conv2(self.conv1(x*y + x*z + x))

        return out


"""unet + 3de + feedback + hidi"""
class UNet_3de_feed_hidi(nn.Module):
    def __init__(self, input_ch=3, output_ch=1):
        super(UNet_3de_feed_hidi, self).__init__()

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

        self.fusion = Fusion(2, 1)
        self.backbone =resnet34(pretrained=True)

        self.feed1 = Feedback(64, 64)
        # self.feed2 = Feedback(64, 128)
        # self.feed3 = Feedback()
    


    def forward(self, x):

        ## x torch.Size([2, 3, 512, 512])

        """encoder"""
        ##--------------resnet----------------
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x1 = self.backbone.layer1(x)      ## torch.Size([2, 64, 512, 512])
        x2 = self.backbone.layer2(x1)     ## torch.Size([2, 128, 256, 256])
        x3 = self.backbone.layer3(x2)     ## torch.Size([2, 256, 128, 128])
        x4 = self.backbone.layer4(x3)     ## torch.Size([2, 512, 64, 64])

        x5 = self.backbone.maxpool(x4)
        x5 = self.Conv5(x5)               ## torch.Size([2, 1024, 32, 32])
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

        #####################################  

        """feedback"""

        x_f = self.feed1(x, d2_2, d2_3)
        # print(x_f.size())   ## [1, 64, 256, 256])
        x1_f = self.backbone.layer1(x_f)

        x1_f = self.feed1(x1_f, d2_2, d2_3)
        # print(x1_f.size())  ## [1, 64, 256, 256])

        x2_f = self.backbone.layer2(x1_f)
        x3_f = self.backbone.layer3(x2_f)    
        x4_f = self.backbone.layer4(x3_f)     
        x5_f = self.backbone.maxpool(x4_f)
        x5_f = self.Conv5(x5_f)

        #####################################

        """decoder"""
        d5_1_f = self.Up5(x5_f)                 ## torch.Size([2, 512, 64, 64])
        d5_1_f = torch.cat((x4_f, d5_1_f), dim=1)
        d5_1_f = self.Up_conv5(d5_1_f)          ## torch.Size([2, 512, 64, 64])
        
        d5_2_f = self.Up5(x5_f)
        d5_2_f = torch.cat((x4_f, d5_2_f), dim=1)
        d5_2_f = self.Up_conv5(d5_2_f)

        d5_3_f = self.Up5(x5_f)
        d5_3_f = torch.cat((x4_f, d5_3_f), dim=1)
        d5_3_f = self.Up_conv5(d5_3_f) 

        #####################################
        d4_1_f = self.Up4(d5_1_f)
        d4_1_f = torch.cat((x3_f, d4_1_f), dim=1)  
        d4_1_f = self.Up_conv4(d4_1_f)          ## torch.Size([2, 256, 128, 128])
        
        d4_2_f = self.Up4(d5_2_f)
        d4_2_f = torch.cat((x3_f, d4_2_f), dim=1)  
        d4_2_f = self.Up_conv4(d4_2_f)  
        
        d4_3_f = self.Up4(d5_3_f)
        d4_3_f = torch.cat((x3_f, d4_3_f), dim=1)  
        d4_3_f = self.Up_conv4(d4_3_f) 

        #####################################
        d3_1_f = self.Up3(d4_1_f)
        d3_1_f = torch.cat((x2_f, d3_1_f), dim=1)
        d3_1_f = self.Up_conv3(d3_1_f)          ## torch.Size([2, 128, 256, 256])
        
        d3_2_f = self.Up3(d4_2_f)
        d3_2_f = torch.cat((x2_f, d3_2_f), dim=1)
        d3_2_f = self.Up_conv3(d3_2_f)  

        d3_3_f = self.Up3(d4_3_f)
        d3_3_f = torch.cat((x2_f, d3_3_f), dim=1)
        d3_3_f = self.Up_conv3(d3_3_f)  

        #####################################
        d2_1_f = self.Up2(d3_1_f)
        d2_1_f = torch.cat((x1_f, d2_1_f), dim=1)
        d2_1_f = self.Up_conv2(d2_1_f)          ## torch.Size([2, 64, 512, 512]) 
        
        d2_2_f = self.Up2(d3_2_f)
        d2_2_f = torch.cat((x1_f, d2_2_f), dim=1)
        d2_2_f = self.Up_conv2(d2_2_f) 

        d2_3_f = self.Up2(d3_3_f)
        d2_3_f = torch.cat((x1_f, d2_3_f), dim=1)
        d2_3_f = self.Up_conv2(d2_3_f) 

        #####################################
        d1_1_f = self.Conv_1x1(d2_1_f)          ## torch.Size([2, 1, 512, 512]) 
        d1_2_f = self.Conv_1x1(d2_2_f) 
        d1_3_f = self.Conv_1x1(d2_3_f)   


        out = self.fusion(d1_2_f, d1_1_f, d1_3_f)

        return d1_1, d1_2, d1_3, d1_1_f, d1_2_f, d1_3_f, out   
        ## _, _, _, pred_raw, pred_edge, pred_skeleton, preds 


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)  





# unet = UNet_3de_feed_hidi()
# a = torch.rand(1, 3, 256, 256)
# _, _, _, output1, output2, output3, out = unet.forward(a)
# print(out.size())
# print(output1.size())   # torch.Size([2, 1, 512, 512])
# print(output2.size()) 
# print(output3.size()) 




