import torch
import torch.nn as nn
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

  

class UNet(nn.Module):
    def __init__(self, input_ch=3, output_ch=1):
        super(UNet, self).__init__()

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


    def forward(self, x):
        ## x torch.Size([2, 3, 512, 512])

        # encoder
        x1 = self.Conv1(x)     ## torch.Size([2, 64, 512, 512])

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)    ## torch.Size([2, 128, 256, 256])

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)    ## torch.Size([2, 256, 128, 128])

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)    ## torch.Size([2, 512, 64, 64])

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)    ## torch.Size([2, 1024, 32, 32])


        # decoder
        d5 = self.Up5(x5)
        # print(d5.size())       ## torch.Size([2, 512, 64, 64])
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)          ## torch.Size([2, 512, 64, 64])

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)  
        d4 = self.Up_conv4(d4)          ## torch.Size([2, 256, 128, 128])

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)          ## torch.Size([2, 128, 256, 256])

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)          ## torch.Size([2, 64, 512, 512]) 

        d1 = self.Conv_1x1(d2)          ## torch.Size([2, 1, 512, 512]) 

        return d1  


class UNet_Multi_decoder(nn.Module):
    def __init__(self, input_ch=3, output_ch=1):
        super(UNet_Multi_decoder, self).__init__()

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


    def forward(self, x):
        ## x torch.Size([2, 3, 512, 512])

        # encoder
        x1 = self.Conv1(x)     ## torch.Size([2, 64, 512, 512])

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)    ## torch.Size([2, 128, 256, 256])

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)    ## torch.Size([2, 256, 128, 128])

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)    ## torch.Size([2, 512, 64, 64])

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)    ## torch.Size([2, 1024, 32, 32])


        # decoder
        d5_1 = self.Up5(x5)
        # print(d5.size())       ## torch.Size([2, 512, 64, 64])
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
   
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0) 





"""随机张量测试"""    
# unet = UNet_Multi_decoder()
# a = torch.rand(2, 3, 512, 512)
# # a = torch.rand(2, 3, 224, 224)
# output1, output2, output3 = unet.forward(a)
# print(output1.size())   # torch.Size([2, 1, 512, 512])
# print(output2.size()) 
# print(output3.size()) 




        # self.backbone =resnet34(pretrained=True)
        ####################################################
        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # c1 = self.backbone.relu(x)#1/2  64
        # ## c1 [2, 64, 512, 512]
 
        # x = self.backbone.maxpool(c1)
        # ## [2, 64, 256, 256]
        
        # c2 = self.backbone.layer1(x)#1/4   64
        # ## [2, 64, 256, 256]

        # c3 = self.backbone.layer2(c2)#1/8   128
        # ## [2, 128, 128, 128]

        # c4 = self.backbone.layer3(c3)#1/16   256
        # ## [2, 256, 64, 64]

        # c5 = self.backbone.layer4(c4)#1/32   512
        # ## [2, 512, 32, 32]

        ####################################################

        # # decoder
        # d5 = self.Up5(x5)
        # # print(d5.size())       ## torch.Size([2, 512, 64, 64])
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)          ## torch.Size([2, 512, 64, 64])

        # d4 = self.Up4(d5)
        # d4 = torch.cat((x3, d4), dim=1)  
        # d4 = self.Up_conv4(d4)          ## torch.Size([2, 256, 128, 128])

        # d3 = self.Up3(d4)
        # d3 = torch.cat((x2, d3), dim=1)
        # d3 = self.Up_conv3(d3)          ## torch.Size([2, 128, 256, 256])

        # d2 = self.Up2(d3)
        # d2 = torch.cat((x1, d2), dim=1)
        # d2 = self.Up_conv2(d2)          ## torch.Size([2, 64, 512, 512]) 

        # d1 = self.Conv_1x1(d2)          ## torch.Size([2, 1, 512, 512]) 

        # return d1  
    
        ####################################################

