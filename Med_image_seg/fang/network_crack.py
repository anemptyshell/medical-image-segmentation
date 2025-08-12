import torch
import torch.nn as nn
# from model_util.crackformer2 import *
from Med_image_seg.fang.model_util.crackformer2 import *


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

"""crackformer2 + unet 3 decoder"""
class UNet_Multi_decoder(nn.Module):
    def __init__(self, input_ch=3, output_ch=1):
        super(UNet_Multi_decoder, self).__init__()

        self.down1 = Down1_new()
        self.down2 = Down2_new()
        self.down3 = Down3_new()
        self.down4 = Down4_new()
        self.down5 = Down5_new()

        self.patch_embed = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)


        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        ## x torch.Size([2, 3, 512, 512])

        # encoder
        out1 = self.down1(inputs)  ## ([1, 64, 512, 512])  
        # print(out1.size())
        
        out2, _ = self.patch_embed(out1)
        out2 = self.down2(out2)    ## ([1, 128, 256, 256]) 

        out3, _  = self.patch_embed(out2)
        out3 = self.down3(out3)    ## ([1, 256, 128, 128]) 

        out4, _  = self.patch_embed(out3)
        out4 = self.down4(out4)    ## ([1, 512, 64, 64])  

        out5, _  = self.patch_embed(out4)
        out5 = self.down5(out5)    ## ([1, 1024, 32, 32])  
        # print(out5.size())


        # decoder
        d5_1 = self.Up5(out5)
        # print(d5.size())          ## torch.Size([2, 512, 64, 64])
        d5_1 = torch.cat((out4, d5_1), dim=1)
        d5_1 = self.Up_conv5(d5_1)          ## torch.Size([2, 512, 64, 64])

        # d5_2 = self.Up5(out5)
        # d5_2 = torch.cat((out4, d5_2), dim=1)
        # d5_2 = self.Up_conv5(d5_2)  

        # d5_3 = self.Up5(out5)
        # d5_3 = torch.cat((out4, d5_3), dim=1)
        # d5_3 = self.Up_conv5(d5_3)  

        #####################################
        d4_1 = self.Up4(d5_1)
        d4_1 = torch.cat((out3, d4_1), dim=1)  
        d4_1 = self.Up_conv4(d4_1)          ## torch.Size([2, 256, 128, 128])

        # d4_2 = self.Up4(d5_2)
        # d4_2 = torch.cat((out3, d4_2), dim=1)  
        # d4_2 = self.Up_conv4(d4_2)  

        # d4_3 = self.Up4(d5_3)
        # d4_3 = torch.cat((out3, d4_3), dim=1)  
        # d4_3 = self.Up_conv4(d4_3)  

        #####################################
        d3_1 = self.Up3(d4_1)
        d3_1 = torch.cat((out2, d3_1), dim=1)
        d3_1 = self.Up_conv3(d3_1)          ## torch.Size([2, 128, 256, 256])

        # d3_2 = self.Up3(d4_2)
        # d3_2 = torch.cat((out2, d3_2), dim=1)
        # d3_2 = self.Up_conv3(d3_2)  

        # d3_3 = self.Up3(d4_3)
        # d3_3 = torch.cat((out2, d3_3), dim=1)
        # d3_3 = self.Up_conv3(d3_3)  

        #####################################
        d2_1 = self.Up2(d3_1)
        d2_1 = torch.cat((out1, d2_1), dim=1)
        d2_1 = self.Up_conv2(d2_1)          ## torch.Size([2, 64, 512, 512]) 

        # d2_2 = self.Up2(d3_2)
        # d2_2 = torch.cat((out1, d2_2), dim=1)
        # d2_2 = self.Up_conv2(d2_2) 

        # d2_3 = self.Up2(d3_3)
        # d2_3 = torch.cat((out1, d2_3), dim=1)
        # d2_3 = self.Up_conv2(d2_3) 
        #####################################

        d1_1 = self.Conv_1x1(d2_1)          ## torch.Size([2, 1, 512, 512]) 
        # d1_2 = self.Conv_1x1(d2_2) 
        # d1_3 = self.Conv_1x1(d2_3) 

        return d1_1 #, d1_2, d1_3  

         



# if __name__ == '__main__':

#     unet = UNet_Multi_decoder()
#     a = torch.rand(2, 3, 256, 256)
#     # a = torch.rand(2, 3, 224, 224)
#     output1= unet.forward(a)
#     # unet.forward(a)
#     print(output1.size())   # torch.Size([2, 1, 512, 512])
    # print(output2.size()) 
    # print(output3.size()) 
    



