import torch
import torch.nn as nn
from Med_image_seg.fang1.model_util.resnet import resnet34
# from model_util.resnet import resnet34

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


class U_Net_resnet(nn.Module):
    def __init__(self, input_ch=3, output_ch=1):
        super(U_Net_resnet, self).__init__()

        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool = nn.MaxPool2d(kernel_size=2)
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

        self.backbone =resnet34(pretrained=True)

    def forward(self, x):
        ## x torch.Size([2, 3, 512, 512])
        # encoder
        ##--------------resnet----------------
        x = self.backbone.conv1(x)  ## [2, 64, 512, 512]
        x = self.backbone.bn1(x)    ## [2, 64, 512, 512]
        x = self.backbone.relu(x)   ## [2, 64, 512, 512]
        # print(x.size())

        x1 = self.backbone.layer1(x)   ## [2, 64, 512, 512]
        x2 = self.backbone.layer2(x1)  ## [2, 128, 256, 256]
        x3 = self.backbone.layer3(x2)  ## [2, 256, 128, 128]
        x4 = self.backbone.layer4(x3)  ## [2, 512, 64, 64]

        x5 = self.backbone.maxpool(x4)
        x5 = self.Conv5(x5) 

        # encoder
        # x1 = self.Conv1(x)     ## torch.Size([2, 64, 512, 512])

        # x2 = self.Maxpool(x1)
        # x2 = self.Conv2(x2)    ## torch.Size([2, 128, 256, 256])

        # x3 = self.Maxpool(x2)
        # x3 = self.Conv3(x3)    ## torch.Size([2, 256, 128, 128])

        # x4 = self.Maxpool(x3)
        # x4 = self.Conv4(x4)    ## torch.Size([2, 512, 64, 64])

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)    ## torch.Size([2, 1024, 32, 32])

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

    
    
"""输入一张图片测试"""
# import torch
# import torchvision
# from PIL import Image
# image_path = "ISIC_0000009.jpg"
# image = Image.open(image_path)
# tensor_image = torchvision.transforms.ToTensor()(image)
# print(tensor_image.size())     # torch.Size([3, 256, 256])


# unet = U_Net()
# output = unet.forward(tensor_image.unsqueeze(0))

# print(output.size())    # torch.Size([1, 1, 256, 256])

# output_image = output.squeeze(0)
# print(output_image.size())   # torch.Size([1, 256, 256])

# output_image = torchvision.transforms.ToPILImage()(output_image)
# output_image.show()


# """随机张量测试"""    
# unet = U_Net_resnet()
# a = torch.rand(2, 3, 512, 512)
# # a = torch.rand(2, 3, 224, 224)
# output = unet.forward(a)
# print(output.size())   # torch.Size([2, 1, 512, 512])

