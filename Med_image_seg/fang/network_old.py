import torch.nn.functional as F
import torch
import torch.nn as nn
# from Med_image_seg.fang.model_util.resnet import resnet34
from model_util.resnet import resnet34


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
            # self.up = nn.Sequential(
            #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #     nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), stride=1)
            # )
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        cat_x = torch.cat((x1, x2), 1)
        output = self.conv(cat_x)
        return output


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


'''
model
'''


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # # decoder
        # self.up1 = Up(1024, 512, bilinear)
        # self.up2 = Up(512, 256, bilinear)
        # self.up3 = Up(256, 128, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.out = OutConv(64, n_classes)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.out = OutConv(64, n_classes)
        self.out = OutConv(32, n_classes)
        
        ## 新加的
        self.backbone =resnet34(pretrained=True)
        self.up = nn.ConvTranspose2d(64, 128 // 2, kernel_size=1, stride=1)
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        # encoder
        # x1 = self.inc(x)

        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # print(x1.size(),x2.size(),x3.size(),x4.size(),x5.size())
        ## torch.Size([2, 64, 512, 512]) torch.Size([2, 128, 256, 256]) torch.Size([2, 256, 128, 128]) torch.Size([2, 512, 64, 64]) torch.Size([2, 1024, 32, 32])

        x = self.backbone.conv1(x)
        print(x.size())         ## torch.Size([1, 64, 512, 512])
        print('-----')
        c1 = self.backbone.relu(x) #1/2  64
        print(c1.size())       ## torch.Size([1, 64, 512, 512])

        # x = self.backbone.maxpool(c1)
        # print(x.size())         ## torch.Size([1, 64, 256, 256])

        # x = self.backbone.bn1(c1)
        c2 = self.backbone.layer1(c1) #1/4   64
        c3 = self.backbone.layer2(c2) #1/8   128
        c4 = self.backbone.layer3(c3) #1/16   256
        c5 = self.backbone.layer4(c4) #1/32   512
        print(c2.size(),c3.size(),c4.size(),c5.size())
        ## torch.Size([1, 64, 512, 512]) torch.Size([1, 128, 256, 256]) torch.Size([1, 256, 128, 128]) torch.Size([1, 512, 64, 64])


        # decoder
        # o_4 = self.up1(x5, x4)
        # o_3 = self.up2(o_4, x3)
        # o_2 = self.up3(o_3, x2)
        # o_1 = self.up4(o_2, x1)
        # o_seg = self.out(o_1)
        # print(o_4.size(),o_3.size(),o_2.size(),o_1.size())
        # ## torch.Size([2, 512, 64, 64]) torch.Size([2, 256, 128, 128]) torch.Size([2, 128, 256, 256]) torch.Size([2, 64, 512, 512])
        # print(o_seg.size())


        o_4 = self.up1(c5, c4) 
        # print('o_4',o_4.size())  ## torch.Size([2, 256, 64, 64])

        o_3 = self.up2(o_4, c3)  
        # print('o_3',o_3.size())  ## torch.Size([2, 128, 128, 128])

        o_2 = self.up3(o_3, c2)
        # print('o_2',o_2.size())  ## torch.Size([2, 64, 256, 256])

        o_11 = self.up(o_2)
        cat_x = torch.cat((o_11, x), 1) 
        o_1 = self.conv(cat_x)   ## torch.Size([2, 64, 256, 256])

        # o_1 = self.up4(o_2, c1)
        # print('o_1',o_1.size())    ## 预计torch.Size([2, 64, 512, 512])


        o_seg = self.out(o_1)
        # o_seg = self.out(o_2)

        if self.n_classes > 1:
            seg = F.softmax(o_seg, dim=1)
            return seg
        elif self.n_classes == 1:
            # seg = torch.sigmoid(o_seg)
            seg = o_seg
            return seg


if __name__ == '__main__':
   

    model = UNet(3)
    input = torch.rand(1, 3, 512, 512)
    output = model.forward(input)
    print(output.size())   ## torch.Size([2, 1, 512, 512])



