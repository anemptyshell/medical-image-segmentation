""" Full assembly of the parts to form the complete network """

from Med_image_seg.selfregunet.model_util.unet_parts import *
# from model_util.unet_parts import *

class SelfRegUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SelfRegUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        encoder = []
        decoder = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        encoder = [x2,x3,x4,x5]
        
        x = self.up1(x5, x4)
        decoder.append(x)
        x = self.up2(x, x3)
        decoder.append(x)
        x = self.up3(x, x2)
        decoder.append(x)
        x = self.up4(x, x1)
        final = x
        logits = self.outc(x)
        if self.training:
            return logits, encoder, decoder, final 
        else:
            return logits


## 注意n_channels
# net =  SelfRegUNet(n_channels=3, n_classes=1, bilinear=True).cuda()

# a = torch.rand(4, 3, 512, 512).cuda()
# torch.cuda.empty_cache()
# output = net.forward(a)
# print('------------------')
# print('output')
# print(len(output))      # 4
# print(type(output))     # <class 'tuple'>
# print(type(output[0]))  # <class 'torch.Tensor'>
# print(output[0].size())  # torch.Size([4, 1, 512, 512])