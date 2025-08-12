import torch
import torch.nn as nn
import torch.nn.functional as F
# from Med_image_seg.fang.model_util.resnet import resnet34
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



"""from TMI2024 小波变换"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, if_relu=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.if_relu = if_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.bn2(out) + identity
        if self.if_relu:
            out = self.relu(out)
        return out



"""HIDI""" """from ISAANet"""
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
        # print("xy,yz:")
        # print(xy.size())
        # print(yz.size())

        xy1 = xy
        xy2 = yz

        result = torch.where(xy1 > xy2, xy1, xy2)
        # print(result.size())

        xx = xy1.view(xy1.shape[0], xy1.shape[1], -1)
        yy = xy2.view(xy2.shape[0], xy2.shape[1], -1)
        # print("xx,yy:")
        # print(xx.size())
        # print(yy.size())

        result = result.view(result.shape[0], result.shape[1], -1)
        # print(result.size())
        #
        #
        sim1 = F.cosine_similarity(xx, result, dim=2)
        sim2 = F.cosine_similarity(yy, result, dim=2)
        # print("sim1,sim2:")
        # print(sim1.size())
        # print(sim2.size())

        a = self.mlp(sim1)
        b = self.mlp(sim2)
        # print("a,b:")
        # print(a.size())   ## torch.Size([2, 1])
        # print(b.size())

        wei = torch.cat((a, b), -1)
        w = F.softmax(wei, dim=-1)
        w1, w2 = torch.split(w, 1, -1)

        ##  [-2, 1], but got 2)    0 1 2 3    0 1 -2 -1
        
        z = self.conv3(torch.cat(
            (xy1 * w1.unsqueeze(-2).unsqueeze(-1).expand_as(xy1), xy2 * w2.unsqueeze(-2).unsqueeze(-1).expand_as(xy2)), 1))
        # print("z")
        # print(z.size())

        out = self.conv(z)
        return out


"""EMI""" 
"""from HIDANet TIP 2023"""
class EMI(nn.Module):    
    def __init__(self,in_dim):
        super(EMI, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.layer_cat1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),)        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        k_size = 3
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_ful, x1, x2):

        ################################
        x = self.layer_cat1(torch.cat([x1, x2, x_ful],dim=1))

        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        out = self.relu(x_ful + x * y.expand_as(x))
        return out



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


class Mine(nn.Module):
    """
    definition of MINE
    """

    def __init__(self, input_size, hidden_size=200):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        return output


class selective_feature_decoupler(nn.Module):
    """
    definition of SFD
    """

    def __init__(self, in_c, out_c, in_h, in_w):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w

        # 3*3CBR 2
        self.c1 = nn.Sequential(
            CBR(in_c, in_c, kernel_size=3, padding=1),
            CBR(in_c, out_c, kernel_size=3, padding=1)
        )
        # 3*3CBR 2
        self.c2 = nn.Sequential(
            CBR(in_c, in_c, kernel_size=3, padding=1),
            CBR(in_c, out_c, kernel_size=3, padding=1)
        )

        # before MINE, reduce channel and size
        self.reduce_c = nn.Sequential(
            CBR(out_c, 16, kernel_size=1, padding=0),
            nn.MaxPool2d(2, stride=2)
        )

        # init MINE
        self.mine_net = Mine(input_size=16 * (in_h // 2) * (in_w // 2) * 2)
        # self.mine_net = Mine(input_size=524288)

    def mutual_information(self, joint, marginal):
        joint = joint.float().cuda() if torch.cuda.is_available() else joint.float()
        marginal = marginal.float().cuda() if torch.cuda.is_available() else marginal.float()
        # print('******************')
        # print(joint.size())     ## [2, 524288]

        t = self.mine_net(joint)
        et = torch.exp(self.mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(1 + torch.mean(et))

        return mi_lb

    def forward(self, x):
        # x:[B,C,H,W]
        s = self.c1(x)  # significant feature
        u = self.c2(x)  # unimportant feature
        # print(s.size())  ## 2, 64, 256, 256

        # reduce channel
        s_16 = self.reduce_c(s)
        u_16 = self.reduce_c(u)
        # print(s_16.size())  ## 2, 16, 128, 128

        # flatten s and u
        s_flat = torch.flatten(s_16, start_dim=1)
        u_flat = torch.flatten(u_16, start_dim=1)
        # print(s_flat.size())  ## 2, 262144

        # create joint and marginal
        joint = torch.cat([s_flat, u_flat], dim=1)
        marginal = torch.cat([s_flat, torch.roll(u_flat, shifts=1, dims=0)], dim=1)
        # print('---------------------')
        # print(joint.size())    ## [2, 524288]

        # calculate mi loss
        mi_lb = self.mutual_information(joint, marginal)
        loss_mi = mi_lb

        # sigmoid
        loss_mi = torch.sigmoid(loss_mi)

        return s, loss_mi




"""unet + 3de + sfd"""
class UNet_3de_sfd(nn.Module):
    def __init__(self, input_ch=3, output_ch=1):
        super(UNet_3de_sfd, self).__init__()

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
        # self.backbone =resnet34(pretrained=True)


        self.sfd = selective_feature_decoupler(64, 64, 256, 256)

        l5c = 64

        self.fa_l5 = BasicBlock(l5c*3, l5c, downsample=nn.Sequential(conv1x1(in_planes=l5c*3, out_planes=l5c), BatchNorm2d(l5c, momentum=BN_MOMENTUM)), if_relu=True)


    def forward(self, x):
        x1 = self.Conv1(x)     ## torch.Size([2, 64, 512, 512])

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)    ## torch.Size([2, 128, 256, 256])

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)    ## torch.Size([2, 256, 128, 128])

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)    ## torch.Size([2, 512, 64, 64])

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)    ## torch.Size([2, 1024, 32, 32])


        ## x torch.Size([2, 3, 512, 512])
        """encoder"""

        ##--------------resnet----------------
        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)

        # x1 = self.backbone.layer1(x)
        # x2 = self.backbone.layer2(x1) 
        # x3 = self.backbone.layer3(x2) 
        # x4 = self.backbone.layer4(x3)

        # x5 = self.backbone.maxpool(x4)
        # x5 = self.Conv5(x5) 
        # ##------------------------------------
       
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

        out = torch.cat((d2_1, d2_2, d2_3), dim=1)
        out = self.fa_l5(out)
        print("out:")
        print(out.size())

        out, loss_mi = self.sfd(out)
      
        #####################################
        d1_1 = self.Conv_1x1(out)          ## torch.Size([2, 1, 512, 512]) 
        d1_2 = self.Conv_1x1(d2_2) 
        d1_3 = self.Conv_1x1(d2_3)   


        return d1_1, d1_2, d1_3, loss_mi    


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)  




unet = UNet_3de_sfd()
a = torch.rand(1, 3, 256, 256)
output1, output2, output3, out, loss = unet.forward(a)
# print(out.size())
# print(output1.size())   # torch.Size([2, 1, 512, 512])
# print(output2.size()) 
# print(output3.size()) 

    













