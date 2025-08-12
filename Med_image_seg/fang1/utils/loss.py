import torch
import torch.nn as nn
import torch.nn.functional as F
from Med_image_seg.fang.utils.loss_iou import IOU
from Med_image_seg.fang.utils.loss_ssim import SSIM

class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy_with_logits(preds, targets)
        smooth = 1e-5
        preds = torch.sigmoid(preds)
        num = targets.size(0)
        preds = preds.view(num, -1)
        targets = targets.view(num, -1)
        intersection = (preds * targets)
        dice = (2. * intersection.sum(1) + smooth) / (preds.sum(1) + targets.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice 

class soft_iou_loss(nn.Module):
    def __init__(self):
        super(soft_iou_loss, self).__init__()

    def forward(self, pred, label):
        b = pred.size()[0]
        pred = pred.view(b, -1)
        label = label.view(b, -1)
        inter = torch.sum(torch.mul(pred, label), dim=-1, keepdim=False)
        unit = torch.sum(torch.mul(pred, pred) + label, dim=-1, keepdim=False) - inter
        return torch.mean(1 - inter / (unit + 1e-10))




# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11,size_average=True)
iou_loss = IOU(size_average=True)
softiou_loss = soft_iou_loss()

class bce_ssim_loss(nn.Module):
    def __init__(self):
        super(bce_ssim_loss, self).__init__()
    
    def forward(self, preds, targets):
        m = nn.Sigmoid()
        lossinput1 = m(preds)
        lossinput2 = m(targets)
        bce_out = bce_loss(lossinput1, lossinput2)

        # bce_out = F.binary_cross_entropy_with_logits(preds, targets)

        ssim_out = 1 - ssim_loss(preds, targets)
        iou_out = iou_loss(preds, targets)
        loss = bce_out + ssim_out + iou_out
        
        return loss

class bceioussim_loss(nn.Module):
    def __init__(self):
        super(bceioussim_loss, self).__init__()

    def forward(self, preds, targets):

        bce_out = F.binary_cross_entropy_with_logits(preds, targets)

        # ssim_out = 1 - ssim_loss(preds, targets)

        iou_out = softiou_loss(preds, targets)
        # loss = bce_out + ssim_out + iou_out
        loss = bce_out + iou_out
        
        return loss




     