import torch
import torch.nn as nn
import torch.nn.functional as F


# class BCELoss(nn.Module):
#     def __init__(self):
#         super(BCELoss, self).__init__()
#         self.bceloss = nn.BCELoss()

#     def forward(self, pred, target):
#         size = pred.size(0)
#         pred_ = pred.view(size, -1)
#         target_ = target.view(size, -1)

#         return self.bceloss(pred_, target_)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, pred, target):
        
        return F.binary_cross_entropy_with_logits(pred, target)
    
    
# class BCEloss(nn.Module):
#     def __init__(self):
#         super(BCELoss, self).__init__()
#         self.bceloss = F.binary_cross_entropy_with_logits()

#     def forward(self, pred, target):
        
#         return self.bceloss(pred, target)

## DiceLoss的实现一样
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-5
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


# class BceDiceLoss(nn.Module):
#     def __init__(self, wb=0.5, wd=1):
#         super(BceDiceLoss, self).__init__()
#         self.bce = BCELoss()
#         self.dice = DiceLoss()
#         self.wb = wb
#         self.wd = wd

#     def forward(self, pred, target):
#         bceloss = self.bce(pred, target)
#         diceloss = self.dice(pred, target)

#         loss = self.wd * diceloss + self.wb * bceloss
#         return loss

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


# class GT_BceDiceLoss(nn.Module):
#     def __init__(self, wb=1, wd=1):
#         super(GT_BceDiceLoss, self).__init__()
#         self.bcedice = BceDiceLoss(wb, wd)

#     def forward(self, gt_pre, out, target):
#         bcediceloss = self.bcedice(out, target)
#         gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
#         gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
#         return bcediceloss + gt_loss
    






# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, target):
#         ## 该类只是多了一个logit过程，nn.BCELoss只需多一个nn.Sigmoid()得到的结果和nn.BCEWithLogitsLoss是一致的？？？？？
#         ## 可以直接将输入的值规范到0和1之间，相当于将Sigmoid和BCELoss集成在了一个方法中
#         bce = F.binary_cross_entropy_with_logits(input, target)

#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         target = target.view(num, -1)

#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#         return 0.5 * bce + dice


# def compute_kl_loss(p, q):
#     p_loss = F.kl_div(F.log_softmax(p, dim=-1),
#                       F.softmax(q, dim=-1), reduction='none')
#     q_loss = F.kl_div(F.log_softmax(q, dim=-1),
#                       F.softmax(p, dim=-1), reduction='none')

#     p_loss = p_loss.mean()
#     q_loss = q_loss.mean()

#     loss = (p_loss + q_loss) / 2
#     return loss