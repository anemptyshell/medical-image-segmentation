import torch
import torch.nn as nn
import torch.nn.functional as F

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