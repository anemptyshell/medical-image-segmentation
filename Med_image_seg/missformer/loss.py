import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

# ce_loss = CrossEntropyLoss().cuda()
## CE-Net 中使用的是 bcedice_loss

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
        # return 0.5 * bce + dice  
        return 0.4 * bce + 0.6 * dice  




 
    
# class DiceLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes

#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss

#     def forward(self, inputs, target, weight=None, softmax=False):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             # weight = [1] * self.n_classes
#             weight = [1]
#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             # dice = self._dice_loss(inputs, target)
#             # class_wise_dice.append(1.0 - dice.item())
#             class_wise_dice.append(1.0 - dice)
#             loss += dice * weight[i]
#         # return loss / self.n_classes   
#         return loss 
 
    
# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()

#     def forward(self, pred, target):
#         smooth = 1e-5

#         size = pred.size(0)

#         pred_ = pred.view(size, -1)
#         target_ = target.view(size, -1)
#         intersection = pred_ * target_
#         dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
#         dice_loss = 1 - dice_score.sum()/size

#         return dice_loss