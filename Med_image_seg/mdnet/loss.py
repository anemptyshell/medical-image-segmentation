import torch.nn as nn
import torch



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
    


