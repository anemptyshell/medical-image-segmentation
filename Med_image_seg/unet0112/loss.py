
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim  # 需要安装：pip install pytorch-msssim


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


class AdaptiveBceDiceLoss(nn.Module):
    def __init__(self, ssim_threshold=0.8, bce_weight=0.5, dice_weight=1.0):
        """
        ssim_threshold: SSIM阈值，低于此值的样本被认为是困难样本
        bce_weight: BCE损失权重
        dice_weight: Dice损失权重
        """
        super().__init__()
        self.ssim_threshold = ssim_threshold
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, preds, targets):
        """
        preds: [bs, 1, H, W] - logits (未经过sigmoid)
        targets: [bs, 1, H, W] 或 [bs, H, W]
        返回：筛选后样本的平均损失
        """
        batch_size = preds.size(0)
        
        # 1. 确保targets维度正确
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # [bs, H, W] -> [bs, 1, H, W]
        
        # 2. 计算每个样本的SSIM
        # 需要将preds转换为概率值计算SSIM
        preds_prob = torch.sigmoid(preds)
        
        # 计算每个样本的SSIM，返回形状为 [bs]
        ssim_values = torch.zeros(batch_size, device=preds.device)
        for i in range(batch_size):
            # ssim需要[1, C, H, W]格式
            ssim_val = ssim(
                preds_prob[i:i+1],  # 保持批次维度
                targets[i:i+1],
                data_range=1.0,  # 数据范围是0-1
                size_average=True  # 返回标量
            )
            ssim_values[i] = ssim_val
        
        # 3. 创建掩码：SSIM < threshold 的样本保留
        mask = ssim_values < self.ssim_threshold  # [bs], bool类型
        num_selected = mask.sum().item()
        
        if num_selected == 0:
            # 如果没有困难样本，可以返回一个很小的损失或者返回0
            print(f"警告：所有样本SSIM >= {self.ssim_threshold}，跳过梯度更新")
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        
        # 4. 计算每个样本的损失（使用reduction='none'）
        # BCE部分（每个样本的平均损失）
        bce_per_sample = F.binary_cross_entropy_with_logits(
            preds, targets, reduction='none'
        )  # [bs, 1, H, W]
        bce_per_sample = bce_per_sample.mean(dim=(1, 2, 3))  # [bs]
        
        # Dice部分（每个样本的损失）
        dice_per_sample = self._dice_loss_per_sample(preds, targets)  # [bs]
        
        # 组合损失
        loss_per_sample = self.bce_weight * bce_per_sample + self.dice_weight * dice_per_sample  # [bs]
        
        # 5. 只保留困难样本的损失
        selected_losses = loss_per_sample[mask]  # [num_selected]
        
        # 6. 返回困难样本的平均损失
        final_loss = selected_losses.mean()
        
        # 可选：记录信息
        self._log_stats(ssim_values, mask, final_loss)
        
        return final_loss
    
    def _dice_loss_per_sample(self, preds, targets):
        """计算每个样本的Dice损失"""
        preds_prob = torch.sigmoid(preds)
        batch_size = preds.size(0)
        
        preds_flat = preds_prob.flatten(start_dim=2)  # [bs, 1, H*W]
        targets_flat = targets.flatten(start_dim=2)   # [bs, 1, H*W]
        
        intersection = (preds_flat * targets_flat).sum(dim=2)  # [bs, 1]
        union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2)  # [bs, 1]
        
        dice_coeff = (2. * intersection + 1e-5) / (union + 1e-5)  # [bs, 1]
        dice_loss_per_sample = 1 - dice_coeff.squeeze(1)  # [bs]
        
        return dice_loss_per_sample
    
    def _log_stats(self, ssim_values, mask, final_loss):
        """记录统计信息"""
        if not hasattr(self, 'stats'):
            self.stats = {
                'ssim_mean': [],
                'selected_ratio': [],
                'loss': []
            }
        
        self.stats['ssim_mean'].append(ssim_values.mean().item())
        self.stats['selected_ratio'].append(mask.float().mean().item())
        self.stats['loss'].append(final_loss.item())