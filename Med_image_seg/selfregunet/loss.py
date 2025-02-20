import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class KDloss(nn.Module):

    def __init__(self,lambda_x):
        super(KDloss,self).__init__()
        self.lambda_x = lambda_x

    def inter_fd(self, f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        
        idx_s = random.sample(range(s_C),min(s_C,t_C))
        idx_t = random.sample(range(t_C),min(s_C,t_C))
        # print(len(idx_s), len(idx_t))   ## 64 64

        # inter_fd_loss = F.mse_loss(f_s[:, 0:min(s_C,t_C), :, :], f_t[:, 0:min(s_C,t_C), :, :].detach())
        # print('fs',f_s[:, idx_s, :, :].size())    
        # print('ft',f_t[:, idx_t, :, :].size())    

        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        # print('inter_fd_loss:',inter_fd_loss)
        return inter_fd_loss

    
    def intra_fd(self,f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
        return intra_fd_loss
    
    def forward(self,feature,feature_decoder,final_up):
        # f1 = feature[0][-1] # 
        # f2 = feature[1][-1]
        # f3 = feature[2][-1]
        # f4 = feature[3][-1] # lower feature 

        f1_0 = feature[0] # 
        f2_0 = feature[1]
        f3_0 = feature[2]
        f4_0 = feature[3] # lower feature 

        # f1_d = feature_decoder[0][-1] # 14 x 14
        # f2_d = feature_decoder[1][-1] # 28 x 28
        # f3_d = feature_decoder[2][-1] # 56 x 56

        f1_d_0 = feature_decoder[0] # 14 x 14
        ## f1_d_0 : torch.Size([2, 256, 28, 28])
        f2_d_0 = feature_decoder[1] # 28 x 28
        f3_d_0 = feature_decoder[2] # 56 x 56

        #print(f3_d.shape)

        final_layer = final_up
        # print(final_layer.shape)  ## torch.Size([2, 64, 224, 224])


        # loss =  (self.intra_fd(f1)+self.intra_fd(f2)+self.intra_fd(f3)+self.intra_fd(f4))/4
        loss = (self.intra_fd(f1_0)+self.intra_fd(f2_0)+self.intra_fd(f3_0)+self.intra_fd(f4_0))/4
        # print('loss:',loss)  ##loss: tensor(0.7073, device='cuda:0', grad_fn=<DivBackward0>)
        loss += (self.intra_fd(f1_d_0)+self.intra_fd(f2_d_0)+self.intra_fd(f3_d_0))/3
        # print('loss:',loss)    ## loss: tensor(1.3790, device='cuda:0', grad_fn=<AddBackward0>)
        # loss += (self.intra_fd(f1_d)+self.intra_fd(f2_d)+self.intra_fd(f3_d))/3

        # loss1 = self.inter_fd(f1_d_0,final_layer)
        # print('loss1:',loss1)

        
        loss += (self.inter_fd(f1_d_0,final_layer)+self.inter_fd(f2_d_0,final_layer)+self.inter_fd(f3_d_0,final_layer)
                   +self.inter_fd(f1_0,final_layer)+self.inter_fd(f2_0,final_layer)+self.inter_fd(f3_0,final_layer)+self.inter_fd(f4_0,final_layer))/7

        
        
        loss = loss * self.lambda_x
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            # print(self.n_classes)  # 1
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            # print('temp:', temp_prob.size())  # torch.Size([2, 1, 224, 224])
            # tensor_list.append(temp_prob.unsqueeze(1))
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        # print('output.size:', output_tensor.size())
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        # print('target.size:', target.size())   ## target.size: torch.Size([2, 1, 1, 224, 224])
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
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