import torch.nn as nn
import torch
import torch.nn.functional as F



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
 
    def forward(self, x, target):
        # 计算类别数量
        n_classes = x.size(1)
        # 生成目标张量
        target = target.unsqueeze(1)
        # 生成标签张量
        one_hot = torch.zeros_like(x)
        one_hot.fill_(self.smoothing / (n_classes - 1))
        one_hot.scatter_(1, target, self.confidence)
        # 计算交叉熵损失
        log_prb = nn.functional.log_softmax(x, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算真实类别的概率
        pt = torch.exp(-ce_loss)

        # 计算 Focal Loss
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        if self.reduction == 'mean':
            return focal_loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss, ce_loss

class Weighted_Focal_Loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)

        # 计算真实类别的概率
        pt = torch.exp(-ce_loss)

        # 计算 Focal Loss
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        if self.reduction == 'mean':
            return focal_loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss, ce_loss
class HardSampleLoss(nn.Module):

    def __init__(self, threshold, zero_partial=True, **kwargs):
        super(HardSampleLoss, self).__init__()
        self.threshold = threshold
        self.zero_partial = zero_partial


    def forward(self, inputs, targets):


        prob = torch.softmax(inputs, dim=1)
        weights = torch.ones_like(targets, dtype=torch.float32)

        correct_predictions = prob.argmax(dim=1) == targets
        correct_high_confidence = (prob.max(dim=1).values > self.threshold) & correct_predictions
        incorrect_predictions = ~correct_predictions


        weight_type = self.weight_type if hasattr(self, 'weight_type') else None

        if weight_type == 'exp':
            if self.zero_partial:
                weights[correct_high_confidence] = 0.0
            else:
                weights[correct_high_confidence] = torch.exp(1 - inputs.max(dim=1).values[correct_high_confidence]) / 2.0
            weights[~correct_high_confidence & ~incorrect_predictions] = torch.exp(1 - inputs.max(dim=1).values[~correct_high_confidence & ~incorrect_predictions])
            weights[incorrect_predictions] = torch.exp(inputs.max(dim=1).values[incorrect_predictions])

        elif weight_type == 'l2':
            if self.zero_partial:
                weights[correct_high_confidence] = 0.0
            else:
                weights[correct_high_confidence] = (1 - inputs.max(dim=1).values[correct_high_confidence]) ** 2 / 2.0
            weights[~correct_high_confidence & ~incorrect_predictions] = (1- inputs.max(dim=1).values[~correct_high_confidence & ~incorrect_predictions]) ** 2
            weights[incorrect_predictions] = (inputs.max(dim=1).values[incorrect_predictions]) ** 2

        elif weight_type == 'l1':
            if self.zero_partial:
                weights[correct_high_confidence] = 0.0
            else:
                weights[correct_high_confidence] = 1 - inputs.max(dim=1).values[correct_high_confidence] / 2.0
            weights[~correct_high_confidence & ~incorrect_predictions] = 1 - inputs.max(dim=1).values[~correct_high_confidence & ~incorrect_predictions]
            weights[incorrect_predictions] = inputs.max(dim=1).values[incorrect_predictions]


        loss = F.cross_entropy(inputs, targets, reduction='none')
        weighted_loss = loss * weights
        return weighted_loss.mean()

def softmax(x):
    """计算softmax函数值"""
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(dim=1, keepdim=True)

if __name__ == '__main__':
    loss_function =HardSampleLoss(threshold=0.7)
    predictions = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])
    targets = torch.tensor([0, 1, 0])  # 真实标签
    loss = loss_function(predictions, targets)
    print(loss.item())  # 打印损失
