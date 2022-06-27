import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Meter:

    def __init__(self):
        self.list = []

    def update(self, item):
        self.list.append(item)

    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None

    def confidence_interval(self):
        if len(self.list) == 0:
            return None
        std = torch.tensor(self.list).std()
        ci = std * 1.96 / math.sqrt(len(self.list))
        return ci

    def avg_and_confidence_interval(self):
        return self.avg(), self.confidence_interval()


def Focal_Loss(preds, labels):
    """
    preds:softmax输出结果
    labels:真实值
    """
    eps = 1e-7
    y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)

    target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

    ce = -1 * torch.log(y_pred + eps) * target
    floss = torch.pow((1 - y_pred), 2) * ce
    floss = torch.mul(floss, 0.25)
    floss = torch.sum(floss, dim=1)
    return torch.mean(floss)


    # def forward(self, preds, labels,self.weight=0.25):
    #     """
    #     preds:softmax输出结果
    #     labels:真实值
    #     """
    #     eps = 1e-7
    #     y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)
    #
    #     target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)
    #
    #     ce = -1 * torch.log(y_pred + eps) * target
    #     floss = torch.pow((1 - y_pred), 2) * ce
    #     floss = torch.mul(floss, self.weight)
    #     floss = torch.sum(floss, dim=1)
    #     return torch.mean(floss)
# class Focal_Loss(nn.Module):
#
#     def __init__(self,
#                  alpha=0.25,
#                  gamma=2,
#                  reduction='mean',):
#         super(Focal_Loss(), self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.crit = nn.BCEWithLogitsLoss(reduction='none')
#
#     def forward(self, logits, label):
#
#         probs = torch.sigmoid(logits)
#         coeff = torch.abs(label - probs).pow(self.gamma).neg()
#         log_probs = torch.where(logits >= 0,
#                 F.softplus(logits, -1, 50),
#                 logits - F.softplus(logits, 1, 50))
#         log_1_probs = torch.where(logits >= 0,
#                 -logits + F.softplus(logits, -1, 50),
#                 -F.softplus(logits, 1, 50))
#         loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
#         loss = loss * coeff
#
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         if self.reduction == 'sum':
#             loss = loss.sum()
#         return loss

