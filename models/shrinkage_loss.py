
import torch.nn as nn
import torch

class RegressionShrinkageLoss(nn.Module):
    def __init__(self, a=10.0, c=0.2):
        super(RegressionShrinkageLoss, self).__init__()
        self.a = a
        self.c = c

    def forward(self, predictions, targets):
        l1 = torch.abs(predictions - targets)
        l2 = l1**2
        loss  = l2 / (1 + torch.exp(self.a * (self.c - l1)))

        if len(loss.shape) > 1:
            loss = loss.sum(1)

        return torch.mean(loss, 0)
    
    