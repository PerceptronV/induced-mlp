from torch import nn
import torch.nn.functional as F


class MSELoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, x, y, **ctx):
        return F.mse_loss(x, y, reduction=self.reduction)

class NormedMSELoss:
    def __init__(self, beta=0.5, reduction='mean'):
        self.reduction = reduction
        self.beta = beta
    
    def __call__(self, x, y, **ctx):
        return F.mse_loss(x, y, reduction=self.reduction) + self.beta * ctx['model'].norm()
