import torch
from torch import nn

from utils import ChainableFn


class RandomUniform(ChainableFn):
    name = 'RandomUniform'
    def fn(self, arr, **ctx):
        return torch.rand_like(arr)

class RandomNormal(ChainableFn):
    name = 'RandomNormal'
    def __init__(self, prev, mean=0., std=1.):
        super().__init__(prev)
        self.mean = float(mean)
        self.std = float(std)
    
    def fn(self, arr, **ctx):
        return torch.normal(mean=self.mean,
                            std=self.std,
                            size=arr.shape)

class Zeros(ChainableFn):
    name = 'Zeros'
    def fn(self, arr, **ctx):
        return torch.zeros_like(arr)

class Triu(ChainableFn):
    name = 'Triu'
    def __init__(self, prev, diagonal=1):
        super().__init__(prev)
        self.diag = diagonal

    def fn(self, arr, **ctx):
        return arr.triu(self.diag)

class _Size(ChainableFn):
    name = 'Size'
    def fn(self, *size, **ctx):
        return torch.zeros(size)

Size = _Size()
Like = Zeros()
Like.name = 'Like'
