import math
import torch

from utils import ChainableFn, filter_topk


class NoPrune(ChainableFn):
    name = "NoPrune"
    def fn(self, arr, **ctx):
        return arr

class RandomPrune(ChainableFn):
    name = "RandomPrune"
    def __init__(self, prev=None, p=0., persistent=True):
        super().__init__(prev)
        self.p = float(p)
        self.persistent = persistent
        self.initialised = False

    def fn(self, arr, **ctx):
        if not self.persistent or not self.initialised:
            self.initialised = True
            self.mask = torch.rand(arr.shape) < self.p
        return arr.masked_fill(self.mask.to(arr.device), 0)

class ThresholdPrune(ChainableFn):
    name = "ThresholdPrune"
    def __init__(self, prev=None, threshold=0.):
        super().__init__(prev)
        self.threshold = float(threshold)

    def fn(self, arr, **ctx):
        idx = arr.abs() < self.threshold
        return arr.masked_fill(idx, 0)

class TopKPrune(ChainableFn):
    name = "TopKPrune"
    def __init__(self, prev=None, k=1.):
        super().__init__(prev)
        self.k = float(k)

    def fn(self, arr, **ctx):
        return filter_topk(arr, self.k, return_mask=False)

class DynamicTopK(ChainableFn):
    name = "DynamicTopK"
    def __init__(self, prev=None, k=1.):
        super().__init__(prev)
        self.eqn = lambda x: 1 - (1-k) * (math.sin(math.pi/2 * x)** 4)

    def fn(self, arr, **ctx):
        return filter_topk(arr, self.eqn(ctx['progress']), return_mask=False)
    
class TriuDamp(ChainableFn):
    name = "TriuDamp"
    def __init__(self, prev=None, diagonal=0, f=0.9):
        super().__init__(prev)
        self.diag = diagonal
        self.f = f
    
    def fn(self, arr, **ctx):
        return arr - arr.tril(self.diag-1) * self.f

class DynamicTriuDamp(ChainableFn):
    name = "DynamicTriuDamp"
    def __init__(self, prev=None, diagonal=0, f=0.9):
        super().__init__(prev)
        self.diag = diagonal
        self.f = f
        self.eqn = lambda x: f * (math.sin(math.pi/2 * x)** 4)
    
    def fn(self, arr, **ctx):
        return arr - arr.tril(self.diag-1) * self.eqn(ctx['progress'])

class TriuPrune(ChainableFn):
    name = "TriuPrune"
    def __init__(self, prev=None, diagonal=0):
        super().__init__(prev)
        self.diag = diagonal
    
    def fn(self, arr, **ctx):
        return arr.triu(self.diag)


class PruneEnsemble:
    def __init__(self, config, requires_grad=False):
        self.cfg = config
        self.requires_grad = requires_grad
    
    def prune(self, module, **ctx):
        if not self.cfg:
            return module
        
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in self.cfg:
                    param.copy_(self.cfg[name](param, **ctx))
        return module

no_pruning = PruneEnsemble(None)
