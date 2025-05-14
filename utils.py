import torch
import itertools
import numpy as np


class ChainableFn:
    name = None
    def __init__(self, prev=None):
        self.prev = prev
        if prev is not None:
            self.name = f"{self.name}({prev.name})"
    
    def __call__(self, *args, **kwargs):
        if self.prev is not None:
            data = self.prev(*args, **kwargs)
            return self.fn(data, **kwargs)
        return self.fn(*args, **kwargs)

    def __str__(self):
        return self.name


def filter_topk(
    tensor: torch.tensor,
    k: float,
    return_mask: bool = False
) -> torch.tensor:
    out = tensor.clone()
    flat = out.flatten()

    order = flat.abs().argsort(descending=True)
    n = int(k * flat.size(0))

    # out is mutable; flat accesses its memory
    flat[order[n:]] = 0
    if return_mask:
        flat[order[:n]] = 1
    
    return out


def permute(x, perm_0, perm_1=None):
    if perm_1 is None:
        perm_1 = perm_0
    return x[perm_0][:, perm_1]


def brute_force_directionality(
    adj_mat: np.ndarray,
    inp_size: int
):
    n = len(adj_mat)
    arr = np.abs(adj_mat)
    header = list(range(inp_size))
    
    min_cost = np.inf
    min_perm = None

    for _p in itertools.permutations(range(inp_size, n)):
        perm = header + list(_p)
        mat = permute(arr, perm)
        cost = np.sum(np.tril(mat, -1))

        if cost < min_cost:
            min_cost = cost
            min_perm = perm

    directionality = 1 - min_cost / np.sum(arr)

    return float(directionality), min_perm
