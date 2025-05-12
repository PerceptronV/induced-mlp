import torch


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
