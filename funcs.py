import torch
from torch.autograd import Function

from utils import filter_topk


class _StraightThrough(Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        weights, k = inputs
        # ctx.save_for_backward(weights, output)

    @staticmethod
    def backward(ctx, grad_output):
        # weights, output = ctx.saved_tensors
        # straight-through estimation
        return grad_output, None

class TopKMask(_StraightThrough):
    @staticmethod
    def forward(weights, k):
        return filter_topk(weights, k, return_mask=True)


def topk_mask(weights, k):
    return TopKMask.apply(weights, k)
