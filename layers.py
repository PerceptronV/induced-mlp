import torch
import torch.nn as nn
import torch.nn.functional as F

from funcs import topk_mask
from inits import Size, Like, RandomUniform, Zeros, Ones


class CompleteLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        values_init=(RandomUniform, False),
        weights_init=(RandomUniform, True),
        bias_init=(RandomUniform, True),
        scores_init=(Ones, False),
        scores_k=1.,
        activation=F.sigmoid,
        use_bias=False
    ):
        super(CompleteLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.units = input_size + hidden_size + output_size

        self.values = nn.Parameter(
            values_init[0](Size)(1, hidden_size+output_size),
            requires_grad = values_init[1]
        )
        self.weights = nn.Parameter(
            weights_init[0](Size)(hidden_size+output_size, self.units),
            requires_grad = weights_init[1]
        )
        self.scores = nn.Parameter(
            scores_init[0](Like)(self.weights),
            requires_grad = scores_init[1]
        )
        self.scores_k = scores_k

        self.activation = activation
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(
                bias_init[0](Like)(self.values),
                requires_grad = bias_init[1]
            )
    
    def apply(
        self,
        inp,
        its,
        values,
        weights,
        scores,
        bias=None,
    ):
        batch = inp.size(0)
        values = values.expand(batch, -1)   # dim-0 of inp is batch
        if self.use_bias:
            bias = bias.expand(batch, -1)   # dim-0 of inp is batch
        
        scores_mask = topk_mask(scores, self.scores_k)
        w = weights * scores_mask
        
        for _ in range(its):
            x = torch.cat((values, inp), 1) # dim-1 of inp is features
            if self.use_bias:
                values = self.activation(x @ w.t() + bias)
            else:
                values = self.activation(x @ w.t())
        
        return values[:, :self.output_size]
    
    def forward(
        self,
        inp,
        its=2,
        **kwargs
    ):
        defaults = {
            'values': self.values,
            'weights': self.weights,
            'scores': self.scores,
            'its': its
        }
        if self.use_bias:
            defaults['bias'] = self.bias
        return self.apply(inp, **(defaults | kwargs))
    
    def norm(self, include_values=True):
        norm = torch.linalg.matrix_norm(self.weights)
        if include_values:
            norm += torch.linalg.vector_norm(self.values)
        if self.use_bias:
            norm += torch.linalg.vector_norm(self.bias)
        return norm
