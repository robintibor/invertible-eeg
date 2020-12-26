from torch import nn
import torch as th

class MultiplyFactors(nn.Module):
    def __init__(self, n_chans):
        super().__init__()
        self.factors = nn.Parameter(th.zeros(n_chans, dtype=th.float32))

    def forward(self, x):
        factors = self.factors.unsqueeze(0)
        assert factors.shape[1] == x.shape[1]
        while factors.ndim < x.ndim:
            factors = factors.unsqueeze(-1)
        return factors * x
