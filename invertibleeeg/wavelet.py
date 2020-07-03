import torch as th
from torch import nn


def wavelet_1d_forward(x):
    means = nn.functional.avg_pool1d(x, 2, stride=2)
    diffs = x[:, :, ::2] - x[:, :, 1::2]
    z = th.cat((means, diffs), dim=1)
    return z

def wavelet_1d_invert(z):
    means, diffs = th.chunk(z, 2, dim=1)
    # x1+x2
    x = means.repeat_interleave(2, dim=2) * 2
    x[:,:,::2] = x[:,:,::2] + diffs
    x[:,:,1::2] = x[:,:,1::2] - diffs
    x = x / 2
    return x


class Haar1dWavelet(nn.Module):
    def __init__(self, chunk_chans_first=False, ):
        super().__init__()
        self.chunk_chans_first = chunk_chans_first

    def forward(self, x, fixed=None):
        if self.chunk_chans_first:
            xs = th.chunk(x, 2, dim=1)
        else:
            xs = [x]
        zs = [wavelet_1d_forward(a_x) for a_x in xs]
        z = th.cat(zs, dim=1)
        return z, 0

    def invert(self, z, fixed=None):
        if self.chunk_chans_first:
            zs = th.chunk(z, 2, dim=1)
        else:
            zs = [z]

        xs = [wavelet_1d_invert(a_z) for a_z in zs]
        x = th.cat(xs, dim=1)
        return x, 0
