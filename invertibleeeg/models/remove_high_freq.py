from torch import nn
import torch  as th
import numpy as np


class RemoveHighFreq(nn.Module):
    def __init__(self, high_cut_hz):
        super().__init__()
        self.high_cut_hz = high_cut_hz

    def forward(self, x, fixed=None):
        x_ffted = th.fft.rfft(x)
        i_stop = np.searchsorted(np.fft.rfftfreq(x.shape[2], 1 / 64.0), self.high_cut_hz)
        x_ffted[:, :, i_stop + 1:] = 0
        x_no_high = th.fft.irfft(x_ffted)
        x_with_noise = x_no_high + th.randn_like(x_no_high) * (5e-3)
        return x_with_noise, 0

    def invert(self, z, fixed=None):
        return z, 0
