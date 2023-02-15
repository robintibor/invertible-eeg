import torch as th
from torch import nn


class GlowLinearClassifier(nn.Module):
    def __init__(self, glow, linear, n_times):
        super().__init__()
        self.glow = glow
        self.linear = linear
        self.n_times = n_times

    def forward(self, x, fixed=None):
        z, _ = self.glow(x)
        # ignore glow lps

        # average features in crop
        zs = nn.AvgPool1d(self.n_times, stride=1)(z)
        # from #examples x # chans x #timecrops
        # to (#examples x #timecrops) x #chans
        zs_flat = zs.permute(0, 2, 1).reshape(-1, zs.shape[1])

        # now linear clf application
        out_flat = self.linear(zs_flat)
        # now to
        # #examples x #timecrops x #chans
        # and average across #timecrops
        lps = out_flat.view(zs.shape[0], zs.shape[2], out_flat.shape[1]).mean(dim=1)

        if fixed is not None and (not fixed.get("sum_dims", True)):
            # add fake dims dimension
            lps = lps.unsqueeze(-1)
        return z, lps
