import torch as th
from torch import nn

from invertible.sequential import InvertibleSequential
from invertible.identity import Identity


class CroppedGlow(nn.Module):
    def __init__(self, glow, n_times):
        super().__init__()
        self.dist = glow.module
        node_until_flat = glow.prev[0]
        if node_until_flat.prev is not None:
            node_before_flat = node_until_flat.prev[0]
        else:
            node_before_flat = Identity()
        self.flat_module = node_until_flat.module.sequential[-1]
        self.mods_until_flat = InvertibleSequential(
            node_before_flat, *node_until_flat.module.sequential[:-1]
        )
        self.n_times = n_times
        if hasattr(glow, 'alphas'):
            self.alphas = glow.alphas

    def forward(self, x, fixed=None):
        z, lps_before_dist = self.mods_until_flat(x, fixed=fixed)
        all_lps_dist = [
            self.dist(
                self.flat_module(
                    z[:, :, i_start : i_start + self.n_times].contiguous()
                )[0],
                fixed=fixed,
            )[1]
            for i_start in range(0, z.shape[2] - self.n_times + 1)
        ]
        lps_dist = th.stack(all_lps_dist).mean(dim=0)
        if hasattr(lps_before_dist, "shape") and len(lps_before_dist.shape) != 0:
            assert len(lps_before_dist.shape) == 1
            # expand dims to match number of dims
            lps_before_dist = lps_before_dist.view(
                -1, *((1,) * (len(lps_dist.shape) - len(lps_before_dist.shape)))
            )
        return z, lps_dist + lps_before_dist
