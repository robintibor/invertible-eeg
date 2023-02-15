from torch import nn


class WrapClfAsGen(nn.Module):
    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def forward(self, x, fixed=None):
        lps = self.clf(x)
        fake_z = x
        if fixed is not None and (not fixed.get('sum_dims', True)):
            # add fake dims dimension
            lps = lps.unsqueeze(-1)
        return fake_z, lps
