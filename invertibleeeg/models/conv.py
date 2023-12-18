import math
from torch import nn
import torch as th


class SeparateTemporalChannelConvFixed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length, **conv_args):
        temporal_weights = nn.Linear(kernel_length, out_channels).weight.detach()
        mix_weights = nn.Linear(
            in_channels,
            out_channels,
        ).weight.detach()
        super().__init__()

        conv_weight = temporal_weights.unsqueeze(1) * mix_weights.unsqueeze(2)
        desired_std = (2 / math.sqrt(in_channels * kernel_length)) / math.sqrt(12)
        factor = desired_std / conv_weight.std()
        temporal_weights.mul_(math.sqrt(factor))
        mix_weights.mul_(math.sqrt(factor))
        self.temporal_weights = nn.Parameter(temporal_weights.detach())
        self.mix_weights = nn.Parameter(mix_weights.detach())
        self.bias = nn.Parameter(th.zeros(out_channels))
        self.conv_args = conv_args

    def forward(self, x):
        conv_weight = self.temporal_weights.unsqueeze(1) * self.mix_weights.unsqueeze(2)
        return nn.functional.conv1d(x, conv_weight, self.bias, **self.conv_args)


class SeparateTemporalChannelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length, **conv_args):
        temporal_weights = nn.Linear(kernel_length, out_channels).weight.detach()
        mix_weights = nn.Linear(
            in_channels,
            out_channels,
        ).weight.detach()
        super().__init__()
        conv_weight = temporal_weights.unsqueeze(1) * mix_weights.unsqueeze(2)
        desired_std = (2 / math.sqrt(in_channels * kernel_length)) / math.sqrt(12)
        factor = desired_std / conv_weight.std()
        conv_weight.data.mul_(factor)
        self.conv_weight = nn.Parameter(conv_weight.detach())
        self.bias = nn.Parameter(th.zeros(out_channels))
        self.conv_args = conv_args

    def forward(self, x):
        return nn.functional.conv1d(x, self.conv_weight, self.bias, **self.conv_args)


class TwoStepSpatialTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length, **conv_args):
        super().__init__()
        # in_channels * 2 just some heuristic
        self.conv_temporal = nn.Conv2d(1, in_channels * 2, kernel_length, **conv_args)
        self.conv_spatial = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=(1, in_channels)
        )

    def forward(self, x):
        # move channels to new empty last dim
        x = x.unsqueeze(-1).permute(0, 3, 2, 1)
        out = self.conv_temporal(x)
        out = self.conv_spatial(out)
        out = out.squeeze(-1)
        assert len(out.shape) == 3
        return out


class TwoStepSpatialTemporalConvFixed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length, padding, **conv_args):
        super().__init__()
        # in_channels * 2 just some heuristic
        self.conv_temporal = nn.Conv2d(
            1,
            in_channels * 2,
            kernel_size=(kernel_length, 1),
            padding=(padding, 0),
            **conv_args
        )
        self.conv_spatial = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=(1, in_channels)
        )

    def forward(self, x):
        # move channels to new empty last dim
        x = x.unsqueeze(-1).permute(0, 3, 2, 1)
        out = self.conv_temporal(x)
        out = self.conv_spatial(out)
        out = out.squeeze(-1)
        assert len(out.shape) == 3
        return out


class TwoStepSpatialTemporalConvMerged(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length, padding, **conv_args):
        super().__init__()
        self.padding = padding
        # in_channels * 2 just some heuristic
        self.conv_temporal_weight = nn.Parameter(
            nn.Conv2d(
                1,
                in_channels * 2,
                (kernel_length, 1),
                padding=(padding, 0),
                **conv_args
            ).weight.detach()
        )
        self.conv_spatial_weight = nn.Parameter(
            nn.Conv2d(
                in_channels * 2, out_channels, kernel_size=(1, in_channels)
            ).weight.detach()
        )
        self.bias = nn.Parameter(th.zeros(out_channels))
        self.conv_args = conv_args

    def forward(self, x):
        merged_weight = self.compute_merged_weight()
        out = th.nn.functional.conv1d(x, merged_weight, bias=self.bias, padding=self.padding, **self.conv_args)
        return out

    def compute_merged_weight(self):
        return (self.conv_temporal_weight * self.conv_spatial_weight.permute(1, 0, 2, 3)).sum(
            dim=0).permute(0, 2, 1)