from collections import namedtuple

import torch as th
import numpy as np

from torch import nn


def predict_outputs(model, X, y, num_classes, ensemble_configurations):
    outputs_model = model((None, X, y),
              single_eval_pos=len(y))[:, :, 0:num_classes]

    reshuffled_outputs = []
    for i, ensemble_configuration in enumerate(ensemble_configurations):
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration
        output_ = outputs_model[:, i:i+1, :]
        output_ = th.cat([output_[..., class_shift_configuration:],output_[..., :class_shift_configuration]],dim=-1)
        reshuffled_outputs.append(output_.squeeze(1))

    merged_output = th.mean(th.stack(reshuffled_outputs), dim=0)
    return merged_output


class EEGCosNetLinearClf(nn.Module):
    def __init__(self, cos_sim_net, linear):
        super().__init__()
        self.cos_sim_net = cos_sim_net
        self.linear = linear

    def forward(self, x, train_x, train_y):
        # ignore train_x, train_y
        out_per_wave = self.cos_sim_net(
            x, norm_inputs=None, return_intermediate_vals=True).out_per_wave
        out = self.linear(out_per_wave)
        return out


class EEGCosNetTabPFN(nn.Module):
    def __init__(self, cos_sim_net, tabpfn_model, n_classes, ensemble_configurations):
        super().__init__()
        self.cos_sim_net = cos_sim_net
        self.tabpfn_model = tabpfn_model
        self.n_classes = n_classes
        self.ensemble_configurations = ensemble_configurations

    def forward(self, x, train_x, train_y):
        overall_x = th.cat((train_x, x))
        out_per_wave = self.cos_sim_net(
            overall_x, norm_inputs=None, return_intermediate_vals=True).out_per_wave
        X_padded = th.cat((out_per_wave,
                           th.zeros(
                               out_per_wave.shape[0],
                               100 - out_per_wave.shape[1],
                               device=out_per_wave.device)), dim=1).unsqueeze(1)
        outs_tabpfn = predict_outputs(
            self.tabpfn_model,
            X_padded, train_y.float().unsqueeze(1),
            self.n_classes, self.ensemble_configurations)
        return outs_tabpfn


def f_cos_sim_net(x, spat_weights, temp_weights, final_weights, final_biases):
    spat_out = th.einsum('bit,oi->bot', x, spat_weights)
    # Apparently group conv still so slow better to just do regular full conv from start input
    merged_weight = temp_weights.unsqueeze(1) * spat_weights.unsqueeze(2)
    conved = th.nn.functional.conv1d(x, merged_weight)
    norm_weights = th.sqrt(th.square(temp_weights).sum(dim=(1)))
    norm_spat_out = th.sqrt(th.nn.functional.avg_pool1d(
        th.square(spat_out), temp_weights.shape[1], 1) * temp_weights.shape[1])

    denominator = norm_weights.view(1, -1, 1) * norm_spat_out
    cos_sims = conved / denominator

    cos_sims = cos_sims * np.sqrt(temp_weights.shape[1])
    abs_cos_sims = th.abs(cos_sims)
    mean_cos_sims = th.mean(abs_cos_sims, dim=(2))
    out_per_wave = mean_cos_sims * final_weights.unsqueeze(0) + final_biases.unsqueeze(0)
    return out_per_wave


class CosSimSpatTempConvNet(nn.Module):
    def __init__(self, n_chans, n_filters, kernel_length):
        super().__init__()
        self.spat_temp_cosconv = CosSimSpatTempConv(n_chans, n_filters, kernel_length)

        self.weight = nn.Parameter(th.zeros(n_filters))
        self.bias = nn.Parameter(th.zeros(n_filters))

    def forward(self, x, norm_inputs=None, return_intermediate_vals=False):
        cos_sims = self.spat_temp_cosconv(x, norm_inputs=norm_inputs)
        abs_cos_sims = th.abs(cos_sims)
        mean_cos_sims = th.mean(abs_cos_sims, dim=(2))
        out_per_wave = mean_cos_sims * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        summed_out = out_per_wave.sum(dim=1)
        if return_intermediate_vals:
            return namedtuple(
                "CosSimResult", ["mean_cos_sims", "out_per_wave", "summed_out"]
            )(
                mean_cos_sims,
                out_per_wave,
                summed_out,
            )
        else:
            return summed_out


class CosSimSpatTempConv(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size):
        conv_weights = th.zeros(
            n_out_channels, kernel_size, requires_grad=True, device="cuda"
        )
        conv_weights.data[:] = th.randn_like(conv_weights) * 0.2
        spat_weights = th.zeros(
            n_out_channels, n_in_channels, requires_grad=True, device="cuda"
        )
        spat_weights.data[:] = th.randn_like(spat_weights) * 0.2
        super().__init__()
        self.conv_weights = nn.Parameter(conv_weights)
        self.spat_weights = nn.Parameter(spat_weights)

    def compute_merged_weight(self):
        return self.conv_weights.unsqueeze(1) * self.spat_weights.unsqueeze(2)

    def forward(self, x, norm_inputs=None):
        merged_weight = self.compute_merged_weight()
        conved = th.nn.functional.conv1d(x, merged_weight)
        norm_weights = th.sqrt(th.square(merged_weight).sum(dim=(1, 2)))
        if norm_inputs is None:
            norm_inputs = th.sqrt(
                th.nn.functional.avg_pool1d(
                    th.square(x), self.conv_weights.shape[1], 1
                ).sum(dim=1)
                * self.conv_weights.shape[1]
            )

        denominator = norm_weights.view(1, -1, 1) * norm_inputs.unsqueeze(1)
        cos_sims = conved / denominator

        return cos_sims * np.sqrt(np.prod(merged_weight.shape[1:]))


class LinCos(nn.Module):
    def __init__(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data)
        self.bias = nn.Parameter(linear.bias.data)

    def forward(self, x):
        cos_sim = th.nn.functional.cosine_similarity(
            x.unsqueeze(2), self.weight.swapdims(0, 1).unsqueeze(0), dim=1
        )
        return cos_sim * np.sqrt(self.weight.shape[1]) + self.bias.unsqueeze(0)


class SimpleActNorm(nn.Module):
    # More like regular linear layer,  multiply and then add bias
    def __init__(self, in_channel, initialized=False):
        super().__init__()
        self.bias = nn.Parameter(th.zeros(in_channel))
        self.scale = nn.Parameter(th.ones(in_channel))

    def forward(self, x):
        bias = self.bias.unsqueeze(0)
        scale = self.scale.unsqueeze(0)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        y = scale * x + bias
        return y


class LinearSpatCosSimTempConvNet(nn.Module):
    def __init__(self, n_chans, n_filters, kernel_length):
        super().__init__()
        self.spat_temp_cosconv = LinearSpatCosSimTempConv(n_chans, n_filters, kernel_length)

        self.weight = nn.Parameter(th.zeros(n_filters))
        self.bias = nn.Parameter(th.zeros(n_filters))

    def forward(self, x, norm_inputs=None, return_intermediate_vals=False):
        cos_sims = self.spat_temp_cosconv(x, norm_inputs=norm_inputs)
        abs_cos_sims = th.abs(cos_sims)
        mean_cos_sims = th.mean(abs_cos_sims, dim=(2))
        out_per_wave = mean_cos_sims * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        summed_out = out_per_wave.sum(dim=1)
        if return_intermediate_vals:
            return namedtuple(
                "CosSimResult", ["mean_cos_sims", "out_per_wave", "summed_out"]
            )(
                mean_cos_sims,
                out_per_wave,
                summed_out,
            )
        else:
            return summed_out


class LinearSpatCosSimTempConv(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size):
        conv_weights = th.zeros(n_out_channels, kernel_size, requires_grad=True, device='cuda')
        conv_weights.data[:] = th.randn_like(conv_weights) * 0.2
        spat_weights = th.zeros(n_out_channels, n_in_channels, requires_grad=True, device='cuda')
        spat_weights.data[:] = th.randn_like(spat_weights) * 0.2
        super().__init__()
        self.conv_weights = nn.Parameter(conv_weights)
        self.spat_weights = nn.Parameter(spat_weights)


    def forward(self, x, norm_inputs=None):
        assert norm_inputs is None
        # Needed to compute norms
        spat_out = th.einsum('bit,oi->bot', x, self.spat_weights)
        # Apparently group conv still so slow better to just do regular full conv from start input
        merged_weight = self.conv_weights.unsqueeze(1) * self.spat_weights.unsqueeze(2)
        conved = th.nn.functional.conv1d(x, merged_weight)
        # this group conv seemed way slower
        #conved = th.nn.functional.conv1d(
        #    spat_out, self.conv_weights.unsqueeze(1), groups=len(self.conv_weights))
        norm_weights = th.sqrt(th.square(self.conv_weights).sum(dim=(1)))
        norm_spat_out = th.sqrt(th.nn.functional.avg_pool1d(
            th.square(spat_out), self.conv_weights.shape[1], 1) * self.conv_weights.shape[1])

        denominator = norm_weights.view(1, -1, 1) * norm_spat_out
        cos_sims = conved / denominator

        return cos_sims * np.sqrt(self.conv_weights.shape[1])