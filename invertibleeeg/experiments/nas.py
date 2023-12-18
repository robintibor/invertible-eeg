import functools
import logging
import os.path
import pickle
import time
import traceback
from copy import deepcopy
from functools import partial
from functools import wraps

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import fasteners
import numpy as np
import pandas as pd
import torch as th
from black import format_str, FileMode
from braindecode.util import set_random_seeds
from hyperoptim.concurrent_file_observer import ConcurrentFileStorageObserver
from sacred.experiment import Experiment
from torch import nn

from invertible.actnorm import ActNorm
from invertible.affine import AdditiveCoefs, AffineCoefs
from invertible.affine import AffineModifier
from invertible.amp_phase import AmplitudePhase
from invertible.coupling import CouplingLayer
from invertible.distribution import (
    PerDimWeightedMix,
    NClassIndependentDist,
    MaskedMixDist,
    ClassWeightedGaussianMixture,
    ClassWeightedPerDimGaussianMixture,
    ClassWeightedHierachicalGaussianMixture, PerClassHierarchical,
)
from invertible.expression import Expression
from invertible.graph import Node
from invertible.init import init_all_modules
from invertible.inverse import Inverse
import invertibleeeg.models.conv
from invertible.permute import InvPermute, Shuffle
from invertible.sequential import InvertibleSequential
from invertible.split_merge import ChunkChansIn2, ChansFraction
from invertible.split_merge import EverySecondChan
from invertible.view_as import Flatten2d
from invertibleeeg.datasets import (
    load_train_valid_test_bcic_iv_2a,
    load_train_valid_test_hgd,
)
from invertibleeeg.models.glow import get_splitter
from invertibleeeg.train import train_glow
from ..factors import MultiplyFactors
from ..models.cropped import CroppedGlow
from ..models.glow_linear_clf import GlowLinearClassifier
from ..scheduler import get_cosine_warmup_scheduler

log = logging.getLogger(__name__)


def copy_clean_encoding_dict(encoding):
    clean_encoding = {}
    for key, val in encoding.items():
        if key not in ["node", "optim_params_per_param", "deep4", "linear_clf"]:
            clean_encoding[key] = copy_clean_encoding_val(val)
    return clean_encoding


def copy_clean_encoding_val(val):
    if isinstance(val, dict):
        return copy_clean_encoding_dict(val)
    elif isinstance(val, list):
        clean_list = [copy_clean_encoding_val(elem) for elem in val]
        return clean_list
    else:
        assert val is None or (
            type(val) in [str, np.str_, int, bool, float]
        ), f"Unexpected encoding value type: {type(val)} for value: {val}"
        return val


def get_optim_choices():
    return [
        CSH.CategoricalHyperparameter(
            "lr",
            choices=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        ),
        CSH.CategoricalHyperparameter(
            "weight_decay",
            choices=[0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        ),
    ]


def sample_optim_config_space(seed):
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add_hyperparameters(get_optim_choices())
    config = cs.sample_configuration()
    return config.get_dictionary()


def mutate_encoding_and_model(
    encoding,
    rng,
    blocks,
    n_start_chans,
    max_n_changes,
    other_encoding,
    fixed_lr,
    max_n_downsample,
    splitter_name,
    mutate_optim_params,
    max_n_deletions,
    min_n_downsample,
):
    encoding = deepcopy(encoding)

    if mutate_optim_params:
        optim_lr_choices, optim_wd_choices = get_optim_choices()
        assert optim_lr_choices.name == "lr"
        assert optim_wd_choices.name == "weight_decay"
        for p in encoding["optim_params_per_param"]:
            optim_params = encoding["optim_params_per_param"][p]
            if fixed_lr is None:
                lr_change = rng.randint(-1, 2)
                new_lr_ind = (
                    np.searchsorted(
                        sorted(optim_lr_choices.choices), optim_params["lr"]
                    )
                    + lr_change
                )
                new_lr_ind = max(0, min(new_lr_ind, len(optim_lr_choices.choices) - 1))
                new_lr = sorted(optim_lr_choices.choices)[new_lr_ind]
                optim_params["lr"] = new_lr

            wd_change = rng.randint(-1, 2)
            new_wd_ind = (
                np.searchsorted(
                    sorted(optim_wd_choices.choices), optim_params["weight_decay"]
                )
                + wd_change
            )
            new_wd_ind = max(0, min(new_wd_ind, len(optim_wd_choices.choices) - 1))
            new_wd = sorted(optim_wd_choices.choices)[new_wd_ind]
            optim_params["weight_decay"] = new_wd

    model = encoding["node"]
    flat_node = model.prev[0]
    n_changes = rng.randint(0, max_n_changes + 1)
    final_blocks = []
    for _ in range(n_changes):
        # Determine which blocks are possible, some are only possible at end
        sample_choices = ["new"]
        if len(encoding["net"]) > 0:
            sample_choices.append("own")
        if len(other_encoding["net"]) > 0:
            sample_choices.append("other")
        sample_method = rng.choice(sample_choices)
        if sample_method == "new":
            assert all([blocks[k]["position"] in ["any", "end"] for k in blocks.keys()])
            block_type = rng.choice(list(blocks.keys()), size=1)[0]

            cs = CS.ConfigurationSpace(seed=rng.randint(2**32))
            cs.add_hyperparameters(blocks[block_type]["params"])
            block_params = cs.sample_configuration().get_dictionary()
            # Now sample optim params
            optim_params = sample_optim_config_space(seed=rng.randint(2**32))
            if fixed_lr is not None:
                optim_params["lr"] = fixed_lr
            n_downsample = rng.randint(min_n_downsample or 0, max_n_downsample + 1)
        elif sample_method == "own":
            i_block_to_copy = rng.choice(len(encoding["net"]))
            block_type = encoding["net"][i_block_to_copy]["key"]
            optim_params = encoding["net"][i_block_to_copy]["optim_params"]
            block_params = encoding["net"][i_block_to_copy]["params"]
            n_downsample = encoding["net"][i_block_to_copy]["n_downsample"]
        elif sample_method == "other":
            i_block_to_copy = rng.choice(len(other_encoding["net"]))
            block_type = other_encoding["net"][i_block_to_copy]["key"]
            optim_params = other_encoding["net"][i_block_to_copy]["optim_params"]
            block_params = other_encoding["net"][i_block_to_copy]["params"]
            n_downsample = other_encoding["net"][i_block_to_copy]["n_downsample"]
        else:
            assert False

        if blocks[block_type]["position"] == "any":
            if sample_method in [
                "new",
                "own",
            ]:
                i_insert_before = rng.choice(len(encoding["net"]) + 1)
            else:
                assert sample_method == "other"
                # map  position in other encoding to this encoding
                i_insert_before = int(
                    np.ceil(
                        (i_block_to_copy / len(other_encoding["net"]))
                        * len(encoding["net"])
                    )
                )
        else:
            assert blocks[block_type]["position"] == "end"
            i_insert_before = len(encoding["net"])

        if i_insert_before == len(encoding["net"]):
            # we are at end of blocks
            next_node = flat_node
        else:
            next_node = encoding["net"][i_insert_before]["node"]
        if i_insert_before == 0:
            prev_node = None
        else:
            prev_node = encoding["net"][i_insert_before - 1]["node"]

        n_cur_chans = n_start_chans
        # Could also sture always n_chans as part of encoding... should not change for a given block
        # so could just set when block added
        for i_block in range(i_insert_before):
            n_cur_chans = blocks[encoding["net"][i_block]["key"]]["chans_after"](
                n_cur_chans
            )
        n_cur_chans = int(n_cur_chans * (2**n_downsample))
        block = blocks[block_type]["func"](n_chans=n_cur_chans, **block_params)

        # Add splitter for downsampling
        # invert logdet sign should not really matter
        splitter_seq = InvertibleSequential(
            *[
                # do not chunk chans first for first downsampling
                # otherwise network only sees half of EEG channels
                get_splitter(
                    splitter_name=splitter_name, chunk_chans_first=(i_downsample > 0)
                )
                for i_downsample in range(n_downsample)
            ]
        )
        block = InvertibleSequential(
            splitter_seq, block, Inverse(splitter_seq, invert_logdet_sign=True)
        )

        for p in block.parameters():
            encoding["optim_params_per_param"][p] = optim_params
        cur_node = Node(prev_node, block, remove_prev_node_next=True)
        next_node.change_prev(
            cur_node, notify_prev_nodes=True, remove_prev_node_next=True
        )

        encoding["net"].insert(
            i_insert_before,
            dict(
                key=block_type,
                params=block_params,
                optim_params=optim_params,
                node=cur_node,
                n_downsample=n_downsample,
            ),
        )
        if block_type in ["splitter"]:
            # invert logdet sign should not really matter
            final_blocks.insert(0, Inverse(block, invert_logdet_sign=True))

    flat_node.module.sequential = nn.Sequential(
        *(final_blocks + list(flat_node.module.sequential.children()))
    )

    n_deletions = rng.randint(0, min(max_n_deletions, len(encoding["net"])) + 1)
    for _ in range(n_deletions):
        i_delete = rng.choice(len(encoding["net"]))
        deleted_part = encoding["net"].pop(i_delete)
        deleted_node = deleted_part["node"]
        if i_delete > 0:
            assert len(deleted_node.prev) == 1
            prev_node = deleted_node.prev[0]
        else:
            prev_node = None
        assert len(deleted_node.next) == 1
        next_node = deleted_node.next[0]
        next_node.change_prev(
            prev_node, notify_prev_nodes=True, remove_prev_node_next=True
        )
        for p in deleted_node.module.parameters():
            encoding["optim_params_per_param"].pop(p)

    init_all_modules(model, None)
    return encoding


def to_code_str(net_list, max_line_length=88):
    m_strs = []
    for m in net_list:
        param_string = ",\n".join([f"{k}={v}" for k, v in m["params"].items()])
        m_str = f"{m['key']}({param_string})"
        m_strs.append(m_str)
    code_str = "[" + ", ".join(m_strs) + "]"
    m_str_clean = format_str(code_str, mode=FileMode(line_length=max_line_length))
    return m_str_clean


def ignore_n_chans(func):
    @wraps(func)
    def without_n_chans(n_chans, **kwargs):
        return func(**kwargs)

    return without_n_chans


def coupling_block(
    n_chans,
    hidden_channels,
    kernel_length,
    affine_or_additive,
    scale_fn,
    dropout=0.0,
    swap_dims=False,
    norm=None,
    nonlin=None,
    multiply_by_zeros=True,
    channel_permutation="none",
    fraction_unchanged=0.5,
    eps=0,
    first_conv_class_name=None
):
    if first_conv_class_name is None:
        first_conv_class = nn.Conv1d
    else:
        first_conv_class = {
            "conv1d": nn.Conv1d,
            "separate_temporal_channel_conv_fixed": invertibleeeg.models.conv.SeparateTemporalChannelConvFixed,
            "twostepspatialtemporalconv": invertibleeeg.models.conv.TwoStepSpatialTemporalConv,
            "twostepspatialtemporalconvfixed": invertibleeeg.models.conv.TwoStepSpatialTemporalConvFixed,
            "twostepspatialtemporalconvmerged": invertibleeeg.models.conv.TwoStepSpatialTemporalConvMerged,
        }[first_conv_class_name]
    n_unchanged = int(np.round(n_chans * fraction_unchanged))
    n_changed = n_chans - n_unchanged
    assert nonlin is not None
    nonlin_layer = {
        "square": Expression(th.square),
        "elu": nn.ELU(),
    }[nonlin]
    assert affine_or_additive in ["affine", "additive"]
    if affine_or_additive == "additive":
        CoefClass = AdditiveCoefs
        n_chans_out = n_changed
    else:
        CoefClass = functools.partial(AffineCoefs, splitter=EverySecondChan())
        n_chans_out = n_changed * 2

    assert kernel_length % 2 == 1
    assert norm in [None, "none", "bnorm"]
    if norm in ["none", None]:
        norm_layer = nn.Identity()
    elif norm == "bnorm":
        norm_layer = nn.BatchNorm1d(hidden_channels)
    else:
        raise ValueError(f"Unexpected norm {norm}")

    last_conv = nn.Conv1d(
        hidden_channels,
        n_chans_out,
        kernel_length,
        padding=kernel_length // 2,
    )
    if multiply_by_zeros:
        post_hoc_coefs = MultiplyFactors(n_chans_out)
    else:
        post_hoc_coefs = nn.Identity()
        last_conv.weight.data.zero_()
        last_conv.bias.data.zero_()

    coupling_layer = CouplingLayer(
        ChansFraction(n_unchanged=n_unchanged, swap_dims=swap_dims),
        CoefClass(
            nn.Sequential(
                first_conv_class(
                    n_unchanged,
                    hidden_channels,
                    kernel_length,
                    padding=kernel_length // 2,
                ),
                nonlin_layer,
                nn.Dropout(dropout),
                norm_layer,
                last_conv,
                post_hoc_coefs,
            )
        ),
        AffineModifier(scale_fn, add_first=True, eps=eps),
    )
    if channel_permutation == "linear":
        permuter = InvPermute(n_chans, fixed=False, use_lu=True, init_identity=False)
        coupling_layer = InvertibleSequential(
            permuter, coupling_layer, Inverse(permuter, invert_logdet_sign=True)
        )
    elif channel_permutation == "shuffle":
        shuffler = Shuffle(n_chans)
        coupling_layer = InvertibleSequential(
            shuffler, coupling_layer, Inverse(shuffler, invert_logdet_sign=True)
        )
    else:
        assert channel_permutation == "none"

    return coupling_layer


def get_deep4_coef_net(n_chans, n_out_chans, dropout):
    from braindecode.models import Deep4Net
    from braindecode.models.util import to_dense_prediction_model

    n_classes = n_out_chans
    model = Deep4Net(
        n_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length=2,
        pool_time_stride=2,
        pool_time_length=2,
        filter_time_length=10,
        filter_length_2=9,
        filter_length_3=7,
        filter_length_4=7,
        drop_prob=dropout,
    )
    to_dense_prediction_model(model)
    new_modules = []
    for n, m in model.named_children():
        if n == "softmax":
            continue
        if hasattr(m, "stride"):
            dilation = m.dilation[0]
            kernel_size = m.kernel_size[0]
            dilated_kernel_size = (kernel_size - 1) * dilation + 1
            new_modules.append(nn.ReflectionPad2d((0, 0, dilated_kernel_size - 1, 0)))
        new_modules.append(m)
    new_model = nn.Sequential(*new_modules)
    return new_model


def deep4_coupling_block(
    n_chans,
    affine_or_additive,
    scale_fn,
    dropout=0.0,
    swap_dims=False,
):
    assert affine_or_additive in ["affine", "additive"]
    if affine_or_additive == "additive":
        CoefClass = AdditiveCoefs
        n_chans_out = n_chans // 2
    else:
        CoefClass = functools.partial(AffineCoefs, splitter=EverySecondChan())
        n_chans_out = n_chans
    return CouplingLayer(
        ChunkChansIn2(swap_dims=swap_dims),
        CoefClass(
            nn.Sequential(
                get_deep4_coef_net(n_chans // 2, n_chans_out, dropout=dropout),
                MultiplyFactors(n_chans_out),
            )
        ),
        AffineModifier(scale_fn, add_first=True, eps=0),
    )


def inv_permute(n_chans, use_lu=True):
    permuter = InvPermute(n_chans, fixed=False, use_lu=use_lu, init_identity=True)
    return permuter


def act_norm(n_chans, scale_fn, eps=1e-8):
    return ActNorm(
        n_chans,
        scale_fn,
        eps=eps,
    )


def get_simpleflow_blocks(include_splitter):
    blocks = {
        "coupling_block": {
            "func": coupling_block,
            "params": [
                CSH.CategoricalHyperparameter(
                    "affine_or_additive",
                    choices=["additive"],
                ),
                CSH.CategoricalHyperparameter(
                    "hidden_channels",
                    choices=[8, 16, 32, 64, 128, 256],
                ),
                CSH.CategoricalHyperparameter(
                    "kernel_length",
                    choices=[3, 5, 7, 9, 11, 13, 15],
                ),
                CSH.CategoricalHyperparameter("scale_fn", choices=["square"]),
                CSH.CategoricalHyperparameter("nonlin", choices=["square"]),
                CSH.CategoricalHyperparameter(
                    "dropout",
                    choices=[
                        0,
                    ],
                ),
                CSH.CategoricalHyperparameter(
                    "swap_dims",
                    choices=[True, False],
                ),
                CSH.CategoricalHyperparameter(
                    "norm",
                    choices=[
                        "none",
                    ],
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "permute": {
            "func": inv_permute,
            "params": [],
            "chans_after": lambda x: x,
            "position": "any",
        },
    }
    if include_splitter:
        blocks["splitter"] = {
            "func": ignore_n_chans(get_splitter),
            "params": [
                CSH.CategoricalHyperparameter(
                    "splitter_name", choices=["haar", "subsample"]
                ),
                CSH.CategoricalHyperparameter(
                    "chunk_chans_first", choices=[True, False]
                ),
            ],
            "chans_after": lambda x: x * 2,
            "position": "end",
        }

    return blocks


def get_downsample_anywhere_blocks(included_blocks):
    blocks = {
        "coupling_block": {
            "func": coupling_block,
            "params": [
                CSH.CategoricalHyperparameter(
                    "affine_or_additive",
                    choices=["affine", "additive"],
                ),
                CSH.CategoricalHyperparameter(
                    "hidden_channels",
                    choices=[8, 16, 32, 64, 128, 256],
                ),
                CSH.CategoricalHyperparameter(
                    "kernel_length",
                    choices=[3, 5, 7, 9, 11, 13, 15],
                ),
                CSH.CategoricalHyperparameter(
                    "scale_fn", choices=["twice_sigmoid", "exp"]
                ),
                CSH.CategoricalHyperparameter("nonlin", choices=["elu"]),
                CSH.CategoricalHyperparameter("dropout", choices=[0, 0.2, 0.5]),
                CSH.CategoricalHyperparameter(
                    "swap_dims",
                    choices=[True, False],
                ),
                CSH.CategoricalHyperparameter(
                    "norm",
                    choices=["none", "bnorm"],
                ),
                CSH.CategoricalHyperparameter(
                    "multiply_by_zeros",
                    choices=[True, False],
                ),
                CSH.CategoricalHyperparameter(
                    "channel_permutation",
                    choices=["none", "shuffle", "linear"],
                    # choices=["none", "shuffle", "linear"],
                ),
                CSH.CategoricalHyperparameter(
                    "fraction_unchanged",
                    choices=[
                        0.5,
                    ],
                    # choices=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                ),
                CSH.CategoricalHyperparameter(
                    "eps",
                    choices=[
                        1e-2,
                    ],
                    # choices=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "permute": {
            "func": inv_permute,
            "params": [
                CSH.CategoricalHyperparameter(
                    "use_lu",
                    choices=[True, False],
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "act_norm": {
            "func": act_norm,
            "params": [
                CSH.CategoricalHyperparameter(
                    "scale_fn", choices=["twice_sigmoid", "exp"]
                ),
                CSH.CategoricalHyperparameter(
                    "eps",
                    choices=[
                        1e-2,
                    ],
                    # choices=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "deep4_coupling": {
            "func": deep4_coupling_block,
            "params": [
                CSH.CategoricalHyperparameter(
                    "affine_or_additive",
                    choices=["affine", "additive"],
                ),
                CSH.CategoricalHyperparameter(
                    "scale_fn", choices=["twice_sigmoid", "exp"]
                ),
                CSH.CategoricalHyperparameter("dropout", choices=[0, 0.2, 0.5]),
                CSH.CategoricalHyperparameter(
                    "swap_dims",
                    choices=[True, False],
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
    }

    wanted_blocks = {name: blocks[name] for name in included_blocks}

    return wanted_blocks


def init_model_encoding(
    amplitude_phase_at_end,
    n_times,
    class_prob_masked,
    alpha_lr,
    dist_module_choices,
    rng,
    just_train_deep4,
    n_virtual_chans,
    linear_glow_clf,
    n_real_chans,
    n_classes,
    dist_lr,
    n_overall_mixes,
    n_dim_mixes,
    reduce_per_dim,
    reduce_overall_mix,
    init_dist_weight_std,
    init_dist_mean_std,
    init_dist_std_std,
):
    init_dist_std = 1e-1
    n_mixes = 8
    n_chans = n_real_chans + n_virtual_chans

    net_encoding = {
        "net": [],
    }
    n_dims = n_times * n_chans

    dist_module_options = {
        "perdimweightedmix": functools.partial(
            PerDimWeightedMix,
            n_classes=n_classes,
            n_mixes=n_mixes,
            n_dims=n_dims,
            optimize_mean=True,
            optimize_std=True,
            init_std=init_dist_std,
        ),
        "nclassindependent": functools.partial(
            NClassIndependentDist,
            n_classes=n_classes,
            n_dims=n_dims,
            optimize_mean=True,
            optimize_std=True,
        ),
        "maskedmix": functools.partial(
            MaskedMixDist,
            n_dims=n_dims,
            dist=PerDimWeightedMix(
                n_classes=n_classes,
                n_mixes=n_mixes,
                n_dims=n_dims,
                optimize_mean=True,
                optimize_std=True,
                init_std=init_dist_std,
            ),
        ),
        "maskedindependent": functools.partial(
            MaskedMixDist,
            n_dims=n_dims,
            dist=NClassIndependentDist(
                n_classes=n_classes,
                n_dims=n_dims,
                optimize_mean=True,
                optimize_std=True,
            ),
        ),
        "weightedgaussianmix": functools.partial(
            ClassWeightedGaussianMixture,
            n_classes=n_classes,
            n_mixes=32,
            n_dims=n_dims,
            init_std=init_dist_std,
            optimize_mean=True,
            optimize_std=True,
        ),
        # for debugging
        "weighteddimgaussianmix": functools.partial(
            ClassWeightedPerDimGaussianMixture,
            n_classes=n_classes,
            n_mixes=32,
            n_dims=n_dims,
            init_std=init_dist_std,
            optimize_mean=True,
            optimize_std=True,
        ),
        "hierarchical": functools.partial(
            ClassWeightedHierachicalGaussianMixture,
            n_classes=n_classes,
            n_overall_mixes=n_overall_mixes,
            n_dim_mixes=n_dim_mixes,
            n_dims=n_dims,
            reduce_per_dim=reduce_per_dim,
            reduce_overall_mix=reduce_overall_mix,
            init_weight_std=init_dist_weight_std,
            init_mean_std=init_dist_mean_std,
            init_std_std=init_dist_std_std,
            optimize_mean=True,
            optimize_std=True,
        ),
        "perclasshierarchical": functools.partial(
            PerClassHierarchical,
            n_classes=n_classes,
            n_overall_mixes=n_overall_mixes,
            n_dim_mixes=n_dim_mixes,
            n_dims=n_dims,
            reduce_per_dim=reduce_per_dim,
            reduce_overall_mix=reduce_overall_mix,
            init_weight_std=init_dist_weight_std,
            init_mean_std=init_dist_mean_std,
            init_std_std=init_dist_std_std,
            optimize_mean=True,
            optimize_std=True,
        ),
    }

    wanted_dist_module = rng.choice(dist_module_choices)
    dist = dist_module_options[wanted_dist_module]()

    cur_node = None

    if amplitude_phase_at_end:
        flat_node = Node(cur_node, InvertibleSequential(AmplitudePhase(), Flatten2d()))
    else:
        flat_node = Node(cur_node, InvertibleSequential(Flatten2d()))
    dist_node = Node(flat_node, dist)

    net = dist_node

    net = net.cuda()
    init_all_modules(net, None)
    # Now also add optim params
    net_encoding["optim_params_per_param"] = {}
    for p in net.parameters():
        net_encoding["optim_params_per_param"][p] = dict(lr=dist_lr, weight_decay=5e-5)
        if hasattr(dist, "alphas"):
            net_encoding["optim_params_per_param"][dist.alphas] = dict(
                lr=alpha_lr, weight_decay=5e-5
            )
        else:
            assert not wanted_dist_module == "masked_mix"
    if class_prob_masked:
        net.alphas = nn.Parameter(
            th.zeros(n_chans * n_times, device="cuda", requires_grad=True)
        )
        net_encoding["optim_params_per_param"][net.alphas] = dict(
            lr=alpha_lr, weight_decay=5e-5
        )
    net_encoding["node"] = net
    if just_train_deep4:
        from braindecode.models import Deep4Net
        from braindecode.models.util import to_dense_prediction_model
        from braindecode.models.util import get_output_shape
        from invertibleeeg.models.wrap_as_gen import WrapClfAsGen

        if n_times == 1000:
            model = Deep4Net(
                n_chans,
                n_classes,
                input_window_samples=None,
                final_conv_length=2,
            )
            to_dense_prediction_model(model)
        else:
            # assert n_times == 128
            input_window_samples = 144

            model = Deep4Net(
                n_chans,
                n_classes,
                input_window_samples=input_window_samples,
                final_conv_length=2,
                pool_time_stride=2,
                pool_time_length=2,
                filter_time_length=10,
                filter_length_2=9,
                filter_length_3=7,
                filter_length_4=7,
            )

            to_dense_prediction_model(model)
            assert get_output_shape(model, n_chans, input_window_samples)[2] == (
                input_window_samples - 128
            )

        new_modules = []
        for n, m in model.named_children():
            if n == "softmax":
                new_modules.append(nn.AdaptiveAvgPool2d(output_size=1))
            new_modules.append(m)
        new_model = nn.Sequential(*new_modules)
        wrapped_model = WrapClfAsGen(new_model)
        net_encoding["deep4"] = wrapped_model.cuda()
    if linear_glow_clf:
        linear = nn.Linear(n_chans, n_classes).cuda()
        net_encoding["linear_clf"] = linear
        net_encoding["optim_params_per_param"][linear.weight] = dict(
            lr=1e-3, weight_decay=5e-5
        )
        net_encoding["optim_params_per_param"][linear.bias] = dict(
            lr=1e-3, weight_decay=5e-5
        )
    return net_encoding


def try_run_backward(model_to_train, batch_size, n_chans, n_times):
    _, lp = model_to_train(th.zeros(batch_size, n_chans, n_times, device="cuda"))
    mean_lp = th.mean(lp)
    mean_lp.backward()
    _ = [p.grad.zero_() for p in model_to_train.parameters() if p.grad is not None]


def try_run_forward(model_to_train, batch_size, n_chans, n_times):
    _, lp = model_to_train(th.zeros(batch_size, n_chans, n_times, device="cuda"))
    mean_lp = th.mean(lp)
    return mean_lp.detach().item()


def run_exp(
    debug,
    np_th_seed,
    worker_folder,
    max_hours,
    n_start_population,
    n_alive_population,
    subject_id,
    n_epochs,
    amplitude_phase_at_end,
    all_subjects_in_each_fold,
    n_times_train,
    n_times_eval,
    max_n_changes,
    fixed_lr,
    searchspace,
    include_splitter,
    class_names,
    fixed_batch_size,
    class_prob_masked,
    nll_loss_factor,
    search_by,
    alpha_lr,
    dist_module_choices,
    scheduler,
    trial_start_offset_sec,
    n_times_crop,
    just_train_deep4,
    n_virtual_chans,
    split_valid_off_train,
    linear_glow_clf,
    splitter_name,
    low_cut_hz,
    high_cut_hz,
    exponential_standardize,
    included_blocks,
    limit_n_downsample,
    sfreq,
    mutate_optim_params,
    max_n_deletions,
    channel_drop_p,
    n_eval_crops,
    dataset_name,
    min_n_downsample,
    hgd_sensors,
    min_improve_fraction,
    n_tuh_recordings,
    dist_lr,
    n_virtual_classes,
    n_overall_mixes,
    n_dim_mixes,
    reduce_per_dim,
    reduce_overall_mix,
    init_dist_weight_std,
    init_dist_mean_std,
    init_dist_std_std,
):
    noise_factor = 5e-3
    start_time = time.time()

    n_classes = len(class_names)
    if dataset_name == "tuh":
        n_classes = 2
    n_classes = n_classes + n_virtual_classes

    # maybe specifiy via hparam to ensure you know of the change class_names = ["left_hand", "right_hand", "feet", "tongue"]

    set_random_seeds(np_th_seed, True)
    log.info("Load data...")
    if dataset_name == "bcic_iv_2a":
        train_set, valid_set, test_set = load_train_valid_test_bcic_iv_2a(
            subject_id,
            class_names,
            split_valid_off_train=split_valid_off_train,
            all_subjects_in_each_fold=all_subjects_in_each_fold,
            sfreq=sfreq,
            trial_start_offset_sec=trial_start_offset_sec,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
            exponential_standardize=exponential_standardize,
        )
    elif dataset_name == "tuh":
        # train_set, valid_set, test_set = load_tuh_abnormal(
        #    n_recordings_to_load=n_tuh_recordings
        # )
        log.info("Load TUH data from pickle...")
        datasets = pickle.load(
            open(
                "/work/dlclarge1/schirrmr-renormalized-flows/tuh_train_valid_test_64_hz_clipped.pkl",
                "rb",
            )
        )
        log.info("Done.")
        train_set = datasets["train"]
        valid_set = datasets["valid"]
        test_set = datasets["test"]
        # if in_clip_val is not None:
        #     train_set.transform = partial(np.clip, a_min=-in_clip_val, a_max=in_clip_val)
        #     valid_set.transform = partial(np.clip, a_min=-in_clip_val, a_max=in_clip_val)
        #     test_set.transform = partial(np.clip, a_min=-in_clip_val, a_max=in_clip_val)

        if n_tuh_recordings is not None:
            train_set = th.utils.data.Subset(train_set, range(n_tuh_recordings))
            valid_set = th.utils.data.Subset(valid_set, range(n_tuh_recordings))
            test_set = th.utils.data.Subset(test_set, range(n_tuh_recordings))
    else:
        assert dataset_name == "hgd"
        train_set, valid_set, test_set = load_train_valid_test_hgd(
            subject_id,
            class_names,
            split_valid_off_train=split_valid_off_train,
            all_subjects_in_each_fold=all_subjects_in_each_fold,
            sfreq=sfreq,
            trial_start_offset_sec=trial_start_offset_sec,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
            exponential_standardize=exponential_standardize,
            sensors=hgd_sensors,
        )
    n_real_chans = train_set[0][0].shape[0]
    n_chans = n_real_chans + n_virtual_chans

    max_n_downsample = int(np.log2(n_times_crop or n_times_train) - 3)
    if limit_n_downsample is not None:
        max_n_downsample = min(max_n_downsample, limit_n_downsample)
    block_fn = dict(
        simpleflow=partial(get_simpleflow_blocks, include_splitter=include_splitter),
        downsample_anywhere=partial(
            get_downsample_anywhere_blocks, included_blocks=included_blocks
        ),
    )[searchspace]

    rng = np.random.RandomState(np_th_seed)
    while (time.time() - start_time) < (max_hours * 3600):
        ex = Experiment()
        ex.observers.append(ConcurrentFileStorageObserver.create(worker_folder))
        config = {}

        @ex.main
        def wrap_run_exp():
            csv_file = os.path.join(worker_folder, "population.csv")
            csv_lock_file = csv_file + ".lock"
            with fasteners.InterProcessLock(csv_lock_file):
                try:
                    population_df = pd.read_csv(csv_file, index_col="pop_id")
                except FileNotFoundError:
                    population_df = None
            if population_df is not None and len(population_df) >= n_start_population:
                # grab randomly one among top n_population
                # change it, with chance of no change as well
                sorted_pop_df = population_df.sort_values(
                    by=[search_by, "pop_id"], ascending=[True, True]
                )
                this_parent = sorted_pop_df.iloc[
                    rng.randint(0, min(len(population_df), n_alive_population))
                ]
                last_metric_val = this_parent[search_by]
                parent_folder = this_parent["folder"]
                parent_encoding_filename = os.path.join(parent_folder, "encoding.pth")
                parent_encoding = th.load(parent_encoding_filename)

                other_parent = sorted_pop_df.iloc[
                    rng.randint(0, min(len(population_df), n_alive_population))
                ]
                other_parent_folder = other_parent["folder"]
                other_parent_encoding_filename = os.path.join(
                    other_parent_folder, "encoding.pth"
                )
                other_parent_encoding = th.load(other_parent_encoding_filename)

                model_worked = False
                log.info("Mutating and trying model...")
                if fixed_batch_size is None:
                    batch_size = 32
                else:
                    batch_size = fixed_batch_size
                while not model_worked:
                    try:
                        encoding = mutate_encoding_and_model(
                            parent_encoding,
                            rng,
                            block_fn(),
                            n_start_chans=n_chans,
                            max_n_changes=max_n_changes,
                            other_encoding=other_parent_encoding,
                            fixed_lr=fixed_lr,
                            max_n_downsample=max_n_downsample,
                            splitter_name=splitter_name,
                            mutate_optim_params=mutate_optim_params,
                            max_n_deletions=max_n_deletions,
                            min_n_downsample=min_n_downsample,
                        )
                        model_to_train = encoding["node"].cuda()
                        if n_times_crop is not None:
                            model_to_train = CroppedGlow(
                                model_to_train, n_times=n_times_crop
                            )

                        # Try a forward-backward to ensure model works
                        # Also here you should check function is unperturbed!!
                        try_run_backward(
                            model_to_train, batch_size, n_chans, n_times_train
                        )
                        try_run_forward(
                            model_to_train, batch_size, n_chans, n_times_eval
                        )
                        model_worked = True
                        log.info("Model:\n" + str(model_to_train))
                    except RuntimeError as e:
                        log.info("Model failed....:\n" + str(model_to_train))
                        traceback.print_exc()
            else:
                encoding = init_model_encoding(
                    amplitude_phase_at_end,
                    n_times=n_times_crop or n_times_train,
                    class_prob_masked=class_prob_masked,
                    alpha_lr=alpha_lr,
                    dist_module_choices=dist_module_choices,
                    rng=rng,
                    just_train_deep4=just_train_deep4,
                    n_virtual_chans=n_virtual_chans,
                    linear_glow_clf=linear_glow_clf,
                    n_real_chans=n_real_chans,
                    n_classes=n_classes,
                    dist_lr=dist_lr,
                    n_overall_mixes=n_overall_mixes,
                    n_dim_mixes=n_dim_mixes,
                    reduce_per_dim=reduce_per_dim,
                    reduce_overall_mix=reduce_overall_mix,
                    init_dist_weight_std=init_dist_weight_std,
                    init_dist_mean_std=init_dist_mean_std,
                    init_dist_std_std=init_dist_std_std,
                )
                last_metric_val = np.inf
                model_to_train = encoding["node"].cuda()
                if n_times_crop is not None:
                    model_to_train = CroppedGlow(model_to_train, n_times=n_times_crop)
                parent_folder = None
            # Now determine suitable batch_size
            model_worked = False
            if fixed_batch_size is None:
                log.info("Determine max batch size...")
                batch_size = 512
                while not model_worked:
                    duplicate_model = deepcopy(model_to_train)
                    try:

                        try_run_backward(
                            duplicate_model, batch_size, n_chans, n_times_train
                        )
                        try_run_forward(
                            duplicate_model, batch_size, n_chans, n_times_eval
                        )
                        model_worked = True
                        log.info(f"Batch size: {batch_size}")
                    except RuntimeError as e:
                        log.info(f"Batch size {batch_size} failed.")
                        del duplicate_model
                        duplicate_model = deepcopy(model_to_train)
                        # clear gpu memory by forwarding minimal batch
                        import gc

                        gc.collect()
                        th.cuda.empty_cache()

                        _, lp = duplicate_model(
                            th.zeros(1, n_chans, n_times_train, device="cuda")
                        )
                        mean_lp = th.mean(lp)
                        mean_lp.backward()
                        batch_size = batch_size // 2
                        if batch_size == 1:
                            raise e
                        traceback.print_exc()
                # still divide batch size by half to esnure some buffer
                batch_size = batch_size // 2
                del duplicate_model
            else:
                batch_size = fixed_batch_size

            # train it get result
            # (maybe also remember how often it was trained, history, or maybe just remember parent id  in df)
            log.info("Train model...")
            if just_train_deep4:
                model_to_train = encoding["deep4"]
                optim_params_per_param = {}
                for p in model_to_train.parameters():
                    optim_params_per_param[p] = dict(
                        lr=1 * 0.01, weight_decay=0.5 * 0.001
                    )
            else:
                optim_params_per_param = encoding["optim_params_per_param"]
            if linear_glow_clf:
                model_to_train = GlowLinearClassifier(
                    model_to_train, encoding["linear_clf"], n_times=n_times_crop
                ).cuda()
            # result and encoding
            import gc

            gc.collect()
            th.cuda.empty_cache()

            scheduler_callable = dict(
                identity=functools.partial(
                    th.optim.lr_scheduler.LambdaLR, lr_lambda=lambda epoch: 1
                ),
                cosine=functools.partial(
                    th.optim.lr_scheduler.CosineAnnealingLR,
                    T_max=n_epochs * (len(train_set) // batch_size),
                ),
                cosine_with_warmup=functools.partial(
                    get_cosine_warmup_scheduler,
                    warmup_steps=len(train_set) // batch_size,  # 1 epoch warmup
                    cosine_steps=(n_epochs - 1) * (len(train_set) // batch_size),
                ),
            )[scheduler]

            results = None
            encoding_before_last_train = None
            log.info("Start training...")
            while (results is None) or (
                results[search_by] < ((1 - min_improve_fraction) * last_metric_val)
            ):
                if results is not None:
                    log.info("Continue training...")
                    last_metric_val = results[search_by]
                    # store encoding before last train (moving node to cpu)
                    # (set this to none before loop)
                    encoding_before_last_train = deepcopy(encoding)
                    encoding_before_last_train["node"].cpu()
                    # also store last results
                    results_before_last_train = results
                start_train_time = time.time()  # Measure only train time
                results = train_glow(
                    model_to_train,
                    class_prob_masked,
                    nll_loss_factor,
                    noise_factor,
                    train_set,
                    valid_set,
                    test_set,
                    n_epochs,
                    n_virtual_chans,
                    n_classes=n_classes,
                    n_times_train=n_times_train,
                    np_th_seed=np_th_seed,
                    scheduler=scheduler_callable,
                    batch_size=batch_size,
                    optim_params_per_param=optim_params_per_param,
                    with_tqdm=False,
                    n_times_eval=n_times_eval,
                    channel_drop_p=channel_drop_p,
                    n_times_crop=n_times_crop,
                    n_eval_crops=n_eval_crops,
                )
            log.info("Finished training...")
            runtime = time.time() - start_train_time
            del model_to_train, optim_params_per_param
            # if result was better one training round earlier
            # reset to that
            if (encoding_before_last_train is not None) and (
                results_before_last_train[search_by] < results[search_by]
            ):
                results = results_before_last_train
                encoding = encoding_before_last_train
            metrics = results
            results["runtime"] = runtime
            for key, val in metrics.items():
                ex.info[key] = val

            file_obs = ex.observers[0]
            output_dir = file_obs.dir
            encoding["node"].cpu()
            th.save(encoding, os.path.join(output_dir, "encoding.pth"))
            clean_encoding = copy_clean_encoding_dict(encoding)
            th.save(clean_encoding, os.path.join(output_dir, "encoding_no_params.pth"))

            with fasteners.InterProcessLock(csv_lock_file):
                try:
                    population_df = pd.read_csv(csv_file, index_col="pop_id")
                except FileNotFoundError:
                    population_df = pd.DataFrame()

                this_dict = dict(
                    folder=output_dir, parent_folder=parent_folder, **metrics
                )
                population_df = population_df.append(this_dict, ignore_index=True)
                population_df.to_csv(csv_file, index_label="pop_id")

            return results

        ex.add_config(config)
        ex.run()
    # Store best result as overall result
    csv_file = os.path.join(worker_folder, "population.csv")
    csv_lock_file = csv_file + ".lock"
    with fasteners.InterProcessLock(csv_lock_file):
        population_df = pd.read_csv(csv_file, index_col="pop_id")
    results = dict(
        population_df.sort_values(by=[search_by, "pop_id"])
        .loc[
            :,
            [
                "runtime",
                "test_mis",
                "test_nll",
                "train_mis",
                "train_nll",
                "valid_mis",
                "valid_nll",
            ],
        ]
        .iloc[0]
    )
    return results
