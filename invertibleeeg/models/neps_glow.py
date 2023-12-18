import time
import traceback
from copy import deepcopy
import pickle

import numpy as np
import torch as th

from invertible.distribution import NClassIndependentDist, ClassWeightedPerDimGaussianMixture, \
    ClassWeightedGaussianMixture, ClassWeightedHierachicalGaussianMixture, PerClassHierarchical, \
    SomeClassIndependentDimsDist
from invertible.distribution import PerClassDists
from invertible.inverse import Inverse
from invertible.sequential import InvertibleSequential
from invertibleeeg.train import train_glow
from invertibleeeg.experiments.nas import (
    coupling_block,
    inv_permute,
    act_norm,
    try_run_backward,
    try_run_forward,
)
from invertibleeeg.models.cropped import CroppedGlow
from invertibleeeg.models.glow import get_splitter
import logging
from invertibleeeg.datasets import (
    load_train_valid_test_hgd,
    load_train_valid_test_bcic_iv_2a,
    load_tuh_abnormal,
)
import neps
import functools
import invertibleeeg.models.conv

def create_eeg_glow_neps_model(
    n_chans,
    n_classes,
    n_times_crop,
    n_blocks,
    n_stages,
    hidden_channels,
    kernel_length,
    channel_permutation,
    dropout_prob,
    dist_name,
    n_mixes,
    eps,
    affine_or_additive,
    n_overall_mixes,
    n_dim_mixes,
    reduce_per_dim,
    reduce_overall_mix,
    init_dist_weight_std,
    init_dist_mean_std,
    init_dist_std_std,
    optimize_std,
    n_class_independent_dims,
    first_conv_class_name,
):

    stages = []
    for n_downsample in range(1, 1 + n_stages):  # 4
        n_this_chans = int(n_chans * (2**n_downsample))
        blocks = []
        for _ in range(n_blocks):
            a_block = act_norm(n_chans=n_this_chans, scale_fn="twice_sigmoid", eps=eps)
            p_block = inv_permute(n_chans=n_this_chans, use_lu=True)
            p_block.reset_to_identity()
            c_block = coupling_block(
                n_chans=n_this_chans,
                hidden_channels=hidden_channels,
                kernel_length=kernel_length,
                affine_or_additive=affine_or_additive,
                scale_fn="twice_sigmoid",
                dropout=dropout_prob,
                swap_dims=False,
                norm="none",
                nonlin="elu",
                multiply_by_zeros=True,
                channel_permutation=channel_permutation,
                fraction_unchanged=0.5,
                eps=eps,
                first_conv_class_name=first_conv_class_name,
            )
            blocks.extend([a_block, p_block, c_block])
        stage = wrap_downsample(InvertibleSequential(*blocks), n_downsample, "haar")
        stages.append(stage)

    from invertible.distribution import MaskedMixDist, PerDimWeightedMix

    n_dims = n_times_crop * n_chans
    init_dist_std = 0.1
    if dist_name == "maskedmix":
        dist = MaskedMixDist(
            n_dims=n_dims,
            dist=PerDimWeightedMix(
                n_classes=n_classes,
                n_mixes=n_mixes,
                n_dims=n_dims,
                optimize_mean=True,
                optimize_std=optimize_std,
                init_std=init_dist_std,
            ),
        )
    elif dist_name == "perdimweightedmix":
        dist = PerDimWeightedMix(
            n_classes=n_classes,
            n_mixes=n_mixes,
            n_dims=n_dims,
            optimize_mean=True,
            optimize_std=optimize_std,
            init_std=init_dist_std,
        )
    elif dist_name == "weightedgaussianmix":
        dist = ClassWeightedGaussianMixture(
            n_classes=n_classes,
            n_mixes=n_mixes,
            n_dims=n_dims,
            init_std=init_dist_std,
            optimize_mean=True,
            optimize_std=optimize_std,
        )
    elif dist_name == "weighteddimgaussianmix":
        dist = ClassWeightedPerDimGaussianMixture(
            n_classes=n_classes,
            n_mixes=n_mixes,
            n_dims=n_dims,
            init_std=init_dist_std,
            optimize_mean=True,
            optimize_std=optimize_std,
        )
    elif dist_name == "maskedweighteddimmmix":
        dist = MaskedMixDist(
            n_dims=n_dims,
            dist=ClassWeightedPerDimGaussianMixture(
                n_classes=n_classes,
                n_mixes=n_mixes,
                n_dims=n_dims,
                init_std=init_dist_std,
                optimize_mean=True,
                optimize_std=optimize_std,
            ),
        )
    elif dist_name == 'hierarchical':
        dist = ClassWeightedHierachicalGaussianMixture(
            n_classes=n_classes,
            n_overall_mixes=n_overall_mixes,
            n_dim_mixes=n_dim_mixes,
            n_dims=n_dims,
            reduce_per_dim=reduce_per_dim,
            reduce_overall_mix=reduce_overall_mix,
            optimize_mean=True,
            optimize_std=optimize_std,
            init_weight_std=init_dist_weight_std,
            init_mean_std=init_dist_mean_std,
            init_std_std=init_dist_std_std,
        )
    elif dist_name == "perclasshierarchical":
       dist = PerClassHierarchical(
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
            optimize_std=optimize_std,
        )
    elif dist_name == "perclassgaussianmix":
        dist = PerClassDists([
            ClassWeightedGaussianMixture(
            n_classes=1,
            n_mixes=n_mixes,
            n_dims=n_dims,
            init_std=init_dist_std,
            optimize_mean=True,
            optimize_std=optimize_std,
            )
        for _ in range(n_classes)])
    elif dist_name == "someindependentdims":
        dist = SomeClassIndependentDimsDist(
            n_classes=n_classes,
            n_dims=n_dims,
            n_class_independent_dims=n_class_independent_dims,
            optimize_std=optimize_std,
        )
    else:
        assert dist_name == "nclassindependent"
        dist = NClassIndependentDist(
            n_classes=n_classes,
            n_dims=n_dims,
            optimize_mean=True,
            optimize_std=optimize_std,
        )

    from invertible.graph import Node
    from invertible.view_as import Flatten2d

    process_node = Node(None, InvertibleSequential(*stages))
    flat_node = Node(process_node, InvertibleSequential(Flatten2d()))

    dist_node = Node(flat_node, dist)

    model = dist_node.cuda()

    from invertible.init import init_all_modules

    init_all_modules(model, None)

    return model