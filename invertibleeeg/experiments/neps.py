import time
import traceback
from copy import deepcopy

import numpy as np
import torch as th

from invertible.distribution import NClassIndependentDist
from invertible.inverse import Inverse
from invertible.sequential import InvertibleSequential
from invertibleeeg.experiments.bcic_iv_2a import train_glow
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
)
import neps
import functools

log = logging.getLogger(__name__)


def wrap_downsample(block, n_downsample, splitter_name):
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
    return block


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
):

    stages = []
    for n_downsample in range(1, 1 + n_stages):  # 4
        n_this_chans = int(n_chans * (2**n_downsample))
        blocks = []
        for _ in range(n_blocks):
            a_block = act_norm(n_chans=n_this_chans, scale_fn="twice_sigmoid")
            p_block = inv_permute(n_chans=n_this_chans, use_lu=True)
            p_block.reset_to_identity()
            c_block = coupling_block(
                n_chans=n_this_chans,
                hidden_channels=hidden_channels,
                kernel_length=kernel_length,
                affine_or_additive="affine",
                scale_fn="twice_sigmoid",
                dropout=dropout_prob,
                swap_dims=False,
                norm="none",
                nonlin="elu",
                multiply_by_zeros=True,
                channel_permutation=channel_permutation,
                fraction_unchanged=0.5,
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
                optimize_std=True,
                init_std=init_dist_std,
            ),
        )
    elif dist_name == "perdimweightedmix":
        dist = PerDimWeightedMix(
            n_classes=n_classes,
            n_mixes=n_mixes,
            n_dims=n_dims,
            optimize_mean=True,
            optimize_std=True,
            init_std=init_dist_std,
        )
    else:
        assert dist_name == "nclassindependent"
        dist = NClassIndependentDist(
            n_classes=n_classes,
            n_dims=n_dims,
            optimize_mean=True,
            optimize_std=True,
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


def train_and_evaluate_glow(
    n_blocks,
    n_stages,
    hidden_channels,
    half_kernel_length,
    channel_permutation,
    dropout_prob,
    dist_name,
    n_mixes,
    start_lr,
    weight_decay,
    scheduler,
    n_epochs,
    train_set,
    valid_set,
    test_set,
    n_chans,
    n_classes,
    n_times_crop,
    n_times_train,
    n_times_eval,
    with_tqdm,
):

    noise_factor = 5e-3
    class_prob_masked = False
    nll_loss_factor = 1
    n_virtual_chans = 0
    n_eval_crops = 3
    np_th_seed = np.random.randint(0, 2**32)
    channel_drop_p = 0
    kernel_length = half_kernel_length * 2 + 1
    cuda = (
        th.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    if cuda:
        th.backends.cudnn.benchmark = True
    model = create_eeg_glow_neps_model(
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
    )
    cropped_model = CroppedGlow(model, n_times_crop)
    batch_size = 256
    model_worked = False
    while not model_worked:
        duplicate_model = deepcopy(cropped_model)
        try:

            try_run_backward(duplicate_model, batch_size, n_chans, n_times_train)
            try_run_forward(duplicate_model, batch_size, n_chans, n_times_eval)
            model_worked = True
            log.info(f"Batch size: {batch_size}")
        except RuntimeError as e:
            log.info(f"Batch size {batch_size} failed.")
            del duplicate_model
            # clear gpu memory by forwarding minimal batch
            import gc

            gc.collect()
            th.cuda.empty_cache()

            batch_size = batch_size // 2
            if batch_size == 1:
                return dict(loss=1, cost=0)
            traceback.print_exc()
    # still divide batch size by half to esnure some buffer
    batch_size = batch_size // 2
    del duplicate_model
    import gc

    gc.collect()
    th.cuda.empty_cache()

    optim_params_per_param = {}
    for p in cropped_model.parameters():
        optim_params_per_param[p] = dict(lr=start_lr, weight_decay=weight_decay)

    from invertibleeeg.scheduler import get_cosine_warmup_scheduler
    import functools

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

    start_time = time.time()
    results = train_glow(
        cropped_model,
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
        with_tqdm=with_tqdm,
        n_times_eval=n_times_eval,
        channel_drop_p=channel_drop_p,
        n_times_crop=n_times_crop,
        n_eval_crops=n_eval_crops,
    )
    results["runtime"] = time.time() - start_time

    neps_results = dict(
        loss=results["valid_mis"], cost=results["runtime"], info_dict=results
    )
    return neps_results


def run_exp(
    debug,
    neps_output_dir,
    max_hours,
    epochs_as_fidelity,
    class_names,
    trial_start_offset_sec,
    subject_id,
    split_valid_off_train,
    sfreq,
    n_times_train,
    n_times_eval,
    all_subjects_in_each_fold,
    exponential_standardize,
    low_cut_hz,
    high_cut_hz,
    hgd_sensors,
    dataset_name,
    n_max_epochs,
    ignore_errors,
):
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
    n_chans = train_set[0][0].shape[0]
    n_classes = len(class_names)
    n_times_crop = 128
    pipeline_space = dict(
        n_blocks=neps.IntegerParameter(lower=0, upper=8),
        n_stages=neps.IntegerParameter(lower=0, upper=4),
        hidden_channels=neps.IntegerParameter(lower=8, upper=512, log=True),
        half_kernel_length=neps.IntegerParameter(
            lower=0,
            upper=15,
        ),
        channel_permutation=neps.CategoricalParameter(
            choices=["shuffle", "linear", "none"]
        ),
        dropout_prob=neps.FloatParameter(lower=0, upper=1),
        start_lr=neps.FloatParameter(lower=1e-5, upper=1, log=True),
        weight_decay=neps.FloatParameter(lower=1e-5, upper=1e-2, log=True),
        dist_name=neps.CategoricalParameter(
            choices=["maskedmix", "perdimweightedmix", "nclassindependent"]
        ),
        n_mixes=neps.IntegerParameter(lower=1, upper=16),
        scheduler=neps.CategoricalParameter(
            choices=["identity", "cosine", "cosine_with_warmup"]
        ),
    )
    if epochs_as_fidelity:
        pipeline_space["n_epochs"] = neps.IntegerParameter(
            lower=2, upper=n_max_epochs, is_fidelity=True
        )
    else:
        pipeline_space["n_epochs"] = neps.CategoricalParameter(choices=[n_max_epochs])


    neps.run(
        run_pipeline=functools.partial(
            train_and_evaluate_glow,
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            n_chans=n_chans,
            n_classes=n_classes,
            n_times_crop=n_times_crop,
            n_times_train=n_times_train,
            n_times_eval=n_times_eval,
            with_tqdm=False,
        ),
        pipeline_space=pipeline_space,
        root_directory=neps_output_dir,
        max_cost_total=max_hours * 3600,
        ignore_errors=ignore_errors,
    )
    return {}
