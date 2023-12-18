import time
from copy import deepcopy
import traceback
import gc
from torch import nn

from braindecode.datasets import BaseConcatDataset

from invertibleeeg.models.remove_high_freq import RemoveHighFreq
from invertibleeeg.train import train_glow
from invertibleeeg.models.cropped import CroppedGlow

from invertible.gaussian import get_gaussian_log_probs
from invertible.view_as import flatten_2d
from invertible.sequential import InvertibleSequential
from skorch.utils import to_numpy
from torch.utils.data import Subset

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
import seaborn
from invertibleeeg.plot import plot_head_signals_tight
import torch as th
from braindecode.util import set_random_seeds
from braindecode.models.modules import Expression
from functools import partial

import os.path

import numpy as np
import pickle

from invertibleeeg.experiments.neps import create_eeg_glow_neps_model
from invertibleeeg.experiments.nas import (
    try_run_backward,
    try_run_forward,
)
import logging

log = logging.getLogger(__name__)


def run_exp(
    np_th_seed,
    n_epochs,
    n_restarts,
    n_blocks,
    n_stages,
    output_dir,
    saved_model_folder,
    debug,
    in_clip_val,
    dist_name,
    affine_or_additive,
    n_times_train_eval,
    n_mixes,
    n_overall_mixes,
    n_dim_mixes,
    reduce_per_dim,
    reduce_overall_mix,
    init_dist_weight_std,
    init_dist_mean_std,
    init_dist_std_std,
    nll_loss_factor,
    temperature_init,
    optimize_std,
    n_class_independent_dims,
    first_conv_class_name,
    high_cut_hz,
    lowpass_for_model,
):
    n_classes = 2
    half_kernel_length = 5
    hidden_channels = 256
    channel_permutation = "linear"
    dropout_prob = 0.5
    start_lr = 5e-4
    weight_decay = 5e-5
    scheduler = "cosine_with_warmup"
    n_times_crop = 128
    n_times_train = n_times_train_eval
    n_times_eval = n_times_train_eval
    n_chans = 21
    dist_lr = 1e-2
    eps = 1e-2

    noise_factor = 5e-3
    class_prob_masked = False
    n_virtual_chans = 0
    n_eval_crops = 3
    channel_drop_p = 0
    kernel_length = half_kernel_length * 2 + 1

    set_random_seeds(np_th_seed, True)
    datasets = pickle.load(
        open(
            "/work/dlclarge1/schirrmr-renormalized-flows/tuh_train_valid_test_64_hz_clipped.pkl",
            "rb",
        )
    )
    train_set = datasets["train"]
    valid_set = datasets["valid"]
    test_set = datasets["test"]

    def clip_and_lowpass(x, in_clip_val, high_cut_hz):
        if in_clip_val is not None:
            x = np.clip(x, a_min=-in_clip_val, a_max=in_clip_val)
        if high_cut_hz is not None:
            i_stop = np.searchsorted(np.fft.rfftfreq(x.shape[1], 1 / 64.0), high_cut_hz)
            a_th = th.tensor(x.astype(np.float32), device='cuda')
            with th.no_grad():
                ffted = th.fft.rfft(a_th)
                ffted[:, i_stop:] = 0
                x = th.fft.irfft(ffted)
        return x
    if in_clip_val is not None:
        train_set.transform = partial(clip_and_lowpass, in_clip_val=in_clip_val, high_cut_hz=high_cut_hz)
        valid_set.transform =  partial(clip_and_lowpass, in_clip_val=in_clip_val, high_cut_hz=high_cut_hz)
        test_set.transform =  partial(clip_and_lowpass, in_clip_val=in_clip_val, high_cut_hz=high_cut_hz)

    if debug:
        n_epochs = 2  # need at least two in case warmup is specified
        n_restarts = 1
        train_set = th.utils.data.Subset(train_set, range(512))
        valid_set = th.utils.data.Subset(valid_set, range(256))
        test_set = th.utils.data.Subset(test_set, range(256))
    train_valid_set = BaseConcatDataset([train_set, valid_set])

    cuda = (
        th.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    if cuda:
        th.backends.cudnn.benchmark = True
    if saved_model_folder is None:
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
            eps,
            affine_or_additive=affine_or_additive,
            n_overall_mixes=n_overall_mixes,
            n_dim_mixes=n_dim_mixes,
            reduce_per_dim=reduce_per_dim,
            reduce_overall_mix=reduce_overall_mix,
            init_dist_weight_std=init_dist_weight_std,
            init_dist_mean_std=init_dist_mean_std,
            init_dist_std_std=init_dist_std_std,
            optimize_std=optimize_std,
            n_class_independent_dims=n_class_independent_dims,
            first_conv_class_name=first_conv_class_name,
        )

        cropped_model = CroppedGlow(model, n_times_crop)
        if lowpass_for_model and (high_cut_hz is not None):
            dist = cropped_model.dist
            cropped_model = InvertibleSequential(RemoveHighFreq(high_cut_hz=high_cut_hz), cropped_model)
            cropped_model.dist = dist
        if temperature_init is not None:
            cropped_model.temperature = nn.Parameter(th.tensor(np.log(temperature_init), device='cuda'))
    else:
        assert saved_model_folder is not None
        cropped_model = th.load(os.path.join(saved_model_folder, "net.th"))

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

            gc.collect()
            th.cuda.empty_cache()

            batch_size = batch_size // 2
            traceback.print_exc()
            if batch_size == 2:
                raise e
    # still divide batch size by half to esnure some buffer
    batch_size = batch_size // 2
    del duplicate_model

    gc.collect()
    th.cuda.empty_cache()

    optim_params_per_param = {}
    for p in cropped_model.parameters():
        optim_params_per_param[p] = dict(lr=start_lr, weight_decay=weight_decay)
    for p in cropped_model.dist.parameters():
        optim_params_per_param[p] = dict(lr=dist_lr, weight_decay=weight_decay)

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
    for _ in range(n_restarts):
        results = train_glow(
            cropped_model,
            class_prob_masked,
            nll_loss_factor,
            noise_factor,
            train_valid_set,
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
            use_temperature=hasattr(cropped_model, 'temperature'),
        )
        print(results)

    train_loader = th.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    valid_loader = th.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = th.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    n_real_chans = 21
    net = cropped_model
    net.eval()
    n_input_times = train_set[0][0].shape[1]
    net.eval()
    results = {}
    for name, loader in (
            ("train", train_loader),
            ("valid", valid_loader),
            ("test", test_loader),
    ):
        all_lps = []
        all_corrects = []
        for X_th, y, _ in loader:
            with th.no_grad():
                i_crop_starts = np.unique(
                    np.int64(
                        np.linspace(
                            0, n_input_times - n_times_eval, n_eval_crops
                        ).round()
                    )
                )
                crop_lps = []
                crop_lp_per_class = []
                for i_crop_start in i_crop_starts:
                    crop_X_th = X_th[:, :, i_crop_start: i_crop_start + n_times_eval]
                    assert n_virtual_chans == 0
                    #crop_X_th = add_virtual_chans(crop_X_th, n_virtual_chans)
                    y_th = th.nn.functional.one_hot(
                        y.type(th.int64), num_classes=n_classes
                    ).cuda()
                    # For now lt's do both with noise, so not like comment below
                    # First with noise to get nll for bpd,
                    # then without noise for accuracy
                    noise = th.randn_like(crop_X_th) * noise_factor
                    noised = crop_X_th + noise
                    noise_log_prob = get_gaussian_log_probs(
                        th.zeros_like(crop_X_th[0]).view(-1),
                        th.log(th.zeros_like(crop_X_th[0]) + noise_factor).view(-1),
                        flatten_2d(noise),
                    )
                    z, lp_per_class = net(
                        noised.cuda(), fixed=dict(y=None, sum_dims=True)
                    )
                    lps = to_numpy(th.sum(lp_per_class * y_th, dim=1) - noise_log_prob.cuda())
                    crop_lp_per_class.append(lp_per_class.detach().cpu())
                    crop_lps.append(lps)

                lp_per_class = th.stack(crop_lp_per_class).mean(dim=0)
                lps = np.mean(np.stack(crop_lps), axis=0)
                all_lps.extend(lps)
                corrects = to_numpy(y.cpu() == lp_per_class.argmax(dim=1))
                all_corrects.extend(corrects)
        acc = np.mean(all_corrects)
        nll = -(np.mean(all_lps) / (n_real_chans * n_times_crop * np.log(2)))
        results[f"{name}_mis"] = 1 - acc
        results[f"{name}_nll"] = nll
    for key, val in results.items():
        print(f"{key:10s}: {val:.3f}")

    th.save(net, os.path.join(output_dir, "net.th"))
    # th.save(net.state_dict(), os.path.join(output_dir, "net_state_dict.th"))


    return results
