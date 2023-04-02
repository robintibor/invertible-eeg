import time
from copy import deepcopy
import traceback
import gc

from braindecode.datasets import BaseConcatDataset

from invertibleeeg.experiments.bcic_iv_2a import train_glow
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
):
    n_classes = 2
    half_kernel_length = 5
    hidden_channels = 256
    channel_permutation = "linear"
    dropout_prob = 0.5
    start_lr = 5e-4
    weight_decay = 5e-5
    n_mixes = 8
    scheduler = "cosine_with_warmup"
    n_times_crop = 128
    n_times_train = n_times_train_eval
    n_times_eval = n_times_train_eval
    n_chans = 21
    dist_lr = 1e-2
    eps = 1e-2

    noise_factor = 5e-3
    class_prob_masked = False
    nll_loss_factor = 1
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
    if in_clip_val is not None:
        train_set.transform = partial(np.clip, a_min=-in_clip_val, a_max=in_clip_val)
        valid_set.transform = partial(np.clip, a_min=-in_clip_val, a_max=in_clip_val)
        test_set.transform = partial(np.clip, a_min=-in_clip_val, a_max=in_clip_val)

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
        )

        cropped_model = CroppedGlow(model, n_times_crop)
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

    class_names = ["healthy", "pathological"]

    # Modify for visualization
    net.mods_until_flat.sequential[0].next = []

    net = InvertibleSequential(net.mods_until_flat, net.flat_module, net.dist)
    net.module = net.sequential[-1]
    n_epochs_visualization = 500
    with th.no_grad():

        X = th.zeros(1, n_chans, n_times_crop)
        X = X.cuda()
        z_zero = net(X.cuda() * 0)[0]
    cross_ent_weight_to_in = {}
    for cross_ent_weight in [1]:  # 100
        print(f"Cross entropy weight {cross_ent_weight}")
        z_class_specific = z_zero.repeat(2, 1).clone().detach_().requires_grad_()
        # z_class_specific = per_class_means.clone().detach_().requires_grad_()
        # z_class_specific = (0.75 * z_zero.repeat(4,1) + 0.25 * per_class_means).clone().detach_().requires_grad_()
        lr = 1e-1
        optim = th.optim.AdamW([z_class_specific], lr=lr, weight_decay=0)
        scheduler = th.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda i_step: min(1, (10 * i_step) / (n_epochs))
        )
        assert not hasattr(net, "alphas")
        for i_epoch in range(n_epochs_visualization):
            if i_epoch > 0:
                inved, lp_per_class = net.invert(z_class_specific)
                lp_per_class = net(inved + th.randn_like(inved) * noise_factor)[1]
                lp_wanted_class = th.diag(lp_per_class)
                _, lp_z_per_class = net.module(
                    z_class_specific, fixed=dict(y=None, sum_dims=True)
                )
                cross_ent_per_example = th.nn.functional.cross_entropy(
                    lp_z_per_class,
                    th.tensor([0, 1], device="cuda"),
                    reduction="none",
                )
                cross_ent = th.mean(cross_ent_per_example)
                # only optimize nll where class already correct
                correct_mask = th.argmax(lp_z_per_class.squeeze(), dim=-1) == th.tensor(
                    [0, 1], device="cuda"
                )

                nll = -th.mean(lp_wanted_class * correct_mask.detach())
                bpd = (nll) / (
                    np.prod(inved.shape[1:]) * (n_real_chans / n_chans) * np.log(2)
                )
                loss = (1 / (cross_ent_weight + 1)) * bpd + (
                    cross_ent_weight / (cross_ent_weight + 1)
                ) * cross_ent
                optim.zero_grad()
                loss.backward()
                optim.step()
                optim.zero_grad()
                scheduler.step()

        in_specific = to_numpy(net.invert(z_class_specific)[0][:].squeeze())
        cross_ent_weight_to_in[cross_ent_weight] = in_specific
        seaborn.set_palette("colorblind")
        seaborn.set_style("darkgrid")

        tight_tuh_positions = [
            ["", "", "FP1", "", "FP2", "", ""],
            [
                "",
                "F7",
                "F3",
                "FZ",
                "F4",
                "F8",
                "",
            ],
            ["A1", "T3", "C3", "CZ", "C4", "T4", "A2"],
            ["", "T5", "P3", "Pz", "P4", "T6", ""],
            ["", "", "O1", "", "O2", "", ""],
        ]
        last_sensor_name = "O2"
        tight_positions = tight_tuh_positions

        ch_names = [
            "A1",
            "A2",
            "C3",
            "C4",
            "CZ",
            "F3",
            "F4",
            "F7",
            "F8",
            "FP1",
            "FP2",
            "FZ",
            "O1",
            "O2",
            "P3",
            "P4",
            "PZ",
            "T3",
            "T4",
            "T5",
            "T6",
        ]

        for cross_ent_weight in cross_ent_weight_to_in:
            in_specific = cross_ent_weight_to_in[cross_ent_weight]
            fig = plot_head_signals_tight(
                in_specific.transpose(1, 2, 0) * 30,
                sensor_names=ch_names,
                plot_args=dict(alpha=0.7),
                figsize=(20, 12),
                sharex=False,
                sharey=False,
                sensor_map=tight_positions,
            )

            last_ax = [
                ax for ax in fig.get_axes() if ax.get_title() == last_sensor_name
            ][0]
            last_ax.set_xlabel("Time [ms]")
            for ax in fig.get_axes():
                ax.set_xticks(range(0, 129, 16))
                ax.set_xticklabels([])
            last_ax.set_xticklabels(
                (np.arange(0, 129, 16) * 1000 / 64).astype(np.int32),
                rotation=45,
                fontsize=12,
            )
            last_ax.set_ylabel("Amplitude [μV]")
            fig.suptitle("Per-Class Maximum Input", fontsize=22)
            abs_max = np.ceil(np.abs(in_specific * 30).max() * 1.2)
            ymin, ymax = -abs_max, +abs_max
            for ax in fig.get_axes():
                ax.set_ylim(ymin, ymax)
            for ax in fig.get_axes():
                ax.yaxis.set_major_locator(AutoLocator())
                if ax.get_title() != last_sensor_name:
                    ax.set_yticklabels([])

            last_ax.legend(class_names, bbox_to_anchor=[1, 0.0, 1, 1])

    plt.savefig(
        os.path.join(output_dir, "class-prototypes.png"), bbox_inches="tight", dpi=300
    )

    return results
