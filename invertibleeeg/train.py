import functools
import logging
from copy import deepcopy
from itertools import islice

import numpy as np
import torch
import torch as th
from torch import nn
from braindecode.models import Deep4Net
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from invertible.gaussian import get_gaussian_log_probs
from invertible.init import init_all_modules
from invertible.optim import grads_all_finite
from invertible.util import weighted_sum
from invertible.view_as import flatten_2d
from invertibleeeg.datasets import load_train_valid_bcic_iv_2a
from invertibleeeg.models.glow import create_eeg_glow, create_eeg_glow_multi_stage
from skorch.utils import to_numpy

# from tensorboardX import SummaryWriter
from tqdm.autonotebook import trange


def add_virtual_chans(x, n_virtual_chans):
    virtual_x = th.zeros(x.shape[0], n_virtual_chans, *x.shape[2:], device=x.device)
    x_with_virtual = th.cat((x, virtual_x), dim=1)
    return x_with_virtual


def train_glow(
    net,
    class_prob_masked,
    nll_loss_factor,
    noise_factor,
    train_set,
    valid_set,
    test_set,
    n_epochs,
    n_virtual_chans,
    n_classes,
    n_times_train,
    np_th_seed,
    scheduler,
    batch_size,
    optim_params_per_param,
    with_tqdm,
    n_times_eval,
    channel_drop_p,
    n_times_crop,
    n_eval_crops,
    use_temperature,
):
    assert n_times_crop == n_times_eval == n_times_train
    assert not class_prob_masked
    if not with_tqdm:
        trange = range
    else:
        from tqdm.autonotebook import trange
    net.train()

    n_real_chans = train_set[0][0].shape[0]
    n_input_times = train_set[0][0].shape[1]
    n_chans = n_real_chans + n_virtual_chans

    train_loader = th.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    valid_loader = th.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = th.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    if any(
        [
            hasattr(m, "initialize_this_forward") and (not m.initialized)
            for m in net.modules()
        ]
    ):
        i_start_center_crop = (n_input_times // 2) - (n_times_train // 2)
        init_all_modules(
            net,
            th.cat(
                [
                    add_virtual_chans(
                        x[
                            :,
                            :,
                            i_start_center_crop : i_start_center_crop + n_times_train,
                        ],
                        n_virtual_chans,
                    )
                    + th.randn(
                        x.shape[0],
                        n_chans,
                        n_times_train,
                    )
                    * noise_factor
                    for x, y, i in islice(train_loader, 10)
                ],
                dim=0,
            ).cuda(),
            init_only_uninitialized=True,
        )

    for n, p in net.named_parameters():
        assert p in optim_params_per_param, f"Parameter {n} does not have optim params"
    net_parameters = list(net.parameters())
    for p in optim_params_per_param:
        assert any(
            [p is param for param in net_parameters]
        ), "every optimized parameter should be a parameter of the network"
    param_dicts = [
        dict(params=[p], **optim_params_per_param[p]) for p in net.parameters()
    ]
    # if not class_prob_masked:
    #    param_dicts = [dict(params=list(net.parameters()), lr=start_lr, weight_decay=5e-5)]
    # else:
    # param_dicts = [dict(params=[p for p in  net.parameters() if p is not net.alphas], lr=5e-4, weight_decay=5e-5)]
    # param_dicts.append(dict(params=[net.alphas], lr=5e-2))

    optim = th.optim.AdamW(param_dicts)
    this_scheduler = scheduler(optim)

    rng = np.random.RandomState(np_th_seed)

    i_epoch = 0
    for i_epoch in trange(n_epochs):
        net.train()
        for X_th, y, _ in train_loader:
            i_start_time = rng.randint(0, X_th.shape[2] - n_times_train + 1)
            X_th = X_th[:, :, i_start_time : i_start_time + n_times_train]
            if channel_drop_p > 0:
                mask = th.bernoulli(
                    th.zeros_like(X_th[:, :, :1]) + (1 - channel_drop_p)
                )
                X_th = X_th * mask
            X_th = add_virtual_chans(X_th, n_virtual_chans)

            # noise added after
            y_th = th.nn.functional.one_hot(
                y.type(th.int64), num_classes=n_classes
            ).cuda()
            noise = th.randn_like(X_th) * noise_factor
            noised = X_th + noise
            z, lp_per_class = net(noised.cuda(), fixed=dict(y=None, sum_dims=True))
            nll = -th.mean(th.sum(lp_per_class * y_th, dim=1))
            if use_temperature:
                lp_for_clf = lp_per_class / th.exp(net.temperature)
            else:
                lp_for_clf = lp_per_class
            cross_ent = th.nn.functional.cross_entropy(
                lp_for_clf,
                y_th.argmax(dim=1),
            )
            # ignoring noise log probs, just for training
            pseudo_bpd = nll / (n_real_chans * n_times_crop * np.log(2))
            if nll_loss_factor < 1e10:
                loss = weighted_sum(1, 1, cross_ent, nll_loss_factor, pseudo_bpd)
            else:
                loss = pseudo_bpd
            optim.zero_grad()
            loss.backward()
            if grads_all_finite:
                optim.step()
            optim.zero_grad()
            this_scheduler.step()
            # print(i_epoch)
            # for g in optim.param_groups:
            #     print("lr", g['lr'])
    # if (i_epoch % max(n_epochs // 10, 1) == 0) or (i_epoch == n_epochs):
    print(i_epoch)
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
                    crop_X_th = X_th[:, :, i_crop_start : i_crop_start + n_times_eval]
                    crop_X_th = add_virtual_chans(crop_X_th, n_virtual_chans)
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
                    lps = to_numpy(
                        th.sum(lp_per_class * y_th, dim=1) - noise_log_prob.cuda()
                    )
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
    return results


def evaluate_net(
    cropped_net,
    train_loader,
    valid_loader,
    test_loader,
    n_real_chans,
    n_eval_crops,
    noise_factor,
    n_times_crop,
    n_times_eval,
):
    assert n_times_crop == n_times_eval
    from tqdm.autonotebook import tqdm

    n_input_times = train_loader.dataset[0][0].shape[1]
    with th.no_grad():
        cropped_net.eval()
        results = {}
        for name, loader in (
            ("train", train_loader),
            ("valid", valid_loader),
            ("test", test_loader),
        ):
            all_lps = []
            all_corrects = []
            for X_th, y, _ in tqdm(loader):
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
                        crop_X_th = X_th[
                            :, :, i_crop_start : i_crop_start + n_times_eval
                        ]
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
                        z, lp_per_class = cropped_net(
                            noised.cuda(), fixed=dict(y=None, sum_dims=True)
                        )
                        lps = to_numpy(
                            lp_per_class.gather(index=(y * 1).cuda().unsqueeze(1), dim=1).squeeze(1)
                            - noise_log_prob.cuda()
                        )
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
        return results

