from copy import deepcopy
import os.path
import logging

import numpy as np
import torch as th
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc
from braindecode.datautil.preprocess import exponential_moving_demean, preprocess
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.util import set_random_seeds
from skorch.utils import to_numpy
from torch.utils.data import Subset
from tensorboardX import SummaryWriter

from invertible.datautil import PreprocessedLoader
from invertible.gaussian import get_gaussian_log_probs
from invertible.init import init_all_modules
from invertible.view_as import flatten_2d
from invertibleeeg.models.glow import  create_eeg_glow_down, create_eeg_glow_up
from invertible.noise import GaussianNoise

log = logging.getLogger(__name__)

def load_tuh(n_subjects):
    dataset = TUHAbnormal(
        '/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/',
        subject_ids=range(n_subjects),
        preload=True)
    return dataset


def preprocess_tuh(dataset, sfreq):
    # making a copy just to be able to rerun preprocessing without
    # waiting later
    dataset = deepcopy(dataset)

    # Define preprocessing steps
    preprocessors = [
        # convert from volt to microvolt, directly modifying the numpy array
        MNEPreproc(fn='pick_channels', ch_names=['EEG T3-REF', ], ordered=True),
        NumpyPreproc(fn=lambda x: x * 1e6),
        NumpyPreproc(fn=lambda x: np.clip(x, -800, 800)),
        NumpyPreproc(fn=lambda x: x / 10),
        MNEPreproc(fn='resample', sfreq=sfreq),
        # keep only EEG sensors
        NumpyPreproc(fn=exponential_moving_demean, init_block_size=sfreq * 10, factor_new=1 / (sfreq * 5)),
    ]

    # Preprocess the data
    preprocess(dataset, preprocessors)
    return dataset

def run_exp(
        n_subjects,
        lr,
        weight_decay,
        np_th_seed,
        debug,
        n_epochs,
        n_virtual_chans,
        hidden_channels,
        n_blocks_up,
        n_blocks_down,
        n_mixes,
        splitter_last,
        init_perm_to_identity,
        output_dir):
    sfreq = 32
    n_real_chans = 1
    kernel_length = 9
    noise_factor = 5e-3
    batch_size = 64
    if debug:
        n_subjects = 10
        batch_size = 10
        n_epochs = 5
    hparams = locals()
    set_random_seeds(np_th_seed, True)

    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    if n_subjects != 1000:
        log.info("Loading data")
        dataset = TUHAbnormal(
            '/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/',
            subject_ids=range(n_subjects),
            preload=False)
        preproced = preprocess_tuh(dataset, sfreq=sfreq)
        # Next, extract the 4-second trials from the dataset.
        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.
    else:
        log.info("Loading preproced data")
        preproced = np.load('/home/schirrmr/data/preproced-tuh/t3-32hz-demeaned.pkl.npy', allow_pickle=True).item()

    windows_dataset = create_fixed_length_windows(
        preproced,
        start_offset_samples=60 * 50,
        stop_offset_samples=60 * 50 + 128 * 3,
        preload=True,
        window_size_samples=128,
        window_stride_samples=64,
        drop_last_window=True,
    )

    whole_train_set = windows_dataset.split('session')['train']
    n_split = int(np.round(len(whole_train_set) * 0.75))
    valid_set = Subset(whole_train_set, range(n_split, len(whole_train_set)))
    train_set = Subset(whole_train_set, range(0, n_split))
    train_loader = th.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)
    valid_loader = th.utils.data.DataLoader(
        valid_set,
        batch_size=len(valid_set),
        shuffle=False,
        num_workers=0)
    preproced_loader = PreprocessedLoader(train_loader, GaussianNoise(noise_factor), False)

    n_chans = n_real_chans + n_virtual_chans

    n_up_block = create_eeg_glow_up(
        n_chans=n_chans, hidden_channels=hidden_channels, kernel_length=kernel_length,
        splitter_first='subsample', splitter_last=splitter_last, n_blocks=n_blocks_up)

    net = create_eeg_glow_down(
        n_up_block,
        n_chans=n_chans, hidden_channels=hidden_channels, kernel_length=kernel_length,
        n_mixes=n_mixes, n_blocks=n_blocks_down, init_dist_std=1e-1)
    if init_perm_to_identity:
        for m in net.modules():
            if hasattr(m, 'use_lu'):
                m.reset_to_identity()
    net = net.cuda()

    def augment_virtual_chans(x, noise_factor, n_virtual_chans):
        if n_virtual_chans > 0:
            virtual_chans = th.randn(
                x.shape[0], n_virtual_chans, x.shape[2], device=x.device).type_as(x) * noise_factor
            return th.cat((x, virtual_chans), dim=1)
        else:
            return x

    init_all_modules(net, th.cat(
        [augment_virtual_chans(x[:, :, 32:32 + 64], noise_factor, n_virtual_chans)
         for x, y, i in preproced_loader], dim=0).cuda())



    optim = th.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    rng = np.random.RandomState(np_th_seed)
    for i_epoch in range(n_epochs + 1):
        if i_epoch > 0:
            for X_th, y, _ in train_loader:
                i_start_time = rng.randint(0,33)
                X_th = X_th[:,:,i_start_time:i_start_time+64]
                # noise added after
                X_th = augment_virtual_chans(X_th, noise_factor=0, n_virtual_chans=n_virtual_chans)
                y_th = th.nn.functional.one_hot(y.type(th.int64), num_classes=2).cuda()
                noise = th.randn_like(X_th) * noise_factor
                noised = X_th + noise
                z, lp = net(noised.cuda(), fixed=dict(y=None))
                cross_ent = th.nn.functional.cross_entropy(
                    lp, y_th.argmax(dim=1), )
                nll = -th.mean(th.sum(lp * y_th, dim=1))
                loss = cross_ent * 10 + nll  # cross_ent*10
                loss.backward()
                optim.step()
                optim.zero_grad()
        if (i_epoch % max(n_epochs // 100, 1) == 0) or (i_epoch == n_epochs):
            print(i_epoch)
            results = {}
            with th.no_grad():
                for name, loader in (('Train', train_loader), ('Valid', valid_loader)):
                    all_lps = []
                    all_corrects = []
                    for X_th, y, _ in loader:
                        X_th = X_th[:,:,32:32+64]
                        X_th = augment_virtual_chans(X_th, noise_factor=0, n_virtual_chans=n_virtual_chans)
                        y_th = th.nn.functional.one_hot(y.type(th.int64), num_classes=2).cuda()
                        # First with noise to get nll for bpd,
                        # then without noise for accuracy
                        noise = th.randn_like(X_th) * noise_factor
                        noised = X_th + noise
                        noise_log_prob = get_gaussian_log_probs(
                            th.zeros_like(X_th[0]).view(-1), th.log(th.zeros_like(X_th[0]) + noise_factor).view(-1),
                            flatten_2d(noise))
                        z, lp = net(noised.cuda())
                        lps = to_numpy(th.sum(lp * y_th, dim=1) - noise_log_prob.cuda())
                        all_lps.extend(lps)
                        z, lp = net(X_th.cuda())
                        corrects = to_numpy(y.cuda() == lp.argmax(dim=1))
                        all_corrects.extend(corrects)
                    acc = np.mean(all_corrects)
                    nll = -(np.mean(all_lps) / (np.prod(X_th.shape[1:]) * np.log(2) * (n_real_chans/n_chans)))
                    print(f"{name} NLL: {nll:.2f}")
                    print(f"{name} Acc: {acc:.1%}")
                    results[f"{name.lower()}_nll"] = nll
                    results[f"{name.lower()}_acc"] = acc
                    writer.add_scalar(f"{name.lower()}_nll", nll, i_epoch)
                    writer.add_scalar(f"{name.lower()}_acc", acc * 100, i_epoch)
            writer.flush()
            if not debug:
                dict_path = os.path.join(output_dir, "model_dict.th")
                th.save(net.state_dict(), open(dict_path, 'wb'))
                model_path = os.path.join(output_dir, "model.th")
                th.save(net, open(model_path, 'wb'))
    return results