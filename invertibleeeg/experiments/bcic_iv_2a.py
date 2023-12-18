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

log = logging.getLogger(__name__)


def train_deep4(train_set, valid_set, n_epochs):
    cuda = torch.cuda.is_available()
    device = "cuda"

    n_classes = 4
    # Extract number of chans and time steps from dataset
    n_chans = train_set[0][0].shape[0]
    input_window_samples = train_set[0][0].shape[1]

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto",
    )
    model = Deep4Net(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto",
        pool_time_stride=2,
        pool_time_length=2,
        filter_time_length=7,
        filter_length_2=7,
        filter_length_3=7,
        filter_length_4=7,
    )
    # Send model to GPU
    if cuda:
        model.cuda()
    from skorch.callbacks import LRScheduler
    from skorch.helper import predefined_split

    from braindecode import EEGClassifier

    # These values we found good for shallow network:
    # lr = 0.0625 * 0.01
    # weight_decay = 0

    # For deep4 they should be:
    lr = 1 * 0.01
    weight_decay = 0.5 * 0.001

    batch_size = 64

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
        ],
        device=device,
    )
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf.fit(train_set, y=None, epochs=n_epochs)
    results_dict = deepcopy(clf.history[-1])
    _ = [
        results_dict.pop(k)
        for k in [
            "batches",
            "epoch",
            "train_batch_count",
            "valid_batch_count",
            "dur",
            "train_loss_best",
            "valid_loss_best",
            "train_accuracy_best",
            "valid_accuracy_best",
            "event_lr",
        ]
    ]
    return results_dict


def run_exp(
    subject_id,
    n_epochs,
    class_names,
    np_th_seed,
    split_valid_off_train,
    nll_loss_factor,
    debug,
    output_dir,
    train_deep4_instead,
    amp_phase_at_end,
    n_virtual_chans,
    n_stages,
    splitter_last,
    n_times,
):
    # hparams = {k: v for k, v in locals().items() if v is not None}
    # writer = SummaryWriter(output_dir)
    # writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    # writer.flush()

    n_blocks_up = 2
    n_blocks_down = 2
    hidden_channels = 32
    n_mixes = 8
    init_perm_to_identity = True
    kernel_length = 9
    affine_or_additive = "affine"

    noise_factor = 5e-3
    class_prob_masked = True
    init_perm_to_identity = True

    n_classes = len(class_names)

    set_random_seeds(np_th_seed, True)
    train_set, valid_set = load_train_valid_bcic_iv_2a(
        subject_id,
        class_names,
        split_valid_off_train,
        all_subjects_in_each_fold=True,
    )

    n_real_chans = train_set[0][0].shape[0]
    n_chans = n_real_chans + n_virtual_chans
    if train_deep4_instead:
        results = train_deep4(train_set, valid_set, n_epochs)

    else:
        net = create_eeg_glow_multi_stage(
            n_chans=n_chans,
            hidden_channels=hidden_channels,
            kernel_length=kernel_length,
            splitter_first="subsample",
            splitter_last=splitter_last,
            n_blocks=n_blocks_up,
            affine_or_additive=affine_or_additive,
            n_mixes=n_mixes,
            n_classes=n_classes,
            amp_phase_at_end=amp_phase_at_end,
            n_stages=n_stages,
            n_times=n_times,
        )

        if init_perm_to_identity:
            for m in net.modules():
                if hasattr(m, "use_lu"):
                    m.reset_to_identity()
        net = net.cuda()

        batch_size = 64
        optim_params_per_param = {
            p: dict(lr=1e-3, weight_decay=5e-5) for p in net.parameters()
        }
        if class_prob_masked:
            net.alphas = nn.Parameter(
                th.zeros(n_chans * n_times, device="cuda", requires_grad=True)
            )
            optim_params_per_param[net.alphas] = dict(lr=5e2, weight_decay=5e-5)

        results = train_glow(
            net=net,
            class_prob_masked=class_prob_masked,
            nll_loss_factor=nll_loss_factor,
            noise_factor=noise_factor,
            train_set=train_set,
            valid_set=valid_set,
            test_set=valid_set,
            n_epochs=n_epochs,
            n_virtual_chans=n_virtual_chans,
            n_classes=n_classes,
            np_th_seed=np_th_seed,
            n_times=n_times,
            scheduler=functools.partial(
                th.optim.lr_scheduler.CosineAnnealingLR,
                T_max=len(train_set) // batch_size,
            ),
            batch_size=batch_size,
            optim_params_per_param=optim_params_per_param,
        )

    # writer.close()
    return results
