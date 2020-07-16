import logging
import os.path
import os
from itertools import islice
import sys

import numpy as np
import torch as th
from braindecode.util import set_random_seeds
from skorch.utils import to_numpy
from tensorboardX import SummaryWriter

from invertible.gaussian import get_gaussian_log_probs
from invertible.init import init_all_modules
from invertible.models.glow import create_glow_model
from invertible.distribution import NClassIndependentDist, PerDimWeightedMix
from invertible.graph import Node
from invertible.lists import ApplyToList
from invertible.models.glow import convert_glow_to_pre_dist_model
from invglow.datasets import load_train_test

log = logging.getLogger(__name__)



def run_exp(
        first_n,
        lr,
        weight_decay,
        cross_ent_weight,
        batch_size,
        np_th_seed,
        debug,
        n_epochs,
        n_mixes,
        output_dir):
    hparams = {k:v for k,v in locals().items() if v is not None}
    noise_factor = 1/256.0
    if debug:
        first_n = 512
        batch_size = 10
        n_epochs = 5
    set_random_seeds(np_th_seed, True)

    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    model = create_glow_model(
        hidden_channels=512,
        K=32,
        L=3,
        flow_permutation='invconv',
        flow_coupling='additive',
        LU_decomposed=True,
        n_chans=3,
        block_type='conv',
        use_act_norm=True
    )
    state_dict = th.load('/home/schirrmr/data/exps/invertible/additive/7/state_dicts_model_250.pth')
    for key in state_dict.keys():
        if 'loc' in key or 'log_scale' in key:
            state_dict[key].squeeze_()

    model.load_state_dict(state_dict)
    pre_dist_model = convert_glow_to_pre_dist_model(model, as_list=True)
    del model
    init_dist_std = 1e-1
    dist0 = PerDimWeightedMix(10, n_mixes=n_mixes, n_dims=3072 // 2,
                              optimize_mean=True, optimize_std=True, init_std=init_dist_std)
    dist1 = PerDimWeightedMix(10, n_mixes=n_mixes, n_dims=3072 // 4,
                              optimize_mean=True, optimize_std=True, init_std=init_dist_std)
    dist2 = PerDimWeightedMix(10, n_mixes=n_mixes, n_dims=3072 // 4,
                              optimize_mean=True, optimize_std=True, init_std=init_dist_std)

    # architecture plan:
    model = Node(pre_dist_model, ApplyToList(dist0, dist1, dist2))
    net = model.cuda()
    init_all_modules(net, None)


    train_loader, valid_loader = load_train_test(
        'cifar10',
        shuffle_train=True,
        drop_last_train=True,
        batch_size=batch_size,
        eval_batch_size=256,
        n_workers=8,
        first_n=first_n,
        augment=True,
        exclude_cifar_from_tiny=False,
    )

    optim = th.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for i_epoch in range(n_epochs + 1):
        if i_epoch > 0:
            for X, y in train_loader:
                y = y.cuda()
                noise = th.rand_like(X) * 1 / 256.0
                noised = X + noise
                _, lp = net(noised.cuda(), fixed=dict(y=None))
                cross_ent = th.nn.functional.cross_entropy(
                    lp, y.argmax(dim=1), )
                nll = -th.mean(th.sum(lp * y, dim=1))
                loss = cross_ent * cross_ent_weight + nll  # cross_ent*10
                optim.zero_grad()
                loss.backward()
                optim.step()
                optim.zero_grad()
                del y, noise, noised, lp, cross_ent, nll, loss

        print(i_epoch)
        results = {}
        with th.no_grad():
            for name, loader in (('Train', train_loader), ('Valid', valid_loader)):
                all_lps = []
                all_corrects = []
                for X, y in loader:
                    y = y.cuda()
                    # First with noise to get nll for bpd,
                    # then without noise for accâuracy
                    noise = th.rand_like(X) * 1 / 256.0
                    noised = X + noise
                    noise_log_prob = np.log(256) * np.prod(X.shape[1:])
                    z, lp = net(noised.cuda())
                    lps = to_numpy(th.sum(lp * y, dim=1) - noise_log_prob)
                    all_lps.extend(lps)
                    z, lp = net(X.cuda() + (1 / (2 * 256.0)))
                    corrects = to_numpy(y.argmax(dim=1) == lp.argmax(dim=1))
                    all_corrects.extend(corrects)
                acc = np.mean(all_corrects)
                nll = -(np.mean(all_lps) / (np.prod(X.shape[1:]) * np.log(2)))
                print(f"{name} NLL: {nll:.2f}")
                print(f"{name} Acc: {acc:.1%}")
                results[f"{name.lower()}_nll"] = nll
                results[f"{name.lower()}_acc"] = acc
                writer.add_scalar(f"{name.lower()}_nll", nll, i_epoch)
                writer.add_scalar(f"{name.lower()}_acc", acc * 100, i_epoch)
                del noise, noised, z, lp, lps
        writer.flush()
        sys.stdout.flush()
        if not debug:
            dict_path = os.path.join(output_dir, "model_dict.th")
            th.save(net.state_dict(), open(dict_path, 'wb'))
            model_path = os.path.join(output_dir, "model.th")
            th.save(net, open(model_path, 'wb'))

    return results