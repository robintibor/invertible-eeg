from tqdm.autonotebook import tqdm
import torch as th
import numpy as np
from invertible.util import th_to_np


def compute_z_lp(loader, net, dist, n_crops):
    noise_factor = 5e-3
    zs = []
    lps = []
    lps_per_mix_and_dim = []

    n_crops = 10
    crop_Xs = []
    ys = []
    for X, y, i in tqdm(loader):
        for i_crop_start in np.linspace(0, X.shape[2] - 128, n_crops):
            i_crop_start = np.int64(np.round(i_crop_start))
            crop_X = X[:, :, i_crop_start : i_crop_start + 128].cuda()
            with th.no_grad():
                z, _ = net(crop_X + th.randn_like(crop_X) * noise_factor)
                lp = dist(
                    z,
                )[1]
                lp_per_mix_and_dim = dist(
                    z,
                    fixed=dict(
                        mixture_reduce="none", reduce_overall_mix="none", sum_dims=False
                    ),
                )[1]
            zs.append(th_to_np(z))
            lps.append(th_to_np(lp))
            lps_per_mix_and_dim.append(th_to_np(lp_per_mix_and_dim))
            crop_Xs.append(th_to_np(crop_X))
            ys.append(th_to_np(y))
    lps_per_mix_and_dim = np.concatenate(lps_per_mix_and_dim)
    lps_topo = lps_per_mix_and_dim.reshape(
        lps_per_mix_and_dim.shape[0], lps_per_mix_and_dim.shape[1], 21, -1
    )
    all_crop_X = np.concatenate(crop_Xs)
    return dict(
        z=np.concatenate(zs),
        y=np.concatenate(ys),
        lps_per_mix_and_dim=lps_per_mix_and_dim,
        lps_topo=lps_topo,
        crop_X=all_crop_X,
    )


def get_per_example(a, n_crops, batch_size):
    parts_per_example = []
    for i_batch in range(int(np.ceil(len(a) / (n_crops * batch_size)))):
        part = a[i_batch * n_crops * batch_size : (i_batch + 1) * n_crops * batch_size]
        part_per_crop_per_example = part.reshape(10, -1, *part.shape[1:])
        parts_per_example.extend(part_per_crop_per_example.swapaxes(0, 1))
    return np.stack(parts_per_example)
