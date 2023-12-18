import os


os.sys.path.insert(0, "/home/schirrmr/code/invertible-public/")
os.sys.path.insert(0, "/home/schirrmr/code/invertible-eeg/")
import time
import logging

from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)

import torch as th
import torch.backends.cudnn as cudnn

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s")


log = logging.getLogger(__name__)
log.setLevel("INFO")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
            "save_folder": "/work/dlclarge1/schirrmr-renormalized-flows/exps/manual/tuh-clipped/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    data_params = dictlistprod({
        "in_clip_val": [7],
        "high_cut_hz": [0.5],
    })

    train_and_model_params = dictlistprod(
        {
            "dist_name": ["nclassindependent"],#["nclassindependent"],#"perclasshierarchical"],#
            "n_class_independent_dims": [None],#1,2,4,8,16,32,64,128],
            'n_epochs': [25],
            'n_restarts': [10,40,],#[10,20,40],
            'n_blocks': [4],#4],#[2,4,8],
            'n_stages': [3],#3],#3,4
            'saved_model_folder': [None],#'/work/dlclarge1/schirrmr-renormalized-flows/exps/manual/tuh/45']
            'affine_or_additive': ["additive", "affine",],#, 'affine'],
            "n_times_train_eval": [128],
            "n_mixes": [64],
            "n_overall_mixes": [16],
            "n_dim_mixes": [16],
            "reduce_per_dim": ["logsumexp"],
            "reduce_overall_mix": ["logsumexp"],
            "init_dist_weight_std": [0.1],
            "init_dist_mean_std": [0.1],
            "init_dist_std_std": [0.1],
            "nll_loss_factor": [1],#[1],#1
            "temperature_init": [200],
            "optimize_std": [False],
            "first_conv_class_name": ["conv1d",],#"twostepspatialtemporalconvfixed",
                                      #"twostepspatialtemporalconvmerged"],#["separate_temporal_channel_conv",],# "twostepspatialtemporalconv"]
            "lowpass_for_model": [True],
        })
    random_params = dictlistprod(
        {
            "np_th_seed": range(2),
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [save_params, data_params, debug_params, train_and_model_params, random_params]
    )

    return grid_params


def run(
    ex,
    debug,
    np_th_seed,
    n_epochs,
    n_restarts,
    n_blocks,
    n_stages,
    saved_model_folder,
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
    kwargs = locals()
    kwargs.pop("ex")
    if not debug:
        log.setLevel("INFO")
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    kwargs["output_dir"] = output_dir

    th.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    start_time = time.time()
    ex.info["finished"] = False
    from invertibleeeg.experiments.manual_tuh import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
