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
    })

    train_and_model_params = dictlistprod(
        {
            "dist_name": ["weighteddimgaussianmix", "weightedgaussianmix"],
            'n_epochs': [25],
            'n_restarts': [10,40],#[10,20,40],
            'n_blocks': [4],#[2,4,8],
            'n_stages': [3],#3,4
            'saved_model_folder': [None],#'/work/dlclarge1/schirrmr-renormalized-flows/exps/manual/tuh/45']
            'affine_or_additive': ['additive'],
            "n_times_train_eval": [128],
        })
    random_params = dictlistprod(
        {
            "np_th_seed": range(3),
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
