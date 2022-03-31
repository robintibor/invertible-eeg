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
            "save_folder": "/work/dlclarge1/schirrmr-renormalized-flows/exps/bcic-iv-2a/fix-virtual-chans/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    train_params = dictlistprod(
        {
            "n_epochs": [800],
            "train_deep4_instead": [True],
        }
    )

    data_params = dictlistprod(
        {
            "subject_id": range(1,10),
            "split_valid_off_train": [False],
            "class_names": [('left_hand', 'right_hand', 'feet', 'tongue')],
        }
    )

    random_params = dictlistprod(
        {
            "np_th_seed": range(0,1),
        }
    )


    optim_params = dictlistprod({
        "nll_loss_factor": [0]
    })

    model_params = dictlistprod({
        "amp_phase_at_end": [False],
        "n_virtual_chans": [0,2],
        "n_stages": [4],
        "splitter_last": [None],
        "n_times": [64],
    })

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            data_params,
            train_params,
            random_params,
            debug_params,
            optim_params,
            model_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    subject_id,
    n_epochs,
    class_names,
    np_th_seed,
    split_valid_off_train,
    nll_loss_factor,
    debug,
    train_deep4_instead,
    amp_phase_at_end,
    n_virtual_chans,
    n_stages,
    splitter_last,
    n_times,
):
    if debug:
        n_epochs = 3
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
    from invertibleeeg.experiments.bcic_iv_2a.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
