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
            "save_folder": "/work/dlclarge1/schirrmr-renormalized-flows/exps/bcic-iv-2a-nas-all-sub-each-fold-new-seeds/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    train_params = dictlistprod(
        {
            "n_epochs": [50],
        }
    )

    data_params = dictlistprod(
        {
            "subject_id": [None],
            "all_subjects_in_each_fold": [True],
        }
    )

    random_params = dictlistprod(
        {
        }
    )


    optim_params = dictlistprod({
    })

    model_params = dictlistprod({
        "amplitude_phase_at_end": [False],
    })

    search_params = [{
        'max_hours': 1,
        'n_population': 15,
    }]

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            data_params,
            train_params,
            random_params,
            debug_params,
            optim_params,
            model_params,
            search_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    params['np_th_seed'] = rng.randint(0, 2**32)
    return params


def run(
    ex,
    debug,
    np_th_seed,
    max_hours,
    n_population,
    subject_id,
    n_epochs,
    amplitude_phase_at_end,
    all_subjects_in_each_fold,
):
    if debug:
        n_population = 2
        n_epochs = 10
    kwargs = locals()
    kwargs.pop("ex")
    if not debug:
        log.setLevel("INFO")
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    worker_folder = os.path.join(*os.path.split(output_dir)[:-1], 'worker')
    if debug:
        worker_folder = os.path.join(*os.path.split(output_dir)[:-1], 'debug-worker')
    kwargs['worker_folder'] = worker_folder

    th.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    start_time = time.time()
    ex.info["finished"] = False
    from invertibleeeg.experiments.nas import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
