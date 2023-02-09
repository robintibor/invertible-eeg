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
            "save_folder": "/work/dlclarge1/schirrmr-renormalized-flows/exps/bcic-iv-2a-crop-128-from-144/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    train_params = dictlistprod(
        {
            "n_epochs": [20],
            "fixed_lr": [None],
            "fixed_batch_size": [None],
            "nll_loss_factor": [3e-2],
            "alpha_lr": [1e-2],
            "n_times_crop": [128],
        }
    )

    data_params = dictlistprod(
        {
            "subject_id": [None],
            "all_subjects_in_each_fold": [True],
            "n_times": [144],
            "class_names": [["left_hand", "right_hand", "feet","tongue"]],# "tongue"]],
            "trial_start_offset_sec": [-0.5],
        }
    )

    random_params = dictlistprod(
        {
            "seed_offset": [0],
        }
    )


    optim_params = dictlistprod({
        "scheduler": ["cosine"],
    })

    model_params = dictlistprod({
        "amplitude_phase_at_end": [False],
        "class_prob_masked": [True],
        "sample_dist_module": [True],
    })

    search_params = [{
        'max_hours': 0.25,
        'n_start_population': 50,
        'n_alive_population': 300,
        "max_n_changes": 3,
        "search_by": "valid_mis",
    }]

    searchspace_params = dictlistprod(
        {
            "searchspace": ["downsample_anywhere"],
            "include_splitter": [False],
        }
    )
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
            searchspace_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    max_seed = 2**32
    params['np_th_seed'] = (rng.randint(0, max_seed) + params['seed_offset']) % max_seed
    params.pop('seed_offset')
    return params


def run(
    ex,
    debug,
    np_th_seed,
    max_hours,
    n_start_population,
    n_alive_population,
    subject_id,
    n_epochs,
    amplitude_phase_at_end,
    all_subjects_in_each_fold,
    n_times,
    max_n_changes,
    fixed_lr,
    searchspace,
    include_splitter,
    class_names,
    fixed_batch_size,
    class_prob_masked,
    nll_loss_factor,
    search_by,
    alpha_lr,
    sample_dist_module,
    scheduler,
    trial_start_offset_sec,
    n_times_crop,
):
    if debug:
        n_start_population = 2
        n_alive_population = 2
        n_epochs = 2
    kwargs = locals()
    kwargs.pop("ex")
    if not debug:
        log.setLevel("INFO")
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    # todo: by default use full subfolder of current exp folder,
    # and allow specifying the folder if you want to run further
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
