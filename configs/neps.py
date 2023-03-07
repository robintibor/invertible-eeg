import os

from braindecode.util import set_random_seeds

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
            "save_folder": "/work/dlclarge1/schirrmr-renormalized-flows/exps/neps/hgd-subject-1-7-no-fidelity-raise-err/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    data_params = dictlistprod(
        {
            "subject_id": [[1,2,3,4,5,6,7]],
            "all_subjects_in_each_fold": [True],
            "n_times_train": [144],
            "n_times_eval": [144],
            "sfreq": [64],  # 32
            "trial_start_offset_sec": [-0.5],
            "split_valid_off_train": [True],
            "low_cut_hz": [4],
            "high_cut_hz": [None],
            "exponential_standardize": [False],
            "class_names": [["left_hand", "right_hand", "feet", "rest"]],
            "dataset_name": ["hgd"],
            "hgd_sensors": ["C"],
        }
    )

    search_params = [
        {
            "max_hours": 15 * 1.5,  # 0.25,
            "epochs_as_fidelity": False,
            "n_max_epochs": 10,
            "seed_offset": 0,
            "ignore_errors": True,
        }
    ]

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            data_params,
            debug_params,
            search_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    max_seed = 2**32
    params["np_th_seed"] = (rng.randint(0, max_seed) + params["seed_offset"]) % max_seed
    params.pop("seed_offset")
    return params


def run(
    ex,
    debug,
    max_hours,
    epochs_as_fidelity,
    class_names,
    trial_start_offset_sec,
    subject_id,
    split_valid_off_train,
    sfreq,
    n_times_train,
    n_times_eval,
    all_subjects_in_each_fold,
    exponential_standardize,
    low_cut_hz,
    high_cut_hz,
    hgd_sensors,
    dataset_name,
    n_max_epochs,
    np_th_seed,
    ignore_errors,
):
    if debug:
        n_epochs = 2
    kwargs = locals()
    kwargs.pop("ex")
    if not debug:
        log.setLevel("INFO")
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    # todo: by default use full subfolder of current exp folder,
    # and allow specifying the folder if you want to run further
    neps_output_dir = os.path.join(*os.path.split(output_dir)[:-1], "neps-results")
    if debug:
        neps_output_dir = os.path.join(
            *os.path.split(output_dir)[:-1], "debug-neps-results"
        )
    kwargs["neps_output_dir"] = neps_output_dir

    th.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    start_time = time.time()
    ex.info["finished"] = False
    from invertibleeeg.experiments.neps import run_exp

    set_random_seeds(np_th_seed, True)
    kwargs.pop("np_th_seed")
    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
