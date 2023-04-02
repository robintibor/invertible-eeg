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
            "save_folder": "/work/dlclarge1/schirrmr-renormalized-flows/exps/tuh-abnormal-all-data-2-class/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    train_params = dictlistprod(
        {
            "n_epochs": [3],  # [20],
            "fixed_lr": [None],
            "fixed_batch_size": [None],
            "alpha_lr": [1e-2],#[1e-2],
            "dist_lr": [1e-3],
            "n_times_crop": [128],
            "channel_drop_p": [0.],
            "n_eval_crops": [3],
            "scheduler": ["cosine"],
        }
    )

    data_params = dictlistprod(
        {
            "subject_id": [[1,2,3,4,5,6,7]],#4
            "all_subjects_in_each_fold": [True],
            "n_times_train": [144],
            "n_times_eval": [144],
            "sfreq": [64],#32
            "trial_start_offset_sec": [-0.5],
            "split_valid_off_train": [True],
            "low_cut_hz": [4],
            "high_cut_hz": [None],
            "exponential_standardize": [False],
            "class_names": [
                #["left_hand", "right_hand", "feet", "tongue"]
                ["left_hand", "right_hand", "feet", "rest"]
            ],
            "dataset_name": ["tuh"],
            "hgd_sensors": ["C"],
            "n_tuh_recordings": [None],
        }
    )

    random_params = dictlistprod(
        {
            "seed_offset": [0],
        }
    )


    variants = dictlistprod(
        # {
        #     "just_train_deep4": [True],
        #     "nll_loss_factor": [0],#3e-2],  # [0],#[3e-2],
        #     "max_n_changes": [0],
        #     "max_n_deletions": [0],
        #      "class_prob_masked": [False],  # [True],
        # },
        # {
        #     "just_train_deep4": [False],
        #     "nll_loss_factor": [1e-5],#3e-2],  # [0],#[3e-2],
        #     "max_n_changes": [0],
        #     "max_n_deletions": [0],
        #      "class_prob_masked": [True],  # [True],
        # },
        {
            "just_train_deep4": [False],
            "nll_loss_factor": [1],#]3e-1],#3e-2],  # [0],#[3e-2],
            "max_n_changes": [2],
            "max_n_deletions": [1],
            "class_prob_masked": [False],#[True],  # [True],
        },
    )

    model_params = dictlistprod(
        {
            "amplitude_phase_at_end": [False],
            "dist_module_choices": [["maskedmix"]],#maskedindependent"perdimweightedmix",
            "n_virtual_chans": [0],
            "linear_glow_clf": [False],
            "splitter_name": ["subsample"],  # "haar"
        }
    )

    search_params = [
        {
            "min_improve_fraction": 0.1,
            "max_hours": 2,#0.25,#0.25,
            "n_start_population": 50,
            "n_alive_population": 300,
            "search_by": "valid_mis",
            "mutate_optim_params": False,
        }
    ]

    searchspace_params = dictlistprod(
        {
            "searchspace": ["downsample_anywhere"],
            "include_splitter": [False],
            # "coupling_block", #"act_norm",
            "included_blocks": [
                ["coupling_block", "permute", "act_norm"],#"act_norm"#"permute",deep4_coupling
            ],
            "limit_n_downsample": [None],
            "min_n_downsample": [0],
        },
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            data_params,
            train_params,
            random_params,
            debug_params,
            model_params,
            search_params,
            searchspace_params,
            variants,
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
    np_th_seed,
    max_hours,
    n_start_population,
    n_alive_population,
    subject_id,
    n_epochs,
    amplitude_phase_at_end,
    all_subjects_in_each_fold,
    n_times_train,
    n_times_eval,
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
    dist_module_choices,
    scheduler,
    trial_start_offset_sec,
    n_times_crop,
    just_train_deep4,
    n_virtual_chans,
    split_valid_off_train,
    linear_glow_clf,
    splitter_name,
    low_cut_hz,
    high_cut_hz,
    exponential_standardize,
    included_blocks,
    limit_n_downsample,
    sfreq,
    mutate_optim_params,
    max_n_deletions,
    channel_drop_p,
    n_eval_crops,
    dataset_name,
    min_n_downsample,
    hgd_sensors,
    min_improve_fraction,
    n_tuh_recordings,
    dist_lr,
    n_virtual_classes,
):
    if debug:
        n_start_population = 2
        n_alive_population = 2
        n_epochs = 2
        n_tuh_recordings = 300
    kwargs = locals()
    kwargs.pop("ex")
    if not debug:
        log.setLevel("INFO")
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    # todo: by default use full subfolder of current exp folder,
    # and allow specifying the folder if you want to run further
    worker_folder = os.path.join(*os.path.split(output_dir)[:-1], "worker")
    if debug:
        worker_folder = os.path.join(*os.path.split(output_dir)[:-1], "debug-worker")
    kwargs["worker_folder"] = worker_folder

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
