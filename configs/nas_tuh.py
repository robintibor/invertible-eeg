# PARENTCONFIG: /home/schirrmr/code/invertible-eeg/configs/nas.py

import os

os.sys.path.insert(0, "/home/schirrmr/code/invertible-public/")
os.sys.path.insert(0, "/home/schirrmr/code/invertible-eeg/")
import time
import logging

from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
            "save_folder": "/work/dlclarge1/schirrmr-renormalized-flows/exps/tuh-abnormal-only-dist-weighted-dim-gaussian-32-mix/",
        },
    ]

    data_params = dictlistprod(
        {
            "subject_id": [None],  # 4
            "all_subjects_in_each_fold": [True],
            "n_times_train": [144],
            "n_times_eval": [144],
            "sfreq": [64],  # 32
            "trial_start_offset_sec": [None],
            "split_valid_off_train": [True],
            "low_cut_hz": [None],
            "high_cut_hz": [None],
            "exponential_standardize": [False],
            "class_names": [
                # ["left_hand", "right_hand", "feet", "tongue"]
                ["healthy", "pathology"]
            ],
            "dataset_name": ["tuh"],
            "hgd_sensors": [None],
            "n_tuh_recordings": [None],
        }
    )

    train_params = dictlistprod(
        {
            "n_epochs": [3],  # [20],
            "fixed_lr": [None],
            "fixed_batch_size": [None],
            "alpha_lr": [1e-2],  # [1e-2],
            "dist_lr": [1e-3],
            "n_times_crop": [128],
            "channel_drop_p": [0.0],
            "n_eval_crops": [3],
            "scheduler": ["cosine"],
        }
    )
    #assert False, "check dist module choices and max_n_changes max_n_deletions"
    variants = dictlistprod(
        {
            "min_improve_fraction": [0.05],
            "just_train_deep4": [False],
            "nll_loss_factor": [1],
            # "max_n_changes": [2],
            # "max_n_deletions": [1],
            "max_n_changes": [0],
            "max_n_deletions": [0],
            #
            "class_prob_masked": [False],
            "max_hours": [0.5],#2,#0.25,#0.25,
            "mutate_optim_params": [True],
            "n_virtual_classes": [0],
            "dist_module_choices": [["weighteddimgaussianmix"]],#maskedindependent"perdimweightedmix",
        },
    )
    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            data_params,
            train_params,
            variants,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params
