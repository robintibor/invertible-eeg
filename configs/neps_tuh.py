# PARENTCONFIG: /home/schirrmr/code/invertible-eeg/configs/neps.py

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
            "save_folder": "/work/dlclarge1/schirrmr-renormalized-flows/exps/neps/tuh-2sec-eps-1e-2/",
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
            "n_times_crop": [128],
        }
    )

    search_params = [
        {
            "max_hours": 15 * 1.5,  # 0.25,
            "epochs_as_fidelity": False,
            "n_max_epochs": 20,
            "seed_offset": 0,
            "ignore_errors": True,
        }
    ]
    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            data_params,
            search_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params
