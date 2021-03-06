# PARENTCONFIG: /home/schirrmr/code/invertible-eeg/invertibleeeg/experiments/config.py

import os
os.sys.path.insert(0, '/home/schirrmr/code/invertible-public/')
os.sys.path.insert(0, '/home/schirrmr/code/invertible-eeg/')
import time
import logging

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
        'save_folder': '/home/schirrmr/data/exps/invertible-eeg/tuh-all-chans/',
    },
    ]

    data_params = dictlistprod({
        'n_subjects': [2076], #'#'
        'n_seconds': [15*4], # 4 is one window
        'dataset_name': ['tuh'], # 4 is one window
    })




    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        data_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params
