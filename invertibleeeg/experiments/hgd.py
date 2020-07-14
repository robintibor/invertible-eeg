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
        'save_folder': '/home/schirrmr/data/exps/invertible-eeg/hgd-21-ch-32-hz/',
    },
    ]

    data_params = dictlistprod({
        'subject_id': list(range(1,15)),
        'dataset_name': ['hgd'], # 4 is one window
    })


    model_params = dictlistprod({
        'hidden_channels': [128,],#512
        'n_virtual_chans': [0,],#1,2
        'n_blocks_up': [8],
        'n_blocks_down': [8],
        'n_mixes': [128],
        'splitter_last': ['haar',],
        'init_perm_to_identity': [True, ],#False#True
    })


    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        data_params,
        model_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params
