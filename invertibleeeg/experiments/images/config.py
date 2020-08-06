import os
os.sys.path.insert(0, '/home/schirrmr/code/invertible-public/')
os.sys.path.insert(0, '/home/schirrmr/code/invertible-eeg/')
os.sys.path.insert(0, '/home/schirrmr/code/invertible-neurips/')
import time
import logging

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts

import torch as th
import torch.backends.cudnn as cudnn
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s')


log = logging.getLogger(__name__)
log.setLevel('INFO')


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
        'save_folder': '/home/schirrmr/data/exps/invertible-eeg/images/',
    },
    ]

    data_params = [{
        'first_n': None,
    }]
    debug_params = [{
        'debug': False,
    }]

    train_params = dictlistprod({
        'n_epochs': [200],
        'batch_size': [32],
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,2),
    })

    optim_params = dictlistprod({
        'lr': [5e-4],
        'weight_decay': [5e-5],
        'cross_ent_weight': [0,],#1,10,100,1000
        'scale_2_cross_ent': [False, True],
        'mask_for_cross_ent': [False],
        'nll_weight': [1],
        'linear_classifier': [False],
        'flow_gmm': [True],
        'flow_coupling': ['affine'],
    })
    model_params = dictlistprod({
        'n_mixes': [32],
    })



    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        train_params,
        debug_params,
        random_params,
        optim_params,
        model_params,
        data_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
        ex,
        first_n,
        lr,
        weight_decay,
        cross_ent_weight,
        batch_size,
        np_th_seed,
        debug,
        n_epochs,
        n_mixes,
        scale_2_cross_ent,
        mask_for_cross_ent,
        nll_weight,
        linear_classifier,
        flow_gmm,
flow_coupling,
):
    kwargs = locals()
    kwargs.pop('ex')
    if not debug:
        log.setLevel('INFO')
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    kwargs['output_dir'] = output_dir
    th.backends.cudnn.benchmark = True
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False

    os.environ['pytorch_data'] = '/home/schirrmr/data/pytorch-datasets/'
    os.environ['tiny_data'] = '/home/schirrmr/data/tiny-images/'
    os.environ['lsun_data'] = '/home/schirrmr/data/lsun/'
    os.environ['brats_data'] = '/home/schirrmr/data/brats-2018//'
    os.environ['celeba_data'] = '/home/schirrmr/data/'  # wrong who cares
    os.environ['tiny_imagenet_data'] = '/home/schirrmr/data/'  # wrong who cares

    from invertibleeeg.experiments.images.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
