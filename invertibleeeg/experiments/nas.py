import functools
import logging
import os.path
import time
from copy import deepcopy
from functools import wraps
import functools

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import fasteners
import numpy as np
import pandas as pd
import torch as th
from black import format_str, FileMode
from braindecode.util import set_random_seeds
from hyperoptim.concurrent_file_observer import ConcurrentFileStorageObserver
from sacred.experiment import Experiment
from torch import nn
from invertible.split_merge import ChunkChansIn2

from invertible.amp_phase import AmplitudePhase
from invertible.distribution import PerDimWeightedMix, NClassIndependentDist
from invertible.graph import Node
from invertible.init import init_all_modules
from invertible.inverse import Inverse
from invertible.sequential import InvertibleSequential
from invertible.view_as import Flatten2d
from invertibleeeg.datasets import load_train_valid_test_bcic_iv_2a
from invertibleeeg.experiments.bcic_iv_2a import train_glow
from invertibleeeg.models.glow import get_splitter, conv_flow_block_nas
from invertible.affine import AdditiveCoefs, AffineCoefs
from invertible.affine import AffineModifier
from ..factors import MultiplyFactors
from invertible.actnorm import ActNorm
from invertible.permute import InvPermute
from invertible.coupling import CouplingLayer
from invertible.split_merge import EverySecondChan
from invertible.expression import Expression
from functools import partial

from ..models.cropped import CroppedGlow

log = logging.getLogger(__name__)


def copy_clean_encoding_dict(encoding):
    clean_encoding = {}
    for key, val in encoding.items():
        if key not in ["node", "optim_params_per_param"]:
            clean_encoding[key] = copy_clean_encoding_val(val)
    return clean_encoding


def copy_clean_encoding_val(val):
    if isinstance(val, dict):
        return copy_clean_encoding_dict(val)
    elif isinstance(val, list):
        clean_list = [copy_clean_encoding_val(elem) for elem in val]
        return clean_list
    else:
        assert val is None or (
            type(val) in [str, np.str_, int, bool, float]
        ), f"Unexpected encoding value type: {type(val)} for value: {val}"
        return val


def get_optim_choices():
    return [
        CSH.CategoricalHyperparameter(
            "lr",
            choices=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        ),
        CSH.CategoricalHyperparameter(
            "weight_decay",
            choices=[0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        ),
    ]


def sample_optim_config_space(seed):
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add_hyperparameters(get_optim_choices())
    config = cs.sample_configuration()
    return config.get_dictionary()


def mutate_encoding_and_model(
    encoding,
    rng,
    blocks,
    n_start_chans,
    max_n_changes,
    other_encoding,
    fixed_lr,
    max_n_downsample,
):
    encoding = deepcopy(encoding)
    model = encoding["node"]
    flat_node = model.prev[0]
    n_changes = rng.randint(0, max_n_changes + 1)
    final_blocks = []
    for _ in range(n_changes):
        # Determine which blocks are possible, some are only possible at end
        sample_choices = ["new"]
        if len(encoding["net"]) > 0:
            sample_choices.append("own")
        if len(other_encoding["net"]) > 0:
            sample_choices.append("other")
        sample_method = rng.choice(sample_choices)
        if sample_method == "new":
            assert all([blocks[k]["position"] in ["any", "end"] for k in blocks.keys()])
            block_type = rng.choice(list(blocks.keys()), size=1)[0]

            cs = CS.ConfigurationSpace(seed=rng.randint(2**32))
            cs.add_hyperparameters(blocks[block_type]["params"])
            block_params = cs.sample_configuration().get_dictionary()
            # Now sample optim params
            optim_params = sample_optim_config_space(seed=rng.randint(2**32))
            if fixed_lr is not None:
                optim_params["lr"] = fixed_lr
            n_downsample = rng.randint(0, max_n_downsample + 1)
        elif sample_method == "own":
            i_block_to_copy = rng.choice(len(encoding["net"]))
            block_type = encoding["net"][i_block_to_copy]["key"]
            optim_params = encoding["net"][i_block_to_copy]["optim_params"]
            block_params = encoding["net"][i_block_to_copy]["params"]
            n_downsample = encoding["net"][i_block_to_copy]["n_downsample"]
        elif sample_method == "other":
            i_block_to_copy = rng.choice(len(other_encoding["net"]))
            block_type = other_encoding["net"][i_block_to_copy]["key"]
            optim_params = other_encoding["net"][i_block_to_copy]["optim_params"]
            block_params = other_encoding["net"][i_block_to_copy]["params"]
            n_downsample = other_encoding["net"][i_block_to_copy]["n_downsample"]
        else:
            assert False

        if blocks[block_type]["position"] == "any":
            if sample_method in [
                "new",
                "own",
            ]:
                i_insert_before = rng.choice(len(encoding["net"]) + 1)
            else:
                assert sample_method == "other"
                # map  position in other encoding to this encoding
                i_insert_before = int(
                    np.ceil(
                        (i_block_to_copy / len(other_encoding["net"]))
                        * len(encoding["net"])
                    )
                )
        else:
            assert blocks[block_type]["position"] == "end"
            i_insert_before = len(encoding["net"])

        if i_insert_before == len(encoding["net"]):
            # we are at end of blocks
            next_node = flat_node
        else:
            next_node = encoding["net"][i_insert_before]["node"]
        if i_insert_before == 0:
            prev_node = None
        else:
            prev_node = encoding["net"][i_insert_before - 1]["node"]

        n_cur_chans = n_start_chans
        # Could also sture always n_chans as part of encoding... should not change for a given block
        # so could just set when block added
        for i_block in range(i_insert_before):
            n_cur_chans = blocks[encoding["net"][i_block]["key"]]["chans_after"](
                n_cur_chans
            )
        n_cur_chans = int(n_cur_chans * (2 ** n_downsample))
        block = blocks[block_type]["func"](n_chans=n_cur_chans, **block_params)

        # Add splitter for downsampling
        # invert logdet sign should not really matter
        splitter_seq = InvertibleSequential(
            *[
                get_splitter(splitter_name="haar", chunk_chans_first=True)
                for _ in range(n_downsample)
            ]
        )
        block = InvertibleSequential(
            splitter_seq, block, Inverse(splitter_seq, invert_logdet_sign=True)
        )

        for p in block.parameters():
            encoding["optim_params_per_param"][p] = optim_params
        cur_node = Node(prev_node, block, remove_prev_node_next=True)
        next_node.change_prev(
            cur_node, notify_prev_nodes=True, remove_prev_node_next=True
        )

        encoding["net"].insert(
            i_insert_before,
            dict(
                key=block_type,
                params=block_params,
                optim_params=optim_params,
                node=cur_node,
                n_downsample=n_downsample,
            ),
        )
        if block_type in ["splitter"]:
            # invert logdet sign should not really matter
            final_blocks.insert(0, Inverse(block, invert_logdet_sign=True))

    flat_node.module.sequential = nn.Sequential(
        *(final_blocks + list(flat_node.module.sequential.children()))
    )
    init_all_modules(model, None)
    return encoding


def to_code_str(net_list, max_line_length=88):
    m_strs = []
    for m in net_list:
        param_string = ",\n".join([f"{k}={v}" for k, v in m["params"].items()])
        m_str = f"{m['key']}({param_string})"
        m_strs.append(m_str)
    code_str = "[" + ", ".join(m_strs) + "]"
    m_str_clean = format_str(code_str, mode=FileMode(line_length=max_line_length))
    return m_str_clean


def ignore_n_chans(func):
    @wraps(func)
    def without_n_chans(n_chans, **kwargs):
        return func(**kwargs)

    return without_n_chans


def coupling_block(
    n_chans,
    hidden_channels,
    kernel_length,
    affine_or_additive,
    scale_fn,
    dropout=0.0,
    swap_dims=False,
    norm=None,
    nonlin=None,
):
    assert nonlin is not None
    nonlin_layer = {
        "square": Expression(th.square),
        "elu": nn.ELU(),
    }[nonlin]
    assert affine_or_additive in ["affine", "additive"]
    if affine_or_additive == "additive":
        CoefClass = AdditiveCoefs
        n_chans_out = n_chans // 2
    else:
        CoefClass = functools.partial(AffineCoefs, splitter=EverySecondChan())
        n_chans_out = n_chans

    assert kernel_length % 2 == 1
    assert norm in [None, "none", "bnorm"]
    if norm in ["none", None]:
        norm_layer = nn.Identity()
    elif norm == "bnorm":
        norm_layer = nn.BatchNorm1d(hidden_channels)
    else:
        raise ValueError(f"Unexpected norm {norm}")

    return CouplingLayer(
        ChunkChansIn2(swap_dims=swap_dims),
        CoefClass(
            nn.Sequential(
                nn.Conv1d(
                    n_chans // 2,
                    hidden_channels,
                    kernel_length,
                    padding=kernel_length // 2,
                ),
                nonlin_layer,
                nn.Dropout(dropout),
                norm_layer,
                nn.Conv1d(
                    hidden_channels,
                    n_chans_out,
                    kernel_length,
                    padding=kernel_length // 2,
                ),
                MultiplyFactors(n_chans_out),
            )
        ),
        AffineModifier(scale_fn, add_first=True, eps=0),
    )


def inv_permute(n_chans):
    permuter = InvPermute(n_chans, fixed=False, use_lu=True, init_identity=True)
    return permuter


def act_norm(n_chans, scale_fn):
    return ActNorm(
        n_chans,
        scale_fn,
    )


def get_simpleflow_blocks(include_splitter):
    blocks = {
        "coupling_block": {
            "func": coupling_block,
            "params": [
                CSH.CategoricalHyperparameter(
                    "affine_or_additive",
                    choices=["additive"],
                ),
                CSH.CategoricalHyperparameter(
                    "hidden_channels",
                    choices=[8, 16, 32, 64, 128, 256],
                ),
                CSH.CategoricalHyperparameter(
                    "kernel_length",
                    choices=[3, 5, 7, 9, 11, 13, 15],
                ),
                CSH.CategoricalHyperparameter("scale_fn", choices=["square"]),
                CSH.CategoricalHyperparameter("nonlin", choices=["square"]),
                CSH.CategoricalHyperparameter(
                    "dropout",
                    choices=[
                        0,
                    ],
                ),
                CSH.CategoricalHyperparameter(
                    "swap_dims",
                    choices=[True, False],
                ),
                CSH.CategoricalHyperparameter(
                    "norm",
                    choices=[
                        "none",
                    ],
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "permute": {
            "func": inv_permute,
            "params": [],
            "chans_after": lambda x: x,
            "position": "any",
        },
    }
    if include_splitter:
        blocks["splitter"] = {
            "func": ignore_n_chans(get_splitter),
            "params": [
                CSH.CategoricalHyperparameter(
                    "splitter_name", choices=["haar", "subsample"]
                ),
                CSH.CategoricalHyperparameter(
                    "chunk_chans_first", choices=[True, False]
                ),
            ],
            "chans_after": lambda x: x * 2,
            "position": "end",
        }

    return blocks


def get_downsample_anywhere_blocks():
    blocks = {
        "coupling_block": {
            "func": coupling_block,
            "params": [
                CSH.CategoricalHyperparameter(
                    "affine_or_additive",
                    choices=["affine", "additive"],
                ),
                CSH.CategoricalHyperparameter(
                    "hidden_channels",
                    choices=[8, 16, 32, 64, 128, 256],
                ),
                CSH.CategoricalHyperparameter(
                    "kernel_length",
                    choices=[3, 5, 7, 9, 11, 13, 15],
                ),
                CSH.CategoricalHyperparameter(
                    "scale_fn", choices=["twice_sigmoid", "exp"]
                ),
                CSH.CategoricalHyperparameter("nonlin", choices=["elu"]),
                CSH.CategoricalHyperparameter("dropout", choices=[0, 0.2, 0.5]),
                CSH.CategoricalHyperparameter(
                    "swap_dims",
                    choices=[True, False],
                ),
                CSH.CategoricalHyperparameter(
                    "norm",
                    choices=["none", "bnorm"],
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "permute": {
            "func": inv_permute,
            "params": [],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "act_norm": {
            "func": act_norm,
            "params": [
                CSH.CategoricalHyperparameter(
                    "scale_fn", choices=["twice_sigmoid", "exp"]
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
    }
    return blocks


def get_blocks():
    blocks = {
        # "conv_flow_block_nas": {
        #     "func": conv_flow_block_nas,
        #     "params": [
        #         CSH.CategoricalHyperparameter(
        #             "affine_or_additive", choices=["affine", "additive"]
        #         ),
        #         CSH.CategoricalHyperparameter(
        #             "hidden_channels", choices=[8, 16, 32, 64, 128, 256]
        #         ),
        #         CSH.CategoricalHyperparameter(
        #             "kernel_length", choices=[3, 5, 7, 9, 11, 13, 15]
        #         ),
        #         CSH.CategoricalHyperparameter(
        #             "scale_fn", choices=["twice_sigmoid", "exp"]
        #         ),
        #         CSH.CategoricalHyperparameter("permute", choices=[True, False]),
        #         CSH.CategoricalHyperparameter("dropout", choices=[0, 0.2, 0.5]),
        #     ],
        #     "chans_after": lambda x: x,
        #     "position": "any",
        # },
        "coupling_block": {
            "func": coupling_block,
            "params": [
                CSH.CategoricalHyperparameter(
                    "affine_or_additive",
                    choices=["affine", "additive"],
                ),
                CSH.CategoricalHyperparameter(
                    "hidden_channels",
                    choices=[8, 16, 32, 64, 128, 256],
                ),
                CSH.CategoricalHyperparameter(
                    "kernel_length",
                    choices=[3, 5, 7, 9, 11, 13, 15],
                ),
                CSH.CategoricalHyperparameter(
                    "scale_fn", choices=["twice_sigmoid", "exp"]
                ),
                CSH.CategoricalHyperparameter("nonlin", choices=["elu"]),
                CSH.CategoricalHyperparameter("dropout", choices=[0, 0.2, 0.5]),
                CSH.CategoricalHyperparameter(
                    "swap_dims",
                    choices=[True, False],
                ),
                CSH.CategoricalHyperparameter(
                    "norm",
                    choices=["none", "bnorm"],
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "permute": {
            "func": inv_permute,
            "params": [],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "act_norm": {
            "func": act_norm,
            "params": [
                CSH.CategoricalHyperparameter(
                    "scale_fn", choices=["twice_sigmoid", "exp"]
                ),
            ],
            "chans_after": lambda x: x,
            "position": "any",
        },
        "splitter": {
            "func": ignore_n_chans(get_splitter),
            "params": [
                CSH.CategoricalHyperparameter(
                    "splitter_name", choices=["haar", "subsample"]
                ),
                CSH.CategoricalHyperparameter(
                    "chunk_chans_first", choices=[True, False]
                ),
            ],
            "chans_after": lambda x: x * 2,
            "position": "end",
        },
    }
    return blocks


def init_model_encoding(
    amplitude_phase_at_end,
    n_times,
    class_prob_masked,
    alpha_lr,
    sample_dist_module,
    rng,
):
    n_classes = 4
    n_real_chans = 22
    init_dist_std = 0.1
    n_mixes = 8
    n_virtual_chans = 0
    n_chans = n_real_chans + n_virtual_chans

    net_encoding = {
        "net": [],
    }
    n_dims = n_times * n_chans

    if sample_dist_module:
        if rng.rand() > 0.5:
            dist = PerDimWeightedMix(
                n_classes,
                n_mixes=n_mixes,
                n_dims=n_dims,
                optimize_mean=True,
                optimize_std=True,
                init_std=init_dist_std,
            )
        else:
            dist = NClassIndependentDist(
                n_classes,
                n_dims=n_dims,
                optimize_mean=True,
                optimize_std=True,
            )
    else:
        dist = PerDimWeightedMix(
            n_classes,
            n_mixes=n_mixes,
            n_dims=n_dims,
            optimize_mean=True,
            optimize_std=True,
            init_std=init_dist_std,
        )

    cur_node = None

    if amplitude_phase_at_end:
        flat_node = Node(cur_node, InvertibleSequential(AmplitudePhase(), Flatten2d()))
    else:
        flat_node = Node(cur_node, InvertibleSequential(Flatten2d()))
    dist_node = Node(flat_node, dist)

    net = dist_node

    init_all_modules(net, None)
    # Now also add optim params
    net_encoding["optim_params_per_param"] = {}
    for p in net.parameters():
        net_encoding["optim_params_per_param"][p] = dict(lr=1e-3, weight_decay=5e-5)
    net = net.cuda()
    if class_prob_masked:
        net.alphas = nn.Parameter(
            th.zeros(n_chans * n_times, device="cuda", requires_grad=True)
        )
        net_encoding["optim_params_per_param"][net.alphas] = dict(
            lr=alpha_lr, weight_decay=5e-5
        )
    net_encoding["node"] = net
    return net_encoding


def run_exp(
    debug,
    np_th_seed,
    worker_folder,
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
    batch_size = 64
    noise_factor = 5e-3
    start_time = time.time()
    n_virtual_chans = 0
    n_real_chans = 22
    n_chans = n_real_chans + n_virtual_chans

    sfreq = 32

    n_classes = 4
    split_valid_off_train = True
    # maybe specifiy via hparam to ensure you know of the change class_names = ["left_hand", "right_hand", "feet", "tongue"]

    set_random_seeds(np_th_seed, True)
    log.info("Load data...")
    train_set, valid_set, test_set = load_train_valid_test_bcic_iv_2a(
        subject_id,
        class_names,
        split_valid_off_train=split_valid_off_train,
        all_subjects_in_each_fold=all_subjects_in_each_fold,
        sfreq=sfreq,
        trial_start_offset_sec=trial_start_offset_sec,
    )

    max_n_downsample = int(np.log2(n_times) - 2)
    block_fn = dict(
        simpleflow=partial(get_simpleflow_blocks, include_splitter=include_splitter),
        default=get_blocks,
        downsample_anywhere=get_downsample_anywhere_blocks,
    )[searchspace]

    rng = np.random.RandomState(np_th_seed)
    while (time.time() - start_time) < (max_hours * 3600):
        ex = Experiment()
        ex.observers.append(ConcurrentFileStorageObserver.create(worker_folder))
        config = {}

        @ex.main
        def wrap_run_exp():
            start_time = time.time()
            csv_file = os.path.join(worker_folder, "population.csv")
            csv_lock_file = csv_file + ".lock"
            csv_file_lock = fasteners.InterProcessLock(csv_lock_file)
            csv_file_lock.acquire()
            try:
                population_df = pd.read_csv(csv_file, index_col="pop_id")
            except FileNotFoundError:
                population_df = None
            csv_file_lock.release()
            if population_df is not None and len(population_df) >= n_start_population:
                # grab randomly one among top n_population
                # change it, with chance of no change as well
                sorted_pop_df = population_df.sort_values(by=search_by)
                this_parent = sorted_pop_df.iloc[
                    rng.randint(0, min(len(population_df), n_alive_population))
                ]
                parent_folder = this_parent["folder"]
                parent_encoding_filename = os.path.join(parent_folder, "encoding.pth")
                parent_encoding = th.load(parent_encoding_filename)

                other_parent = sorted_pop_df.iloc[
                    rng.randint(0, min(len(population_df), n_alive_population))
                ]
                other_parent_folder = other_parent["folder"]
                other_parent_encoding_filename = os.path.join(
                    other_parent_folder, "encoding.pth"
                )
                other_parent_encoding = th.load(other_parent_encoding_filename)

                model_worked = False
                log.info("Mutating and trying model...")
                if fixed_batch_size is None:
                    batch_size = 32
                else:
                    batch_size = fixed_batch_size
                while not model_worked:
                    try:
                        encoding = mutate_encoding_and_model(
                            parent_encoding,
                            rng,
                            block_fn(),
                            n_start_chans=n_chans,
                            max_n_changes=max_n_changes,
                            other_encoding=other_parent_encoding,
                            fixed_lr=fixed_lr,
                            max_n_downsample=max_n_downsample,
                        )
                        model_to_train = encoding["node"].cuda()
                        if n_times_crop is not None:
                            model_to_train = CroppedGlow(model_to_train, n_times=n_times_crop)
                        # Try a forward-backward to ensure model works
                        # Also here you should check function is unperturbed!!
                        _, lp = model_to_train(
                            th.zeros(batch_size, n_chans, n_times, device="cuda")
                        )
                        mean_lp = th.mean(lp)
                        mean_lp.backward()
                        _ = [
                            p.grad.zero_()
                            for p in model_to_train.parameters()
                            if p.grad is not None
                        ]
                        model_worked = True
                        log.info("Model:\n" + str(model_to_train))
                    except RuntimeError as e:
                        log.info("Model failed....:\n" + str(model_to_train))
                        print(e)
            else:
                encoding = init_model_encoding(
                    amplitude_phase_at_end,
                    n_times=n_times_crop or n_times,
                    class_prob_masked=class_prob_masked,
                    alpha_lr=alpha_lr,
                    sample_dist_module=sample_dist_module,
                    rng=rng,
                )
                model_to_train = encoding["node"].cuda()
                if n_times_crop is not None:
                    model_to_train = CroppedGlow(model_to_train, n_times=n_times_crop)
                parent_folder = None
            # Now determine suitable batch_size
            model_worked = False
            if fixed_batch_size is None:
                log.info("Determine max batch size...")
                batch_size = 512
                while not model_worked:
                    try:
                        # Try a forward-backward to ensure model works with batch size
                        _, lp = model_to_train(
                            th.zeros(batch_size, n_chans, n_times, device="cuda")
                        )
                        mean_lp = th.mean(lp)
                        mean_lp.backward()
                        _ = [
                            p.grad.zero_()
                            for p in model_to_train.parameters()
                            if p.grad is not None
                        ]
                        model_worked = True
                        log.info(f"Batch size: {batch_size}")
                    except RuntimeError as e:
                        log.info(f"Batch size {batch_size} failed.")
                        batch_size = batch_size // 2
                        print(e)
                        if batch_size == 0:
                            raise e

            # train it get result
            # (maybe also remember how often it was trained, history, or maybe just remember parent id  in df)
            log.info("Train model...")
            # result and encoding
            results = train_glow(
                model_to_train,
                class_prob_masked,
                nll_loss_factor,
                noise_factor,
                train_set,
                valid_set,
                test_set,
                n_epochs,
                n_virtual_chans,
                n_classes=n_classes,
                n_times=n_times,
                np_th_seed=np_th_seed,
                scheduler=dict(
                    identity=functools.partial(
                        th.optim.lr_scheduler.LambdaLR, lr_lambda=lambda epoch: 1
                    ),
                    cosine=functools.partial(
                        th.optim.lr_scheduler.CosineAnnealingLR,
                        T_max=len(train_set) // batch_size,
                    ),
                )[scheduler],
                batch_size=batch_size,
                optim_params_per_param=encoding["optim_params_per_param"],
                with_tqdm=False,
            )
            metrics = results
            runtime = time.time() - start_time
            results["runtime"] = runtime
            for key, val in metrics.items():
                ex.info[key] = val

            file_obs = ex.observers[0]
            output_dir = file_obs.dir
            encoding["node"].cpu()
            th.save(encoding, os.path.join(output_dir, "encoding.pth"))
            clean_encoding = copy_clean_encoding_dict(encoding)
            th.save(clean_encoding, os.path.join(output_dir, "encoding_no_params.pth"))

            csv_file_lock = fasteners.InterProcessLock(csv_lock_file)
            csv_file_lock.acquire()
            try:
                population_df = pd.read_csv(csv_file, index_col="pop_id")
            except FileNotFoundError:
                population_df = pd.DataFrame()

            this_dict = dict(folder=output_dir, parent_folder=parent_folder, **metrics)
            population_df = population_df.append(this_dict, ignore_index=True)
            population_df.to_csv(csv_file, index_label="pop_id")
            csv_file_lock.release()

            return results

        ex.add_config(config)
        ex.run()
    results = {}
    return results
