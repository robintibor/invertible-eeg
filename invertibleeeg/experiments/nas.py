import logging
from copy import deepcopy
from itertools import islice
import os.path
import numpy as np
import torch
import torch as th
from torch import nn
from braindecode.models import Deep4Net
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from invertible.gaussian import get_gaussian_log_probs
from invertible.init import init_all_modules
from invertible.util import weighted_sum
from invertible.view_as import flatten_2d
from invertibleeeg.datasets import load_train_valid_bcic_iv_2a
from invertibleeeg.models.glow import create_eeg_glow, create_eeg_glow_multi_stage
from skorch.utils import to_numpy
from tensorboardX import SummaryWriter
from tqdm.autonotebook import trange
from sacred.experiment import Experiment
from hyperoptim.concurrent_file_observer import ConcurrentFileStorageObserver
import time
import fasteners
import pandas as pd
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS
from functools import wraps
from invertibleeeg.models.glow import get_splitter, conv_flow_block_nas
from invertible.init import init_all_modules
from invertible.view_as import Flatten2d
import torch as th
from torch import nn
from invertible.actnorm import ActNorm
from invertible.affine import AdditiveCoefs, AffineCoefs
from invertible.affine import AffineModifier
from invertible.amp_phase import AmplitudePhase
from invertible.branching import ChunkChans
from invertible.conditional import ConditionTransformWrapper, CatCondAsList
from invertible.coupling import CouplingLayer
from invertible.distribution import NClassIndependentDist, PerDimWeightedMix
from invertible.expression import Expression
from invertible.graph import Node, SelectNode, CatAsListNode, ConditionalNode
from invertible.lists import ApplyToList, ApplyToListNoLogdets
from invertible.multi_in_out import MultipleInputOutput
from invertible.permute import InvPermute
from invertible.pure_model import NoLogDet
from invertible.sequential import InvertibleSequential
from invertible.split_merge import ChunkChansIn2
from invertible.splitter import SubsampleSplitter
from invertible.view_as import Flatten2d, ViewAs
from invertible.split_merge import EverySecondChan
from invertibleeeg.wavelet import Haar1dWavelet
import functools
from black import format_str, FileMode
from invertible.inverse import Inverse
from invertibleeeg.datasets import load_train_valid_test_bcic_iv_2a
from tqdm import trange
from invertibleeeg.experiments.bcic_iv_2a import train_glow


log = logging.getLogger(__name__)


def mutate_encoding_and_model(encoding, rng, blocks):
    encoding = deepcopy(encoding)
    model = encoding["node"]
    flat_node = model.prev[0]
    try:
        cur_node = flat_node.prev[0]
    except TypeError:
        assert flat_node.prev is None
        cur_node = flat_node.prev
    n_changes = rng.randint(0, 4)
    n_cur_chans = model.n_last_chans
    final_blocks = []
    for _ in range(n_changes):
        i_block = rng.choice(len(blocks.keys()), size=1)[0]
        block_type = list(blocks.keys())[i_block]
        cs = CS.ConfigurationSpace(seed=rng.randint(2 ** 32))
        cs.add_hyperparameters(blocks[block_type]["params"])
        config = cs.sample_configuration()
        block = blocks[block_type]["func"](n_chans=n_cur_chans, **config)
        cur_node = Node(cur_node, block)
        encoding["net"].append(
            dict(
                key=block_type,
                params=config.get_dictionary(),
                optim_params=dict(lr=5e-4, weight_decay=5e-5),
                node=cur_node,
            )
        )
        n_cur_chans = blocks[block_type]["chans_after"](n_cur_chans)
        if block_type in ["splitter"]:
            final_blocks.insert(0, Inverse(block, invert_logdet_sign=True))
    model.n_last_chans = n_cur_chans

    flat_node.change_prev(cur_node, notify_prev_nodes=True)
    flat_node.module.sequential = nn.Sequential(
        *(final_blocks + list(flat_node.module.sequential.children()))
    )
    init_all_modules(model, None)
    return encoding


def to_code_str(net_list):
    m_strs = []
    for m in net_list:
        param_string = ",\n".join([f"{k}={v}" for k, v in m["params"].items()])
        m_str = f"{m['key']}({param_string})"
        m_strs.append(m_str)
    code_str = "[" + ", ".join(m_strs) + "]"
    m_str_clean = format_str(code_str, mode=FileMode())
    return m_str_clean


def ignore_n_chans(func):
    @wraps(func)
    def without_n_chans(n_chans, **kwargs):
        return func(**kwargs)

    return without_n_chans


def get_blocks():
    blocks = {
        "conv_flow_block_nas": {
            "func": conv_flow_block_nas,
            "params": [
                CSH.CategoricalHyperparameter(
                    "affine_or_additive", choices=["affine", "additive"]
                ),
                CSH.CategoricalHyperparameter(
                    "hidden_channels", choices=[8, 16, 32, 64, 128, 256]
                ),
                CSH.CategoricalHyperparameter(
                    "kernel_length", choices=[3, 5, 7, 9, 11, 13, 15]
                ),
                CSH.CategoricalHyperparameter(
                    "scale_fn", choices=["twice_sigmoid", "exp"]
                ),
                CSH.CategoricalHyperparameter("permute", choices=[True, False]),
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


def init_model_encoding(amplitude_phase_at_end, n_times):
    n_classes = 4
    n_real_chans = 22
    init_dist_std = 0.1
    n_mixes = 8
    n_virtual_chans = 0
    n_chans = n_real_chans + n_virtual_chans

    blocks = get_blocks()

    net_encoding = {
        "net": [],
    }
    n_dims = n_times * n_chans

    dist = PerDimWeightedMix(
        n_classes,
        n_mixes=n_mixes,
        n_dims=n_dims,
        optimize_mean=True,
        optimize_std=True,
        init_std=init_dist_std,
    )

    cur_node = None

    for n in net_encoding["net"]:
        func = blocks[n["key"]]["func"]
        module = func(n_chans=n_chans, **n["params"])
        cur_node = Node(cur_node, module)

    if amplitude_phase_at_end:
        flat_node = Node(cur_node, InvertibleSequential(AmplitudePhase(), Flatten2d()))
    else:
        flat_node = Node(cur_node, InvertibleSequential(Flatten2d()))
    dist_node = Node(flat_node, dist)

    net = dist_node

    net.n_last_chans = n_chans  # hack for now

    init_all_modules(net, None)
    net = net.cuda()
    net.alphas = nn.Parameter(
        th.zeros(n_chans * n_times, device="cuda", requires_grad=True)
    )
    net_encoding["node"] = net
    return net_encoding


def run_exp(
    debug,
    np_th_seed,
    worker_folder,
    max_hours,
    n_population,
    subject_id,
    n_epochs,
    amplitude_phase_at_end,
    all_subjects_in_each_fold,
    n_times,
):
    batch_size = 64
    class_prob_masked = True
    nll_loss_factor = 1e-4
    noise_factor = 5e-3
    start_time = time.time()
    n_virtual_chans = 0
    n_real_chans = 22
    n_chans = n_real_chans + n_virtual_chans

    n_classes = 4
    split_valid_off_train = True
    class_names = ["left_hand", "right_hand", "feet", "rest"]

    set_random_seeds(np_th_seed, True)
    log.info("Load data...")
    train_set, valid_set, test_set = load_train_valid_test_bcic_iv_2a(
        subject_id,
        class_names,
        split_valid_off_train=split_valid_off_train,
        all_subjects_in_each_fold=all_subjects_in_each_fold,
    )

    rng = np.random.RandomState(np_th_seed)
    while (time.time() - start_time) < (max_hours * 3600):
        ex = Experiment()
        ex.observers.append(ConcurrentFileStorageObserver.create(worker_folder))
        config = {}

        @ex.main
        def wrap_run_exp():
            csv_file = os.path.join(worker_folder, "population.csv")
            csv_lock_file = csv_file + ".lock"
            csv_file_lock = fasteners.InterProcessLock(csv_lock_file)
            csv_file_lock.acquire()
            try:
                population_df = pd.read_csv(csv_file, index_col="pop_id")
            except FileNotFoundError:
                population_df = None
            csv_file_lock.release()
            if population_df is not None and len(population_df) >= n_population:
                # grab randomly one among top 9
                # change it, with chance of no change as well
                population_df = population_df.sort_values(by="valid_mis")
                this_parent = population_df.iloc[rng.randint(0, n_population)]
                parent_encoding_filename = os.path.join(
                    this_parent["folder"], "encoding.pth"
                )
                parent_folder = this_parent["folder"]
                parent_encoding = th.load(parent_encoding_filename)

                model_worked = False
                log.info("Mutating and trying model...")
                while not model_worked:
                    try:
                        encoding = mutate_encoding_and_model(
                            parent_encoding, rng, get_blocks()
                        )
                        encoding["node"].cuda()
                        # Try a forward-backward to ensure model works
                        _, lp = encoding["node"](
                            th.zeros(batch_size, n_chans, n_times, device="cuda")
                        )
                        mean_lp = th.mean(lp)
                        mean_lp.backward()
                        _ = [
                            p.grad.zero_()
                            for p in encoding["node"].parameters()
                            if p.grad is not None
                        ]
                        model_worked = True
                    except RuntimeError:
                        log.info("Model failed....")
                        pass
            else:
                encoding = init_model_encoding(
                    amplitude_phase_at_end, n_times=n_times)
                parent_folder = None

            # train it get result
            # (maybe also remember how often it was trained, history, or maybe just remember parent id  in df)

            log.info("Train model...")
            # result and encoding
            results = train_glow(
                encoding["node"].cuda(),
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
                scheduler=functools.partial(
                    th.optim.lr_scheduler.CosineAnnealingLR, T_max=len(train_set) // batch_size
                ),
                batch_size=batch_size,
            )
            metrics = results
            for key, val in metrics.items():
                ex.info[key] = val

            file_obs = ex.observers[0]
            output_dir = file_obs.dir
            encoding["node"].cpu()
            th.save(encoding, os.path.join(output_dir, "encoding.pth"))

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

            # afterwards lock global dataframe, add your id,
            # check if you are superior, if yes, add to active population/parents,
            # and remove another one
            # then
            return results

        ex.add_config(config)
        ex.run()
    results = {}
    return results
