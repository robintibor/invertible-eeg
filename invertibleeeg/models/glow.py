from torch import nn

from invertible.actnorm import ActNorm
from invertible.affine import AdditiveCoefs
from invertible.affine import AffineModifier
from invertible.amp_phase import AmplitudePhase
from invertible.branching import ChunkChans
from invertible.coupling import CouplingLayer
from invertible.distribution import NClassIndependentDist, PerDimWeightedMix
from invertible.graph import Node, SelectNode, CatAsListNode
from invertible.lists import ApplyToList
from invertible.permute import InvPermute
from invertible.sequential import InvertibleSequential
from invertible.split_merge import ChunkChansIn2
from invertible.subsample_split import SubsampleSplitter
from invertible.view_as import Flatten2d
from ..factors import MultiplyFactors
from ..wavelet import Haar1dWavelet


def conv_flow_block(n_chans, hidden_channels, kernel_length):
    assert kernel_length % 2 == 1
    return InvertibleSequential(
        ActNorm(n_chans, 'exp', ),
        InvPermute(n_chans, fixed=False, use_lu=True),
        CouplingLayer(
            ChunkChansIn2(swap_dims=False),
            AdditiveCoefs(nn.Sequential(
                nn.Conv1d(n_chans // 2, hidden_channels, kernel_length, padding=kernel_length // 2),
                nn.ELU(),
                nn.Conv1d(hidden_channels, n_chans // 2, kernel_length, padding=kernel_length // 2),
                MultiplyFactors(n_chans // 2),
            )),
            AffineModifier('sigmoid', add_first=True, eps=0)))


def dense_flow_block(n_chans, hidden_channels):
    return InvertibleSequential(ActNorm(n_chans, 'exp', ),
                                InvPermute(n_chans, fixed=False, use_lu=True),
                                CouplingLayer(
                                    ChunkChansIn2(swap_dims=False),
                                    AdditiveCoefs(nn.Sequential(
                                        nn.Linear(n_chans // 2, hidden_channels),
                                        nn.ELU(),
                                        nn.Linear(hidden_channels, n_chans // 2),
                                        MultiplyFactors(n_chans // 2),
                                    )),
                                    AffineModifier('sigmoid', add_first=True, eps=0)))


def create_eeg_glow_full_dist(n_chans, hidden_channels, kernel_length,):
    n_merged = create_eeg_glow_no_dist(n_chans, hidden_channels, kernel_length)
    dist0 = NClassIndependentDist(2, 64 * n_chans // 2, optimize_mean=True, optimize_std=False)
    dist1 = NClassIndependentDist(2, 64 * n_chans // 4, optimize_mean=True, optimize_std=False)
    dist2 = NClassIndependentDist(2, 64 * n_chans // 4, optimize_mean=True, optimize_std=False)
    # architecture plan:
    n_dist = Node(n_merged, ApplyToList(dist0, dist1, dist2))
    net = n_dist
    return net


def create_eeg_glow_per_dim_dist(n_chans, hidden_channels, kernel_length, n_mixes,
                                 init_dist_std=1e-1):
    n_merged = create_eeg_glow_no_dist(n_chans, hidden_channels, kernel_length)
    dist0 = PerDimWeightedMix(2, n_mixes=n_mixes, n_dims=64 * n_chans // 2,
                              optimize_mean=True, optimize_std=True, init_std=init_dist_std)
    dist1 = PerDimWeightedMix(2, n_mixes=n_mixes, n_dims=64 * n_chans // 4,
                              optimize_mean=True, optimize_std=True,  init_std=init_dist_std)
    dist2 = PerDimWeightedMix(2, n_mixes=n_mixes, n_dims=64 * n_chans // 4,
                              optimize_mean=True, optimize_std=True,  init_std=init_dist_std)
    # architecture plan:
    n_dist = Node(n_merged, ApplyToList(dist0, dist1, dist2))
    net = n_dist
    return net


def create_eeg_glow_no_dist(n_chans, hidden_channels, kernel_length):
    block_a_out = InvertibleSequential(
        AmplitudePhase(),
        Flatten2d(),
    )
    block_b_out = InvertibleSequential(
        AmplitudePhase(),
        Flatten2d(),
    )
    block_c_out = InvertibleSequential(
        AmplitudePhase(),
        Flatten2d(),
    )

    n_before_block = create_eeg_glow_before_amp_phase(
        n_chans=n_chans, hidden_channels=hidden_channels,
        kernel_length=kernel_length)
    n_a_out = Node(SelectNode(n_before_block, 0), block_a_out)
    n_b_out = Node(SelectNode(n_before_block, 1), block_b_out)
    n_c_out = Node(SelectNode(n_before_block, 2), block_c_out)
    n_merged = CatAsListNode([n_a_out, n_b_out, n_c_out])
    net = n_merged
    return n_merged


def create_eeg_glow_before_amp_phase(n_chans, hidden_channels, kernel_length, splitter):
    assert splitter in ['haar', 'subsample']
    if splitter == 'haar':
        def splitter_fn(chunk_chans_first):
            return Haar1dWavelet(chunk_chans_first=chunk_chans_first)
    else:
        def splitter_fn(chunk_chans_first):
            return SubsampleSplitter((2,), chunk_chans_first=chunk_chans_first)
    block_a = InvertibleSequential(
        splitter_fn(chunk_chans_first=False),
        conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length),
        splitter_fn(chunk_chans_first=False),
    )

    block_b = InvertibleSequential(
        splitter_fn(chunk_chans_first=True),
        conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length),
    )


    block_c = InvertibleSequential(
        splitter_fn(chunk_chans_first=False),
        splitter_fn(chunk_chans_first=True),
        Flatten2d(),
        dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels, ),
        dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels, ),
        dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels, ),
        dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels, ),
    )

    n_a = Node(None, block_a)
    n_a_split = Node(n_a, ChunkChans(2))
    n_a_out = SelectNode(n_a_split, 1)
    n_b = Node(SelectNode(n_a_split, 0), block_b)
    n_b_split = Node(n_b, ChunkChans(2))
    n_b_out = SelectNode(n_b_split, 1)
    n_c = Node(SelectNode(n_b_split, 0), block_c)
    n_c_out = n_c
    n_merged = CatAsListNode([n_a_out, n_b_out, n_c_out])
    net = n_merged
    return n_merged


def create_eeg_glow_before_amp_phase_2(n_chans, hidden_channels, kernel_length,):
    block_a = InvertibleSequential(
        SubsampleSplitter((2,), chunk_chans_first=False),
        conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length),
        Haar1dWavelet(chunk_chans_first=False),
    )

    block_b = InvertibleSequential(
        SubsampleSplitter((2,), chunk_chans_first=False),
        conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length),
        conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length),
        Haar1dWavelet(chunk_chans_first=False),
    )


    block_c = InvertibleSequential(
        Flatten2d(),
        dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels, ),
        dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels, ),
        dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels, ),
        dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels, ),
    )

    n_a = Node(None, block_a)
    n_a_split = Node(n_a, ChunkChans(2))
    n_a_out = SelectNode(n_a_split, 1)
    n_b = Node(SelectNode(n_a_split, 0), block_b)
    n_b_split = Node(n_b, ChunkChans(2))
    n_b_out = SelectNode(n_b_split, 1)
    n_c = Node(SelectNode(n_b_split, 0), block_c)
    n_c_out = n_c
    n_merged = CatAsListNode([n_a_out, n_b_out, n_c_out])
    net = n_merged
    return n_merged
