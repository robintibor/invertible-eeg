from torch import nn

from invertible.actnorm import ActNorm
from invertible.affine import AdditiveCoefs
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
from invertible.subsample_split import SubsampleSplitter
from invertible.view_as import Flatten2d, ViewAs
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


def create_eeg_glow_up(n_chans, hidden_channels, kernel_length, splitter_first,
                       splitter_last, n_blocks):

    def get_splitter(splitter_name, chunk_chans_first):
        if splitter_name == 'haar':
            return Haar1dWavelet(chunk_chans_first=False)
        else:
            assert splitter_name == 'subsample'
            return SubsampleSplitter((2,), chunk_chans_first=chunk_chans_first)
    block_a = InvertibleSequential(
        get_splitter(splitter_first, chunk_chans_first=False),
        *[conv_flow_block(n_chans * 2, hidden_channels=hidden_channels, kernel_length=kernel_length)
         for _ in range(n_blocks)],
        get_splitter(splitter_last, chunk_chans_first=True),
    )

    block_b = InvertibleSequential(
        get_splitter(splitter_first, chunk_chans_first=False),
        *[conv_flow_block(n_chans * 4, hidden_channels=hidden_channels, kernel_length=kernel_length)
         for _ in range(n_blocks)],
        get_splitter(splitter_last, chunk_chans_first=True),
    )

    block_c = InvertibleSequential(
        get_splitter(splitter_first, chunk_chans_first=False),
        Flatten2d(),
        *[dense_flow_block(n_chans * 64 // 4, hidden_channels=hidden_channels,)
         for _ in range(n_blocks)],
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
    return n_merged


def flow_block_down_1(n_chans, hidden_channels):
    in_to_outs = [[
        nn.Sequential(nn.Conv1d(n_chans * 2, hidden_channels, 5, padding=5 // 2)),
        nn.Sequential(nn.Conv1d(n_chans * 8, hidden_channels, 3, padding=3 // 2),
                      nn.Upsample(scale_factor=2))
    ]]
    m = nn.Sequential(
        MultipleInputOutput(in_to_outs),
        Expression(unwrap_single_element),
        nn.Conv1d(hidden_channels, n_chans * 4  // 2, 5, padding=5 // 2),
        MultiplyFactors(n_chans * 4 // 2),
    )
    c = CouplingLayer(
        ChunkChansIn2(swap_dims=False),
        AdditiveCoefs(m),
        AffineModifier('sigmoid', add_first=True, eps=0),
        condition_merger=CatCondAsList(cond_preproc=None)
    )

    f = InvertibleSequential(
        ActNorm(n_chans * 4, 'exp', ),
        InvPermute(n_chans * 4, fixed=False, use_lu=True),
        c,
    )
    return f

def flow_block_down_0(n_chans, hidden_channels, kernel_length):
    in_to_outs = [[
        nn.Sequential(nn.Conv1d(n_chans, hidden_channels, kernel_length, padding=kernel_length // 2)),
        nn.Sequential(nn.Conv1d(n_chans * 4, hidden_channels, 5, padding=5 // 2),
                      nn.Upsample(scale_factor=4)),
        nn.Sequential(nn.Conv1d(n_chans * 8, hidden_channels, 3, padding=3 // 2),
                      nn.Upsample(scale_factor=8))
    ]]

    m = nn.Sequential(
        MultipleInputOutput(in_to_outs),
        Expression(unwrap_single_element),
        nn.Conv1d(hidden_channels, n_chans * 2 // 2, kernel_length, padding=kernel_length // 2),
        MultiplyFactors(n_chans * 2 // 2),
    )

    c = CouplingLayer(ChunkChansIn2(swap_dims=False),
                      AdditiveCoefs(m),
                      AffineModifier('sigmoid', add_first=True, eps=0),
                      condition_merger=CatCondAsList(cond_preproc=None, cond_is_list=True)
                      )
    f = InvertibleSequential(
        ActNorm(n_chans * 2, 'exp', ),
        InvPermute(n_chans * 2, fixed=False, use_lu=True),
        c,
    )
    return f

def unwrap_single_element(x):
    assert len(x) == 1
    return x[0]


def create_eeg_glow_down(
        n_up_block, n_chans,
        hidden_channels, kernel_length, n_mixes, n_blocks, init_dist_std=1e-1):
    n_up_0, n_up_1, n_up_2 = [SelectNode(n_up_block, i) for i in (0, 1, 2)]

    # maybe add n_down_2 as well, postprocessing

    flow_down_1 = InvertibleSequential(*[
        flow_block_down_1(n_chans, hidden_channels) for _ in range(n_blocks)])
    cond_preproc = nn.Sequential(
        nn.Linear(n_chans * 16, 128),
        nn.ELU(),
        nn.Linear(128, n_chans * 16),
        NoLogDet(ViewAs((-1, n_chans * 16), (-1, n_chans * 8, 2))))
    flow_down_1 = ConditionTransformWrapper(flow_down_1, cond_preproc=cond_preproc)

    n_down_1 = ConditionalNode(n_up_1, flow_down_1, condition_nodes=n_up_2)

    flow_down_0 = InvertibleSequential(*[
        flow_block_down_0(n_chans, hidden_channels, kernel_length) for _ in range(n_blocks)])
    cond_preproc = ApplyToListNoLogdets(
        nn.Sequential(
            nn.Conv1d(n_chans * 4, hidden_channels, 5, padding=5 // 2),
            nn.ELU(),
            nn.Conv1d(hidden_channels, n_chans * 4, 5, padding=5 // 2),
        ),
        nn.Sequential(
            nn.Linear(n_chans * 16, 128),
            nn.ELU(),
            nn.Linear(128, n_chans * 16),
            NoLogDet(ViewAs((-1, n_chans * 16), (-1, n_chans * 8, 2))), ))

    flow_down_0 = ConditionTransformWrapper(flow_down_0, cond_preproc=cond_preproc)

    n_down_0 = ConditionalNode(n_up_0, flow_down_0, condition_nodes=[n_down_1, n_up_2])

    block_a_out = InvertibleSequential(
        Flatten2d(),
    )
    block_b_out = InvertibleSequential(
        Flatten2d(),
    )
    block_c_out = InvertibleSequential(
        Flatten2d(),
    )

    n_0_out = Node(n_down_0, block_a_out)
    n_1_out = Node(n_down_1, block_b_out)
    n_2_out = Node(n_up_2, block_c_out)

    n_merged = CatAsListNode([n_0_out, n_1_out, n_2_out])

    dist0 = PerDimWeightedMix(2, n_mixes=n_mixes, n_dims=64 * n_chans // 2,
                              optimize_mean=True, optimize_std=True, init_std=init_dist_std)
    dist1 = PerDimWeightedMix(2, n_mixes=n_mixes, n_dims=64 * n_chans // 4,
                              optimize_mean=True, optimize_std=True, init_std=init_dist_std)
    dist2 = PerDimWeightedMix(2, n_mixes=n_mixes, n_dims=64 * n_chans // 4,
                              optimize_mean=True, optimize_std=True, init_std=init_dist_std)
    # architecture plan:
    n_dist = Node(n_merged, ApplyToList(dist0, dist1, dist2))
    net = n_dist
    return net