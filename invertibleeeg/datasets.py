from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import Preprocessor
from braindecode.datautil.preprocess import preprocess, scale
from braindecode.datautil.windowers import create_windows_from_events
from torch.utils.data import Subset
import numpy as np
import torch as th


def load_and_preproc_bcic_iv_2a(subject_id):
    # using the moabb dataset to load our data
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])
    sfreq = 32
    # Define preprocessing steps
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(fn='set_eeg_reference', ref_channels='average', ),
        Preprocessor(fn='resample', sfreq=sfreq),
        Preprocessor(scale, factor=1e6 / 6, apply_on_array=True),  # Convert from V to uV
    ]
    # Preprocess the data
    preprocess(dataset, preprocessors)
    return dataset


def create_window_dataset(preproced_set, class_names):
    # Next, extract the 4-second trials from the dataset.
    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    class_mapping = {name: i_cls for i_cls, name in enumerate(class_names)}

    windows_dataset = create_windows_from_events(
        preproced_set,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
        mapping=class_mapping,
    )
    return windows_dataset


def split_bcic_iv_2a(windows_dataset, split_valid_off_train):
    train_whole_set = windows_dataset.split('session')['session_T']
    if split_valid_off_train:
        n_split = int(np.round(0.75 * len(train_whole_set)))
        valid_set = Subset(train_whole_set, range(n_split, len(train_whole_set)))
        train_set = Subset(train_whole_set, range(0, n_split))
    else:
        train_set = train_whole_set
        valid_set = windows_dataset.split('session')['session_E']

    return train_set, valid_set


def load_train_valid_bcic_iv_2a(subject_id, class_names, split_valid_off_train):
    preproced_set = load_and_preproc_bcic_iv_2a(subject_id)
    windows_dataset = create_window_dataset(preproced_set, class_names)
    train_set, valid_set = split_bcic_iv_2a(windows_dataset, split_valid_off_train)
    return train_set, valid_set

