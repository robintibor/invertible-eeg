from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import Preprocessor
from braindecode.datautil.preprocess import preprocess, scale
from braindecode.datautil.windowers import create_windows_from_events
from torch.utils.data import Subset
import numpy as np


def load_and_preproc_bcic_iv_2a(subject_id, sfreq):
    # using the moabb dataset to load our data
    if not (subject_id is None or hasattr(subject_id, "__len__")):
        subject_id = [subject_id]
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_id)
    # Define preprocessing steps
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(
            fn="set_eeg_reference",
            ref_channels="average",
        ),
        Preprocessor(fn="resample", sfreq=sfreq),
        Preprocessor(
            scale, factor=1e6 / 6, apply_on_array=True
        ),  # Convert from V to uV
    ]
    # Preprocess the data
    preprocess(dataset, preprocessors)
    return dataset


def create_window_dataset(preproced_set, class_names):
    # Next, extract the 4-second trials from the dataset.
    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    existing_classes = set()
    for ds in preproced_set.datasets:
        unique_events = set(np.unique(ds.raw.annotations.description))
        existing_classes = existing_classes | unique_events
    assert all([c in existing_classes for c in class_names]), (
        f"These classes do not exist: {set(class_names) - existing_classes}")
    class_mapping = {name: i_cls for i_cls, name in enumerate(class_names)}

    windows_dataset = create_windows_from_events(
        preproced_set,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
        mapping=class_mapping,
    )
    return windows_dataset


def split_bcic_iv_2a(windows_dataset, split_valid_off_train, all_subjects_in_each_fold):
    train_whole_set = windows_dataset.split("session")["session_T"]
    eval_whole_set = windows_dataset.split("session")["session_E"]
    if split_valid_off_train:
        if all_subjects_in_each_fold:
            run_splitted = train_whole_set.split("run")
            train_set = BaseConcatDataset([run_splitted[f"run_{i}"] for i in range(4)])
            valid_set = BaseConcatDataset(
                [run_splitted[f"run_{i}"] for i in range(4, 6)]
            )
        else:
            n_split = int(np.round(0.75 * len(train_whole_set)))
            print("n split", n_split)
            train_set = Subset(train_whole_set, range(0, n_split))
            valid_set = Subset(train_whole_set, range(n_split, len(train_whole_set)))
    else:
        train_set = train_whole_set
        valid_set = eval_whole_set

    return train_set, valid_set


def load_train_valid_bcic_iv_2a(
    subject_id, class_names, split_valid_off_train, all_subjects_in_each_fold,
        sfreq,
        #sfreq 32 originally
):
    preproced_set = load_and_preproc_bcic_iv_2a(subject_id, sfreq)
    windows_dataset = create_window_dataset(preproced_set, class_names)
    train_set, valid_set = split_bcic_iv_2a(
        windows_dataset,
        split_valid_off_train,
        all_subjects_in_each_fold=all_subjects_in_each_fold,
    )
    return train_set, valid_set


def split_bcic_iv_2a_train_valid_test(
    windows_dataset, split_valid_off_train, all_subjects_in_each_fold
):
    train_whole_set = windows_dataset.split("session")["session_T"]
    eval_whole_set = windows_dataset.split("session")["session_E"]
    if split_valid_off_train:
        test_set = eval_whole_set
        if all_subjects_in_each_fold:
            run_splitted = train_whole_set.split("run")
            train_set = BaseConcatDataset([run_splitted[f"run_{i}"] for i in range(4)])
            valid_set = BaseConcatDataset(
                [run_splitted[f"run_{i}"] for i in range(4, 6)]
            )
        else:
            n_split = int(np.round(0.75 * len(train_whole_set)))
            train_set = Subset(train_whole_set, range(0, n_split))
            valid_set = Subset(train_whole_set, range(n_split, len(train_whole_set)))
    else:
        train_set = train_whole_set
        if all_subjects_in_each_fold:
            run_splitted = eval_whole_set.split("run")
            valid_set = BaseConcatDataset([run_splitted[f"run_{i}"] for i in range(4)])
            test_set = BaseConcatDataset(
                [run_splitted[f"run_{i}"] for i in range(4, 6)]
            )
        else:
            n_split = int(np.round(0.5 * len(eval_whole_set)))
            valid_set = Subset(eval_whole_set, range(0, n_split))
            test_set = Subset(eval_whole_set, range(n_split, len(eval_whole_set)))
    return train_set, valid_set, test_set


def load_train_valid_test_bcic_iv_2a(
    subject_id, class_names, split_valid_off_train, all_subjects_in_each_fold,
        sfreq
):
    preproced_set = load_and_preproc_bcic_iv_2a(subject_id, sfreq)
    windows_dataset = create_window_dataset(preproced_set, class_names)
    train_set, valid_set, test_set = split_bcic_iv_2a_train_valid_test(
        windows_dataset, split_valid_off_train, all_subjects_in_each_fold
    )
    return train_set, valid_set, test_set
