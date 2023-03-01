from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.preprocessing import Preprocessor
from braindecode.preprocessing import preprocess, scale
from braindecode.preprocessing.windowers import create_windows_from_events
from torch.utils.data import Subset
import numpy as np
import torch as th


def load_and_preproc_bcic_iv_2a(
    subject_id, sfreq, low_cut_hz, high_cut_hz, exponential_standardize
):
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
        Preprocessor(
            "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
        ),  # Bandpass filter
        Preprocessor(fn="resample", sfreq=sfreq),
        Preprocessor(
            scale, factor=1e6 / 6, apply_on_array=True
        ),  # Convert from V to uV
    ]

    if exponential_standardize:
        factor_new = 1e-3 * (250 / sfreq)
        init_block_size = int(1000 * (sfreq / 250))
        preprocessors.append(Preprocessor(
            exponential_moving_standardize,  # Exponential moving standardization
            factor_new=factor_new,
            init_block_size=init_block_size,
            )
        )
    # Preprocess the data
    preprocess(dataset, preprocessors)
    return dataset


def create_window_dataset(preproced_set, class_names, trial_start_offset_samples):
    # Next, extract the 4-second trials from the dataset.
    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    existing_classes = set()
    for ds in preproced_set.datasets:
        unique_events = set(np.unique(ds.raw.annotations.description))
        existing_classes = existing_classes | unique_events
    assert all(
        [c in existing_classes for c in class_names]
    ), f"These classes do not exist: {set(class_names) - existing_classes}"
    class_mapping = {name: i_cls for i_cls, name in enumerate(class_names)}

    windows_dataset = create_windows_from_events(
        preproced_set,
        trial_start_offset_samples=trial_start_offset_samples,
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
    subject_id,
    class_names,
    split_valid_off_train,
    all_subjects_in_each_fold,
    sfreq,
    trial_start_offset_sec,
    low_cut_hz,
    high_cut_hz,
    exponential_standardize
    # sfreq 32 originally
):
    preproced_set = load_and_preproc_bcic_iv_2a(
        subject_id,
        sfreq,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        exponential_standardize=exponential_standardize,
    )
    trial_start_offset_samples = int(np.round(trial_start_offset_sec * sfreq))
    windows_dataset = create_window_dataset(
        preproced_set,
        class_names,
        trial_start_offset_samples=trial_start_offset_samples,
    )
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
    subject_id,
    class_names,
    split_valid_off_train,
    all_subjects_in_each_fold,
    sfreq,
    trial_start_offset_sec,
    low_cut_hz,
    high_cut_hz,
    exponential_standardize,
):
    preproced_set = load_and_preproc_bcic_iv_2a(
        subject_id,
        sfreq,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        exponential_standardize=exponential_standardize,
    )
    trial_start_offset_samples = int(np.round(trial_start_offset_sec * sfreq))
    windows_dataset = create_window_dataset(
        preproced_set,
        class_names,
        trial_start_offset_samples=trial_start_offset_samples,
    )
    train_set, valid_set, test_set = split_bcic_iv_2a_train_valid_test(
        windows_dataset, split_valid_off_train, all_subjects_in_each_fold
    )
    return train_set, valid_set, test_set


def load_and_preproc_hgd(
        subject_id, sfreq, low_cut_hz, high_cut_hz, exponential_standardize
):
    # using the moabb dataset to load our data
    if not (subject_id is None or hasattr(subject_id, "__len__")):
        subject_id = [subject_id]
    dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=subject_id)
    hgd_names = ['Fp2', 'Fp1', 'F4', 'F3', 'C4', 'C3', 'P4', 'P3', 'O2', 'O1', 'F8',
                 'F7', 'T8', 'T7', 'P8', 'P7', 'M2', 'M1', 'Fz', 'Cz', 'Pz']
    # Define preprocessing steps
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(fn=lambda x: np.clip(x, -800/1e6, 800/1e6)),
        Preprocessor(
            fn="set_eeg_reference",
            ref_channels="average",
        ),
        Preprocessor(fn='pick_channels', ch_names=hgd_names, ordered=True),
        Preprocessor(
            "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
        ),  # Bandpass filter
        Preprocessor(
            scale, factor=1e6, apply_on_array=True
        ),  # Convert from V to uV
        Preprocessor(fn=lambda x: x / 6),
        Preprocessor(fn="resample", sfreq=sfreq),
    ]

    if exponential_standardize:
        factor_new = 1e-3 * (250 / sfreq)
        init_block_size = int(1000 * (sfreq / 250))
        preprocessors.append(Preprocessor(
            exponential_moving_standardize,  # Exponential moving standardization
            factor_new=factor_new,
            init_block_size=init_block_size,
        )
        )
    # Preprocess the data
    preprocess(dataset, preprocessors)
    return dataset


def load_train_valid_test_hgd(
        subject_id,
        class_names,
        split_valid_off_train,
        all_subjects_in_each_fold,
        sfreq,
        trial_start_offset_sec,
        low_cut_hz,
        high_cut_hz,
        exponential_standardize,
):
    preproced_set = load_and_preproc_hgd(
        subject_id,
        sfreq,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        exponential_standardize=exponential_standardize,
    )
    trial_start_offset_samples = int(np.round(trial_start_offset_sec * sfreq))
    windows_dataset = create_window_dataset(
        preproced_set,
        class_names,
        trial_start_offset_samples=trial_start_offset_samples,
    )
    train_set, valid_set, test_set = split_hgd_train_valid_test(
        windows_dataset, split_valid_off_train, all_subjects_in_each_fold
    )
    return train_set, valid_set, test_set


def split_hgd_train_valid_test(
        windows_dataset, split_valid_off_train, all_subjects_in_each_fold
):
    train_whole_set = windows_dataset.split("run")["train"]
    eval_whole_set = windows_dataset.split("run")["test"]

    if split_valid_off_train:
        test_set = eval_whole_set
        if all_subjects_in_each_fold:
            train_sets_per_subject = train_whole_set.split('subject')
            train_sets = []
            valid_sets = []
            for subject in train_sets_per_subject:
                subject_train_set = train_sets_per_subject[subject]
                n_split = int(np.round(0.75 * len(subject_train_set)))
                train_set = Subset(subject_train_set, range(0, n_split))
                valid_set = Subset(subject_train_set, range(n_split, len(subject_train_set)))
                train_sets.append(train_set)
                valid_sets.append(valid_set)
            train_set = th.utils.data.ConcatDataset(train_sets)
            valid_set = th.utils.data.ConcatDataset(valid_sets)
        else:
            n_split = int(np.round(0.75 * len(train_whole_set)))
            train_set = Subset(train_whole_set, range(0, n_split))
            valid_set = Subset(train_whole_set, range(n_split, len(train_whole_set)))
    else:
        train_set = train_whole_set
        if all_subjects_in_each_fold:
            test_sets_per_subject = eval_whole_set.split('subject')
            valid_sets = []
            test_sets = []
            for subject in test_sets_per_subject:
                subject_test_set = test_sets_per_subject[subject]
                n_split = int(np.round(0.5 * len(subject_test_set)))
                valid_set = Subset(subject_test_set, range(0, n_split))
                test_set = Subset(subject_test_set, range(n_split, len(subject_test_set)))
                valid_sets.append(valid_set)
                test_sets.append(test_set)
            valid_set = th.utils.data.ConcatDataset(valid_sets)
            test_set = th.utils.data.ConcatDataset(test_sets)
        else:
            n_split = int(np.round(0.5 * len(eval_whole_set)))
            valid_set = Subset(eval_whole_set, range(0, n_split))
            test_set = Subset(eval_whole_set, range(n_split, len(eval_whole_set)))
    return train_set, valid_set, test_set
