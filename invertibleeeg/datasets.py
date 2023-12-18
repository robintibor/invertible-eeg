from braindecode.datasets.moabb import MOABBDataset
import numpy as np
import torch as th
from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.preprocessing.windowers import create_windows_from_events
from torch.utils.data import Subset
import logging
log = logging.getLogger(__name__)


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
        Preprocessor(lambda x: x * 1e6/6
        ),  # Convert from V to uV
    ]

    if exponential_standardize:
        factor_new = 1e-3 * (250 / sfreq)
        init_block_size = int(1000 * (sfreq / 250))
        preprocessors.append(
            Preprocessor(
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
    subject_id,
    sfreq,
    low_cut_hz,
    high_cut_hz,
    exponential_standardize,
    sensors,
):
    # using the moabb dataset to load our data
    if not (subject_id is None or hasattr(subject_id, "__len__")):
        subject_id = [subject_id]
    dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=subject_id)
    C_sensors = [
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "C3",
        "Cz",
        "C4",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "FC3",
        "FCz",
        "FC4",
        "C5",
        "C1",
        "C2",
        "C6",
        "CP3",
        "CPz",
        "CP4",
        "FFC5h",
        "FFC3h",
        "FFC4h",
        "FFC6h",
        "FCC5h",
        "FCC3h",
        "FCC4h",
        "FCC6h",
        "CCP5h",
        "CCP3h",
        "CCP4h",
        "CCP6h",
        "CPP5h",
        "CPP3h",
        "CPP4h",
        "CPP6h",
        "FFC1h",
        "FFC2h",
        "FCC1h",
        "FCC2h",
        "CCP1h",
        "CCP2h",
        "CPP1h",
        "CPP2h",
    ]
    EEG_sensors = [
        "Fp1",
        "Fp2",
        "Fpz",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "M1",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "M2",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "POz",
        "O1",
        "Oz",
        "O2",
        "AF7",
        "AF3",
        "AF4",
        "AF8",
        "F5",
        "F1",
        "F2",
        "F6",
        "FC3",
        "FCz",
        "FC4",
        "C5",
        "C1",
        "C2",
        "C6",
        "CP3",
        "CPz",
        "CP4",
        "P5",
        "P1",
        "P2",
        "P6",
        "PO5",
        "PO3",
        "PO4",
        "PO6",
        "FT7",
        "FT8",
        "TP7",
        "TP8",
        "PO7",
        "PO8",
        "FT9",
        "FT10",
        "TPP9h",
        "TPP10h",
        "PO9",
        "PO10",
        "P9",
        "P10",
        "AFF1",
        "AFz",
        "AFF2",
        "FFC5h",
        "FFC3h",
        "FFC4h",
        "FFC6h",
        "FCC5h",
        "FCC3h",
        "FCC4h",
        "FCC6h",
        "CCP5h",
        "CCP3h",
        "CCP4h",
        "CCP6h",
        "CPP5h",
        "CPP3h",
        "CPP4h",
        "CPP6h",
        "PPO1",
        "PPO2",
        "I1",
        "Iz",
        "I2",
        "AFp3h",
        "AFp4h",
        "AFF5h",
        "AFF6h",
        "FFT7h",
        "FFC1h",
        "FFC2h",
        "FFT8h",
        "FTT9h",
        "FTT7h",
        "FCC1h",
        "FCC2h",
        "FTT8h",
        "FTT10h",
        "TTP7h",
        "CCP1h",
        "CCP2h",
        "TTP8h",
        "TPP7h",
        "CPP1h",
        "CPP2h",
        "TPP8h",
        "PPO9h",
        "PPO5h",
        "PPO6h",
        "PPO10h",
        "POO9h",
        "POO3h",
        "POO4h",
        "POO10h",
        "OI1h",
        "OI2h",
    ]
    hgd_names = [
        "Fp2",
        "Fp1",
        "F4",
        "F3",
        "C4",
        "C3",
        "P4",
        "P3",
        "O2",
        "O1",
        "F8",
        "F7",
        "T8",
        "T7",
        "P8",
        "P7",
        "M2",
        "M1",
        "Fz",
        "Cz",
        "Pz",
    ]
    sensor_names = {
        "10-20": hgd_names,
        "all": EEG_sensors,
        "C": C_sensors,
    }[sensors]

    # Define preprocessing steps
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(fn=lambda x: np.clip(x, -800 / 1e6, 800 / 1e6)),
        Preprocessor(
            fn="set_eeg_reference",
            ref_channels="average",
        ),
        Preprocessor(fn="pick_channels", ch_names=sensor_names, ordered=True),
        Preprocessor(
            "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
        ),  # Bandpass filter
        Preprocessor(lambda x : x * 1e6),  # Convert from V to uV
        Preprocessor(fn=lambda x: x / 6),
        Preprocessor(fn="resample", sfreq=sfreq),
    ]

    if exponential_standardize:
        factor_new = 1e-3 * (250 / sfreq)
        init_block_size = int(1000 * (sfreq / 250))
        preprocessors.append(
            Preprocessor(
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
    sensors,
):
    preproced_set = load_and_preproc_hgd(
        subject_id,
        sfreq,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        exponential_standardize=exponential_standardize,
        sensors=sensors,
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
            train_sets_per_subject = train_whole_set.split("subject")
            train_sets = []
            valid_sets = []
            for subject in train_sets_per_subject:
                subject_train_set = train_sets_per_subject[subject]
                n_split = int(np.round(0.75 * len(subject_train_set)))
                train_set = Subset(subject_train_set, range(0, n_split))
                valid_set = Subset(
                    subject_train_set, range(n_split, len(subject_train_set))
                )
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
            test_sets_per_subject = eval_whole_set.split("subject")
            valid_sets = []
            test_sets = []
            for subject in test_sets_per_subject:
                subject_test_set = test_sets_per_subject[subject]
                n_split = int(np.round(0.5 * len(subject_test_set)))
                valid_set = Subset(subject_test_set, range(0, n_split))
                test_set = Subset(
                    subject_test_set, range(n_split, len(subject_test_set))
                )
                valid_sets.append(valid_set)
                test_sets.append(test_set)
            valid_set = th.utils.data.ConcatDataset(valid_sets)
            test_set = th.utils.data.ConcatDataset(test_sets)
        else:
            n_split = int(np.round(0.5 * len(eval_whole_set)))
            valid_set = Subset(eval_whole_set, range(0, n_split))
            test_set = Subset(eval_whole_set, range(n_split, len(eval_whole_set)))
    return train_set, valid_set, test_set


def load_tuh_abnormal(n_recordings_to_load):
    n_max_minutes = 3
    sfreq = 64
    n_minutes = 2
    log.info("Load TUH train data...")
    # atm window stride determined automatically as n_preds_per_input, could also parametrize it

    data_path = "/data/datasets/TUH/EEG/tuh_eeg_abnormal/v2.0.0/edf/train/"
    if n_recordings_to_load:
        recording_ids = range(
            n_recordings_to_load
        )
    else:
        recording_ids = None

    train_set = TUHAbnormal(
        path=data_path,
         recording_ids=recording_ids,  #   loads the n chronologically first recordings
        target_name="pathological",  # age, gender, pathology
        preload=False,
        add_physician_reports=False,
    )

    log.info("Load TUH test data...")
    data_path = "/data/datasets/TUH/EEG/tuh_eeg_abnormal/v2.0.0/edf/eval/"
    test_set = TUHAbnormal(
        path=data_path,
        target_name="pathological",  # age, gender, pathology
        preload=False,
        add_physician_reports=False,
    )
    complete_set = BaseConcatDataset([train_set, test_set])

    log.info("Preprocess Data..")
    ar_ch_names = sorted(
        [
            "EEG A1-REF",
            "EEG A2-REF",
            "EEG FP1-REF",
            "EEG FP2-REF",
            "EEG F3-REF",
            "EEG F4-REF",
            "EEG C3-REF",
            "EEG C4-REF",
            "EEG P3-REF",
            "EEG P4-REF",
            "EEG O1-REF",
            "EEG O2-REF",
            "EEG F7-REF",
            "EEG F8-REF",
            "EEG T3-REF",
            "EEG T4-REF",
            "EEG T5-REF",
            "EEG T6-REF",
            "EEG FZ-REF",
            "EEG CZ-REF",
            "EEG PZ-REF",
        ]
    )

    preprocessors = [
        Preprocessor("crop", tmin=0, tmax=n_max_minutes * 60, include_tmax=True),
        Preprocessor(fn="pick_channels", ch_names=ar_ch_names, ordered=True),
        # convert from volt to microvolt, directly modifying the numpy array
        Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),
        Preprocessor(fn=lambda x: np.clip(x, -800, 800), apply_on_array=True),
        Preprocessor("set_eeg_reference", ref_channels="average"),
        Preprocessor(fn=lambda x: x / 30, apply_on_array=True),  # this seemed best
        Preprocessor(fn="resample", sfreq=sfreq),
        Preprocessor("crop", tmin=1 * 60, tmax=n_max_minutes * 60, include_tmax=False),
    ]
    # Preprocess the data
    preprocess(complete_set, preprocessors)

    log.info("Window Data..")
    window_set = create_fixed_length_windows(
        complete_set,
        preload=True,
        window_size_samples=n_minutes * 60 * sfreq,
        window_stride_samples=1,
        drop_last_window=True,
    )
    whole_train_set = window_set.split("train")["True"]

    log.info("Split Data..")
    subject_datasets = whole_train_set.split("subject")
    n_subjects = len(subject_datasets)

    n_split = int(np.round(n_subjects * 0.75))
    keys = list(subject_datasets.keys())
    train_sets = [d for i in range(n_split) for d in subject_datasets[keys[i]].datasets]
    train_set = BaseConcatDataset(train_sets)
    valid_sets = [
        d
        for i in range(n_split, n_subjects)
        for d in subject_datasets[keys[i]].datasets
    ]
    valid_set = BaseConcatDataset(valid_sets)
    test_set = window_set.split("train")["False"]
    return train_set, valid_set, test_set
