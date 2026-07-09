import multiprocessing as mp
from functools import partial
from scipy.io import loadmat
from loguru import logger
import numpy as np
from einops import repeat

_EEG_FILES = [
    ['1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
     '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
     '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
     '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
     '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
    ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
     '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
     '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
     '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
     '13_20151125.mat', '14_20151208.mat', '15_20150514.mat'],
    ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
     '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
     '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
     '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
     '13_20161130.mat', '14_20151215.mat', '15_20150527.mat'],
]

_SES_LABELS = [
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
]

_FEATURE_INDEX = {
    "de_movingAve": 0, "de_LDS": 1, "psd_movingAve": 2, "psd_LDS": 3,
}


def _parallel_read_seed_iv_feature(feature_id, dir_path, file):
    n_samples = 100
    subject_data = loadmat(f"{dir_path}/{file}")
    keys = list(subject_data.keys())[3:]  # skip __header__, __version__, __globals__
    trail_datas = []
    for i in range(24):
        trail_data = list(np.array(subject_data[keys[i * 4 + feature_id]]).transpose((1, 0, 2)))
        if (len(trail_data) - n_samples) > 0:
            logger.info("len of trial{}: {}", i, len(trail_data))
            pos = len(trail_data) - n_samples
            trail_data = trail_data[pos:]

        trail_datas.append(trail_data)
    return trail_datas


def load_seed_iv(
    dataset_path: str,
    feature_type: str = "de_LDS",
    trim_trial_start_pct: float = 0.0,
) -> tuple[list, list, int, int, int, int]:
    """
    Load SEED-IV pre-extracted features from eeg_feature_smooth/.

    Returns:
        data:   list, shape (session=3, subject=15, trial=24, sample=var, electrode=62, band=5)
        labels: list, shape (session=3, subject=15, trial=24, sample=var) — values 0..3
        num_subjects=15, num_electrodes=62, num_features=5, num_classes=4
    """
    logger.info(f"Loading SEED-IV dataset from path {dataset_path} ...")

    dir_path = dataset_path.rstrip("/") + "/eeg_feature_smooth"
    feature_id = _FEATURE_INDEX[feature_type]

    # Build labels: (3, 15, 24), tiled across subjects
    label = np.array(_SES_LABELS)  # (3, 24)
    label = repeat(label, 'session trial -> session subject trial', subject=15)
    num_classes = 4
    label = label.tolist()

    # Load data: (session, subject), each element is a list of 24 trials
    eeg_data = [[] for _ in range(3)]
    for session_id in range(3):
        eeg_data[session_id] = [[] for _ in range(15)]

    for session_id, session_files in enumerate(_EEG_FILES):
        session_dir = f"{dir_path}/{session_id + 1}"
        logger.debug(f"Reading session {session_id + 1} from {session_dir}")
        with mp.Pool(processes=5) as pool:
            result_session = pool.map(
                partial(_parallel_read_seed_iv_feature, feature_id, session_dir),
                session_files,
            )
        for subject_id in range(15):
            eeg_data[session_id][subject_id] = result_session[subject_id]

    # Replicate labels: one label per window (all windows in a trial share the trial label)
    for session_id in range(3):
        for subject_id in range(15):
            for trial_id in range(24):
                trial_label = label[session_id][subject_id][trial_id]
                n_windows = len(eeg_data[session_id][subject_id][trial_id])
                label[session_id][subject_id][trial_id] = [trial_label] * n_windows

    # Optional: trim the start of each trial (e.g. to remove transition artifacts)
    if trim_trial_start_pct > 0.0:
        for session_id in range(3):
            for subject_id in range(15):
                for trial_id in range(24):
                    t = eeg_data[session_id][subject_id][trial_id]
                    skip = int(len(t) * trim_trial_start_pct / 100.0)
                    eeg_data[session_id][subject_id][trial_id] = t[skip:]
                    label[session_id][subject_id][trial_id] = label[session_id][subject_id][trial_id][skip:]

    return eeg_data, label, 15, 62, 5, num_classes
