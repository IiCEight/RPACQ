

import awkward as ak
from einops import rearrange
from loguru import logger
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from constant import enums as CLI_arguments_enum


def merge_and_split(data:list, labels:list, task_type, session_id, subject_id, split_ratio, data_random,
                    )-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_DEPENDENT:
        train_data, train_labels, test_data, test_labels = (
            split_data_wrt_trials(
                data[session_id][subject_id],
                labels[session_id][subject_id], split_ratio, data_random)
        )
        train_data, train_labels = merge_for_one_subject(train_data, train_labels)
        test_data, test_labels = merge_for_one_subject(test_data, test_labels)

    else:
        # For subject-independent setting, we leave current subject out as test data
        # and merge the rest subjects' data as train data.
        train_data, train_labels, test_data, test_labels = (
            split_data_wrt_subjects(
                data[session_id], labels[session_id], subject_id)
        )

        train_data, train_labels = merge_for_all_subjects(train_data, train_labels)

        test_data, test_labels = merge_for_one_subject(test_data, test_labels)

    return train_data, train_labels, test_data, test_labels


def merge_for_all_subjects(
    data: list,
    labels: list,
    merge_subject_dim = True,
)-> tuple[np.ndarray, np.ndarray]:
    """
    input:
        data:  list, shape (subject, trial, sample, electrode, feature)
        label: list, shape (subject, trial, sample)

    return:
        if not merge_subject_dim:
            data:  np.ndarray, shape (subject, new_sample (trial * sample), electrode, feature)
            label: np.ndarray, shape (subject, new_sample (trial * sample))

        else:
            data:  np.ndarray, shape (new_sample (subject  * trial * sample), electrode, feature)
            label: np.ndarray, shape (new_sample (subject * trial * sample))
    """
    logger.info(f"Merging data and labels....")

    grouped, grouped_labels = [], []
    for sub_trials, sub_labels in zip(data, labels):
        sub_grouped, sub_gl = [], []
        for trial, tlabels in zip(sub_trials, sub_labels):
            t = np.array(trial)
            l = np.array(tlabels)
            sub_grouped.append(t)
            sub_gl.append(l)
        grouped.append(np.concatenate(sub_grouped, axis=0))
        grouped_labels.append(np.concatenate(sub_gl, axis=0))
    data = np.stack(grouped, axis=0)
    labels = np.stack(grouped_labels, axis=0)

    if not merge_subject_dim:
        logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)
        return data, labels

    if merge_subject_dim:
        data = rearrange(data, "subject samples ... -> (subject samples) ...")
        labels = rearrange(labels, "subject samples -> (subject samples)")

    logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)

    return data, labels

def merge_for_one_subject(
    data: list,
    labels: list,
)-> tuple[np.ndarray, np.ndarray]:
    """
    input:
        data: list, shape (trial, sample(may different), electrode, feature)
        label: list, shape (trial, sample)

    return:
        data: np.ndarray, shape (new_sample (trial * sample), electrode, feature)
        label: np.ndarray, shape (new_sample (trial * sample))
    """
    logger.info(f"Merging data and labels....")

    grouped, grouped_labels = [], []
    for trial, tlabels in zip(data, labels):
        t = np.array(trial)
        l = np.array(tlabels)
        grouped.append(t)
        grouped_labels.append(l)
    data = np.concatenate(grouped, axis=0)
    labels = np.concatenate(grouped_labels, axis=0)

    logger.info("Finish merging! data shape: {}, label shape: {}", data.shape, labels.shape)

    return data, labels
    
def split_data_wrt_trials(data:list, labels:list, split_ratio:float, random = False
                          )->tuple[list, list, list, list]:
    """
    For one subject, we split 60% of the trials as train data 
        and the rest 40% as test data.
        
    input:
        data: list, shape (trial, sample, electrode, feature)
        label: list, shape (trial, sample)
    return:
        train_data: list, the same as input
        train_label: list, the same as input
        test_data: list, the same as input
        test_label: list, the same as input
    """
    data = ak.Array(data)
    labels = ak.Array(labels)

    # number of trails for each subject,
    num_trials = len(data)
    logger.debug(f"Number of trials: {num_trials}")
    num_trails_per_train = int(num_trials * split_ratio)
    logger.info("Number of trials for train {}", num_trails_per_train)
    mask = [True] * num_trails_per_train + [False] * (num_trials - num_trails_per_train)

    mask = np.array(mask)

    if random:
        np.random.shuffle(mask)

    # Use bool index of ak
    # convert back to list
    train_data = data[mask].to_list()
    train_labels = labels[mask].to_list()
    test_data = data[~mask].to_list()
    test_labels = labels[~mask].to_list()

    
    return train_data, train_labels, test_data, test_labels


def split_data_wrt_subjects(data:list, labels:list, subject_id:int
                          )->tuple[list, list, list, list]:
    """
    leave one subject(subject_id:int) as test set.
            
    input:
        data:  list, shape (subject, trial, sample, electrode, feature)
        label: list, shape (subject, trial, sample)
    return:
        train_data:  list, shape (subject - 1, trial, sample, electrode, feature)
        train_label: list, shape (subject - 1, trial, sample)
        test_data:   list, shape (trial, sample, electrode, feature)
        test_label:  list, shape (trial, sample)
    """
    # We need the function of boolean array index.
    data = ak.Array(data)
    labels = ak.Array(labels)

    # number for each subject,
    num_subjects = len(labels)

    mask = np.ones(num_subjects, dtype=bool)
    mask[subject_id] = False

    # convert back to list
    train_data = data[mask].to_list()
    train_labels = labels[mask].to_list()
    test_data = data[subject_id].to_list()
    test_labels = labels[subject_id].to_list()
    
    return train_data, train_labels, test_data, test_labels


def normalization_wrt_trial(data:list, type = 'min_max'):
    '''
    param {type}: min_max, z_score
    
    input:
        data: list shape (session, subject, trial, sample, electrode, feature)
        type: str, 'min_max' or 'z_score'

    return:
        data the same shape as input
    '''

    for session_id in range(len(data)):
        for subject_id in range(len(data[session_id])):
            for trial_id in range(len(data[session_id][subject_id])):
                    data[session_id][subject_id][trial_id]= normalize_one_trial(
                        data[session_id][subject_id][trial_id], type)

    return data

def zscore_wrt_subject(data: list) -> list:
    """Z-score normalisation matching Tianyang/MSMDAERNet pipeline.

    Each subject's samples are flattened to (N, 310), z-scored with per-feature
    mean/std computed over that subject's samples, then reshaped back.
    Sessions are normalised separately.

    input:  list shape (session, subject, trial, sample, electrode, feature)
    output: same shape, float32
    """
    for session_id in range(len(data)):
        for subject_id in range(len(data[session_id])):
            subj = data[session_id][subject_id]
            arr = ak.Array(subj)
            flat = ak.flatten(arr, axis=1)
            X = ak.to_numpy(flat).astype(np.float32)   # (N, 62, 5)
            N, E, F = X.shape
            X2d = X.reshape(N, E * F)                  # (N, 310)
            mean = X2d.mean(axis=0)
            std = X2d.std(axis=0)
            X2d = (X2d - mean) / (std + 1e-9)
            X3d = X2d.reshape(N, E, F)
            ret = ak.unflatten(X3d, ak.num(arr, axis=1), axis=0)
            data[session_id][subject_id] = ret.to_list()
    return data


def normalization_wrt_subject(data: list, type: str = 'min_max', band_major: bool = False):
    '''
    param {type}: min_max, z_score
    param band_major: if True, flatten in band-major order (N,5,62) before normalizing,
        matching utils_PCL.get_data_label_frommat. Default False keeps electrode-major (N,62,5).

    NOTE: Different sessions are separately normalized.

    input:
        data: list shape (session, subject, trial, sample, electrode, feature)
        type: str, 'min_max' or 'z_score'

    return:
        data the same shape as input
    '''

    for session_id in range(len(data)):
        for subject_id in range(len(data[session_id])):
            data[session_id][subject_id] = normalize_one_subject(
                data[session_id][subject_id], type, band_major=band_major)

    return data



def normalize_one_subject(data, type: str = 'min_max', band_major: bool = False):
    '''
    description: Normalizes a jagged EEG array of shape (trials, var_samples, 62, 5)
    param {type}: 'min_max', 'z_score'
    param band_major: if True, transpose to (N,5,62) before flattening to (N,310),
        so each column corresponds to one (band, electrode) pair — matches utils_PCL ordering.
    return: list
    '''
    data = ak.Array(data)

    # Flatten (trials, var_samples, 62, 5) → (N, 62, 5)
    flat_data = ak.flatten(data, axis=1)
    flat_np = ak.to_numpy(flat_data)   # (N, 62, 5)
    N, E, F = flat_np.shape

    if type == 'min_max':
        if band_major:
            # (N, 62, 5) → (N, 5, 62) → (N, 310): each column = one (band, electrode) pair
            flat_2d = flat_np.transpose(0, 2, 1).reshape(N, E * F)
        else:
            flat_2d = flat_np.reshape(N, E * F)

        # Cast to float32 BEFORE fit to match utils_PCL.get_data_label_frommat byte-for-byte.
        # Fitting on float64 then downcasting drifts at the LSB and propagates into KMeans.
        flat_2d = flat_2d.astype(np.float32)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        flat_2d = scaler.fit_transform(flat_2d).astype(np.float32)

        if band_major:
            flat_3d = flat_2d.reshape(N, F, E).transpose(0, 2, 1)  # back to (N, 62, 5)
        else:
            flat_3d = flat_2d.reshape(N, E, F)

        ret = ak.unflatten(flat_3d, ak.num(data, axis=1), axis=0)

    elif type == 'z_score':
        flat_np_work = flat_np.transpose(0, 2, 1) if band_major else flat_np
        x_mean = flat_np_work.mean(axis=0)
        x_std = flat_np_work.std(axis=0)
        flat_np_work = (flat_np_work - x_mean) / (x_std + 1e-8)
        if band_major:
            flat_np_work = flat_np_work.transpose(0, 2, 1)  # back to (N, 62, 5)
        ret = ak.unflatten(flat_np_work.astype(np.float32), ak.num(data, axis=1), axis=0)

    return ret.to_list()
def normalize_one_trial(data, type = 'min_max'):
    '''
    description:
    param {type}: min_max, z_score
    return {type}
    '''
    if type == 'min_max':
        _range = np.max(data) - np.min(data)
        ret = (data - np.min(data)) / _range
    elif type == 'z_score':
        x_mean = np.mean(data)
        x_std = np.std(data)
        ret = (data - x_mean) / x_std

    return ret