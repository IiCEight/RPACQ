from re import split
from loguru import logger
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GroupShuffleSplit
from constant import CLI_arguments_enum


def merge_and_split_deap(data: list, labels: list, num_classes: int, task_type: str):
    """
    Split the data into training, testing, validation sets according
    to the task type (subject-dependent or subject-independent)

    subject-independent: 
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
        with shape (sample_new, electrode, feature) and (sample_new, class)
    
    subject-dependent: 
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
        with shape (subject, sample_new, electrode, feature) and 
        (subject, sample_new, class) 
    """
    data = np.array(data)
    labels = np.array(labels)

    # Transpose (session, subject, trial, sample, electrode, feature) into
    #     (subject, session, trial, sample, electrode, feature)
    # And transpose labels into (subject, session, trial, sample, class)
    data = data.transpose(1, 0, 2, 3, 4, 5)     # put subject to first dimension
    labels = labels.transpose(1, 0, 2, 3, 4)          # put subject to first dimension

    if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_INDEPENDENT:
        return get_subject_independent_splits(data, labels)
    else:
        return get_subject_dependent_splits(data,labels)

def merge_and_split_seed(data: list, labels: np.ndarray, num_classes: int, task_type: str, split_type: str):
    """
    input data shape (session, subject, trail, sample(different), electrode, band)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
    For subject-independent (sample_new, electrode, feature) and (sample_new, class)
    For subject-dependent (subject, sample_new, electrode, feature) and (subject, sample_new, class) 
    """

    logger.info("Start merging and splitting SEED dataset...")

    if task_type == CLI_arguments_enum.TaskTypeName.SUBJECT_INDEPENDENT:
        return subject_independent_splits(data, labels, split_type)
    else:
        return subject_dependent_splits(data, labels, split_type)
        
def subject_independent_splits(data, labels, test_fraction=0.15, val_fraction=0.1):
    """
    input data shape (session, subject, trail, sample(different), electrode, band)

    return (sample_new, electrode, feature) and (sample_new, class)
    """
    split_data= [[], [], [], [], [], []]

    num_session = len(data)
    num_subject = len(data[0])

    # shuffle the subject order, and the last subject will be used as test set,
    # and the last second subject will be used as validation set
    subject_split = np.arange(num_subject)
    np.random.shuffle(subject_split)

    # split data into 3 parts (num_subject_test, num_subject_val, num_subject_train)
    num_subject_test = int(num_subject * test_fraction)
    num_subject_val = int(num_subject * val_fraction)
    num_subject_train = num_subject - num_subject_test - num_subject_val
    
    # construct the map from subject id to split id (train, val, test)
    split_indx = [0] * num_subject_train + [1] * num_subject_val + [2] * num_subject_test
    split_map = zip(subject_split, split_indx)
    split_map = sorted(split_map, key=lambda x: x[0])

    for session_id in range(num_session):
        for subject_id in range(num_subject):
            for trail, label in zip(data[session_id][subject_id], labels[session_id][subject_id]):
                split_id = split_map[subject_id][1]
                split_data[split_id * 2].extend(trail)
                split_data[split_id * 2 + 1].extend(label)


    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data
    logger.debug("Finished merging and splitting data. The shape of train data {}, "
                 "The shape of train labels {}", np.array(train_data).shape,
                 np.array(train_labels).shape)
    return np.array(train_data), np.array(train_labels), np.array(val_data), np.array(val_labels), np.array(test_data), np.array(test_labels)

def subject_dependent_splits(
    data, labels, test_fraction=0.15, val_fraction=0.1
):
    """
        input data shape (session, subject, trail, sample(different), electrode, band)

        For each subject, we need to split the data into training, testing, 
        validation sets.
        
        We use Mixed-Session Validation, which means that the session can be mixed 
        in the training, validation, and testing sets. 
        
        But the trial cannot be mixed in the training, validation, and testing sets
        to prevent data leakage.
            
    """

    num_session = len(data)
    num_subject = len(data[0])
    num_trial = len(data[0][0])
    split_data= [[[] for _ in range(num_subject)] for _ in range(6)]

    num_trial_each_subject = num_trial * num_session

    trial_split = np.arange(num_trial_each_subject)
    np.random.shuffle(trial_split)
    
    num_trial_test = int(num_trial_each_subject * test_fraction)
    num_trial_val = int(num_trial_each_subject * val_fraction)
    num_trial_train = num_trial_each_subject - num_trial_test - num_trial_val

    # construct the map from subject id to split id (train, val, test)
    split_index = [0] * num_trial_train + [1] * num_trial_val + [2] * num_trial_test
    split_map = zip(trial_split, split_index)
    split_map = sorted(split_map, key=lambda x: x[0])


    for subject_id in range(num_subject):
        for session_id in range(num_session):
            for trial_id, (trail, label) in enumerate(zip(data[session_id][subject_id], labels[session_id][subject_id])):
                split_id = split_map[session_id * num_trial +trial_id][1]
                split_data[split_id * 2][subject_id].append(trail)
                split_data[split_id * 2 + 1][subject_id].append(label)


    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data
    logger.debug("Finished merging and splitting data. The shape of train data {}, " \
                 "The shape of train labels {}",
                 np.array(train_data).shape, np.array(train_labels).shape)
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def get_subject_independent_splits(data, labels, test_fraction=0.15, val_fraction=0.1):
    """
    data shape: (subject, session, trial, sample, electrode, feature)
    labels shape: (subject, session, trial, sample, class)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
    with shape (sample_new, electrode, feature) and (sample_new, class) since there is a merge.
    We need to keep the same subject in the same split to prevent data leakage.
    """

    # merge into shape (subject, sample_new, electrode, feature) and 
    # (subject, sample_new, class)
    data = data.reshape(data.shape[0], -1, data.shape[4], data.shape[5])
    labels = labels.reshape(labels.shape[0], -1, labels.shape[4])

    logger.debug(f"Shape of data after merge: {data.shape}, shape of labels after merge: {labels.shape}")

    num_subject = data.shape[0]
    # shuffle the subject order, and the last subject will be used as test set,
    # and the last second subject will be used as validation set
    subject_split = np.arange(num_subject)
    np.random.shuffle(subject_split)

    # split data into 3 parts (num_subject_test, num_subject_val, num_subject_train)
    num_subject_test = int(num_subject * test_fraction)
    num_subject_val = int(num_subject * val_fraction)
    num_subject_train = num_subject - num_subject_test - num_subject_val

    train_data, train_labels = (
        data[subject_split[:num_subject_train]],
        labels[subject_split[:num_subject_train]],
    )
    val_data, val_labels = (
        data[subject_split[num_subject_train : num_subject_train + num_subject_val]],
        labels[subject_split[num_subject_train : num_subject_train + num_subject_val]],
    )
    test_data, test_labels = (
        data[subject_split[-num_subject_test:]],
        labels[subject_split[-num_subject_test:]],
    )

    # Since it is a subject-independent split, we need
    # to flatten the data and labels into shape (sample, electrode, feature)
    # and (sample, class)
    train_data = train_data.reshape(-1, train_data.shape[2], train_data.shape[3])
    train_labels = train_labels.reshape(-1, train_labels.shape[2])
    val_data = val_data.reshape(-1, val_data.shape[2], val_data.shape[3])
    val_labels = val_labels.reshape(-1, val_labels.shape[2])
    test_data = test_data.reshape(-1, test_data.shape[2], test_data.shape[3])
    test_labels = test_labels.reshape(-1, test_labels.shape[2])

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def get_subject_dependent_splits(
    data, labels, test_fraction=0.15, val_fraction=0.1
):
    """
    data shape: (subject, session, trial, sample, electrode, feature)
    labels shape: (subject, session, trial, sample, class)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
    with shape (subject, sample_new, electrode, feature) and (subject, sample_new, class) since
    We need to keep the same trial in the same split to prevent data leakage
    """

    num_subject = data.shape[0]
    num_session = data.shape[1]
    num_trial = data.shape[2]
    num_electrode = data.shape[4]
    feature = data.shape[5]
    num_classes = labels.shape[4]

    # merge into shape (subject, session * trial, sample, electrode, feature)
    data = data.reshape(data.shape[0], -1, data.shape[3], data.shape[4], data.shape[5])
    labels = labels.reshape(labels.shape[0], -1, labels.shape[3], labels.shape[4])

    train_data, train_labels, val_data, val_labels, test_data, test_labels = (
        [], [], [], [], [], []
    )

    # since each subject is trained independently, we need to split the data for
    # each subject separately
    for s_i, subject_data in enumerate(data):
        # each subject_data shape is (session * trial, sample, electrode, feature)
        # And there are num_session * num_trial trials in total 
        num_total_trial = num_session * num_trial
        
        assert num_total_trial == subject_data.shape[0]
        # create index for trial, and shuffle it.
        split_trial = np.arange(num_total_trial)
        np.random.shuffle(split_trial)

        # split data into 3 parts (num_trial_test, num_trial_val, num_trial_train)
        num_trial_test = int(num_total_trial * test_fraction)
        num_trial_val = int(num_total_trial * val_fraction)
        num_trial_train = num_total_trial - num_trial_test - num_trial_val

        train_data.append(subject_data[split_trial[:num_trial_train]])
        train_labels.append(labels[s_i][split_trial[:num_trial_train]])
        val_data.append(subject_data[split_trial[num_trial_train : num_trial_train + num_trial_val]])
        val_labels.append(labels[s_i][split_trial[num_trial_train : num_trial_train + num_trial_val]])
        test_data.append(subject_data[split_trial[-num_trial_test:]])
        test_labels.append(labels[s_i][split_trial[-num_trial_test:]])

        # reshape each subject's data and labels into 
        # (sample, electrode, feature) and (sample, class)
        train_data[-1] = np.array(train_data[-1]).reshape(-1, num_electrode, feature)
        train_labels[-1] = np.array(train_labels[-1]).reshape(-1, num_classes)
        val_data[-1] = np.array(val_data[-1]).reshape(-1, num_electrode, feature)
        val_labels[-1] = np.array(val_labels[-1]).reshape(-1, num_classes)
        test_data[-1] = np.array(test_data[-1]).reshape(-1, num_electrode, feature)
        test_labels[-1] = np.array(test_labels[-1]).reshape(-1, num_classes)

        logger.debug(
            f"Finished splitting data for subject {s_i}. "
            + f"train_data shape: {train_data[-1].shape}, train_labels "
            + f"shape: {train_labels[-1].shape}, ")

    return train_data, train_labels, val_data, val_labels, test_data, test_labels