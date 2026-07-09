import multiprocessing as mp
from functools import partial
from scipy.io import loadmat
from loguru import logger
import numpy as np
import awkward as ak
from einops import rearrange, repeat

eeg_files = [
                ['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
                  '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
                  '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
                  '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
                  '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
                ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
                  '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
                  '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
                  '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
                  '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
                ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
                  '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
                  '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
                  '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
                  '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
                ]

feature_index = {
    "de": 0, "de_lds": 1, "psd": 2, "psd_lds": 3, "dasm": 4, "dasm_lds": 5,
    "rasm": 6, "rasm_lds": 7, "asm": 8, "asm_lds": 9, "dcau": 10, "dcau_lds": 11
}

def load_seed(dataset_path: str, feature_type: str = "de_lds",
              trim_trial_start_pct: float = 0.0) -> tuple[list, list, int, int, int, int]:
    """
    feature_type: "raw", "de_lds"...

    return:
        data:list  shape: (session(3), subject(15), trial(15), sample(different), electrode, frequency band)
        label:list shape: (session(3), subject(15), trial(15), sample(different)) NOT ONE-HOT ENCODED, 
            value is 0, 1, 2 represent the emotion label.

    SEED dataset has two folders
    1. ExtractedFeatures: contains the extracted features for each subject and session.
    2. Preprocessed_EEG: contains the raw EEG data after preprocessing 
        like filtering and artifact removal.
    
    For ExtractedFeatures folder
        there are 3 folders named 1, 2, 3, each represent one session and one label.mat.
        label.mat contains the label for all trail in three sessions.
        label is a dict with a key 'label', and the value is a 1*15 array, 
        each element represent the emotion label of the stimulus. 
        
        And under each session folder, there are 15 .mat files, each 
        represent one subject.
        It's a dict with keys like 'de_LDS1', 'de_LDS2', 'psd_LDS1'..,
            de = Differential Entropy
            LDS = Linear Dynamic System (This is the smoothing method)
            1 = Clip Number (1-15) i.e., the trial number.
        NOTE: The shape each trial is different!
        In one value for key 'de_LDS1', 
            it is a (electrode, sample, frequency band) array,
            but the sample dimension is different for different trial.
    
    """
    logger.info(f"Loading SEED dataset from path {dataset_path} ...")

    if feature_type == "raw":
        pass
    else:
        dir_path = dataset_path + "/ExtractedFeatures"

    # Extract the label for all trail in three sessions, label shape : (15)

    
    label = loadmat(f"{dir_path}/label.mat");
    logger.debug(f"Type of label: {type(label)}\n keys of label: {list(label.keys())}")

    label = label['label']
    
    logger.debug(f"Type of label: {type(label)}")
    # since the label value is -1, 0, 1, we add 1 to make it 0, 1, 2 
    # for easier processing later. 
    # And reshape the label to (3, 15, 1) to match the shape of data
    label = repeat(label, '1 label -> session subject label',subject=15, session=3) + 1
    
    num_classes = len(np.unique(label))

    label = label.tolist()

    # Set index based on selected characteristics
    feature_id = feature_index[feature_type]

    # shape (session, subject) each element is a list of trial data, 
    # and each trial data is a (electrode, time window, frequency band) array.
    eeg_data = [[] for _ in range(3)]
    for session_id in range(3):
            eeg_data[session_id] = [[] for _ in range(15)]
    # Define a function to read a single MAT file
    for session_id, subject_file_of_one_session in enumerate(eeg_files):
        logger.debug(f"Reading session {session_id+1} files: {subject_file_of_one_session}")

        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            # note pool.map only supports functions with a single argument, 
            # so we use partial to fix the other arguments
            result_session = pool.map(
                partial(
                    parallel_read_seed_feature, 
                    feature_id, 
                    dir_path +f"/{session_id+1}", 
                    label
                ), 
                subject_file_of_one_session
            )
        
        for subject_id in range(15):
            eeg_data[session_id][subject_id] = result_session[subject_id]
    
    # extend the label shape to (15, 3, 15) to match the shape of data
    for session_id in range(3):
        for subject_id in range(15):
            for trial_id in range(15):
                current_lable = label[session_id][subject_id][trial_id]
                label[session_id][subject_id][trial_id] = (
                    [current_lable] * len(eeg_data[session_id][subject_id][trial_id])
                )

    if trim_trial_start_pct > 0.0:
        for session_id in range(3):
            for subject_id in range(15):
                for trial_id in range(15):
                    t = eeg_data[session_id][subject_id][trial_id]
                    skip = int(len(t) * trim_trial_start_pct / 100.0)
                    eeg_data[session_id][subject_id][trial_id] = t[skip:]
                    label[session_id][subject_id][trial_id] = label[session_id][subject_id][trial_id][skip:]

    # No sampling rate for the extracted features, since SEED did it already.
    return eeg_data, label, 15, 62, 5, num_classes

def parallel_read_seed_feature(feature_id, dir_path, label, file):
    subject_data = loadmat(f"{dir_path}/{file}")
    logger.debug(f"dir_path: {dir_path}, file: {file}")
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(15):
        trail_data = list(np.array(subject_data[keys[i * 12+feature_id]]).transpose((1, 0, 2)))
        # logger.info(f"Subject {file} session {dir_path[-1]} trail {i} data shape: {np.array(trail_data).shape}")
        trail_datas.append(trail_data)
    
    return trail_datas

