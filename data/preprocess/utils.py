from loguru import logger
import numpy as np


def segment_data(data, sample_length, stride):
    """
    Segment the data using a sliding window approach.
    A length of the window is defined by sample_length,
    and the step size between windows is defined by stride.

    return data after segmentation and the length of each seg_data_points
    i.e., the feature dimension.

    feature:
    input: original band features of EEG signal
    output:
    input shape -> data:  (session, subject, trail, time window, electrode, band)
    (sampling twice, first by time window, then by sample)
    output shape -> data:  (session, subject, trail, sample, time window, electrode, band)
    if sample_length == 1, then the output shape -> data: 
        (session, subject, trail, sample(time window), electrode, band)

    raw_data:
    input: original data of EEG signal
    input shape -> data: (session, subject, trail, electrode, data_points)
    output shape -> data: (session, subject, trail, sample, electrode, seg_data_points)
    """

    logger.debug("Segmenting data ...")

    if sample_length  == 1:
        logger.info("Sample length is 1, so use the original time window as sample")

        return data, len(data[0][0][0][0][0])

    seg_data = []
    for ses_i, session in enumerate(data):
        seg_session = []
        for sub_i, subject in enumerate(data[ses_i]):
            seg_sub = []
            seg_sub_label = []
            for t_i, trail in enumerate(data[ses_i][sub_i]):
                seg_trail = None
                trail = np.array(trail)
                if len(trail.shape) == 3:
                    # trail shape -> (time window, channel, band)
                    trail = np.asarray(trail)
                    num_sample = (len(trail) - sample_length) // stride + 1
                    seg_trail = np.zeros(
                        (num_sample, sample_length, len(trail[0]), len(trail[0][0]))
                    )
                    # Cutting a one-dimensional array through a sliding window
                    # to form a two-dimensional array
                    for i in range(num_sample):
                        seg_trail[i] = trail[i * stride : i * stride + sample_length]
                elif len(trail.shape) == 2:
                    # trail shape -> (channel, data_points)

                    # Use a window sliding to segment the data
                    # window size: sample_length
                    # every step size: stride
                    # so samples can overlap

                    num_sample = (len(trail[0]) - sample_length) // stride + 1
                    seg_trail = np.zeros((num_sample, len(trail), sample_length))
                    for i in range(num_sample):
                        seg_trail[i] = trail[:, i * stride : i * stride + sample_length]
                seg_sub.append(seg_trail)
            seg_session.append(seg_sub)
        seg_data.append(seg_session)

    logger.debug("Finished segmenting. Data shape: {}", np.array(seg_data).shape)

    if len(seg_data[0][0][0].shape) == 4:
        return seg_data, len(seg_data[0][0][0][0][0][0])
    elif len(seg_data[0][0][0].shape) == 3:
        return seg_data, len(seg_data[0][0][0][0][0])


def label_process(
    data: list, label: list, bounds: list = None, onehot=True, label_used: list = None
) -> tuple[list, list, int]:
    """
    input shape -> label: (session, subject, trail)
    output shape -> label: (session, subject, trail, sample)

    bounds are used to process the label into binary classification,
    if bounds is not None, then the label will be processed into binary classification
    according to the bounds, otherwise, the label will be processed into multi-class
    classification according to the unique values of the label.
    bounds shape -> 2, high emotion state > bounds[1], low emotion state < bounds[0]
    if dataset is hci, deap, dreamer, then label will be ordered by valence, arousal,
    dominance, liking

    return 
        processed data, processed label, num_classes(e.g.,
        2 forbinary classification).
    """

    logger.debug("Processing labels ...")

    available_label = ["valence", "arousal", "dominance", "liking"]
    if label_used is None:
        label_used = ["valence"]
    used_id = [available_label.index(item) for item in label_used]
    if type(label[0][0][0]) is np.ndarray:
        num_classes = np.power(2, len(used_id))
    else:
        num_classes = len(np.unique(label))
    new_label = []
    new_data = []
    for ses_i, ses_label in enumerate(label):
        new_ses_label = []
        new_ses_data = []
        for sub_i, sub_label in enumerate(ses_label):
            new_sub_label = []
            new_sub_data = []
            for trail_i, trail_label in enumerate(sub_label):
                new_trail_label = []
                new_trail_data = data[ses_i][sub_i][trail_i]
                num_sample = len(new_trail_data)
                if type(trail_label) is np.ndarray:
                    pro_label = []
                    for value_id in used_id:
                        value = trail_label[value_id]
                        if value <= bounds[0]:
                            pro_label.append(0)
                        elif value >= bounds[1]:
                            pro_label.append(1)
                    # pro_label shape -> (num_used_label, 2)
                    # processing into the ordinary labels
                    if len(pro_label) == len(used_id):
                        trail_label = int("".join(str(i) for i in pro_label), 2)
                    else:
                        # discard the data and label
                        continue
                if onehot:
                    oh_code = np.zeros((1, num_classes), dtype="int32")
                    # print(trail_label)
                    oh_code[0][trail_label] = 1
                    trail_label = oh_code
                    new_trail_label = np.tile(trail_label, (num_sample, 1))
                else:
                    trail_label = np.ones(1, dtype="int32") * trail_label
                    new_trail_label = np.tile(trail_label, num_sample)
                new_sub_data.append(new_trail_data)
                new_sub_label.append(new_trail_label)
            new_ses_label.append(new_sub_label)
            new_ses_data.append(new_sub_data)
        new_label.append(new_ses_label)
        new_data.append(new_ses_data)

    logger.debug("Processed len of label(session) {}, lable[0](subject){},"
                 " lable[0][0](trial){}, shape of lable[0][0][0](sample) {}",
                 len(new_label), len(new_label[0]), len(new_label[0][0]), new_label[0][0][0].shape)

    return new_data, new_label, num_classes
