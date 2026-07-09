from loguru import logger
import numpy as np

from data.preprocess.utils import label_process, segment_data


def preprocess_seed(
    data: list,
    labels: list,
    sampling_rate: int,
    num_electrodes: int,
    sample_length: int,
    stride: int,
    label_type: str,
):
    """
    Preprocess SEED dataset if needed

    return:
    """
    logger.info("Preprocessing SEED dataset ...")

    # Add any preprocessing steps here if needed


    data, num_feature = segment_data(data, sample_length=sample_length, stride=stride)

    data, labels, num_classes = label_process(
        data, labels, None, onehot=False, label_used=None
    )


    logger.debug(
        "Finished preprocessing SEED dataset. Len of data(session): {}, "
        "data[0](subject): {}, data[0][0](trial): {}, data[0][0][0](sample): {}, "
        "shape of labels: {},num_electrodes: {}, num_feature: {}, num_classes: {}",
        len(data),
        len(data[0]),
        len(data[0][0]),
        len(data[0][0][0]),
        data[0][0][0][0].shape,
        num_electrodes,
        num_feature,
        num_classes,
    )

    return data, labels, num_electrodes, num_feature, num_classes
