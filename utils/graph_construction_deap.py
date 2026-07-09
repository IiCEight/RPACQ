import numpy as np
from utils.graph_construction import (
    format_adj_matrix_from_standard,
    STANDARD_1005_CHANNEL_LOCATION_DICT,
)

DEAP_CHANNEL_LIST = [
    'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
    'P3', 'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4',
    'F8', 'FC6', 'FC2', 'CZ', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8',
    'PO4', 'O2',
]


def get_deap_adj_from_standard() -> np.ndarray:
    return format_adj_matrix_from_standard(
        DEAP_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT
    )
