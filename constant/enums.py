from enum import Enum


class DatasetName(str, Enum):
    DEAP = "DEAP"
    SEED = "SEED"
    SEED_IV = "SEED_IV"


class LevelName(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    ERROR = "ERROR"


class TaskTypeName(str, Enum):
    SUBJECT_DEPENDENT = "dep"
    SUBJECT_INDEPENDENT = "indep"


class SplitTypeName(str, Enum):
    KFOLD = "kfold"
    LEAVE_ONE_SUBJECT_OUT = "loso"
    TRAIN_TEST_VALIDATION = "ttv"
