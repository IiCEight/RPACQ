import numpy as np
from loguru import logger


class Metric:
    def __init__(self, num_subjects: int, num_session: int):
        self.accuracy = np.zeros((num_subjects, num_session))

    def update(self, subject_id: int, session_id: int, acc: float):
        if acc > self.accuracy[subject_id, session_id]:
            logger.info(
                "--> A new better acc {:<.4f} on subject {} session {}",
                acc, subject_id, session_id,
            )
            self.accuracy[subject_id, session_id] = acc

    def all_sessions_mean_acc(self) -> tuple[float, float]:
        return self.accuracy.mean(), self.accuracy.std()

    def two_best_sessions_mean_acc(self) -> tuple[float, float]:
        np.sort(self.accuracy, axis=1)
        return self.accuracy[:, :2].mean(), self.accuracy[:, :2].std()

    def one_best_session_mean_acc(self) -> tuple[float, float]:
        return self.accuracy.max(axis=1).mean(), self.accuracy.max(axis=1).std()
