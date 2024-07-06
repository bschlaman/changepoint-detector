import abc
import logging
import numpy as np
from hmmlearn import hmm

from changepoint_detector.modules.detection.config import MIN_PARTITION_SIZE

log = logging.getLogger(__name__)

def compute_stat(a: np.ndarray) -> float:
    return a.mean()

class ChangepointDetectStrategy(abc.ABC):
    """Strategy for identifying changepoints"""

    @abc.abstractmethod
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        return np.array([1, 2, 3])


class RandChangepoints(ChangepointDetectStrategy):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        return np.random.randint(0, len(ts), np.random.randint(2, min(10, len(ts))))


class NChangepoints(ChangepointDetectStrategy):
    def __init__(self, n: int):
        self.n = n

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        return np.sort(
            np.random.randint(0, len(ts), np.random.randint(2, min(10, len(ts))))
        )


class AMOC(ChangepointDetectStrategy):
    """The changepoint occurs when the statistic
    is the most different between the subsequences"""
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        stat_diffs: dict[int, float] = {}
        for i in range(MIN_PARTITION_SIZE, len(ts) - MIN_PARTITION_SIZE +1):
            stat_diffs[i] = compute_stat(ts[:i]) - compute_stat(ts[i:])
        cpt = max(stat_diffs, key=lambda x: abs(stat_diffs[x]))
        return np.array([cpt])


class HiddenMarkovModel(ChangepointDetectStrategy):
    def __init__(self):
        self.model = hmm.GaussianHMM()

    def preprocess(self, ts: np.ndarray) -> np.ndarray:
        return ts.reshape(-1, 1)

    def fit(self, ts: np.ndarray):
        candidate_models = []
        for n in range(1, 9, 2):
            log.debug(f"[HMM] fitting {n=}")
            model = hmm.GaussianHMM(n_components=n, n_iter=100)
            model.fit(ts)
            candidate_models.append(model)
        best_model = min(candidate_models, key=lambda m: m.bic(ts))
        log.debug(f"[HMM] best model (BIC): {best_model}")
        self.model = best_model

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        ts = self.preprocess(ts)
        self.fit(ts)
        return self.model.predict(ts)


class LikelihoodRatioMethod(ChangepointDetectStrategy):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        return np.array([200])


class ChangepointMethod(ChangepointDetectStrategy):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        return np.array([30, 1000])


class BayesianMethod(ChangepointDetectStrategy):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        return np.array([400])
