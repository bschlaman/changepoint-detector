import abc
import logging
import numpy as np
import pandas as pd
import scipy.stats
from hmmlearn import hmm
from pychangepoints import algo_changepoints

from changepoint_detector.modules.detection.config import MIN_PARTITION_SIZE

log = logging.getLogger(__name__)


def compute_stat(a: np.ndarray) -> float:
    return a.mean()


class ChangepointDetectStrategy(abc.ABC):
    """Strategy for identifying changepoints"""

    # TODO: prevent this function and its dependents
    # from being called before fitting
    def aic(self, ts: np.ndarray) -> float:
        return 2 * self.get_n_params() - 2 * self.get_log_likelihood(ts)

    def bic(self, ts: np.ndarray) -> float:
        return np.log(len(ts)) * self.get_n_params() - 2 * self.get_log_likelihood(ts)

    @property
    def display_name(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def get_n_params(self) -> int: ...

    @abc.abstractmethod
    def get_log_likelihood(self, ts: np.ndarray) -> float: ...

    @abc.abstractmethod
    def __call__(self, ts: np.ndarray) -> np.ndarray: ...


class OneRegimePerQuarter(ChangepointDetectStrategy):
    TRADING_DAYS_PER_QUARTER = 63

    def __init__(self):
        self.cpts = np.array([])

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        self.cpts = (
            np.arange(len(ts) // self.TRADING_DAYS_PER_QUARTER)
            * self.TRADING_DAYS_PER_QUARTER
        )[1:]
        return self.cpts

    @property
    def display_name(self):
        return "[BASELINE] " + super().display_name

    def get_n_params(self) -> int:
        return 1

    def get_log_likelihood(self, ts: np.ndarray) -> float:
        assert len(self.cpts)
        cpts = self.cpts.copy()
        if self.cpts[-1] != len(ts) - 1:
            np.append(cpts, len(ts) - 1)
        ll = 0
        for i, cpt in enumerate(cpts):
            segment = ts[0 if i == 0 else cpts[i - 1] : cpt]
            ll += scipy.stats.norm.logpdf(segment, loc=segment.mean(), scale=segment.std(ddof=1)).sum()
        return ll


class RandChangepoints(ChangepointDetectStrategy):
    def __init__(self, n: int):
        self.n = n
        self.cpts = np.array([])

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        assert self.n < len(ts)
        return np.sort(np.random.randint(0, len(ts), self.n))

    @property
    def display_name(self):
        return "[BASELINE] " + super().display_name + f" n={self.n}"

    def get_n_params(self) -> int:
        return self.n

    def get_log_likelihood(self, ts: np.ndarray) -> float:
        assert len(self.cpts)
        cpts = self.cpts.copy()
        if self.cpts[-1] != len(ts) - 1:
            np.append(cpts, len(ts) - 1)
        ll = 0
        for i, cpt in enumerate(cpts):
            segment = ts[0 if i == 0 else cpts[i - 1] : cpt]
            ll += scipy.stats.norm.logpdf(segment, loc=segment.mean(), scale=segment.std(ddof=1)).sum()
        return ll


class AMOC(ChangepointDetectStrategy):
    """The changepoint occurs when the statistic
    is the most different between the subsequences"""

    def __init__(self):
        self.changepoint = -1

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        stat_diffs: dict[int, float] = {}
        for i in range(MIN_PARTITION_SIZE, len(ts) - MIN_PARTITION_SIZE + 1):
            stat_diffs[i] = compute_stat(ts[:i]) - compute_stat(ts[i:])
        cpt = max(stat_diffs, key=lambda x: abs(stat_diffs[x]))
        self.changepoint = cpt
        return np.array([cpt])

    @property
    def display_name(self):
        return "[BASELINE] " + super().display_name

    def get_n_params(self) -> int:
        return 1

    def get_log_likelihood(self, ts: np.ndarray) -> float:
        assert self.changepoint != -1
        seg0, seg1 = ts[: self.changepoint], ts[self.changepoint :]
        return (
            scipy.stats.norm.logpdf(seg0, loc=ts.mean(), scale=ts.std(ddof=1)).sum()
            + scipy.stats.norm.logpdf(seg1, loc=ts.mean(), scale=ts.std(ddof=1)).sum()
        )


class HiddenMarkovModel(ChangepointDetectStrategy):
    def __init__(self, n_states: int):
        self.model = hmm.GaussianHMM()
        self.n_states = n_states

    def _preprocess(self, ts: np.ndarray) -> np.ndarray:
        log.debug(f"[HMM] preprocessing...")
        return ts.reshape(-1, 1)

    def _postprocess(self, states: np.ndarray) -> np.ndarray:
        log.debug(f"[HMM] postprocessing...")
        return np.where(states[:-1] != states[1:])[0] + 1

    @property
    def display_name(self):
        return (
            ("[BASELINE] " if self.n_states == 1 else "")
            + super().display_name
            + f" (n={self.n_states})"
        )

    # def fit(self, ts: np.ndarray):
    #     candidate_models = []
    #     for n in range(1, 9, 2):
    #         log.debug(f"[HMM] fitting {n=}")
    #         model = hmm.GaussianHMM(n_components=n, n_iter=100)
    #         model.fit(ts)
    #         candidate_models.append(model)
    #     best_model = min(candidate_models, key=lambda m: m.bic(ts))
    #     log.debug(f"[HMM] best model (BIC): {best_model}")
    #     self.model = best_model
    def fit(self, ts: np.ndarray):
        log.debug(f"[HMM] fitting n={self.n_states}")
        model = hmm.GaussianHMM(n_components=self.n_states, n_iter=100)
        model.fit(ts)
        self.model = model

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        ts = self._preprocess(ts)
        self.fit(ts)
        states = self.model.predict(ts)
        return self._postprocess(states)

    # note that hmm.GaussianHMM already has a builtin `bic` method,
    # but specifying following my api is more flexible for other penalties
    def get_n_params(self) -> int:
        return sum(self.model._get_n_fit_scalars_per_param().values())

    def get_log_likelihood(self, ts: np.ndarray) -> float:
        ts = self._preprocess(ts)
        return self.model.score(ts)


class LikelihoodRatioMethod(ChangepointDetectStrategy):
    def __init__(self):
        self.cpts = np.array([])

    def _preprocess(self, ts: np.ndarray) -> pd.DataFrame:
        log.debug(f"[LRM] preprocessing...")
        return pd.DataFrame(ts, columns=["logret"])

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        df = self._preprocess(ts)
        cpts, _ = algo_changepoints.pelt(df, pen_=3, minseg=2, method="mbic_meanvar")
        log.debug(f"[LRM] found {len(cpts)} changepoints")
        self.cpts = np.sort(cpts)
        return cpts

    def get_n_params(self) -> int:
        # number of changepoints + (var, mean) for each segment
        return len(self.cpts) + 2 * (1 + len(self.cpts))

    def get_log_likelihood(self, ts: np.ndarray) -> float:
        assert len(self.cpts)
        cpts = self.cpts.copy()
        if self.cpts[-1] != len(ts) - 1:
            np.append(cpts, len(ts) - 1)
        ll = 0
        for i, cpt in enumerate(cpts):
            segment = ts[0 if i == 0 else cpts[i - 1] : cpt]
            ll += scipy.stats.norm.logpdf(segment, loc=segment.mean(), scale=segment.std(ddof=1)).sum()
        return ll


class ChangepointMethod(ChangepointDetectStrategy):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BayesianMethod(ChangepointDetectStrategy):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        raise NotImplementedError
