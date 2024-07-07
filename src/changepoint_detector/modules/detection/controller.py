import numpy as np
from changepoint_detector.modules.detection import strategy


class Controller:
    def __init__(self):
        self.strategies: list[strategy.ChangepointDetectStrategy] = [
            strategy.OneRegimePerQuarter(),
            strategy.AMOC(),
            strategy.HiddenMarkovModel(1),
            strategy.HiddenMarkovModel(3),
            strategy.HiddenMarkovModel(5),
            strategy.HiddenMarkovModel(7),
            strategy.LikelihoodRatioMethod(),
        ]

    @staticmethod
    def _postprocess(res: np.ndarray) -> int:
        """Implement various postprocessing strategies here"""
        return len(res)

    def process_sync(
        self, data: np.ndarray
    ) -> dict[strategy.ChangepointDetectStrategy, int]:
        res: dict[strategy.ChangepointDetectStrategy, int] = {}
        for strat in self.strategies:
            res[strat] = self._postprocess(strat(data))
        return res

    def calculate_metric_aic(
        self, data: np.ndarray
    ) -> dict[strategy.ChangepointDetectStrategy, float]:
        res = {}
        for strategy in self.strategies:
            res[strategy] = strategy.aic(data)
        return res

    def calculate_metric_bic(
        self, data: np.ndarray
    ) -> dict[strategy.ChangepointDetectStrategy, float]:
        res = {}
        for strategy in self.strategies:
            res[strategy] = strategy.bic(data)
        return res
