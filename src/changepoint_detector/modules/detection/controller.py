import numpy as np
from changepoint_detector.modules.detection import strategy

class Controller:
    def __init__(self):
        self.strategies = [strategy.RandChangepoints(), strategy.AMOC(), strategy.HiddenMarkovModel()]

    def process_sync(self, data: np.ndarray) -> dict[str, np.ndarray]:
        res = {}
        for strategy in self.strategies:
            res[strategy.__class__.__name__] = strategy(data)
        return res