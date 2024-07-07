import logging
import numpy as np
import pandas as pd
import prettytable
import changepoint_detector.utils.data_load
import changepoint_detector.modules.detection.controller
from changepoint_detector.modules.detection import strategy
from bpyutils.formatting.colors import bld, yel

log = logging.getLogger(__name__)


def run(path: str):
    """
    1. ingest
    2. preprocess
    3. controller
    4. compute strategy fan-out
    5. aggregator
    6. postprocessing
    7. view
    """
    log.debug(f"attempting to load {path}")
    df = changepoint_detector.utils.data_load.load_from_file_dataframe(
        path,
        index_col=0,
        parse_dates=True,
    )
    log.debug(f"loaded time series with {len(df)} rows")
    log.debug(f"using index: {df.index.__class__.__name__}")
    preprocess(df)
    controller = changepoint_detector.modules.detection.controller.Controller()
    res = controller.process_sync(df['logret'].to_numpy())
    display_res(res)


def preprocess(df: pd.DataFrame):
    security = df.columns[0]
    log.debug(f"ticker: {security}")
    df.rename(columns={security: "price"}, inplace=True)
    df["logret"] = df.apply(np.log).diff()
    df.dropna(inplace=True)


def display_res(res: dict[strategy.ChangepointDetectStrategy, int]):
    def _fmt_name(strat: strategy.ChangepointDetectStrategy) -> str:
        return strat.display_name
    pt = prettytable.PrettyTable([bld("strategy"), bld("num changepoints")])
    pt.float_format = ".5"
    pt.align = "l"
    for strat, chpts in res.items():
        pt.add_row([yel(_fmt_name(strat)), chpts])
    print(pt)
