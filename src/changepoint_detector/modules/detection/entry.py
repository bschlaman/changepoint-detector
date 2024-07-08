import logging
import numpy as np
import pandas as pd
import prettytable
import changepoint_detector.utils.data_load
import changepoint_detector.modules.detection.controller
from changepoint_detector.modules.detection import strategy
from bpyutils.formatting.colors import bld, yel, dim

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

    log.info(f"fitting {len(controller.strategies)} models...")
    res = controller.process_sync(df["logret"].to_numpy())
    metric_aic = controller.calculate_metric_aic(df["logret"].to_numpy())
    metric_bic = controller.calculate_metric_bic(df["logret"].to_numpy())

    display_res(res, metric_aic, metric_bic)


def preprocess(df: pd.DataFrame):
    security = df.columns[0]
    log.debug(f"ticker: {security}")
    df.rename(columns={security: "price"}, inplace=True)
    df["logret"] = df.apply(np.log).diff()
    df.dropna(inplace=True)
    mask = df["logret"].abs() <= 0.0001
    df.drop(df[mask].index, inplace=True)


def display_res(
    num_changepoints: dict[strategy.ChangepointDetectStrategy, int],
    aic: dict[strategy.ChangepointDetectStrategy, float],
    bic: dict[strategy.ChangepointDetectStrategy, float],
):
    def _fmt_name(strat: strategy.ChangepointDetectStrategy) -> str:
        return strat.display_name

    pt = prettytable.PrettyTable(
        [bld("strategy"), bld("num changepoints"), bld("aic"), bld("bic")]
    )
    pt.float_format = ".5"
    pt.align = "l"
    for strat, n_cpts in num_changepoints.items():
        pt.add_row([yel(_fmt_name(strat)), n_cpts, aic[strat], bic[strat]])
    pt.add_rows([
        [dim("ChangepointMethods (TODO)"), "-", "-","-"],
        [dim("BayesianMethods (TODO)"), "-", "-","-"],
    ])
    print(pt)
