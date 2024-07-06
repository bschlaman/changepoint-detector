import argparse
import logging
import os
import sys
import textwrap

from bpyutils.formatting.colors import bld, blu, mag
import changepoint_detector.modules.detection.entry

logging.basicConfig(
    format="[%(levelname)-8s] (%(name)s) %(message)s",
    level=logging.DEBUG,
)

log = logging.getLogger(__name__)


APPLICATION_NAME = "Changepoint Detector"


def main():
    # making a parse of myself
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            f"""{bld(APPLICATION_NAME)}
            This tool detects significant regime changes
            in mean and variance in time series using various
            changepoint detection methods.

            It is intended only as a learning tool
            and should not be relied upon for anything important.

            Below are a list of modes.
            """
        ),
    )
    modules_arg_group = parser.add_mutually_exclusive_group(required=True)

    modules_arg_group.add_argument(
        "--file",
        "-f",
        metavar="name_of_datafile",
        help="relative path of input csv file",
    )

    args = parser.parse_args()

    bin_name = os.path.basename(sys.argv[0])

    log.info(mag(APPLICATION_NAME))
    log.info(f"{blu('try running ')}{bin_name} -h{blu(' for a list of modes')}")

    if not args.file:
        parser.error("must specify a csv file with time series data")

    changepoint_detector.modules.detection.entry.run(args.file)
