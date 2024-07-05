import csv
import logging
import os

import pandas as pd

log = logging.getLogger(__name__)

RELATIVE_DATA_LAKE_PATH = "../../../../data"


def _construct_filepath(path_in_datalake: str) -> str:
    """Returns the relative file path that can be accessed
    from this utility"""
    rel_path = os.path.join(
        os.path.dirname(__file__), RELATIVE_DATA_LAKE_PATH, path_in_datalake
    )
    if not (os.path.isfile(rel_path) and os.access(rel_path, os.R_OK)):
        raise Exception(f"cannot read from file: {rel_path}")
    return rel_path


def load_from_file_csv(path_in_datalake: str):
    """Load data from a csv file into memory"""
    rel_path = _construct_filepath(path_in_datalake)
    with open(rel_path) as csvfile:
        log.debug(f"loaded data file: {path_in_datalake}")
        reader = csv.DictReader(csvfile)
        log.debug(f"loaded data fields: {reader.fieldnames}")
        return list(reader)


def load_from_file_dataframe(path_in_datalake: str, **args):
    """Load data from a csv file into memory as a pandas DataFrame

    :param path_in_datalake: file location within the datalake dir
    :param index_col: column name to use as the index to the dataframe
    """
    rel_path = _construct_filepath(path_in_datalake)
    # detect common problems with CSVs sourced from the internet
    # this is hacky, since I load the file into memory.
    with open(rel_path, "rb") as f:
        content = f.read()
        if b"\r\n" in content:
            raise Exception("DOS file format detected!")
        if b",\n" in content:
            raise Exception("Trailing commas detected!")
    return pd.read_csv(rel_path, **args)
