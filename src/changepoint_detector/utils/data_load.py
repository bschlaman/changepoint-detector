import pandas as pd

def load_from_file_dataframe(path: str, **kwargs) -> pd.DataFrame:
    """Load data from a csv file into memory as a pandas DataFrame

    :param path_in_datalake: file location within the datalake dir
    :param index_col: column name to use as the index to the dataframe
    """
    # detect common problems with CSVs sourced from the internet
    # this is hacky, since I load the file into memory.
    with open(path, "rb") as f:
        content = f.read()
        if b"\r\n" in content:
            raise Exception("DOS file format detected!")
        if b",\n" in content:
            raise Exception("Trailing commas detected!")
    return pd.read_csv(path, **kwargs)
