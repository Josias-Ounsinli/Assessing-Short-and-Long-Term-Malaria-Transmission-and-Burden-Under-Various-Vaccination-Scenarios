""" Data Transformer Class """

import dvc.api
import pandas as pd


class DataTransformer:
    """Transform dataset into a more suitable dataset"""

    COUNTRIES = ["Nigeria", "Burkina Faso", "Ghana", "Kenya", "Malawi"]

    def __init__(self):
        """Init method"""

    def load_data(self, ext: str, filepath, repo="./", **kwargs):
        """Load a dataset"""

        exts = ["xlsx", "csv"]

        with dvc.api.open(filepath, repo=repo, mode="rb") as file:
            if ext not in exts:
                raise ValueError(f"Invalid ext type. Expected one of: {exts}")

            if ext == "xlsx":
                data = pd.read_excel(file, nrows=kwargs["nrows"])
            else:
                data = pd.read_csv(file, nrows=kwargs["header"])

        return data

    def subset_study_countries(self):
        """Subset the five study countries"""

    def convert_to_dateformat(self):
        """convert Year column to date"""

    def extract_series(self):
        """extract series from a dataset"""
