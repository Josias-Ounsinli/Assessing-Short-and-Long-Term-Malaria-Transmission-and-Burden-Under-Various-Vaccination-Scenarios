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

            if kwargs:
                if ext == "xlsx":
                    data = pd.read_excel(file, nrows=kwargs["nrows"])
                else:
                    data = pd.read_csv(file, header=kwargs["header"])
            else:
                if ext == "xlsx":
                    data = pd.read_excel(file)
                else:
                    data = pd.read_csv(file)

        return data

    def subset_study_countries(self, data, country_column):
        """Subset the five study countries"""

        # Subset data
        cleaned_data = data[data[country_column].isin(self.COUNTRIES)]
        # Reset data
        cleaned_data.reset_index(drop=True, inplace=True)

        return cleaned_data

    def convert_to_dateformat(self, data, year_column):
        """convert Year column to date"""

        # Convert year_column to datetime format
        data.loc[:, year_column] = pd.to_datetime(data[year_column], format="%Y")

        # Change the date to 31/12
        data.loc[:, year_column] = (
            data[year_column] - pd.DateOffset(days=1) + pd.DateOffset(years=1)
        )

        return data

    def extract_series(self):
        """extract series from a dataset"""
