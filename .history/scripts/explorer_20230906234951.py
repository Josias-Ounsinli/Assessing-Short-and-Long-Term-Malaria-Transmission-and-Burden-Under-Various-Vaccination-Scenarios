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

    def convert_to_dateformat(self, data, country_column):
        """convert Year column to date"""
        # Convert year to datetime format
        cleaned_df1.loc[:, "Year"] = pd.to_datetime(cleaned_df1["Year"], format="%Y")

        # Change the date to 31/12
        cleaned_df1.loc[:, "Year"] = cleaned_df1["Year"] - pd.DateOffset(days=1) + pd.DateOffset(years=1)

    def extract_series(self):
        """extract series from a dataset"""
