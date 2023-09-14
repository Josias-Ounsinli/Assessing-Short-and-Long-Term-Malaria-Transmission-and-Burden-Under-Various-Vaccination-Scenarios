""" Data Transformer Class """

import dvc.api
import pandas as pd


class DataTransformer:
    """Transform dataset into a more suitable dataset"""

    COUNTRIES = ["Nigeria", "Burkina Faso", "Ghana", "Kenya", "Malawi"]

    def __init__(self):
        """Init method"""

    def load_data(self, ext: str, filepath, repo="./", rev: str = None, **kwargs):
        """Load a dataset

        Parameters
        ----------
        ext: str :

        filepath :

        repo :
             (Default value = "./")
        **kwargs :


        Returns
        -------

        """
        exts = ["xlsx", "csv"]

        with dvc.api.open(filepath, repo=repo, mode="rb", rev=rev) as file:
            if ext not in exts:
                raise ValueError(f"Invalid ext type. Expected one of: {exts}")

            if kwargs:
                if ext == "xlsx":
                    data = pd.read_excel(file, nrows=kwargs["nrows"])
                else:
                    data = pd.read_csv(
                        file, header=kwargs["header"], parse_dates=kwargs["parse_dates"]
                    )
            else:
                if ext == "xlsx":
                    data = pd.read_excel(file)
                else:
                    data = pd.read_csv(file)

        return data

    def subset_study_countries(self, data, country_column):
        """Subset the five study countries

        Parameters
        ----------
        data : pd.DataFrame

        country_column : str

        Returns
        -------

        """
        # Subset data
        cleaned_data = data[data[country_column].isin(self.COUNTRIES)]
        # Reset data
        cleaned_data.reset_index(drop=True, inplace=True)

        return cleaned_data

    def convert_to_dateformat(self, data, year_column):
        """convert Year column to date

        Parameters
        ----------
        data : pd.DataFrame

        year_column : str

        Returns
        -------

        """
        # Convert year_column to datetime format
        data.loc[:, year_column] = pd.to_datetime(data[year_column], format="%Y")

        # Change the date to 31/12
        data.loc[:, year_column] = (
            data[year_column] - pd.DateOffset(days=1) + pd.DateOffset(years=1)
        )

        return data

    def extract_unique_serie(self, data, country_column, serie_name):
        """Extract a unique serie without too much constraint

        Parameters
        ----------
        data : pd.DataFrame

        country_column : str

        serie_name : str


        Returns
        -------

        """
        data = (
            data.set_index(country_column)
            .stack(dropna=False)
            .reset_index()
            .rename(columns={"level_1": "Year", 0: serie_name})
        )

        return data

    def extract_series(
        self,
        series_metadata: list,
        source_data: pd.DataFrame,
        immutable_columns: list,
        multiple_index=False,
    ) -> pd.DataFrame:
        """Extract a series from source_data

        Parameters
        ----------
        series_metadata: list :

        source_data: pd.DataFrame :

        immutable_columns: list :

        multiple_index :
             (Default value = False)

        Returns
        -------

        """
        series_columns = series_metadata[0]
        series_name = series_metadata[1]
        series_keyword = series_metadata[2]

        # Subset concerned columns and stack
        data = (
            source_data[immutable_columns + series_columns]
            .set_index(immutable_columns)
            .stack()
            .rename(series_name)
            .reset_index()
            .rename(
                columns={
                    "level_2": "Year",
                }
            )
        )

        # Clean Year column
        data["Year"] = data["Year"].str.replace(series_keyword, "").astype(int)

        # Convert year to datetime format
        data = self.convert_to_dateformat(data=data, year_column="Year")

        if multiple_index:
            data.set_index(["Year"] + immutable_columns, inplace=True)
            data = data[series_name].unstack().reset_index()

        return data

    def get_country_climate_data(
        self, source_df: pd.DataFrame, variable_name: str, start: int
    ):
        """Subset country climate data

        Parameters
        ----------
        source_df: pd.DataFrame :

        variable_name: str :

        start: int :


        Returns
        -------

        """
        # subset dataset
        data = source_df[source_df.columns[:2]]

        # Create Country columns
        data = data.assign(Country=source_df.columns[1])

        # Rename columns
        new_names = ["Year", variable_name]
        data.rename(
            columns=dict(zip(source_df.columns[:2], new_names)),
            inplace=True,
        )

        # Subset Year from 2000
        data = data[data["Year"] >= start]

        # Reset index
        data.reset_index(drop=True, inplace=True)

        # Convert year to datetime format
        data = self.convert_to_dateformat(data=data, year_column="Year")

        return data
