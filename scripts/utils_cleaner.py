""" Get information on a dataframe """

from IPython.display import display
import pandas as pd
import numpy as np

from dataframe_info_extractor import DataFrameInfo


class DataFrameCleaner:
    """DataFrame Cleaner"""

    def __init__(self, data: pd.DataFrame) -> None:
        """Init function"""
        self.data = data

    def drop_columns(self, columns: list):
        """Drop columns from dataframe"""
        self.data.drop(columns=columns, inplace=True)

    def convert_to_date(self, column: str):
        """Convert to datetime"""
        self.data[column] = pd.to_datetime(self.data[column])

    def replace_in_string_to_int(self, column: str, to_replace: str, replace_by: str):
        """Replace in string column and convert to int"""
        self.data[column] = self.data[column].str.replace(to_replace, replace_by)
        self.data[column] = self.data[column].fillna(-1)
        self.data[column] = self.data[column].astype(int)
        self.data[column] = self.data[column].replace(-1, np.nan)

    def split_in_subframes(self, column: str, external_data=False, data=None):
        """Split data in sub dataframes"""
        if external_data:
            keys = list(data[column].unique())

            subsets = []

            for key in keys:
                subset = data.loc[data[column] == key]
                subsets.append(subset)
        else:
            keys = list(self.data[column].unique())

            subsets = []

            for key in keys:
                subset = self.data.loc[self.data[column] == key]
                subsets.append(subset)

        return subsets

    def remove_more_than_percent_missing_values(self, column: str, level):
        """Remove columns with missing value above 20% in country dataset"""

        subsets = self.split_in_subframes(column)

        datainfo = [DataFrameInfo(frame) for frame in subsets]

        drop_columns = [
            set(frameinfo.missing_values_table(missing_level=level).index)
            for frameinfo in datainfo
        ]

        drop_columns = set.union(*drop_columns)

        self.drop_columns(list(drop_columns))

    def _fillna_numbers(self, data: pd.DataFrame):
        """Fill missing values in a DataFrame using average growth rate"""
        data = data.copy()
        data = data.reset_index(drop=True)
        for column in data.select_dtypes(include="number").columns:
            growth_rate = (data[column].pct_change()).mean()
            for i in list(data[data[column].isna()].index):
                index = data.loc[:i, column].last_valid_index()
                if index is None:
                    index = data.loc[i:, column].first_valid_index()
                value = data[column].loc[index]
                if index < i:
                    data.loc[i, column] = value * ((1 + growth_rate) ** (i - index))
                else:
                    data.loc[i, column] = value / ((1 + growth_rate) ** (index - i))
        return data

    def fill_missing(self, column: str):
        """Fill missing values using ffill and bfill method"""

        subsets = self.split_in_subframes(column)

        frames_filled = [self._fillna_numbers(frame) for frame in subsets]

        self.data = pd.concat(frames_filled)

    def _get_threshold(self, column: str):
        """Get lower and upper to identify outlier"""
        quartile_1 = self.data[column].quantile(0.05)
        quartile_3 = self.data[column].quantile(0.95)

        # Defining inter quartile range
        inter_quartile_range = quartile_3 - quartile_1

        # Get lower and upper bond
        lower = quartile_1 - 1.5 * inter_quartile_range
        upper = quartile_3 + 1.5 * inter_quartile_range

        return lower, upper

    def manage_outlier(
        self, columns: list, drop=False, cat_columns=False, cat_values=False
    ):
        """Find and manage (drop or not) outlier"""

        isoutlier = []

        for column in columns:
            lower, upper = self._get_threshold(column)
            isoutlier.append((self.data[column] < lower) | (self.data[column] > upper))

        outlier_table = (
            pd.DataFrame(
                np.array(isoutlier).transpose(),
                columns=[f"{column}" for column in columns],
            )
            .sum()
            .reset_index()
            .rename(columns={"index": "Columns", 0: "nb_outliers"})
        )

        outlier_columns = outlier_table[outlier_table["nb_outliers"] > 0][
            "Columns"
        ].to_list()

        if cat_columns:
            return outlier_columns

        if cat_values:
            for column in outlier_columns:
                lower, upper = self._get_threshold(column)
                mask_outlier = (self.data[column] < lower) | (self.data[column] > upper)

                display(f"Outliers in {column}")
                display(self.data[mask_outlier])

            return outlier_columns

        if drop:
            for column in outlier_columns:
                lower, upper = self._get_threshold(column)
                mask_outlier = (self.data[column] > lower) | (self.data[column] < upper)
                self.data = self.data[mask_outlier]

            return self.data

        return outlier_table
