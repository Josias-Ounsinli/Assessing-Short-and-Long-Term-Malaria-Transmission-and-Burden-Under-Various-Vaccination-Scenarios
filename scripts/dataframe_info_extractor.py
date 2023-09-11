""" Get information on a dataframe """

import pandas as pd
import numpy as np


class DataFrameInfo:
    """Dataframe Info class"""

    def __init__(self, data: pd.DataFrame):
        """Initialize the DataFrame"""

        self.data = data

    def get_dimension(self):
        """Get dataframe dimensions"""

        print(f" There are {self.data.shape[0]} rows and {self.data.shape[1]} columns")

    def get_data_types(self):
        """Get value counts of data types in the dataframe"""

        dtypes = self.data.dtypes.value_counts()
        return dtypes

    def get_variables_types(self):
        """Get variables types"""
        dtypes = self.data.dtypes
        return dtypes

    def get_percent_missing(self):
        """Get percentage of missing values"""

        # Calculate total number of cells in dataframe
        totalcells = np.prod(self.data.shape)

        # Count number of missing values per column
        missingcount = self.data.isnull().sum()

        # Calculate total number of missing values
        totalmissing = missingcount.sum()

        # Calculate percentage of missing values
        print(
            "The dataset contains",
            round(((totalmissing / totalcells) * 100), 2),
            "%",
            "missing values.",
        )

    def missing_values_table(self, missing_level=0):
        """Get missing values table"""
        # Total missing values
        missing_value = self.data.isnull().sum()

        # Percentage of missing values
        missing_value_percent = 100 * self.data.isnull().sum() / len(self.data)

        # dtype of missing values
        missing_value_dtype = self.data.dtypes

        # Make a table with the results
        missing_value_table = pd.concat(
            [missing_value, missing_value_percent, missing_value_dtype], axis=1
        )

        # Rename the columns
        missing_value_table_ren_columns = missing_value_table.rename(
            columns={0: "Missing Values", 1: "% of Total Values", 2: "Dtype"}
        )

        # Sort the table by percentage of missing descending
        missing_value_table_ren_columns = (
            missing_value_table_ren_columns[
                missing_value_table_ren_columns.iloc[:, 1] > missing_level
            ]
            .sort_values("% of Total Values", ascending=False)
            .round(1)
        )

        # Print some summary information
        print(
            "Your selected dataframe has " + str(self.data.shape[1]) + " columns.\n"
            "There are "
            + str(missing_value_table_ren_columns.shape[0])
            + " columns that have missing values greater than "
            + str(missing_level)
            + "%."
        )

        # Return the dataframe with missing information
        return missing_value_table_ren_columns
