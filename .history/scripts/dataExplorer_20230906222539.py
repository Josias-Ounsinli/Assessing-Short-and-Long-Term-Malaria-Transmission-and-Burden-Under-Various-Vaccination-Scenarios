""" Data Transformer Class """

class DataTransformer:
    """ Transform dataset into a more suitable dataset """

    COUNTRIES = ["Nigeria", "Burkina Faso", "Ghana", "Kenya", "Malawi"]

    def __init__(self) -> None:
        pass

    def load_data(self):
        """ Load a dataset """
        pass

    def subset_study_countries(self):
        """ Subset the five study countries """
        pass

    def convert_to_dateformat(self):
        """ convert Year column to date """
        pass

    def extract_series(
        self,
        series_columns: list,
        series_name: str,
        series_keyword:str,
        source_data=source_df3,
        immutable_columns=country_columns,
        multiple=False
    )->pd.DataFrame:
        """ Extract a series from source_data """

        # Subset concerned columns and stack
        df = (
            source_data[immutable_columns+series_columns].
            set_index(immutable_columns).
            stack().rename(series_name).
            reset_index().
            rename(
                columns={
                    "level_2": "Year",
                }
            )
        )

        # Clean Year column
        df["Year"] = df["Year"].str.replace(series_keyword,"").astype(int)

        # Convert year to datetime format
        df.loc[:, "Year"] = pd.to_datetime(df["Year"], format="%Y")

        # Change the date to 31/12
        df.loc[:, "Year"] = df["Year"] - pd.DateOffset(days=1) + pd.DateOffset(years=1)

        if multiple:
            df.set_index(["Year"] + immutable_columns, inplace=True)
            df = (
                df[series_name].
                unstack().
                reset_index()
            )

        return df