""" Data Transformer Class """

import dvc.api
import pandas as pd

class DataTransformer:
    """ Transform dataset into a more suitable dataset """

    COUNTRIES = ["Nigeria", "Burkina Faso", "Ghana", "Kenya", "Malawi"]

    def __init__(self):
        """ Init method """
        pass

    def load_data(self, ext: str, filepath, repo="./", **kwargs):
        """ Load a dataset """

        exts = ["xlsx", "csv"]

        with dvc.api.open(
            filepath,
            repo=repo,
            mode="rb"
        ) as fd:
            if ext not i exts:
                raise ValueError("Invalid ext type. Expected one of: %s" % ext)
            
            if ext="xlsx":
                df = pd.read_excel(fd, nrows=kwargs["nrows"])
            else:
                df = pd.read_csv(fd, nrows=kwargs["header"])


    def subset_study_countries(self):
        """ Subset the five study countries """
        pass


    def convert_to_dateformat(self):
        """ convert Year column to date """
        pass

    def extract_series():
        pass