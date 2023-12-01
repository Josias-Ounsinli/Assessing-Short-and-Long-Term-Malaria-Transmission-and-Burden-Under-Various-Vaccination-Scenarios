""" Util functions to create new features """

import pandas as pd
import numpy as np

from utils_cleaner import DataFrameCleaner


class FeatureEnginnering(DataFrameCleaner):
    """Feature Engineering Class"""

    def __init__(self, data: pd.DataFrame) -> None:
        """Init function"""

        self.data = data

    def _neg_exp(self, time, b_0, b_1=0.5):
        return b_0 * np.exp(-b_1 * time)

    def _creating_appropiate_susceptible_and_vaccinated_for_future(
        self, data, coverage_0, coverage_2, initial_efficacy, efficacy_booster
    ):
        """Util function thta create new features"""

        pop_columns = [
            "Total Aged 0 (thousand)",
            "Total Aged 1 (thousand)",
            "Total Aged 2 (thousand)",
            "Total Aged 3 (thousand)",
            "Total Aged 4 (thousand)",
            "Total Aged 5 (thousand)",
        ]
        vac_columns = [
            "Vaccinated Aged 0",
            "Vaccinated Aged 1",
            "Vaccinated Aged 2",
            "Vaccinated Aged 3",
            "Vaccinated Aged 4",
            "Vaccinated Aged 5",
        ]
        vac_boost_columns = [
            "Vaccinated Booster y0 (Aged 2)",
            "Vaccinated Booster y1 (Aged 3)",
            "Vaccinated Booster y2 (Aged 4)",
            "Vaccinated Booster y3 (Aged 5)",
        ]

        # new_features = [
        #     "Vaccinated Aged 0",
        #     "Susceptibles, not vaccinated (0-5)",
        #     "Effectively_protected (0-5)",
        #     "Vaccinated_still_susceptibles (0-5)",
        # ]

        # feat1: Vaccinated Aged 0
        data["Vaccinated Aged 0"] = data["Total Aged 0 (thousand)"] * coverage_0

        # feat2: Susceptibles, not vaccinated (0-5)
        for i in range(1, 6):
            data[f"Vaccinated Aged {i}"] = (
                data[f"Vaccinated Aged {i-1}"].shift(1)
                * data[f"Probablity of surviving at age {i-1}"]
            )

        data[vac_columns] = data[vac_columns].fillna(0)
        data["Susceptibles, not vaccinated (0-5)"] = (
            np.array(data[pop_columns]) - np.array(data[vac_columns])
        ).sum(axis=1)

        # feat3: Effectively_protected (0-5)
        data["Vaccinated Booster y0 (Aged 2)"] = data["Vaccinated Aged 2"] * coverage_2
        for i in range(3, 6):
            data[f"Vaccinated Booster y{i-2} (Aged {i})"] = (
                data[f"Vaccinated Booster y{i-2-1} (Aged {i-1})"].shift(1)
                * data[f"Probablity of surviving at age {i-1}"]
            )
        data[vac_boost_columns] = data[vac_boost_columns].fillna(0)

        for i in range(2, 6):
            data[f"Vaccinated Aged {i}"] = (
                data[f"Vaccinated Aged {i}"]
                - data[f"Vaccinated Booster y{i-2} (Aged {i})"]
            )

        primary_efficacy_coefs = []
        boost_efficacy_coefs = []

        for time in range(6):
            primary_efficacy_coefs.append(self._neg_exp(time, initial_efficacy))

        for time in range(4):
            boost_efficacy_coefs.append(self._neg_exp(time, efficacy_booster))

        data["Effectively_protected (0-5)"] = (
            np.array(primary_efficacy_coefs) * np.array(data[vac_columns])
        ).sum(axis=1) + (
            np.array(boost_efficacy_coefs) * np.array(data[vac_boost_columns])
        ).sum(
            axis=1
        )

        # feat4: Vaccinated_still_susceptibles (0-5)
        data["Vaccinated_still_susceptibles (0-5)"] = (
            (1 - np.array(primary_efficacy_coefs)) * np.array(data[vac_columns])
        ).sum(axis=1) + (
            1 - (np.array(boost_efficacy_coefs)) * np.array(data[vac_boost_columns])
        ).sum(
            axis=1
        )

        return data

    def create_new_features_for_future(
        self, column: str, coverage_0, coverage_2, initial_efficacy, efficacy_booster
    ):
        """Creating new_features"""

        subsets = self.split_in_subframes(column)

        added_subframes = [
            self._creating_appropiate_susceptible_and_vaccinated_for_future(
                frame, coverage_0, coverage_2, initial_efficacy, efficacy_booster
            )
            for frame in subsets
        ]

        updated_data = pd.concat(added_subframes)

        return updated_data

    def _creating_appropiate_susceptible_and_vaccinated_for_present(
        self, data, initial_efficacy
    ):
        """Util function thta create new features"""

        pop_columns = [
            "Total Aged 0 (thousand)",
            "Total Aged 1 (thousand)",
            "Total Aged 2 (thousand)",
            "Total Aged 3 (thousand)",
            "Total Aged 4 (thousand)",
            "Total Aged 5 (thousand)",
        ]
        vac_columns = [
            "Vaccinated Aged 0",
            "Vaccinated Aged 1",
            "Vaccinated Aged 2",
            "Vaccinated Aged 3",
            "Vaccinated Aged 4",
            "Vaccinated Aged 5",
        ]

        # feat2: Susceptibles, not vaccinated (0-5)
        for i in range(1, 6):
            data[f"Vaccinated Aged {i}"] = (
                data[f"Vaccinated Aged {i-1}"].shift(1)
                * data[f"Probablity of surviving at age {i-1}"]
            )

        data[vac_columns] = data[vac_columns].fillna(0)
        data["Susceptibles, not vaccinated (0-5)"] = (
            np.array(data[pop_columns]) - np.array(data[vac_columns])
        ).sum(axis=1)

        # feat3: Effectively_protected (0-5)
        primary_efficacy_coefs = []

        for time in range(6):
            primary_efficacy_coefs.append(self._neg_exp(time, initial_efficacy))

        data["Effectively_protected (0-5)"] = (
            np.array(primary_efficacy_coefs) * np.array(data[vac_columns])
        ).sum(axis=1)

        # feat4: Vaccinated_still_susceptibles (0-5)
        data["Vaccinated_still_susceptibles (0-5)"] = (
            (1 - np.array(primary_efficacy_coefs)) * np.array(data[vac_columns])
        ).sum(axis=1)

        return data

    def create_new_features_for_present(
        self, column: str, initial_efficacy
    ):
        """Creating new_features"""

        subsets = self.split_in_subframes(column)

        added_subframes = [
            self._creating_appropiate_susceptible_and_vaccinated_for_present(
                frame, initial_efficacy
            )
            for frame in subsets
        ]

        updated_data = pd.concat(added_subframes)

        return updated_data
