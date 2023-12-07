""" Uitls predictions functions """

from datetime import datetime
from IPython.display import display

import numpy as np
import pandas as pd

from modeling import build_train_test_datasets
from modeling import build_sequences


def format_data_column_names(data):
    """Rename data columns"""
    ### Renaming the columns
    potential_targets = ["Malaria_Incidence", "Malaria_Deaths_U5", "Malaria_Deaths"]

    potential_features = [
        "ITN_Access",
        "PopDensity",
        "MedianAgePop",
        "PopGrowthRate",
        "TFR",
        "IMR",
        "Q5",
        "CNMR",
        "Population ages 0-14 (% of total population)",
        "Population ages 15-64 (% of total population)",
        "Domestic general government health expenditure (% of general government expenditure)",
        "External health expenditure (% of current health expenditure)",
        "People using at least basic sanitation services, rural (% of rural population)",
        "People using safely managed sanitation services, rural (% of rural population)",
        "Population living in slums (% of urban population)",
        "Foreign direct investment, net inflows (% of GDP)",
        "Mortality rate, under-5 (per 1,000 live births)",
        "Population growth (annual %)",
        "Population in urban agglomerations of more than 1 million (% of total population)",
        "Urban population (% of total population)",
        "Urban population growth (annual %)",
        "Rural population",
        "Precipitation",
        "Average Mean Surface Air Temperature",
        "Average Minimum Surface Air Temperature",
        "Leveraged RTS Vaccine",
        "Leveraged R21 Vaccine",
        "Vaccinated Aged 0",
        "Effectively_protected (0-5)",
        "Vaccinated_still_susceptibles (0-5)",
    ]

    new_column_names = {
        i: f"Target_{potential_targets.index(i)+1}" for i in potential_targets
    }
    features_names = {
        i: f"Feature_{potential_features.index(i)+1}" for i in potential_features
    }

    new_column_names.update(features_names)

    index_cols = ["Country", "ISO3", "Date"]
    # Renaming the columns
    data = data.rename(columns=new_column_names).set_index(index_cols)

    return data


def predict_malaria_dynamics_without_target_lag(
    data, model, index_cols, input_columns, **kwargs
):
    """predict"""

    d_train_test, scaler_inputs, scaler_inputs = build_train_test_datasets(
        data,
        index_cols=index_cols,
        input_cols=input_columns,
        target_cols=kwargs["target_columns"],
        train_limit="2070",
        scaler_str="standard",
        target=False,
        split=False,
        test=False,
    )

    X = d_train_test["Scaled"]["train"]["Inputs"]

    X = build_sequences(
        X, X, seq_length=kwargs["seq_length"], index_cols=index_cols, target=False
    )

    predictions = model.predict(X)

    inverted_predictions = kwargs["scaler_targets"].inverse_transform(predictions)

    inverted_predictions = pd.DataFrame(
        data=inverted_predictions,
        columns=kwargs["target_columns"],
        # index=futur_data.set_index(index_cols).index,
    )

    return inverted_predictions


# def predict_next_year(model, end_year, data, group, **kwargs):
#     """Predict next year"""
#     seq_length = kwargs["seq_length"]
#     scaler_inputs = kwargs["scaler_inputs"]
#     scaler_targets = kwargs["scaler_targets"]
#     input_columns = kwargs["input_columns"]
#     target_columns = kwargs["target_columns"]

#     pred_2023 = {}

#     for country in data[group].unique():
#         country_data = data[data[group] == country]
#         last_sequence = country_data[input_columns].tail(seq_length)
#         last_sequence = scaler_inputs.transform(last_sequence)
#         last_sequence = np.expand_dims(last_sequence, axis=0)
#         predictions = model.predict(last_sequence)

#         # Inverse transform predictions using scalers
#         inverted_predictions = scaler_targets.inverse_transform(predictions)
#         inverted_predictions = pd.DataFrame(
#             data=inverted_predictions,
#             columns=target_columns,
#             index=[datetime(end_year + 1, 12, 31)],
#         )

#         print(f"Predictions for {country} in {end_year + 1}:")
#         display(inverted_predictions)

#         pred_2023[country] = inverted_predictions

#     return pred_2023


def predict_malaria_dynamics_with_target_lag(data, model, **kwargs):
    """Predict to 2070"""

    seq_length = kwargs["seq_length"]
    scaler_inputs = kwargs["scaler_inputs"]
    scaler_targets = kwargs["scaler_targets"]
    input_columns = kwargs["input_columns"]
    target_columns = kwargs["target_columns"]
    # feature_columns = kwargs["feature_columns"]
    # Extend the dataset and make predictions for 50 years into the future
    future_years = 48
    data = data.copy()

    for year in range(2023, 2023 + future_years):
        data_year = data[data["Date"] <= f"{year-1}-12-31"]
        for country in data_year["ISO3"].unique():
            country_data = data_year[data_year["ISO3"] == country]
            last_sequence = country_data[input_columns].tail(seq_length)
            last_sequence = scaler_inputs.transform(last_sequence)
            last_sequence = np.expand_dims(last_sequence, axis=0)
            prediction_next_year = model.predict(last_sequence)
            # Inverse transform predictions using scalers
            inverted_prediction_next_year = scaler_targets.inverse_transform(
                prediction_next_year
            )
            data.loc[
                (data["Date"] == f"{year}-12-31") & (data["ISO3"] == country),
                target_columns,
            ] = inverted_prediction_next_year

    return data


# def predict_to_2070(model, data, vaccine_features=None, **kwargs):
#     """Predict to 2070"""
#     # Extend the dataset and make predictions for 50 years into the future
#     future_years = 48
#     data = data.copy()
#     data["Predicted"] = 0

#     extended_data = []

#     if not vaccine_features:
#         vaccine_features = (
#             data[["Feature_32", "Feature_33", "Feature_34"]]
#             .tail(1)
#             .to_dict("records")[0]
#         )

#     seq_length = kwargs["seq_length"]
#     scaler_inputs = kwargs["scaler_inputs"]
#     scaler_targets = kwargs["scaler_targets"]
#     input_columns = kwargs["input_columns"]
#     target_columns = kwargs["target_columns"]
#     feature_columns = kwargs["feature_columns"]

#     for year in range(2023, 2023 + future_years):
#         last_sequence = data[input_columns].tail(seq_length)
#         last_sequence = scaler_inputs.transform(last_sequence)
#         # Predict for the next year
#         predicted_values = model.predict(np.expand_dims(last_sequence, axis=0))

#         # Inverse transform the predictions
#         inverted_predictions = scaler_targets.inverse_transform(predicted_values)
#         inverted_predictions = pd.DataFrame(
#             data=inverted_predictions,
#             columns=target_columns,
#             index=[datetime(year, 12, 31)],
#         )

#         next_row_pred = inverted_predictions.to_dict("records")[0]
#         next_row_values = data[feature_columns].tail(5).mean().to_frame().to_dict()[0]
#         next_row_values.update(vaccine_features)

#         next_row_values.update(next_row_pred)

#         # Append the predictions to the dataset
#         next_date = pd.to_datetime(f"{year}-12-31")
#         next_row = {
#             "Date": next_date,
#             "ISO3": data.ISO3.unique()[0],
#         }
#         next_row.update(next_row_values)
#         extended_data.append(next_row)

#         # Update the last_sequence for the next prediction
#         new_row = pd.DataFrame(next_row_values, index=[data.index[-1] + 1])
#         data = pd.concat([data, new_row])

#     # Extend the dataset with the predictions
#     extended_data = pd.DataFrame(extended_data)
#     extended_data["Predicted"] = 1

#     updated_data = pd.concat([data, extended_data], ignore_index=True)

#     return updated_data
