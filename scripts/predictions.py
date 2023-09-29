""" Uitls predictions functions """

from datetime import datetime
from IPython.display import display

import numpy as np
import pandas as pd


def predict_next_year(model, end_year, data, group, **kwargs):
    """Predict next year"""
    seq_length = kwargs["seq_length"]
    scaler_inputs = kwargs["scaler_inputs"]
    scaler_targets = kwargs["scaler_targets"]
    input_columns = kwargs["input_columns"]
    target_columns = kwargs["target_columns"]

    pred_2023 = {}

    for country in data[group].unique():
        country_data = data[data[group] == country]
        last_sequence = country_data[input_columns].tail(seq_length)
        last_sequence = scaler_inputs.transform(last_sequence)
        last_sequence = np.expand_dims(last_sequence, axis=0)
        predictions = model.predict(last_sequence)

        # Inverse transform predictions using scalers
        inverted_predictions = scaler_targets.inverse_transform(predictions)
        inverted_predictions = pd.DataFrame(
            data=inverted_predictions,
            columns=target_columns,
            index=[datetime(end_year + 1, 12, 31)],
        )

        print(f"Predictions for {country} in {end_year + 1}:")
        display(inverted_predictions)

        pred_2023[country] = inverted_predictions

    return pred_2023


def predict_to_2070(model, data, vaccine_features=None, **kwargs):
    """Predict to 2070"""
    # Extend the dataset and make predictions for 50 years into the future
    future_years = 48
    data = data.copy()
    data["Predicted"] = 0

    extended_data = []

    if not vaccine_features:
        vaccine_features = (
            data[["Feature_32", "Feature_33", "Feature_34"]]
            .tail(1)
            .to_dict("records")[0]
        )

    seq_length = kwargs["seq_length"]
    scaler_inputs = kwargs["scaler_inputs"]
    scaler_targets = kwargs["scaler_targets"]
    input_columns = kwargs["input_columns"]
    target_columns = kwargs["target_columns"]
    feature_columns = kwargs["feature_columns"]

    for year in range(2023, 2023 + future_years):
        last_sequence = data[input_columns].tail(seq_length)
        last_sequence = scaler_inputs.transform(last_sequence)
        # Predict for the next year
        predicted_values = model.predict(np.expand_dims(last_sequence, axis=0))

        # Inverse transform the predictions
        inverted_predictions = scaler_targets.inverse_transform(predicted_values)
        inverted_predictions = pd.DataFrame(
            data=inverted_predictions,
            columns=target_columns,
            index=[datetime(year, 12, 31)],
        )

        next_row_pred = inverted_predictions.to_dict("records")[0]
        next_row_values = data[feature_columns].tail(5).mean().to_frame().to_dict()[0]
        next_row_values.update(vaccine_features)

        next_row_values.update(next_row_pred)

        # Append the predictions to the dataset
        next_date = pd.to_datetime(f"{year}-12-31")
        next_row = {
            "Date": next_date,
            "ISO3": data.ISO3.unique()[0],
        }
        next_row.update(next_row_values)
        extended_data.append(next_row)

        # Update the last_sequence for the next prediction
        new_row = pd.DataFrame(next_row_values, index=[data.index[-1] + 1])
        data = pd.concat([data, new_row])

    # Extend the dataset with the predictions
    extended_data = pd.DataFrame(extended_data)
    extended_data["Predicted"] = 1

    updated_data = pd.concat([data, extended_data], ignore_index=True)

    return updated_data
