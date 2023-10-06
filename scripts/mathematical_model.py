""" SVIRD Model Class """

import pickle

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize

from keras.models import load_model
from modeling import build_sequences


class SIRVD:
    """SIRVD Model Class"""

    def __init__(self, params: dict):
        """Init function"""

        self.params = params

    def _sirvd_model(self, y, t, params):
        """Model definition"""
        S, I, R, V, D = y
        biglambda, lamda, theta, omega_v, omega_r, mu, tau, gamma, delta = params
        N = S + I + R + V

        dSdt = biglambda * N + omega_v * V + omega_r * R - (mu + theta + lamda) * S
        dIdt = lamda * S + lamda * (1 - tau) * V - (mu + gamma + delta) * I
        dRdt = gamma * I - (mu + omega_r) * R
        dVdt = theta * S - (mu + omega_v + lamda * (1 - tau)) * V
        dDdt = delta * I

        return [dSdt, dIdt, dRdt, dVdt, dDdt]

    def _cost_function(self, params, y0, t, observed):
        """cost function"""

        solution = odeint(self._sirvd_model, y0, t, args=(params,))
        predicted = solution[:, [1, 4]]
        squared_errors = (predicted - observed) ** 2
        mse = np.mean(squared_errors)
        return mse

    def get_plausible_observations(self, observed_rates_dataframe: pd.DataFrame):
        """Get plausible observed data from observed rates"""
        S_data = [1000000]
        I_data = []
        D_data = []

        for rows, columns in observed_rates_dataframe.iterrows():
            I_data.append(int((columns["Target_1"] / 1000) * S_data[-1]))
            D_data.append(int((columns["Target_2"] / 1000) * I_data[-1]))
            S_data.append(S_data[-1] - I_data[-1])

        plausible_I_D = np.transpose(np.array([I_data, D_data]))

        return plausible_I_D

    def optimize_params(self, y0, t, plausible_I_D):
        """Optimize parameters by minimizing the cost function"""

        initial_guess = self.params
        result = minimize(
            self._cost_function, initial_guess, args=(y0, t, plausible_I_D)
        )

        return result.x

    def update_params(self, params):
        """Update parameters"""
        self.params = params

    def estimate_classes_dynamics(self, y0, t_pred):
        """Estimate classes dynamics

        Fit model and generate data
        """
        params_for_gen = self.params
        solution = odeint(self._sirvd_model, y0, t_pred, args=(params_for_gen,))

        return solution.T

    def get_corresponding_targets_values(self, solutions, weights, target_columns):
        """Get corresponding targets values"""
        S, I, R, V, D = solutions

        generated_table = pd.DataFrame({"S": S, "I": I, "R": R, "V": V, "D": D})
        generated_table["Target_1"] = generated_table["I"] * 1000 / generated_table["S"]
        generated_table["D_true"] = generated_table["D"] - generated_table["D"].shift(1)
        generated_table["Target_2"] = (
            generated_table["D_true"] * 1000 / generated_table["I"]
        )
        generated_table["Target_3"] = weights[0] * generated_table["Target_2"] * 5
        generated_table["Target_4"] = weights[1] * generated_table["Target_2"] * 5
        generated_table["Target_5"] = weights[2] * generated_table["Target_2"] * 5
        generated_table["Target_6"] = weights[3] * generated_table["Target_2"] * 5
        generated_table["Target_7"] = weights[4] * generated_table["Target_2"] * 5

        pred_data = generated_table[target_columns]
        pred_data = pred_data.loc[1:,].reset_index(drop=True)

        return pred_data


class PredictionFromSIVRDGeneration:
    """Prediction using generated data"""

    def __init__(
        self,
        model=load_model(
            "../models/LSTM_MBIOST/Time_lag_2/20230921_150200/20230921_150200/models/lstm_20230921_150200.h5"
        ),
        scores_file="../models/LSTM_MBIOST/Time_lag_2/20230921_150200/lstm_scores_20230921_150200.pkl",
    ):
        """Init function"""
        self.model = model
        self.scores_file = scores_file

    def _get_model_scalers(self):
        """Get model scalars"""

        with open(self.scores_file, "rb") as f:
            actefacts = pickle.load(f)

        scaler_inputs, scaler_targets = actefacts["scalers"]

        return scaler_inputs, scaler_targets

    def get_vaccine_features(self, feature_32, feature_33, feature_34):
        """Get vaccine features"""

        vaccine_features = {
            "Feature_32": feature_32,
            "Feature_33": feature_33,
            "Feature_34": feature_34,
        }
        return vaccine_features

    def get_model_inputs_data(self, data, feature_columns, pred_data, **kwargs):
        """Get input data for model"""

        feature_32, feature_33, feature_34 = (
            kwargs["feature_32"],
            kwargs["feature_33"],
            kwargs["feature_34"],
        )

        future_years = 50
        extended_feature_data = []

        vaccine_features = self.get_vaccine_features(feature_32, feature_33, feature_34)

        for year in range(2023, 2023 + future_years):
            features_data = data.copy()
            next_row_values = (
                data[feature_columns].tail(5).mean().to_frame().to_dict()[0]
            )

            next_row_values.update(vaccine_features)

            next_date = pd.to_datetime(f"{year}-12-31")
            next_row = {
                "Date": next_date,
                "ISO3": data.ISO3.unique()[0],
            }
            next_row.update(next_row_values)
            extended_feature_data.append(next_row)

            new_row = pd.DataFrame(next_row_values, index=[data.index[-1] + 1])

            features_data = pd.concat([features_data, new_row])

        # Extend the dataset with the predictions
        extended_feature_data = pd.DataFrame(extended_feature_data)

        input_pred_data = pd.concat([extended_feature_data, pred_data], axis=1)

        return input_pred_data

    def predict_scenario(self, data, nga_data, **kwargs):
        """Make prediction for a given scenario using the model"""
        seq_length = 2

        scaler_inputs, scaler_targets = self._get_model_scalers()

        scaled_pred_data = scaler_inputs.transform(data[kwargs["input_columns"]])

        scaled_pred_data = pd.DataFrame(
            scaled_pred_data, columns=kwargs["input_columns"]
        )
        scaled_pred_data[kwargs["index_cols"]] = data[kwargs["index_cols"]]

        scaled_pred_data_for_seq = scaled_pred_data.set_index(kwargs["index_cols"])

        X, X_ = build_sequences(
            scaled_pred_data_for_seq,
            scaled_pred_data_for_seq,
            seq_length=seq_length,
            index_cols=kwargs["index_cols"],
        )

        predicted_data = self.model.predict(X)

        predicted_data = scaler_targets.inverse_transform(predicted_data)

        predicted_data = pd.DataFrame(
            data=predicted_data,
            columns=kwargs["target_columns"],
            index=list(scaled_pred_data["Date"][: len(predicted_data)]),
        )
        predicted_data["Date"] = predicted_data.index
        predicted_data["Predicted"] = 1

        original_data = nga_data[kwargs["target_columns"] + ["Date"]]
        original_data.index = original_data["Date"]
        original_data["Predicted"] = 0

        total_predicted_data = pd.concat([original_data, predicted_data])

        return total_predicted_data
