""" Modeling utils"""

import os
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras import callbacks

from tensorflow.keras.utils import plot_model


def build_train_test_datasets(
    data: pd.DataFrame,
    index_cols: list,
    input_cols: list,
    target_cols: list,
    train_limit="2019",
    # val_limit="2021",
    scaler_str="standard",
):
    """
    Creates training and test sets while applying normalization if specified in the parameters.

    Parameters
    ----------
    data : pd.DataFrame
        The input data matrix passed as a parameter.
    index_cols: list, optional
        The list of columns considered as index.
    input_cols : list, optional
        The list of columns considered as model inputs.
    target_cols : list, optional
        The columns considered as the values to predict.
    train_limit : str, optional
        The final date to be used for training.
        The default value is 2020.
    val_limit: str, optional
        The final date to be used for validation
    scaler_str : str, optional
        A string indicating which normalization should be applied to the data.
        If the value is valid, normalization is calculated on the training data
        and applied to the entire dataset.
        The default value is 'standard' to apply mean-centered standardization.

    Returns
    -------
    d_train_test : dict
        A dictionary containing normalized and non-normalized training and test data.
    scaler_inputs : sklearn.preprocessing.MinMaxScaler or sklearn.preprocessing.StandardScaler
        The normalization coefficients for the scaling operation applied to the inputs data.
    scaler_targets : sklearn.preprocessing.MinMaxScaler or sklearn.preprocessing.StandardScaler
        The normalization coefficients for the "Min-Max" operation applied to the targets data.
    """

    # Build train set using data from 2000 to 2018
    train_inputs = (
        data[index_cols + input_cols].loc[data.Date < train_limit].set_index(index_cols)
    )
    train_targets = (
        data[index_cols + target_cols]
        .loc[data.Date < train_limit]
        .set_index(index_cols)
    )

    # Build test set using data from 2020
    # val_inputs = (
    #     data[index_cols + input_cols]
    #     .loc[(data.Date >= train_limit) & (data.Date < val_limit)]
    #     .set_index(index_cols)
    # )
    # val_targets = (
    #     data[index_cols + target_cols]
    #     .loc[(data.Date >= train_limit) & (data.Date < val_limit)]
    #     .set_index(index_cols)
    # )

    # Build test set using data from 2020
    test_inputs = (
        data[index_cols + input_cols]
        .loc[data.Date >= train_limit]
        .set_index(index_cols)
    )
    test_targets = (
        data[index_cols + target_cols]
        .loc[data.Date >= train_limit]
        .set_index(index_cols)
    )

    d_train_test = {}
    d_train_test["NonScaled"] = {
        "train": {"Inputs": train_inputs, "Targets": train_targets},
        "test": {"Inputs": test_inputs, "Targets": test_targets},
        # "val": {"Inputs": val_inputs, "Targets": val_targets},
    }

    if scaler_str is None:
        scaler_inputs, scaler_targets = None, None
        d_train_test["Scaled"] = None

    elif scaler_str.lower() == "minmax":
        inputs_train_index = train_inputs.index
        inputs_test_index = test_inputs.index
        # inputs_val_index = val_inputs.index

        targets_train_index = train_targets.index
        targets_test_index = test_targets.index
        # targets_val_index = val_targets.index

        # Apply rescaling
        scaler_inputs = MinMaxScaler()
        train_inputs = scaler_inputs.fit_transform(train_inputs)
        test_inputs = scaler_inputs.transform(test_inputs)
        # val_inputs = scaler_inputs.transform(val_inputs)

        scaler_targets = MinMaxScaler()
        train_targets = scaler_targets.fit_transform(train_targets)
        test_targets = scaler_targets.transform(test_targets)
        # val_targets = scaler_targets.transform(val_targets)

        # Convert normalized values to dataframe
        train_inputs = pd.DataFrame(
            data=train_inputs, index=inputs_train_index, columns=input_cols
        )
        test_inputs = pd.DataFrame(
            data=test_inputs, index=inputs_test_index, columns=input_cols
        )
        # val_inputs = pd.DataFrame(
        #     data=val_inputs, index=inputs_val_index, columns=input_cols
        # )

        train_targets = pd.DataFrame(
            data=train_targets, index=targets_train_index, columns=target_cols
        )
        test_targets = pd.DataFrame(
            data=test_targets, index=targets_test_index, columns=target_cols
        )

        # val_targets = pd.DataFrame(
        #     data=val_targets, index=targets_val_index, columns=target_cols
        # )

        d_train_test["Scaled"] = {
            "train": {"Inputs": train_inputs, "Targets": train_targets},
            "test": {"Inputs": test_inputs, "Targets": test_targets},
            # "val": {"Inputs": val_inputs, "Targets": val_targets},
        }

    elif scaler_str.lower() == "std" or scaler_str.lower() == "standard":
        inputs_train_index = train_inputs.index
        inputs_test_index = test_inputs.index
        # inputs_val_index = val_inputs.index

        targets_train_index = train_targets.index
        targets_test_index = test_targets.index
        # targets_val_index = val_targets.index

        # Apply rescaling
        scaler_inputs = StandardScaler()
        train_inputs = scaler_inputs.fit_transform(train_inputs)
        test_inputs = scaler_inputs.transform(test_inputs)
        # val_inputs = scaler_inputs.transform(val_inputs)

        scaler_targets = StandardScaler()
        train_targets = scaler_targets.fit_transform(train_targets)
        test_targets = scaler_targets.transform(test_targets)
        # val_targets = scaler_targets.transform(val_targets)

        # Convert normalized values to dataframe
        train_inputs = pd.DataFrame(
            data=train_inputs, index=inputs_train_index, columns=input_cols
        )
        test_inputs = pd.DataFrame(
            data=test_inputs, index=inputs_test_index, columns=input_cols
        )
        # val_inputs = pd.DataFrame(
        #     data=val_inputs, index=inputs_val_index, columns=input_cols
        # )

        train_targets = pd.DataFrame(
            data=train_targets, index=targets_train_index, columns=target_cols
        )
        test_targets = pd.DataFrame(
            data=test_targets, index=targets_test_index, columns=target_cols
        )
        # val_targets = pd.DataFrame(
        #     data=val_targets, index=targets_val_index, columns=target_cols
        # )

        d_train_test["Scaled"] = {
            "train": {"Inputs": train_inputs, "Targets": train_targets},
            "test": {"Inputs": test_inputs, "Targets": test_targets},
            # "val": {"Inputs": val_inputs, "Targets": val_targets},
        }

    else:
        scaler_inputs, scaler_targets = None, None
        d_train_test["Scaled"] = None

    return d_train_test, scaler_inputs, scaler_targets


def build_sequences(df_inputs, df_targets, seq_length, index_cols):
    """Build sequences"""
    # Reset set index
    df_inputs = df_inputs.reset_index()
    df_targets = df_targets.reset_index()

    X, y = [], []

    for country in df_inputs["ISO3"].unique():
        country_data_inputs = df_inputs[df_inputs["ISO3"] == country]
        country_data_targets = df_targets[df_targets["ISO3"] == country]
        country_data_inputs = country_data_inputs.set_index(index_cols)
        country_data_targets = country_data_targets.set_index(index_cols)
        for i in range(len(country_data_inputs) - seq_length):
            X.append(country_data_inputs.iloc[i : i + seq_length].values)
            y.append(country_data_targets.iloc[i + seq_length].values)

    X = np.array(X)
    y = np.array(y)

    return X, y


def define_LSTM_net(
    seq_length, n_features, n_targets, layers: list = [100, 50, 10], fn_act="tanh"
):
    """Create the model"""

    model_lstm = Sequential()
    model_lstm.add(
        LSTM(
            layers[0],
            activation=fn_act,
            input_shape=(seq_length, n_features),
            return_sequences=True,
        )
    )
    model_lstm.add(Dropout(0.2))

    for idx in range(1, len(layers) - 1):
        model_lstm.add(LSTM(layers[idx], activation=fn_act, return_sequences=True))
        model_lstm.add(Dropout(0.2))

    model_lstm.add(LSTM(layers[-1], activation=fn_act))
    model_lstm.add(Dropout(0.2))

    model_lstm.add(Dense(n_targets))

    # Summary of the model
    model_lstm.summary()

    # Plot model architecture
    plot_model(
        model_lstm,
        to_file="../plots/model.png",
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
    )

    return model_lstm


def train_lstm(X_train, y_train, X_test, y_test, d_learning_params, results_dir):
    """Train model"""

    # Get today datetime to create the model file
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_dir, current_datetime)
    params_file = os.path.join(
        results_dir, "lstm_params_{}.pkl".format(str(current_datetime))
    )

    # Create results directory if not exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(params_file, "wb") as fp:
        pickle.dump(d_learning_params, fp)

    model_file = os.path.join(
        results_dir, "models", "lstm_{}.h5".format(current_datetime)
    )
    log_dir = os.path.join(results_dir, "logs")

    # Défine learning settings
    # Using tensorboard to track learning performance
    my_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=model_file,
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
            verbose=0,
            save_weights_only=False,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", mode="auto", patience=15, verbose=0
        ),
        callbacks.TensorBoard(log_dir=log_dir),
    ]

    # Create model network
    model = define_LSTM_net(
        seq_length=X_train.shape[1],
        n_features=X_train.shape[2],
        n_targets=y_train.shape[1],
        layers=d_learning_params["layers"],
        fn_act=d_learning_params["activation_fn"],
    )

    # compile model using optimizer and loss
    model.compile(
        optimizer=d_learning_params["optimizer"],
        loss=d_learning_params["loss_fn"],
        metrics=d_learning_params["metrics"],
    )

    # Training
    history = model.fit(
        X_train,
        y_train,
        batch_size=d_learning_params["batch_size"],
        epochs=d_learning_params["epochs"],
        validation_data=(X_test, y_test),
        # validation_split=d_learning_params["val_ratio"],
        callbacks=my_callbacks,
        verbose=1,
    )

    # Display loss and metrics
    fig_perf, axis_perf = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, constrained_layout=True
    )

    axis_perf[0].plot(history.history["loss"], label="Training loss")
    axis_perf[0].plot(history.history["val_loss"], label="Validation loss")
    axis_perf[0].legend()
    axis_perf[0].set_title("Evolution de la fonction de coût pendant l'apprentissage")

    axis_perf[1].plot(
        history.history[d_learning_params["metrics"][0]],
        label="Training {}".format(d_learning_params["metrics"][0]),
    )
    axis_perf[1].plot(
        history.history["val_{}".format(d_learning_params["metrics"][0])],
        label="Validation {}".format(d_learning_params["metrics"][0]),
    )
    axis_perf[1].legend()
    axis_perf[0].set_title(
        "Evolution de la métrique'{}' pendant l'apprentissage".format(
            d_learning_params["metrics"][0]
        )
    )

    # Display model performances
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    scores = dict()
    scores["R2"] = {
        "train": np.round(r2_score(y_train, y_train_pred), 3),
        "test": np.round(r2_score(y_test, y_test_pred), 3),
    }
    scores["MAE"] = {
        "train": np.round(mean_absolute_error(y_train, y_train_pred), 3),
        "test": np.round(mean_absolute_error(y_test, y_test_pred), 3),
    }

    return model, scores
