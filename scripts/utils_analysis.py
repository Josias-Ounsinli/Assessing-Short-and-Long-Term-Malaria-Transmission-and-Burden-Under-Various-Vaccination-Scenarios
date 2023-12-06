import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings


def estimate_var_effect_with_arima(
    data: pd.DataFrame,
    targets,
    indep_vars,
):
    # Iterate through each target variable
    for target in targets:
        # Specify the endogenous (dependent) variable for the current target
        endog = data[target]

        # Set a range for p, d, q values for the ARIMA model
        p_range = range(3)  # Change this range based on your data
        d_range = range(3)  # Change this range based on your data
        q_range = range(3)  # Change this range based on your data

        best_aic = np.inf
        best_order = None
        best_results = None

        # Iterate through combinations of p, d, q values
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    # Specify the exogenous variables (if any) for the current target
                    exog = data[
                        indep_vars
                    ]  # Adjust for your exog variables for each target

                    # Fit the ARIMA model for the current target
                    try:
                        # Suppress warnings temporarily
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            model = sm.tsa.ARIMA(endog, order=(p, d, q), exog=exog)
                            results = model.fit()

                        # Calculate AIC
                        current_aic = results.aic

                        # Update the best model if current AIC is lower
                        if current_aic < best_aic:
                            best_aic = current_aic
                            best_order = (p, d, q)
                            best_results = results
                    except:
                        continue

        # Print only the table summary for the best model of the current target
        print(f"Best ARIMA order for {target}:", best_order)
        print(best_results.summary().tables[0])
        print(best_results.summary().tables[1])
