import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import astsadata


# Load dataset
def load_data():
    """Load the recruitment dataset and convert it to a time series."""
    # Load dataset from astsa package
    data = astsadata.rec
    # Convert to time series
    ts = pd.Series(data.iloc[:, 0].values, index=pd.date_range(start="1950", periods=len(data), freq="M"))
    return ts


# Plot time series and ACF/PACF
def plot_time_series(ts):
    """Plot the recruitment time series and ACF/PACF plots."""
    plt.figure(figsize=(8, 5))
    plt.plot(ts, label="Recruitment Series")
    plt.xlabel("Year")
    plt.ylabel("Number of New Fish")
    plt.title("Recruitment Time Series")
    plt.grid()
    plt.legend()
    plt.show()

    plot_acf(ts, lags=40)
    plt.grid()
    plt.show()

    plot_pacf(ts, lags=40)
    plt.grid()
    plt.show()


# Method of Moments (MoM) Estimation
def estimate_mom(ts, p):
    """Estimate AR coefficients using Yule-Walker equations."""
    phi = sm.regression.yule_walker(ts, order=p, method="mle")[0]
    return phi


# ARMA Model Estimation using MLE and OLS
def estimate_arma(ts, p, q, method="mle"):
    """Estimate ARMA parameters using Maximum Likelihood Estimation (MLE) or OLS."""
    model = sm.tsa.SARIMAX(ts, order=(p, 0, q), enforce_stationarity=False, enforce_invertibility=False)
    return model.fit()


# Sliding Window Forecasting for MoM
def sliding_window_forecast_mom(ts, p, steps=24):
    """Perform sliding window forecasting using the MoM estimator."""
    ts_diff = ts.diff().dropna()
    mo_coeffs = sm.regression.yule_walker(ts_diff, order=p, method="mle")[0]

    forecast_values = list(ts.iloc[-p:])
    for _ in range(steps):
        next_val = sum(mo_coeffs * np.array(forecast_values[-p:]))
        forecast_values.append(next_val)

    return forecast_values[p:]


# Forecast using ARMA model
def forecast_arma(model, steps=24):
    """Forecast future values using a fitted ARMA model."""
    return model.forecast(steps=steps)


# Main Execution
if __name__ == "__main__":
    ts = load_data()
    plot_time_series(ts)

    # Define ARMA order based on ACF/PACF
    p, q = 2, 2

    # Estimate parameters
    phi_mom = estimate_mom(ts, p)
    print("MoM AR Coefficients:", phi_mom)

    arma_mle = estimate_arma(ts, p, q, method="mle")
    print("MLE Parameters:\n", arma_mle.params)

    arma_ols = estimate_arma(ts, p, q, method="ols")
    print("OLS Parameters:\n", arma_ols.params)

    # Forecast using MoM, MLE, and OLS
    forecast_horizon = 24
    forecast_mom = sliding_window_forecast_mom(ts, p, steps=forecast_horizon)
    forecast_mle = forecast_arma(arma_mle, steps=forecast_horizon)
    forecast_ols = forecast_arma(arma_ols, steps=forecast_horizon)

    # Create forecast dates
    forecast_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq="M")

    # Plot Forecasts
    plt.figure(figsize=(10, 6))
    plt.plot(ts, label="Original Data", color='black')
    plt.plot(forecast_dates, forecast_mle, linestyle="dashed", label="MLE Forecast", color='blue')
    plt.plot(forecast_dates, forecast_ols, linestyle="dotted", label="OLS Forecast", color='red')
    plt.plot(forecast_dates, forecast_mom, linestyle="dashdot", label="MoM Forecast", color='green')

    plt.xlabel("Year")
    plt.ylabel("Recruitment")
    plt.title("24-Month Forecast using Different Estimators")
    plt.legend()
    plt.show()
