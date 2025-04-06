import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import astsadata
from sklearn.linear_model import LinearRegression

# Load dataset from astsa package
data = astsadata.rec

# Convert to time series
ts = pd.Series(data.iloc[:, 0].values, index=pd.date_range(start="1950", periods=len(data), freq="M"))

# --- Enhanced Time Series Plot ---
plt.figure(figsize=(12, 6))
plt.plot(ts, label="Recruitment Series", color='black', linewidth=2)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of New Fish", fontsize=14)
plt.title("Recruitment Time Series", fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Improved ACF & PACF Plots ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ts, lags=40, ax=ax[0], title="Autocorrelation Function (ACF)")
plot_pacf(ts, lags=40, ax=ax[1], title="Partial Autocorrelation Function (PACF)")
plt.show()

# Model order selection based on ACF/PACF (assume p=1, q=1)
p, q = 1, 1

# --- Maximum Likelihood Estimation (MLE) ---
arma_mle = sm.tsa.SARIMAX(ts, order=(p, 0, q), enforce_stationarity=False, enforce_invertibility=False).fit()
print("MLE Parameters:\n", arma_mle.params)

# --- Ordinary Least Squares (OLS) ---
lagged_data = pd.DataFrame({'Y': ts})
for i in range(1, p + 1):
    lagged_data[f'Y_lag{i}'] = ts.shift(i)
lagged_data.dropna(inplace=True)

X = lagged_data.drop(columns=['Y'])
Y = lagged_data['Y']

ols_model = LinearRegression()
ols_model.fit(X, Y)
ols_params = np.insert(ols_model.coef_, 0, ols_model.intercept_)
print("OLS Parameters:\n", ols_params)

# --- Method of Moments (MoM) using Yule-Walker ---
mo_coeffs = sm.regression.yule_walker(ts, order=p, method="mle")[0]
print("MoM AR Coefficients:", mo_coeffs)

# --- Forecasting ---
forecast_horizon = 24
forecast_mle = arma_mle.get_forecast(steps=forecast_horizon)
forecast_mle_mean = forecast_mle.predicted_mean
forecast_mle_ci = forecast_mle.conf_int()

forecast_ols = []
last_values = list(ts.iloc[-p:].values)
for _ in range(forecast_horizon):
    next_val = ols_params[0] + sum(ols_params[1:] * np.array(last_values[-p:]))
    forecast_ols.append(next_val)
    last_values.append(next_val)

forecast_mom = [ts.iloc[-1]]
for _ in range(forecast_horizon):
    next_val = sum(mo_coeffs * np.array(forecast_mom[-p:]))
    forecast_mom.append(next_val)
forecast_mom = forecast_mom[1:]

# --- Improved Forecast Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(ts, label="Original Data", color='black', linewidth=2)

forecast_index = pd.date_range(ts.index[-1], periods=forecast_horizon + 1, freq="M")[1:]
plt.plot(forecast_index, forecast_mle_mean, linestyle="dashed", color='blue', label="MLE Forecast", marker='o')
plt.fill_between(forecast_index, forecast_mle_ci.iloc[:, 0], forecast_mle_ci.iloc[:, 1], color='blue', alpha=0.2)
plt.plot(forecast_index, forecast_ols, linestyle="dotted", color='red', label="OLS Forecast", marker='s')
plt.plot(forecast_index, forecast_mom, linestyle="dashdot", color='green', label="MoM Forecast", marker='^')

plt.xlabel("Year", fontsize=14)
plt.ylabel("Recruitment", fontsize=14)
plt.title("24-Month Forecast using Different Estimators", fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()