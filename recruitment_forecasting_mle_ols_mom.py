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
ts = pd.Series(data.iloc[:, 0].values, index=pd.date_range(start="1950",
periods=len(data), freq="M"))
# Plot time series
plt.figure(figsize=(8,5))
plt.plot(ts, label="Recruitment Series")
plt.xlabel("Year")
plt.ylabel("Number of New Fish")
plt.title("Recruitment Time Series")
plt.legend()
plt.show()
# ACF & PACF plots
plot_acf(ts, lags=40)
plt.show()
plot_pacf(ts, lags=40)
plt.show()

# Model order selection based on ACF/PACF (assume p=1, q=1)
p, q = 1, 1
### 1. Maximum Likelihood Estimation (MLE) ###
arma_mle = sm.tsa.SARIMAX(ts, order=(p, 0, q), enforce_stationarity=False,
enforce_invertibility=False).fit()
print("MLE Parameters:\n", arma_mle.params)
### 2. Ordinary Least Squares (OLS) ###
# Create lagged variables for OLS regression
lagged_data = pd.DataFrame({'Y': ts})
for i in range(1, p + 1):
 lagged_data[f'Y_lag{i}'] = ts.shift(i)
lagged_data = lagged_data.dropna() # Drop missing values due to shifting
# Define X (independent) and Y (dependent)
X = lagged_data.drop(columns=['Y'])
Y = lagged_data['Y']
# Perform OLS regression
ols_model = LinearRegression()
ols_model.fit(X, Y)
ols_params = np.insert(ols_model.coef_, 0, ols_model.intercept_) # Add

print("OLS Parameters:\n", ols_params)
### 3. Method of Moments (MoM) using Yule-Walker ###
mo_coeffs = sm.regression.yule_walker(ts, order=p, method="mle")[0]
print("MoM AR Coefficients:", mo_coeffs)
# --- Forecasting ---
forecast_horizon = 24
# Forecast using MLE
forecast_mle = arma_mle.forecast(steps=forecast_horizon)
# Forecast using OLS
forecast_ols = []
last_values = list(ts.iloc[-p:].values)
for _ in range(forecast_horizon):
 next_val = ols_params[0] + sum(ols_params[1:] *
np.array(last_values[-p:]))
 forecast_ols.append(next_val)
 last_values.append(next_val)
# Forecast using MoM (Yule-Walker)
forecast_mom = [ts.iloc[-1]]
for _ in range(forecast_horizon):
 next_val = sum(mo_coeffs * np.array(forecast_mom[-p:]))
 forecast_mom.append(next_val)
forecast_mom = forecast_mom[1:]
# --- Plot Forecasts ---
plt.figure(figsize=(10, 6))
plt.plot(ts, label="Original Data", color='black')
plt.plot(pd.date_range(ts.index[-1], periods=forecast_horizon, freq="M"),
forecast_mle, linestyle="dashed", label="MLE Forecast", color='blue')
plt.plot(pd.date_range(ts.index[-1], periods=forecast_horizon, freq="M"),
forecast_ols, linestyle="dotted", label="OLS Forecast", color='red')
plt.plot(pd.date_range(ts.index[-1], periods=forecast_horizon, freq="M"),
forecast_mom, linestyle="dashdot", label="MoM Forecast", color='green')
plt.xlabel("Year")
plt.ylabel("Recruitment")
plt.title("24-Month Forecast using Different Estimators")
plt.legend()
plt.show()