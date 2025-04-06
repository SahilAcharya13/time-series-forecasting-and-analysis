import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import itertools
import warnings
warnings.filterwarnings("ignore")

# Download Data
nifty = yf.download("^NSEI", start="2004-01-01",end="2024-12-26")[['Close']]

plt.figure(figsize=(12, 6))
plt.plot(nifty['Close'], label='NIFTY Closing Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('NIFTY 50 Closing Price Time Series')
plt.legend()
plt.grid()

plt.show()

# Perform ADF Test for Stationarity
def adf_test(series):
     result = adfuller(series.dropna())
     print(f'ADF Statistic: {result[0]}')
     print(f'p-value: {result[1]}')
     if result[1] <= 0.05:
        print("Series is stationary.")
     else:
        print("Series is NOT stationary. Differencing is needed.")
adf_test(nifty['Close'])

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(nifty['Close'].dropna(), lags=50, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(nifty['Close'].dropna(), lags=50, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Fit Five Different ARIMA Models
arima_orders = [(1, 1, 1), (2, 1, 2), (0, 1, 1), (1, 1, 0), (2, 1, 1)]
models = {}
for order in arima_orders:
     try:
        model = ARIMA(nifty['Close'], order=order)
        model_fit = model.fit()
        models[order] = model_fit
        print(f"Fitted ARIMA{order} Model Successfully")
     except Exception as e:
        print(f"Error fitting ARIMA{order}: {e}")


# Select the Best ARIMA Model Based on AIC/BIC
best_aic, best_bic = float("inf"), float("inf")
best_model_aic, best_model_bic = None, None
for order, model_fit in models.items():
     aic, bic = model_fit.aic, model_fit.bic
     if aic < best_aic:
        best_aic = aic
        best_model_aic = order
     if bic < best_bic:
         best_bic = bic
         best_model_bic = order
print(f"Best model based on AIC: ARIMA{best_model_aic} with AIC={best_aic:.2f}")
print(f"Best model based on BIC: ARIMA{best_model_bic} with BIC={best_bic:.2f}")

# Fit the Best ARIMA Model
best_order = best_model_aic
model = ARIMA(nifty['Close'], order=best_order)
model_fit = model.fit()

# Forecast Using the Best Model
forecast_steps = [1, 2, 5]
forecast_results = {}
actual_values = nifty['Close'].iloc[-5:]
forecast_5_days = model_fit.forecast(steps=5)
for m in forecast_steps:
    forecast_results[m] = forecast_5_days.iloc[:m]

# Evaluate Forecasting Performance & Plot Results
metrics_results = {}
fig, axes = plt.subplots(1, len(forecast_steps), figsize=(15, 5))
for idx, m in enumerate(forecast_steps):
     rmse = np.sqrt(mean_squared_error(actual_values.iloc[:m], forecast_results[m]))
     mae = mean_absolute_error(actual_values.iloc[:m], forecast_results[m])
     r2 = r2_score(actual_values.iloc[:m], forecast_results[m])
     metrics_results[m] = {'RMSE': rmse, 'MAE': mae, 'R2-Score': r2}
     print(f'm={m}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2-Score={r2:.4f}')
     # Plot forecast vs actual
     axes[idx].plot(range(len(actual_values.iloc[:m])),actual_values.iloc[:m], marker='o', label='Actual', color='blue')
     axes[idx].plot(range(len(forecast_results[m])), forecast_results[m],marker='s', label='Forecasted', color='red')
     axes[idx].set_title(f'Forecast vs Actual (m={m})')
     axes[idx].set_xlabel('Time Steps')
     axes[idx].set_ylabel('Closing Price')
     axes[idx].legend()
plt.tight_layout()
plt.show()


# Fit five different ARIMA models and plot their ACF and PACF
from statsmodels.tsa.arima.model import ARIMA
fig, axes = plt.subplots(5, 2, figsize=(14, 20))
orders = [(1, 1, 1), (2, 1, 2), (3, 1, 3), (1, 1, 2), (2, 1, 1)]
for i, order in enumerate(orders):
    model = ARIMA(nifty['Close'], order=order)
    fitted_model = model.fit()
    # Get residuals
    residuals = fitted_model.resid
    # Plot ACF and PACF of residuals
    plot_acf(residuals, lags=50, ax=axes[i, 0])
    axes[i, 0].set_title(f'ACF of Residuals (ARIMA{order})')
    plot_pacf(residuals, lags=50, ax=axes[i, 1])
    axes[i, 1].set_title(f'PACF of Residuals (ARIMA{order})')
plt.tight_layout()
plt.show()