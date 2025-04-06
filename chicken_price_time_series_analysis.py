import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from astsadata import chicken
from statsmodels.tsa.stattools import acf
chicken_series = pd.Series(chicken.values.flatten(),
              index=pd.date_range(start="1924-01",periods=len(chicken),freq="M"))
plt.figure(figsize=(7,5))
plt.plot(chicken_series, label="Chicken Prices", color="blue")
plt.title("Monthly U.S. Retail Broiler Prices")
plt.xlabel("Time")
plt.ylabel("Cents per Pound")
plt.legend()
plt.show()

sm.graphics.tsa.plot_acf(chicken_series, lags=40, title="ACF of Chicken Prices")
plt.show()

X = sm.add_constant(np.arange(len(chicken_series)))
model = sm.OLS(chicken_series, X).fit()
detrended_series = chicken_series - model.predict(X)

plt.figure(figsize=(7,5))
plt.plot(detrended_series, label="Detrended Chicken Prices", color="red")
plt.title("Detrended Chicken Prices (Residuals)")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.legend()
plt.show()


sm.graphics.tsa.plot_acf(detrended_series, lags=40, title="ACF of Detrended Series")
plt.show()


diff_series = chicken_series.diff().dropna()

plt.figure(figsize=(7,5))
plt.plot(diff_series, label="First Difference of Chicken Prices", color="green")
plt.title("First Difference of Chicken Prices")
plt.xlabel("Time")
plt.ylabel("Differenced Values")
plt.legend()
plt.show()

#ACF of first differenced series
sm.graphics.tsa.plot_acf(diff_series, lags=40, title="ACF of First Differenced Series")
plt.show()


mean_original = chicken_series.mean()
mean_detrended = detrended_series.mean()
mean_diff = diff_series.mean()
print(f"Mean of Original Series: {mean_original:.4f}")
print(f"Mean of Detrended Series: {mean_detrended:.4f}")
print(f"Mean of First Differenced Series: {mean_diff:.4f}")

acf_temp = acf(chicken_series, nlags=5)
acf_detrended = acf(detrended_series, nlags=5)
acf_diff_temp = acf(diff_series, nlags=5)


print("ACF Values for First 5 Lags:")
print(f"Original Original Series ACF: {acf_temp}")
print(f"Detrended Series ACF: {acf_detrended}")
print(f"First-Differenced  Series ACF: {acf_diff_temp}")