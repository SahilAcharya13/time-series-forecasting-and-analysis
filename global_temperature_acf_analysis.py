import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, acf
from astsadata import varve
data = varve.squeeze()
data.index = data.index.astype(str).astype(int)
plt.plot(data.index, data.values, marker='o', linestyle='-', color='b', label='Original Varve Series')
plt.xlabel("Year")
plt.ylabel("Varve Thickness")
plt.title("Paleoclimatic Glacial Varves Time Series")
plt.legend()
plt.grid()
plt.show()

plot_acf(data, lags=20)
plt.title("ACF of Original Varve Series-1")
plt.show()

log_data = np.log(data)

plot_acf(log_data, lags=20)
plt.title("ACF of Log-Transformed Varve Series")
plt.show()

diff_log_data = log_data.diff().dropna()
plot_acf(diff_log_data, lags=20)
plt.title("ACF of First-Differenced Log-Transformed Varve Series")
plt.show()

mean_original = data.mean()
mean_log = log_data.mean()
mean_differenced_log = diff_log_data.mean()

# Step 9: Compute ACF Values for First 5 Lags
acf_original = acf(data, nlags=5)
acf_log = acf(log_data, nlags=5)
acf_diff_log = acf(diff_log_data, nlags=5)

# Print Results
print(f"Mean of Original Time Series: {mean_original:.4f}")
print(f"Mean of Log-Transformed Series: {mean_log:.4f}")
print(f"Mean of First-Differenced Log Series: {mean_differenced_log:.4f}\n")

print("ACF Values for First 5 Lags:")
print(f"Original Series ACF: {acf_original}")
print(f"Log-Transformed Series ACF: {acf_log}")
print(f"First-Differenced Log Series ACF: {acf_diff_log}")
