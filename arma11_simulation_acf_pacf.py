import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

# Set seed for reproducibility
np.random.seed(42)

# Define ARMA(1,1) processes with different parameter combinations
arma_params = [
    (0.9, 0.5),
    (0.9, -0.5),
    (-0.9, 0.5),
    (-0.9, -0.5)
]


# Simulate and plot ACF and PACF for each case
def simulate_and_plot(ar_coeff, ma_coeff, title):
    ar = np.array([1, -ar_coeff])  # AR coefficients
    ma = np.array([1, ma_coeff])  # MA coefficients
    process = ArmaProcess(ar, ma)
    simulated_data = process.generate_sample(nsample=500)

    # Plot time series
    plt.figure(figsize=(12, 4))
    plt.plot(simulated_data, label=title)
    plt.title(title)
    plt.legend()
    plt.show()

    # Plot ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sm.graphics.tsa.plot_acf(simulated_data, lags=40, ax=axes[0])
    sm.graphics.tsa.plot_pacf(simulated_data, lags=40, ax=axes[1])
    axes[0].set_title(f"ACF of {title}")
    axes[1].set_title(f"PACF of {title}")
    plt.show()


# Run simulations for all four cases
for ar_coeff, ma_coeff in arma_params:
    title = f"ARMA(1,1) with phi={ar_coeff}, theta={ma_coeff}"
    simulate_and_plot(ar_coeff, ma_coeff, title)
