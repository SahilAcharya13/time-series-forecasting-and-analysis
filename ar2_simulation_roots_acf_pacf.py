import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

# Set seed for reproducibility
np.random.seed(42)

# Define AR(2) process with Xt = 1.5Xt-1 - 0.75Xt-2 + Wt
ar2 = np.array([1, -1.5, 0.75])  # AR coefficients
ma = np.array([1])  # No MA terms

# Find roots of the characteristic equation
roots = np.roots(ar2)
print("Roots of AR(2) polynomial:", roots)


# Simulate 500 observations
def simulate_and_plot(ar, ma, title):
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


# Run simulation and plot results
simulate_and_plot(ar2, ma, "AR(2) with phi1 = 1.5, phi2 = -0.75")
