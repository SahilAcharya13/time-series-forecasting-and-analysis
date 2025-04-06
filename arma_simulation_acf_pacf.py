import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

np.random.seed(42)

ar1_positive = np.array([1, -0.9])
ar1_negative = np.array([1, 0.9])
ma = np.array([1])

ar = np.array([1])
ma1_positive = np.array([1, 0.5])
ma1_negative = np.array([1, -0.5])

def simulate_and_plot(ar, ma, title):
    process = ArmaProcess(ar, ma)
    simulated_data = process.generate_sample(nsample=500)

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

simulate_and_plot(ar1_positive, ma, "AR(1) with phi = 0.9")
simulate_and_plot(ar1_negative, ma, "AR(1) with phi = -0.9")
simulate_and_plot(ar, ma1_positive, "MA(1) with theta = 0.5")
simulate_and_plot(ar, ma1_negative, "MA(1) with theta = -0.5")