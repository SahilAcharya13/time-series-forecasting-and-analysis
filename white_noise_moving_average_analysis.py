import numpy as np
import matplotlib.pyplot as plt

# Part (a): White Noise Time Series

# Generate white noise random process
N = 2000
white_noise = np.random.normal(0, 1, N)
time = np.arange(N)

# Plot the white noise time series
plt.figure(figsize=(12, 6))
plt.plot(time, white_noise, label="White Noise")
plt.title("White Noise Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

# Check stationarity by dividing into chunks of 100 samples
chunk_size = 100
num_chunks = N // chunk_size
means = []
variances = []

for i in range(num_chunks):
    chunk = white_noise[i * chunk_size:(i + 1) * chunk_size]
    means.append(np.mean(chunk))
    variances.append(np.var(chunk))

# Plot the chunk means and variances
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(range(num_chunks), means, marker='o', label="Chunk Means")
plt.axhline(np.mean(white_noise), color='r', linestyle='--', label="Global Mean")
plt.title("Chunk Means")
plt.xlabel("Chunk Index")
plt.ylabel("Mean")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(range(num_chunks), variances, marker='o', label="Chunk Variances")
plt.axhline(np.var(white_noise), color='r', linestyle='--', label="Global Variance")
plt.title("Chunk Variances")
plt.xlabel("Chunk Index")
plt.ylabel("Variance")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Plot ACF of white noise using matplotlib.pyplot.acorr
plt.figure(figsize=(12, 6))
plt.acorr(white_noise, maxlags=50, usevlines=True, normed=True)
plt.title("Autocorrelation Function of White Noise")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.grid()
plt.show()


# Input: Number of taps (M)
M = int(input("Enter the number of taps (M > 1): "))

# Apply M-tap Moving Average Filter
filtered_series = np.convolve(white_noise, np.ones(M)/M, mode='valid')
time_filtered = np.arange(len(filtered_series))

# Plot the filtered time series
plt.figure(figsize=(12, 6))
plt.plot(time_filtered, filtered_series, label=f"Filtered Series (M={M})")
plt.title("Filtered Time Series (M-tap Moving Average)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

# Plot ACF of filtered time series using matplotlib.pyplot.acorr
plt.figure(figsize=(12, 6))
plt.acorr(filtered_series, maxlags=50, usevlines=True, normed=True)
plt.title("Autocorrelation Function of Filtered Series")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.grid()
plt.show()

# Conclusion: Compare results and summarize findings
print("\nConclusion:")
print("- The white noise series has zero mean and unity variance, with no significant autocorrelations.")
print(f"- The {M}-tap moving average filter smooths the series and introduces correlations in the filtered data.")
