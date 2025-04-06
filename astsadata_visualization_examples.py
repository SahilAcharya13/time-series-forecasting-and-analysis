import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astsadata import *

# Example 1.1
jj.plot(xlabel='Time', ylabel='Quarterly Earnings per Share', marker='o',
        legend=False)
plt.show()

# Example 1.2
globtemp.plot(xlabel='Time', ylabel='Global Temperature Deviations',
                marker='o', legend=False)
plt.show()

# Example 1.3
speech.plot(legend=False)
plt.show()

# Example 1.4
djiar = djia['Close'].apply(np.log).diff()
djiar.plot(title="DJIA Returns")
plt.show()

# Example 1.5
fig, axes = plt.subplots(nrows=2, figsize=(7, 5))
soi.plot(ax=axes[0], title="Southern Oscillation Index", legend=False)
rec.plot(ax=axes[1], title="Recruitment", legend=False)
fig.tight_layout()
plt.show()

# Example 1.6
fig, axes = plt.subplots(nrows=2, figsize=(7, 5))
fmri1.iloc[:, 1:5].plot(ax=axes[0], ylabel="BOLD", xlabel="", title="Cortex",
                        legend=False)
fmri1.iloc[:, 5:10].plot(ax=axes[1], ylabel="BOLD", xlabel="Time (1 pt = 2 sec)",
                         title="Thalamus & Cerebellum", legend=False)
fig.tight_layout()
plt.show()

# Example 1.7
fig, axes = plt.subplots(nrows=2, figsize=(7, 5))
EQ5.plot(ax=axes[0], ylabel="EQ5", title="Earthquake", legend=False)
EXP6.plot(ax=axes[1], ylabel="EXP6", title="Explosion", legend=False)
fig.tight_layout()
plt.show()