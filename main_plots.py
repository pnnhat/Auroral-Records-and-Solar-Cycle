import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq
from astropy.timeseries import LombScargle

korean = pd.read_excel("data/KoreanAuroraRecords/Ancient Korean Aurora.xlsx")
chinese = pd.read_excel("data/ChineseDynastyRecords/Chinese Aurora Records.xlsx")

# Stacked Histogram
k_years = korean["Year"]
c_years = chinese["Year"]
min_year = min(k_years.min(), c_years.min())
max_year = max(k_years.max(), c_years.max())

bins = np.arange(min_year, max_year + 2, 1)

fig, ax = plt.subplots(figsize=(15, 6))

ax.hist(
    [k_years, c_years],
    bins=bins,
    stacked=True,
    histtype="step",
    linewidth=1.8,
    edgecolor=["red", "blue"],
    label=["Korean Auroras", "Chinese Auroras"],
)

ax.axvspan(1645, 1715, color="lightgray", alpha=0.5)
ax.text(
    1647,
    ax.get_ylim()[1] * 0.9,
    "Maunder Minimum\n(1645–1715)",
    fontsize=12,
    ha="left",
    va="top",
)
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Aurora Records", fontsize=14)
ax.set_title("Histogram of Korean and Chinese Auroral Records", fontsize=16)

ax.legend(frameon=False)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

plt.tight_layout()
plt.show()

# Periodogram Analysis for Aurora Records
years_event = pd.concat([korean["Year"], chinese["Year"]]).dropna().astype(int)
year_min = years_event.min()
year_max = years_event.max()

years_full = np.arange(year_min, year_max + 1)

counts = pd.Series(years_event).value_counts().sort_index()
counts_full = []
for y in years_full:
    if y in counts:
        counts_full.append(counts[y])
    else:
        counts_full.append(0)
counts_full = np.array(counts_full)

counts_full = counts_full - np.mean(counts_full)

ls = LombScargle(years_full, counts_full, normalization="psd")

min_period = 2.0
max_period = 30.0

min_freq = 1 / max_period
max_freq = 1 / min_period

Nf = 40000
frequency = np.linspace(min_freq, max_freq, Nf)
power_ls = ls.power(frequency)
period_ls = 1 / frequency

plt.figure(figsize=(10, 6))
plt.plot(period_ls, power_ls, color="black")
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.title("Lomb–Scargle Periodogram (combined Aurora Records)")
plt.xlim(2, 30)
plt.grid(True, linestyle="--", alpha=0.4)

plt.axvline(8, color="blue", linestyle="--", label="8-year")
plt.axvline(11, color="red", linestyle="--", label="11-year")
plt.axvline(22, color="green", linestyle="--", label="22-year")

plt.xticks([6, 8, 10, 20, 30])
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
