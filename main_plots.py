import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq
from astropy.timeseries import LombScargle

korean = pd.read_excel("data/KoreanAuroraRecords/Korean_Auroral_Full.xlsx")
chinese = pd.read_excel("data/ChineseDynastyRecords/Chinese Aurora Records.xlsx")
korean_grades = pd.read_excel("data/KoreanAuroraRecords/Korean_Aurora_Grades_918_1392.xlsx")

# Histograms
## Histogram of Korean auroral records
year = korean["Year"].astype(int).values

year_min = int(year.min())
year_max = int(year.max())
years = np.arange(year_min, year_max + 1)

yearly_counts = pd.Series(1, index=year).groupby(level=0).sum()
counts = np.array(
    [yearly_counts.get(y, 0) for y in years],
    dtype=float
)

plt.figure(figsize=(12, 4))
plt.bar(years, counts, width=1.0, color="gray", edgecolor="gray", alpha=0.8)

smooth_counts = gaussian_filter1d(counts, sigma=2.0)
plt.plot(years, smooth_counts, color="black", linewidth=2.2, zorder=10)

plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.xlim(year_min, year_max)
plt.xticks([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700])
plt.text(1010, 4.0, "Oort", fontsize=18)
plt.text(1290, 3.5, "Wolf", fontsize=18)
plt.text(1450, 5.5, "Sporer", fontsize=18)

plt.axvspan(1645, 1715, color="gray", alpha=0.2)
plt.text(1680, 10, "Maunder\nMinimum", fontsize=18, ha="center", va="center")

plt.tight_layout()
plt.show()

## Stacked Histogram
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


# Stem plot
test_korea = pd.read_excel("data/KoreanAuroraRecords/Korean_Aurora_Grades_918_1392.xlsx")
year = test_korea["Year"].astype(int).values
magni = test_korea["Magnitude"].astype(int).values
### restrict to 1000–1400
mask = (year >= 1000) & (year <= 1400)
year = year[mask]
magni = magni[mask]
plt.figure(figsize=(12, 4))
plt.vlines(year, ymin=0, ymax=magni, color="black", linewidth=1)
plt.xlim(1000, 1400)
plt.ylim(0, 5.5)

plt.xlabel("Year")
plt.ylabel("Magnitude")
plt.yticks([1, 2, 3, 4, 5])
plt.xticks(np.arange(1000, 1401, 50))
plt.grid(False)
plt.tight_layout()
plt.show()

# Periodogram Analysis
## Korean as event-series
# Build binned annual series
year = korean["Year"].astype(int).to_numpy()
year_min = int(year.min())
year_max = int(year.max())
years = np.arange(year_min, year_max + 1)

yearly_counts = pd.Series(1, index=year).groupby(level=0).sum()
counts = np.array(
    [float(yearly_counts.loc[y]) if y in yearly_counts.index else 0.0
     for y in years],
    dtype=float,
)

# counts = np.clip(counts, 0, 1)   # <-- uncomment for binary: 1 if any aurora that year, else 0

counts = counts - counts.mean()

min_period = 6.0
max_period = 50.0
nf = 60000
frequency = np.linspace(1.0 / max_period, 1.0 / min_period, nf)
period = 1.0 / frequency

ls = LombScargle(years, counts, normalization="psd")
power = np.maximum(ls.power(frequency), 0.0)

plt.figure(figsize=(10, 6))
plt.plot(period, power, color="black")
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.xlim(min_period, max_period)
plt.grid(True, linestyle="--", alpha=0.4)
plt.axvline(11.0, color="red", linestyle="--", label="~11-year")
plt.xticks([6, 8, 10, 20, 40, 50])
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


## Korean as yearly series (magnitude-weighted)
# Build binned annual series, summing grades per year
year = korean_grades["Year"].astype(int).values
magni = korean_grades["Magnitude"].astype(float).values

year_min = int(year.min())
year_max = int(year.max())
years = np.arange(year_min, year_max + 1)

yearly_amp = pd.Series(magni, index=year).groupby(level=0).sum()
amp = np.array(
    [float(yearly_amp.loc[y]) if y in yearly_amp.index else 0.0
     for y in years],
    dtype=float,
)

# Subtract 40-year rolling baseline (high-pass)
w = 40
baseline = pd.Series(amp).rolling(window=w, center=True, min_periods=1).mean().to_numpy()
amp_hp = amp - baseline

min_period = 6.0
max_period = 110.0
nf = 60000
frequency = np.linspace(1.0 / max_period, 1.0 / min_period, nf)
period = 1.0 / frequency

ls = LombScargle(years, amp_hp, normalization="psd")
power = np.maximum(ls.power(frequency), 0.0)

plt.figure(figsize=(10, 6))
plt.plot(period, power, color="black")
plt.title("Lomb-Scargle Periodogram with yearly series")
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.xlim(min_period, max_period)
plt.grid(True, linestyle="--", alpha=0.4)
plt.axvline(11.4, color="red", linestyle="--", label="~11-year", alpha=0.8)
plt.xticks([6, 8, 10, 20, 40, 60, 80, 100])
plt.legend(frameon=False)
plt.tight_layout()
plt.show()