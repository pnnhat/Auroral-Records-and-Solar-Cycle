import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from astropy.timeseries import LombScargle
korean = pd.read_excel("data/KoreanAuroraRecords/Korean_Auroral_Full.xlsx")
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

# labels
plt.text(1010, 4.0, "Oort", fontsize=18)
plt.text(1290, 3.5, "Wolf", fontsize=18)
plt.text(1450, 5.5, "Sporer", fontsize=18)

plt.axvspan(1645, 1715, color="gray", alpha=0.2)
plt.text(1680, 10, "Maunder\nMinimum", fontsize=18, ha="center", va="center")

plt.tight_layout()
plt.show()


## Korean as event-series
year = korean["Year"].astype(int).to_numpy()
year_min = int(year.min())
year_max = int(year.max())
years = np.arange(year_min, year_max + 1)
yearly_counts = pd.Series(1, index=year).groupby(level=0).sum()

counts = np.array(
    [float(yearly_counts.loc[y]) if y in yearly_counts.index else 0.0
     for y in years],
    dtype=float
)
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


chinese = pd.read_excel("data/ChineseDynastyRecords/Chinese Aurora Records.xlsx")
test_korea = pd.read_excel("data/Korean_Aurora_Grades_918_1392.xlsx")

# Stem plot
year = test_korea["Year"].astype(int).values
magni = test_korea["Magnitude"].astype(int).values

# restrict to 1000–1400
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

# Periodogram Analysis for Korean Aurora Records
year = test_korea["Year"].astype(int).values
magni = test_korea["Magnitude"].astype(float).values

year_min = year.min()
year_max = year.max()
years = np.arange(year_min, year_max + 1)  # yearly grid

yearly_amp = (
    pd.Series(magni, index=year).groupby(level=0).sum()
)  # groups all events that occurred in the same year and sums their grades

amp = []  # fill missing years with 0, so the series is a binned annual time seris
for y in years:
    if y in yearly_amp.index:
        amp.append(yearly_amp.loc[y])
    else:
        amp.append(0.0)

w = 40  # compute moving average baseline and subtract it
baseline = (
    pd.Series(amp).rolling(window=w, center=True, min_periods=1).mean().to_numpy()
)
amp_hp = amp - baseline

ls = LombScargle(years, amp_hp, normalization="psd")

min_period = 6.0
max_period = 110.0
min_freq = 1.0 / max_period
max_freq = 1.0 / min_period

nf = 60000
frequency = np.linspace(min_freq, max_freq, nf)
power = ls.power(frequency)
power = np.maximum(power, 0.0)

period = 1 / frequency

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


# Lomb–Scargle with Monte Carlo significance levels
year = korean["Year"].astype(int).values

year_min = int(year.min())
year_max = int(year.max())
years = np.arange(year_min, year_max + 1)
yearly_counts = pd.Series(1, index=year).groupby(level=0).sum()

amp = np.array(
    [float(yearly_counts.loc[y]) if y in yearly_counts.index else 0.0 for y in years],
    dtype=float,
)
amp = amp - amp.mean()

min_period = 6.0
max_period = 110.0
nf = 60000

frequency = np.linspace(1.0 / max_period, 1.0 / min_period, nf)
period = 1.0 / frequency

ls_obs = LombScargle(years, amp, normalization="psd")
power_obs = np.maximum(ls_obs.power(frequency), 0.0)

n_mc = 10000
rng = np.random.default_rng(42)

power_mc = np.zeros((n_mc, nf))

for i in range(n_mc):
    year_rand = rng.integers(year_min, year_max + 1, size=len(year))

    yearly_rand = pd.Series(1, index=year_rand).groupby(level=0).sum()

    amp_rand = np.array(
        [float(yearly_rand.loc[y]) if y in yearly_rand.index else 0.0 for y in years],
        dtype=float,
    )

    amp_rand = amp_rand - amp_rand.mean()

    ls_rand = LombScargle(years, amp_rand, normalization="psd")

    power_mc[i] = np.maximum(ls_rand.power(frequency), 0.0)

sig_014 = np.percentile(power_mc, 99.86, axis=0)
sig_030 = np.percentile(power_mc, 99.70, axis=0)
plt.figure(figsize=(10,6))

plt.plot(period, power_obs, color="black", label="Observed")
plt.plot(period, sig_014, "k:", linewidth=1.2, label="Upper 0.14%")
plt.plot(period, sig_030, "k--", linewidth=1.2, label="Upper 0.30%")
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.xlim(min_period, max_period)
plt.grid(True, linestyle="--", alpha=0.4)
plt.axvline(11.0, color="red", linestyle="--", label="~11-year")
plt.xticks([6,8,10,20,40,60,80,100])
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


