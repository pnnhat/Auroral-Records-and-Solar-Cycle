import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from astropy.timeseries import LombScargle

korean = pd.read_excel("data/KoreanAuroraRecords/Ancient Korean Aurora.xlsx")
chinese = pd.read_excel("data/ChineseDynastyRecords/Chinese Aurora Records.xlsx")

test_korea = pd.read_excel("data/Korean_Aurora_Grades_918_1392.xlsx")

# Periodogram Analysis for Korean Aurora Records (as yearly binned data)
year = test_korea["Year"].astype(int).values
grade = test_korea["Grade"].astype(float).values

year_min = year.min()
year_max = year.max()
years = np.arange(year_min, year_max + 1)  # yearly grid

yearly_amp = (
    pd.Series(grade, index=year).groupby(level=0).sum()
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

nf = 60000  # number of frequency points
frequency = np.linspace(min_freq, max_freq, nf)
power = ls.power(frequency)
power = np.maximum(power, 0.0)

period = 1 / frequency  # convert to period

plt.figure(figsize=(10, 6))
plt.plot(period, power, color="black")
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.xlim(min_period, max_period)
plt.grid(True, linestyle="--", alpha=0.4)
plt.axvline(11.4, color="red", linestyle="--", label="~11-year", alpha=0.8)
plt.xticks([6, 8, 10, 20, 40, 60, 80, 100])
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# zoom in
year = test_korea["Year"].astype(int).values
grade = test_korea["Grade"].astype(float).values

year_min = year.min()
year_max = year.max()
years = np.arange(year_min, year_max + 1)

yearly_amp = pd.Series(grade, index=year).groupby(level=0).sum()

amp = []
for y in years:
    if y in yearly_amp.index:
        amp.append(yearly_amp.loc[y])
    else:
        amp.append(0.0)

w = 40
baseline = (
    pd.Series(amp).rolling(window=w, center=True, min_periods=1).mean().to_numpy()
)
amp_hp = amp - baseline

ls = LombScargle(years, amp_hp, normalization="psd")

min_period = 6.0
max_period = 20.0
min_freq = 1.0 / max_period
max_freq = 1.0 / min_period

nf = 60000
frequency = np.linspace(min_freq, max_freq, nf)
power = ls.power(frequency)
power = np.maximum(power, 0.0)

period = 1.0 / frequency

plt.figure(figsize=(10, 6))
plt.plot(period, power, color="black")
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.xlim(min_period, max_period)
plt.grid(True, linestyle="--", alpha=0.4)
plt.axvline(11.4, color="red", linestyle="--", label="~11-year", alpha=0.8)
plt.xticks([6, 8, 10, 20])
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


## as event-series
t = test_korea["Year"].astype(int).values
y = test_korea["Grade"].astype(float).values

y = y - np.mean(y)

min_period = 6.0
max_period = 20.0
frequency = np.linspace(1 / max_period, 1 / min_period, 60000)
period = 1.0 / frequency

ls = LombScargle(t, y, normalization="psd")
power = ls.power(frequency)

plt.figure(figsize=(10, 6))
plt.plot(period, power, color="black")
plt.title("Lomb-Scargle Periodogram with event series")
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.xlim(min_period, max_period)
plt.grid(True, linestyle="--")
plt.axvline(13.2, color="red", linestyle="--", label="~13-year")

plt.xticks([6, 8, 10, 20])
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
