import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq
from astropy.timeseries import LombScargle

korean = pd.read_excel("data/KoreanAuroraRecords/Ancient Korean Aurora.xlsx")
chinese = pd.read_excel("data/ChineseDynastyRecords/Chinese Aurora Records.xlsx")

# Remake korean auroral power by Gaussian smoothed counts
years_event = korean["Year"].dropna().astype(int)
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
activity_proxy = gaussian_filter1d(counts_full.astype(float), sigma=2.5)

ls = LombScargle(years_full, activity_proxy)

frequency, power = ls.autopower(minimum_frequency=1 / 120, maximum_frequency=1 / 2)

period = 1 / frequency

plt.figure(figsize=(6, 4))

plt.plot(period, power, color="black", linewidth=1.5)

plt.xscale("log")
plt.xlim(6, 120)
plt.axvline(11, linestyle="--", color="black", linewidth=1)
plt.axvline(22, linestyle=":", color="black", linewidth=1)

plt.xlabel("Period (Year)")
plt.ylabel("Power")
plt.title("Aurora Power Spectrum")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# Periodogram Analysis for Korean Aurora Records (tuning parameters)
korean_pre1700 = korean.loc[korean["Year"] < 1400, "Year"]
years_event = pd.concat([korean_pre1700]).dropna().astype(int)

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

ls = LombScargle(years_full, counts_full)

min_period = 2.0
max_period = 30.0

min_freq = 1 / max_period
max_freq = 1 / min_period

Nf = 30000
frequency = np.linspace(min_freq, max_freq, Nf)
power_ls = ls.power(frequency)
period_ls = 1 / frequency

plt.figure(figsize=(7, 5))
plt.plot(period_ls, power_ls, color="black")
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.title("Lomb–Scargle Periodogram (combined Aurora Records)")
plt.xlim(2, 30)
plt.grid(True, linestyle="--", alpha=0.4)

plt.axvline(8, color="blue", linestyle="--", label="8-year")
plt.axvline(11, color="red", linestyle="--", label="11-year")
plt.axvline(22, color="green", linestyle="--", label="22-year")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


# Lomb-Scargle with Monte Carlo
years_event_korea = korean["Year"].dropna().astype(float).values
years_event_china = chinese["Year"].dropna().astype(float).values
N_korea = len(years_event_korea)
N_china = len(years_event_china)

density_ratio = N_china / N_korea
values_event_korea = np.ones_like(years_event_korea) * density_ratio
values_event_china = np.ones_like(years_event_china)
years_event = np.concatenate([years_event_korea, years_event_china])
values_event = np.concatenate([values_event_korea, values_event_china])


min_period = 2.0
max_period = 30.0

frequency = np.linspace(1 / max_period, 1 / min_period, 40000)

ls_obs = LombScargle(
    years_event,
    values_event,
)

power_obs = ls_obs.power(frequency)
period = 1 / frequency

N_events = len(years_event)
t_min = years_event.min()
t_max = years_event.max()

n_mc = 10000
power_mc = np.zeros((n_mc, len(frequency)))

rng = np.random.default_rng(seed=42)

for i in range(n_mc):
    years_random = rng.uniform(t_min, t_max, size=N_events)
    values_random = values_event.copy()

    ls_rand = LombScargle(years_random, values_random, normalization="psd")
    power_mc[i] = ls_rand.power(frequency)

sig_95 = np.percentile(power_mc, 95, axis=0)
sig_99 = np.percentile(power_mc, 99, axis=0)
sig_999 = np.percentile(power_mc, 99.9, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(period, power_obs, color="black", label="Observed")
plt.plot(period, sig_95, color="gray", linestyle="--", label="95%")
plt.plot(period, sig_99, color="gray", linestyle="-.", label="99%")
plt.plot(period, sig_999, color="gray", linestyle=":", label="99.9%")

plt.axvline(11, color="red", linestyle="--", label="11-year")
plt.axvline(22, color="green", linestyle="--", label="22-year")

plt.xlim(2, 30)
plt.xlabel("Period (years)")
plt.ylabel("Power (PSD)")
plt.title("Density-Weighted Event-Based Lomb–Scargle\nwith Monte Carlo Significance")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# Monte Carlo Significance Testing for FFT Power Spectrum
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

x = counts_full - counts_full.mean()

# remove slow secular structure
t = (years_full - years_full.mean()).astype(float)
deg = 2
trend = np.polyval(np.polyfit(t, x, deg=deg), t)
x = x - trend

window = np.hanning(len(x))
xw = x * window

pad_factor = 8
nfft = pad_factor * len(xw)

fft_vals = rfft(xw, n=nfft)
freqs = rfftfreq(nfft, d=1.0)

power_obs = np.abs(fft_vals) ** 2
period = 1 / freqs

mask = (freqs > 0) & (period >= 2) & (period <= 30)
period_p = period[mask]
power_p = power_obs[mask]

n_mc = 10000
rng = np.random.default_rng(42)

power_mc = np.zeros((n_mc, power_p.size))

for i in range(n_mc):
    years_random = rng.integers(year_min, year_max + 1, size=len(years_event))

    counts_rand = pd.Series(years_random).value_counts()
    x_rand = np.array([counts_rand.get(y, 0) for y in years_full], dtype=float)

    x_rand = x_rand - x_rand.mean()
    trend_r = np.polyval(np.polyfit(t, x_rand, deg=deg), t)
    x_rand = x_rand - trend_r

    xw_rand = x_rand * window

    fft_rand = rfft(xw_rand, n=nfft)
    power_rand = np.abs(fft_rand) ** 2

    power_mc[i] = power_rand[mask]

conf_997 = np.percentile(power_mc, 99.7, axis=0)
conf_999 = np.percentile(power_mc, 99.9, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(period_p, power_p, color="black", linewidth=1.5, label="Observed")
plt.plot(period_p, conf_997, color="gray", linestyle="--", label="99.7%")
plt.plot(period_p, conf_999, color="gray", linestyle=":", label="99.9%")

plt.axvline(11, color="red", linestyle="--", label="11-year")
plt.axvline(22, color="green", linestyle="--", label="22-year")

plt.xlim(2, 30)
plt.xlabel("Period (years)")
plt.ylabel("Fourier Power")
plt.title("FFT Power Spectrum with Monte Carlo Confidence")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


# Take before 1400 only
korean_pre1400 = korean.loc[korean["Year"] < 1400, "Year"]
chinese_pre1400 = chinese.loc[chinese["Year"] < 1400, "Year"]

years_event = pd.concat([korean_pre1400, chinese_pre1400]).dropna().astype(int)
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

x = counts_full - counts_full.mean()

# remove slow secular structure
t = (years_full - years_full.mean()).astype(float)
deg = 2
trend = np.polyval(np.polyfit(t, x, deg=deg), t)
x = x - trend

window = np.hanning(len(x))
xw = x * window

pad_factor = 8
nfft = pad_factor * len(xw)

fft_vals = rfft(xw, n=nfft)
freqs = rfftfreq(nfft, d=1.0)

power_obs = np.abs(fft_vals) ** 2
period = 1 / freqs

mask = (freqs > 0) & (period >= 2) & (period <= 30)
period_p = period[mask]
power_p = power_obs[mask]

n_mc = 10000
rng = np.random.default_rng(42)

power_mc = np.zeros((n_mc, power_p.size))

for i in range(n_mc):
    years_random = rng.integers(year_min, year_max + 1, size=len(years_event))

    counts_rand = pd.Series(years_random).value_counts()
    x_rand = np.array([counts_rand.get(y, 0) for y in years_full], dtype=float)

    x_rand = x_rand - x_rand.mean()
    trend_r = np.polyval(np.polyfit(t, x_rand, deg=deg), t)
    x_rand = x_rand - trend_r

    xw_rand = x_rand * window

    fft_rand = rfft(xw_rand, n=nfft)
    power_rand = np.abs(fft_rand) ** 2

    power_mc[i] = power_rand[mask]

conf_997 = np.percentile(power_mc, 99.7, axis=0)
conf_999 = np.percentile(power_mc, 99.9, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(period_p, power_p, color="black", linewidth=1.5, label="Observed")
plt.plot(period_p, conf_997, color="gray", linestyle="--", label="99.7%")
plt.plot(period_p, conf_999, color="gray", linestyle=":", label="99.9%")

plt.axvline(11, color="red", linestyle="--", label="11-year")
plt.axvline(22, color="green", linestyle="--", label="22-year")

plt.xlim(2, 30)
plt.xlabel("Period (years)")
plt.ylabel("Fourier Power")
plt.title("FFT Power Spectrum with Monte Carlo Confidence")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


## Test gfor
korean_pre1400 = korean.loc[korean["Year"] < 1400, "Year"]

years_event = korean_pre1400.values.astype(int)

values_event = np.ones_like(years_event)

min_period = 2.0
max_period = 30.0

frequency = np.linspace(1 / max_period, 1 / min_period, 30000)

ls = LombScargle(years_event, values_event, normalization="psd")

power_ls = ls.power(frequency)
period = 1 / frequency

plt.figure(figsize=(9, 5))
plt.plot(period, power_ls, color="black")

plt.axvline(8, color="blue", linestyle="--", label="8-year")
plt.axvline(11, color="red", linestyle="--", label="11-year")
plt.axvline(22, color="green", linestyle="--", label="22-year")

plt.xlim(2, 30)
plt.xlabel("Period (years)")
plt.ylabel("Power")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 3))
plt.scatter(years_event, np.ones_like(years_event), s=5)
plt.xlabel("Year")
plt.yticks([])
plt.title("korean")
plt.show()


P = 11.0
phase = (years_event % P) / P

plt.hist(phase, bins=20, color="gray", edgecolor="black")
plt.xlabel("Phase (11-year cycle)")
plt.ylabel("Counts")
plt.title("Phase Folding for korean Aurora Records (<1400)")
plt.show()