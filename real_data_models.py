import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
from astropy.timeseries import LombScargle

from helpers import * 


P_grid_min, P_grid_max, nP = 7.0, 16.0, 900
phi_grid = np.linspace(0.0, 2.0 * np.pi, 721)
b1_grid = np.linspace(0.0, 1.5, 601)

korean = pd.read_excel("data/KoreanAuroraRecords/Korean_Auroral_Full.xlsx")

years_real = korean["Year"].astype(int).values
t0_real = int(years_real.min())
t_real = (years_real - t0_real).astype(float) 
T_real = float(t_real.max())

event_times = np.asarray(t_real, dtype=float)
T = T_real

print(f"Real data: N={len(event_times)}, t0={t0_real}, T={T:.1f} years")
print("First few event times (years since t0):", event_times[:10])

rough_rate = np.log((len(event_times) + 1e-9) / (T + 1e-9))
beta0_guess = rough_rate
beta1_guess = 0.5
phi_guess = 0.5

# Scan parameters
periods_scan, logL_P_scan, P_hat = scan_period(
    event_times,
    beta0_guess,
    beta1_guess,
    phi_guess,
    T,
    P_min=P_grid_min,
    P_max=P_grid_max,
    n_P=nP,
    ll_grid=2500,
)

omega_hat = 2.0 * np.pi / P_hat

b0g = beta0_grid(event_times, T, width=3.0, n=601)
logL_b0_scan, b0_hat = scan_beta0(
    event_times, beta1_guess, omega_hat, phi_guess, T, b0g, ll_grid=2500
)

logL_b1_scan, b1_hat = scan_beta1(
    event_times, b0_hat, omega_hat, phi_guess, T, b1_grid, ll_grid=2500
)

logL_phi_scan, phi_hat = scan_phi(
    event_times, b0_hat, b1_hat, omega_hat, T, phi_grid, ll_grid=2500
)

theta_center = np.array([b0_hat, b1_hat, np.log(P_hat), np.log(phi_hat)], dtype=float)
print("Init center: [beta0, beta1, logP, logphi] =", theta_center)
print("Init center as P_hat:", np.exp(theta_center[2]))

# Run emcee
ndim = 4
nwalkers = 64

seed = 2200
rng = np.random.default_rng(seed)

p0 = theta_center + 1e-2 * rng.standard_normal(size=(nwalkers, ndim))

# wrap P
logP_min, logP_max = np.log(P_grid_min), np.log(P_grid_max)
p0[:, 2] = np.clip(p0[:, 2], logP_min + 1e-6, logP_max - 1e-6)

# wrap phi
logphi_min = np.log(1e-12)
logphi_max = np.log(np.pi - 1e-12)
p0[:, 3] = np.clip(p0[:, 3], logphi_min, logphi_max)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(event_times, T),
    kwargs={"P_min": P_grid_min, "P_max": P_grid_max, "ll_grid": 4000},
)

nburn = 50000
state = sampler.run_mcmc(p0, nburn, progress=True, rstate0=rng)
sampler.reset()

nsteps = 30000
sampler.run_mcmc(state, nsteps, progress=True, rstate0=rng)

print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# Summaries
thin = 10
flat = sampler.get_chain(discard=0, thin=thin, flat=True)

beta0_s = flat[:, 0]
beta1_s = flat[:, 1]
logP_s  = flat[:, 2]
logphi_s   = flat[:, 3]

P_s = np.exp(logP_s)
phi_s = np.exp(logphi_s)

print("Posterior (16, 50, 84 percentiles):")
print("beta0:", q16_50_84(beta0_s))
print("beta1:", q16_50_84(beta1_s))
print("P    :", q16_50_84(P_s))
print("phi  :", q16_50_84(phi_s))
    
# Trace plots
chain = sampler.get_chain()
labels = [r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"]
P_chain = np.exp(chain[:, :, 2])
phi_chain = np.exp(chain[:, :, 3])

fig, axes = plt.subplots(ndim, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
for i in range(ndim):
    ax = axes[i]
    if i == 2:
        for w in range(chain.shape[1]):
            ax.plot(P_chain[:, w], alpha=0.2, color="blue")
        ax.axhline(P_hat, ls="--", color="black", label="P_hat")
        ax.legend(frameon=False)
    elif i == 3:
        for w in range(chain.shape[1]):
            ax.plot(phi_chain[:, w], alpha=0.2, color="blue")
        ax.legend(frameon=False)
    else:
        for w in range(chain.shape[1]):
            ax.plot(chain[:, w, i], alpha=0.2, color="blue")
    ax.set_ylabel(labels[i], fontsize=12)

axes[-1].set_xlabel("Step", fontsize=12)
plt.show()

# Corner plot
nsteps, nwalkers, ndim = chain.shape
flat2 = chain.reshape(-1, ndim)

beta0_s2 = flat2[:, 0]
beta1_s2 = flat2[:, 1]
P_s2     = np.exp(flat2[:, 2])
phi_s2   = np.exp(flat2[:, 3])

samples_corner = np.column_stack([beta0_s2, beta1_s2, P_s2, phi_s2])

fig = corner.corner(
    samples_corner,
    labels=[r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"],
    truths=[b0_hat, b1_hat, P_hat, np.exp(theta_center[3])],
    color="blue",
    show_titles=True,
    title_fmt=".3f",
    title_kwargs={"fontsize": 12},
)
plt.show()


# Posterior predictive checks
t_grid = np.linspace(0.0, T, 2000)

bins = 50
counts, edges = np.histogram(event_times, bins=bins, range=(0.0, T))
bin_width = edges[1] - edges[0]
rate_hist = counts / bin_width
centers = 0.5 * (edges[:-1] + edges[1:])

plt.figure(figsize=(10, 5))
plt.step(centers, rate_hist, where="mid", lw=2, label="Empirical rate", color="black")

rng = np.random.default_rng(2200)
n_draws = 300
idx = rng.integers(0, len(beta0_s), size=n_draws)

for k in idx:
    b0 = beta0_s[k]
    b1 = beta1_s[k]
    P  = P_s[k]
    ph = phi_s[k]
    om = 2.0 * np.pi / P
    lam = np.exp(b0 + b1 * np.sin(om * t_grid + ph))
    plt.plot(t_grid, lam, alpha=0.12, lw=1, color="blue")

plt.xlabel("Time (years since t0)")
plt.ylabel("Rate")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Let's try to filter the dominant P cluster
thin = 10
flat = sampler.get_chain(discard=0, thin=thin, flat=True)

beta0_s = flat[:, 0]
beta1_s = flat[:, 1]
P_s     = np.exp(flat[:, 2])
phi_s   = np.exp(flat[:, 3])

hist, edges = np.histogram(P_s, bins=250)
k = np.argmax(hist)
P_mode = 0.5 * (edges[k] + edges[k + 1])

dP = 0.03
mask = np.abs(P_s - P_mode) < dP

print("P_mode:", P_mode, " kept fraction:", mask.mean())

beta0_m = beta0_s[mask]
beta1_m = beta1_s[mask]
P_m     = P_s[mask]
phi_m   = phi_s[mask]

print("Posterior (16, 50, 84) dominant mode:")
print("beta0:", q16_50_84(beta0_m))
print("beta1:", q16_50_84(beta1_m))
print("P    :", q16_50_84(P_m))
print("phi  :", q16_50_84(phi_m))

chain = sampler.get_chain()
P_last = np.exp(chain[-1, :, 2])
good_w = np.abs(P_last - P_mode) < dP

print("Kept walkers for trace:", good_w.sum(), "/", chain.shape[1])

beta0_chain = chain[:, good_w, 0]
beta1_chain = chain[:, good_w, 1]
P_chain     = np.exp(chain[:, good_w, 2])
phi_chain   = np.exp(chain[:, good_w, 3])

labels = [r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"]

# Trace plots for dominant P cluster
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
axes[0].plot(beta0_chain, alpha=0.2, color="blue"); axes[0].set_ylabel(labels[0], fontsize=16)
axes[1].plot(beta1_chain, alpha=0.2, color="blue"); axes[1].set_ylabel(labels[1], fontsize=16)
axes[2].plot(P_chain, alpha=0.2, color="blue")
axes[2].axhline(P_mode, ls="--", color="black", label="P_mode"); axes[2].legend(frameon=False)
axes[2].set_ylabel(labels[2], fontsize=16)
axes[3].plot(phi_chain, alpha=0.2, color="blue"); axes[3].set_ylabel(labels[3], fontsize=16)
axes[-1].set_xlabel("Step", fontsize=16)
axes[0].set_title("Trace plots", fontsize=20)
plt.show()

# Corner plot for dominant P cluster
samples_corner = np.column_stack([beta0_m, beta1_m, P_m, phi_m])
fig = corner.corner(
    samples_corner,
    labels=[r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"],
    color="blue",
    show_titles=True,
    title_fmt=".3f",
    title_kwargs={"fontsize": 16},
)
fig.suptitle("Corner plot", fontsize=20)
plt.show()

# Posterior predictive checks for dominant P cluster
t_grid = np.linspace(0.0, T, 2000)
bins = 50
counts, edges = np.histogram(event_times, bins=bins, range=(0.0, T))
bin_width = edges[1] - edges[0]
rate_hist = counts / bin_width
centers = 0.5 * (edges[:-1] + edges[1:])

plt.figure(figsize=(10, 5))
plt.step(centers, rate_hist, where="mid", lw=2, label="Empirical rate", color="black")

rng = np.random.default_rng(2200)
n_draws = 300
idx = rng.integers(0, len(beta0_m), size=n_draws)

for j in idx:
    b0 = beta0_m[j]
    b1 = beta1_m[j]
    P  = P_m[j]
    ph = phi_m[j]
    om = 2.0 * np.pi / P
    lam = np.exp(b0 + b1 * np.sin(om * t_grid + ph))
    plt.plot(t_grid, lam, alpha=0.12, lw=1, color="blue")

plt.xlabel("Time (years since t0)")
plt.ylabel("Rate")
plt.legend(loc="best")
plt.tight_layout()
plt.show()



# Multi-start MCMC over multiple P modes from scan_period
P_vals = periods_scan
L_vals = logL_P_scan

is_peak = np.r_[False, (L_vals[1:-1] > L_vals[:-2]) & (L_vals[1:-1] > L_vals[2:]), False]
peak_idx = np.where(is_peak)[0]

if peak_idx.size == 0:
    peak_idx = np.array([int(np.argmax(L_vals))])

K = 4
peak_idx = peak_idx[np.argsort(L_vals[peak_idx])[::-1]]
peak_idx = peak_idx[: min(K, peak_idx.size)]

P_seeds = P_vals[peak_idx]
print("P seeds:", P_seeds)

nburn = 50000
nsteps = 30000
thin = 10

results = []

for r, P0 in enumerate(P_seeds):
    omega0 = 2.0 * np.pi / P0

    logL_b0_scan, b0_0 = scan_beta0(
        event_times, beta1_guess, omega0, phi_guess, T, b0g, ll_grid=2500
    )
    logL_b1_scan, b1_0 = scan_beta1(
        event_times, b0_0, omega0, phi_guess, T, b1_grid, ll_grid=2500
    )
    logL_phi_scan, phi_0 = scan_phi(
        event_times, b0_0, b1_0, omega0, T, phi_grid, ll_grid=2500
    )

    if phi_0 <= 0.0:
        phi_0 = 1e-6
    if beta1_guess <= 0.0:
        beta1_guess = 0.5

    theta_center_r = np.array([b0_0, b1_0, np.log(P0), np.log(phi_0)], dtype=float)
    print(f"Run {r}: theta_center =", theta_center_r, " | P0 =", P0, " | phi0 =", phi_0)

    rng_r = np.random.default_rng(seed + 1000 * r)

    p0 = theta_center_r + 1e-2 * rng_r.standard_normal(size=(nwalkers, ndim))

    p0[:, 2] = np.clip(p0[:, 2], logP_min + 1e-6, logP_max - 1e-6)
    p0[:, 3] = np.clip(p0[:, 3], logphi_min, logphi_max)

    sampler_r = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(event_times, T),
        kwargs={"P_min": P_grid_min, "P_max": P_grid_max, "ll_grid": 4000},
    )

    state_r = sampler_r.run_mcmc(p0, nburn, progress=True, rstate0=rng_r)
    sampler_r.reset()
    sampler_r.run_mcmc(state_r, nsteps, progress=True, rstate0=rng_r)

    chain_r = sampler_r.get_chain()
    logp_r = sampler_r.get_log_prob()

    mean_logp = np.mean(logp_r)
    max_logp = np.max(logp_r)
    acc = float(np.mean(sampler_r.acceptance_fraction))

    flat_r = sampler_r.get_chain(discard=0, thin=thin, flat=True)
    P_r = np.exp(flat_r[:, 2])
    phi_r = np.exp(flat_r[:, 3])

    results.append(
        {
            "run": r,
            "P_seed": float(P0),
            "theta_center": theta_center_r,
            "mean_logp": float(mean_logp),
            "max_logp": float(max_logp),
            "acc": acc,
            "P_med": float(np.percentile(P_r, 50)),
            "P_16": float(np.percentile(P_r, 16)),
            "P_84": float(np.percentile(P_r, 84)),
            "phi_med": float(np.percentile(phi_r, 50)),
            "phi_16": float(np.percentile(phi_r, 16)),
            "phi_84": float(np.percentile(phi_r, 84)),
            "sampler": sampler_r,
        }
    )
results_sorted = sorted(results, key=lambda d: d["mean_logp"], reverse=True)

print("Multi-start results")
for d in results_sorted:
    print(
        f"run={d['run']}  P_seed={d['P_seed']:.6f}  "
        f"mean_logp={d['mean_logp']:.2f}  max_logp={d['max_logp']:.2f}  acc={d['acc']:.3f}  "
        f"P={d['P_med']:.6f} [{d['P_16']:.6f},{d['P_84']:.6f}]  "
        f"phi={d['phi_med']:.3f} [{d['phi_16']:.3f},{d['phi_84']:.3f}]"
    )

# Pick the best run
best = results_sorted[0]

# Get the best sampler and its chain
sampler = best["sampler"]

chain = sampler.get_chain()
P_last = np.exp(chain[-1, :, 2])

thin = 10
flat_best = sampler.get_chain(discard=0, thin=thin, flat=True)

beta0_s = flat_best[:, 0]
beta1_s = flat_best[:, 1]
P_s     = np.exp(flat_best[:, 2]) 
phi_s   = np.exp(flat_best[:, 3])

# Find the dominant P cluster and filter to it
hist, edges = np.histogram(P_s, bins=250)
k = np.argmax(hist)
P_mode = 0.5 * (edges[k] + edges[k + 1])

dP = 0.015
mask = np.abs(P_s - P_mode) < dP

print("Dominant P cluster:")
print("P_mode =", P_mode, "| kept fraction =", mask.mean())

beta0_m = beta0_s[mask]
beta1_m = beta1_s[mask]
P_m     = P_s[mask]
phi_m   = phi_s[mask]

print("Posterior (16, 50, 84) for dominant P cluster:")
print("beta0:", q16_50_84(beta0_m))
print("beta1:", q16_50_84(beta1_m))
print("P    :", q16_50_84(P_m))
print("phi  :", q16_50_84(phi_m))

# Trace plots
chain = sampler.get_chain()
P_last = np.exp(chain[-1, :, 2])
good_w = np.abs(P_last - P_mode) < dP

print("Trace plot filtering:")
print("kept walkers =", good_w.sum(), "/", chain.shape[1])

beta0_chain = chain[:, good_w, 0]
beta1_chain = chain[:, good_w, 1]
P_chain     = np.exp(chain[:, good_w, 2])
phi_chain   = np.exp(chain[:, good_w, 3])

labels = [r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"]

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

axes[0].plot(beta0_chain, alpha=0.2, color="blue")
axes[0].set_ylabel(labels[0], fontsize=12)

axes[1].plot(beta1_chain, alpha=0.2, color="blue")
axes[1].set_ylabel(labels[1], fontsize=12)

axes[2].plot(P_chain, alpha=0.2, color="blue")
axes[2].axhline(P_mode, ls="--", color="black", label="P_mode")
axes[2].legend(frameon=False)
axes[2].set_ylabel(labels[2], fontsize=12)

axes[3].plot(phi_chain, alpha=0.2, color="blue")
axes[3].set_ylabel(labels[3], fontsize=12)

axes[-1].set_xlabel("Step", fontsize=12)
plt.show()

# Filtered Corner plot 
samples_corner = np.column_stack([beta0_m, beta1_m, P_m, phi_m])

fig = corner.corner(
    samples_corner,
    labels=[r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"],
    color="blue",
    show_titles=True,
    title_fmt=".3f",
    title_kwargs={"fontsize": 12},
)
plt.show()

# Posterior predictive checks filtered
t_grid = np.linspace(0.0, T, 2000)

bins = 50
counts, edges = np.histogram(event_times, bins=bins, range=(0.0, T))
bin_width = edges[1] - edges[0]
rate_hist = counts / bin_width
centers = 0.5 * (edges[:-1] + edges[1:])

plt.figure(figsize=(10, 5))
plt.step(centers, rate_hist, where="mid", lw=2, label="Empirical rate", color="black")

rng_ppc = np.random.default_rng(2200)
n_draws = 300
idx = rng_ppc.integers(0, len(beta0_m), size=n_draws)

for j in idx:
    b0 = beta0_m[j]
    b1 = beta1_m[j]
    P  = P_m[j]
    ph = phi_m[j]
    om = 2.0 * np.pi / P
    lam = np.exp(b0 + b1 * np.sin(om * t_grid + ph))
    plt.plot(t_grid, lam, alpha=0.12, lw=1, color="blue")

plt.xlabel("Time (years since t0)")
plt.ylabel("Rate")
plt.legend(loc="best")
plt.tight_layout()
plt.show()