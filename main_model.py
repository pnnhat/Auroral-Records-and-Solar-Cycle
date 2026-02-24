import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import trapezoid
import emcee
import corner
import pandas as pd
from astropy.timeseries import LombScargle

T = 354.0
P_true = 11.0
omega_true = 2 * np.pi / P_true
beta1_true = 0.6
phi_true = np.pi / 4
N_target = 200

def beta0(beta1, omega, phi, T, N_target, n_grid=20000):
    t_grid = np.linspace(0.0, T, n_grid)
    base = np.exp(beta1 * np.sin(omega * t_grid + phi))
    integral_base = trapezoid(base, t_grid)
    beta0 = np.log(N_target / integral_base)
    return beta0


beta0_true = beta0(beta1_true, omega_true, phi_true, T, N_target)

theta_true = {
    "beta0": beta0_true,
    "beta1": beta1_true,
    "phi": phi_true,
    "T": T,
    "P_true": P_true,
    "omega_true": omega_true,
    "N_target": N_target,
}

print("theta_true:", theta_true)

seed = 2200
rng = np.random.default_rng(seed)


def lambda_func(t, beta0, beta1, omega, phi):
    t = np.asarray(t, dtype=float)
    return np.exp(beta0 + beta1 * np.sin(omega * t + phi))


def cdf_inversion(beta0, beta1, omega, phi, T, n_grid=20000, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed)

    t_grid = np.linspace(0.0, T, n_grid)
    lam_grid = lambda_func(t_grid, beta0, beta1, omega, phi)

    dt = np.diff(t_grid)
    incr = 0.5 * (lam_grid[:-1] + lam_grid[1:]) * dt
    Lambda_grid = np.concatenate([[0.0], np.cumsum(incr)])
    Lambda_T = Lambda_grid[-1]

    N = rng.poisson(Lambda_T)
    u = rng.random(N) * Lambda_T

    inv_Lambda = interp1d(
        Lambda_grid,
        t_grid,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, T),
        assume_sorted=True,
    )

    t_events = inv_Lambda(u)
    t_events.sort()
    return t_events, Lambda_T


t_events, Lambda_T = cdf_inversion(
    beta0=theta_true["beta0"],
    beta1=theta_true["beta1"],
    omega=theta_true["omega_true"],
    phi=theta_true["phi"],
    T=theta_true["T"],
    n_grid=20000,
    rng=rng,
)

print("Simulated N:", len(t_events))
print("First few events:", t_events[:10])
print("Last event time:", t_events[-1])


def log_likelihood_params(beta0, beta1, omega, phi, event_times, T, n_grid=4000):
    if len(event_times) == 0:
        # logL = - integral λ
        t_grid = np.linspace(0.0, T, n_grid)
        lam_grid = np.exp(beta0 + beta1 * np.sin(omega * t_grid + phi))
        return -trapezoid(lam_grid, t_grid)

    term_events = np.sum(beta0 + beta1 * np.sin(omega * event_times + phi))
    t_grid = np.linspace(0.0, T, n_grid)
    lam_grid = np.exp(beta0 + beta1 * np.sin(omega * t_grid + phi))
    integral = trapezoid(lam_grid, t_grid)
    return term_events - integral


def scan_period(
    prefix, beta0, beta1, phi, T, P_min=7.0, P_max=16.0, n_P=900, ll_grid=2500
):
    periods = np.linspace(P_min, P_max, n_P)
    logLs = np.array(
        [
            log_likelihood_params(
                beta0, beta1, 2 * np.pi / P, phi, prefix, T, n_grid=ll_grid
            )
            for P in periods
        ],
        dtype=float,
    )
    i = int(np.argmax(logLs))
    return periods, logLs, periods[i]


def scan_beta0(prefix, beta1, omega, phi, T, b0_grid, ll_grid=2500):
    logLs = np.array(
        [
            log_likelihood_params(b0, beta1, omega, phi, prefix, T, n_grid=ll_grid)
            for b0 in b0_grid
        ],
        dtype=float,
    )
    i = int(np.argmax(logLs))
    return logLs, b0_grid[i]


def scan_beta1(prefix, beta0, omega, phi, T, b1_grid, ll_grid=2500):
    logLs = np.array(
        [
            log_likelihood_params(beta0, b1, omega, phi, prefix, T, n_grid=ll_grid)
            for b1 in b1_grid
        ],
        dtype=float,
    )
    i = int(np.argmax(logLs))
    return logLs, b1_grid[i]


def scan_phi(prefix, beta0, beta1, omega, T, phi_grid, ll_grid=2500):
    logLs = np.array(
        [
            log_likelihood_params(beta0, beta1, omega, ph, prefix, T, n_grid=ll_grid)
            for ph in phi_grid
        ],
        dtype=float,
    )
    i = int(np.argmax(logLs))
    return logLs, phi_grid[i]


P_grid_min, P_grid_max, nP = 7.0, 16.0, 900
phi_grid = np.linspace(-np.pi, np.pi, 721)
b1_grid = np.linspace(-1.5, 1.5, 601)


def beta0_grid(prefix, T, width=3.0, n=601):
    n_evt = len(prefix)
    rough = np.log((n_evt + 1e-9) / (T + 1e-9))
    return np.linspace(rough - width, rough + width, n)


hist_counts, bin_edges = np.histogram(t_events, bins=200, range=(0.0, T))
hist_ymax = max(1, int(1.1 * hist_counts.max()))

fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
ax_hist = axes[0, 0]
ax_P = axes[0, 1]
ax_b0 = axes[1, 0]
ax_b1 = axes[1, 1]
ax_phi = axes[2, 0]
axes[2, 1].axis("off")


def init():
    for ax in [ax_hist, ax_P, ax_b0, ax_b1, ax_phi]:
        ax.clear()
    axes[2, 1].axis("off")
    return []


def update(frame_idx):
    j = frame_idx + 1
    prefix = t_events[:j]
    periods, logL_P, P_hat = scan_period(
        prefix,
        beta0_true,
        beta1_true,
        phi_true,
        T,
        P_min=P_grid_min,
        P_max=P_grid_max,
        n_P=nP,
        ll_grid=2500,
    )

    b0g = beta0_grid(prefix, T, width=3.0, n=601)
    logL_b0, b0_hat = scan_beta0(
        prefix, beta1_true, omega_true, phi_true, T, b0g, ll_grid=2500
    )
    logL_b1, b1_hat = scan_beta1(
        prefix, beta0_true, omega_true, phi_true, T, b1_grid, ll_grid=2500
    )
    logL_phi, phi_hat = scan_phi(
        prefix, beta0_true, beta1_true, omega_true, T, phi_grid, ll_grid=2500
    )

    # Histogram
    ax_hist.clear()
    ax_hist.hist(prefix, bins=bin_edges, density=False, alpha=0.6, color="black")
    ax_hist.set_xlim(0.0, T)
    ax_hist.set_ylim(0, hist_ymax)
    ax_hist.set_xlabel("Event times")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Num events: {j}")

    # logL vs Period
    ax_P.clear()
    ax_P.plot(periods, logL_P, lw=2, color="black")
    ax_P.axvline(P_true, ls="--", label="True P", color="red")
    ax_P.axvline(P_hat, ls=":", label=f"P̂={P_hat:.2f}")
    ax_P.set_xlim(P_grid_min, P_grid_max)
    ax_P.set_xlabel("Period")
    ax_P.set_ylabel("logL")
    ax_P.set_title("Period")
    ax_P.legend(loc="best")

    # logL vs beta0
    ax_b0.clear()
    ax_b0.plot(b0g, logL_b0, lw=2, color="black")
    ax_b0.axvline(beta0_true, ls="--", label="True β0", color="red")
    ax_b0.axvline(b0_hat, ls=":", label=f"β0̂={b0_hat:.2f}")
    ax_b0.set_xlabel("β0")
    ax_b0.set_ylabel("logL")
    ax_b0.set_title("β0")
    ax_b0.legend(loc="best")

    # logL vs beta1
    ax_b1.clear()
    ax_b1.plot(b1_grid, logL_b1, lw=2, color="black")
    ax_b1.axvline(beta1_true, ls="--", label="True β1", color="red")
    ax_b1.axvline(b1_hat, ls=":", label=f"β1̂={b1_hat:.2f}")
    ax_b1.set_xlabel("β1")
    ax_b1.set_ylabel("logL")
    ax_b1.set_title("β1")
    ax_b1.legend(loc="best")

    # logL vs phi
    ax_phi.clear()
    ax_phi.plot(phi_grid, logL_phi, lw=2, color="black")
    ax_phi.axvline(phi_true, ls="--", label="True φ", color="red")
    ax_phi.axvline(phi_hat, ls=":", label=f"φ̂={phi_hat:.2f}")
    ax_phi.set_xlabel("φ")
    ax_phi.set_ylabel("logL")
    ax_phi.set_title("φ")
    ax_phi.legend(loc="best")
    return []


anim = FuncAnimation(
    fig,
    update,
    frames=len(t_events),
    init_func=init,
    interval=80,
    blit=False,
)

anim.save("animation.gif", writer=PillowWriter(fps=10))
plt.show()

#! What's the minimum number of events needed to get a good estimate of the period?
# Sequentially adding events, add 1 prefix each time
P_min, P_max, nP = 7.0, 16.0, 900
periods = np.linspace(P_min, P_max, nP)
omegas = 2 * np.pi / periods

ll_grid = 3000
t_grid_ll = np.linspace(0.0, T, ll_grid)

sin_grid = np.sin(omegas[:, None] * t_grid_ll[None, :] + phi_true)
lam_grid = np.exp(beta0_true + beta1_true * sin_grid)
integrals = trapezoid(lam_grid, t_grid_ll, axis=1)

err = 0.5  # years


def min_events(t_events):
    N = len(t_events)
    if N == 0:
        return np.nan

    sin_events = np.sin(omegas[:, None] * t_events[None, :] + phi_true)
    cumsum_sin = np.cumsum(sin_events, axis=1)

    P_hat = np.empty(N, dtype=float)
    for j in range(1, N + 1):
        sum_sin = cumsum_sin[:, j - 1]
        logL = beta0_true * j + beta1_true * sum_sin - integrals
        P_hat[j - 1] = periods[int(np.argmax(logL))]

    within = np.abs(P_hat - P_true) <= err
    suffix_all_true = np.logical_and.accumulate(within[::-1])[::-1]
    idx = np.where(suffix_all_true)[0]
    if len(idx) == 0:
        return np.nan
    return float(idx[0] + 1)


K = 300
jstars = np.empty(K, dtype=float)

for k in range(K):
    t_ev, _ = cdf_inversion(
        beta0_true, beta1_true, omega_true, phi_true, T, n_grid=20000, rng=rng
    )
    jstars[k] = min_events(t_ev)

valid = np.isfinite(jstars)
jv = jstars[valid]

N_90 = int(np.quantile(jv, 0.90, method="higher"))
N_95 = int(np.quantile(jv, 0.95, method="higher"))

print(f"±{err} years around P_true={P_true}")
print(f"Minimum events for 90% confidence: {N_90} events")
print(f"Minimum events for 95% confidence: {N_95} events")

# Survival curve
N_max = int(np.nanmax(jv))
Ns = np.arange(1, N_max + 1)

success_prob = np.array([np.mean(jv <= N) for N in Ns])
plt.figure(figsize=(7, 4))

plt.plot(Ns, success_prob, lw=2, color="black")
plt.axhline(0.90, ls=":", color="black", label="90%")
plt.axhline(0.95, ls=":", color="gray", label="95%")

plt.axvline(N_90, ls="--", color="red", label=f"N = {N_90}")
plt.axvline(N_95, ls="--", color="blue", label=f"N = {N_95}")

plt.xlabel("Number of events")
plt.xlim(0, N_max)
plt.ylim(0, 1.02)
plt.legend()
plt.tight_layout()
plt.show()


# Random across full T
seed = 2200
rng = np.random.default_rng(seed)
perm = rng.permutation(len(t_events))  # random subset every run
frame_js = np.arange(len(t_events))
err = 0.5

fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
ax_hist, ax_P, ax_err = axes

err_trace = []
n_trace = []


def init():
    ax_hist.clear()
    ax_P.clear()
    ax_err.clear()
    err_trace.clear()
    n_trace.clear()
    return []


def update(frame_idx):
    j = int(frame_idx) + 1

    subset = t_events[perm[:j]]
    subset_sorted = np.sort(subset)

    periods, logL_P, P_hat = scan_period(
        subset_sorted,
        beta0_true,
        beta1_true,
        phi_true,
        T,
        P_min=P_grid_min,
        P_max=P_grid_max,
        n_P=nP,
        ll_grid=2500,
    )

    err_trace.append(float(np.abs(P_hat - P_true)))
    n_trace.append(j)

    ax_hist.clear()
    ax_hist.hist(subset_sorted, bins=bin_edges, density=False, alpha=0.6, color="black")
    ax_hist.set_xlim(0.0, T)
    ax_hist.set_ylim(0, hist_ymax)
    ax_hist.set_xlabel("Event times")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Num events: {j}")

    ax_P.clear()
    ax_P.plot(periods, logL_P, lw=2, color="black")
    ax_P.axvline(P_true, ls="--", label="True P", color="red")
    ax_P.axvline(P_hat, ls=":", label=f"P̂={P_hat:.2f}", color="blue")
    ax_P.set_xlim(P_grid_min, P_grid_max)
    ax_P.set_xlabel("Period")
    ax_P.set_ylabel("logL")
    ax_P.set_title("Period")
    ax_P.legend(loc="best")

    # Error vs N
    ax_err.clear()
    ax_err.plot(n_trace, err_trace, lw=2, color="black")
    ax_err.axhline(err, ls="--", color="red", label=f"Error = {err}")
    ax_err.set_xlim(0, len(t_events))
    ax_err.set_ylim(0, max(err * 2, max(err_trace) * 1.05))
    ax_err.set_xlabel("Number of events")
    ax_err.set_ylabel(r"$|\hat P - P_{\mathrm{true}}|$")
    ax_err.set_title("Error vs Num Events")
    ax_err.legend(loc="best")

    return []


anim = FuncAnimation(
    fig,
    update,
    frames=len(frame_js),
    init_func=init,
    interval=80,
    blit=False,
)

anim.save("random_animation.gif", writer=PillowWriter(fps=10))
plt.show()


# Random, get minimum events
seed = 2200
rng = np.random.default_rng(seed)


def min_events_random(t_events, rng):
    N = len(t_events)
    if N == 0:
        return np.nan
    perm = rng.permutation(N)
    sin_all = np.sin(omegas[:, None] * t_events[None, :] + phi_true)

    sin_reordered = sin_all[:, perm]

    cumsum_sin = np.cumsum(sin_reordered, axis=1)

    P_hat = np.empty(N, dtype=float)
    for j in range(1, N + 1):
        sum_sin = cumsum_sin[:, j - 1]
        logL = beta0_true * j + beta1_true * sum_sin - integrals
        P_hat[j - 1] = periods[int(np.argmax(logL))]

    within = np.abs(P_hat - P_true) <= err

    suffix_all_true = np.logical_and.accumulate(within[::-1])[::-1]
    idx = np.where(suffix_all_true)[0]
    if len(idx) == 0:
        return np.nan

    return float(idx[0] + 1)


K = 300
jstars = np.empty(K, dtype=float)

for k in range(K):
    t_ev, _ = cdf_inversion(
        beta0_true, beta1_true, omega_true, phi_true, T, n_grid=20000, rng=rng
    )
    jstars[k] = min_events_random(t_ev, rng=rng)

valid = np.isfinite(jstars)
jv = jstars[valid]

N_90 = int(np.quantile(jv, 0.90, method="higher"))
N_95 = int(np.quantile(jv, 0.95, method="higher"))

print(f"Random set across full T, ±{err} years around P_true={P_true}")
print(f"Minimum events for 90% confidence: {N_90} events")
print(f"Minimum events for 95% confidence: {N_95} events")

# Success curve (empirical CDF)
N_max = int(np.nanmax(jv))
Ns = np.arange(1, N_max + 1)

success_prob = np.array([np.mean(jv <= N) for N in Ns])

plt.figure(figsize=(7, 4))
plt.plot(Ns, success_prob, lw=2, color="black")
plt.axhline(0.90, ls=":", color="black", label="90%")
plt.axhline(0.95, ls=":", color="gray", label="95%")
plt.axvline(N_90, ls="--", color="red", label=f"N90 = {N_90}")
plt.axvline(N_95, ls="--", color="blue", label=f"N95 = {N_95}")
plt.xlabel("Number of events")
plt.ylabel("Probability of accurate P̂")
plt.xlim(0, N_max)
plt.ylim(0, 1.02)
plt.legend()
plt.tight_layout()
plt.show()


# Random across full T, multiple seeds
seeds = [2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209]
K = 300
results = []

for seed in seeds:
    rng = np.random.default_rng(seed)

    jstars = np.empty(K, dtype=float)

    for k in range(K):
        t_ev, _ = cdf_inversion(
            beta0_true, beta1_true, omega_true, phi_true, T, n_grid=20000, rng=rng
        )
        jstars[k] = min_events_random(t_ev, rng=rng)

    valid = np.isfinite(jstars)
    jv = jstars[valid]

    N_90 = int(np.quantile(jv, 0.90, method="higher"))
    N_95 = int(np.quantile(jv, 0.95, method="higher"))

    results.append(
        {
            "seed": seed,
            "N90": N_90,
            "N95": N_95,
        }
    )

print(f"Random set across full T, ±{err} years around P_true={P_true}")
for r in results:
    print(f"seed={r['seed']} | N90={r['N90']} | N95={r['N95']}")

N90s = np.array([r["N90"] for r in results], dtype=int)
N95s = np.array([r["N95"] for r in results], dtype=int)

print("Range across seeds:")
print(f"Minimum events for 90% confidence:: min={N90s.min()}, max={N90s.max()}")
print(f"Minimum evetns for 95% confidence: min={N95s.min()}, max={N95s.max()}")


#! What does it look like if the function is changed to a Fourier Series?
# 2 harmonics
def lambda_func_2harm(t, beta0, omega, a1, b1, a2, b2):
    t = np.asarray(t, dtype=float)
    return np.exp(
        beta0
        + a1 * np.sin(omega * t)
        + b1 * np.cos(omega * t)
        + a2 * np.sin(2 * omega * t)
        + b2 * np.cos(2 * omega * t)
    )


def beta0_2harm(a1, b1, a2, b2, omega, T, N_target, n_grid=20000):
    t = np.linspace(0.0, T, n_grid)
    base = np.exp(
        a1 * np.cos(omega * t)
        + b1 * np.sin(omega * t)
        + a2 * np.cos(2 * omega * t)
        + b2 * np.sin(2 * omega * t)
    )
    return np.log(N_target / trapezoid(base, t))


# 1 harmonic
def lambda_func_1harm(t, beta0, omega, a1, b1):
    t = np.asarray(t, dtype=float)
    return np.exp(beta0 + a1 * np.sin(omega * t) + b1 * np.cos(omega * t))


def beta0_1harm(a1, b1, omega, T, N_target, n_grid=20000):
    t = np.linspace(0.0, T, n_grid)
    base = np.exp(a1 * np.cos(omega * t) + b1 * np.sin(omega * t))
    return np.log(N_target / trapezoid(base, t))


def log_likelihood_1harm(
    event_times, T, N_target, omega, a1_grid, b1_grid, ll_grid=4000
):
    if len(event_times) == 0:
        return -np.inf, (np.nan, np.nan, np.nan)

    # compute cos and sin at event times
    s1 = np.sin(omega * event_times)
    c1 = np.cos(omega * event_times)

    # integration grid
    t_grid = np.linspace(0.0, T, ll_grid)
    c1g = np.cos(omega * t_grid)
    s1g = np.sin(omega * t_grid)

    bestL = -np.inf
    best = None

    for a1 in a1_grid:
        for b1 in b1_grid:
            b0 = beta0_1harm(a1, b1, omega, T, N_target, n_grid=ll_grid)

            term = np.sum(b0 + a1 * c1 + b1 * s1)
            lam_grid = np.exp(b0 + a1 * c1g + b1 * s1g)
            integral = trapezoid(lam_grid, t_grid)
            L = term - integral

            if L > bestL:
                bestL = L
                best = (b0, a1, b1)

    return bestL, best


def log_likelihood_2harm(
    event_times, T, N_target, omega, a1_grid, b1_grid, a2_grid, b2_grid, ll_grid=4000
):
    if len(event_times) == 0:
        return -np.inf, (np.nan, np.nan, np.nan, np.nan, np.nan)

    s1 = np.sin(omega * event_times)
    c1 = np.cos(omega * event_times)
    s2 = np.sin(2 * omega * event_times)
    c2 = np.cos(2 * omega * event_times)

    t_grid = np.linspace(0.0, T, ll_grid)
    s1g = np.sin(omega * t_grid)
    c1g = np.cos(omega * t_grid)
    s2g = np.sin(2 * omega * t_grid)
    c2g = np.cos(2 * omega * t_grid)

    bestL = -np.inf
    best = None

    for a1 in a1_grid:
        for b1 in b1_grid:
            for a2 in a2_grid:
                for b2 in b2_grid:
                    b0 = beta0_2harm(a1, b1, a2, b2, omega, T, N_target, n_grid=ll_grid)

                    term = np.sum(b0 + a1 * c1 + b1 * s1 + a2 * c2 + b2 * s2)
                    lam_grid = np.exp(b0 + a1 * c1g + b1 * s1g + a2 * c2g + b2 * s2g)
                    integral = trapezoid(lam_grid, t_grid)
                    L = term - integral

                    if L > bestL:
                        bestL = L
                        best = (b0, a1, b1, a2, b2)

    return bestL, best


# What if the true is 2 harmonics, fit 1 harmonic?
# get amplitudes and phases by randomizing
A1, A2 = 0.6, 0.3
rng = np.random.default_rng(2200)

phi1 = rng.uniform(-np.pi, np.pi)
phi2 = rng.uniform(-np.pi, np.pi)

a1_true = A1 * np.cos(phi1)
b1_true = A1 * np.sin(phi1)
a2_true = A2 * np.cos(phi2)
b2_true = A2 * np.sin(phi2)

omega_true = 2 * np.pi / P_true
beta0_true_2 = beta0_2harm(a1_true, b1_true, a2_true, b2_true, omega_true, T, N_target)

a1_grid = np.linspace(-1.2, 1.2, 121)
b1_grid = np.linspace(-1.2, 1.2, 121)

L1_at_trueP, (b0_1hat, a1_hat, b1_hat) = log_likelihood_1harm(
    t_events, T, N_target, omega_true, a1_grid, b1_grid, ll_grid=4000
)

t_plot = np.linspace(0.0, T, 3000)
lam_true = lambda_func_2harm(
    t_plot, beta0_true_2, omega_true, a1_true, b1_true, a2_true, b2_true
)
lam_fit1 = lambda_func_1harm(t_plot, b0_1hat, omega_true, a1_hat, b1_hat)

plt.figure(figsize=(10, 4))
plt.plot(t_plot, lam_true, lw=2, label="True intensity (2 harmonics)", color="black")
plt.plot(t_plot, lam_fit1, lw=2, ls="--", label="Best 1-harm fit", color="C1")
plt.xlabel("t")
plt.ylabel("λ(t)")
plt.legend()
plt.tight_layout()
plt.show()

print("Best 1-harm fit at P_true = 11.0:")
print(
    f"  beta0={b0_1hat:.4f}, a1={a1_hat:.4f}, b1={b1_hat:.4f}, logL={L1_at_trueP:.2f}"
)
print("\nTrue 2-harm parameters:")
print(
    f""
    f"  beta0={beta0_true_2:.4f}, a1={a1_true:.4f}, b1={b1_true:.4f}, "
    f"a2={a2_true:.4f}, b2={b2_true:.4f}"
)

# What if the true is 1 harmonic, fit 2 harmonics?
A1 = 0.7
rng = np.random.default_rng(2200)
phi1 = rng.uniform(-np.pi, np.pi)

# again
a1_true = A1 * np.cos(phi1)
b1_true = A1 * np.sin(phi1)
omega_true = 2 * np.pi / P_true

beta0_true_1 = beta0_1harm(a1_true, b1_true, omega_true, T, N_target)
a1_grid = np.linspace(-1.2, 1.2, 121)
b1_grid = np.linspace(-1.2, 1.2, 121)
a2_grid = np.linspace(-0.8, 0.8, 81)
b2_grid = np.linspace(-0.8, 0.8, 81)

L2_at_trueP, (b0_2hat, a1_hat, b1_hat, a2_hat, b2_hat) = log_likelihood_2harm(
    t_events, T, N_target, omega_true, a1_grid, b1_grid, a2_grid, b2_grid, ll_grid=4000
)

t_plot = np.linspace(0.0, T, 3000)
lam_true = lambda_func_1harm(t_plot, beta0_true_1, omega_true, a1_true, b1_true)
lam_fit2 = lambda_func_2harm(
    t_plot, b0_2hat, omega_true, a1_hat, b1_hat, a2_hat, b2_hat
)

plt.figure(figsize=(10, 4))
plt.plot(t_plot, lam_true, lw=2, label="True intensity (1 harmonic)", color="black")
plt.plot(t_plot, lam_fit2, lw=2, ls="--", label="Best 2-harm fit", color="C1")
plt.xlabel("t")
plt.ylabel("λ(t)")
plt.legend()
plt.tight_layout()
plt.show()

print("Best 2-harm fit at P_true = 11.0:")
print(
    f"  beta0={b0_2hat:.4f}, a1={a1_hat:.4f}, b1={b1_hat:.4f}, "
    f"a2={a2_hat:.4f}, b2={b2_hat:.4f}, logL={L2_at_trueP:.2f}"
)

print("\nTrue 1-harm parameters:")
print(f"  beta0={beta0_true_1:.4f}, a1={a1_true:.4f}, b1={b1_true:.4f}")


def cdf_inversion_harm(beta0, omega, T, lam_on_grid, n_grid=20000, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed)

    t_grid = np.linspace(0.0, T, n_grid)
    lam_grid = lam_on_grid(t_grid, beta0, omega)

    dt = np.diff(t_grid)
    incr = 0.5 * (lam_grid[:-1] + lam_grid[1:]) * dt
    Lambda_grid = np.concatenate([[0.0], np.cumsum(incr)])
    Lambda_T = Lambda_grid[-1]

    N = rng.poisson(Lambda_T)
    u = rng.random(N) * Lambda_T

    inv_Lambda = interp1d(
        Lambda_grid,
        t_grid,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, T),
        assume_sorted=True,
    )
    t_events = inv_Lambda(u)
    t_events.sort()
    return t_events, Lambda_T

# Get the minimum number of events for Fourier models
rng0 = np.random.default_rng(2200)

# TRUE K=1
A1 = 0.6
phi1 = rng0.uniform(-np.pi, np.pi)
a1_1, b1_1 = A1 * np.cos(phi1), A1 * np.sin(phi1)
omega_true = 2 * np.pi / P_true
beta0_true_1 = beta0_1harm(a1_1, b1_1, omega_true, T, N_target)


def lam1(t, beta0, omega):
    return lambda_func_1harm(t, beta0, omega, a1_1, b1_1)


# TRUE K=2
A2 = 0.3
phi2 = rng0.uniform(-np.pi, np.pi)
a1_2, b1_2 = a1_1, b1_1
a2_2, b2_2 = A2 * np.cos(phi2), A2 * np.sin(phi2)
beta0_true_2 = beta0_2harm(a1_2, b1_2, a2_2, b2_2, omega_true, T, N_target)


def lam2(t, beta0, omega):
    return lambda_func_2harm(t, beta0, omega, a1_2, b1_2, a2_2, b2_2)


# Get integrals for K=1
lam_grid_1 = np.exp(
    beta0_true_1
    + a1_1 * np.sin(omegas[:, None] * t_grid_ll[None, :])
    + b1_1 * np.cos(omegas[:, None] * t_grid_ll[None, :])
)
integrals_1 = trapezoid(lam_grid_1, t_grid_ll, axis=1)

# get integrals for K=2
sin1 = np.sin(omegas[:, None] * t_grid_ll[None, :])
cos1 = np.cos(omegas[:, None] * t_grid_ll[None, :])
sin2 = np.sin(2 * omegas[:, None] * t_grid_ll[None, :])
cos2 = np.cos(2 * omegas[:, None] * t_grid_ll[None, :])
lam_grid_2 = np.exp(
    beta0_true_2 + a1_2 * sin1 + b1_2 * cos1 + a2_2 * sin2 + b2_2 * cos2
)
integrals_2 = trapezoid(lam_grid_2, t_grid_ll, axis=1)


def min_events_random_fourier(
    t_events, rng, beta0_true, integrals, a1, b1, a2=None, b2=None
):
    N = len(t_events)
    if N == 0:
        return np.nan

    perm = rng.permutation(N)

    t = t_events
    s1 = np.sin(omegas[:, None] * t[None, :])
    c1 = np.cos(omegas[:, None] * t[None, :])
    eta_all = a1 * s1 + b1 * c1

    if (a2 is not None) and (b2 is not None):
        s2 = np.sin(2 * omegas[:, None] * t[None, :])
        c2 = np.cos(2 * omegas[:, None] * t[None, :])
        eta_all = eta_all + a2 * s2 + b2 * c2

    eta_reordered = eta_all[:, perm]
    cumsum_eta = np.cumsum(eta_reordered, axis=1)

    P_hat = np.empty(N, dtype=float)
    for j in range(1, N + 1):
        sum_eta = cumsum_eta[:, j - 1]
        logL = beta0_true * j + sum_eta - integrals
        P_hat[j - 1] = periods[int(np.argmax(logL))]

    within = np.abs(P_hat - P_true) <= err
    suffix_all_true = np.logical_and.accumulate(within[::-1])[::-1]
    idx = np.where(suffix_all_true)[0]
    if len(idx) == 0:
        return np.nan
    return float(idx[0] + 1)


Kmc = 300


def run_mc_for_model(
    seed_base, beta0_true, integrals, a1, b1, a2=None, b2=None, lam_on_grid=None
):
    rng = np.random.default_rng(seed_base)
    jstars = np.empty(Kmc, dtype=float)

    for k in range(Kmc):
        t_ev, _ = cdf_inversion_harm(
            beta0_true, omega_true, T, lam_on_grid=lam_on_grid, n_grid=20000, rng=rng
        )
        jstars[k] = min_events_random_fourier(
            t_ev, rng, beta0_true, integrals, a1, b1, a2=a2, b2=b2
        )

    jv = jstars[np.isfinite(jstars)]
    N90 = int(np.quantile(jv, 0.90, method="higher"))
    N95 = int(np.quantile(jv, 0.95, method="higher"))
    return jv, N90, N95


# True K=1, fit K=1
jv1, N90_1, N95_1 = run_mc_for_model(
    seed_base=3100,
    beta0_true=beta0_true_1,
    integrals=integrals_1,
    a1=a1_1,
    b1=b1_1,
    a2=None,
    b2=None,
    lam_on_grid=lam1,
)

# True K=2, fit K=2
jv2, N90_2, N95_2 = run_mc_for_model(
    seed_base=4100,
    beta0_true=beta0_true_2,
    integrals=integrals_2,
    a1=a1_2,
    b1=b1_2,
    a2=a2_2,
    b2=b2_2,
    lam_on_grid=lam2,
)

print(f"Random set across full T, ±{err} years around P_true={P_true}")
print(f"K=1 (true & fit): N90={N90_1}  N95={N95_1}")
print(f"K=2 (true & fit): N90={N90_2}  N95={N95_2}")

def success_curve(jv):
    N_max = int(np.nanmax(jv))
    Ns = np.arange(1, N_max + 1)
    success_prob = np.array([np.mean(jv <= N) for N in Ns])
    return Ns, success_prob


Ns1, sp1 = success_curve(jv1)
Ns2, sp2 = success_curve(jv2)

plt.figure(figsize=(7, 4))
plt.plot(Ns1, sp1, lw=2, label=f"K=1 (N90={N90_1}, N95={N95_1})")
plt.plot(Ns2, sp2, lw=2, ls="--", label=f"K=2 (N90={N90_2}, N95={N95_2})")
plt.axhline(0.90, ls=":", color="black", label="90%")
plt.axhline(0.95, ls=":", color="gray", label="95%")
plt.xlabel("Number of events")
plt.ylabel("Probability of accurate P̂")
plt.ylim(0, 1.02)
plt.legend()
plt.tight_layout()
plt.show()

# try MCMC, with our simulated data
event_times = np.asarray(t_events, dtype=float)
N_obs = len(event_times)
print(f"Using full simulated catalog: N_obs={N_obs}, T={T}")


## The Priors
def log_prior(theta, event_times, T, P_min=7.0, P_max=16.0):
    beta0_, beta1_, logP_, phi_ = theta
    P_ = np.exp(logP_)

    if not (P_min < P_ < P_max):
        return -np.inf
    if not (-3.0 < beta1_ < 3.0):
        return -np.inf
    if not (-np.pi <= phi_ <= np.pi):
        return -np.inf
    if not (-20.0 < beta0_ < 20.0):
        return -np.inf
    rough_rate = np.log((len(event_times) + 1e-9) / (T + 1e-9)) 

    lp = 0.0
    lp += -0.5 * ((beta1_ - 0.0) / 1.0) ** 2
    lp += -0.5 * ((beta0_ - rough_rate) / 3.0) ** 2 

    return lp

## Posterior log probability
def log_probability(theta, event_times, T, P_min=7.0, P_max=16.0, ll_grid=4000):
    beta0_, beta1_, logP_, phi_ = theta

    lp = log_prior(theta, event_times, T, P_min=P_min, P_max=P_max)
    if not np.isfinite(lp):
        return -np.inf

    P_ = np.exp(logP_)
    omega_ = 2.0 * np.pi / P_

    ll = log_likelihood_params(beta0_, beta1_, omega_, phi_, event_times, T, n_grid=ll_grid)
    return lp + ll


# Scan
# These parameters are the global maximum likelihood estimate, before MCMC sampling
periods_scan, logL_P_scan, P_hat = scan_period(
    event_times,
    beta0_true,
    beta1_true,
    phi_true,
    T,
    P_min=P_grid_min,
    P_max=P_grid_max,
    n_P=nP,
    ll_grid=2500,
)

b0g = beta0_grid(event_times, T, width=3.0, n=601)
logL_b0_scan, b0_hat = scan_beta0(
    event_times, beta1_true, omega_true, phi_true, T, b0g, ll_grid=2500
)
logL_b1_scan, b1_hat = scan_beta1(
    event_times, beta0_true, omega_true, phi_true, T, b1_grid, ll_grid=2500
)
logL_phi_scan, phi_hat = scan_phi(
    event_times, beta0_true, beta1_true, omega_true, T, phi_grid, ll_grid=2500
)

theta_center = np.array([b0_hat, b1_hat, np.log(P_hat), phi_hat], dtype=float)
print("Init center: [beta0, beta1, logP, phi] =", theta_center)
print("Init center as P_hat:", np.exp(theta_center[2]))

# Run emcee
ndim = 4 
nwalkers = 64 

seed = 2200
rng = np.random.default_rng(seed)

p0 = theta_center + 1e-2 * rng.standard_normal(size=(nwalkers, ndim))

p0[:, 3] = (p0[:, 3] + np.pi) % (2.0 * np.pi) - np.pi

logP_min, logP_max = np.log(P_grid_min), np.log(P_grid_max)
p0[:, 2] = np.clip(p0[:, 2], logP_min + 1e-6, logP_max - 1e-6)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(event_times, T),
    kwargs={"P_min": P_grid_min, "P_max": P_grid_max, "ll_grid": 4000},
)


nburn = 50000
state = sampler.run_mcmc(p0, nburn, progress=True, rstate0 = rng)
sampler.reset()

nsteps = 30000
sampler.run_mcmc(state, nsteps, progress=True, rstate0 = rng)

print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

thin = 10
flat = sampler.get_chain(discard=0, thin=thin, flat=True)

beta0_s = flat[:, 0]
beta1_s = flat[:, 1]
logP_s  = flat[:, 2]
phi_s   = flat[:, 3]

P_s = np.exp(logP_s)
phi_s = (phi_s + np.pi) % (2.0 * np.pi) - np.pi 

def q16_50_84(x):
    return np.percentile(x, [16, 50, 84])

print("Posterior (16, 50, 84 percentiles):")
print("beta0:", q16_50_84(beta0_s), " | true:", beta0_true)
print("beta1:", q16_50_84(beta1_s), " | true:", beta1_true)
print("P    :", q16_50_84(P_s),     " | true:", P_true)
print("phi  :", q16_50_84(phi_s),   " | true:", phi_true)


chain = sampler.get_chain()

labels = [r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"]

P_chain = np.exp(chain[:, :, 2])

# Trace plots
fig, axes = plt.subplots(ndim, 1, figsize=(10, 8), 
                         sharex=True, constrained_layout=True)
for i in range(ndim):
    ax = axes[i]

    if i == 2:
        # Plot P instead of logP
        for w in range(chain.shape[1]):
            ax.plot(P_chain[:, w], alpha=0.2, color ="blue")
        ax.axhline(P_true, ls="--", color="black")
    else:
        for w in range(chain.shape[1]):
            ax.plot(chain[:, w, i], alpha=0.2, color ="blue")
        if i == 0:
            ax.axhline(beta0_true, ls="--", color="black")
        elif i == 1:
            ax.axhline(beta1_true, ls="--", color="black")
        elif i == 3:
            ax.axhline(phi_true, ls="--", color="black")

    ax.set_ylabel(labels[i], fontsize=12)

axes[-1].set_xlabel("Step", fontsize=12)
plt.show()

# Corner plot
nsteps, nwalkers, ndim = chain.shape

flat = chain.reshape(-1, ndim)

beta0_s = flat[:, 0]
beta1_s = flat[:, 1]
P_s     = np.exp(flat[:, 2])
phi_s   = flat[:, 3]

phi_s = (phi_s + np.pi) % (2*np.pi) - np.pi

samples_corner = np.column_stack([beta0_s, beta1_s, P_s, phi_s])

fig = corner.corner(
    samples_corner,
    labels=[r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"],
    truths=[beta0_true, beta1_true, P_true, phi_true], color="blue", show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12}
)
plt.show()


# Posterior predictive checks
t_grid = np.linspace(0.0, T, 2000)

bins = 60
counts, edges = np.histogram(t_events, bins=bins, range=(0.0, T))
bin_width = edges[1] - edges[0]
rate_hist = counts / bin_width
centers = 0.5 * (edges[:-1] + edges[1:])

plt.figure(figsize=(10, 5))
plt.step(centers, rate_hist, where="mid", lw=2, 
         label="Empirical rate", color="black")

# Draw 300 posterior samples and plot their rate curves
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
    plt.plot(t_grid, lam, alpha=0.10, lw=1, color="blue")
plt.plot(t_events, np.zeros_like(t_events), "|", color="black", alpha=0.4)
plt.xlabel("Events Time")
plt.ylabel("Rate")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# Real data
test_korea = pd.read_excel("data/Korean_Aurora_Grades_918_1392.xlsx")

years_real = test_korea["Year"].astype(int).values
t0_real = int(years_real.min())
t_real = (years_real - t0_real).astype(float) 
T_real = float(t_real.max())

event_times = np.asarray(t_real, dtype=float)
T = T_real

print(f"Real data: N={len(event_times)}, t0={t0_real}, T={T:.1f} years")
print("First few event times (years since t0):", event_times[:10])

rough_rate = np.log((len(event_times) + 1e-9) / (T + 1e-9))

beta0_guess = rough_rate
beta1_guess = 0.0
phi_guess   = 0.0

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

theta_center = np.array([b0_hat, b1_hat, np.log(P_hat), phi_hat], dtype=float)
print("Init center: [beta0, beta1, logP, phi] =", theta_center)
print("Init center as P_hat:", np.exp(theta_center[2]))


# Run emcee
ndim = 4
nwalkers = 64

seed = 2200
rng = np.random.default_rng(seed)

p0 = theta_center + 1e-2 * rng.standard_normal(size=(nwalkers, ndim))

# wrap phi 
logP_min, logP_max = np.log(P_grid_min), np.log(P_grid_max)
p0[:, 2] = np.clip(p0[:, 2], logP_min + 1e-6, logP_max - 1e-6)

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

# Summaries (uncertainties)
thin = 10
flat = sampler.get_chain(discard=0, thin=thin, flat=True)

beta0_s = flat[:, 0]
beta1_s = flat[:, 1]
logP_s  = flat[:, 2]
phi_s   = flat[:, 3]

P_s = np.exp(logP_s)
phi_s = (phi_s + np.pi) % (2.0 * np.pi) - np.pi

print("Posterior (16, 50, 84 percentiles):")
print("beta0:", q16_50_84(beta0_s))
print("beta1:", q16_50_84(beta1_s))
print("P    :", q16_50_84(P_s))
print("phi  :", q16_50_84(phi_s))
    
# Trace plots
chain = sampler.get_chain()
labels = [r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"]
P_chain = np.exp(chain[:, :, 2])

fig, axes = plt.subplots(ndim, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
for i in range(ndim):
    ax = axes[i]
    if i == 2:
        for w in range(chain.shape[1]):
            ax.plot(P_chain[:, w], alpha=0.2, color="blue")
        ax.axhline(P_hat, ls="--", color="black", label="P_hat")
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
phi_s2   = (flat2[:, 3] + np.pi) % (2*np.pi) - np.pi

samples_corner = np.column_stack([beta0_s2, beta1_s2, P_s2, phi_s2])

fig = corner.corner(
    samples_corner,
    labels=[r"$\beta_0$", r"$\beta_1$", r"$P$", r"$\phi$"],
    truths=[b0_hat, b1_hat, P_hat, phi_hat], 
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