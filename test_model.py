import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, PillowWriter

T = 354.0
P_true = 11.0
omega_true = 2 * np.pi / P_true
beta1_true = 0.6
phi_true = np.pi / 4
N_target = 200


def beta0(beta1, omega, phi, T, N_target, n_grid=20000):
    t_grid = np.linspace(0.0, T, n_grid)
    base = np.exp(beta1 * np.sin(omega * t_grid + phi))
    integral_base = np.trapz(base, t_grid)
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
        return -np.trapz(lam_grid, t_grid)

    term_events = np.sum(beta0 + beta1 * np.sin(omega * event_times + phi))
    t_grid = np.linspace(0.0, T, n_grid)
    lam_grid = np.exp(beta0 + beta1 * np.sin(omega * t_grid + phi))
    integral = np.trapz(lam_grid, t_grid)
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
integrals = np.trapz(lam_grid, t_grid_ll, axis=1)

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

anim.save("random_subset_animation.gif", writer=PillowWriter(fps=10))
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

print("\nRange across seeds:")
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
    return np.log(N_target / np.trapz(base, t))


# 1 harmonic
def lambda_func_1harm(t, beta0, omega, a1, b1):
    t = np.asarray(t, dtype=float)
    return np.exp(beta0 + a1 * np.sin(omega * t) + b1 * np.cos(omega * t))


def beta0_1harm(a1, b1, omega, T, N_target, n_grid=20000):
    t = np.linspace(0.0, T, n_grid)
    base = np.exp(a1 * np.cos(omega * t) + b1 * np.sin(omega * t))
    return np.log(N_target / np.trapz(base, t))


# Get log likelihood functions
def log_likelihood_1harm(
    event_times, T, N_target, omega, a1_grid, b1_grid, ll_grid=4000
):
    if len(event_times) == 0:
        return -np.inf, (np.nan, np.nan, np.nan)

    # compute cos and sin at event times
    c1 = np.cos(omega * event_times)
    s1 = np.sin(omega * event_times)

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
            integral = np.trapz(lam_grid, t_grid)
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

    # compute cos and sin at event times
    c1 = np.cos(omega * event_times)
    s1 = np.sin(omega * event_times)
    c2 = np.cos(2 * omega * event_times)
    s2 = np.sin(2 * omega * event_times)

    # integration grid
    t_grid = np.linspace(0.0, T, ll_grid)
    c1g = np.cos(omega * t_grid)
    s1g = np.sin(omega * t_grid)
    c2g = np.cos(2 * omega * t_grid)
    s2g = np.sin(2 * omega * t_grid)

    bestL = -np.inf
    best = None

    for a1 in a1_grid:
        for b1 in b1_grid:
            for a2 in a2_grid:
                for b2 in b2_grid:
                    b0 = beta0_2harm(a1, b1, a2, b2, omega, T, N_target, n_grid=ll_grid)

                    term = np.sum(b0 + a1 * c1 + b1 * s1 + a2 * c2 + b2 * s2)
                    lam_grid = np.exp(b0 + a1 * c1g + b1 * s1g + a2 * c2g + b2 * s2g)
                    integral = np.trapz(lam_grid, t_grid)
                    L = term - integral

                    if L > bestL:
                        bestL = L
                        best = (b0, a1, b1, a2, b2)

    return bestL, best


a1_true, b1_true = 0.0, 0.6
a2_true, b2_true = 0.0, 0.3

omega_true = 2 * np.pi / P_true
beta0_true_2 = beta0_2harm(a1_true, b1_true, a2_true, b2_true, omega_true, T, N_target)


# At P_true (2 harmonics), fit 1 harmonic, then plot intensity shapes
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

# Log likelihood vs period
P_min, P_max, nP = 7.0, 16.0, 300
periods = np.linspace(P_min, P_max, nP)

a1_grid = np.linspace(0.0, 1.0, 21)
b1_grid = np.linspace(-np.pi, np.pi, 61)
a2_grid = np.linspace(0.0, 0.6, 13)
b2_grid = np.linspace(-np.pi, np.pi, 61)

logL1 = np.empty(nP)
logL2 = np.empty(nP)

best1_params = []
best2_params = []

for i, P in enumerate(periods):
    omega = 2 * np.pi / P

    L1, p1 = log_likelihood_1harm(
        t_events, T, N_target, omega, a1_grid, b1_grid, ll_grid=2500
    )
    logL1[i] = L1
    best1_params.append(p1)

    L2, p2 = log_likelihood_2harm(
        t_events, T, N_target, omega, a1_grid, b1_grid, a2_grid, b2_grid, ll_grid=2500
    )
    logL2[i] = L2
    best2_params.append(p2)

P_hat_1 = periods[int(np.argmax(logL1))]
P_hat_2 = periods[int(np.argmax(logL2))]

plt.figure(figsize=(10, 4))
plt.plot(periods, logL1, lw=2, label="K=1")
plt.plot(periods, logL2, lw=2, ls="--", label="LK=2")
plt.axvline(P_true, ls=":", label="P_true")
plt.axvline(P_hat_1, ls="-.", label=f"P_hat (K=1) = {P_hat_1:.2f}")
plt.axvline(P_hat_2, ls="-.", label=f"P_hat (K=2) = {P_hat_2:.2f}")
plt.xlabel("Period P")
plt.ylabel("Profile log-likelihood")
plt.legend()
plt.tight_layout()
plt.show()

print(f"P_hat (fit K=1): {P_hat_1:.3f}")
print(f"P_hat (fit K=2): {P_hat_2:.3f}")
