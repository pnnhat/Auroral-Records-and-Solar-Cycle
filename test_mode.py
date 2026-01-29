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


def lambda_func(t, beta0, beta1, omega, phi):
    t = np.asarray(t, dtype=float)
    return np.exp(beta0 + beta1 * np.sin(omega * t + phi))


def cdf_inversion(beta0, beta1, omega, phi, T, n_grid=20000, seed=123):
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
    seed=42,
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


hist_counts, _ = np.histogram(t_events, bins=200)
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
    ax_hist.hist(prefix, bins=200, density=False, alpha=0.6, color="black")
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
    fig, update, frames=len(t_events), init_func=init, interval=80, blit=False
)

anim.save("animation.gif", writer=PillowWriter(fps=10))
print("Saved: animation.gif")

plt.show()
