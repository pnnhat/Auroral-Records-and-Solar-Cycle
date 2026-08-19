## Model-free period estimators
# Estimate the cycle without assuming any NHPP rate model, as a cross-check on the likelihood result
# point_process_periodogram: Bartlett (1963) spectral estimate I(f)=(1/N)|sum exp(2*pi*i*f*t_n)|^2
# modified_lls: Quinn et al. (2012) phase-misalignment estimator, robust to intensity non-stationarity
# refine_peak: Brent refinement of a coarse periodogram peak
# recovery_test: self-contained simulation validation of both estimators
# plot_periodogram_suite: point-process vs modified-LLS comparison figure
# animate_periodogram: sequential build-up of I(f) as events accumulate

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from numpy import cumsum, concatenate, diff
from matplotlib.animation import FuncAnimation, PillowWriter

## Point-process periodogram


def point_process_periodogram(
    event_times: np.ndarray,
    f_grid: np.ndarray,
) -> np.ndarray:
    # Bartlett / Fogel–Gavish point-process periodogram.
    #
    # I(f) = (1/N) |Σ_n exp(2πi f t_n)|²
    # = (1/N) [(Σ cos(2πf t_n))² + (Σ sin(2πf t_n))²]
    #
    # Parameters
    # ----------
    # event_times : 1-D array of event arrival times.
    # f_grid      : 1-D array of frequencies (cycles per unit time).
    #
    # Returns
    # -------
    # I : 1-D array, shape (len(f_grid),).
    t = np.asarray(event_times, dtype=float)
    f = np.asarray(f_grid, dtype=float)
    N = len(t)
    # Compute outer product safely in float64 to avoid memory issues for large grids
    chunk = max(1, int(20_000_000 // N))
    I = np.empty(len(f))
    for start in range(0, len(f), chunk):
        sl = f[start : start + chunk]
        phase = 2.0 * np.pi * np.outer(sl, t)
        cs = np.sum(np.cos(phase), axis=1)
        ss = np.sum(np.sin(phase), axis=1)
        I[start : start + chunk] = (cs**2 + ss**2) / N
    return I


## Modified least-squares


def _ssmod_at_gamma(gtn: np.ndarray, n_nu: int) -> float:
    # Compute  min_ν Σ ⟨gtn − ν⟩²  via a fine ν-grid search.
    #
    # gtn : pre-computed γ * t_n values (1-D array).
    nu = np.linspace(0.0, 1.0, n_nu, endpoint=False)  # (n_nu,)
    r = gtn[None, :] - nu[:, None]  # (n_nu, N)
    r -= np.round(r)
    return float(np.sum(r**2, axis=1).min())


def modified_lls(
    event_times: np.ndarray,
    P_min: float,
    P_max: float,
    nP: int = 800,
    n_nu: int = 400,
) -> tuple[np.ndarray, np.ndarray, float]:
    # Modified least-squares period estimator (Quinn, Clarkson & McKilliam 2012).
    #
    # Minimises  SSMOD(γ) = min_ν  Σ_n ⟨γ t_n − ν⟩²
    # over a grid γ = 1/P, then returns the period at the minimum.
    #
    # Parameters
    # ----------
    # event_times : 1-D array of event arrival times.
    # P_min, P_max: period search bounds (same units as event_times).
    # nP          : number of γ grid points.
    # n_nu        : ν-grid resolution for the inner minimisation.
    #
    # Returns
    # -------
    # P_grid : 1-D array of period values (ascending).
    # ssmod  : 1-D array of SSMOD values (same order as P_grid).
    # P_hat  : float, period at minimum SSMOD.
    t = np.asarray(event_times, dtype=float)
    gamma_grid = np.linspace(1.0 / P_max, 1.0 / P_min, nP)
    ssmod = np.empty(nP)
    for j, gamma in enumerate(gamma_grid):
        ssmod[j] = _ssmod_at_gamma(gamma * t, n_nu=n_nu)
    # Return in ascending P order (gamma_grid is descending in P)
    P_grid = 1.0 / gamma_grid[::-1]
    ssmod = ssmod[::-1]
    P_hat = float(P_grid[np.argmin(ssmod)])
    return P_grid, ssmod, P_hat


## Peak refinement


def refine_peak(
    event_times: np.ndarray,
    f_coarse: float,
    half_width: float,
) -> tuple[float, float]:
    # Refine a coarse frequency estimate using Brent's method on the
    # point-process periodogram.
    #
    # Parameters
    # ----------
    # event_times : 1-D array.
    # f_coarse    : coarse frequency estimate (cycles / unit time).
    # half_width  : search half-width around f_coarse.
    #
    # Returns
    # -------
    # f_hat : refined frequency.
    # I_hat : periodogram value at f_hat.
    t = np.asarray(event_times, dtype=float)
    N = len(t)

    def neg_I(f: float) -> float:
        phase = 2.0 * np.pi * f * t
        cs = float(np.sum(np.cos(phase)))
        ss = float(np.sum(np.sin(phase)))
        return -(cs**2 + ss**2) / N

    res = minimize_scalar(
        neg_I,
        bounds=(f_coarse - half_width, f_coarse + half_width),
        method="bounded",
    )
    return float(res.x), float(-res.fun)


## Validation: simulation-based recovery test


def simulate_sparse_events(
    P_true: float,
    T: float,
    N_target: int,
    beta1: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    # Simulate event times from a sinusoidal NHPP for validation purposes.
    # Uses the same cdf_inversion approach as helpers.py.

    rng = np.random.default_rng(seed)
    omega = 2.0 * np.pi / P_true
    n_grid = 20000
    t_grid = np.linspace(0.0, T, n_grid)
    # Set beta0 so E[N] = N_target
    base = np.exp(beta1 * np.sin(omega * t_grid))

    beta0 = np.log(N_target / trapezoid(base, t_grid))
    lam = np.exp(beta0 + beta1 * np.sin(omega * t_grid))

    incr = 0.5 * (lam[:-1] + lam[1:]) * diff(t_grid)
    Lambda = concatenate([[0.0], cumsum(incr)])
    Lambda_T = Lambda[-1]
    N = rng.poisson(Lambda_T)
    u = rng.random(N) * Lambda_T
    inv_L = interp1d(
        Lambda, t_grid, kind="linear", bounds_error=False, fill_value=(0.0, T)
    )
    t_ev = np.sort(inv_L(u))
    return t_ev


def recovery_test(
    P_true: float,
    T: float,
    N_target: int,
    P_min: float = 7.0,
    P_max: float = 16.0,
    nP: int = 600,
    nf: int = 6000,
    n_sims: int = 20,
    seed: int = 42,
) -> dict:
    # Simulate n_sims datasets under P_true and report period recovery
    # statistics for both the point-process periodogram and modified LLS.
    #
    # Returns dict with keys: P_true, pp_errors, lls_errors, pp_median, lls_median.
    f_grid = np.linspace(1.0 / P_max, 1.0 / P_min, nf)
    pp_errors = []
    lls_errors = []
    for k in range(n_sims):
        t_ev = simulate_sparse_events(P_true, T, N_target, seed=seed + k)
        # Point-process periodogram
        I = point_process_periodogram(t_ev, f_grid)
        f_peak = float(f_grid[np.argmax(I)])
        f_ref, _ = refine_peak(t_ev, f_peak, half_width=2.0 / T)
        pp_errors.append(abs(1.0 / f_ref - P_true))
        # Modified LLS
        _, _, P_lls = modified_lls(t_ev, P_min, P_max, nP=nP)
        lls_errors.append(abs(P_lls - P_true))
    return {
        "P_true": P_true,
        "pp_errors": pp_errors,
        "lls_errors": lls_errors,
        "pp_median": float(np.median(pp_errors)),
        "lls_median": float(np.median(lls_errors)),
    }


## Main comparison plot


def plot_periodogram_suite(
    event_times: np.ndarray,
    T: float,
    P_min: float = 7.0,
    P_max: float = 16.0,
    nf: int = 8000,
    nP_lls: int = 800,
    figsize: tuple = (12, 4.5),
) -> plt.Figure:
    # Two-panel figure:
    # Left  : Point-process periodogram  I(f)
    # Right : Modified LLS  SSMOD(P)  — period at minimum
    #
    # Parameters
    # ----------
    # event_times : 1-D array of event arrival times.
    # T           : observation window length.
    # P_min, P_max: period search range (years).
    # nf          : frequency grid resolution.
    # nP_lls      : period grid resolution for modified LLS.
    f_lo = 1.0 / P_max
    f_hi = 1.0 / P_min
    f_grid = np.linspace(f_lo, f_hi, nf)
    P_grid_f = 1.0 / f_grid[::-1]  # ascending period axis

    print("Computing point-process periodogram…")
    I_pp = point_process_periodogram(event_times, f_grid)[::-1]

    print(f"Computing modified LLS (nP={nP_lls}, n_nu=400)…")
    P_lls, ssmod, P_hat_lls = modified_lls(event_times, P_min, P_max, nP=nP_lls)

    # Refine point-process peak
    P_hat_pp = float(P_grid_f[np.argmax(I_pp)])
    f_peak = 1.0 / P_hat_pp
    delta_f = 2.0 / T
    f_ref, _ = refine_peak(event_times, f_peak, half_width=delta_f)
    P_hat_pp_refined = 1.0 / f_ref

    print(f"\nPeriod estimates:")
    print(f"  Point-process (refined) : {P_hat_pp_refined:.4f} yr")
    print(f"  Modified LLS            : {P_hat_lls:.4f} yr")

    fig, (ax_pp, ax_lls) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Left: Point-process periodogram
    ax_pp.plot(P_grid_f, I_pp, color="black", lw=0.7)
    ax_pp.axvline(11.0, ls="--", color="red", alpha=0.7, label="11 yr")
    ax_pp.axvline(
        P_hat_pp_refined,
        ls=":",
        color="steelblue",
        lw=1.5,
        label=f"Peak  {P_hat_pp_refined:.3f} yr",
    )
    ax_pp.set_xlabel("Period P (years)")
    ax_pp.set_ylabel(r"$I(f) = \frac{1}{N}|\sum e^{2\pi i f t_n}|^2$")
    ax_pp.set_title("Point-process periodogram")
    ax_pp.set_xlim(P_min, P_max)
    ax_pp.legend(frameon=False, fontsize=9)

    # Right: Modified LLS
    ssmod_norm = ssmod / ssmod.max()
    ax_lls.plot(P_lls, ssmod_norm, color="darkgreen", lw=0.7)
    ax_lls.axvline(11.0, ls="--", color="red", alpha=0.7, label="11 yr")
    ax_lls.axvline(
        P_hat_lls, ls=":", color="steelblue", lw=1.5, label=f"Min  {P_hat_lls:.3f} yr"
    )
    ax_lls.set_xlabel("Period P (years)")
    ax_lls.set_ylabel(r"$\mathrm{SSMOD}(\gamma)$  (normalised)")
    ax_lls.set_title("Modified least-squares periodogram")
    ax_lls.set_xlim(P_min, P_max)
    ax_lls.legend(frameon=False, fontsize=9)

    fig.suptitle("Period estimation from auroral event times", fontsize=13)
    return fig


## Sequential periodogram animation


def animate_periodogram(
    event_times,
    years_abs,
    T: float,
    P_min: float = 7.0,
    P_max: float = 16.0,
    nf: int = 3000,
    step: int = 5,
    fps: int = 12,
    out_path: str = "plots/periodogram_animation.gif",
) -> None:
    # Animate the point-process periodogram I(f) as events are added one
    # batch at a time (in chronological order).
    #
    # Each frame adds `step` events and shows:
    # Top panel    : event rug plot (absolute calendar years) up to current N.
    # Bottom panel : I(f) on a period axis, with the current dominant peak marked.
    #
    # The animation reveals:
    # - How quickly the dominant peak near ~9.9 yr stabilises.
    # - The visible impact of the 1535-1563 CE burst epoch on the spectrum.
    # - The aliasing structure (multiple near-equal peaks) across the search range.
    #
    # Parameters
    # ----------
    # event_times : 1-D array of relative event times (years since t0), sorted.
    # years_abs   : 1-D array of absolute calendar years, same order as event_times.
    # T           : total observation window (years).
    # P_min, P_max: period search range (years).
    # nf          : frequency grid resolution (lower = faster render).
    # step        : number of events added per frame.
    # fps         : frames per second in output GIF.
    # out_path    : save path for the GIF.

    t = np.asarray(event_times, dtype=float)
    yrs = np.asarray(years_abs, dtype=float)
    N_total = len(t)

    f_grid = np.linspace(1.0 / P_max, 1.0 / P_min, nf)
    P_grid = 1.0 / f_grid[::-1]  # ascending period axis

    t0_abs = int(yrs.min())
    bin_edges = np.linspace(t0_abs, t0_abs + T, 60)

    frame_ends = list(range(max(10, step), N_total, step)) + [N_total]

    fig, (ax_hist, ax_per) = plt.subplots(
        2, 1, figsize=(12, 8), constrained_layout=True
    )

    def _update(n_events):
        t_sub = t[:n_events]
        y_sub = yrs[:n_events]

        # Top: histogram of calendar years seen so far
        ax_hist.clear()
        ax_hist.hist(y_sub, bins=bin_edges, color="black", alpha=0.7)
        ax_hist.axvspan(1535, 1563, color="orange", alpha=0.3)
        ax_hist.text(
            1549,
            ax_hist.get_ylim()[1] * 0.85,
            "Burst 1535–1563 CE",
            fontsize=10,
            color="darkorange",
            ha="center",
        )
        ax_hist.set_xlim(t0_abs, t0_abs + T)
        ax_hist.set_title(f"Number of events: {n_events}", fontsize=20)
        ax_hist.set_xlabel("Calendar year (CE)", fontsize=16)
        ax_hist.set_ylabel("Count", fontsize=16)
        ax_hist.tick_params(axis="both", labelsize=14)

        # Bottom: point-process periodogram
        ax_per.clear()
        I_raw = point_process_periodogram(t_sub, f_grid)[::-1]
        P_hat = float(P_grid[np.argmax(I_raw)])
        ax_per.plot(P_grid, I_raw, color="black", lw=2.5)
        ax_per.axvline(11.0, ls="--", color="red", label="11 yr")
        ax_per.axvline(
            P_hat, ls=":", color="blue", label=f"Estimated period = {P_hat:.2f} yr"
        )
        ax_per.set_xlim(P_min, P_max)
        ax_per.set_title("Period Recovery (Point-Process Periodogram)", fontsize=20)
        ax_per.set_xlabel("Period (years)", fontsize=16)
        ax_per.set_ylabel("I(f)", fontsize=16)
        ax_per.tick_params(axis="both", labelsize=14)
        ax_per.legend(loc="upper right", fontsize=14, frameon=False)

        return []

    anim = FuncAnimation(
        fig,
        _update,
        frames=frame_ends,
        interval=1000 // fps,
        blit=False,
    )

    print(f"Rendering {len(frame_ends)} frames -> {out_path}")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    print("Saved.")
    plt.close(fig)


## Build event times from the full Korean record
# Rebase calendar years to years-since-first-event; T is the observation span
korean_pp = pd.read_excel("data/KoreanAuroraRecords/Korean_Auroral_Full.xlsx")
years_pp = korean_pp["Year"].astype(int).values
t0_pp = int(years_pp.min())
event_times_pp = (years_pp - t0_pp).astype(float)
T_pp = float(event_times_pp.max())
print(f"N={len(event_times_pp)}, T={T_pp:.1f} yr")

## Point-process periodogram vs modified least-squares
# Two-panel comparison figure over the 7-16 yr search range
fig_pp = plot_periodogram_suite(
    event_times_pp, T_pp, P_min=7.0, P_max=16.0, nf=8000, nP_lls=800
)
plt.show()

## Sequential build-up animation on the real data
# Add events in batches of 5 and recompute I(f); saves a GIF to out_path (create the folder yourself)
animate_periodogram(
    event_times=event_times_pp,
    years_abs=years_pp,
    T=T_pp,
    P_min=7.0,
    P_max=16.0,
    nf=3000,
    step=5,
    fps=9,
    out_path="plots/periodogram_animation.gif",
)
