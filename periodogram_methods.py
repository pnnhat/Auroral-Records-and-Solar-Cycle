"""
periodogram_methods.py
======================

Period estimation via periodogram and modified-least-squares methods,
as alternatives to the stationary NHPP likelihood approach.

Three estimators are implemented:

1. point_process_periodogram
   Bartlett (1963) / Fogel–Gavish spectral estimate for point processes:
       I(f) = (1/N) |sum_n exp(2πi f t_n)|^2
   Requires no model for the event rate.  Robust to non-stationarity in
   intensity because it does not compare observed counts to a modelled rate.

2. lomb_scargle_events
   Lomb–Scargle regression periodogram applied directly to event-time data
   (each event = amplitude 1), with proper DC-term handling via mean-
   correction (Quinn 2016).  Equivalent to fitting
       y(t) = μ + α cos(2πft) + β sin(2πft) + ε
   at each frequency.

3. modified_lls
   Modified least-squares estimator of Quinn, Clarkson & McKilliam (2012).
   Minimises
       SSMOD(γ) = min_ν  Σ_n ⟨γ t_n − ν⟩²
   where ⟨x⟩ = x − round(x) and γ = 1/P.
   Each event is treated as a noisy measurement of a periodic phase; the
   estimator is STRONGLY CONSISTENT with N^{3/2} convergence rate under
   mild distributional assumptions.  It does not model the event rate and
   is therefore insensitive to non-stationarity in intensity.

4. refine_peak
   Fine-tunes a coarse frequency estimate using scalar Brent optimisation
   on the point-process periodogram.

5. plot_periodogram_suite
   Four-panel comparison figure: (a) point-process, (b) Lomb-Scargle,
   (c) modified LLS, (d) zoom near dominant peak.

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from astropy.timeseries import LombScargle


# ---------------------------------------------------------------------------
# 1.  Point-process periodogram
# ---------------------------------------------------------------------------

def point_process_periodogram(
    event_times: np.ndarray,
    f_grid: np.ndarray,
) -> np.ndarray:
    """
    Bartlett / Fogel–Gavish point-process periodogram.

        I(f) = (1/N) |Σ_n exp(2πi f t_n)|²
             = (1/N) [(Σ cos(2πf t_n))² + (Σ sin(2πf t_n))²]

    Parameters
    ----------
    event_times : 1-D array of event arrival times.
    f_grid      : 1-D array of frequencies (cycles per unit time).

    Returns
    -------
    I : 1-D array, shape (len(f_grid),).
    """
    t = np.asarray(event_times, dtype=float)
    f = np.asarray(f_grid, dtype=float)
    N = len(t)
    # Compute outer product safely in float64 to avoid memory issues for large grids
    # Process in chunks if nf * N > 20M to keep memory < ~320 MB
    chunk = max(1, int(20_000_000 // N))
    I = np.empty(len(f))
    for start in range(0, len(f), chunk):
        sl = f[start:start + chunk]
        phase = 2.0 * np.pi * np.outer(sl, t)  # (chunk, N)
        cs = np.sum(np.cos(phase), axis=1)
        ss = np.sum(np.sin(phase), axis=1)
        I[start:start + chunk] = (cs ** 2 + ss ** 2) / N
    return I



# ---------------------------------------------------------------------------
# 2.  Modified least-squares (Quinn, Clarkson & McKilliam 2012)
# ---------------------------------------------------------------------------

def _ssmod_at_gamma(gtn: np.ndarray, n_nu: int) -> float:
    """
    Compute  min_ν Σ ⟨gtn − ν⟩²  via a fine ν-grid search.

    gtn : pre-computed γ * t_n values (1-D array).
    """
    nu = np.linspace(0.0, 1.0, n_nu, endpoint=False)  # (n_nu,)
    r = gtn[None, :] - nu[:, None]                      # (n_nu, N)
    r -= np.round(r)
    return float(np.sum(r ** 2, axis=1).min())


def modified_lls(
    event_times: np.ndarray,
    P_min: float,
    P_max: float,
    nP: int = 800,
    n_nu: int = 400,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Modified least-squares period estimator (Quinn, Clarkson & McKilliam 2012).

    Minimises  SSMOD(γ) = min_ν  Σ_n ⟨γ t_n − ν⟩²
    over a grid γ = 1/P, then returns the period at the minimum.

    Parameters
    ----------
    event_times : 1-D array of event arrival times.
    P_min, P_max: period search bounds (same units as event_times).
    nP          : number of γ grid points.
    n_nu        : ν-grid resolution for the inner minimisation.

    Returns
    -------
    P_grid : 1-D array of period values (ascending).
    ssmod  : 1-D array of SSMOD values (same order as P_grid).
    P_hat  : float, period at minimum SSMOD.

    Notes
    -----
    Strongly consistent estimator with N^{3/2} convergence rate under
    mild distributional assumptions (Quinn et al. 2012, Theorem 1 & 3).
    Does not model the event rate; robust to intensity non-stationarity.
    Runtime: O(nP * n_nu * N).  With nP=800, n_nu=400, N=787 ≈ 250M ops
    — takes ~30 s in pure Python; vectorised here via NumPy broadcasting.
    """
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


# ---------------------------------------------------------------------------
# 4.  Peak refinement
# ---------------------------------------------------------------------------

def refine_peak(
    event_times: np.ndarray,
    f_coarse: float,
    half_width: float,
) -> tuple[float, float]:
    """
    Refine a coarse frequency estimate using Brent's method on the
    point-process periodogram.

    Parameters
    ----------
    event_times : 1-D array.
    f_coarse    : coarse frequency estimate (cycles / unit time).
    half_width  : search half-width around f_coarse.

    Returns
    -------
    f_hat : refined frequency.
    I_hat : periodogram value at f_hat.
    """
    t = np.asarray(event_times, dtype=float)
    N = len(t)

    def neg_I(f: float) -> float:
        phase = 2.0 * np.pi * f * t
        cs = float(np.sum(np.cos(phase)))
        ss = float(np.sum(np.sin(phase)))
        return -(cs ** 2 + ss ** 2) / N

    res = minimize_scalar(
        neg_I,
        bounds=(f_coarse - half_width, f_coarse + half_width),
        method="bounded",
    )
    return float(res.x), float(-res.fun)


# ---------------------------------------------------------------------------
# 5.  Validation: simulation-based recovery test
# ---------------------------------------------------------------------------

def simulate_sparse_events(
    P_true: float,
    T: float,
    N_target: int,
    beta1: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate event times from a sinusoidal NHPP for validation purposes.
    Uses the same cdf_inversion approach as helpers.py.
    """
    from scipy.interpolate import interp1d
    rng = np.random.default_rng(seed)
    omega = 2.0 * np.pi / P_true
    n_grid = 20000
    t_grid = np.linspace(0.0, T, n_grid)
    # Set beta0 so E[N] = N_target
    base = np.exp(beta1 * np.sin(omega * t_grid))
    from scipy.integrate import trapezoid
    beta0 = np.log(N_target / trapezoid(base, t_grid))
    lam = np.exp(beta0 + beta1 * np.sin(omega * t_grid))
    from numpy import cumsum, concatenate, diff
    incr = 0.5 * (lam[:-1] + lam[1:]) * diff(t_grid)
    Lambda = concatenate([[0.0], cumsum(incr)])
    Lambda_T = Lambda[-1]
    N = rng.poisson(Lambda_T)
    u = rng.random(N) * Lambda_T
    inv_L = interp1d(Lambda, t_grid, kind="linear",
                     bounds_error=False, fill_value=(0.0, T))
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
    """
    Simulate n_sims datasets under P_true and report period recovery
    statistics for both the point-process periodogram and modified LLS.

    Returns dict with keys: P_true, pp_errors, lls_errors, pp_median, lls_median.
    """
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


# ---------------------------------------------------------------------------
# 6.  Main comparison plot
# ---------------------------------------------------------------------------

def plot_periodogram_suite(
    event_times: np.ndarray,
    T: float,
    P_min: float = 7.0,
    P_max: float = 16.0,
    nf: int = 8000,
    nP_lls: int = 800,
    figsize: tuple = (12, 4.5),
) -> plt.Figure:
    """
    Two-panel figure:
      Left  : Point-process periodogram  I(f)
      Right : Modified LLS  SSMOD(P)  — period at minimum

    Parameters
    ----------
    event_times : 1-D array of event arrival times.
    T           : observation window length.
    P_min, P_max: period search range (years).
    nf          : frequency grid resolution.
    nP_lls      : period grid resolution for modified LLS.
    """
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
    ax_pp.axvline(P_hat_pp_refined, ls=":", color="steelblue", lw=1.5,
                  label=f"Peak  {P_hat_pp_refined:.3f} yr")
    ax_pp.set_xlabel("Period P (years)")
    ax_pp.set_ylabel(r"$I(f) = \frac{1}{N}|\sum e^{2\pi i f t_n}|^2$")
    ax_pp.set_title("Point-process periodogram")
    ax_pp.set_xlim(P_min, P_max)
    ax_pp.legend(frameon=False, fontsize=9)

    # Right: Modified LLS
    ssmod_norm = ssmod / ssmod.max()
    ax_lls.plot(P_lls, ssmod_norm, color="darkgreen", lw=0.7)
    ax_lls.axvline(11.0, ls="--", color="red", alpha=0.7, label="11 yr")
    ax_lls.axvline(P_hat_lls, ls=":", color="steelblue", lw=1.5,
                   label=f"Min  {P_hat_lls:.3f} yr")
    ax_lls.set_xlabel("Period P (years)")
    ax_lls.set_ylabel(r"$\mathrm{SSMOD}(\gamma)$  (normalised)")
    ax_lls.set_title("Modified least-squares periodogram")
    ax_lls.set_xlim(P_min, P_max)
    ax_lls.legend(frameon=False, fontsize=9)

    fig.suptitle("Period estimation from auroral event times", fontsize=13)
    return fig


# ---------------------------------------------------------------------------
# 7.  Subsampling robustness check
# ---------------------------------------------------------------------------

def subsampling_robustness(
    event_times: np.ndarray,
    T: float,
    keep_fractions: tuple = (0.9, 0.7, 0.5),
    n_reps: int = 30,
    P_min: float = 7.0,
    P_max: float = 16.0,
    nf: int = 4000,
    seed: int = 0,
) -> plt.Figure:
    """
    Assess robustness of the point-process periodogram by randomly dropping
    a fraction of events and re-estimating the period.

    Returns a figure showing the distribution of P_hat across replicates
    for each keep fraction.
    """
    rng = np.random.default_rng(seed)
    f_grid = np.linspace(1.0 / P_max, 1.0 / P_min, nf)
    delta_f = 2.0 / T
    results = {frac: [] for frac in keep_fractions}

    # Full-data estimate
    I_full = point_process_periodogram(event_times, f_grid)
    f_full = float(f_grid[np.argmax(I_full)])
    f_full_ref, _ = refine_peak(event_times, f_full, half_width=delta_f)
    P_full = 1.0 / f_full_ref
    print(f"Full-data peak: {P_full:.4f} yr")

    for frac in keep_fractions:
        n_keep = int(frac * len(event_times))
        for _ in range(n_reps):
            idx = rng.choice(len(event_times), size=n_keep, replace=False)
            t_sub = event_times[idx]
            I_sub = point_process_periodogram(t_sub, f_grid)
            f_peak = float(f_grid[np.argmax(I_sub)])
            f_ref, _ = refine_peak(t_sub, f_peak, half_width=delta_f)
            results[frac].append(1.0 / f_ref)
        med = np.median(results[frac])
        std = np.std(results[frac])
        print(f"  keep {int(100*frac)}%: P_hat = {med:.4f} ± {std:.4f} yr")

    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.axvline(P_full, ls="--", color="black", lw=1.5, label=f"Full data  {P_full:.3f} yr")
    ax.axvline(11.0, ls=":", color="red", alpha=0.7, label="11 yr")
    colours = ["steelblue", "darkorange", "darkgreen"]
    p_plot = np.linspace(P_min, P_max, 500)
    for (frac, vals), col in zip(results.items(), colours):
        vals_arr = np.array(vals)
        if len(np.unique(vals_arr)) > 1:
            kde = gaussian_kde(vals_arr)
            ax.plot(p_plot, kde(p_plot), color=col, lw=2,
                    label=f"keep {int(100*frac)}%  (n={n_reps})")
            ax.fill_between(p_plot, kde(p_plot), alpha=0.2, color=col)
    ax.set_xlabel("P̂  (years)")
    ax.set_ylabel("Density")
    ax.set_title("Subsampling robustness for Point-process periodogram")
    ax.legend(frameon=False, fontsize=9)
    return fig
