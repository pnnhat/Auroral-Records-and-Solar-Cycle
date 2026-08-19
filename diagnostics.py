## Validation diagnostics for the real-data NHPP fit (Part 3)
# All functions are read-only w.r.t. the inference: they check the fit, they do not alter posterior samples
# profile_logL_over_P: profile logL of P, maximised over (beta0, a, b); concave at fixed omega, so one L-BFGS-B run finds the global max
# windowed_period_scan: slide a fixed-width window over the record to test whether the dominant period is stationary across epochs
# run_sbc: simulation-based calibration; a uniform rank histogram means the estimator is well-calibrated
# plot_windowed_scan / plot_sbc: two-panel figures for the scans above
# plot_ppc_epoch: posterior predictive checks split by epoch, the key check for non-stationarity near grand minima
# profile_logL_2harm_over_P / plot_profile_comparison: 1- vs 2-harmonic profile, to show the multimodality is structural aliasing not a harmonic-count artefact

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import trapezoid

from helpers import log_likelihood_ab, cdf_inversion
from helpers import beta0 as _compute_beta0  # renamed to avoid shadowing


## Profile log-likelihood over P


def _neg_ll_and_grad(params, sin_ev, cos_ev, N, T, omega, ll_grid):
    # Negative NHPP log-likelihood and its analytical gradient w.r.t. (beta0, a, b)
    # for a fixed angular frequency omega.
    #
    # The log-likelihood is:
    # L = N*beta0 + a*sum(sin) + b*sum(cos) - integral_0^T exp(beta0 + a*sin + b*cos) dt
    #
    # Because L is strictly concave in (beta0, a, b), a single gradient-based
    # optimisation reliably finds the global maximum.
    beta0_, a_, b_ = params

    evt_term = float(N) * beta0_ + a_ * float(sin_ev.sum()) + b_ * float(cos_ev.sum())

    t_grid = np.linspace(0.0, T, ll_grid)
    sin_g = np.sin(omega * t_grid)
    cos_g = np.cos(omega * t_grid)
    lam_g = np.exp(beta0_ + a_ * sin_g + b_ * cos_g)
    integral = trapezoid(lam_g, t_grid)

    neg_ll = -(evt_term - integral)
    grad = np.array(
        [
            -(float(N) - integral),
            -(float(sin_ev.sum()) - trapezoid(lam_g * sin_g, t_grid)),
            -(float(cos_ev.sum()) - trapezoid(lam_g * cos_g, t_grid)),
        ]
    )
    return float(neg_ll), grad


def profile_logL_over_P(event_times, T, P_grid, ll_grid=4000):
    # Compute the profile log-likelihood of P by maximising over (beta0, a, b).
    #
    # Parameters
    # ----------
    # event_times : 1-D array, event arrival times (years).
    # T           : float, length of observation window (years).
    # P_grid      : 1-D array, candidate periods (years).
    # ll_grid     : int, number of trapezoid quadrature points.
    #
    # Returns
    # -------
    # profile    : 1-D array of shape (len(P_grid),), profile log-likelihood values.
    # best_params: list of length len(P_grid), each element is a (beta0, a, b) array
    # at the profile maximum for that P.
    #
    # Notes
    # -----
    # The log-likelihood is strictly concave in (beta0, a, b) for any fixed omega,
    # so L-BFGS-B with analytical gradients converges to the unique global maximum
    # without restarts.  Runtime is O(len(P_grid) * ll_grid).
    N = len(event_times)
    rough_b0 = float(np.log((N + 1e-9) / (T + 1e-9)))

    profile = np.empty(len(P_grid))
    best_params = []

    for j, P in enumerate(P_grid):
        omega = 2.0 * np.pi / P
        sin_ev = np.sin(omega * event_times)
        cos_ev = np.cos(omega * event_times)

        res = minimize(
            _neg_ll_and_grad,
            x0=np.array([rough_b0, 0.0, 0.0]),
            args=(sin_ev, cos_ev, N, T, omega, ll_grid),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        profile[j] = -res.fun
        best_params.append(res.x.copy())

    return profile, best_params


## Rolling-window period scan


def windowed_period_scan(
    event_times,
    years_abs,
    window_years: int = 100,
    step_years: int = 25,
    P_min: float = 7.0,
    P_max: float = 16.0,
    nP: int = 200,
    ll_grid: int = 2000,
):
    # Slide a fixed-width window over the aurora record and compute the profile
    # log-likelihood scan inside each window.
    #
    # Parameters
    # ----------
    # event_times  : 1-D array, relative times (years since t0).
    # years_abs    : 1-D array, absolute calendar years, same length as event_times.
    # window_years : Width of each window (years).
    # step_years   : Stride between successive windows (years).
    # P_min, P_max : Period search range (years).
    # nP           : Number of period grid points per window.
    # ll_grid      : Quadrature resolution for profile_logL_over_P.
    #
    # Returns
    # -------
    # List of dicts, one per window, each with keys:
    # 'start', 'end', 'n_events', 'P_grid', 'profile', 'P_hat'
    #
    # Interpretation
    # --------------
    # - If all windows agree on P_hat ≈ 11 yr, the period is stationary and the
    # multimodality is structural aliasing (many near-equal modes at different phi).
    # - If P_hat varies substantially across windows, or windows covering grand
    # solar minima show different dominant periods, the record is nonstationary
    # and a time-varying rate model should be considered.
    year_min = int(years_abs.min())
    year_max = int(years_abs.max())
    P_grid = np.linspace(P_min, P_max, nP)
    results = []

    for start in range(year_min, year_max - window_years + 1, step_years):
        end = start + window_years
        mask = (years_abs >= start) & (years_abs < end)
        ev = event_times[mask]
        if len(ev) < 10:
            continue

        # Rebase times to window interior
        t_w = ev - ev.min()
        T_w = float(t_w.max())
        if T_w < P_min:
            continue

        prof, _ = profile_logL_over_P(t_w, T_w, P_grid, ll_grid=ll_grid)
        i_hat = int(np.argmax(prof))
        results.append(
            {
                "start": start,
                "end": end,
                "n_events": int(len(ev)),
                "P_grid": P_grid.copy(),
                "profile": prof,
                "P_hat": float(P_grid[i_hat]),
            }
        )

    return results


## Plot windowed scan


def plot_windowed_scan(results, figsize=(14, 5)):
    # Two-panel figure for windowed_period_scan output.
    #
    # Left panel  : Profile logL curves per window (colour = window midpoint year).
    # Right panel : Profile MLE (P_hat) vs window midpoint year.
    # Marker size is proportional to number of events in the window.
    if not results:
        print("No windowed scan results to plot.")
        return None

    midpoints = np.array([0.5 * (r["start"] + r["end"]) for r in results])
    norm = plt.Normalize(midpoints.min(), midpoints.max())
    cmap = plt.get_cmap("plasma")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    for r, mid in zip(results, midpoints):
        prof_rel = r["profile"] - r["profile"].max()  # normalise for display
        ax_l.plot(r["P_grid"], prof_rel, color=cmap(norm(mid)), lw=1.2, alpha=0.75)

    ax_l.set_xlabel("Period P (years)")
    ax_l.set_ylabel("Profile logL (relative to window max)")
    ax_l.set_title("Profile logL per window")
    ax_l.set_xlim(results[0]["P_grid"][0], results[0]["P_grid"][-1])
    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_l,
        label="Window midpoint year",
    )

    phats = [r["P_hat"] for r in results]
    sizes = [max(15, r["n_events"]) for r in results]
    sc = ax_r.scatter(
        midpoints,
        phats,
        c=midpoints,
        cmap="plasma",
        s=sizes,
        vmin=midpoints.min(),
        vmax=midpoints.max(),
        zorder=3,
    )
    ax_r.axhline(11.0, ls="--", color="red", alpha=0.8, label="11 yr reference")
    ax_r.set_xlabel("Window midpoint year")
    ax_r.set_ylabel("P̂  (years)")
    ax_r.set_title("Profile MLE per window")
    ax_r.legend(frameon=False)
    plt.colorbar(sc, ax=ax_r, label="Window midpoint year")

    plt.show()
    return fig


## Simulation-based calibration


def run_sbc(
    n_sim: int,
    T_val: float,
    N_target: int,
    P_range: tuple = (9.0, 13.0),
    P_min: float = 7.0,
    P_max: float = 16.0,
    nP: int = 300,
    ll_grid: int = 2000,
    seed: int = 42,
):
    # Simulation-based calibration for the profile-logL period estimator.
    #
    # For each replicate:
    # 1. Draw P_true ~ Uniform(P_range), phi_true ~ Uniform(0, 2*pi),
    # beta1_true ~ Uniform(0.3, 1.0).
    # 2. Compute beta0_true to achieve expected N_target events over T_val.
    # 3. Simulate event times with cdf_inversion.
    # 4. Compute profile_logL_over_P on the simulated data.
    # 5. Record the CDF rank of P_true in the profile (i.e. fraction of P
    # values with lower profile logL than P_true).
    #
    # Parameters
    # ----------
    # n_sim    : Number of simulation replicates.
    # T_val    : Observation span (years) — use T_real from real_data_models.
    # N_target : Target expected event count — use len(event_times).
    # P_range  : (low, high) for Uniform prior on P_true.
    # P_min, P_max : Profile search range (years).
    # nP       : Number of period grid points.
    # ll_grid  : Quadrature resolution.
    # seed     : RNG seed.
    #
    # Returns
    # -------
    # List of dicts with keys:
    # 'sim', 'P_true', 'P_mle', 'n_events',
    # 'logL_cdf_rank_of_true', 'abs_error_yr'
    #
    # Interpretation
    # --------------
    # The rank histogram should be uniform if the profile estimator is calibrated.
    # - A spike near rank=1 means P_true is almost always at the global maximum:
    # the estimator is too confident (overfitting).
    # - A bimodal or flat rank distribution is expected under true structural
    # aliasing: P_true is just one of many near-equal peaks, so its rank varies.
    # - Concentration near rank=0 means the estimator is systematically biased.
    rng = np.random.default_rng(seed)
    P_grid = np.linspace(P_min, P_max, nP)
    records = []

    for i in range(n_sim):
        P_true = float(rng.uniform(*P_range))
        phi_true = float(rng.uniform(0.0, 2.0 * np.pi))
        beta1_true = float(rng.uniform(0.3, 1.0))
        omega_true = 2.0 * np.pi / P_true

        b0_true = _compute_beta0(beta1_true, omega_true, phi_true, T_val, N_target)
        t_ev, _ = cdf_inversion(
            b0_true, beta1_true, omega_true, phi_true, T_val, rng=rng
        )
        if len(t_ev) < 5:
            continue

        prof, _ = profile_logL_over_P(t_ev, T_val, P_grid, ll_grid=ll_grid)
        P_mle = float(P_grid[np.argmax(prof)])
        logL_at_true = float(np.interp(P_true, P_grid, prof))
        cdf_rank = float(np.mean(prof <= logL_at_true))

        records.append(
            {
                "sim": i,
                "P_true": P_true,
                "P_mle": P_mle,
                "n_events": len(t_ev),
                "logL_cdf_rank_of_true": cdf_rank,
                "abs_error_yr": abs(P_mle - P_true),
            }
        )

    return records


def plot_sbc(records, figsize=(11, 4)):
    # Two-panel SBC diagnostic figure.
    #
    # Left  : P_mle vs P_true scatter with 1:1 line.
    # Right : Histogram of the CDF rank of P_true in the profile logL.
    # A uniform histogram means the estimator is calibrated.
    P_true_v = [r["P_true"] for r in records]
    P_mle_v = [r["P_mle"] for r in records]
    ranks = [r["logL_cdf_rank_of_true"] for r in records]

    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax_s.scatter(P_true_v, P_mle_v, alpha=0.6, s=30, color="steelblue")
    lo = min(min(P_true_v), min(P_mle_v)) - 0.3
    hi = max(max(P_true_v), max(P_mle_v)) + 0.3
    ax_s.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax_s.set_xlabel("P_true (years)")
    ax_s.set_ylabel("P_mle — profile MLE (years)")
    ax_s.set_title("SBC: period recovery")

    n_bins = 20
    ax_r.hist(ranks, bins=n_bins, range=(0, 1), color="steelblue", edgecolor="white")
    ax_r.axhline(len(ranks) / n_bins, ls="--", color="red", label="Expected if uniform")
    ax_r.set_xlabel("CDF rank of P_true in profile logL")
    ax_r.set_ylabel("Count")
    ax_r.set_title("SBC: rank histogram\n(uniform → calibrated estimator)")
    ax_r.legend(frameon=False)

    plt.show()
    return fig


## Posterior predictive checks split by epoch


def plot_ppc_epoch(
    flat_ab_samples,
    event_times,
    years_abs,
    T,
    epoch_edges=None,
    n_draws: int = 200,
    seed: int = 0,
):
    # Posterior predictive checks split into time epochs.
    #
    # This is the key diagnostic for *nonstationarity*: if the posterior intensity
    # envelope consistently over- or under-covers the empirical rate in a specific
    # epoch (e.g. covering a grand solar minimum), the stationary sinusoidal model
    # is misspecified for that period.
    #
    # Parameters
    # ----------
    # flat_ab_samples : (n_samples, 4) array with columns [beta0, a, b, logP].
    # Use the flat chain from the (a, b) MCMC defined below.
    # event_times     : 1-D array, relative event times (years since t0).
    # years_abs       : 1-D array, absolute calendar years.
    # T               : float, total observation span (years).
    # epoch_edges     : List of absolute year boundaries defining epochs.
    # Defaults to quartiles of years_abs.
    # n_draws         : Number of posterior draws to overlay.
    # seed            : RNG seed for draw selection.
    #
    # Returns
    # -------
    # matplotlib Figure.
    rng = np.random.default_rng(seed)

    if epoch_edges is None:
        epoch_edges = list(np.quantile(years_abs, [0.0, 0.25, 0.5, 0.75, 1.0]))

    t0_abs = float(years_abs.min())
    n_epochs = len(epoch_edges) - 1
    idx_draws = rng.integers(0, len(flat_ab_samples), size=n_draws)

    fig, axes = plt.subplots(
        1, n_epochs, figsize=(5 * n_epochs, 4), constrained_layout=True
    )
    if n_epochs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        y_lo = float(epoch_edges[i])
        y_hi = float(epoch_edges[i + 1])
        mask = (years_abs >= y_lo) & (years_abs < y_hi)
        ev = event_times[mask]
        t_lo_rel = y_lo - t0_abs
        t_hi_rel = y_hi - t0_abs

        # Empirical rate
        counts, edges = np.histogram(ev, bins=30, range=(t_lo_rel, t_hi_rel))
        bw = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(
            centers,
            counts / bw,
            where="mid",
            color="black",
            lw=2,
            label="Empirical",
            zorder=3,
        )

        # Posterior predictive draws
        t_plot = np.linspace(t_lo_rel, t_hi_rel, 500)
        for k in idx_draws:
            b0, a, b, logP = flat_ab_samples[k]
            omega = 2.0 * np.pi / np.exp(logP)
            lam = np.exp(b0 + a * np.sin(omega * t_plot) + b * np.cos(omega * t_plot))
            ax.plot(t_plot, lam, alpha=0.08, lw=0.8, color="steelblue")

        ax.set_title(f"{int(y_lo)}–{int(y_hi)}\n(N = {len(ev)})")
        ax.set_xlabel("Relative time (years)")
        if i == 0:
            ax.set_ylabel("Rate (events / year)")
            ax.legend(frameon=False)

    fig.suptitle("Posterior predictive checks by epoch", fontsize=13)
    plt.show()
    return fig


## Compare 1-harmonic vs 2-harmonic profile logL


def _neg_ll_and_grad_2harm(
    params, sin_ev, cos_ev, sin2_ev, cos2_ev, N, T, omega, ll_grid
):
    # Negative NHPP log-likelihood and gradient for the 2-harmonic model
    beta0_, a1_, b1_, a2_, b2_ = params

    evt_term = (
        float(N) * beta0_
        + a1_ * float(sin_ev.sum())
        + b1_ * float(cos_ev.sum())
        + a2_ * float(sin2_ev.sum())
        + b2_ * float(cos2_ev.sum())
    )

    t_grid = np.linspace(0.0, T, ll_grid)
    sin_g = np.sin(omega * t_grid)
    cos_g = np.cos(omega * t_grid)
    sin2_g = np.sin(2.0 * omega * t_grid)
    cos2_g = np.cos(2.0 * omega * t_grid)
    lam_g = np.exp(beta0_ + a1_ * sin_g + b1_ * cos_g + a2_ * sin2_g + b2_ * cos2_g)
    integral = trapezoid(lam_g, t_grid)

    neg_ll = -(evt_term - integral)
    grad = np.array(
        [
            -(float(N) - integral),
            -(float(sin_ev.sum()) - trapezoid(lam_g * sin_g, t_grid)),
            -(float(cos_ev.sum()) - trapezoid(lam_g * cos_g, t_grid)),
            -(float(sin2_ev.sum()) - trapezoid(lam_g * sin2_g, t_grid)),
            -(float(cos2_ev.sum()) - trapezoid(lam_g * cos2_g, t_grid)),
        ]
    )
    return float(neg_ll), grad


def profile_logL_2harm_over_P(event_times, T, P_grid, ll_grid=4000):
    # Profile log-likelihood of P for the 2-harmonic model:
    # log lambda(t) = beta0 + a1*sin(wt) + b1*cos(wt) + a2*sin(2wt) + b2*cos(2wt)
    # Maximised over (beta0, a1, b1, a2, b2) for each P.
    #
    # Returns
    # -------
    # profile    : 1-D array of profile log-likelihood values.
    # best_params: list of (beta0, a1, b1, a2, b2) arrays.
    N = len(event_times)
    rough_b0 = float(np.log((N + 1e-9) / (T + 1e-9)))
    profile = np.empty(len(P_grid))
    best_params = []

    for j, P in enumerate(P_grid):
        omega = 2.0 * np.pi / P
        sin_ev = np.sin(omega * event_times)
        cos_ev = np.cos(omega * event_times)
        sin2_ev = np.sin(2.0 * omega * event_times)
        cos2_ev = np.cos(2.0 * omega * event_times)

        res = minimize(
            _neg_ll_and_grad_2harm,
            x0=np.array([rough_b0, 0.0, 0.0, 0.0, 0.0]),
            args=(sin_ev, cos_ev, sin2_ev, cos2_ev, N, T, omega, ll_grid),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        profile[j] = -res.fun
        best_params.append(res.x.copy())

    return profile, best_params


def plot_profile_comparison(event_times, T, P_grid, ll_grid=4000, figsize=(10, 4)):
    # Plot the 1-harmonic and 2-harmonic profile log-likelihoods side-by-side on
    # the same axes, normalised to their respective maxima.
    #
    # This is the key diagnostic for the question:
    # "Is the period posterior multimodal because of a modelling choice
    # (too few harmonics), or because of structural aliasing?"
    #
    # If both curves show the same pattern of peaks, the aliasing is structural.
    # If the 2-harmonic profile has a single cleaner peak, the 1-harmonic model
    # is missing nonsinusoidal structure in the solar cycle shape.
    print("Computing 1-harmonic profile logL...")
    prof1, _ = profile_logL_over_P(event_times, T, P_grid, ll_grid=ll_grid)
    print("Computing 2-harmonic profile logL...")
    prof2, _ = profile_logL_2harm_over_P(event_times, T, P_grid, ll_grid=ll_grid)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(P_grid, prof1 - prof1.max(), lw=2, label="1-harmonic", color="steelblue")
    ax.plot(
        P_grid,
        prof2 - prof2.max(),
        lw=2,
        label="2-harmonic",
        color="darkorange",
        ls="--",
    )
    ax.axvline(11.0, ls=":", color="red", alpha=0.8, label="11 yr")
    ax.set_xlabel("Period P (years)")
    ax.set_ylabel("Profile logL (relative to model max)")
    ax.set_title("Profile logL: 1-harmonic vs 2-harmonic NHPP")
    ax.legend(frameon=False)
    ax.set_xlim(P_grid[0], P_grid[-1])
    plt.tight_layout()
    plt.show()
    return fig, prof1, prof2
