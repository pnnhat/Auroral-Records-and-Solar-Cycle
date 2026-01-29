import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

#! Step 0 Injection Recovery Method
# 0.1 Choose the true parameters for the injection
T = 354.0
P_true = 11.0
omega_true = 2 * np.pi / P_true
beta1_true = 0.6
phi_true = np.pi / 4
N_target = 68


def solve_beta0(beta1, omega, phi, T, N_target, n_grid=20000):
    t_grid = np.linspace(0.0, T, n_grid)
    base = np.exp(beta1 * np.sin(omega * t_grid + phi))
    integral_base = np.trapz(base, t_grid)
    beta0 = np.log(N_target / integral_base)
    return beta0


beta0_true = solve_beta0(beta1_true, omega_true, phi_true, T, N_target)

theta_true = {
    "beta0": beta0_true,
    "beta1": beta1_true,
    "omega": omega_true,
    "period": P_true,
    "phi": phi_true,
    "T": T,
    "N_target": N_target,
}

theta_true


# 0.2 Simulate data from the true parameters using CDF inversion
def lambda_func(t, beta0, beta1, omega, phi):
    t = np.asarray(t, dtype=float)
    return np.exp(beta0 + beta1 * np.sin(omega * t + phi))


def cdf_inversion(theta_true, n_grid=20000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    beta0 = theta_true["beta0"]
    beta1 = theta_true["beta1"]
    omega = theta_true["omega"]
    phi = theta_true["phi"]
    T = float(theta_true["T"])

    t_grid = np.linspace(0.0, T, n_grid)

    lambda_grid = lambda_func(t_grid, beta0, beta1, omega, phi)

    dt = np.diff(t_grid)

    incr = 0.5 * (lambda_grid[:-1] + lambda_grid[1:]) * dt
    Lambda_grid = np.concatenate([[0.0], np.cumsum(incr)])

    Lambda_T = Lambda_grid[-1]  # expected total events

    N = rng.poisson(Lambda_T)

    if N == 0:
        return np.array([], dtype=float)
    u = rng.random(N) * Lambda_T
    # invert the CDF using interpolation
    inv_cdf = interp1d(
        Lambda_grid, t_grid, kind="linear", bounds_error=False, fill_value=(0.0, T)
    )

    t_events = inv_cdf(u)
    t_events.sort()
    return t_events


t_sim = cdf_inversion(theta_true, n_grid=20000)

print("Simulated N:", len(t_sim))
print("First few events:", t_sim[:10])
print("Last event time:", t_sim[-1] if len(t_sim) else None)


# 0.3 Period scan and Maximum Likelihood Estimation optimization
def log_likelihood(beta0, beta1, omega, phi, t_events, T, n_grid=4000):
    # Term 1: sum log λ(t_i)
    term_events = np.sum(beta0 + beta1 * np.sin(omega * t_events + phi))

    # Term 2: integral of λ(t) over [0, T]
    t_grid = np.linspace(0.0, T, n_grid)
    lam_grid = np.exp(beta0 + beta1 * np.sin(omega * t_grid + phi))
    integral = np.trapz(lam_grid, t_grid)

    return term_events - integral


# Maximamize over (beta0, beta1, phi) for fixed omega
def negative_log_likelihood_fixed_omega(params, omega, t_events, T, n_grid=4000):
    n = len(t_events)
    beta0_init = np.log((n + 1e-9) / (T + 1e-9))
    x0 = np.array([beta0_init, 0.3, 0.0])

    # make it stable by bounds
    bounds = [
        (None, None),
        (-3.0, 3.0),
        (-np.pi, np.pi),
    ]

    def objective(x):
        beta0, beta1, phi = x
        return -log_likelihood(beta0, beta1, omega, phi, t_events, T, n_grid=n_grid)

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    beta0_hat, beta1_hat, phi_hat = res.x
    logL_hat = -res.fun

    return {
        "omega": omega,
        "period": 2 * np.pi / omega,
        "beta0": beta0_hat,
        "beta1": beta1_hat,
        "phi": phi_hat,
        "logL": logL_hat,
        "success": res.success,
        "message": res.message,
    }


# Pick best omega
def scan_fit(t_events, T, P_min=7.0, P_max=16.0, n_P=200, n_grid=4000):
    periods = np.linspace(P_min, P_max, n_P)
    omegas = 2 * np.pi / periods

    fits = []
    for omega in omegas:
        fits.append(
            negative_log_likelihood_fixed_omega(None, omega, t_events, T, n_grid=n_grid)
        )

    best = max(fits, key=lambda d: d["logL"])
    return fits, best


# One injection
t_sim = cdf_inversion(theta_true, n_grid=20000)
fits, best = scan_fit(t_sim, theta_true["T"], P_min=7, P_max=16, n_P=250)
print("True period:", theta_true["period"])
print("Recovered period:", best["period"])
print("True beta1:", theta_true["beta1"], "Recovered beta1:", best["beta1"])
print("True phi:", theta_true["phi"], "Recovered phi:", best["phi"])


# 0.4 Run many trials and compare inferred vs true parameters
def wrap_phase_diff(phi_hat, phi_true):
    d = phi_hat - phi_true
    return (d + np.pi) % (2 * np.pi) - np.pi


def injection_recovery_trials(
    theta_true,
    K=200,
    seed=0,
    sim_grid=20000,
    P_min=7.0,
    P_max=16.0,
    n_P=250,
    ll_grid=4000,
):
    rng = np.random.default_rng(seed)

    true_period = theta_true["period"]
    true_beta1 = theta_true["beta1"]
    true_phi = theta_true["phi"]
    T = float(theta_true["T"])

    recovered_periods = []
    recovered_beta1s = []
    recovered_phis = []
    recovered_logL = []
    success_flags = []

    for k in range(K):
        t_sim = cdf_inversion(theta_true, n_grid=sim_grid, rng=rng)

        # if a draw gives zero events (rare if N_target ~ 68), handle gracefully
        if len(t_sim) == 0:
            recovered_periods.append(np.nan)
            recovered_beta1s.append(np.nan)
            recovered_phis.append(np.nan)
            recovered_logL.append(np.nan)
            success_flags.append(False)
            continue

        fits, best = scan_fit(
            t_sim, T, P_min=P_min, P_max=P_max, n_P=n_P, n_grid=ll_grid
        )

        recovered_periods.append(best["period"])
        recovered_beta1s.append(best["beta1"])
        recovered_phis.append(best["phi"])
        recovered_logL.append(best["logL"])
        success_flags.append(bool(best["success"]))

    recovered_periods = np.array(recovered_periods, dtype=float)
    recovered_beta1s = np.array(recovered_beta1s, dtype=float)
    recovered_phis = np.array(recovered_phis, dtype=float)
    recovered_logL = np.array(recovered_logL, dtype=float)
    success_flags = np.array(success_flags, dtype=bool)

    # Keep ony trials with finite recovered parameters
    ok = (
        np.isfinite(recovered_periods)
        & np.isfinite(recovered_beta1s)
        & np.isfinite(recovered_phis)
    )

    period = recovered_periods[ok]
    b1 = recovered_beta1s[ok]
    ph = recovered_phis[ok]

    period_err = period - true_period
    b1_err = b1 - true_beta1
    phi_err = np.array([wrap_phase_diff(p, true_phi) for p in ph], dtype=float)

    within_1yr = np.mean(np.abs(period_err) <= 1.0) if len(period_err) else np.nan
    within_2yr = np.mean(np.abs(period_err) <= 2.0) if len(period_err) else np.nan

    summary = {
        "K": K,
        "valid_trials": int(ok.sum()),
        "optimizer_success_rate": float(np.mean(success_flags[ok]))
        if ok.sum()
        else np.nan,
        "true_period": true_period,
        "recovered_period_mean": float(np.mean(period)) if len(period) else np.nan,
        "recovered_period_std": float(np.std(period, ddof=1))
        if len(period) > 1
        else np.nan,
        "period_bias": float(np.mean(period_err)) if len(period_err) else np.nan,
        "period_rmse": float(np.sqrt(np.mean(period_err**2)))
        if len(period_err)
        else np.nan,
        "period_within_1yr_rate": float(within_1yr),
        "period_within_2yr_rate": float(within_2yr),
        "true_beta1": true_beta1,
        "recovered_beta1_mean": float(np.mean(b1)) if len(b1) else np.nan,
        "recovered_beta1_std": float(np.std(b1, ddof=1)) if len(b1) > 1 else np.nan,
        "beta1_bias": float(np.mean(b1_err)) if len(b1_err) else np.nan,
        "beta1_rmse": float(np.sqrt(np.mean(b1_err**2))) if len(b1_err) else np.nan,
        "true_phi": true_phi,
        "phi_error_mean": float(np.mean(phi_err)) if len(phi_err) else np.nan,
        "phi_error_std": float(np.std(phi_err, ddof=1)) if len(phi_err) > 1 else np.nan,
    }

    outputs = {
        "recovered_periods": recovered_periods,
        "recovered_beta1s": recovered_beta1s,
        "recovered_phis": recovered_phis,
        "recovered_logL": recovered_logL,
        "success_flags": success_flags,
        "ok_mask": ok,
        "errors": {
            "period_err": period_err,
            "beta1_err": b1_err,
            "phi_err": phi_err,
        },
        "summary": summary,
    }

    return outputs


out = injection_recovery_trials(theta_true, K=500, seed=42)
out["summary"]


ok = out["ok_mask"]
period = out["recovered_periods"][ok]
b1 = out["recovered_beta1s"][ok]
ph = out["recovered_phis"][ok]
trueP = theta_true["period"]
trueB1 = theta_true["beta1"]
truePhi = theta_true["phi"]

# Histogram of recovered periods
plt.figure()
plt.hist(period, bins=30, density=True, alpha=0.6, color="black")
plt.axvline(trueP, linestyle="--", color="red")
plt.xlabel("Recovered period")
plt.ylabel("Density")

# Hisogram of recovered beta1
plt.figure()
plt.hist(b1, bins=30, density=True, alpha=0.6, color="black")
plt.axvline(trueB1, linestyle="--", color="red")
plt.xlabel("Recovered beta1")
plt.ylabel("Density")

# Histogram of phase errors, can we locate where the cycle events tend to cluster?
phi_err = out["errors"]["phi_err"]
plt.figure()
plt.hist(phi_err, bins=30, density=True, alpha=0.6, color="black")
plt.axvline(0.0, linestyle="--", color="red")
plt.xlabel("Phase error")
plt.ylabel("Density")
plt.show()


# Log_likeli vs period for 1 trial
def logL_vs_period(fits, true_period=None):
    periods = np.array([f["period"] for f in fits])
    logL = np.array([f["logL"] for f in fits])

    plt.figure(figsize=(7, 4))
    plt.plot(periods, logL, lw=2)
    plt.xlabel("Period (years)")
    plt.ylabel("Log-likelihood")
    plt.title("NHPP log-likelihood vs period")
    plt.axvline(true_period, color="red", ls="--", label="True period")

    best_idx = np.argmax(logL)
    plt.axvline(periods[best_idx], color="black", ls=":", label="Best-fit")
    plt.legend()
    plt.tight_layout()
    plt.show()


logL_vs_period(fits, true_period=theta_true["period"])


# Overlay several single-trial likelihood curves
def logL_constant_rate(t_events, T):
    N = len(t_events)
    rate_hat = N / T
    beta0_hat = np.log(rate_hat)
    logL0 = N * beta0_hat - rate_hat * T
    return logL0


N_show = 50
seed_overlay = 123

rng = np.random.default_rng(seed_overlay)

plt.figure(figsize=(8, 5))

for i in range(N_show):
    t_sim = cdf_inversion(theta_true, n_grid=20000, rng=rng)

    fits, best = scan_fit(
        t_sim, theta_true["T"], P_min=7.0, P_max=16.0, n_P=250, n_grid=4000
    )

    periods = np.array([f["period"] for f in fits], dtype=float)
    logL = np.array([f["logL"] for f in fits], dtype=float)

    # Convert to ΔlogL vs constant-rate
    logL0 = logL_constant_rate(t_sim, theta_true["T"])
    delta_logL = logL - logL0

    plt.plot(periods, delta_logL, alpha=0.5)

plt.axvline(theta_true["period"], color="red", ls="--", label="True period")
plt.axhline(0.0, color="gray", ls="--")

plt.xlabel("Period (years)")
plt.ylabel("ΔlogL vs constant rate")
plt.title(f"Single-trial likelihood curves (N={N_show} random injections)")
plt.legend()
plt.tight_layout()
plt.show()


# Log-likelihood difference vs constant rate model
def delta_logL_vs_period(fits, t_events, T, true_period=None):
    periods = np.array([f["period"] for f in fits])
    logLs = np.array([f["logL"] for f in fits])

    logL0 = logL_constant_rate(t_events, T)
    delta_logL = logLs - logL0

    plt.figure(figsize=(7, 4))
    plt.plot(periods, delta_logL, lw=2)
    plt.axhline(0.0, color="gray", ls="--")

    plt.axvline(true_period, color="red", ls="--", label="True period")

    best_idx = np.argmax(delta_logL)
    plt.axvline(periods[best_idx], color="black", ls=":", label="Best-fit")

    plt.xlabel("Period (years)")
    plt.ylabel(r"$\Delta \log L$  (vs constant rate)")
    plt.title("Log-likelihood over constant-rate NHPP")
    plt.legend()
    plt.tight_layout()
    plt.show()


delta_logL_vs_period(fits, t_sim, theta_true["T"], true_period=theta_true["period"])


# Averaged likelihood curves over K trials
def averaged_likelihood_curves(
    theta_true,
    K,
    seed=0,
    sim_grid=20000,
    P_min=7.0,
    P_max=16.0,
    n_P=250,
    ll_grid=4000,
    use_delta_vs_const=True,
    align_by_max=False,
):
    rng = np.random.default_rng(seed)
    T = float(theta_true["T"])

    periods = np.linspace(P_min, P_max, n_P)
    curves = np.empty((K, n_P), dtype=float)

    for k in range(K):
        t_sim = cdf_inversion(theta_true, n_grid=sim_grid, rng=rng)
        fits, best = scan_fit(
            t_sim, T, P_min=P_min, P_max=P_max, n_P=n_P, n_grid=ll_grid
        )
        logL = np.array([f["logL"] for f in fits], dtype=float)

        if use_delta_vs_const:
            logL0 = logL_constant_rate(t_sim, T)
            curve = logL - logL0
        else:
            curve = logL
        if align_by_max:
            curve = curve - np.max(curve)

        curves[k, :] = curve

    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0, ddof=1) if K > 1 else np.zeros_like(mean_curve)

    return periods, mean_curve, std_curve, curves


Ks = [50, 100, 200, 500]

plt.figure(figsize=(8, 5))

for K in Ks:
    periods, mean_curve, std_curve, curves = averaged_likelihood_curves(
        theta_true, K=K, seed=42, use_delta_vs_const=True, align_by_max=False
    )
    plt.plot(periods, mean_curve, label=f"K={K}")

plt.axvline(theta_true["period"], ls="--", color="red", label="True period")
plt.axhline(0.0, ls="--", color="gray")
plt.xlabel("Period (years)")
plt.ylabel("Averaged ΔlogL vs constant rate")
plt.title("Averaged likelihood vs period")
plt.legend()
plt.tight_layout()
plt.show()


#! Step 1: Fit on the real data with the inference pipeline
test_korea = pd.read_excel("data/Korean_Aurora_Grades_918_1392.xlsx")

years_real = test_korea["Year"].astype(int).values

t0_real = years_real.min()
t_real = (years_real - t0_real).astype(float)
T_real = float(t_real.max())

fits_real, best_real = scan_fit(
    t_real, T_real, P_min=7.0, P_max=16.0, n_P=250, n_grid=4000
)

print("Best-fit period:", best_real["period"])
print("Best-fit omega:", best_real["omega"])
print("Best-fit beta0:", best_real["beta0"])
print("Best-fit beta1:", best_real["beta1"])
print("Best-fit phi:", best_real["phi"])
print("Best-fit logL:", best_real["logL"])

# Plot
logL_vs_period(fits_real, true_period=11.0)
delta_logL_vs_period(fits_real, t_real, T_real, true_period=11.0)

# Print more
periods = np.array([f["period"] for f in fits_real])
logLs = np.array([f["logL"] for f in fits_real])

logL0 = logL_constant_rate(t_real, T_real)
delta = logLs - logL0

best_idx = np.argmax(delta)
best_period = periods[best_idx]
best_delta = delta[best_idx]

idx_11 = np.argmin(np.abs(periods - 11.0))
delta_11 = delta[idx_11]

print("Best period:", best_period)
print("Best ΔlogL:", best_delta)
print("ΔlogL at 11y:", delta_11)


T = T_real
N_target = len(t_real)

P_true = 11.0
omega_true = 2 * np.pi / P_true
beta1_true = 0.6
phi_true = np.pi / 4

beta0_true = solve_beta0(beta1_true, omega_true, phi_true, T, N_target)

theta_true = {
    "beta0": beta0_true,
    "beta1": beta1_true,
    "omega": omega_true,
    "period": P_true,
    "phi": phi_true,
    "T": T,
    "N_target": N_target,
}

ok = out["ok_mask"]
period_rec = out["recovered_periods"][ok]

plt.figure()
plt.hist(period_rec, bins=30, density=True, alpha=0.6, color="black")
plt.axvline(
    theta_true["period"], linestyle="--", color="red", label="Injected true P=11"
)
plt.axvline(best_real["period"], linestyle=":", color="blue", label="Real-data best P")
plt.xlabel("Recovered period (years)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
