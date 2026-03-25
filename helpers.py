import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import emcee


def beta0(beta1, omega, phi, T, N_target, n_grid=20000):
    t_grid = np.linspace(0.0, T, n_grid)
    base = np.exp(beta1 * np.sin(omega * t_grid + phi))
    integral_base = trapezoid(base, t_grid)
    beta0 = np.log(N_target / integral_base)
    return beta0


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


def log_likelihood_params(beta0, beta1, omega, phi, event_times, T, n_grid=4000):
    if len(event_times) == 0:
        t_grid = np.linspace(0.0, T, n_grid)
        lam_grid = np.exp(beta0 + beta1 * np.sin(omega * t_grid + phi))
        return -trapezoid(lam_grid, t_grid)

    term_events = np.sum(beta0 + beta1 * np.sin(omega * event_times + phi))
    t_grid = np.linspace(0.0, T, n_grid)
    lam_grid = np.exp(beta0 + beta1 * np.sin(omega * t_grid + phi))
    integral = trapezoid(lam_grid, t_grid)
    return term_events - integral


def scan_period(prefix, beta0, beta1, phi, T, P_min=7.0, P_max=16.0, n_P=900, ll_grid=2500):
    periods = np.linspace(P_min, P_max, n_P)
    logLs = np.array(
        [
            log_likelihood_params(beta0, beta1, 2 * np.pi / P, phi, prefix, T, n_grid=ll_grid)
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


def beta0_grid(prefix, T, width=3.0, n=601):
    n_evt = len(prefix)
    rough = np.log((n_evt + 1e-9) / (T + 1e-9))
    return np.linspace(rough - width, rough + width, n)


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


def log_likelihood_1harm(event_times, T, N_target, omega, a1_grid, b1_grid, ll_grid=4000):
    if len(event_times) == 0:
        return -np.inf, (np.nan, np.nan, np.nan)

    s1 = np.sin(omega * event_times)
    c1 = np.cos(omega * event_times)

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


def log_likelihood_2harm(event_times, T, N_target, omega, a1_grid, b1_grid, a2_grid, b2_grid, ll_grid=4000):
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


def success_curve(jv):
    N_max = int(np.nanmax(jv))
    Ns = np.arange(1, N_max + 1)
    success_prob = np.array([np.mean(jv <= N) for N in Ns])
    return Ns, success_prob


def log_prior(theta, event_times, T, P_min=7.0, P_max=16.0):
    beta0_, beta1_, logP_, logphi_ = theta
    P_ = np.exp(logP_)
    phi_ = np.exp(logphi_)

    if not (P_min < P_ < P_max):
        return -np.inf
    if not (0.0 < beta1_ < 3.0):
        return -np.inf
    if not (0.0 < phi_ < 2.0 * np.pi):
        return -np.inf
    if not (-20.0 < beta0_ < 20.0):
        return -np.inf
    rough_rate = np.log((len(event_times) + 1e-9) / (T + 1e-9))

    lp = 0.0
    lp += -0.5 * ((beta1_ - 0.0) / 1.0) ** 2
    lp += -0.5 * ((beta0_ - rough_rate) / 3.0) ** 2
    lp += logphi_
    return lp


def log_probability(theta, event_times, T, P_min=7.0, P_max=16.0, ll_grid=4000):
    beta0_, beta1_, logP_, logphi_ = theta

    lp = log_prior(theta, event_times, T, P_min=P_min, P_max=P_max)
    if not np.isfinite(lp):
        return -np.inf

    P_ = np.exp(logP_)
    omega_ = 2.0 * np.pi / P_
    phi_ = np.exp(logphi_)

    ll = log_likelihood_params(beta0_, beta1_, omega_, phi_, event_times, T, n_grid=ll_grid)
    return lp + ll


def q16_50_84(x):
    return np.percentile(x, [16, 50, 84])


def run_emcee_sampler(p0, event_times, T, P_grid_min=7.0, P_grid_max=16.0, ll_grid=4000, nburn=50000, nsteps=30000):
    ndim = p0.shape[1]
    nwalkers = p0.shape[0]
    rng = np.random.default_rng(2200)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(event_times, T),
        kwargs={"P_min": P_grid_min, "P_max": P_grid_max, "ll_grid": ll_grid},
    )
    state = sampler.run_mcmc(p0, nburn, progress=True, rstate0=rng)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True, rstate0=rng)
    return sampler


def q16_50_84(x):
    return np.percentile(np.asarray(x), [16, 50, 84])


# -----------------------------
# Semiparametric basis functions
# -----------------------------
def make_rbf_basis(t, T, n_knots=8, width_scale=1.25):
    """
    Low-rank smooth basis on [0, T] using Gaussian RBFs.
    Returns a design matrix of shape (len(t), n_knots).

    We center each column so the spline part has mean ~ 0 over the grid,
    which helps identifiability with beta0.
    """
    t = np.asarray(t, dtype=float)

    centers = np.linspace(0.0, T, n_knots)
    if n_knots > 1:
        spacing = centers[1] - centers[0]
    else:
        spacing = T if T > 0 else 1.0

    width = width_scale * spacing
    B = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / width) ** 2)

    # center columns for identifiability with beta0
    B = B - B.mean(axis=0, keepdims=True)
    return B, centers, width


def phi_from_raw(raw_phi):
    """
    Map unconstrained real number -> (0, 2*pi)
    """
    return 2.0 * np.pi * expit(raw_phi)


def semiparam_log_rate(t, beta0, beta1, omega, raw_phi, gamma, basis_info):
    """
    log lambda(t) = beta0 + beta1*sin(omega t + phi) + B(t) @ gamma
    """
    B_eval, _, _ = basis_info
    phi = phi_from_raw(raw_phi)
    return beta0 + beta1 * np.sin(omega * t + phi) + B_eval @ gamma


def semiparam_lambda(t, beta0, beta1, omega, raw_phi, gamma, basis_info):
    return np.exp(semiparam_log_rate(t, beta0, beta1, omega, raw_phi, gamma, basis_info))


def build_semiparam_basis(T, n_knots=8, grid_size=4000, width_scale=1.25):
    """
    Build basis once on a dense integration grid and return an evaluator.
    """
    t_grid = np.linspace(0.0, T, grid_size)
    B_grid, centers, width = make_rbf_basis(t_grid, T, n_knots=n_knots, width_scale=width_scale)

    def eval_basis(t):
        t = np.asarray(t, dtype=float)
        B = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / width) ** 2)
        B = B - B_grid.mean(axis=0, keepdims=True)
        return B

    return t_grid, B_grid, centers, width, eval_basis


def log_likelihood_semiparametric(theta, event_times, T, basis_bundle):
    """
    theta = [beta0, beta1, logP, raw_phi, gamma_1, ..., gamma_K]
    """
    beta0 = theta[0]
    beta1 = theta[1]
    logP  = theta[2]
    raw_phi = theta[3]
    gamma = theta[4:]

    P = np.exp(logP)
    omega = 2.0 * np.pi / P

    t_grid, B_grid, centers, width, eval_basis = basis_bundle

    if len(event_times) == 0:
        log_lam_grid = beta0 + beta1 * np.sin(omega * t_grid + phi_from_raw(raw_phi)) + B_grid @ gamma
        lam_grid = np.exp(log_lam_grid)
        return -trapezoid(lam_grid, t_grid)

    B_evt = eval_basis(event_times)
    phi = phi_from_raw(raw_phi)

    log_lam_evt = beta0 + beta1 * np.sin(omega * event_times + phi) + B_evt @ gamma
    term_events = np.sum(log_lam_evt)

    log_lam_grid = beta0 + beta1 * np.sin(omega * t_grid + phi) + B_grid @ gamma
    lam_grid = np.exp(log_lam_grid)
    integral = trapezoid(lam_grid, t_grid)

    return term_events - integral


def log_prior_semiparametric(
    theta,
    P_min,
    P_max,
    sigma_beta0=5.0,
    sigma_beta1=1.0,
    sigma_gamma=0.35
):
    """
    Weakly informative priors + shrinkage on spline coefficients gamma.
    Shrinkage is what keeps the semiparametric part smooth-ish / controlled.
    """
    beta0 = theta[0]
    beta1 = theta[1]
    logP = theta[2]
    raw_phi = theta[3]
    gamma = theta[4:]

    P = np.exp(logP)

    if not (P_min < P < P_max):
        return -np.inf

    if abs(beta1) > 3.0:
        return -np.inf

    # weak Gaussian priors
    lp = 0.0
    lp += -0.5 * (beta0 / sigma_beta0) ** 2
    lp += -0.5 * (beta1 / sigma_beta1) ** 2

    # raw_phi kept unconstrained; logistic map handles support
    lp += -0.5 * (raw_phi / 3.0) ** 2

    # ridge prior on spline coefficients
    lp += -0.5 * np.sum((gamma / sigma_gamma) ** 2)

    return lp


def log_probability_semiparametric(
    theta,
    event_times,
    T,
    basis_bundle,
    P_min=7.0,
    P_max=16.0,
    sigma_beta0=5.0,
    sigma_beta1=1.0,
    sigma_gamma=0.35,
):
    lp = log_prior_semiparametric(
        theta,
        P_min=P_min,
        P_max=P_max,
        sigma_beta0=sigma_beta0,
        sigma_beta1=sigma_beta1,
        sigma_gamma=sigma_gamma,
    )
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood_semiparametric(theta, event_times, T, basis_bundle)
    return lp + ll