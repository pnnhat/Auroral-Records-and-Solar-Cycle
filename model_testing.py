import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

import sympy as sp

t0, t_max = 0, 10
t = np.linspace(t0, t_max, 1000)

# Define a symbolic time variable
t_sym = sp.symbols("t", real=True)
period = sp.symbols("p", positive=True)
phase = sp.symbols("phi", real=True)
baseline = sp.symbols("b", positive=True)
amplitude = sp.symbols("a", positive=True)

# Define a concrete SymPy rate expression (edit as needed)
# Example: sin(t) + t^2
rate_expr = baseline + amplitude * sp.sin(2 * sp.pi * t_sym / (period * 2) + phase) ** 2
vals = {baseline: 0, amplitude: 5000, period: 0.4, phase: np.pi / 4}

# Create a callable function from the SymPy expression
rate_function = sp.lambdify(
    (t_sym, baseline, amplitude, period, phase), rate_expr, "numpy"
)
cdf_expr = sp.integrate(rate_expr, (t_sym, 0, t_sym))
cdf_function = sp.lambdify(
    (t_sym, baseline, amplitude, period, phase), cdf_expr, "numpy"
)

# Evaluate on the existing variable `t` (numpy array or scalar)
rate_values = rate_function(t, *vals.values())
cdf_values = cdf_function(t, *vals.values())

plt.plot(t, rate_values)
plt.xlim(t.min(), t.max())
plt.ylim(0, np.max(rate_values) * 1.1)

plt.plot(t, cdf_values / cdf_values.max())
plt.xlim(t.min(), t.max())
plt.ylim(0, 1.0)

# poisson_draws
N_draws = poisson.rvs(mu=cdf_values.max())
print(f"Number of events: {N_draws}")


# interpolate the cdf to invert it
from scipy.interpolate import interp1d

uniform_draws = np.random.rand(N_draws)

rescaled_draws = interp1d(cdf_values / cdf_values.max(), t, kind="linear")(
    uniform_draws
)
plt.hist(rescaled_draws, bins=200, density=True, alpha=0.5)
plt.xlim(t.min(), t.max())


# now let's infer the value of period, holding the other parameters fixed

# log likelihood is the sum of - log n! - integrated_rate + sum rate at event times


def log_likelihood(period_value, event_times, vals_fixed):
    vals = vals_fixed.copy()
    vals[period] = period_value

    integrated_rate = cdf_function(t_max, *vals.values())
    rate_at_events = rate_function(event_times, *vals.values())

    log_lik = -integrated_rate + np.sum(np.log(rate_at_events))
    # use stirling's approximation for log n!
    n = len(event_times)
    log_lik -= n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)

    return log_lik


periods = np.linspace(0.1, 6, 1000)
log_likelihoods = [log_likelihood(p, rescaled_draws, vals) for p in periods]
plt.plot(periods, log_likelihoods)
plt.xlabel("Period")
plt.xlim(periods.min(), periods.max())
plt.axvline(vals[period], color="red", linestyle="--", label="True value")
plt.ylabel("Log Likelihood")
plt.legend()


from celluloid import Camera
import tqdm

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
camera = Camera(fig)

for j in tqdm.tqdm(range(len(rescaled_draws))):
    if j == 0:
        continue
    periods = np.linspace(0.1, 6, 1000)
    log_likelihoods = [log_likelihood(p, rescaled_draws[:j], vals) for p in periods]

    ax1.hist(rescaled_draws[:j], bins=200, density=False, alpha=0.5, color="C0")
    ax1.set_xlim(t.min(), t.max())
    ax1.set_ylim(0, 1.1 * np.max(np.histogram(rescaled_draws, bins=200)[0]))
    ax1.set_xlabel("Event times")

    ax2.plot(periods, log_likelihoods, color="C0")
    ax2.set_xlabel("Period")
    ax2.set_xlim(periods.min(), periods.max())
    ax2.axvline(vals[period], color="red", linestyle="--", label="True value")
    ax2.axvline(
        periods[np.argmax(log_likelihoods)],
        color="green",
        linestyle="--",
        label="Inferred value",
    )
    # ax2.set_ylabel("Log Likelihood")
    # ax2.legend()
    plt.suptitle(f"Inference after {j} events")
    camera.snap()

animation = camera.animate()
animation.save("inference_animation.gif", writer="imagemagick", fps=10)
print("Animation saved as inference_animation.gif")
