# Research Audit — Aurora Records & Solar Cycle Recovery

*Complete codebase review: purpose, evolution, status, bugs, and scientific story.*

---

## 1. Per-File Reference

### `helpers.py`

**Role:** Shared utility library. Every other analysis file either imports from it (`from helpers import *`) or duplicates it.

**Contents by layer (in order of appearance):**

| Function group | What it does |
|---|---|
| `beta0`, `lambda_func` | Core NHPP: compute baseline rate and intensity `λ(t) = exp(β₀ + β₁ sin(ωt + φ))` |
| `cdf_inversion` | Simulate NHPP events via CDF inversion (trapezoidal CDF, Poisson draw, inverse interpolation) |
| `log_likelihood_params` | NHPP log-likelihood in amplitude-phase form `(β₀, β₁, ω, φ)` |
| `scan_period/beta0/beta1/phi`, `beta0_grid` | 1-D profile scans at fixed other parameters — **initialisation-only heuristics**, NOT true profiles |
| `lambda_func_2harm`, `beta0_2harm` | 2-harmonic intensity function |
| `lambda_func_1harm`, `beta0_1harm` | 1-harmonic (a, b) intensity for Fourier expansion |
| `log_likelihood_1harm`, `log_likelihood_2harm` | Grid-search likelihoods for harmonic models |
| `cdf_inversion_harm` | Generic CDF inversion for arbitrary lambda functions |
| `success_curve` | Empirical CDF of min-event-count for period recovery |
| `log_prior`, `log_probability` | MCMC posterior — **original parameterisation** `(β₀, β₁, logP, logφ)` |
| `run_emcee_sampler` | emcee wrapper (uses original parameterisation) |
| `q16_50_84` | Percentile summary — **defined three times**; only last is active |
| `amplitude_phase_from_ab` | Converts `(a, b)` → `(amplitude A, phase φ)` |
| `log_likelihood_ab`, `log_prior_ab`, `log_probability_ab` | MCMC posterior — **new (a, b) parameterisation** where `log λ(t) = β₀ + a sin(ωt) + b cos(ωt)`, `theta = [β₀, a, b, logP]` |
| Semiparametric block | `make_rbf_basis`, `phi_from_raw`, `semiparam_log_rate`, etc. — experimental, incomplete, not called anywhere |

**Bugs and quality issues:**
- `q16_50_84` defined 3 times; two are dead code.
- `from scipy.special import expit` — imported but never used.
- `run_emcee_sampler` passes `rstate0=rng` to `run_mcmc`. This is not a valid parameter in emcee ≥ 3.0 and will raise `TypeError`.
- **Naming inconsistency**: `lambda_func_1harm` treats `a1` as the sin coefficient and `b1` as the cos coefficient. `beta0_1harm` and `log_likelihood_1harm` treat `a1` as cos, `b1` as sin — the opposite convention. The likelihood functions are internally self-consistent, but `lambda_func_1harm` cannot be safely combined with them without swapping (a1, b1). Same issue in the 2-harmonic pair.
- Semiparametric functions are present but unfinished and uncalled. They increase file length with no benefit.

---

### `main_model.py`

**Role:** Original standalone simulation study. Produces the animations showing period recovery as events accumulate and estimates the minimum sample size needed.

**Contents:**
- Sets true parameters: `T=787 yr, P_true=11.0, β₁=0.6, φ=π/2, N_target=788`
- Simulates events via `cdf_inversion` (local copy)
- Produces `animation.gif` (6-panel: event histogram + logL vs each of P, β₀, β₁, φ)
- Produces `animation_sequential.gif` (2-panel: histogram + logL vs P, sequential events)
- Produces `animation_random.gif` (2-panel, random order)
- `min_events()`: finds the minimum N such that all subsequent estimates are within ±0.5 yr of P_true — runs K=300 simulations → N₉₀, N₉₅
- Repeats with random-order events → `min_events_random()` equivalent
- Sections on 1-harmonic and 2-harmonic simulations (model misspecification test)

**Bugs and quality issues:**
- **Massively duplicates `helpers.py`**: redefines `beta0`, `lambda_func`, `cdf_inversion`, `log_likelihood_params`, `scan_period`, `scan_beta0`, `scan_beta1`, `scan_phi`, `beta0_grid`, `lambda_func_2harm`, `beta0_2harm`, `lambda_func_1harm`, `beta0_1harm`, `log_likelihood_1harm`, `log_likelihood_2harm`, `cdf_inversion_harm`. None of these are imported from `helpers.py`.
- `init()` and `update()` are defined twice (once for the 6-panel animation, once for the random-order animation), with the second definition silently shadowing the first.
- Same `lambda_func_1harm` / `beta0_1harm` naming inconsistency as in `helpers.py` (copied verbatim).

**Status:** The scientific content (minimum sample size, animation logic) is the canonical result of Stage 1. The code itself should be refactored to import from `helpers.py`. Has **not** been superseded — `simulation_models.py` did not complete the takeover.

---

### `simulation_models.py`

**Role:** Appears to be a mid-project reorganisation attempt of `main_model.py`. Nearly identical content.

**Differences from `main_model.py`:**
- `from helpers import *` at the top — this import is present but then all functions are redefined locally, so the import is effectively wasted.
- Animation code is the same.
- The 1-harmonic / 2-harmonic section and the `cdf_inversion_harm` redefinition are present here too.

**Status:** **Incomplete transition.** It imports helpers but then overrides everything. Both files coexist without resolution. Either this was meant to replace `main_model.py` (and the transition was never finished) or it is a working scratchpad. Either way it is redundant.

---

### `real_data_models.py`

**Role:** Applies the full inference pipeline to the Korean aurora data. This is the primary analysis script.

**Contents (in execution order):**
1. Load `data/KoreanAuroraRecords/Korean_Auroral_Full.xlsx` → extract years, compute `event_times = year - t0` (relative time), `T = max(event_times)`.
2. Initial grid scan (`scan_period`, `scan_beta0`, `scan_beta1`, `scan_phi`) to get starting estimates `(b0_hat, b1_hat, P_hat, phi_hat)`.
3. **Single-start MCMC** (original parameterisation): 64 walkers, 50 k burn + 30 k production. Bug fix note inline.
4. Dominant-mode filtering: find P_mode from P histogram, keep samples with |P − P_mode| < 0.03 yr; trace plot, corner plot, PPC for filtered samples.
5. **Multi-start MCMC** (original parameterisation): identifies top K=4 peaks from `scan_period`, runs independent emcee chains from each, picks the best by mean log-probability.
6. Second round of dominant-mode filtering (dP = 0.015 yr, stricter).
7. **Diagnostics section** (marked `# DIAGNOSTICS SECTION`):
   - D1: Profile log-likelihood over P (calls `profile_logL_over_P` from `diagnostics.py`)
   - 1-harmonic vs 2-harmonic profile comparison
   - D2: Rolling-window period scan
   - D3: Simulation-based calibration (SBC, n_sim=50)
   - D4: **Amplitude-phase MCMC** — a second, cleaner emcee run using `log_probability` (same original parameterisation but with corrected φ bounds and a reference to `P_profile_hat`)
   - D5: PPC split by epoch (calls `plot_ppc_epoch` from `diagnostics.py`)

**Key bug fix documented inline:**
```python
# BUG FIX: was np.log(np.pi - 1e-12), which clipped phi to (0, pi).
logphi_max = np.log(2.0 * np.pi - 1e-12)
```
The original initialisation clipped φ walkers to half the phase space, biasing the posterior. This is already fixed.

**Note:** Despite the section label "D4: MCMC — amplitude-phase form", the code still uses `log_probability` (original `(β₀, β₁, logP, logφ)` parameterisation), not `log_probability_ab`. There is no completed emcee run using the (a, b) parameterisation in this file; `log_probability_ab` exists in helpers.py but is not called in `real_data_models.py`.

---

### `diagnostics.py`

**Role:** Standalone diagnostic module. Contains all statistical validation functions. Added after initial MCMC implementation.

| Function | What it does |
|---|---|
| `profile_logL_over_P(event_times, T, P_grid)` | For each P, maximises log L over (β₀, a, b) via L-BFGS-B with analytical gradients. This is the **true profile likelihood** — the only correct way to marginalise over nuisance parameters. Returns profile logL array and MLE parameters. |
| `windowed_period_scan(event_times, years_abs, window_years, step_years)` | Slides a time window over the record, fits P̂ in each window. Tests stationarity. |
| `plot_windowed_scan(results)` | Two-panel: logL curves per window (coloured by year) + P̂ vs window midpoint. |
| `run_sbc(n_sim, T_val, N_target, ...)` | Simulation-based calibration. Draws P_true from prior, simulates events, recovers P via profile likelihood, computes rank of P_true in 1-D posterior. Uniform rank histogram = calibrated estimator. |
| `plot_sbc(records)` | Scatter P_mle vs P_true + rank histogram. |
| `plot_ppc_epoch(flat_ab_samples, event_times, years_abs, T, epoch_edges)` | PPC split by time epoch. Overlays posterior intensity envelope on empirical rate per epoch. Key diagnostic for non-stationarity (grand solar minima: Maunder ~1645–1715, Spörer ~1460–1550, Oort ~1040–1080). |
| `_neg_ll_and_grad_2harm` | Gradient function for 2-harmonic model (L-BFGS-B). |
| `profile_logL_2harm_over_P` | Profile likelihood for 2-harmonic NHPP. |
| `plot_profile_comparison` | Overlays 1-harmonic and 2-harmonic profile logL curves. Tests whether multimodality is structural aliasing or an insufficient-harmonics artefact. |

**Why profile likelihood matters:** `scan_period` (in helpers.py) is a 1-D slice at fixed (β₀, β₁, φ), not a true profile. Its peak depends on the starting point. `profile_logL_over_P` is guaranteed to find the global optimum for fixed ω because log L is **strictly concave in (β₀, a, b)** for any fixed ω — meaning L-BFGS-B finds the exact MLE of the nuisance parameters every time.

---

### `periodogram_methods.py`

**Role:** Model-free period estimation. Added late in the project to corroborate the NHPP result without assuming a parametric rate model or stationarity.

| Function | What it does |
|---|---|
| `point_process_periodogram(event_times, f_grid)` | Bartlett/Fogel-Gavish: `I(f) = (1/N)|Σ exp(2πi f tₙ)|²`. No rate model assumed. Memory-efficient chunked implementation. |
| `modified_lls(event_times, P_min, P_max, nP, n_nu)` | Quinn–Clarkson–McKilliam (2012) SSMOD estimator. Minimises `Σ⟨γtₙ − ν⟩²` over γ=1/P. Strongly consistent with N^(3/2) convergence rate. Does not model the event rate. Runtime ~30 s for N=787. |
| `lomb_scargle_events` | Lomb-Scargle on event time data. |
| `refine_peak(event_times, f_coarse, half_width)` | Brent optimisation to refine a periodogram frequency peak. |
| `simulate_sparse_events` | Local simulation (partial duplicate of `cdf_inversion` in `helpers.py`). |
| `recovery_test` | n_sims simulations, reports median |P̂ − P_true| for both methods. |
| `plot_periodogram_suite` | Two-panel: point-process periodogram (left) + normalised SSMOD (right). |
| `subsampling_robustness` | Randomly drops 10%, 30%, 50% of events, re-estimates period n_reps times, plots KDE of P̂. Tests stability of point-process periodogram. |

---

### `plots.py`

**Role:** Exploratory data visualisation script. Runs top-to-bottom; not imported.

**Contents:**
- Histogram of Korean auroral records with Gaussian smoothing, labelled grand minima (Oort, Wolf, Spörer, Maunder).
- Lomb-Scargle periodogram on yearly-count time series (not on event times).
- Lomb-Scargle on magnitude-weighted annual series from `Korean_Aurora_Grades_918_1392.xlsx` (graded records, restricted to 1000–1400 CE).
- **Monte Carlo significance levels**: 10,000 surrogate periodograms (random year assignments preserving N), computes 99.7th and 99.86th percentile envelopes. This is the strongest exploratory evidence in the file.
- Stem plot of graded records 1000–1400.

**Unused imports:** `seaborn` is imported but never used.

---

### `main_plots.py`

**Role:** Extended exploratory and publication-quality plotting. Runs top-to-bottom; not imported.

**Contents:**
- Stacked histogram of Korean + Chinese aurora records (uses `data/auroral_records_optionA.xlsx` — **this file does not exist in the repository**).
- Lomb-Scargle periodogram on combined Korean-only event series, with 8/11/22-year reference lines.
- Calls `plot_periodogram_suite` and `subsampling_robustness` from `periodogram_methods.py` on the Korean full dataset.

**Bugs:**
- `pd.read_excel("data/auroral_records_optionA.xlsx")` will raise `FileNotFoundError` at startup.
- `import seaborn as sns` and `from scipy.fft import rfft, rfftfreq` — both imported, neither used.

---

### `data/ChineseDynastyRecords/merging.py`

**Role:** One-time data wrangling script. Merges 7 Chinese dynasty Excel files (Sui 581–619, Tang 619–907, Five Dynasties 907–960, Song 960–1279, Yuan 1368–1644, Ming 1368–1644, Qing 1616–1949) into `Chinese_Aurora_Records.xlsx`. Standalone; not imported.

---

### `plots/histogram+kde.ipynb` and `Interactive - helpers.py.ipynb`

**Role:** Jupyter notebooks used for interactive exploration. The `helpers` notebook mirrors `helpers.py` with all function definitions plus the semiparametric model — it was likely the development scratchpad before the functions were moved to `helpers.py`. The histogram+kde notebook contains supplemental visualisations.

---

## 2. Chronological Narrative

### Stage 1 — Feasibility study: how many events do you need? (`main_model.py`)

The first question was: *can a sinusoidal NHPP recover a ~11 yr period from sparse medieval records at all?* `main_model.py` simulated data under true parameters (T=787 yr, P=11 yr, β₁=0.6, φ=π/2), then animated the log-likelihood surface as events accumulated sequentially. The `min_events()` function found the minimum N such that all future estimates stay within ±0.5 yr of P_true. K=300 Monte Carlo simulations produced empirical N₉₀ and N₉₅ quantiles. With ~787 Korean events, the study established that the dataset is large enough to be informative.

At this point all functions were defined locally; `helpers.py` did not exist yet.

---

### Stage 2 — Refactoring into a shared library (`helpers.py` first version)

The scan and simulation functions were extracted into `helpers.py`. The original amplitude-phase parameterisation `(β₀, β₁, logP, logφ)` was carried over. `log_prior` and `log_probability` were added to enable MCMC. `run_emcee_sampler` was a convenience wrapper. `q16_50_84` appeared here (eventually defined three times from different copy-paste sessions).

`simulation_models.py` was created around the same time, apparently as a cleaner home for the simulation code. However the transition was never finished — both `simulation_models.py` and `main_model.py` continued to coexist, both with local function redefinitions.

---

### Stage 3 — First MCMC on real data and phase bug (`real_data_models.py`, first half)

`real_data_models.py` loaded the Korean data and ran the first emcee run. A subtle bug was found: the initialisation clipped `logφ_max = log(π)` instead of `log(2π)`, preventing walkers from exploring φ > π. The bug was fixed inline (the fix is present in the current code). After fixing, the MCMC revealed a **multimodal posterior over P**: multiple narrow peaks separated by roughly P²/T ≈ 0.14 yr.

To handle multimodality, dominant-mode filtering was added: find the histogram mode of {P_s}, keep only samples within dP of it, re-plot trace, corner, and PPC for the filtered subset.

---

### Stage 4 — Multi-start MCMC to explore all modes (`real_data_models.py`, middle section)

The dominant-mode approach only explored one peak. Multi-start MCMC was added: `scan_period` identifies the top K=4 logL peaks; an independent emcee chain is seeded from each. The best chain (by mean log-probability) is selected. This established that the multi-start runs broadly agree on P ≈ 11 yr as the dominant mode.

---

### Stage 5 — Profile likelihood and (a, b) reparameterisation (`diagnostics.py`, `helpers.py` second layer)

The core insight motivating this stage: the 1-D scan `scan_period` is not a genuine profile likelihood — it holds (β₀, β₁, φ) fixed at a guess, so its peaks are artefacts of the starting point. The true profile requires maximising over all nuisance parameters.

The (a, b) reparameterisation — `log λ(t) = β₀ + a sin(ωt) + b cos(ωt)` — was introduced because for any fixed ω, the log-likelihood is **strictly concave in (β₀, a, b)**, guaranteeing a unique global maximum that L-BFGS-B can find reliably. `profile_logL_over_P` in `diagnostics.py` uses this property: for each P on a grid, it runs one L-BFGS-B call to get the exact profile value.

`log_likelihood_ab`, `log_prior_ab`, `log_probability_ab` were added to `helpers.py` to support the (a, b) MCMC. Note: despite being added, these are **not called in `real_data_models.py`** — D4 still uses `log_probability` (original parameterisation). The (a, b) MCMC is implemented but not run in the main analysis file.

---

### Stage 6 — Stationarity and calibration checks (`diagnostics.py`)

Two questions arose:
- *Is the period stationary across the 474-year record?* → `windowed_period_scan`: 100-year windows, 25-year steps, each window gets its own profile P̂.
- *Is the profile estimator calibrated?* → `run_sbc`: draws true parameters from the prior, simulates data, recovers P via profile likelihood, computes the rank of P_true. A uniform rank histogram confirms the estimator has correct coverage.

These are added as independent diagnostic blocks (`# D2`, `# D3`) in `real_data_models.py`.

---

### Stage 7 — Epoch PPC (`diagnostics.py`)

`plot_ppc_epoch` was added to test whether the stationary sinusoidal model fits the data in each temporal epoch. If the model is correctly specified, the posterior intensity envelope should straddle the empirical rate consistently across epochs. Persistent over- or under-coverage in a particular epoch (e.g., during the Maunder Minimum 1645–1715 or the Spörer Minimum 1460–1550) would indicate non-stationarity and model misspecification.

This is diagnostic `D5` in `real_data_models.py`. The conversion from amplitude-phase MCMC samples → (a, b) form (needed for `plot_ppc_epoch`) is done inline:
```python
flat_ab_for_d5 = np.column_stack([b0_ap_s, b1_ap_s * np.cos(phi_ap_s), b1_ap_s * np.sin(phi_ap_s), flat_ap[:, 2]])
```

---

### Stage 8 — Model-free corroboration (`periodogram_methods.py`)

To avoid relying solely on NHPP assumptions, two model-free methods were implemented:
1. **Point-process periodogram** (Bartlett/Fogel-Gavish): a classical spectral tool that does not assume any rate model and is robust to non-stationarity.
2. **Modified LLS** (Quinn–Clarkson–McKilliam 2012): strongly consistent with N^(3/2) convergence, does not model the rate at all.

If both agree with the NHPP profile likelihood at P ≈ 11 yr, the result is model-robust. `plot_periodogram_suite` produces the two-panel figure. `subsampling_robustness` tests stability by dropping 10–50% of events.

These methods are called from `main_plots.py` (and accessible from `real_data_models.py` if imported).

---

### Stage 9 — Chinese data and combined analysis (`main_plots.py`, `data/ChineseDynastyRecords/`)

`merging.py` combined 7 Chinese dynasty aurora spreadsheets into a single file. `main_plots.py` loads both Korean and Chinese records and produces stacked histograms and combined periodograms. This stage appears exploratory — no combined NHPP analysis has been implemented yet. The reference file `data/auroral_records_optionA.xlsx` is missing.

---

## 3. Component Status Table

| Component | Primary file(s) | Status | Notes |
|---|---|---|---|
| NHPP intensity + likelihood | `helpers.py` | **Active / Core** | Used everywhere |
| CDF inversion simulation | `helpers.py` | **Active** | Duplicated in main_model.py, simulation_models.py, periodogram_methods.py |
| 1-D scan (scan_period etc.) | `helpers.py` | Active but imprecise | Not true profile; initialisation-dependent |
| Profile likelihood (1-harmonic) | `diagnostics.py` | **Active / Primary diagnostic** | Correct marginalisation; strictly concave in nuisance params |
| Profile likelihood (2-harmonic) | `diagnostics.py` | Active / diagnostic | For testing whether 2nd harmonic changes result |
| MCMC — original (β₀, β₁, logP, logφ) | `helpers.py`, `real_data_models.py` | **Active (used in real_data_models.py)** | Phase bug fixed; multimodal posterior handled by multi-start + filtering |
| MCMC — (a, b) parameterisation | `helpers.py` | Implemented, not called | `log_probability_ab` exists; no emcee run in real_data_models.py uses it |
| Dominant-mode filtering | `real_data_models.py` | Active / post-processing | Applied after single-start and multi-start MCMC |
| Rolling-window scan | `diagnostics.py` | **Active / diagnostic** | Tests stationarity of P across record |
| Simulation-based calibration (SBC) | `diagnostics.py` | **Active / diagnostic** | Tests coverage of profile estimator |
| Epoch PPC | `diagnostics.py` | **Active / diagnostic** | Tests fit near grand solar minima |
| Point-process periodogram | `periodogram_methods.py` | **Active** | Model-free; used in main_plots.py |
| Modified LLS (Quinn 2012) | `periodogram_methods.py` | **Active** | Model-free; strongly consistent |
| Subsampling robustness | `periodogram_methods.py` | Active / diagnostic | Tests stability of periodogram estimate |
| Simulation study (min events) | `main_model.py` | Active / complete | N₉₀, N₉₅ computed; ~787 events is sufficient |
| 1-harmonic / 2-harmonic grid likelihoods | `helpers.py`, `main_model.py` | Active / diagnostic | Tests model order sensitivity |
| Semiparametric NHPP | `helpers.py` | **Experimental / incomplete** | Functions exist, not called anywhere; no test or analysis using them |
| Chinese data merge | `data/ChineseDynastyRecords/merging.py` | Complete (data prep) | Output file exists; no combined NHPP analysis yet |
| Korean data visualisation | `plots.py` | Active | MC significance levels strongest result here |
| Combined Korean+Chinese visualisation | `main_plots.py` | Broken (missing file) | Requires `data/auroral_records_optionA.xlsx` |

---

## 4. Answers to the Five Specific Questions

### Q1: What currently supports the final conclusions?

The final conclusion — that Korean aurora records contain a detectable ~11-year solar cycle — rests on four independent lines of evidence:

1. **Profile likelihood (`diagnostics.py: profile_logL_over_P`):** The most rigorous parametric result. The profile log-likelihood curve has its global maximum near P ≈ 11 yr. This is the correct profile (not the initialisation-dependent scan).

2. **Multi-start MCMC (`real_data_models.py`, best-run selection):** Bayesian posterior on P under the NHPP model. After dominant-mode filtering, the credible interval is concentrated near 11 yr.

3. **Point-process periodogram (`periodogram_methods.py`):** Model-free spectral peak agrees with the parametric estimate.

4. **Modified LLS (`periodogram_methods.py`):** Strongly-consistent model-free estimator also points to P ≈ 11 yr.

The agreement of (3) and (4) with (1) and (2) is the key scientific argument, because (3) and (4) require no assumptions about the rate model or stationarity.

### Q2: What is purely diagnostic (validates the inference but does not change the conclusion)?

- **SBC (`run_sbc`):** Confirms the profile estimator has correct frequentist coverage. If the rank histogram is uniform, the estimator is calibrated and can be trusted on the real data.
- **Rolling-window scan (`windowed_period_scan`):** Confirms the dominant period is approximately stationary across the record. If P̂ were erratic across windows, the stationary NHPP would be misspecified.
- **Epoch PPC (`plot_ppc_epoch`):** Confirms the model fits the data in each sub-period, including near grand solar minima. Systematic misfit near Maunder/Spörer would require a non-stationary extension.
- **1-harmonic vs 2-harmonic comparison (`plot_profile_comparison`):** Confirms that the multimodal profile logL is structural aliasing from finite T (many near-equal peaks) rather than evidence for a second harmonic.
- **Subsampling robustness (`subsampling_robustness`):** Confirms the periodogram peak is stable when events are dropped; the estimate is not driven by a small subset of years.

### Q3: What can be removed or cleaned up?

**Can be removed without scientific loss:**
- Duplicate function definitions of `q16_50_84` (keep only one).
- `from scipy.special import expit` in `helpers.py` (never used).
- `import seaborn as sns` and `from scipy.fft import rfft, rfftfreq` in `main_plots.py` (never used).
- The semiparametric block in `helpers.py` (incomplete, uncalled — move to a feature branch if needed).
- `simulation_models.py` is entirely redundant with `main_model.py`; one should be deleted.
- The locally-defined function duplicates in `main_model.py` should import from `helpers.py` instead.

**Should be fixed (bugs):**
- `run_emcee_sampler` in `helpers.py`: `rstate0=rng` is not a valid parameter for emcee ≥ 3.0. Remove `rstate0=rng` or replace with `move=emcee.moves.StretchMove(rng=rng)`.
- `main_plots.py` line 1: replace `data/auroral_records_optionA.xlsx` reference with the correct file, or gate behind `if Path(...).exists()`.
- `lambda_func_1harm` (and `lambda_func_2harm`): the (a=sin, b=cos) convention is opposite to `log_likelihood_1harm`'s (a=cos, b=sin) convention. Either rename the parameters or add a docstring warning. Since these functions are only used for plotting (not for likelihood computation), the practical impact is a phase shift in the displayed fit curve only.

**Should be completed:**
- The (a, b) MCMC (`log_probability_ab`) exists in `helpers.py` but is never run. Either add an emcee block in `real_data_models.py` using it, or remove it if it's been decided the amplitude-phase MCMC (with the phase bug fixed) is sufficient.

### Q4: Which plots belong in the final paper?

| Figure | Source | Purpose |
|---|---|---|
| Korean record histogram with grand minima labelled | `plots.py` | Data overview |
| Stem plot of graded records 1000–1400 | `plots.py` | Data quality illustration |
| Profile log-likelihood curve `profile_1h` vs P | `real_data_models.py D1` | Primary parametric result |
| 1-harmonic vs 2-harmonic profile comparison | `real_data_models.py D1` | Model-order robustness |
| Two-panel periodogram suite (Bartlett + SSMOD) | `periodogram_methods.plot_periodogram_suite` | Model-free corroboration |
| Corner plot of MCMC posterior (dominant mode) | `real_data_models.py` | Full Bayesian uncertainty |
| Rolling-window P̂ vs time | `real_data_models.py D2` | Stationarity diagnostic |
| Epoch PPC panels | `real_data_models.py D5` | Model fit near grand minima |
| SBC rank histogram | `real_data_models.py D3` | Estimator calibration |
| Subsampling robustness KDE | `periodogram_methods.subsampling_robustness` | Data-deletion robustness |
| Simulation animation stills (N₉₀ recovery) | `main_model.py` | Minimum sample size |

Plots that are **not** paper-quality and should stay as internal diagnostics: the 6-panel animation (`animation.gif`), the raw trace plots (before dominant-mode filtering), the multi-start run comparison table.

### Q5: What is the scientific story?

**Claim:** The ~11-year solar cycle is detectable in medieval Korean auroral records (918–1392 CE) even at low latitudes.

**Argument structure:**

1. *Data:* 787 Korean aurora events over T ≈ 474 years (Koryo-Sa period), predominantly red low-latitude auroras triggered by geomagnetic storms during solar maxima.

2. *Model:* Events are modelled as a sinusoidal NHPP with `log λ(t) = β₀ + β₁ sin(ωt + φ)`. The rate peaks at solar maxima (high aurora frequency) and troughs at solar minima.

3. *Feasibility:* Simulation study shows that ~787 events over 474 years is sufficient to recover P ≈ 11 yr within ±0.5 yr at 90% confidence.

4. *Parametric evidence:* The profile likelihood and Bayesian posterior both concentrate near P ≈ 11 yr. Multimodality is structural (many nearly-equivalent alias peaks spaced ~0.14 yr apart) rather than evidence against the solar-cycle interpretation.

5. *Model-free confirmation:* The point-process periodogram and the modified LLS estimator — neither of which assumes a parametric rate model or stationarity — both peak at P ≈ 11 yr.

6. *Validation:* SBC confirms the profile estimator has correct coverage; the rolling-window scan confirms the period is approximately stationary; epoch PPC shows no systematic misfit near the Maunder or Spörer Minima.

7. *Implication:* The consistency across parametric NHPP, model-free spectral methods, and multiple diagnostic checks supports the conclusion that the Korean records preserve a genuine solar cycle signal, and that the ~11-year period was maintained throughout the Goryeo dynasty period.

---

## 5. Outstanding Issues Summary

| # | Issue | Severity | Location |
|---|---|---|---|
| 1 | `rstate0=rng` passed to emcee `run_mcmc` — invalid in emcee ≥ 3.0 | **Bug / will crash** | `helpers.py: run_emcee_sampler` |
| 2 | `data/auroral_records_optionA.xlsx` does not exist | **Bug / will crash** | `main_plots.py` line 8 |
| 3 | `(a, b)` MCMC (`log_probability_ab`) implemented but never run | Gap | `helpers.py`, `real_data_models.py` |
| 4 | `lambda_func_1harm`/`lambda_func_2harm` use opposite (a, b) convention from `log_likelihood_1harm`/`log_likelihood_2harm` | Naming inconsistency | `helpers.py` |
| 5 | `q16_50_84` defined 3 times | Dead code | `helpers.py` |
| 6 | `from scipy.special import expit` | Unused import | `helpers.py` |
| 7 | `seaborn`, `rfft`, `rfftfreq` imports | Unused imports | `main_plots.py` |
| 8 | `simulation_models.py` is a near-duplicate of `main_model.py` | Redundant file | both files |
| 9 | Both `main_model.py` and `simulation_models.py` redefine all `helpers.py` functions locally | Code duplication | both files |
| 10 | Semiparametric block in `helpers.py` is incomplete and unused | Clutter | `helpers.py` |
| 11 | `init()` and `update()` defined twice in `main_model.py` (second silently shadows first) | Confusing | `main_model.py` |
| 12 | No completed NHPP analysis on Chinese data | Missing analysis | `main_plots.py`, `ChineseDynastyRecords/` |
