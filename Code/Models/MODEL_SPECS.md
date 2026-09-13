# Revenue Index Model Specs

**Fitted model artifacts — Northern Italy grid point 45.5°N, 11.25°E**
(roughly 115 km from Bologna proper, inside the same NORD bidding zone;
file and variable names still say "Bologna" for historical reasons)

Every parameter below is read from the saved artifact in `Code/Models/`, not from the
notebook source — these are the values the simulation actually loads.

*Regenerated 2026-09-12 after the Modelling-stage rework · fit window
2005-01-01 → 2025-12-31, n = 7,662 days (copula: 2015-01-03 → 2025-12-31,
n = 4,018)*

> **What changed since the last read (2026-08-20).** Two fixes rippled
> through every number below: (1) Transformation now joins price and
> irradiance on the true UTC instant instead of a naive hour that silently
> mixed Italian local time with UTC; (2) the capture-rate notebook's
> uniformity gate on `U_CR` is now a recorded warning instead of a hard
> `assert` that previously blocked the pipeline before either artifact was
> written. `U_CR` still does not clear the gate (KS p = 0.023) — this is
> tracked as `U_CR_ks_pvalue` in `cr_ar_params.pkl` and as a known
> limitation below, not hidden. The copula's fitting sample also grew from
> n = 3,639 to n = 4,018 after price was dropped from its join (see §4).

---

## Dependency structure

| | Model | Spec |
|---|---|---|
| Weather margin | Clearness index Kt | logit + ARMA(2,1) + GMM |
| Weather margin | Temperature | Fourier(3) + AR(2)-t |
| Market margin | Capture rate | SARIMA × GARCH skew-t |
| **Dependence** | **t-copula, ν = 17** | **PIT of the three residual series** |

The three marginal models feed the copula; the copula is a dependence layer over them,
not a fourth stage in a sequence.

---

## 1. Clearness index Kt

**Spec:** rescaled logit → 1-harmonic seasonal → ARMA(2,1) → monthly 2-component GMM

- Source: `Code/Modelling/01_solar_irradiation.ipynb`
- Artifacts: `Code/Models/GHI - KT/{ghi_arma_result, ghi_mixture_params, ghi_seasonal_coeffs}.pkl`

Unaffected by the timezone fix (weather-only; no join with price), so every
value in this section is unchanged from the previous read.

### Transform & deterministic seasonal

```
X_t   = 1 - K_t                        (cloudiness, increasing in overcast)
X'_t  = (X_t - alpha) / beta           in (0,1) strictly
Y_t   = log( X'_t / (1 - X'_t) )       logit link
Y_bar = a0 + a1*cos(2*pi*doy/365) + a2*sin(2*pi*doy/365)
Y~_t  = Y_t - Y_bar
```

| Parameter | Value | Note |
|---|---:|---|
| `alpha` | −0.001000 | logit offset, = min(X_t) − ε |
| `beta` | 0.9113408 | logit scale, = range(X_t) + 2ε |
| `a0` | −2.10331 | seasonal level |
| `a1` | 0.88853 | cos, annual harmonic |
| `a2` | 0.21977 | sin, annual harmonic |
| `n_harmonics` | 1 | day-of-year basis, period 365 |

### ARMA(2,1) on the deseasonalized series

Order selected by BIC over p, q ∈ 0..3.

### Innovation distribution — 2-component Gaussian mixture per calendar month

Component 0 is the **clear-sky regime** (lower mean ε, low cloudiness),
component 1 the **cloudy** regime (higher mean ε). A previous version of the
fitting notebook and its exported figure had these two labels swapped; the
values below were never wrong, only their names — fixed in
`01_solar_irradiation.ipynb`, cells tagged D4.

ΔBIC = BIC(1 component) − BIC(2 components); positive means the 2-component
mixture is preferred. Every month prefers two components.

| Month | μ (clear) | σ (clear) | w (clear) | μ (cloudy) | σ (cloudy) | w (cloudy) | ΔBIC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jan | −2.149 | 1.964 | 0.357 | 1.220 | 1.385 | 0.643 | 37.3 |
| Feb | −2.787 | 1.715 | 0.359 | 1.311 | 1.556 | 0.641 | 25.8 |
| Mar | −3.041 | 1.705 | 0.322 | 1.181 | 1.527 | 0.678 | 44.7 |
| Apr | −2.503 | 1.665 | 0.324 | 1.298 | 1.468 | 0.676 | 31.5 |
| May | −1.740 | 1.988 | 0.311 | 1.268 | 1.424 | 0.689 | 30.5 |
| Jun | −2.482 | 1.443 | 0.299 | 1.096 | 1.368 | 0.701 | 31.0 |
| Jul | −2.425 | 1.293 | 0.408 | 1.372 | 1.392 | 0.592 | 37.1 |
| Aug | −2.458 | 1.375 | 0.416 | 1.394 | 1.436 | 0.584 | 32.1 |
| Sep | −2.438 | 1.410 | 0.389 | 1.507 | 1.393 | 0.611 | 39.6 |
| Oct | −2.373 | 1.773 | 0.320 | 1.389 | 1.498 | 0.680 | 30.4 |
| Nov | −2.170 | 2.050 | 0.349 | 1.448 | 1.340 | 0.651 | 66.8 |
| Dec | −2.458 | 1.807 | 0.340 | 1.178 | 1.296 | 0.660 | 58.4 |

### Recorded in the notebook

- Seasonality is split across two places by design: the smooth annual harmonic in
  `Y_bar`, plus a month-specific residual offset absorbed by the mixture means. One
  harmonic cannot carry the full intra-year cycle, and `ghi_seasonal_coeffs.pkl` is a
  3-element contract that downstream notebooks unpack positionally.
- `Kt_fitted` maps the conditional *mean* of Y_t through a nonlinear inverse logit, so
  it is a conditional-median-like series, not a level forecast — the Jensen gap is a
  property of the transform. Monte-Carlo simulation integrates the innovation
  distribution instead and is unaffected.
- Support is capped by construction: `alpha`/`beta` come from the sample's own
  min/max, so no simulated day can be cloudier than the cloudiest day observed
  2005-2025.

---

## 2. Air temperature

**Spec:** Fourier(k=3) + linear trend + AR(2) with Student-t innovations

- Source: `Code/Modelling/02_temperature.ipynb`
- Artifact: `Code/Models/Temperature/temp_model_params.pkl`

Also unaffected by the timezone fix; values are unchanged except for naming
(the model is AR(2), previously named `ar1_*` throughout including in the
persisted parquet columns — renamed to `ar2_*`, see the notebook's header).

### Specification

```
T_sim(t) = mu + SUM_{k=1..3} [ a_k*cos(2*pi*k*doy/365) + b_k*sin(2*pi*k*doy/365) ]
                + trend_slope * (year(t) - 2005)
                + resid(t)
resid(t) = phi1*resid(t-1) + phi2*resid(t-2) + sqrt(omega)*eps(t)
eps(t)  ~ t(nu_temp)
```

| Parameter | Value | Note |
|---|---:|---|
| `mu` | 12.6925 | °C, seasonal mean level |
| `a1, a2, a3` | −9.8854, −0.3215, −0.2713 | cosine harmonics k = 1, 2, 3 |
| `b1, b2, b3` | −3.0700, 0.6440, −0.1985 | sine harmonics k = 1, 2, 3 |
| `trend_slope` | 0.0585095 | °C per year, reference year 2005 |
| `phi1, phi2` | 1.0412, −0.2284 | AR(2) on the residual |
| `arch_omega` | 1.7583 | constant innovation variance |
| `innovation_sd` | 1.32601 | = sqrt(omega); scales unit-variance draws |
| `nu_temp` | 6.90809 | Student-t d.o.f. of the AR(2) innovations |

### Cell-temperature uplift (NOCT)

```
T_cell(h)    = T_amb(h) + (NOCT - 20)/800 * GHI(h)
T_eff_daily  = GHI-weighted mean T_cell over daylight hours
T_eff_sim(t) = T_sim(t) + uplift_doy[ doy(t) ]
```

Calibrated in `Code/Modelling/03_noct_uplift.ipynb`, the sole producer of
`Code/Models/Temperature/noct_uplift_params.pkl`. It is a Stage-1 marginal fit
in its own right, separate from `02_temperature.ipynb`: the temperature model
supplies the daily mean, this supplies the daylight offset applied to it.

| Parameter | Value | Note |
|---|---:|---|
| `NOCT` | 45 | °C, nominal operating cell temperature |
| `NOCT_COEFF` | 0.03125 | = (45 − 20) / 800 |
| `uplift_by_doy` | 366 values | 8.72 (doy 355) → 22.71 (doy 195) °C |
| annual mean | 16.0230 | °C, over 7,670 calibration days |
| `MAX_UPLIFT` | 40 | °C clip; 0 day(s) clipped |
| calibration window | 2005-01-01 → 2025-12-31 | |

### Recorded limitations

- Innovation variance is constant. An ARCH-LM test on the AR(2) residuals flags
  remaining heteroskedasticity concentrated in high-irradiance months — exactly where
  the revenue index is most temperature-sensitive. A GARCH or seasonal-variance
  extension is noted as future work, not adopted.
- Fitting a t to the raw pre-AR residuals is degenerate (ν diverges to Gaussian)
  because autocorrelation inflates the tails. The genuine heavy tails (ν ≈ 6.9) appear
  only after the AR(2) filter, so `nu_temp` describes the innovations, not the raw
  daily residual series.
- The AR order (2) is fixed by the production simulator's hardcoded two-lag
  recursion, not by the BIC grid the notebook also runs and reports openly.

---

## 3. Solar capture rate

**Spec:** exp-decay penetration trend + Fourier(2) + weekend dummy →
SARIMA(1,0,0)×(1,0,1)₇ → GARCH(1,1) skewed-t

- Source: `Code/Modelling/05_capture_rate.ipynb`
- Artifact: `Code/Models/Capture rate/cr_ar_params.pkl` · 15 parameters
  (mean 7, SARIMA 3, GARCH 3, dist 2)

All values below reflect the corrected UTC join and therefore differ from the
2026-08-20 read.

### Conditional mean — cannibalisation trend

```
mu_CR(t) = L + (U - L) * exp( -lambda_decay * pen(t) )
           + Fourier2(t) @ beta_fourier          # beta_fourier[0] = level offset c
           + beta_weekend * weekend(t)
```

| Parameter | Value | Note |
|---|---:|---|
| `L_FIXED` | 0.20000 | imposed asymptotic floor |
| `U_FIXED` | 1.00235 | zero-penetration ceiling (was 1.07807) |
| `lambda_decay` | 0.123124 | std. err 0.0078998 (was 0.185085) |
| `beta_fourier` | 0.00939, −0.02076, −0.01228, 0.02497, 0.00511 | 2 harmonics; element 0 is the level offset c |
| `beta_weekend` | −0.031540 | weekend cannibalisation shift |
| `beta_log` | 1 (fixed) | constrained — offset form |
| `effective_L / U` | 0.20939 / 1.01175 | after the level offset |

### Residual dynamics — SARIMA and GARCH

```
CR_resid(t) = phi_s*CR_resid(t-1) + Phi_s*CR_resid(t-7)
              - phi_s*Phi_s*CR_resid(t-8) + theta_q*u(t-7) + u(t)
sigma^2(t)  = omega/garch_scale^2 + alpha*u^2(t-1) + beta*sigma^2(t-1)
z(t)        = u(t)/sigma(t)  ~  skewed-t(nu_g, eta_g), unit variance
```

| Parameter | Value | Note |
|---|---:|---|
| `phi_s` | 0.344634 | AR(1) |
| `Phi_s` | 0.896207 | seasonal AR, lag 7 |
| `theta_q` | −0.658831 | seasonal MA, lag 7 |
| `garch_omega` | 0.375634 | on a ×100 scale (`garch_scale = 100`) |
| `garch_alpha` | 0.086446 | ARCH term |
| `garch_beta` | 0.900364 | GARCH term |
| `alpha + beta` | 0.986811 | near-unit persistence — very slow vol decay |
| `nu_g` | 4.14938 | skewed-t d.o.f. (`arch.SkewStudent`) |
| `eta_g` | −0.100269 | skew — negative, fatter left tail |

`nu_g`/`eta_g` are now extracted from the fitted GARCH by position (the last
two parameters after `omega`/`alpha[1]`/`beta[1]`), not by name — `arch`'s own
naming is inconsistent across distributions (`dist='studentst'` calls its
shape parameter `nu`; `dist='skewt'` calls the same concept `eta` and names
skew `lambda`), so a fixed name would have broken on exactly the distribution
this notebook fits.

### PIT uniformity — now a recorded warning, not a gate

| Field | Value |
|---|---|
| `U_CR_ks_pvalue` | 0.023414 |

The notebook no longer hard-stops when this falls below 0.05 — it prints a
warning and records the p-value here, matching the policy `06_copula.ipynb`
already used (2 of 3 margins ship non-uniform there, documented rather than
blocked). Mean (0.509) and std (0.288) of `U_CR` are both close to target
(0.5, 0.289), so this reads as a small miscalibration, not a structural one.
A genuine likelihood-ratio test (fitting a true symmetric-t alongside the
skewed-t, rather than the skewed-t against a re-fit of itself as a previous
version of the notebook did) confirms the skewed shape is load-bearing:
LR = 52.6, p ≈ 0.

### Recorded limitations

- The weekend effect is a single dummy with no interaction with the penetration
  scenario, so it does not deepen as solar grows. This understates weekend
  cannibalisation in far, high-penetration scenarios such as 2050.
- A Gaussian innovation would understate the probability of a 4-sigma downside shock by
  roughly two orders of magnitude relative to the fitted skewed-t — exactly the region
  driving CR-put hedge payoffs. Downstream VaR and CVaR figures depend on the skewed-t
  choice holding.
- `U_CR` fails its own uniformity check (see above); the copula margin built
  from it may be mis-specified to a similar degree as the two margins already
  flagged as non-uniform in §4.

---

## 4. Student-t copula

**Spec:** 3-dimensional t-copula, ν = 17, on the PIT of the three residual series

- Source: `Code/Modelling/06_copula.ipynb`
- Artifacts: `Code/Models/Copula/{copula_params, marginal_params}.pkl`,
  `{corr_matrix, chol_matrix}.npy`

**Price is no longer part of this join.** A previous version read a fourth
series, `eps_price` from `df_energy_modelled.parquet` — an artifact that no
longer exists, since the notebook that produced it was an orphan nothing
downstream modelled. Loading it cost ~380 post-2015 observations for a
series that was never fed to the copula anyway; dropping it recovered the
full post-2015 sample (n = 3,639 → 4,018) and fixed a `KeyError` the missing
file was causing.

### Correlation matrix

| | U_cr | U_kt | U_temp |
|---|---:|---:|---:|
| **U_cr** | 1.0000 | 0.3100 | −0.0739 |
| **U_kt** | 0.3100 | 1.0000 | −0.2388 |
| **U_temp** | −0.0739 | −0.2388 | 1.0000 |

ν = 17 · ΔAIC vs Gaussian copula = −36.42 (t preferred) · n = 4,018 ·
Cholesky factor stored alongside

### Marginals feeding the PIT

| Margin | Family | Parameters | Note |
|---|---|---:|---|
| `U_cr` | skewed-t | ν = 4.14938, η = −0.100269 | CDF of the GARCH z_t, no rescaling; σ = 0.99764 |
| `U_kt` | Gaussian mixture | 2 components × 12 months | the same monthly GMM listed in §1 |
| `U_temp` | Student-t | ν = 7.73636 | scale 0.867415, σ = 1.00734 |

### Recorded limitations

- Two of the three margins fail a Kolmogorov–Smirnov uniformity test on this fitting
  window (post-2015, crisis included). The parametric PIT feeding the copula may
  therefore attenuate the true dependence toward zero.
- Empirical upper-tail dependence is roughly twice the lower-tail dependence, but an
  elliptical copula is symmetric by construction. Both Gaussian and t therefore
  *overstate* clustering in the lower tail — the revenue-relevant corner of sunny days
  with a low capture rate. The bias is conservative for revenue risk, so it was
  accepted; a skewed-t or vine copula is noted as future work.
- The fit window is the post-2015 sub-sample (n = 4,018), while the marginal parameters
  above are fitted on the full normal-regime window. The two are not the same sample.
- CR-KT dependence is still strengthening over time (rank correlation +0.17 in
  2015-18, +0.26 in 2019-21, +0.31 in 2022-25 — see the notebook's TT2a check),
  so a copula with fixed correlations likely understates it in the later
  projection years. Conservative for a hedging study; named as future work.

---

## One inconsistency worth resolving

The temperature margin still carries **two different Student-t degrees of
freedom** across the artifacts, as it did before this rework.
`Code/Models/Temperature/temp_model_params.pkl` stores `nu_temp = 6.90809`,
fitted to the AR(2) innovations. `Code/Models/Copula/marginal_params.pkl`
stores `nu_temp = 7.73636` with `scale_temp = 0.867415`, used for the PIT
that feeds the copula. The gap moved slightly with the refit (previously
6.91 vs 8.02) but did not close.

If both are meant to describe the same residual series, the simulation is drawing
temperature shocks from a slightly heavier tail (ν = 6.91) than the one the dependence
structure was calibrated against (ν = 7.74). That may well be deliberate — the copula
was fitted on the post-2015 window and the temperature model on the full sample, which
alone would move ν — but it is not recorded as such in either metadata file, so it is
worth either documenting the reason or reconciling the two.

---

## Solar penetration (feeds §3's `pen(t)`)

**Spec:** hardcoded annual series, 2005-2025, normalised to a penetration index

- Source: `Code/Modelling/04_solar_share_historical.ipynb`
- Artifact: `Data/Cleaned/df_solar_share.parquet`

| Field | Value |
|---|---|
| `REFERENCE_SHARE` | 13.00% (2025, the anchor year — pen = 1.0) |
| Series | 21 annual values, Italy national solar share of total generation, source GSE/Terna |

**Caveat, carried from the source notebook's Decision Log:** 2020, 2021 and
2022 are all recorded at exactly 9.50%, and 2024 (11.50%) sits below 2023
(12.30%) despite a period of strong Italian PV capacity growth. This series
sets `lambda_decay` above and every downstream capture-rate projection and
hedge premium — verify the 2020-2024 figures against a primary GSE/Terna
source before treating those projections as final.

---

*Values for §1-2 read directly from the pickled artifacts; §3-4 and this
document as a whole regenerated 2026-09-12 from the freshly re-fitted
artifacts after the Modelling-stage timezone and documentation rework. No
value in this document is restated from notebook source or recomputed by
hand.*

*Artifact set: GHI - KT · Temperature · Capture rate · Copula · Solar share.
The Energy-price branch (`price_ou_params.pkl`, `price_model_params.pkl`,
`df_energy_modelled.parquet`) has been removed entirely — it was an orphan
nothing downstream read.*
