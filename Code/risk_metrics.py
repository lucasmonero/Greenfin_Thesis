"""Shared risk-metric functions for the revenue-index / hedging notebooks.

Convention used throughout this module (see thesis notebooks for the prior,
inconsistent conventions this replaces):
  - dispersion always uses ddof=1 (sample statistics), matching the majority
    convention already used across the codebase and the pandas default.
  - VaR/CVaR are parameterised by `confidence` (e.g. 0.95), not by the tail
    probability alpha, matching the dominant naming (VaR95/CVaR95) found in
    Simulation/03_revenue_index_historical.ipynb and 05_hedge_contract.ipynb.
  - VaR/CVaR are returned as raw levels of the input series (e.g. raw revenue),
    not as losses relative to the mean. Loss-relative-to-mean, when wanted, is
    the caller's responsibility (`ref_mean - result["VaR"]`).
"""

import numpy as np


def _as_array(x):
    return np.asarray(x, dtype=float)


def var_cvar(x, confidence=0.95):
    """VaR/CVaR of `x` at a given confidence level.

    VaR  = the `1 - confidence` percentile of x (e.g. the 5th percentile for
           confidence=0.95) — the level not exceeded with probability
           `1 - confidence`.
    CVaR = mean of x in the tail at or below VaR (expected shortfall).

    Returns {"VaR": float, "CVaR": float}, both raw levels of `x`.
    """
    x = _as_array(x)
    alpha = 1.0 - confidence
    var = float(np.percentile(x, alpha * 100))
    cvar = float(x[x <= var].mean())
    return {"VaR": var, "CVaR": cvar}


def var_cvar_table(x, confidences=(0.90, 0.95, 0.99)):
    """var_cvar for several confidence levels at once.

    Returns a dict keyed "VaR90"/"CVaR90"/"VaR95"/"CVaR95"/... (confidence-level
    naming, matching the majority convention).
    """
    out = {}
    for c in confidences:
        r = var_cvar(x, confidence=c)
        tag = f"{int(round(c * 100))}"
        out[f"VaR{tag}"] = r["VaR"]
        out[f"CVaR{tag}"] = r["CVaR"]
    return out


def expected_shortfall(x, tail_pct=10):
    """Mean of x in the tail at or below its `tail_pct`-th percentile.

    Equivalent to var_cvar(x, confidence=1 - tail_pct/100)["CVaR"] — this is
    the single source of truth for what the notebooks separately called
    "ES10" and "CVaR_0.1" (byte-identical formula, previously computed twice).
    """
    return var_cvar(x, confidence=1.0 - tail_pct / 100.0)["CVaR"]


def dispersion_stats(x):
    """mean, sample std (ddof=1), and coefficient of variation of x."""
    x = _as_array(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    return {"mean": mean, "std": std, "CoV": std / mean}


def variance_reduction(unhedged, hedged):
    """Classic variance-reduction hedge-effectiveness ratio.

    vr = 1 - Var(hedged) / Var(unhedged), both with ddof=1.
    A dimensionless ratio in [~-inf, 1]; 1 = hedge eliminates all variance,
    0 = hedge has no effect on variance, negative = hedge increases variance.

    Not to be confused with `certainty_equivalent_gain`, a monetary
    certainty-equivalent gain (EUR) used by the Lee & Oren (2009) equilibrium
    pricing block — the two measure different things despite both being
    informally called "hedge effectiveness".
    """
    unhedged = _as_array(unhedged)
    hedged = _as_array(hedged)
    var_unh = unhedged.var(ddof=1)
    if var_unh < 1e-8:
        return float("nan")
    return float(1.0 - hedged.var(ddof=1) / var_unh)


def certainty_equivalent(mu, sigma2, risk_aversion):
    """Mean-variance certainty equivalent: CE = mu - (risk_aversion / 2) * sigma2."""
    return mu - (risk_aversion / 2.0) * sigma2


def certainty_equivalent_gain(mu_unhedged, sigma2_unhedged, mu_hedged, sigma2_hedged, risk_aversion):
    """CE(hedged) - CE(unhedged): the monetary certainty-equivalent hedging gain
    used in the Lee & Oren (2009) equilibrium pricing block (e.g. HE_spp).

    Not to be confused with `variance_reduction`, a dimensionless variance
    ratio — the two are unrelated measures that both get called "hedge
    effectiveness" informally.
    """
    ce_unh = certainty_equivalent(mu_unhedged, sigma2_unhedged, risk_aversion)
    ce_hed = certainty_equivalent(mu_hedged, sigma2_hedged, risk_aversion)
    return ce_hed - ce_unh


def _detrend(ri, how):
    """Divide each path by the cross-sectional mean path, isolating the
    stochastic component from the deterministic trend common to all paths."""
    if how is None:
        return ri
    if how != "cross_sectional_mean":
        raise ValueError("detrend must be None or 'cross_sectional_mean'")
    mean_path = ri.mean(axis=0, keepdims=True)
    return np.divide(ri, mean_path, out=np.zeros_like(ri), where=mean_path != 0)


def max_drawdown_per_path(ri_annual_matrix, detrend=None):
    """Worst per-path peak-to-trough decline in annual revenue.

    ri_annual_matrix: (N_SIM, n_years) array of annual revenue per simulated
    path, where each row is the same simulated path traced across years.

    Applied to the annual revenue LEVEL, not its cumulative sum — since revenue
    is non-negative, a cumulative sum is monotonically non-decreasing and would
    make "drawdown" trivially zero. At each year t, drawdown =
    (ri[t] - running_max(ri[:t+1])) / running_max(ri[:t+1]). Returns the most
    negative value per path, shape (N_SIM,); values <= 0.

    IMPORTANT — `detrend`: when the revenue process has a deterministic trend
    common to every path (here, solar cannibalisation drives mean revenue
    steadily down to 2050), raw drawdown mostly measures that trend rather than
    risk: a path can decline monotonically with zero volatility and still post a
    large drawdown. Pass detrend="cross_sectional_mean" to divide each path by
    the mean path first, leaving only the stochastic component. Report both, and
    use `trend_drawdown` to quantify how much of the raw figure is deterministic.
    """
    ri = _detrend(_as_array(ri_annual_matrix), detrend)
    running_max = np.maximum.accumulate(ri, axis=1)
    drawdown = np.divide(
        ri - running_max,
        running_max,
        out=np.zeros_like(ri),
        where=running_max != 0,
    )
    return drawdown.min(axis=1)


def trend_drawdown(ri_annual_matrix):
    """Max drawdown of the cross-sectional MEAN path alone.

    This is the portion of the raw per-path drawdown attributable purely to the
    deterministic trend — i.e. what a zero-volatility path would still show.
    """
    ri = _as_array(ri_annual_matrix)
    mean_path = ri.mean(axis=0)[None, :]
    return float(max_drawdown_per_path(mean_path)[0])


def mean_growth(ri_annual_matrix, detrend=None):
    """Mean year-over-year growth rate per path, shape (N_SIM,)."""
    ri = _detrend(_as_array(ri_annual_matrix), detrend)
    return (ri[:, 1:] / ri[:, :-1] - 1.0).mean(axis=1)


def min_variance_hedge(revenue, payoff):
    """Minimum-variance hedge ratio and the ceiling it implies.

    For a single hedging instrument, no sizing can remove more than rho^2 of
    revenue variance, where rho = corr(revenue, payoff). That ceiling — not the
    variance reduction actually achieved — is what determines whether the
    instrument can work at all.

    Returns {"h_star", "rho", "max_var_red"}; h_star is in the same units as
    the payoff series, so compare it against the notional actually traded.
    """
    r = _as_array(revenue)
    g = _as_array(payoff)
    var_g = g.var(ddof=1)
    if var_g < 1e-15:
        return {"h_star": float("nan"), "rho": float("nan"), "max_var_red": 0.0}
    cov = float(np.cov(r, g, ddof=1)[0, 1])
    rho = cov / np.sqrt(r.var(ddof=1) * var_g)
    return {"h_star": -cov / var_g, "rho": float(rho), "max_var_red": float(rho ** 2)}


def _resolve_target(ri, target):
    """target may be a number, or "expected" to use the deterministic mean
    growth of the cross-sectional mean path."""
    if target != "expected":
        return float(target)
    mean_path = ri.mean(axis=0)
    return float((mean_path[1:] / mean_path[:-1] - 1.0).mean())


def downside_deviation(ri_annual_matrix, target=0.0, detrend=None):
    """Per-path downside deviation of year-over-year revenue growth vs `target`.

    growth[t] = ri[t] / ri[t-1] - 1; dd = sqrt(mean(min(growth - target, 0)^2)).
    Returns shape (N_SIM,) — one Sortino-style denominator per path.

    `target` may be "expected", which uses the deterministic mean growth of the
    cross-sectional mean path instead of 0. With a structurally declining
    revenue process, target=0 counts every year of ordinary expected decline as
    "downside", inflating the denominator with deterministic content.
    """
    ri = _detrend(_as_array(ri_annual_matrix), detrend)
    tgt = _resolve_target(ri, target)
    growth = ri[:, 1:] / ri[:, :-1] - 1.0
    downside = np.minimum(growth - tgt, 0.0)
    return np.sqrt((downside ** 2).mean(axis=1))


def sortino_ratio_per_path(ri_annual_matrix, target=0.0, detrend=None):
    """Per-path Sortino-style ratio: mean(growth - target) / downside_deviation.

    Returns shape (N_SIM,); NaN for paths with zero downside deviation.

    NOTE on `target`: with target=0 on a structurally declining revenue process,
    the numerator is negative for essentially every path, so the ratio is
    mechanically negative and merely restates the assumed decline. Use
    target="expected" (deviation from the EXPECTED decline) or detrend the
    series for the ratio to carry information about risk.
    """
    ri = _detrend(_as_array(ri_annual_matrix), detrend)
    tgt = _resolve_target(ri, target)
    growth = ri[:, 1:] / ri[:, :-1] - 1.0
    excess = growth.mean(axis=1) - tgt
    dd = downside_deviation(ri_annual_matrix, target=target, detrend=detrend)
    return np.divide(excess, dd, out=np.full_like(excess, np.nan), where=dd != 0)


# --------------------------------------------------------------------------- #
# Sampling uncertainty
#
# Every metric above is estimated from a finite set of Monte Carlo paths. VaR95
# is interpolated between two order statistics and CVaR95 averages only the ~5%
# of paths below it, so both are far noisier than their point estimates suggest.
# The functions below quantify that, and are paired where the underlying
# comparison is paired (hedged and unhedged share the same draws).
# --------------------------------------------------------------------------- #

def _bootstrap_indices(n, B, seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def var_cvar_ci(x, confidence=0.95, B=1000, ci=0.95, seed=None):
    """var_cvar plus a bootstrap confidence interval on both VaR and CVaR.

    Returns {"VaR", "VaR_lo", "VaR_hi", "VaR_se", "CVaR", "CVaR_lo", "CVaR_hi",
    "CVaR_se"}. `ci` is the width of the reported interval (0.95 -> 2.5/97.5
    percentiles of the bootstrap distribution); `confidence` is the VaR level.
    """
    x = _as_array(x)
    n = x.size
    point = var_cvar(x, confidence=confidence)
    alpha = 1.0 - confidence

    samples = x[_bootstrap_indices(n, B, seed)]                  # (B, n)
    thr = np.percentile(samples, alpha * 100, axis=1, keepdims=True)
    mask = samples <= thr
    counts = mask.sum(axis=1)
    cvar_boot = np.where(counts > 0, (samples * mask).sum(axis=1) / np.maximum(counts, 1), np.nan)
    var_boot = thr.ravel()

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "VaR": point["VaR"],
        "VaR_lo": float(np.percentile(var_boot, lo_q)),
        "VaR_hi": float(np.percentile(var_boot, hi_q)),
        "VaR_se": float(np.std(var_boot, ddof=1)),
        "CVaR": point["CVaR"],
        "CVaR_lo": float(np.nanpercentile(cvar_boot, lo_q)),
        "CVaR_hi": float(np.nanpercentile(cvar_boot, hi_q)),
        "CVaR_se": float(np.nanstd(cvar_boot, ddof=1)),
    }


def paired_delta_ci(unhedged, hedged, confidence=0.95, B=1000, ci=0.95, seed=None):
    """Bootstrap CI for the hedge's effect on VaR and CVaR, resampling PATHS jointly.

    Hedged and unhedged come from the same simulated paths (common random
    numbers), so the two series must be resampled with the SAME indices —
    otherwise the CI is inflated by variation that the paired design removes.

    Returns {"dVaR", "dVaR_lo", "dVaR_hi", "dCVaR", "dCVaR_lo", "dCVaR_hi",
    "p_dVaR_gt0", "p_dCVaR_gt0"} where d = hedged - unhedged.
    """
    u = _as_array(unhedged)
    h = _as_array(hedged)
    if u.shape != h.shape:
        raise ValueError("unhedged and hedged must be the same length (paired)")
    n = u.size
    alpha = 1.0 - confidence

    idx = _bootstrap_indices(n, B, seed)
    du, dh = [], []
    for arr, out in ((u, du), (h, dh)):
        s = arr[idx]
        thr = np.percentile(s, alpha * 100, axis=1, keepdims=True)
        mask = s <= thr
        counts = np.maximum(mask.sum(axis=1), 1)
        out.append(thr.ravel())
        out.append((s * mask).sum(axis=1) / counts)

    dvar = dh[0] - du[0]
    dcvar = dh[1] - du[1]
    point = var_cvar(h, confidence), var_cvar(u, confidence)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "dVaR": point[0]["VaR"] - point[1]["VaR"],
        "dVaR_lo": float(np.percentile(dvar, lo_q)),
        "dVaR_hi": float(np.percentile(dvar, hi_q)),
        "dCVaR": point[0]["CVaR"] - point[1]["CVaR"],
        "dCVaR_lo": float(np.percentile(dcvar, lo_q)),
        "dCVaR_hi": float(np.percentile(dcvar, hi_q)),
        "p_dVaR_gt0": float((dvar > 0).mean()),
        "p_dCVaR_gt0": float((dcvar > 0).mean()),
    }


def variance_reduction_ci(unhedged, hedged, B=1000, ci=0.95, seed=None):
    """variance_reduction plus a PAIRED bootstrap CI (same indices both series)."""
    u = _as_array(unhedged)
    h = _as_array(hedged)
    if u.shape != h.shape:
        raise ValueError("unhedged and hedged must be the same length (paired)")
    idx = _bootstrap_indices(u.size, B, seed)
    vu = u[idx].var(axis=1, ddof=1)
    vh = h[idx].var(axis=1, ddof=1)
    boot = np.where(vu > 1e-12, 1.0 - vh / vu, np.nan)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "var_red": variance_reduction(u, h),
        "lo": float(np.nanpercentile(boot, lo_q)),
        "hi": float(np.nanpercentile(boot, hi_q)),
        "se": float(np.nanstd(boot, ddof=1)),
    }


def pitman_morgan_test(unhedged, hedged):
    """Pitman-Morgan test of equal variances for PAIRED samples.

    For paired x, y the variances are equal iff corr(x-y, x+y) = 0, which turns
    the variance comparison into a correlation test with n-2 degrees of freedom.
    Valid here precisely because hedged and unhedged share the same draws.

    Returns {"r", "t", "df", "p_two_sided", "var_unhedged", "var_hedged"}.
    p is computed from Student's t if SciPy is available, else NaN.
    """
    u = _as_array(unhedged)
    h = _as_array(hedged)
    if u.shape != h.shape:
        raise ValueError("unhedged and hedged must be the same length (paired)")
    n = u.size
    d = u - h
    s = u + h
    if d.std() == 0 or s.std() == 0:
        # Identical (or perfectly offsetting) series: variances are trivially
        # equal, and corrcoef of a constant vector is undefined.
        return {"r": 0.0, "t": 0.0, "df": n - 2, "p_two_sided": 1.0,
                "var_unhedged": float(u.var(ddof=1)), "var_hedged": float(h.var(ddof=1))}
    r = float(np.corrcoef(d, s)[0, 1])
    denom = 1.0 - r ** 2
    t = float(r * np.sqrt((n - 2) / denom)) if denom > 0 else float("inf")
    try:
        from scipy import stats as _stats
        p = float(2 * _stats.t.sf(abs(t), df=n - 2))
    except Exception:
        p = float("nan")
    return {"r": r, "t": t, "df": n - 2, "p_two_sided": p,
            "var_unhedged": float(u.var(ddof=1)), "var_hedged": float(h.var(ddof=1))}


def shortfall_gap(x, confidence=0.95):
    """mean(x) - CVaR(x): expected shortfall measured from the mean.

    Reporting a change in CVaR as a fraction of the CVaR *level* understates it,
    because the level is dominated by the revenue base rather than by risk. The
    gap below the mean is the part a hedge can actually act on, and is the
    correct denominator for a percentage improvement.
    """
    x = _as_array(x)
    return float(x.mean() - var_cvar(x, confidence=confidence)["CVaR"])
