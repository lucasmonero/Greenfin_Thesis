"""Shared helpers for the Modelling / Scenarios notebooks.

Convention used throughout this module (see the notebooks for the prior,
inconsistent conventions this replaces):
  - `find_project_root()` replaces three competing path idioms (
    `Path(os.getcwd())` + Windows raw strings, bare `Path("../../Data")`, and
    a two-directory existence-check fallback) with the one that is actually
    robust to the notebook's working directory, already used in
    `Code/Simulation/05_hedge_contract.ipynb` and every Transformation/EDA
    notebook.
  - `daily_features()` replaces the daily-aggregation block duplicated in
    `01_solar_irradiation.ipynb` and `02_temperature.ipynb` (the solar
    version is a strict subset of the temperature one, and the temperature
    notebook discards the four solar columns it computes) with one function
    both can call, so the two can no longer drift apart.
  - `fourier_design()` replaces four separate re-implementations of the same
    day-of-year harmonic basis across the two marginal-model notebooks, each
    re-hardcoding the annual period.
  - `point_metrics()` replaces the RMSE/MAE/MAPE formula written twice, once
    per holdout section, with the same rationale spelled out in both places.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root() -> Path:
    """Walk up from the cwd until the directory holding Code/ and Data/ is found.

    Notebooks have no __file__, and a bare Path("..") is cwd-relative, so it
    breaks whenever the notebook is run from anywhere but its own folder.
    """
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "Code").is_dir() and (cand / "Data").is_dir():
            return cand
    raise RuntimeError(f"Project root not found above {here}")


def daily_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Daily aggregates shared by the solar and temperature marginal models.

    Always computes GHI_sum, GHI_cs_sum, GHI_var, Kt_daily, year and month.
    Temperature aggregates (t2m_mean/min/max/range) are added automatically
    when the input frame carries a `t2m` column, so the same call serves both
    the solar-only and the temperature notebook.
    """
    dates = df[time_col].dt.date

    agg = {
        "GHI_sum": ("GHI", "sum"),
        "GHI_cs_sum": ("CLEAR_SKY_GHI", "sum"),
        "GHI_var": ("GHI", "var"),
    }
    if "t2m" in df.columns:
        agg.update({
            "t2m_mean": ("t2m", "mean"),
            "t2m_min": ("t2m", "min"),
            "t2m_max": ("t2m", "max"),
        })

    out = df.groupby(dates).agg(**agg).rename_axis("date").reset_index()
    out["Kt_daily"] = out["GHI_sum"] / out["GHI_cs_sum"]
    if "t2m_mean" in out.columns:
        out["t2m_range"] = out["t2m_max"] - out["t2m_min"]

    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    return out


def fourier_design(doy, K: int, period: float = 365.0) -> np.ndarray:
    """Day-of-year Fourier design columns [cos(k*t), sin(k*t)] for k=1..K.

    No constant or trend column — callers that need one (the temperature
    model's linear warming trend) prepend it themselves; the solar model,
    which has no trend term, uses this output directly after adding a
    constant column.
    """
    doy = np.asarray(doy, dtype=float)
    t = 2 * np.pi * doy / period
    cols = []
    for k in range(1, K + 1):
        cols.append(np.cos(k * t))
        cols.append(np.sin(k * t))
    return np.column_stack(cols)


def point_metrics(obs, pred) -> dict:
    """RMSE, MAE and MAPE of `pred` against `obs`.

    The one formula both holdout sections computed independently
    (Solar Irradiation model and Temperature model), differing only in
    whether MAPE was reported.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = obs - pred
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    mape = float((np.abs(err) / np.abs(obs)).mean() * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# ─────────────────────────────────────────────────────────────────────────────
# Simulation stage (Stage 3/4) helpers
#
# The three functions below replace logic that was copy-pasted across the
# Simulation notebooks, in each case with at least one copy that had drifted:
#   - the production chain was written three times with two different values of
#     GAMMA (-0.004 and -0.0043), producing three incompatible "historical
#     modelled RI" series;
#   - the path-layout guard existed in three notebooks and was missing from the
#     fourth, which is how ri_annual_results_multiscenario.parquet came to hold
#     a mix of 1,000-path and 5,000-path scenario cells.
# ─────────────────────────────────────────────────────────────────────────────

# Physical constants of the 1 kWp reference array. Definitional values (G_STC,
# T_STC) and datasheet values (GAMMA, PR, NOCT) — see the Decision Log in
# Code/Simulation/03_revenue_index_historical.ipynb for the sourcing of each.
G_STC = 1000.0    # W/m2   — standard test condition irradiance (definitional)
T_STC = 25.0      # degC   — STC reference temperature (definitional)
GAMMA = -0.0043   # /degC  — temperature coefficient, monocrystalline Si (D2)
PR = 0.80         # —      — performance ratio, ex-temperature (D1)
NOCT = 45.0       # degC   — nominal operating cell temperature (D5)


def production_chain(ghi, t_amb, uplift, gamma=GAMMA, pr=PR,
                     g_stc=G_STC, t_stc=T_STC):
    """The four production formulas, applied once so they cannot drift apart.

        T_eff  = t_amb + uplift                     [degC]
        f_temp = 1 + gamma * (T_eff - t_stc)        [-]
        P      = (ghi / g_stc) * f_temp * pr        [kWh/kWp/day]

    `ghi` is a DAILY TOTAL in Wh/m2, so `ghi / g_stc` is equivalent peak-sun-
    hours: the kWh a lossless 1 kWp array would produce that day.

    Works unchanged on numpy arrays and pandas Series. `uplift` broadcasts, so
    a (N_DAYS,) seasonal profile can be applied to an (N_SIM, N_DAYS) grid of
    temperature paths by passing `uplift[np.newaxis, :]`.

    Returns a dict with keys "T_eff", "f_temp", "P" — RI = P * CR is left to
    the caller, since the capture rate is the one input that is not shared
    between the historical and simulated versions of this chain.
    """
    t_eff = t_amb + uplift
    f_temp = 1 + gamma * (t_eff - t_stc)
    p = (ghi / g_stc) * f_temp * pr
    return {"T_eff": t_eff, "f_temp": f_temp, "P": p}


def uplift_for_calendar(uplift_by_doy: dict, doys) -> np.ndarray:
    """Map a {doy: uplift} profile onto a calendar, as a (len(doys),) array.

    Falls back to the annual mean for any day-of-year the profile is missing,
    which is what every caller did inline with its own copy of the default.
    """
    default = float(np.mean(list(uplift_by_doy.values())))
    return np.array([uplift_by_doy.get(int(d), default) for d in doys])


def assert_path_layout(df: pd.DataFrame, name: str = "frame",
                       n_sim: int = None, n_days: int = None,
                       id_col: str = "sim_id", date_col: str = "date"):
    """Verify a long simulation frame can be reshaped to (N_SIM, N_DAYS) safely.

    Checks, in the order that produces the most useful failure message:
      1. every path has the same number of rows (otherwise reshape interleaves
         paths instead of raising);
      2. the total row count is exactly N_SIM x N_DAYS;
      3. all paths share one date grid;
      4. the frame is in sim-major order, so reshape(N_SIM, N_DAYS) puts one
         path per row.

    Pass `n_sim` / `n_days` to also pin the expected dimensions — this is the
    check whose absence let a 1,000-path file and a 5,000-path file be combined
    into one results artifact. Returns the verified (N_SIM, N_DAYS).
    """
    sizes = df.groupby(id_col, observed=True).size()
    if sizes.nunique() != 1:
        raise ValueError(
            f"STOP — {name}: paths have differing lengths (min {sizes.min()}, "
            f"max {sizes.max()}). Cannot reshape safely."
        )
    got_sim, got_days = int(sizes.size), int(sizes.iloc[0])

    if len(df) != got_sim * got_days:
        raise ValueError(
            f"STOP — {name}: {len(df):,} rows is not N_SIM x N_DAYS "
            f"({got_sim} x {got_days} = {got_sim * got_days:,})."
        )
    if n_sim is not None and got_sim != n_sim:
        raise ValueError(
            f"STOP — {name}: {got_sim} paths, expected {n_sim}. Mixing path "
            f"counts across scenarios silently compares Monte Carlo estimates "
            f"at different precision — re-run the reconstruction so every "
            f"scenario file has the same N_SIM."
        )
    if n_days is not None and got_days != n_days:
        raise ValueError(
            f"STOP — {name}: {got_days} days per path, expected {n_days}."
        )

    n_dates = df[date_col].nunique()
    if n_dates != got_days:
        raise ValueError(
            f"STOP — {name}: {n_dates} distinct dates for {got_days} days per "
            f"path; the paths do not share one date grid."
        )

    ids = df[id_col].to_numpy()
    if not (ids[::got_days] == np.arange(got_sim)).all() or not (
            ids.reshape(got_sim, got_days) == ids[::got_days][:, None]).all():
        raise ValueError(
            f"STOP — {name}: rows are not in sim-major order. Sort by "
            f"[{id_col}, {date_col}] before reshaping."
        )

    return got_sim, got_days
