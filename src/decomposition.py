"""STL decomposition helpers with national-holiday context.

Stage 3 splits each daily series (Stage 1 output) into trend + seasonal +
resid with STL, quantifies how much variation each component absorbs via the
Wang/Smith/Hyndman strength statistics, and decides additive vs multiplicative
from a log-proxy fit. Holiday annotations and the model choice live here so
notebooks interpret results instead of re-deriving them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl_decompose(series: pd.Series, period: int = 7, robust: bool = True) -> pd.DataFrame:
    """Season-trend decomposition via STL (statsmodels), additive by construction.

    STL (Seasonal-Trend decomposition using LOESS, Cleveland et al. 1990) fits
    each component with local regression instead of fixed seasonal means, so a
    single holiday spike deforms only its neighborhood rather than the whole
    seasonal band or trend. The decomposition is additive by construction:
    observed = trend + seasonal + resid — which is exactly why a multiplicative
    model is later *proxied* by STL on ``log(series)`` instead of a separate
    estimator.

    ``period=7`` is the weekly seasonality measured in Stage 1 (Sun and Sat
    well above midweek): STL's seasonal component cycles with this many days,
    and any other period would leave part of the weekday rhythm inside
    ``resid``.

    ``robust=True`` enables STL's iterated re-weighted fitting pass, which
    down-weights outliers. Retail data is full of them (Christmas closures are
    0-filled, New Year bursts spike the series), and the non-robust LOESS pass
    would let those events bend the trend around the holidays.

    Parameters
    ----------
    series:
        Daily sales with a continuous ``DatetimeIndex``; no missing values,
        no duplicate dates. Must be at least ``2 * period + 1`` points so STL
        can form a full seasonal cycle.
    period:
        Seasonal period in days (``period=7`` = weekly). ``int >= 2``.
    robust:
        Use the outlier-robust LOESS pass.

    Returns
    -------
    pd.DataFrame
        Columns ``trend``, ``seasonal``, ``resid`` on the original index.
        The columns sum exactly back to the input (floating-point rounding
        aside), which tests verify.

    Raises
    ------
    ValueError
        If the series is not a continuous daily grid (gaps, duplicates,
        NaN) or is shorter than ``2 * period + 1`` points.
    TypeError
        If ``series`` is not a ``pd.Series`` with a ``DatetimeIndex``.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"series must be a pd.Series, got {type(series).__name__}")
    if isinstance(period, bool) or not isinstance(period, int) or period < 2:
        raise ValueError(f"period must be an int >= 2, got {period!r}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex (daily dates)")
    if series.isna().any():
        raise ValueError("series contains NaN/NaT values; fill or drop them before STL")
    index = series.index
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("series index must be unique and sorted ascending")
    # STL assumes a regular grid; a hole would silently mis-align the
    # seasonal cycle, so demand the index equal the full daily range.
    expected = pd.date_range(index.min(), index.max(), freq="D")
    if not index.equals(expected):
        raise ValueError(
            f"index is not a continuous daily grid: {len(expected)} calendar days "
            f"expected between {index.min().date()} and {index.max().date()}"
        )
    min_len = 2 * period + 1
    if len(series) < min_len:
        raise ValueError(
            f"series length {len(series)} is below the STL minimum 2*period+1 = {min_len}"
        )
    fit = STL(series, period=period, robust=robust).fit()
    return pd.DataFrame(
        {"trend": fit.trend, "seasonal": fit.seasonal, "resid": fit.resid},
        index=series.index,
    )


def _variance_strength(resid: pd.Series, explained_sum: pd.Series, part: str) -> float:
    """Shared 1 - var(resid)/var(resid + component) computation.

    ``explained_sum`` is ``resid + seasonal`` (or ``resid + trend``); when its
    variance is zero the ratio is undefined. For a constant series STL returns
    all-zero components (0/0), which we define as perfect strength: there is
    no variation left to explain. The ``part`` label only names the component
    in the error message.
    """
    var_resid = float(resid.var())
    var_total = float(explained_sum.var())
    if var_total == 0.0:
        # Degenerate 0/0: resid + <part> is constant, so the <part> band (and
        # typically the residual too, as on a constant series) carries no
        # variation to explain; strength is taken as 1.0 instead of NaN.
        return 1.0
    return max(0.0, 1.0 - var_resid / var_total)


def seasonality_strength(components: pd.DataFrame) -> float:
    """Strength of seasonality F_s = max(0, 1 - var(resid)/var(resid + seasonal)).

    Wang, Smith & Hyndman (2006): the seasonal band absorbs a share of the
    detrended variation equal to 1 minus the residual's share of the combined
    ``resid + seasonal`` spread. F_s = 1 means the seasonal component explains
    everything the trend leaves behind; F_s = 0 that it explains none of it.
    ``max(0, ...)`` clamps the estimator when a model over-fits (ratio > 1 from
    sampling noise); the true population value lies in [0, 1].

    Parameters
    ----------
    components:
        Output of ``stl_decompose`` (columns ``trend``, ``seasonal``, ``resid``).

    Returns
    -------
    float
        F_s in [0, 1]; 1.0 for the degenerate zero-variance case.
    """
    resid = components["resid"].astype(float)
    return _variance_strength(resid, resid + components["seasonal"].astype(float), "seasonal")


def trend_strength(components: pd.DataFrame) -> float:
    """Strength of trend F_t = max(0, 1 - var(resid)/var(resid + trend)).

    Same variance-ratio logic as ``seasonality_strength``, applied to the
    trend band (Wang, Smith & Hyndman 2006). F_t near 1 means the trend
    accounts for almost all non-seasonal movement; near 0 that the series has
    no lasting level change relative to its residual wobble.

    Parameters
    ----------
    components:
        Output of ``stl_decompose`` (columns ``trend``, ``seasonal``, ``resid``).

    Returns
    -------
    float
        F_t in [0, 1]; 1.0 for the degenerate zero-variance case.
    """
    resid = components["resid"].astype(float)
    return _variance_strength(resid, resid + components["trend"].astype(float), "trend")


def decompose_additive_vs_multiplicative(series: pd.Series, period: int = 7) -> dict:
    """Fit additive and multiplicative-proxy STL models and recommend one.

    Additive: plain ``stl_decompose`` on the original scale. Multiplicative
    proxy: STL on ``log(series)`` (additive in log space approximates
    multiplicative in original space), with trend/seasonal mapped back via
    ``exp`` so ``trend * seasonal`` reads as the deterministic multiplicative
    prediction. This is a *proxy*: true multiplicative STL would need a
    dedicated implementation, and exp-of-log residuals are not exactly the
    original additive residual (see notes/03, pitfalls).

    ``log(0)`` is undefined, so every value must be strictly positive. The
    policy for structural zeros (the 0-filled Christmas closures in the
    Stage-1 daily series) is the caller's decision: the notebook replaces
    exactly those days with neighbor means so both branches see the identical
    input. Raising here keeps the function fail-fast at its boundary instead
    of silently inventing a zero policy.

    Returns a dict with documented keys:

    ``result["additive"]`` / ``result["multiplicative"]`` each carry
    ``trend``, ``seasonal``, ``resid`` (the multiplicative resid is the
    original-scale reconstruction error ``y - trend*seasonal`` — it is *not*
    ``exp(log-resid)``, which would be a unitless factor and break the
    comparability of ``resid_relative_std``), plus ``seasonality_strength``,
    ``trend_strength`` and ``resid_relative_std = resid.std() / series.mean()``.

    Strengths need care across models: the formula is defined for an additive
    decomposition, and the multiplicative fit is only additive in *log* space,
    so its strengths are computed on the log-space components. ``resid_relative_std``
    is the one quantity measured identically on both models (same input, same
    sales units), which is why the recommendation rule uses it.

    Recommendation rule: pick the model with the smaller ``resid_relative_std``
    (less relative unexplained variation); on a tie, prefer ``additive`` as
    the simpler model on the original scale.

    Parameters
    ----------
    series:
        Strictly positive daily series (``log`` domain); otherwise the same
        contract as ``stl_decompose``.
    period:
        Seasonal period passed through to ``stl_decompose``.

    Returns
    -------
    dict
        ``{"additive": {...}, "multiplicative": {...}, "recommendation": str}``
        as documented above; ``recommendation`` is ``"additive"`` or
        ``"multiplicative"``.

    Raises
    ------
    ValueError
        If any value is <= 0 (log undefined).
    """
    ser = series.astype(float)
    n_non_positive = int((ser <= 0).sum())
    if n_non_positive:
        raise ValueError(
            f"multiplicative proxy takes log(series) and log(0) is undefined: "
            f"{n_non_positive} non-positive values found; the caller must "
            f"prepare a strictly positive input first"
        )
    add = stl_decompose(ser, period=period)
    log_comp = stl_decompose(np.log(ser), period=period)
    # exp-back only trend/seasonal for comparability of shapes; resid is the
    # original-scale deviation from the deterministic multiplicative fit so
    # resid_relative_std stays comparable with the additive branch.
    mult_trend = np.exp(log_comp["trend"]).rename("trend")
    mult_seasonal = np.exp(log_comp["seasonal"]).rename("seasonal")
    mult_resid = (ser - mult_trend * mult_seasonal).rename("resid")

    # Strengths for the multiplicative branch come from the log-space fit:
    # the formula is defined for additive components, and the multiplicative
    # model is only additive in log space.
    add_entry = {
        "trend": add["trend"],
        "seasonal": add["seasonal"],
        "resid": add["resid"],
        "seasonality_strength": seasonality_strength(add),
        "trend_strength": trend_strength(add),
        "resid_relative_std": float(add["resid"].std() / ser.mean()),
    }
    mult_entry = {
        "trend": mult_trend,
        "seasonal": mult_seasonal,
        "resid": mult_resid,
        "seasonality_strength": seasonality_strength(log_comp),
        "trend_strength": trend_strength(log_comp),
        "resid_relative_std": float(mult_resid.std() / ser.mean()),
    }
    add_rrs = add_entry["resid_relative_std"]
    mult_rrs = mult_entry["resid_relative_std"]
    recommendation = "multiplicative" if mult_rrs < add_rrs else "additive"
    return {
        "additive": add_entry,
        "multiplicative": mult_entry,
        "recommendation": recommendation,
    }


def national_holidays(holidays_path: str, start: str, end: str) -> pd.DataFrame:
    """National holidays in ``holidays_events.csv`` within ``[start, end]``.

    Only ``locale == "National"`` rows are returned: local/regional holidays
    do not affect the network-wide series, and the ``transferred`` column is
    deliberately ignored (a transferred holiday is still a national holiday on
    its official date for annotation purposes).

    Parameters
    ----------
    holidays_path:
        Path to ``holidays_events.csv`` (columns date, type, locale,
        locale_name, description, transferred).
    start, end:
        Inclusive date range; anything pandas can parse as a date. Holidays
        outside the range are dropped because the STL series only spans
        ``[start, end]``.

    Returns
    -------
    pd.DataFrame
        Two columns, ``date`` and ``description``, sorted by date (stable, so
        same-day rows keep source order); index reset to 0..n-1.
    """
    holidays = pd.read_csv(
        Path(holidays_path),
        usecols=["date", "locale", "description"],
        parse_dates=["date"],
    )
    if "locale" not in holidays.columns or "description" not in holidays.columns:
        raise ValueError(f"{holidays_path} is missing expected columns (date/locale/description)")
    mask = (holidays["locale"] == "National") & holidays["date"].between(
        pd.Timestamp(start), pd.Timestamp(end)
    )
    subset = holidays.loc[mask, ["date", "description"]]
    return subset.sort_values("date", kind="stable").reset_index(drop=True)