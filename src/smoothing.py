"""Smoothing helpers: simple (MA) and exponential (EMA) moving averages.

Stage 1 showed a strong 7-day cycle in the network daily series; these two
functions are the only smoothers this project is allowed to use (plan ruling
— no ARIMA/spectral extras). Edge and warm-up rules live here once, next to
their tests, so notebooks never re-implement them inline.
"""

from __future__ import annotations

import pandas as pd


def simple_moving_average(series: pd.Series, window: int, center: bool = False) -> pd.Series:
    """Simple moving average over exactly ``window`` observations, NaN at incomplete edges.

    The window order *is* the smoothing period: every reported point is the
    plain mean of ``window`` observations, so window=7 is the natural "trend
    without weekly noise" read (one full week per point) and window=30 a
    roughly monthly level. ``min_periods=window`` is what enforces that
    guarantee — an incomplete window becomes NaN instead of a mean over fewer
    points, which would silently bias the edges toward recent data.

    ``center=False`` (default, trailing) timestamps each mean at the window's
    right edge: causal, usable as "level up to day t", but it inherits a
    ``(window - 1) / 2``-observation phase lag. ``center=True`` timestamps
    the mean at the window's middle — no lag for plots, but every point then
    borrows future observations (display-only, not a causal filter).

    Parameters
    ----------
    series:
        Input values; the index and ``name`` are preserved.
    window:
        Observations per mean. Must be an ``int >= 1``; ``bool`` is rejected
        explicitly because it subclasses ``int`` and ``True`` would silently
        mean window=1.
    center:
        Align the window to its middle instead of its right edge.

    Returns
    -------
    pd.Series
        Float series on the original index. NaN on the first ``window - 1``
        points when trailing, and on the first ``window // 2`` and last
        ``(window - 1) // 2`` points when centered — pandas anchors an even
        window at ``i - window // 2``, so the extra incomplete observation
        falls on the left edge.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be an int >= 1, got {window!r}")
    # min_periods=window: report a full-window mean or NaN, never a partial
    # mean that would quietly change the smoothing order at the edges.
    return series.rolling(window=window, center=center, min_periods=window).mean()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average parameterized by pandas' ``span``.

    ``span`` is the EMA counterpart of an MA window: the decay is
    ``alpha = 2 / (span + 1)`` and the mean age of the weights is
    ``(span - 1) / 2`` observations — the same average lag as a trailing MA
    with ``window = span``, but shaped differently: each new point enters
    with weight ``alpha`` while every older point keeps a positive,
    geometrically shrinking tail. EMA therefore reacts faster to a level
    change than the equal-weight MA of the same order, yet never fully
    forgets the past (infinite memory) — that is the "lag" seen against the
    raw series.

    ``adjust=False`` keeps the textbook recursion honest:
    ``y[t] = (1 - alpha) * y[t-1] + alpha * x[t]`` seeded with ``y[0] = x[0]``.
    ``adjust=True`` would instead renormalize a truncated weight history —
    a different estimator, not the recursion above, and the notebooks/tests
    would be checking a moving target.

    Unlike MA, the recursion itself is defined from observation 0 (there is
    no "incomplete window" concept, hence no structural NaN edges).
    ``min_periods=span`` still blanks the first ``span - 1`` outputs so the
    warm-up matches an MA of the same order and notebook comparisons are
    fair; after warm-up the series is gapless — but every value lags the raw
    series by the weight age above.

    Parameters
    ----------
    series:
        Input values; the index and ``name`` are preserved.
    span:
        EMA span (``alpha = 2 / (span + 1)``). Must be an ``int >= 1``;
        ``bool`` is rejected for the same reason as in
        ``simple_moving_average``.

    Returns
    -------
    pd.Series
        Float series on the original index; NaN only during the warm-up
        (first ``span - 1`` points).

    Raises
    ------
    ValueError
        If ``span`` is not an integer >= 1.
    """
    if isinstance(span, bool) or not isinstance(span, int) or span < 1:
        raise ValueError(f"span must be an int >= 1, got {span!r}")
    # adjust=False + min_periods=span is the documented contract: honest
    # recursion, warm-up aligned with an MA of the same order.
    return series.ewm(span=span, adjust=False, min_periods=span).mean()
