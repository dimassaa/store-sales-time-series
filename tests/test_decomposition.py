"""Unit tests for src/decomposition.py — synthetic fixtures only.

train.csv is never loaded. Expectations are hand-verified on seeded synthetic
series; tolerances carry their reason inline (project rule: a tolerance must
say what it covers and what a correct-wrong split would look like).
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import STL

from src.decomposition import (
    decompose_additive_vs_multiplicative,
    national_holidays,
    seasonality_strength,
    stl_decompose,
    trend_strength,
)

# Fixed seed so every run sees the same series (reproducibility golden
# standard; only the noise draws are random, everything else is exact math).
RNG = np.random.default_rng(7)


def daily_synth(n: int = 300, amp: float = 50.0, noise: float = 5.0, base: float = 1000.0) -> pd.Series:
    """Additive synthetic: trend-less seasonal + Gaussian noise.

    Hand-checkable structure: y = base + amp*sin(2*pi*day/7) + noise. The
    period-7 sine is the seasonality STL must recover; amp/noise set the
    signal-to-noise ratio the strength tests rely on.
    """
    t = np.arange(n)
    y = base + amp * np.sin(2 * np.pi * t / 7) + RNG.normal(0, noise, n)
    return pd.Series(y, index=pd.date_range("2020-01-01", periods=n, name="date"), name="sales")


def multiplicative_synth(n: int = 300) -> pd.Series:
    """Multiplicative synthetic: level*seasonal*log-normal noise, all positive.

    The level grows (base + 0.5*t) so an additive fit cannot absorb the
    seasonal factor: its reconstruction error grows with the level, which is
    exactly the signal ``decompose_additive_vs_multiplicative`` should read as
    "pick multiplicative".
    """
    t = np.arange(n)
    level = 1000.0 + 0.5 * t
    factor = 1 + 0.05 * np.sin(2 * np.pi * t / 7)
    y = level * factor * np.exp(RNG.normal(0, 0.02, n))
    return pd.Series(y, index=pd.date_range("2020-01-01", periods=n, name="date"), name="sales")


# --- stl_decompose ---------------------------------------------------------


def test_stl_components_reconstruct_series() -> None:
    s = daily_synth()
    comp = stl_decompose(s)
    assert list(comp.columns) == ["trend", "seasonal", "resid"]
    assert comp.index.equals(s.index)
    # Additive identity: trend + seasonal + resid == observed. rel=1e-8 is
    # looser than the ~1e-15 float round-off we measure (see below) but fails
    # loudly if STL ever returns non-additive components (e.g. a period/robust
    # plumbing bug that mixes scales); a broken identity is off by orders.
    recon = comp["trend"] + comp["seasonal"] + comp["resid"]
    assert (recon / s.mean()).sub(s / s.mean()).abs().max() < 1e-8


def test_stl_seasonal_component_captures_period() -> None:
    s = daily_synth()
    comp = stl_decompose(s)
    # A period-7 seasonal component must be ~7 distinct weekday values that
    # repeat every week: the weekday means of the seasonal component should
    # match the generating sine in *shape* — but STL's LOESS seasonal filter
    # is free to land at any weekday phase (measured: +5-day shift), so the
    # assertion takes the best correlation over the 7 possible cyclic shifts
    # rather than demanding the exact phase.
    weekday_mean = comp["seasonal"].groupby(comp.index.dayofweek).mean().to_numpy(dtype=float, copy=True)
    weekday_mean -= weekday_mean.mean()
    phase = np.sin(2 * np.pi * np.arange(7) / 7)
    phase -= phase.mean()
    best = max(np.corrcoef(np.roll(weekday_mean, shift), phase)[0, 1] for shift in range(7))
    # Noise std is 10% of the seasonal amplitude, so the recovered pattern is
    # near-perfect (0.9999 measured at the best shift); 0.9 leaves room for
    # LOESS edge effects without being a tautology.
    assert best > 0.9
    # Seasonality must dominate residual spread for this SNR: otherwise the
    # component captured noise, not the period.
    assert comp["seasonal"].std() > 5 * comp["resid"].std()


def test_stl_cross_check_against_statsmodels() -> None:
    s = daily_synth(n=120)
    comp = stl_decompose(s, period=7, robust=True)
    reference = STL(s, period=7, robust=True).fit()
    # The wrapper IS this estimator, so components must match bit-for-near:
    # this test guards the API contract (fit.trend/.seasonal/.resid fields,
    # df/Series plumbing) against a statsmodels upgrade, not re-derives the
    # algorithm. rtol=1e-12 covers float reordering only.
    np.testing.assert_allclose(comp["trend"].to_numpy(), reference.trend, rtol=1e-12)
    np.testing.assert_allclose(comp["seasonal"].to_numpy(), reference.seasonal, rtol=1e-12)
    np.testing.assert_allclose(comp["resid"].to_numpy(), reference.resid, rtol=1e-12)


def test_stl_rejects_series_shorter_than_2p1() -> None:
    short = daily_synth(n=14)  # 2*7 + 1 = 15
    with pytest.raises(ValueError, match="2\\*period\\+1"):
        stl_decompose(short, period=7)


def test_stl_rejects_irregular_grid() -> None:
    s = daily_synth(n=40)
    gap = s.drop(s.index[10])
    with pytest.raises(ValueError, match="continuous daily grid"):
        stl_decompose(gap)


def test_stl_rejects_duplicate_dates() -> None:
    s = daily_synth(n=40)
    dup = pd.concat([s, s.iloc[[5]]]).sort_index()
    with pytest.raises(ValueError, match="unique and sorted"):
        stl_decompose(dup)


def test_stl_rejects_nan_and_naive_index() -> None:
    s = daily_synth(n=40)
    s_nan = s.copy()
    s_nan.iloc[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        stl_decompose(s_nan)
    plain = s.reset_index(drop=True)
    plain.name = "sales"
    with pytest.raises(TypeError, match="DatetimeIndex"):
        stl_decompose(plain)


@pytest.mark.parametrize("bad", [0, 1, 2.5, True, None])
def test_stl_rejects_invalid_period(bad: object) -> None:
    with pytest.raises(ValueError, match="period"):
        stl_decompose(daily_synth(n=40), period=bad)  # type: ignore[arg-type]


# --- strength statistics ----------------------------------------------------


def test_strengths_in_range_and_extreme_on_synthetic() -> None:
    seasonal = stl_decompose(daily_synth(amp=50.0, noise=5.0))
    noise = stl_decompose(daily_synth(amp=0.0, noise=50.0))
    for comp in (seasonal, noise):
        for strength in (seasonality_strength(comp), trend_strength(comp)):
            assert 0.0 <= strength <= 1.0
    # Strong seasonality: SNR is 10x, so the seasonal band should absorb the
    # vast majority of the detrended variation. 0.9 is far above the pure
    # noise case measured below (0.24), short of being a 1.0 tautology.
    assert seasonality_strength(seasonal) > 0.9
    # Pure noise: STL's LOESS seasonal smoother still digests some noise
    # (~0.24 measured), so assert well below seasonality but not zero.
    assert seasonality_strength(noise) < 0.5


def test_strength_degenerate_zero_variance_returns_one() -> None:
    # Hand-built zero-variance components: STL on a constant series returns
    # near-zero (not exactly zero) variances and the stats come out noisy
    # (see notes/03 pitfall); the 0/0 guard is tested here on exact zeros.
    zeros = pd.DataFrame(
        {"trend": 0.0, "seasonal": 0.0, "resid": 0.0},
        index=pd.date_range("2020-01-01", periods=20),
    )
    assert seasonality_strength(zeros) == 1.0
    assert trend_strength(zeros) == 1.0


def test_strength_requires_decomposition_columns() -> None:
    bogus = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    with pytest.raises(KeyError):
        seasonality_strength(bogus)  # type: ignore[arg-type]


# --- additive vs multiplicative ----------------------------------------------


def test_decompose_additive_vs_mult_keys_and_recommendation() -> None:
    s = multiplicative_synth()
    out = decompose_additive_vs_multiplicative(s)
    assert set(out) == {"additive", "multiplicative", "recommendation"}
    for model in ("additive", "multiplicative"):
        entry = out[model]
        assert set(entry) == {
            "trend",
            "seasonal",
            "resid",
            "seasonality_strength",
            "trend_strength",
            "resid_relative_std",
        }
        assert entry["trend"].index.equals(s.index)
        assert entry["seasonal"].index.equals(s.index)
        assert entry["resid"].index.equals(s.index)
        assert 0.0 <= entry["seasonality_strength"] <= 1.0
        assert 0.0 <= entry["trend_strength"] <= 1.0
        assert entry["resid_relative_std"] >= 0.0
    # On a multiplicative-generating series the level-proportional mismatch of
    # the additive fit dominates, so the relative residual must be smaller for
    # the multiplicative model — this is the documented recommendation rule.
    assert out["additive"]["resid_relative_std"] > out["multiplicative"]["resid_relative_std"]
    assert out["recommendation"] == "multiplicative"


def test_decompose_mult_components_are_consistent() -> None:
    s = multiplicative_synth()
    out = decompose_additive_vs_multiplicative(s)
    mult = out["multiplicative"]
    # Documented identity: upstream, resid is stored as y - trend*seasonal on
    # the original scale (not the log residual), so the three components
    # reconstruct the series through that identity. exact (float-equal within
    # ulp) because the same subtraction is replayed.
    assert (s - mult["trend"] * mult["seasonal"] - mult["resid"]).abs().max() == pytest.approx(0.0, abs=1e-9)
    # The exp-back seasonal factor is a unitless multiplier near 1 for the
    # unit-mean synthetic factor — a sanity check that the log-space fit
    # indeed came back to the original scale.
    assert abs(float(mult["seasonal"].mean()) - 1.0) < 0.01


def test_decompose_raises_on_non_positive_values() -> None:
    s = daily_synth(n=40)
    s.iloc[2] = 0.0
    with pytest.raises(ValueError, match="log"):
        decompose_additive_vs_multiplicative(s)


# --- national holidays --------------------------------------------------------


def test_national_holidays_filters_locale_and_range(tmp_path) -> None:
    csv = tmp_path / "holidays.csv"
    rows = [
        "date,type,locale,locale_name,description,transferred",
        "2013-01-01,Holiday,National,Ecuador,Navidad,False",
        "2013-01-02,Holiday,Local,Quito,local day,False",
        "2013-06-01,Holiday,National,Ecuador,rare national,False",
        "2013-06-02,Holiday,Regional,Guayas,regional day,False",
        "2013-12-31,Holiday,National,Ecuador,NYE,False",
        "2013-06-01,Holiday,National,Ecuador,block transfer,True",
    ]
    csv.write_text("\n".join(rows))
    out = national_holidays(str(csv), "2013-01-02", "2013-06-30")
    # Only National within [start, end]: same-day rows in file order are kept.
    assert list(out.columns) == ["date", "description"]
    assert out["date"].dtype.kind == "M"  # datetime64, ready for plotting
    assert list(out["description"]) == ["rare national", "block transfer"]
    assert out.index.tolist() == [0, 1]
    # Transferred rows are NOT analysed (ruling): block transfer survives the
    # National filter unchanged — filtering on transferred is deliberately out.
    assert (out["date"].isna()).sum() == 0


def test_national_holidays_sorted_and_empty_outside_range(tmp_path) -> None:
    csv = tmp_path / "holidays.csv"
    csv.write_text(
        "date,type,locale,locale_name,description,transferred\n"
        "2013-05-24,Holiday,National,Ecuador,Pichincha,False\n"
        "2013-08-10,Holiday,National,Ecuador,Grito,False\n"
    )
    out = national_holidays(str(csv), "2013-08-01", "2013-12-31")
    assert list(out["description"]) == ["Grito"]  # Pichincha dropped by range
    empty = national_holidays(str(csv), "2014-01-01", "2014-12-31")
    assert empty.empty