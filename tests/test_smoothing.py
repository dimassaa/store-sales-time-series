"""Unit tests for src/smoothing.py — synthetic series only.

train.csv is never loaded: every expectation below is hand-computed on a
small fixture, so a wrong window/edge/decay rule fails loudly instead of
hiding behind real-data noise.
"""

import numpy as np
import pandas as pd
import pytest

from src.smoothing import exponential_moving_average, simple_moving_average

# 5-point float fixture on a daily datetime index. Hand expectations for a
# trailing window of 3: NaN, NaN, (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4.
FIVE = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    index=pd.date_range("2020-01-01", periods=5, name="date"),
    name="sales",
)

# 6 points for the even-window centered case (a 5-point fixture cannot show
# the left/right NaN asymmetry of window=4).
SIX = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    index=pd.date_range("2020-01-01", periods=6, name="date"),
    name="sales",
)

# Triangle: symmetric up-down shape makes a wrong centered window obvious
# (phase error moves the peak), 9 points so window=3 leaves both edges short.
TRIANGLE = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    index=pd.date_range("2020-01-01", periods=9, name="date"),
    name="sales",
)


def test_ma_trailing_hand_computed() -> None:
    out = simple_moving_average(FIVE, 3)
    # Incomplete leading window is NaN — never a partial mean.
    assert out.isna().tolist() == [True, True, False, False, False]
    # Each valid point is an integer sum divided by 3, exact in float64
    # (6/3, 9/3, 12/3), so no tolerance is needed.
    assert out.iloc[2] == 2.0
    assert out.iloc[3] == 3.0
    assert out.iloc[4] == 4.0


def test_ma_triangle_hand_computed() -> None:
    out = simple_moving_average(TRIANGLE, 3)
    # Hand trailing means at positions 2..8: 2, 3, 4, 13/3, 4, 3, 2
    # (the last point still has a full window [3,2,1]); only positions 0-1
    # are short -> NaN (full-window contract). First written by hand with a
    # wrong last point — a test-arithmetic error, not a code one (see
    # notes/01 pitfall #3: re-check the hand expectation before the code).
    assert out.isna().tolist() == [True, True] + [False] * 7
    expected = [2.0, 3.0, 4.0, 13 / 3, 4.0, 3.0, 2.0]
    # rel=1e-12: only 13/3 is not exactly representable (a single division
    # rounding); the tolerance covers that rounding and nothing else —
    # a wrong window shifts these means by orders of magnitude more.
    for pos, want in enumerate(expected, start=2):
        assert out.iloc[pos] == pytest.approx(want, rel=1e-12)


def test_ma_constant_series_stays_constant() -> None:
    constant = pd.Series([3.0] * 10, name="sales")
    out = simple_moving_average(constant, 7)
    # Warm-up is NaN; afterwards 7 x 3.0 sums exactly and dividing by 7
    # returns 3.0 exactly in float64 — compared with == on purpose.
    assert out.iloc[:6].isna().all()
    assert (out.iloc[6:] == 3.0).all()


def test_ma_centered_odd_window_shifts_phase() -> None:
    trailing = simple_moving_average(FIVE, 3)
    centered = simple_moving_average(FIVE, 3, center=True)
    # Same interior means, different timestamps: at pos 2 trailing reads
    # mean(1,2,3)=2 (window ends at pos 2) while centered reads
    # mean(2,3,4)=3 (window centered on pos 2) — center removes the lag
    # by borrowing the future, it does not change the smoothing itself.
    assert trailing.iloc[2] == 2.0
    assert centered.iloc[2] == 3.0
    # Odd window: one incomplete point on each edge.
    assert centered.isna().tolist() == [True, False, False, False, True]


def test_ma_centered_even_window_nans_on_both_sides() -> None:
    out = simple_moving_average(SIX, 4, center=True)
    # pandas anchors the window at i - window//2, so window=4 covers
    # [i-2, i+1]: positions 0-1 lack observations on the LEFT and position 5
    # on the RIGHT — both edges are NaN, the extra gap falls on the left.
    assert out.isna().tolist() == [True, True, False, False, False, True]
    # Hand means of [1..4], [2..5], [3..6] = 2.5, 3.5, 4.5 — exact in float64.
    assert out.iloc[2] == 2.5
    assert out.iloc[3] == 3.5
    assert out.iloc[4] == 4.5


def test_ma_integer_input_returns_float_and_keeps_index() -> None:
    ints = pd.Series(
        [1, 2, 3, 4],
        index=pd.Index(["a", "b", "c", "d"], name="day"),
        name="n",
    )
    out = simple_moving_average(ints, 2)
    # A mean is never an integer, even for integer input.
    assert pd.api.types.is_float_dtype(out)
    assert out.index.equals(ints.index)
    assert out.index.name == "day"
    assert out.name == "n"
    assert out.iloc[1] == 1.5


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_ma_rejects_invalid_window(bad: object) -> None:
    # bool is rejected although it subclasses int: True would silently
    # become window=1 and hide the caller's bug (fail-fast at the boundary).
    with pytest.raises(ValueError, match="window"):
        simple_moving_average(FIVE, bad)  # type: ignore[arg-type]


def test_ema_matches_manual_recursion() -> None:
    span = 4
    alpha = 2 / (span + 1)  # 0.4 — the span->alpha mapping from the docstring
    out = exponential_moving_average(FIVE, span)
    # Hand recursion y_t = (1-alpha)*y_{t-1} + alpha*x_t, y_0 = x_0, alpha=0.4:
    # y_1=1.4, y_2=2.04, y_3=0.6*2.04+0.4*4=2.824 (anchor for the loop below).
    y = float(FIVE.iloc[0])
    expected = []
    for x in FIVE:
        y = (1 - alpha) * y + alpha * float(x)
        expected.append(y)
    assert expected[3] == pytest.approx(2.824, rel=1e-12)
    # rel=1e-9: the same closed-form recursion, but pandas evaluates it in C
    # — rounding may differ by ulps; a wrong alpha or adjust flag would move
    # the values by far more than 1e-9 relative.
    assert out.isna().tolist()[: span - 1] == [True] * (span - 1)
    for pos in range(span - 1, len(FIVE)):
        assert out.iloc[pos] == pytest.approx(expected[pos], rel=1e-9)


def test_ema_matches_pandas_reference() -> None:
    out = exponential_moving_average(TRIANGLE, 5)
    reference = TRIANGLE.ewm(span=5, adjust=False, min_periods=5).mean()
    # rel=1e-9 (plan contract): the wrapper must reproduce pandas' documented
    # semantics modulo float noise — a different adjust/min_periods choice
    # would diverge by orders of magnitude, not by ulps. equal_nan keeps the
    # warm-up NaN pattern part of the comparison.
    np.testing.assert_allclose(out.to_numpy(), reference.to_numpy(), rtol=1e-9, equal_nan=True)


def test_ema_warmup_then_gapless() -> None:
    out = exponential_moving_average(TRIANGLE, 3)
    # min_periods=span blanks only the warm-up; unlike MA there is no
    # structural edge NaN at the END — the recursion is defined for every t.
    assert out.isna().tolist() == [True, True] + [False] * 7


def test_ema_preserves_index_and_name() -> None:
    out = exponential_moving_average(FIVE, 3)
    assert out.index.equals(FIVE.index)
    assert out.index.name == "date"
    assert out.name == "sales"
    assert pd.api.types.is_float_dtype(out)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_ema_rejects_invalid_span(bad: object) -> None:
    # Same boundary rule as window: non-int (bool included) or < 1 fails loud.
    with pytest.raises(ValueError, match="span"):
        exponential_moving_average(FIVE, bad)  # type: ignore[arg-type]
