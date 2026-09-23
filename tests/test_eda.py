"""Unit tests for src/eda.py — hand-built Kaggle-schema fixtures only.

train.csv (~122 MB) is never loaded here: tests must stay fast and hermetic,
so every case runs on a 16-row synthetic frame whose aggregates are known by
hand (values chosen to be exactly representable in float32).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.eda import (
    assert_integrity,
    daily_network_sales,
    integrity_report,
    load_train,
    pick_stl_stores,
    promotion_share,
    store_daily_sales,
    store_type,
)

# 3 consecutive days, 5 stores (types A, A, D, E, B), 2 families.
# Hand-checked aggregates used throughout the tests:
#   daily network  = 25 / 18 / 14.5
#   store totals   = 1:28 (A), 2:6 (A), 3:7 (D), 4:4.5 (E), 5:12 (B)
TRAIN_ROWS = [
    # id, date, store_nbr, family, sales, onpromotion
    [0, "2013-01-01", 1, "GROCERY", 10.0, 0],
    [1, "2013-01-01", 1, "BEVERAGES", 5.0, 1],
    [2, "2013-01-01", 2, "GROCERY", 1.0, 0],
    [3, "2013-01-01", 3, "GROCERY", 2.0, 0],
    [4, "2013-01-01", 4, "GROCERY", 3.0, 1],
    [5, "2013-01-01", 5, "GROCERY", 4.0, 0],
    [6, "2013-01-02", 1, "GROCERY", 6.0, 0],
    [7, "2013-01-02", 2, "GROCERY", 2.0, 1],
    [8, "2013-01-02", 3, "GROCERY", 1.0, 0],
    [9, "2013-01-02", 4, "GROCERY", 1.0, 0],
    [10, "2013-01-02", 5, "GROCERY", 8.0, 1],
    [11, "2013-01-03", 1, "GROCERY", 7.0, 0],
    [12, "2013-01-03", 2, "GROCERY", 3.0, 0],
    [13, "2013-01-03", 3, "GROCERY", 4.0, 1],
    [14, "2013-01-03", 4, "GROCERY", 0.5, 0],
    [15, "2013-01-03", 5, "GROCERY", 0.0, 0],
]

# Full Kaggle stores.csv schema; the loader must join only `type` from it.
STORE_ROWS = [
    # store_nbr, city, state, type, cluster
    [1, "Quito", "Pichincha", "A", 1],
    [2, "Quito", "Pichincha", "A", 2],
    [3, "Cuenca", "Azuay", "D", 3],
    [4, "Ambato", "Tungurahua", "E", 4],
    [5, "Quito", "Pichincha", "B", 1],
]


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write train.csv plus sibling stores.csv so load_train's join runs."""
    train = pd.DataFrame(
        TRAIN_ROWS,
        columns=["id", "date", "store_nbr", "family", "sales", "onpromotion"],
    )
    stores = pd.DataFrame(
        STORE_ROWS,
        columns=["store_nbr", "city", "state", "type", "cluster"],
    )
    train.to_csv(tmp_path / "train.csv", index=False)
    stores.to_csv(tmp_path / "stores.csv", index=False)
    return tmp_path / "train.csv"


@pytest.fixture
def clean_df(sample_csv: Path) -> pd.DataFrame:
    """The typed, type-joined frame every non-gap test runs on."""
    return load_train(sample_csv)


@pytest.fixture
def gap_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Frame with 2013-01-02 removed — mirrors the real dataset's Christmas gaps."""
    return clean_df[clean_df["date"] != pd.Timestamp("2013-01-02")].copy()


def test_load_train_applies_contract_dtypes(clean_df: pd.DataFrame) -> None:
    assert clean_df["id"].dtype == "int64"
    # pandas 3 parses dates as datetime64[us] rather than [ns]; any unit
    # satisfies the contract, so we check the kind, not the unit.
    assert pd.api.types.is_datetime64_any_dtype(clean_df["date"])
    assert clean_df["store_nbr"].dtype == "int16"
    assert clean_df["family"].dtype.name == "category"
    assert clean_df["sales"].dtype == "float32"
    assert clean_df["onpromotion"].dtype == "int32"


def test_load_train_joins_only_store_type(clean_df: pd.DataFrame) -> None:
    # Only `type` is joined by ruling; city/state/cluster stay out of scope.
    assert "type" in clean_df.columns
    assert not {"city", "state", "cluster"} & set(clean_df.columns)
    # A left join must preserve the train row count exactly.
    assert len(clean_df) == 16
    assert clean_df["type"].notna().all()
    assert store_type(clean_df, 1) == "A"


def test_integrity_report_clean_frame(clean_df: pd.DataFrame) -> None:
    report = integrity_report(clean_df)
    assert report["n_rows"] == 16
    assert report["date_min"] == "2013-01-01"
    assert report["date_max"] == "2013-01-03"
    assert report["n_stores"] == 5
    assert report["n_families"] == 2
    assert report["n_duplicate_keys"] == 0
    assert report["missing_per_column"].empty
    assert report["sales_min"] == pytest.approx(0.0)
    assert report["sales_max"] == pytest.approx(10.0)
    # 1 of 16 rows has sales == 0; rel tolerance because the share is a
    # fractional mean over a float32 column — last bits may drift.
    assert report["zero_sales_share"] == pytest.approx(1 / 16, rel=1e-6)
    assert report["onpromotion_min"] == 0
    assert report["onpromotion_max"] == 1
    assert report["n_missing_calendar_days"] == 0
    assert report["missing_calendar_days"] == []


def test_integrity_report_lists_missing_calendar_days(gap_df: pd.DataFrame) -> None:
    report = integrity_report(gap_df)
    assert report["n_missing_calendar_days"] == 1
    assert report["missing_calendar_days"] == ["2013-01-02"]


def test_assert_integrity_passes_for_clean_frame(clean_df: pd.DataFrame) -> None:
    assert_integrity(clean_df)


def test_assert_integrity_rejects_duplicates(clean_df: pd.DataFrame) -> None:
    duplicated = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        assert_integrity(duplicated)


def test_assert_integrity_rejects_missing_values(clean_df: pd.DataFrame) -> None:
    broken = clean_df.copy()
    broken.loc[0, "sales"] = None
    with pytest.raises(ValueError, match="missing values"):
        assert_integrity(broken)


def test_assert_integrity_rejects_negative_sales(clean_df: pd.DataFrame) -> None:
    broken = clean_df.copy()
    broken.loc[0, "sales"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        assert_integrity(broken)


def test_assert_integrity_rejects_calendar_gap_with_dates(gap_df: pd.DataFrame) -> None:
    # The message must name the exact missing dates so the 0-fill decision
    # is made consciously, never silently.
    with pytest.raises(ValueError, match="calendar"):
        assert_integrity(gap_df)
    with pytest.raises(ValueError, match="2013-01-02"):
        assert_integrity(gap_df)


def test_daily_network_sales_hand_known_sums(clean_df: pd.DataFrame) -> None:
    daily = daily_network_sales(clean_df)
    # rel=1e-6: sums are fractional (14.5) and accumulate in float32 —
    # tolerance covers fp noise without masking a wrong aggregation.
    assert daily.index.is_monotonic_increasing
    assert list(daily.index.strftime("%Y-%m-%d")) == [
        "2013-01-01",
        "2013-01-02",
        "2013-01-03",
    ]
    assert daily.iloc[0] == pytest.approx(25.0, rel=1e-6)
    assert daily.iloc[1] == pytest.approx(18.0, rel=1e-6)
    assert daily.iloc[2] == pytest.approx(14.5, rel=1e-6)


def test_daily_network_sales_fills_calendar_gap(gap_df: pd.DataFrame) -> None:
    daily = daily_network_sales(gap_df)
    # Approved closure handling: the missing day stays in the index as 0,
    # so the index remains a regular daily grid for MA/STL.
    assert len(daily) == 3
    assert daily.iloc[1] == pytest.approx(0.0, rel=1e-6)
    assert daily.iloc[2] == pytest.approx(14.5, rel=1e-6)


def test_store_daily_sales_hand_known(clean_df: pd.DataFrame) -> None:
    series = store_daily_sales(clean_df, 1)
    # Store 1: 10+5, 6, 7 by hand from TRAIN_ROWS.
    assert series.iloc[0] == pytest.approx(15.0, rel=1e-6)
    assert series.iloc[1] == pytest.approx(6.0, rel=1e-6)
    assert series.iloc[2] == pytest.approx(7.0, rel=1e-6)


def test_store_daily_sales_fills_gap_and_rejects_unknown(clean_df: pd.DataFrame, gap_df: pd.DataFrame) -> None:
    filled = store_daily_sales(gap_df, 1)
    assert len(filled) == 3
    assert filled.iloc[1] == pytest.approx(0.0, rel=1e-6)
    with pytest.raises(ValueError, match="unknown store"):
        store_daily_sales(clean_df, 99)


def test_store_type_returns_type_letter(clean_df: pd.DataFrame) -> None:
    assert store_type(clean_df, 1) == "A"
    assert store_type(clean_df, 5) == "B"
    with pytest.raises(ValueError, match="unknown store"):
        store_type(clean_df, 99)


def test_pick_stl_stores_default_picks_largest_a_and_smallest_de(clean_df: pd.DataFrame) -> None:
    # store 1 (A, total 28) outranks store 2 (A, 6); store 4 (E, 4.5)
    # is the smallest of D/E ahead of store 3 (D, 7). Type B is never picked.
    assert pick_stl_stores(clean_df) == [1, 4]


def test_pick_stl_stores_respects_counts(clean_df: pd.DataFrame) -> None:
    assert pick_stl_stores(clean_df, n_top=2, n_small=1) == [1, 2, 4]
    assert pick_stl_stores(clean_df, n_top=1, n_small=2) == [1, 4, 3]


def test_pick_stl_stores_fails_when_cohort_too_small(clean_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="type-A"):
        pick_stl_stores(clean_df, n_top=3)


def test_promotion_share_is_bounded_and_hand_known(clean_df: pd.DataFrame) -> None:
    share = promotion_share(clean_df)
    assert share.between(0, 1).all()
    assert list(share.index.strftime("%Y-%m-%d")) == [
        "2013-01-01",
        "2013-01-02",
        "2013-01-03",
    ]
    # Day 1: 2 of 6 rows on promo = 1/3 (stores 1-F2, 4); day 2: 2 of 5 = 0.4
    # (stores 2 and 5); day 3: 1 of 5 = 0.2 — 1/3 and 0.2 have no exact binary
    # representation, hence rel=1e-6 instead of ==.
    assert share.iloc[0] == pytest.approx(1 / 3, rel=1e-6)
    assert share.iloc[1] == pytest.approx(0.4, rel=1e-6)
    assert share.iloc[2] == pytest.approx(0.2, rel=1e-6)
