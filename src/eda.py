"""Data integrity and EDA helpers for the Favorita store-sales analysis.

Golden standard #1 (data first): every downstream smoothing/STL claim is only
as trustworthy as the frame it runs on, so loading, validation, calendar
handling and the pre-registered store pick live here as reusable functions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_train(path: str | Path) -> pd.DataFrame:
    """Load train.csv with contract dtypes and the store ``type`` column attached.

    The explicit dtype map is the optimized read for the ~122 MB CSV: letting
    pandas infer would upcast store_nbr/sales and waste memory for no benefit.

    Store type is left-joined from the sibling ``stores.csv`` because
    ``store_type``/``pick_stl_stores`` are defined on a single frame — callers
    must not have to remember a manual merge. Only ``type`` is joined
    (city/state/cluster are out of scope for this stage).

    Returns the frame unvalidated; ``assert_integrity`` is the gate. A store
    missing from stores.csv would surface there as a NaN ``type``.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        dtype={
            "id": "int64",
            "store_nbr": "int16",
            "family": "category",
            "sales": "float32",
            "onpromotion": "int32",
        },
        parse_dates=["date"],
    )
    stores = pd.read_csv(
        path.parent / "stores.csv",
        usecols=["store_nbr", "type"],
        dtype={"store_nbr": "int16", "type": "category"},
    )
    # many_to_one guards against duplicated store rows in stores.csv, which
    # would silently multiply train rows if left unchecked.
    return df.merge(stores, on="store_nbr", how="left", validate="many_to_one")


def integrity_report(df: pd.DataFrame) -> dict[str, object]:
    """Summarize every integrity dimension the analysis depends on.

    Returns a dict so notebooks/terminal can render it; the function takes no
    decisions. Calendar continuity is reported as the exact missing dates
    because MA/STL stages need a regular daily index and must see what will
    be filled (see ``daily_network_sales``).
    """
    duplicate_keys = df.duplicated(["date", "store_nbr", "family"])
    missing = df.isna().sum()
    missing = missing[missing > 0] if missing.any() else pd.Series(dtype="int64")
    full_calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_days = full_calendar.difference(pd.DatetimeIndex(df["date"].unique()))
    return {
        "n_rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_stores": int(df["store_nbr"].nunique()),
        "n_families": int(df["family"].nunique()),
        "n_duplicate_keys": int(duplicate_keys.sum()),
        "missing_per_column": missing,
        "sales_min": float(df["sales"].min()),
        "sales_max": float(df["sales"].max()),
        "zero_sales_share": float((df["sales"] == 0).mean()),
        "onpromotion_min": int(df["onpromotion"].min()),
        "onpromotion_max": int(df["onpromotion"].max()),
        "n_missing_calendar_days": int(len(missing_days)),
        "missing_calendar_days": [str(day.date()) for day in missing_days],
    }


def assert_integrity(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` on any defect that would invalidate downstream work.

    Checks, in order: missing values, duplicate ``(date, store_nbr, family)``
    keys, negative sales, calendar gaps. Calendar gaps are fatal because the
    smoothing/STL stages require a gapless daily index; the message names the
    exact dates so the caller applies the documented 0-fill consciously rather
    than smoothing over unknown holes.
    """
    missing = df.isna().sum()
    if missing.any():
        raise ValueError(f"missing values per column: {missing[missing > 0].to_dict()}")
    n_duplicates = int(df.duplicated(["date", "store_nbr", "family"]).sum())
    if n_duplicates:
        raise ValueError(f"{n_duplicates} duplicate (date, store_nbr, family) keys")
    n_negative = int((df["sales"] < 0).sum())
    if n_negative:
        raise ValueError(f"{n_negative} rows with negative sales")
    full_calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_days = full_calendar.difference(pd.DatetimeIndex(df["date"].unique()))
    if len(missing_days):
        raise ValueError(
            "calendar is not continuous; missing dates: "
            + ", ".join(str(day.date()) for day in missing_days)
        )


def _on_full_calendar(daily: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Reindex an observed-daily series onto every calendar day in [start, end].

    Days without rows (Christmas closures 2013-2016 in this dataset) become 0:
    an absent row means zero recorded sales for that day, and downstream
    MA/STL reject irregular indices, while NaN would poison every rolling
    window around the gap.
    """
    full_calendar = pd.date_range(start, end, freq="D")
    filled = daily.reindex(full_calendar, fill_value=0)
    filled.index.name = "date"
    return filled


def daily_network_sales(df: pd.DataFrame) -> pd.Series:
    """Network-wide daily sales (all stores x families) over the full calendar.

    Sorted by date with a gapless daily index: closure days are 0-filled per
    ``_on_full_calendar``. The series keeps the name ``sales`` for axis labels.
    """
    daily = df.groupby("date", sort=True)["sales"].sum()
    return _on_full_calendar(daily, df["date"].min(), df["date"].max())


def store_daily_sales(df: pd.DataFrame, store_nbr: int) -> pd.Series:
    """Daily sales for one store over the full training calendar.

    Same continuity rule as ``daily_network_sales`` — the later STL stage
    requires a gapless index for every store. The calendar spans the whole
    frame (not the store's first/last row), so a genuinely late-opened store
    would get pre-opening zeros; in this dataset every store covers the full
    period, so no false zeros are introduced.

    Raises ``ValueError`` for an unknown store so a typo fails loud instead of
    returning an empty series.
    """
    if store_nbr not in set(df["store_nbr"].unique()):
        raise ValueError(f"unknown store_nbr: {store_nbr}")
    daily = df.loc[df["store_nbr"] == store_nbr].groupby("date", sort=True)["sales"].sum()
    return _on_full_calendar(daily, df["date"].min(), df["date"].max())


def store_type(df: pd.DataFrame, store_nbr: int) -> str:
    """Return the store's type letter (A-E) as a plain ``str``.

    ``type`` is joined in ``load_train``; an unknown store raises rather than
    returning None, because the STL pick must be traceable to a real store.
    """
    types = df.loc[df["store_nbr"] == store_nbr, "type"]
    if types.empty:
        raise ValueError(f"unknown store_nbr: {store_nbr}")
    return str(types.iloc[0])


def pick_stl_stores(df: pd.DataFrame, n_top: int = 1, n_small: int = 1) -> list[int]:
    """Deterministically pick STL stores: largest type-A + smallest type-D/E.

    Single pre-registered decision source for notebooks 01/03: the ranking
    rule lives here so later stages cannot shop for convenient stores after
    seeing results. Type-A stores are ranked by total sales descending, D/E
    ascending; a stable sort with store_nbr already ascending from groupby
    breaks total-sales ties deterministically across platforms.

    Returns ``store_nbr``'s: top-A stores first, then small D/E stores.
    Raises ``ValueError`` when a cohort has fewer stores than requested.
    """
    totals = df.groupby("store_nbr").agg(total=("sales", "sum"), type=("type", "first"))
    type_a = totals[totals["type"] == "A"].sort_values("total", ascending=False, kind="stable")
    small_de = (
        totals[totals["type"].isin(["D", "E"])]
        .sort_values("total", ascending=True, kind="stable")
    )
    if len(type_a) < n_top:
        raise ValueError(f"requested {n_top} type-A stores, only {len(type_a)} available")
    if len(small_de) < n_small:
        raise ValueError(f"requested {n_small} type-D/E stores, only {len(small_de)} available")
    picked = [int(store) for store in type_a.index[:n_top]]
    picked += [int(store) for store in small_de.index[:n_small]]
    return picked


def promotion_share(df: pd.DataFrame) -> pd.Series:
    """Share of rows with an active promotion per observed date, in [0, 1].

    A row-level fraction answers "how much of the assortment is on promo that
    day" — the raw promo-activity view plotted in the EDA notebook. Only
    observed dates are returned: closure days carry no rows, and promo
    continuity is not required anywhere downstream (unlike the sales series).
    """
    share = (df["onpromotion"] > 0).groupby(df["date"], sort=True).mean()
    return share.rename("promotion_share")
