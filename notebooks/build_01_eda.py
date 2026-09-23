"""Builds notebooks/01_eda_integrity.ipynb (Stage 1: data integrity + EDA)."""

from __future__ import annotations

from builders import build

CELLS: list[dict[str, str]] = [
    {
        "type": "markdown",
        "source": (
            "# 01 — Data integrity and EDA\n\n"
            "**Stage 1** of the time-series analytics project. Before any\n"
            "smoothing or STL claim we verify the data can support a regular\n"
            "daily index (data integrity, golden standard #1) and describe the\n"
            "network's shape and scale (EDA).\n\n"
            "**Status: EXPLORATORY (hypothesis-generating).** Nothing in this\n"
            "notebook is a significance claim; findings here only inform later\n"
            "stages.\n\n"
            "**Unit of analysis (unit of analysis):** one row = sales of one\n"
            "product `family` in one `store_nbr` on one `date`, the Kaggle\n"
            "schema `id, date, store_nbr, family, sales, onpromotion`.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "%matplotlib inline\n"
            "import sys, pathlib\n"
            "root = pathlib.Path.cwd()\n"
            "while root != root.parent and not (root / 'data' / 'train.csv').exists():\n"
            "    root = root.parent\n"
            "if not (root / 'data' / 'train.csv').exists():\n"
            "    raise FileNotFoundError('could not locate data/train.csv')\n"
            "if str(root) not in sys.path:\n"
            "    sys.path.insert(0, str(root))\n"
            "DATA = root / 'data' / 'train.csv'\n"
            "from src.eda import (\n"
            "    load_train, integrity_report, assert_integrity,\n"
            "    daily_network_sales, store_daily_sales, store_type,\n"
            "    pick_stl_stores, promotion_share,\n"
            ")\n"
            "df = load_train(DATA)\n"
            "df.head()"
        ),
    },
    {
        "type": "code",
        "source": (
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "pd.set_option('display.float_format', lambda x: f'{x:,.2f}')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 1. Data integrity (data integrity)\n\n"
            "Checks performed: completeness (no missing per column), uniqueness\n"
            "(no duplicate `(date, store_nbr, family)` keys), value sanity\n"
            "(`sales >= 0`, `onpromotion` range), and calendar continuity\n"
            "(every day between min and max `date` present). The calendar check\n"
            "matters most: smoothing and STL in later stages assume a regular\n"
            "daily index.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "report = integrity_report(df)\n"
            "print(f\"rows: {report['n_rows']:,}\")\n"
            "print(f\"date range: {report['date_min']} .. {report['date_max']}\")\n"
            "print(f\"stores: {report['n_stores']}, families: {report['n_families']}\")\n"
            "print(f\"duplicate (date, store, family) keys: {report['n_duplicate_keys']}\")\n"
            "print(f\"missing values per column: {report['missing_per_column'].to_dict()}\")\n"
            "print(f\"sales min/max: {report['sales_min']:,.2f} / {report['sales_max']:,.2f}\")\n"
            "print(f\"zero-sales row share: {report['zero_sales_share']:.2%}\")\n"
            "print(f\"onpromotion min/max: {report['onpromotion_min']} / {report['onpromotion_max']}\")\n"
            "print(f\"missing calendar days: {report['n_missing_calendar_days']} -> \"\n"
            "      f\"{report['missing_calendar_days']}\")"
        ),
    },
    {
        "type": "code",
        "source": (
            "try:\n"
            "    assert_integrity(df)\n"
            "    print('assert_integrity: PASS — frame is valid for analysis.')\n"
            "except ValueError as err:\n"
            "    print(f'assert_integrity surfaced: {err}')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "### Calendar verdict and the 0-fill decision (pre-registered)\n\n"
            "The frame spans **2013-01-01 .. 2017-08-15**, but only **1684** of\n"
            "**1688** calendar days are present. The four missing days are\n"
            "**Dec 25, 2013-2016**: no rows exist for any store on those days\n"
            "(and `transactions.csv` records no activity either), i.e. the whole\n"
            "network was closed for Christmas.\n\n"
            "Handling (decided here and re-used by every downstream stage):\n"
            "`daily_network_sales` / `store_daily_sales` reindex to the full\n"
            "calendar and **fill the closure days with 0** — an absent row means\n"
            "zero recorded sales, and a gapless daily index is what MA/STL\n"
            "require. `NaN` is rejected because it would poison every rolling\n"
            "window around the gap; dropping the days would break the regular\n"
            "grid. `assert_integrity` still raises on gaps — the hard gate stays\n"
            "strict, and the 0-fill is a conscious, documented exception.\n"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 2. Daily network sales (decomposition target)\n\n"
            "**Question:** what does the whole network's daily sales look like\n"
            "over 2013-2017 — level, spread, spikes? This is the series the MA/STL\n"
            "stages smooth and decompose.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "daily = daily_network_sales(df)\n"
            "print(f\"series length: {len(daily)} (full calendar incl. 4 filled closures)\\n\")\n"
            "print(daily.describe().to_string())\n"
            "print(f\"\\nmin..max: {daily.min():,.0f} .. {daily.max():,.0f}\")"
        ),
    },
    {
        "type": "code",
        "source": (
            "fig, ax = plt.subplots(figsize=(11, 4.2))\n"
            "ax.plot(daily.index, daily.values, lw=0.8, color='#4C72B0')\n"
            "ax.set_title('Daily network sales, 2013-01-01 .. 2017-08-15')\n"
            "ax.set_xlabel('date')\n"
            "ax.set_ylabel('sales (units)')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/eda_daily_sales.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 3. Weekly profile (weekday seasonality)\n\n"
            "**Question:** which days of the week sell more? A stable weekday\n"
            "pattern is exactly what a `period=7` STL is meant to capture.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\n"
            "weekly = daily.groupby(daily.index.dayofweek).mean()\n"
            "weekly.index = [weekday_names[i] for i in weekly.index]\n"
            "print(weekly.to_string())"
        ),
    },
    {
        "type": "code",
        "source": (
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(weekly.index, weekly.values, color='#4C72B0')\n"
            "ax.set_title('Mean network sales by day of week')\n"
            "ax.set_xlabel('day of week')\n"
            "ax.set_ylabel('mean daily sales (units)')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/eda_weekday_profile.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 4. Promotions over time (promo share)\n\n"
            "**Question:** how much of the assortment runs a promotion, and is\n"
            "promo activity stationary? `promotion_share` is the raw share of\n"
            "rows with `onpromotion > 0` per day — a visibility view, not a\n"
            "causal claim.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "promo = promotion_share(df)\n"
            "print(f\"overall share of promo rows: {(df['onpromotion'] > 0).mean():.2%}\")\n"
            "print(f\"daily share min / mean / max: \"\n"
            "      f\"{promo.min():.2%} / {promo.mean():.2%} / {promo.max():.2%}\")\n"
            "print(\"\\nmean daily promo share by year:\")\n"
            "print(promo.groupby(promo.index.year).mean()\n"
            "      .map(lambda v: f'{v:.2%}').to_string())"
        ),
    },
    {
        "type": "code",
        "source": (
            "# Embedded EDA figure only (option from the plan: promo is not an\n"
            "# assets/ file — the stage's two named figures are the sales series\n"
            "# and the weekday profile).\n"
            "fig, ax = plt.subplots(figsize=(11, 3.6))\n"
            "ax.plot(promo.index, promo.values, lw=0.7, color='#DD8452')\n"
            "ax.set_title('Share of rows with active promotions, per day (raw view)')\n"
            "ax.set_xlabel('date')\n"
            "ax.set_ylabel('share of rows on promo')\n"
            "ax.set_ylim(0, 1)\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 5. Top stores by total sales\n\n"
            "**Question:** how concentrated is the network? This sets the scale\n"
            "for store-level smoothing later.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "totals = df.groupby('store_nbr')['sales'].sum().sort_values(ascending=False)\n"
            "top = totals.head(10).to_frame('total')\n"
            "top['type'] = [store_type(df, s) for s in top.index]\n"
            "print(top.round(0).to_string())"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 6. STL store pick (pre-registered, fixed here)\n\n"
            "Task 3 decomposes **two** stores. The rule is fixed now so the pick\n"
            "cannot be revisited after decomposition results are visible: the\n"
            "largest type-**A** store (same format for peak seasonality) plus the\n"
            "smallest type-**D**/**E** store (a different channel scale). The pick\n"
            "is computed by `pick_stl_stores`, the single decision source used by\n"
            "notebooks 01 and 03.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "picked = pick_stl_stores(df)\n"
            "for s in picked:\n"
            "    total = float(df.loc[df['store_nbr'] == s, 'sales'].sum())\n"
            "    print(f'store {s} (type {store_type(df, s)}): total sales {total:,.0f}')\n"
            "print(f'\\nSTL stores for Task 3: {picked}')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 7. Stage conclusion\n\n"
            "- Frame is clean: **3,000,888** rows, 2013-01-01 .. 2017-08-15,\n"
            "  **54** stores, **33** families; no duplicates, no missing values,\n"
            "  `sales` in [0, 124,717], `onpromotion` in [0, 741]; 31.3% of rows\n"
            "  are zero-sales (assortment days without a sale — structural zeros).\n"
            "- Calendar is **not** gapless: **Dec 25, 2013-2016** are absent for\n"
            "  the whole network (Christmas closure). They are 0-filled in both\n"
            "  daily series functions; `assert_integrity` surfaced this loudly and\n"
            "  the decision is documented in notes/01.\n"
            "- Network daily sales: mean ≈ 636k, median ≈ 632k, max ≈ 1.46M;\n"
            "  a strong weekend effect — Sun ≈ 822k, Sat ≈ 772k vs a Thu trough\n"
            "  ≈ 503k — confirms a clear weekly seasonality for `period=7` STL.\n"
            "- Promotions are **not** stationary: daily promo share climbed from\n"
            "  ~0% (2013) to ~45% (2017), peaking at 56% of rows — promo is a\n"
            "  growing business policy and a candidate driver of level shifts.\n"
            "- STL stores fixed here: **store 44 (type A, largest)** and\n"
            "  **store 26 (type D, smallest D/E)** — used unchanged by Task 3.\n"
        ),
    },
]


def main() -> None:
    out = build(CELLS, "01_eda_integrity")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()