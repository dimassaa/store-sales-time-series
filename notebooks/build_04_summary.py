"""Builds notebooks/04_executive_summary.ipynb (Stage 4: summary + HTML report).

This notebook is the deliverable readers open instead of the three working
notebooks: it answers the four dashboard questions (trend, seasonal peak,
weekly profile, additive-vs-multiplicative) with ONE findings table computed
by executed src/ code and at most three figures REUSED as embedded PNGs from
assets/ (built by stages 1-3). It re-derives nothing beyond the headline
summary numbers, and its verdict text quotes only printed values.

The prose sections below were written after pass-A printed the exact values;
every literal number in a markdown paragraph appears verbatim in a printed or
displayed output cell above it.
"""

from __future__ import annotations

from builders import build

_ANCHOR = (
    "import sys, pathlib\n"
    "root = pathlib.Path.cwd()\n"
    "while root != root.parent and not (root / 'data' / 'train.csv').exists():\n"
    "    root = root.parent\n"
    "if not (root / 'data' / 'train.csv').exists():\n"
    "    raise FileNotFoundError('could not locate data/train.csv')\n"
    "if str(root) not in sys.path:\n"
    "    sys.path.insert(0, str(root))\n"
    "DATA = root / 'data' / 'train.csv'\n"
    "HOLIDAYS = root / 'data' / 'holidays_events.csv'\n"
)

# All prose numbers below are executed outputs of cells 2, 4, 8 and 12 (and
# the chained stages they cite). Nothing is typed that a cell did not print.
_TREND_SECTION = (
    "## 1. Trend — is there a long-term one, and where does it point?\n\n"
    "The STL trend (top panel below; raw series in thin gray for context)\n"
    "climbs from **462,484** (first day) to a peak of **1,010,730** in late\n"
    "2016, then eases to a last value of **745,999** (2017-08-15). The yearly\n"
    "trend means quantify the direction: **385,121** in 2013, **931,604** as\n"
    "the 2016-12 monthly mean, **846,530** in 2017. The trend strength\n"
    "**F_t = 0.793** (Wang, Smith & Hyndman 2006) says the trend band absorbs\n"
    "most of the non-seasonal variation — a durable level story, not random\n"
    "wobble. It is **not** a monotone ramp: it peaks in **late 2016** and\n"
    "softens through 2017, so \u00abup then easing\u00bb is the honest shape.\n\n"
    "The 0-filled Christmas closures are visible as the needle-dips in the\n"
    "top panel; they belong to the residual story (section 2), not to the\n"
    "trend."
)

_PEAK_SECTION = (
    "## 2. Seasonal peak — when, and why (national holidays)\n\n"
    "Two different \u00abpeaks\u00bb answer this from the STL decomposition:\n\n"
    "- **The regular weekly peak is Sunday.** With `period=7` the seasonal\n"
    "  band cycles every week; mean seasonal by weekday peaks on **Sunday\n"
    "  (+215,863 units)** and bottoms on **Thursday (\u2212136,391)** — the same\n"
    "  weekday shape Stage 1 measured on raw means (Sun \u2248 822k vs Thu \u2248\n"
    "  503k, printed in cell 4), now isolated from noise.\n"
    "- **The calendar peaks are national-holiday events.** The weekly band\n"
    "  cannot encode a specific date, so single-day events land in the\n"
    "  residual: **14 of the top-15 largest |resid| days are exactly a\n"
    "  national-holiday date** (table in the cell below). The biggest are the\n"
    "  four **Navidad** (Christmas) closures — **2016-12-25 reaches\n"
    "  \u22121,321,021** — and the **New Year** bursts (**+1,262,485** on\n"
    "  2017-01-01). One top-15 day has **no named holiday**: **2017-04-01**,"
    " the largest raw-sales day of the whole series (1,463,084, printed in\n"
    " the setup cell) — the holiday file does not explain it.\n\n"
    "So: expect a **Sunday-heavy week every week**, and the **outlier days\n"
    "around Christmas / New Year** as the seasonal peaks for calendar dates."
)

_WEEKLY_SECTION = (
    "## 3. Weekly profile — where the weekday rhythm lives\n\n"
    "The bar figure (reused from Stage 1) is the network-wide mean by day of\n"
    "week: **Sun \u2248 821,794** and **Sat \u2248 772,206** against a Thursday\n"
    "trough of **\u2248 503,173** — an amplitude of **318,621** units (exact\n"
    "numbers in the findings table, row 4). This is the pattern the\n"
    "`period=7` STL seasonal band captures and isolates, and it is why\n"
    "smoothing with a 7-day window (stage 2) removes the noise while keeping\n"
    "the level."
)

_MODEL_SECTION = (
    "## 4. Model choice — additive or multiplicative?\n\n"
    "`decompose_additive_vs_multiplicative` fits an **additive** STL on the\n"
    "original scale and a **multiplicative proxy** (STL on log, components\n"
    "exp-mapped back) on the identical, strictly-positive input. The four\n"
    "0-filled closure days are replaced by neighbour means **only here**, and\n"
    "the adjusted rows are printed (cell 12) so the policy is auditable.\n\n"
    "Decision rule (documented in `src/decomposition.py`): pick the model\n"
    "with the smaller `resid_relative_std`. Result on the network series:\n"
    "additive **0.1323** vs multiplicative **0.1311** — the multiplicative\n"
    "proxy wins by **0.94%** of relative residual spread. That margin is tiny,\n"
    "**no test was run on the gap**, and both fits tell the same monthly-report\n"
    "story; we record **multiplicative** as the pick and report the near-tie\n"
    "rather than overselling it."
)

_VERDICT = (
    "## 5. Verdict — the answer for a monthly report\n\n"
    "This is the **executive answer** to the question a monthly report asks:\n"
    "<em>what happened to daily network sales, descriptively?</em>\n\n"
    "> **The network grew strongly from 2013 through late 2016 (STL trend\n"
    "> strength F_t = 0.793; yearly trend means 385,121 \u2192 846,530) and then\n"
    "> eased into 2017. Sales follow a stable weekly rhythm that peaks on\n"
    "> Sunday (mean +215,863 seasonal units above the weekday midpoint;\n"
    "> trough Thursday at \u2212136,391), and the calendar outliers are national\n"
    "> holidays: 14 of the 15 largest residual days fall exactly on a\n"
    "> holiday, led by the Christmas-network closures (Navidad 2016 at\n"
    "> \u22121,321,021) and New Year bursts (+1,262,485 on 2017-01-01). Between\n"
    "> the additive and multiplicative seasonal models the data pick\n"
    "> multiplicative, but only by 0.94% of relative residual spread — the\n"
    "> two are effectively tied for reporting purposes.**\n\n"
    "**Honest boundary of this verdict:** it is **descriptive and\n"
    "exploratory** (status of all stages 1-4). It describes what the STL\n"
    "decomposition showed; it does **not** forecast, fit an ARIMA family\n"
    "model, or claim statistical significance for any single number. The\n"
    "trend's late-2016 easing and the one unexplained top-residual day\n"
    "(2017-04-01) are facts we state without a causal story. For a decision\n"
    "that forecasts — e.g. \u00abwill 2017 keep growing\u00bb — a separate,\n"
    "pre-registered modeling stage would be needed; this report's job is\n"
    "only to say what the series has done."
)

CELLS: list[dict[str, str]] = [
    {
        "type": "markdown",
        "source": (
            "# 04 — Executive summary (monthly-report story)\n\n"
            "**Stage 4** of the time-series analytics project. Everything the\n"
            "working notebooks 01-03 found is condensed here into one\n"
            "report: the **table of findings**, the four dashboard answers\n"
            "(\u00a71 trend, \u00a72 seasonal peak, \u00a73 weekly profile, \u00a74 model\n"
            "choice), and a single **verdict for a monthly report**.\n\n"
            "**Status: DESCRIPTIVE / EXPLORATORY.** No forecasting, no\n"
            "significance tests; the verdict reports what the data showed.\n\n"
            "**Inputs:** this notebook reuses the three PNG figures built in\n"
            "stages 1-3 (embedded from `assets/`) and recomputes the headline\n"
            "summary numbers from `src/` so every claim below is reproduced by\n"
            "executed code in this notebook."
        ),
    },
    {
        "type": "code",
        "source": (
            "%matplotlib inline\n"
            + _ANCHOR
            + "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display, Image\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "pd.set_option('display.float_format', lambda v: f'{v:,.0f}')\n"
            "from src.eda import (\n"
            "    load_train, daily_network_sales, store_type, pick_stl_stores,\n"
            ")\n"
            "from src.decomposition import (\n"
            "    stl_decompose, seasonality_strength, trend_strength,\n"
            "    decompose_additive_vs_multiplicative, national_holidays,\n"
            ")\n"
            "df = load_train(DATA)\n"
            "daily = daily_network_sales(df)\n"
            "print(f'data rows: {len(df):,} | stores: {df[\"store_nbr\"].nunique()} | '\n"
            "      f'families: {df[\"family\"].nunique()}')\n"
            "print(f'daily series: {len(daily)} days, '\n"
            "      f\"{daily.index.min().date()} .. {daily.index.max().date()}\")\n"
            "zero_days = daily[daily == 0]\n"
            "print('0-filled days (Christmas closures): '\n"
            "      f\"{list(zero_days.index.strftime('%Y-%m-%d'))}\")\n"
            "picked = pick_stl_stores(df)\n"
            "print(f'STL stores (pick from src, never hardcoded): '\n"
            "      f\"{[p for p in picked]}\")\n"
            "print(f'largest raw-sales day: {daily.idxmax().date()} '\n"
            "      f'({daily.max():,.0f})')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## How to read this notebook\n\n"
            "1. **Every number in the prose is an executed value.** The\n"
            "   findings table below (and the per-section cells) computes each\n"
            "   headline number from `src/` in this notebook; the verdict\n"
            "   quotes only printed values. Nothing is typed from memory.\n"
            "2. **The table of findings is the centerpiece** — it is the\n"
            "   one screen that carries the whole monthly-report story.\n"
            "3. **Figures are reused, not re-drawn:** three PNGs from stages\n"
            "   1-3 are embedded, one per dashboard question (trend, peak,\n"
            "   weekly profile). The model-choice question is answered by the\n"
            "   numbers in \u00a74, not by a fourth figure."
        ),
    },
    {
        "type": "code",
        "source": (
            "comp_net = stl_decompose(daily)\n"
            "tr = comp_net['trend']\n"
            "weekly = daily.groupby(daily.index.dayofweek).mean()\n"
            "seasonal_dow = comp_net['seasonal'].groupby(comp_net.index.dayofweek).mean()\n"
            "weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\n"
            "\n"
            "pos = daily.copy()\n"
            "for day in zero_days.index:\n"
            "    pos[day] = (daily.shift(1)[day] + daily.shift(-1)[day]) / 2\n"
            "cmp = decompose_additive_vs_multiplicative(pos)\n"
            "add_rrs = cmp['additive']['resid_relative_std']\n"
            "mult_rrs = cmp['multiplicative']['resid_relative_std']\n"
            "\n"
            "holidays = national_holidays(HOLIDAYS, daily.index.min().date(),\n"
            "                            daily.index.max().date())\n"
            "top = comp_net['resid'].abs().nlargest(15)\n"
            "hd = holidays.set_index('date')['description']\n"
            "def holiday_names(d):\n"
            "    if d not in hd.index:\n"
            "        return ''\n"
            "    vals = hd.loc[d]\n"
            "    vals = [vals] if isinstance(vals, str) else list(vals)\n"
            "    return ', '.join(vals)\n"
            "n_holiday = int(sum(holiday_names(d) != '' for d in top.index))\n"
            "\n"
            "tr_yearly = tr.groupby(tr.index.year).mean()\n"
            "findings = pd.DataFrame({\n"
            "    'metric': [\n"
            "        'Data range',\n"
            "        'Rows / stores / families',\n"
            "        'Christmas closures (0-filled)',\n"
            "        'Raw weekday mean — Sun / Thu',\n"
            "        'Raw weekday amplitude (max - min of mean)',\n"
            "        'STL seasonality strength F_s (Wang et al.)',\n"
            "        'STL trend strength F_t',\n"
            "        'Trend first / last (STL)',\n"
            "        'Trend peak (value, date)',\n"
            "        'Trend yearly means 2013 / 2017',\n"
            "        'Trend monthly mean 2016-12',\n"
            "        'Seasonal weekday peak / trough (mean)',\n"
            "        'Top-15 |resid| days exactly on a national holiday',\n"
            "        'Largest holiday residual (Navidad 2016)',\n"
            "        'Model pick (min resid_relative_std)',\n"
            "        'RRS additive / multiplicative / gap',\n"
            "    ],\n"
            "    'value': [\n"
            "        f\"{daily.index.min().date()} .. {daily.index.max().date()}\",\n"
            "        f'{len(df):,} / {df[\"store_nbr\"].nunique()} / '\n"
            "            f'{df[\"family\"].nunique()}',\n"
            "        ', '.join(zero_days.index.strftime('%Y-%m-%d')),\n"
            "        f\"{weekly[6]:,.0f} / {weekly[3]:,.0f}\",\n"
            "        f'{weekly.max() - weekly.min():,.0f}',\n"
            "        f\"{seasonality_strength(comp_net):.3f}\",\n"
            "        f\"{trend_strength(comp_net):.3f}\",\n"
            "        f'{tr.iloc[0]:,.0f} / {tr.iloc[-1]:,.0f}',\n"
            "        f'{tr.max():,.0f} on {tr.idxmax().date()}',\n"
            "        f'{tr_yearly[2013]:,.0f} / {tr_yearly[2017]:,.0f}',\n"
            "        f'{tr[\"2016-12\"].mean():,.0f}',\n"
            "        f'{weekday_names[int(seasonal_dow.idxmax())]} '\n"
            "            f'{seasonal_dow.max():,.0f} / '\n"
            "            f'{weekday_names[int(seasonal_dow.idxmin())]} '\n"
            "            f'{seasonal_dow.min():,.0f}',\n"
            "        f'{n_holiday} of 15',\n"
            "        f\"{comp_net['resid'].min():,.0f} on \"\n"
            "            f\"{comp_net['resid'].idxmin().strftime('%Y-%m-%d')}\",\n"
            "        cmp['recommendation'],\n"
            "        f'{add_rrs:.4f} / {mult_rrs:.4f} / '\n"
            "            f'{(mult_rrs - add_rrs) / add_rrs:+.2%}',\n"
            "    ],\n"
            "    'produced by': [\n"
            "        'eda.daily_network_sales (Stage 1)',\n"
            "        'eda.load_train + integrity (Stage 1)',\n"
            "        'eda.daily_network_sales 0-fill (Stage 1)',\n"
            "        'weekday means computed here from daily (Stage 1 claim)',\n"
            "        'same',\n"
            "        'decomposition.seasonality_strength (Stage 3)',\n"
            "        'decomposition.trend_strength (Stage 3)',\n"
            "        'decomposition.stl_decompose (Stage 3)',\n"
            "        'same',\n"
            "        'same',\n"
            "        'same',\n"
            "        'seasonal band by weekday, computed here (Stage 3 claim)',\n"
            "        'decomposition.national_holidays + resid (Stage 3)',\n"
            "        'stl resid min (Stage 3, Navidad closure)',\n"
            "        'decomposition.decompose_additive_vs_multiplicative',\n"
            "        'same (0.94% near-tie)',\n"
            "    ],\n"
            "})\n"
            "display(findings)\n"
            "print('\\nRaw weekday mean sales (series units):')\n"
            "for k in range(7):\n"
            "    print(f'{weekday_names[k]}: {weekly[k]:,.0f}')\n"
            "print('\\nMean STL seasonal band by weekday (units):')\n"
            "for k in range(7):\n"
            "    print(f'{weekday_names[k]}: {seasonal_dow[k]:,.0f}')\n"
            "print('\\nTrend yearly mean (units):')\n"
            "for year, v in tr_yearly.items():\n"
            "    print(f'{year}: {v:,.0f}')"
        ),
    },
    {
        "type": "markdown",
        "source": _TREND_SECTION,
    },
    {
        "type": "code",
        "source": (
            "# Reused Stage-3 figure: the full STL decomposition of the network\n"
            "# series; the trend panel is the evidence for section 1. Embedded\n"
            "# directly so the notebook stays honest to the committed PNG.\n"
            "display(Image(filename=str(root / 'assets' / 'stl_network.png')))"
        ),
    },
    {
        "type": "markdown",
        "source": _PEAK_SECTION,
    },
    {
        "type": "code",
        "source": (
            "top_tbl = top.rename('abs_resid').to_frame()\n"
            "top_tbl['date'] = top_tbl.index\n"
            "top_tbl['holiday'] = [holiday_names(d) for d in top_tbl['date']]\n"
            "print('top-15 |resid| days (executed table behind section 2):')\n"
            "display(top_tbl[['date', 'abs_resid', 'holiday']].reset_index(drop=True))\n"
            "print(f'{n_holiday} of 15 top residual days are exactly a '\n"
            "      'national-holiday date')\n"
            "# Reused Stage-3 figure: raw sales with holiday lines (top) and the\n"
            "# residual (bottom) where the holiday spikes land.\n"
            "display(Image(filename=str(root / 'assets' / 'seasonal_peaks_holidays.png')))"
        ),
    },
    {
        "type": "markdown",
        "source": _WEEKLY_SECTION,
    },
    {
        "type": "code",
        "source": (
            "# Reused Stage-1 figure: mean network sales by day of week — the\n"
            "# evidence behind section 3 and behind period=7 for all earlier\n"
            "# decomposition stages.\n"
            "display(Image(filename=str(root / 'assets' / 'eda_weekday_profile.png')))"
        ),
    },
    {
        "type": "markdown",
        "source": _MODEL_SECTION,
    },
    {
        "type": "code",
        "source": (
            "zero_days = daily[daily == 0]\n"
            "adj = []\n"
            "for day in zero_days.index:\n"
            "    left, right = daily.shift(1)[day], daily.shift(-1)[day]\n"
            "    adj.append((day.date(), float(daily[day]), float(left),\n"
            "                float(right), float(pos[day])))\n"
            "print('adjusted closure days (original -> neighbour mean):')\n"
            "print(pd.DataFrame(adj, columns=['date', 'original', 'prev', 'next',\n"
            "                                'replacement']).to_string(index=False))\n"
            "rows = []\n"
            "for model in ('additive', 'multiplicative'):\n"
            "    e = cmp[model]\n"
            "    rows.append({\n"
            "        'model': model,\n"
            "        'resid_relative_std': e['resid_relative_std'],\n"
            "        'resid std': float(e['resid'].std()),\n"
            "    })\n"
            "with pd.option_context('display.float_format', lambda v: f'{v:.4f}'):\n"
            "    display(pd.DataFrame(rows))\n"
            "print(f'recommendation: {cmp[\"recommendation\"]} (documented rule: '\n"
            "      'smaller resid_relative_std wins)')\n"
            "print(f'gap (mult - add)/add: {(mult_rrs - add_rrs) / add_rrs:+.2%}')"
        ),
    },
    {
        "type": "markdown",
        "source": _VERDICT,
    },
]


def main() -> None:
    out = build(CELLS, "04_executive_summary")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()