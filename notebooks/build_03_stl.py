"""Builds notebooks/03_stl_decomposition.ipynb (Stage 3: STL + holidays).

Decomposes the network daily series and the two pre-registered stores (the
pick from `pick_stl_stores`, never hardcoded), annotates non-weekly peaks with
national holidays, reports seasonality/trend strengths, and decides the
additive-vs-multiplicative model from executed numbers.
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

# Interim one-liner; replaced from the pass-A executed outputs so every number
# in the written conclusion is a real executed value (project rule).
# All figures below come from the pass-A printed outputs of cells 3/6/7/9/12/14.
CONCLUSION = (
    "## 8. Conclusion\n\n"
    "All numbers below are executed outputs printed by the cells above\n"
    "(\u00a7network, \u00a7stores, \u00a7holidays, \u00a7strengths, \u00a7model \u2014 nothing here is\n"
    "written without a matching printed value).\n\n"
    "**1. The weekly seasonal peak is Sunday.** The STL seasonal band of the\n"
    "network series swings from **\u2212241,757 to +376,388** units over the\n"
    "week; mean seasonal by weekday peaks on **Sunday (+215,863)** and bottoms\n"
    "on **Thursday (\u2212136,391)** \u2014 the same weekday shape Stage 1 measured\n"
    "in raw means (Sun \u2248 822k vs Thu \u2248 503k), now isolated from noise.\n\n"
    "**2. The non-weekly peaks are national holidays \u2014 14 of the top-15\n"
    "|resid| days are exactly a national-holiday date.** With period=7 the\n"
    "weekly band cannot encode calendar-date events, so holiday effects land\n"
    "in **resid**, and the list confirms it: four **Navidad** (Christmas)\n"
    "closure dips \u2014 2016-12-25 reaches \u22121,321,021 \u2014 and the New-Year\n"
    "bursts (**Primer dia del ano** +1,262,485 on 2017-01-01, plus 2013-2016\n"
    "New Year days), the 2016 **Terremoto Manabi** aftershock week\n"
    "(+506,248 on 2016-04-18), and **Dia del Trabajo** (2017-05-01). The one\n"
    "top-15 day without a named holiday is **2017-04-01**, the largest raw\n"
    "sales day of the whole series (1,463,084) \u2014 a calendar-independent\n"
    "spike the holiday file does not explain.\n\n"
    "**3. Trends: strong and rising for the network, questionably flat for\n"
    "the small store.** Network trend first/last = 462,484 \u2192 745,999, yearly\n"
    "mean trend 385,121 (2013) \u2192 846,530 (2017); F_t = 0.793 \u2014 the trend\n"
    "band explains most non-seasonal movement. It is *not* monotonic: it\n"
    "peaked at 1,010,730 around end-2016 (Dec 2016 monthly mean 931,604) and\n"
    "eased through 2017 (last value 745,999). Store 44 (type A) follows the\n"
    "same up pattern (trend 28,809 \u2192 39,725). Store 26 (type D/E) has a\n"
    "weak unfocused signal: **F_seasonality = 0.171, F_trend = 0.306** and\n"
    "residual std 1,891 at a mean of 4,594 daily units \u2014 the level is too\n"
    "small relative to daily noise to call a durable trend (yearly means\n"
    "3,906 \u2192 4,960 within a highly erratic band).\n\n"
    "**4. Strengths (Wang, Smith & Hyndman 2006):**\n\n"
    "| series | F_seasonality | F_trend | resid std / mean |\n"
    "|---|---|---:|---:|\n"
    "| network | 0.570 | 0.793 | 0.150 |\n"
    "| store 44 (A) | 0.633 | 0.711 | \u2014 |\n"
    "| store 26 (D/E) | 0.171 | 0.306 | \u2014 |\n\n"
    "Seasonality is a real but secondary driver next to the trend; the weekly\n"
    "band peaks become the holiday residuals that dominate the tail.\n\n"
    "**5. Model choice: multiplicative-proxy by a hair \u2014 and the margin\n"
    "matters for honesty.** On the network series the additive fit leaves\n"
    "relative residual std 0.1323, the multiplicative proxy 0.1311 \u2014 the\n"
    "multiplicative branch is **0.94%** lower. By the documented rule\n"
    "(smaller resid_relative_std wins; strengths are not cross-comparable\n"
    "across log/original space) the choice is **multiplicative**, but the\n"
    "scientific statement is that the two models are **statistically almost\n"
    "tied**: the level-proportional component the multiplicative proxy absorbs\n"
    "is real yet tiny, so either view tells the same story for a monthly\n"
    "report. We record multiplicative as the pick and flag the margin rather\n"
    "than overselling it.\n\n"
    "**Verdict:** the network's daily story is a strong, still-easing 2013\u20132016\n"
    "climb (F_t 0.793), a robust Sunday-peaking weekly rhythm (F_seasonality\n"
    "0.570), and national-holiday shocks \u2014 Christmas closures and New Year \u2014\n"
    "as the dominant non-weekly peaks; the small store's week and level are\n"
    "too noisy to carry either claim.\n"
)

_STL_METHOD = (
    "## 2. Method: why STL, why period=7, why robust\n\n"
    "`stl_decompose` wraps `statsmodels.tsa.seasonal.STL`, an additive\n"
    "decomposition `observed = trend + seasonal + resid`\n"
    "(Cleveland et al. 1990). Three choices are deliberate:\n\n"
    "- **STL over classical decomposition**: each component is fitted with\n"
    "  LOESS at every point, so a single holiday spike deforms only its\n"
    "  neighbourhood (it lands in `resid`) instead of bending the whole\n"
    "  seasonal band or trend.\n"
    "- **period=7**: Stage 1 measured the weekday rhythm (Sun > Sat > midweek)\n"
    "  — the seasonal band must cycle every 7 days or the weekday pattern\n"
    "  leaks into `resid`.\n"
    "- **robust=True**: the iterated re-weighted pass down-weights outliers.\n"
    "  The 0-filled Christmas closures and New Year bursts would otherwise\n"
    "  pull the trend toward them.\n\n"
    "Everything below is computed by `src/decomposition.py`; this notebook\n"
    "only interprets."
)

_STRENGTHS_METHOD = (
    "## 5. How strong are the components? (Wang, Smith & Hyndman 2006)\n\n"
    "F_s = max(0, 1 \u2212 var(resid)/var(resid + seasonal)) and\n"
    "F_t = max(0, 1 \u2212 var(resid)/var(resid + trend)) measure how much of the\n"
    "detrended (F_s) / deseasonalised (F_t) variation each band absorbs.\n"
    "1.0 = the band explains everything left over; 0.0 = it explains nothing.\n"
    "A zero-variance denominator (constant series) returns 1.0 by convention."
)

_ADD_MULT_METHOD = (
    "## 6. Additive vs multiplicative model\n\n"
    "`decompose_additive_vs_multiplicative` fits the **additive** model by STL\n"
    "on the original scale and a **multiplicative proxy** by STL on\n"
    "`log(series)` (additive in log space \u2248 multiplicative in original space),\n"
    "mapping trend/seasonal back with `exp`. The multiplicative branch needs\n"
    "strictly positive input: the four **0-filled Christmas closures** (Stage-1\n"
    "decision) have `log(0)` undefined, so — only for this comparison — those\n"
    "four days are replaced by the mean of their neighbours, identically for\n"
    "both branches, and the adjusted rows are printed (nothing is silent).\n\n"
    "Decision rule: pick the model with the smaller `resid_relative_std`\n"
    "(= resid.std()/series.mean()) — the one quantity both fits report in the\n"
    "same sales units. The strength statistics *cannot* cross-compare (the\n"
    "multiplicative fit is only additive in log space), so they are reported\n"
    "for diagnostics, not used for the choice."
)

CELLS: list[dict[str, str]] = [
    {
        "type": "markdown",
        "source": (
            "# 03 \u2014 STL decomposition with national-holiday context\n\n"
            "**Stage 3** of the time-series analytics project. We split the\n"
            "network daily series (Stage 1) and the two pre-registered stores\n"
            "into trend + seasonality + resid with STL, explain the non-weekly\n"
            "peaks by **national holidays**, and justify the additive vs\n"
            "multiplicative model choice with executed numbers.\n\n"
            "**Status: EXPLORATORY (descriptive).** No significance claims;\n"
            "this feed the monthly-report story of \u00abtrend without weekly\n"
            "noise\u00bb and \u00abwhen the seasonal peak hits\u00bb.\n"
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
            "from IPython.display import display\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "pd.set_option('display.float_format', lambda v: f'{v:,.0f}')\n"
            "from src.eda import (\n"
            "    load_train, daily_network_sales, store_daily_sales, store_type,\n"
            "    pick_stl_stores,\n"
            ")\n"
            "from src.decomposition import (\n"
            "    stl_decompose, seasonality_strength, trend_strength,\n"
            "    decompose_additive_vs_multiplicative, national_holidays,\n"
            ")\n"
            "df = load_train(DATA)\n"
            "daily = daily_network_sales(df)\n"
            "# Single decision source (plan ruling): never hardcode 44/26 here.\n"
            "picked = pick_stl_stores(df)\n"
            "store_a, store_small = picked[0], picked[1]\n"
            "print(f'network daily series: {len(daily)} days '\n"
            "      f\"({daily.index.min().date()} .. {daily.index.max().date()})\")\n"
            "zero_days = daily[daily == 0]\n"
            "print(f'zero days (0-filled Christmas closures): '\n"
            "      f\"{list(zero_days.index.strftime('%Y-%m-%d'))}\")\n"
            "for s in picked:\n"
            "    print(f'STL store {s}: type {store_type(df, s)}, '\n"
            "          f'total sales {df.loc[df.store_nbr == s, \"sales\"].sum():,.0f}')"
        ),
    },
    {
        "type": "markdown",
        "source": _STL_METHOD,
    },
    {
        "type": "code",
        "source": (
            "comp_net = stl_decompose(daily)\n"
            "tr = comp_net['trend']\n"
            "print(f'network trend: first {tr.iloc[0]:,.0f} | last {tr.iloc[-1]:,.0f} | '\n"
            "      f\'min {tr.min():,.0f} | max {tr.max():,.0f}\')\n"
            "print(f'network seasonal: min {comp_net[\"seasonal\"].min():,.0f} | '\n"
            "      f'max {comp_net[\"seasonal\"].max():,.0f}')\n"
            "print(f'network resid: std {comp_net[\"resid\"].std():,.0f} | '\n"
            "      f'min {comp_net[\"resid\"].min():,.0f} | max {comp_net[\"resid\"].max():,.0f}')\n"
            "print(f'max daily sales: {daily.idxmax().date()} ({daily.max():,.0f})')\n"
            "# Additive identity printed here once so later numbers trust it.\n"
            "recon = comp_net.sum(axis=1)\n"
            "print(f'max |reconstruct - observed|: {(recon - daily).abs().max():.6f}')"
        ),
    },
    {
        "type": "code",
        "source": (
            "# 4-panel view of the network decomposition. Top panel pairs observed\n"
            "# (thin gray) with the STL trend so the level change is legible;\n"
            "# below are the three components on honest per-panel scales.\n"
            "fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)\n"
            "axes[0].plot(daily.index, daily.values, lw=0.6, color='0.78',\n"
            "             label='observed')\n"
            "axes[0].plot(comp_net.index, comp_net['trend'].values, lw=1.5,\n"
            "             color='#4C72B0', label='STL trend')\n"
            "axes[0].legend(loc='upper left', ncol=2)\n"
            "axes[1].plot(comp_net.index, comp_net['trend'].values, lw=1.2,\n"
            "             color='#4C72B0')\n"
            "axes[2].plot(comp_net.index, comp_net['seasonal'].values, lw=1.2,\n"
            "             color='#55A868')\n"
            "axes[3].plot(comp_net.index, comp_net['resid'].values, lw=0.8,\n"
            "             color='#C44E52')\n"
            "axes[0].set_title('STL decomposition: network daily sales')\n"
            "axes[1].set_ylabel('trend (units)')\n"
            "axes[2].set_ylabel('seasonal (units)')\n"
            "axes[3].set_ylabel('resid (units)')\n"
            "axes[3].set_xlabel('date')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/stl_network.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "code",
        "source": (
            "# Milestones the written conclusions cite, all computed here:\n"
            "# the weekly seasonal peak day, the per-year trend levels, and the\n"
            "# end-2016 vs 2017 trend reading (peak then easing).\n"
            "seasonal_dow = comp_net['seasonal'].groupby(comp_net.index.dayofweek).mean()\n"
            "weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\n"
            "for k in range(7):\n"
            "    print(f'{weekday_names[k]}: seasonal {seasonal_dow[k]:,.0f}')\n"
            "peak_dow = int(seasonal_dow.idxmax())\n"
            "trough_dow = int(seasonal_dow.idxmin())\n"
            "print(f'peak weekday: {weekday_names[peak_dow]} '\n"
            "      f'({seasonal_dow[peak_dow]:,.0f}) | trough: '\n"
            "      f'{weekday_names[trough_dow]} ({seasonal_dow[trough_dow]:,.0f})')\n"
            "tr = comp_net['trend']\n"
            "print('trend yearly mean:')\n"
            "print(tr.groupby(tr.index.year).mean().round(0).to_string())\n"
            "print(f'trend max {tr.idxmax().date()} ({tr.max():,.0f}) | '\n"
            "      f'trend last {tr.iloc[-1]:,.0f}')\n"
            "print(f'trend monthly mean 2016-12: {tr[\"2016-12\"].mean():,.0f}')\n"
            "print(f'trend mean 2017: {tr[\"2017\":].mean():,.0f}')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 3. STL on the two pre-registered stores\n\n"
            "The pick is the **largest type-A store** (`store_a`) and the\n"
            "**smallest type-D/E store** (`store_small`), recomputed here by\n"
            "`pick_stl_stores` (the single decision source from Stage 1).\n"
            "Same period/robust settings: the weekly rhythm drives both, and a\n"
            "smaller store is the scale contrast for the seasonal read."
        ),
    },
    {
        "type": "code",
        "source": (
            "s_a = store_daily_sales(df, store_a)\n"
            "comp_a = stl_decompose(s_a)\n"
            "tr_a = comp_a['trend']\n"
            "print(f'store {store_a} (type {store_type(df, store_a)}): '\n"
            "      f'mean {s_a.mean():,.0f} | max {s_a.max():,.0f}')\n"
            "print(f'  trend first {tr_a.iloc[0]:,.0f} | last {tr_a.iloc[-1]:,.0f} | '\n"
            "      f'min {tr_a.min():,.0f} | max {tr_a.max():,.0f}')\n"
            "print(f'  resid std {comp_a[\"resid\"].std():,.0f} | '\n"
            "      f'max |resid| {comp_a[\"resid\"].abs().max():,.0f}')\n"
            "print('  trend yearly mean: ' + ', '.join(\n"
            "    f'{y}: {v:,.0f}' for y, v in tr_a.groupby(tr_a.index.year).mean().items()))\n"
            "fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)\n"
            "axes[0].plot(s_a.index, s_a.values, lw=0.6, color='0.78', label='observed')\n"
            "axes[0].plot(comp_a.index, comp_a['trend'].values, lw=1.5,\n"
            "             color='#4C72B0', label='STL trend')\n"
            "axes[0].legend(loc='upper left', ncol=2)\n"
            "axes[1].plot(comp_a.index, comp_a['trend'].values, lw=1.2, color='#4C72B0')\n"
            "axes[2].plot(comp_a.index, comp_a['seasonal'].values, lw=1.2, color='#55A868')\n"
            "axes[3].plot(comp_a.index, comp_a['resid'].values, lw=0.8, color='#C44E52')\n"
            "axes[0].set_title(f'STL decomposition: store {store_a} (type A)')\n"
            "axes[1].set_ylabel('trend (units)')\n"
            "axes[2].set_ylabel('seasonal (units)')\n"
            "axes[3].set_ylabel('resid (units)')\n"
            "axes[3].set_xlabel('date')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/stl_store_a.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "code",
        "source": (
            "s_m = store_daily_sales(df, store_small)\n"
            "comp_m = stl_decompose(s_m)\n"
            "tr_m = comp_m['trend']\n"
            "print(f'store {store_small} (type {store_type(df, store_small)}): '\n"
            "      f'mean {s_m.mean():,.0f} | max {s_m.max():,.0f}')\n"
            "print(f'  trend first {tr_m.iloc[0]:,.0f} | last {tr_m.iloc[-1]:,.0f} | '\n"
            "      f'min {tr_m.min():,.0f} | max {tr_m.max():,.0f}')\n"
            "print(f'  resid std {comp_m[\"resid\"].std():,.0f} | '\n"
            "      f'max |resid| {comp_m[\"resid\"].abs().max():,.0f}')\n"
            "print('  trend yearly mean: ' + ', '.join(\n"
            "    f'{y}: {v:,.0f}' for y, v in tr_m.groupby(tr_m.index.year).mean().items()))\n"
            "fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)\n"
            "axes[0].plot(s_m.index, s_m.values, lw=0.6, color='0.78', label='observed')\n"
            "axes[0].plot(comp_m.index, comp_m['trend'].values, lw=1.5,\n"
            "             color='#4C72B0', label='STL trend')\n"
            "axes[0].legend(loc='upper left', ncol=2)\n"
            "axes[1].plot(comp_m.index, comp_m['trend'].values, lw=1.2, color='#4C72B0')\n"
            "axes[2].plot(comp_m.index, comp_m['seasonal'].values, lw=1.2, color='#55A868')\n"
            "axes[3].plot(comp_m.index, comp_m['resid'].values, lw=0.8, color='#C44E52')\n"
            "axes[0].set_title(f'STL decomposition: store {store_small} (type D/E)')\n"
            "axes[1].set_ylabel('trend (units)')\n"
            "axes[2].set_ylabel('seasonal (units)')\n"
            "axes[3].set_ylabel('resid (units)')\n"
            "axes[3].set_xlabel('date')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/stl_store_small.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 4. Do national holidays explain the non-weekly peaks?\n\n"
            "With `period=7` the seasonal band is a *weekly* pattern \u2014 it\n"
            "cannot encode calendar-date events like Christmas. Holiday effects\n"
            "therefore land in **resid** (single-day bursts). To test that story\n"
            "we cross the largest residual days against the national holidays in\n"
            "`holidays_events.csv` (locale == National, within the series range;\n"
            "transferred rows kept, not analysed)."
        ),
    },
    {
        "type": "code",
        "source": (
            "holidays = national_holidays(HOLIDAYS, daily.index.min().date(),\n"
            "                            daily.index.max().date())\n"
            "print(f'national holidays in range: {len(holidays)} rows on '\n"
            "      f'{holidays[\"date\"].nunique()} unique dates')\n"
            "top = comp_net['resid'].abs().nlargest(15).rename('abs_resid')\n"
            "hd = holidays.set_index('date')['description']\n"
            "top = top.to_frame()\n"
            "top['date'] = top.index\n"
            "# hd.loc[d] is a scalar str for single-holiday days, a Series\n"
            "# otherwise; normalize both to a list before joining.\n"
            "def holiday_names(d):\n"
            "    if d not in hd.index:\n"
            "        return ''\n"
            "    vals = hd.loc[d]\n"
            "    vals = [vals] if isinstance(vals, str) else list(vals)\n"
            "    return ', '.join(vals)\n"
            "top['holiday'] = [holiday_names(d) for d in top['date']]\n"
            "top = top[['date', 'abs_resid', 'holiday']].reset_index(drop=True)\n"
            "print('top-15 |resid| days and whether a national holiday falls on them:')\n"
            "display(top)\n"
            "print()\n"
            "print(f'{int((top[\"holiday\"] != \"\").sum())} of 15 top residual days '\n"
            "      'are exactly a national-holiday date')"
        ),
    },
    {
        "type": "code",
        "source": (
            "# Upper panel: raw sales with every national holiday as a thin gray\n"
            "# line. Lower panel: resid (same x-range). Red lines + rotated labels\n"
            "# mark the holidays that coincide with a top-15 residual day \u2014 the\n"
            "# data-driven subset, so labels stay readable (~a handful).\n"
            "labeled = top.loc[top['holiday'] != '', 'date']\n"
            "fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)\n"
            "for ax, series, color in ((axes[0], daily, '#4C72B0'),\n"
            "                          (axes[1], comp_net['resid'], '#C44E52')):\n"
            "    ax.plot(series.index, series.values, lw=0.7, color=color)\n"
            "    for day in holidays['date']:\n"
            "        ax.axvline(day, color='0.75', lw=0.7, zorder=0)\n"
            "    for day in labeled:\n"
            "        ax.axvline(day, color='#8C1515', lw=1.6, zorder=1)\n"
            "axes[0].set_title('Daily network sales with national holidays '\n"
            "                  '(gray = all, red = on a top residual day)')\n"
            "axes[0].set_ylabel('sales (units)')\n"
            "axes[1].set_title('STL residual \u2014 holiday spikes land here, '\n"
            "                   'not in the weekly seasonal band')\n"
            "axes[1].set_ylabel('resid (units)')\n"
            "axes[1].set_xlabel('date')\n"
            "ymax = axes[0].get_ylim()[1]\n"
            "for i, day in enumerate(labeled):\n"
            "    vals = hd.loc[day]\n"
            "    vals = [vals] if isinstance(vals, str) else list(vals)\n"
            "    notes = '\\n'.join(vals)\n"
            "    axes[0].text(day, ymax * 0.98, notes, rotation=90,\n"
            "                 fontsize=7, va='top', ha='right',\n"
            "                 color='#8C1515')\n"
            "from matplotlib.lines import Line2D\n"
            "axes[0].legend(handles=[\n"
            "    Line2D([0], [0], color='0.75', lw=1.2, label='national holiday'),\n"
            "    Line2D([0], [0], color='#8C1515', lw=1.6,\n"
            "           label='holiday on a top-15 |resid| day'),\n"
            "], loc='upper left')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/seasonal_peaks_holidays.png', dpi=150,\n"
            "            bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": _STRENGTHS_METHOD,
    },
    {
        "type": "code",
        "source": (
            "strengths = pd.DataFrame({\n"
            "    'series': ['network', f'store {store_a} (A)', f'store {store_small} (D/E)'],\n"
            "    'F_seasonality': [seasonality_strength(comp_net),\n"
            "                      seasonality_strength(comp_a),\n"
            "                      seasonality_strength(comp_m)],\n"
            "    'F_trend': [trend_strength(comp_net),\n"
            "                trend_strength(comp_a),\n"
            "                trend_strength(comp_m)],\n"
            "    'resid std / mean': [\n"
            "        comp_net['resid'].std() / daily.mean(),\n"
            "        comp_a['resid'].std() / s_a.mean(),\n"
            "        comp_m['resid'].std() / s_m.mean(),\n"
            "    ],\n"
            "})\n"
            "# Cell-1 float_format is integer; the strength table needs 3 dp,\n"
            "# so render it in an isolated option context instead of mutating\n"
            "# the global display state.\n"
            "with pd.option_context('display.float_format', lambda v: f'{v:.3f}'):\n"
            "    display(strengths)\n"
            "for _, row in strengths.iterrows():\n"
            "    print(f\"{row['series']}: F_seasonality {row['F_seasonality']:.3f} | \"\n"
            "          f\"F_trend {row['F_trend']:.3f} | resid/mean {row['resid std / mean']:.4f}\")"
        ),
    },
    {
        "type": "markdown",
        "source": _ADD_MULT_METHOD,
    },
    {
        "type": "code",
        "source": (
            "# Strictly-positive input for the log proxy. Only the four 0-filled\n"
            "# closure days change, each replaced by the mean of its neighbours;\n"
            "# the adjusted rows are printed so the policy is auditable.\n"
            "zero_days = daily[daily == 0]\n"
            "pos = daily.copy()\n"
            "adj = []\n"
            "for day in zero_days.index:\n"
            "    left, right = daily.shift(1)[day], daily.shift(-1)[day]\n"
            "    pos[day] = (left + right) / 2\n"
            "    adj.append((day.date(), float(daily[day]), float(left),\n"
            "                float(right), float(pos[day])))\n"
            "print('adjusted closure days (original -> neighbour mean):')\n"
            "print(pd.DataFrame(adj, columns=['date', 'original', 'prev', 'next', 'replacement'])\n"
            "      .to_string(index=False))\n"
            "cmp = decompose_additive_vs_multiplicative(pos)\n"
            "rows = []\n"
            "for model in ('additive', 'multiplicative'):\n"
            "    e = cmp[model]\n"
            "    rows.append({\n"
            "        'model': model,\n"
            "        'F_seasonality': e['seasonality_strength'],\n"
            "        'F_trend': e['trend_strength'],\n"
            "        'resid_relative_std': e['resid_relative_std'],\n"
            "        'resid std': float(e['resid'].std()),\n"
            "    })\n"
            "cmp_table = pd.DataFrame(rows)\n"
            "with pd.option_context('display.float_format', lambda v: f'{v:.4f}'):\n"
            "    display(cmp_table)\n"
            "add_rrs = cmp['additive']['resid_relative_std']\n"
            "mult_rrs = cmp['multiplicative']['resid_relative_std']\n"
            "print(f'recommendation: {cmp[\"recommendation\"]}')\n"
            "print(f'relative difference (mult - add)/add: '\n"
            "      f'{(mult_rrs - add_rrs) / add_rrs:+.2%}')"
        ),
    },
    {
        "type": "code",
        "source": (
            "# Residuals of both fits on one panel, same units: the visual story\n"
            "# behind the relative-std numbers. Structureless, level-independent\n"
            "# scatter supports the selected model; a funnel growing with the\n"
            "# level supports the other.\n"
            "fig, ax = plt.subplots(figsize=(11, 4.4))\n"
            "ax.plot(cmp['additive']['resid'].index,\n"
            "        cmp['additive']['resid'].values, lw=0.7, alpha=0.8,\n"
            "        color='#4C72B0',\n"
            "        label=f\"additive resid (rel_std {add_rrs:.4f})\")\n"
            "ax.plot(cmp['multiplicative']['resid'].index,\n"
            "        cmp['multiplicative']['resid'].values, lw=0.7,\n"
            "        color='#DD8452',\n"
            "        label=f\"multiplicative resid (rel_std {mult_rrs:.4f})\")\n"
            "ax.axhline(0, color='0.4', lw=0.8)\n"
            "ax.set_title('Additive vs multiplicative-proxy residuals on the '\n"
            "             'network series')\n"
            "ax.set_xlabel('date')\n"
            "ax.set_ylabel('reconstruction residual (sales units)')\n"
            "ax.legend(loc='upper left')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/additive_vs_multiplicative.png', dpi=150,\n"
            "            bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": CONCLUSION,
    },
]


def main() -> None:
    out = build(CELLS, "03_stl_decomposition")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()