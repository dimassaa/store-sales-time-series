"""Builds notebooks/02_smoothing.ipynb (Stage 2: MA and EMA smoothing)."""

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
)

# Interim one-liner; replaced from the pass-A executed outputs so every
# number in the written conclusion is a real executed value (project rule).
# All figures below come from the pass-A prints of cells 3, 6 and 9.
CONCLUSION = (
    "## 4. Conclusion: which view answers the trend question\n\n"
    "The weekly rhythm from Stage 1 (Sun \u2248 822k vs Thu \u2248 503k) is exactly\n"
    "the noise a \u00abtrend\u00bb read must remove. Measured on the smoothed series:\n\n"
    "| view | weekday amplitude | day-to-day std | mean level |\n"
    "|---|---|---:|---:|\n"
    "| raw | 318,621 | 172,063 | 636,046 |\n"
    "| MA7 | 881 | 19,481 | \u2014 |\n"
    "| MA14 | 1,069 | 10,664 | \u2014 |\n"
    "| MA30 | 17,748 | 8,002 | 636,922 |\n\n"
    "- **Window 7 is the answer to \u00abtrend without weekly noise\u00bb.** It\n"
    "  collapses the weekday amplitude from **318,621 to 881** (\u221299.7%)\n"
    "  and the day-to-day variation from **172,063 to 19,481** (\u221288.7%).\n"
    "  MA14 is calmer still (std 10,664) but at double the phase lag\n"
    "  ((14\u22121)/2 vs (7\u22121)/2 days) \u2014 more smoothing buys you nothing for\n"
    "  the weekly-cycle question, only lag.\n"
    "- **Window 30 is the monthly-level read.** Its window is 30 days, not a\n"
    "  multiple of 7, so a small slice of the weekly cycle leaks through\n"
    "  (17,748, ~5.6% of raw) \u2014 but the level it tracks is stable: mean\n"
    "  636,922 across the run, matching the raw mean 636,046, and a last\n"
    "  value of 832,844. \u00abMonthly level\u00bb rests on exactly this: a 30-day\n"
    "  trailing mean is the long-run level by construction.\n"
    "- **EMA7 sits closer to the observed series than MA7**: mean |raw \u2212 EMA7|\n"
    "  = 92,308 vs 109,751 for MA7, and its empirical lag is **1 day vs 3**\n"
    "  days. The reason is the weight shape: the newest point enters EMA with\n"
    "  weight \u03b1 = 1/4 against MA's equal 1/7, so EMA reacts within days\n"
    "  while MA waits for the full window to turn over. The two stay on the\n"
    "  same level (mean EMA7 \u2212 MA7 = \u2212149.5; mean |EMA7 \u2212 MA7| = 30,138\n"
    "  over the run) \u2014 they differ in the *shape* of response, not in where\n"
    "  they point.\n"
    "- **Which view to keep:** MA7 for a clean, weekly-noise-free trend read\n"
    "  with a known 3-day lag (STL in stage 3 will handle the phase); EMA7 if\n"
    "  the question is \u00abwhere is the level *right now*\u00bb \u2014 visibly closer,\n"
    "  still no weekly noise. Both warm up in 6 days; EMA30/MA30 in 29.\n\n"
    "**Verdict:** on this network daily series, the view that answers\n"
    "    \u00abthe trend without the weekly noise\u00bb is a **7-day moving average**\n"
    "    (window equals the seasonal period), with EMA7 as the\n"
    "    faster-responding alternative."
)

CELLS: list[dict[str, str]] = [
    {
        "type": "markdown",
        "source": (
            "# 02 — Smoothing: MA and EMA\n\n"
            "**Stage 2** of the time-series analytics project. We smooth the\n"
            "weekly/holiday noise of the daily network series (Stage 1 output)\n"
            "with a simple moving average (MA) and an exponential moving\n"
            "average (EMA), then fix which view answers the question\n"
            "**\u00abwhat is the trend once the weekly noise is gone\u00bb**.\n\n"
            "**Status: EXPLORATORY (hypothesis-generating).** Nothing here is a\n"
            "significance claim; smoothing is a descriptive read that feeds the\n"
            "STL stage. Both smoothers live in `src/smoothing.py` — this\n"
            "notebook only chooses windows/spans and interprets the result.\n"
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
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "pd.set_option('display.float_format', lambda v: f'{v:,.0f}')\n"
            "from src.eda import load_train, daily_network_sales\n"
            "from src.smoothing import simple_moving_average, exponential_moving_average\n"
            "df = load_train(DATA)\n"
            "daily = daily_network_sales(df)\n"
            "# Same series as Stage 1: full calendar 2013-01-01..2017-08-15,\n"
            "# network total per day, 0-filled only on the four Christmas days.\n"
            "print(f'daily series: {len(daily)} days, '\n"
            "      f\"{daily.index.min().date()} .. {daily.index.max().date()}\")\n"
            "daily.head()"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 1. Simple moving average: windows 7 / 14 / 30\n\n"
            "`simple_moving_average(daily, window)` uses a trailing window:\n"
            "each point is the mean of exactly `window` observations, and\n"
            "incomplete windows are NaN (warm-up), never partial means. Window\n"
            "order = smoothing period: **7** is one full week (the weekly\n"
            "rhythm found in Stage 1), **14** two weeks, **30** a roughly\n"
            "monthly level.\n\n"
            "Two summary numbers tell the trade-off: the **weekday amplitude**\n"
            "(max\u2212min of mean sales by day of week \u2014 how much weekly cycle\n"
            "survives) and the **day-to-day std** (volatility left after\n"
            "smoothing)."
        ),
    },
    {
        "type": "code",
        "source": (
            "ma7 = simple_moving_average(daily, 7)\n"
            "ma14 = simple_moving_average(daily, 14)\n"
            "ma30 = simple_moving_average(daily, 30)\n"
            "\n"
            "def weekday_amplitude(s):\n"
            "    \"\"\"Max-min of mean daily sales by weekday \u2014 the size of the\n"
            "    weekly cycle that survives a smoothing window.\"\"\"\n"
            "    by_dow = s.groupby(s.index.dayofweek).mean()\n"
            "    return float(by_dow.max() - by_dow.min())\n"
            "\n"
            "summary = {}\n"
            "for label, s in {'raw': daily, 'MA7': ma7, 'MA14': ma14, 'MA30': ma30}.items():\n"
            "    summary[label] = {\n"
            "        'weekday amplitude': weekday_amplitude(s),\n"
            "        'day-to-day std': float(s.diff().std()),\n"
            "        'mean': float(s.mean()),\n"
            "        'last value': float(s.iloc[-1]),\n"
            "    }\n"
            "pd.DataFrame(summary).T\n"
            "\n"
            "amp = {k: v['weekday amplitude'] for k, v in summary.items()}\n"
            "vlt = {k: v['day-to-day std'] for k, v in summary.items()}\n"
            "print('weekday amplitude (raw %.0f | MA7 %.0f | MA14 %.0f | MA30 %.0f)'\n"
            "      % (amp['raw'], amp['MA7'], amp['MA14'], amp['MA30']))\n"
            "print('day-to-day std    (raw %.0f | MA7 %.0f | MA14 %.0f | MA30 %.0f)'\n"
            "      % (vlt['raw'], vlt['MA7'], vlt['MA14'], vlt['MA30']))\n"
            "print('mean level: raw %.0f | MA30 %.0f | MA30 last value %.0f'\n"
            "      % (summary['raw']['mean'], summary['MA30']['mean'],\n"
            "         summary['MA30']['last value']))"
        ),
    },
    {
        "type": "code",
        "source": (
            "fig, ax = plt.subplots(figsize=(11, 4.2))\n"
            "# Raw series in thin gray is honest context: it shows exactly how\n"
            "# much noise each window removes (nothing is hidden or decimated).\n"
            "ax.plot(daily.index, daily.values, lw=0.6, color='0.75', label='raw daily')\n"
            "ax.plot(ma7.index, ma7.values, lw=1.3, color='#4C72B0', label='MA window=7')\n"
            "ax.plot(ma14.index, ma14.values, lw=1.3, color='#DD8452', label='MA window=14')\n"
            "ax.plot(ma30.index, ma30.values, lw=1.5, color='#55A868', label='MA window=30')\n"
            "ax.set_title('Daily network sales smoothed: MA7 / MA14 / MA30')\n"
            "ax.set_xlabel('date')\n"
            "ax.set_ylabel('sales (units)')\n"
            "ax.legend(loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/smoothing_ma.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 2. Exponential moving average: spans 7 / 30\n\n"
            "`exponential_moving_average(daily, span)` runs `ewm(span=span,\n"
            "adjust=False, min_periods=span)`: the decay is `alpha = 2 / (span\n"
            "+ 1)`, so the mean age of the weights is `(span \u2212 1) / 2` days\n"
            "\u2014 the same average lag as an MA of `window = span`, but spread\n"
            "differently: each new point enters with weight `alpha` while every\n"
            "older point keeps a positive, geometrically shrinking tail.\n"
            "`adjust=False` keeps the textbook recursion\n"
            "`y[t] = (1 \u2212 alpha) y[t\u22121] + alpha x[t]`. The first `span \u2212 1`\n"
            "points are NaN (warm-up matching an MA of the same order), then the\n"
            "series is gapless."
        ),
    },
    {
        "type": "code",
        "source": (
            "ema7 = exponential_moving_average(daily, 7)\n"
            "ema30 = exponential_moving_average(daily, 30)\n"
            "print('fidelity to raw: mean |raw \u2212 MA7| %.0f | EMA7 %.0f |'\n"
            "      ' MA30 %.0f | EMA30 %.0f'\n"
            "      % ((daily - ma7).abs().mean(), (daily - ema7).abs().mean(),\n"
            "         (daily - ma30).abs().mean(), (daily - ema30).abs().mean()))\n"
            "print('last values: MA7 %.0f | EMA7 %.0f | MA30 %.0f | EMA30 %.0f'\n"
            "      % (ma7.iloc[-1], ema7.iloc[-1], ma30.iloc[-1], ema30.iloc[-1]))\n"
            "print('EMA30 warm-up NaN: %d of %d; EMA7 warm-up NaN: %d of %d'\n"
            "      % (int(ema30.isna().sum()), len(ema30),\n"
            "         int(ema7.isna().sum()), len(ema7)))"
        ),
    },
    {
        "type": "code",
        "source": (
            "fig, ax = plt.subplots(figsize=(11, 4.2))\n"
            "ax.plot(daily.index, daily.values, lw=0.6, color='0.75', label='raw daily')\n"
            "ax.plot(ema7.index, ema7.values, lw=1.3, color='#DD8452', label='EMA span=7')\n"
            "ax.plot(ema30.index, ema30.values, lw=1.5, color='#4C72B0', label='EMA span=30')\n"
            "ax.set_title('Daily network sales: EMA7 and EMA30 (span = smoothing horizon)')\n"
            "ax.set_xlabel('date')\n"
            "ax.set_ylabel('sales (units)')\n"
            "ax.legend(loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/smoothing_ema.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 3. MA7 vs EMA7: same order, different lag profile\n\n"
            "Both have a mean weight age of 3 days for window/span 7, but the\n"
            "weight shapes differ \u2014 MA cuts off hard after 7 days, EMA decays\n"
            "forever. The empirical lag below regresses each smoother on a\n"
            "back-shifted raw series (the shift that minimizes MSE is how many\n"
            "days the smoother trails reality), then the final segment makes\n"
            "the phase difference visible."
        ),
    },
    {
        "type": "code",
        "source": (
            "def best_lag(smoothed, raw, max_lag=10):\n"
            "    \"\"\"Empirical lag in days: the back-shift of *raw* that makes\n"
            "    the smoother most closely track it (minimal MSE).\"\"\"\n"
            "    valid = smoothed.dropna()\n"
            "    best_d, best_mse = 0, float('inf')\n"
            "    for d in range(max_lag + 1):\n"
            "        aligned = raw.shift(d).loc[valid.index]\n"
            "        mse = float(((valid - aligned) ** 2).mean())\n"
            "        if mse < best_mse:\n"
            "            best_d, best_mse = d, mse\n"
            "    return best_d, best_mse\n"
            "\n"
            "lag_ma7, mse_ma7 = best_lag(ma7, daily)\n"
            "lag_ema7, mse_ema7 = best_lag(ema7, daily)\n"
            "print('empirical lag: MA7 %d days | EMA7 %d days' % (lag_ma7, lag_ema7))\n"
            "print('mean signed (EMA7 \u2212 MA7): %.1f  |  mean |EMA7 \u2212 MA7|: %.0f'\n"
            "      % ((ema7 - ma7).mean(), (ema7 - ma7).abs().mean()))"
        ),
    },
    {
        "type": "code",
        "source": (
            "# A lag of 3 days is invisible at the 4.5-year scale, so compare on\n"
            "# the final segment where the weekly rhythm is still strong.\n"
            "zoom = slice(pd.Timestamp('2017-01-01'), pd.Timestamp('2017-08-15'))\n"
            "fig, ax = plt.subplots(figsize=(11, 4.2))\n"
            "ax.plot(daily.loc[zoom].index, daily.loc[zoom].values, lw=0.7,\n"
            "        color='0.75', label='raw daily')\n"
            "ax.plot(ma7.loc[zoom].index, ma7.loc[zoom].values, lw=1.5,\n"
            "        color='#4C72B0', label='MA window=7')\n"
            "ax.plot(ema7.loc[zoom].index, ema7.loc[zoom].values, lw=1.5,\n"
            "        color='#DD8452', label='EMA span=7')\n"
            "ax.set_title('MA7 vs EMA7, 2017-01-01 .. 2017-08-15: weekly peaks traced '\n"
            "             'with different lag profiles')\n"
            "ax.set_xlabel('date')\n"
            "ax.set_ylabel('sales (units)')\n"
            "ax.legend(loc='best')\n"
            "fig.tight_layout()\n"
            "fig.savefig('../assets/smoothing_ma_vs_ema.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
    },
    {
        "type": "markdown",
        "source": CONCLUSION,
    },
]


def main() -> None:
    out = build(CELLS, "02_smoothing")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()