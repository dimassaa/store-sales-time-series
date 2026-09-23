# Store Sales — Descriptive Time-Series Analysis

A complete, reproducible descriptive cycle on the Kaggle Corporación Favorita
store-sales dataset: data-integrity checks, trend/seasonality smoothing,
STL decomposition, national-holiday mapping, and an honest verdict framed as
a monthly report. The project is intentionally *descriptive*: no forecasting,
no significance tests, no store/region drill-down.

[Русская версия](README_ru.md)

![Python](https://img.shields.io/badge/Python-3.12.3-3776AB)
![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Method](https://img.shields.io/badge/method-STL%20%2B%20MA%2FEMA-blue)

![STL decomposition of daily network sales: observed, trend, weekly seasonal band and residual](assets/stl_network.png)

> **TL;DR — the monthly-report answer.** Daily network sales grew strongly
> from 2013 through late 2016 (STL trend strength F_t = 0.793; yearly trend
> means 385,121 → 846,530) and then eased into 2017. Sales keep a stable
> weekly rhythm that peaks on Sunday (mean 821,794/day vs Thursday trough
> 503,173/day) and the calendar outliers are national holidays: 14 of the 15
> largest residual days fall exactly on a holiday, led by the Christmas
> closures (Navidad 2016 at −1,321,021) and New Year bursts (+1,262,485 on
> 2017-01-01). Between the additive and multiplicative models the data pick
> multiplicative, but only by 0.94% of relative residual spread — the two are
> effectively tied. The verdict is descriptive and exploratory: it does **not**
> forecast, fit an ARIMA-family model, or claim statistical significance for
> any single number. Full reasoning: `reports/04_executive_summary.html`.

## Table of Contents

- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Installation and Usage](#detailed-installation-and-usage)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Methodology](#methodology)
- [Results](#results)
- [Learning Resources](#learning-resources)
- [Testing](#testing)
- [Limitations](#limitations)
- [Recommendations / Next Steps](#recommendations--next-steps)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project simulates a regular analyst task: answer the questions a monthly
report asks about daily sales, in descriptive terms. It is built for a
teaching/portfolio context — every stage is a tested, reusable `src/`
function, and the reasoning behind each choice is documented in `notes/`.

The four questions a monthly report asks, and where each is answered:

| Monthly-report question | Where it lives |
| ----------------------- | -------------- |
| Is there a long-term trend, and where is it heading? | `src/smoothing.py`, notebooks 02–03 |
| When is the seasonal peak and what explains it? | `src/decomposition.py`, notebook 03 |
| What is the weekly rhythm (which weekdays peak)? | `src/eda.py`, notebook 04 §3 |
| Additive or multiplicative seasonal model? | `src/decomposition.py`, notebook 04 §4 |

Main contributions:

- Data integrity verified before any conclusion (unit of analysis, duplicates,
  missing calendar days, negative sales, value ranges).
- Trend views with simple (MA) and exponential (EMA) moving averages —
  tested formulas, cross-checked against `pandas`.
- STL decomposition (trend + weekly seasonal band + residual) with the
  seasonality/trend strength metrics F_s and F_t (Wang, Smith & Hyndman).
- National-holiday mapping of the non-weekly peak days — 14 of the top 15
  residual days land exactly on a national holiday.
- An honest additive-vs-multiplicative comparison reported as the near-tie it
  is (0.94% gap, no test run on it).

> [!NOTE]
> What this project is **not**: it does not forecast future sales, it does
> not run statistical significance tests, and it does not drill into
> store/region-level analytics. Those are deliberate scope decisions, not
> oversights — see [Limitations](#limitations).

## Tech Stack

| Category            | Technologies                                            |
| ------------------- | ------------------------------------------------------- |
| Language            | Python 3.12.3                                           |
| Statistics          | statsmodels (STL), NumPy                                |
| Data processing     | pandas 3.0.6                                            |
| Visualization       | Matplotlib                                              |
| Notebooks           | Jupyter (nbformat + nbclient reproducible builds)       |
| Testing             | pytest (56 tests)                                       |
| Environment         | uv / venv, `requirements.txt` with pinned versions      |

## Project Structure

```
├── src/                  Reusable, tested functions (eda, smoothing, decomposition)
├── tests/                pytest coverage for src/ (56 tests)
├── notebooks/            Stages 01–04, built from build_*.py scripts (outputs embedded)
├── reports/              Standalone HTML executive summary (nbconvert)
├── assets/               PNG figures reused by the README and notebooks
├── notes/                Learning notes per stage (Russian, key terms in English)
├── data/                 Dataset (source documented in data/README.md, CSV not committed)
├── README.md             This file
├── README_ru.md          Russian mirror of this file
├── LICENSE               MIT
└── requirements.txt      Pinned Python dependencies
```

## Quick Start

Prerequisite: **Python 3.12** and the dataset placed under `data/` (the CSV
is git-ignored — see `data/README.md` for re-download instructions). Then set
up the environment by either option below.

**Option A: uv (recommended)**

```bash
git clone https://github.com/<your-user>/store-sales-time-series.git
cd store-sales-time-series
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 56 passed
```

**Option B: classic venv + pip**

```bash
git clone https://github.com/<your-user>/store-sales-time-series.git
cd store-sales-time-series
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q                     # 56 passed
```

**Option C: explore the report without installing anything**

Open `reports/04_executive_summary.html` in a browser — it is a standalone
HTML file with all figures embedded (no Jupyter required).

## Detailed Installation and Usage

Prerequisites and environment setup as in [Quick Start](#quick-start).

Rebuild every notebook from the tested `src/` functions (the build scripts
must run from the `notebooks/` directory):

```bash
cd notebooks
for b in build_01_eda build_02_smoothing build_03_stl build_04_summary; do
  ../.venv/bin/python "$b.py"
done
cd ..
```

Export the executive summary to a standalone HTML report:

```bash
.venv/bin/python -m jupyter nbconvert --to html \
  --output-dir reports notebooks/04_executive_summary.ipynb
```

Open a notebook interactively:

```bash
.venv/bin/python -m jupyter notebook notebooks/04_executive_summary.ipynb
```

Notebook rebuilds are deterministic: running the loop twice produces
byte-identical `.ipynb` files (stable cell ids, no execution timestamps), so
`git status` stays clean after any rebuild.

## Data

Source: **Store Sales — Time Series Forecasting**, a Kaggle competition run
by Corporación Favorita (a major Ecuadorian grocery retailer). Redownload
instructions and full schema are in `data/README.md`.

| Stat                          | Value            |
| ----------------------------- | ---------------- |
| Rows                          | 3,000,888        |
| Stores                        | 54               |
| Product families              | 33               |
| Date range                    | 2013-01-01 .. 2017-08-15 (1,688 days) |
| Zero-filled closure days      | 25 Dec 2013–2016 (Christmas, documented and auditable) |
| Largest single day            | 2017-04-01 → 1,463,084 (not a holiday) |

Files used: `train.csv` (schema `id`, `date`, `store_nbr`, `family`, `sales`,
`onpromotion`), `stores.csv` (city, state, type A–E, cluster), and
`holidays_events.csv` (national/regional/local holidays with transfer dates).

Files **not** used — a documented design decision: `oil.csv`,
`transactions.csv`, `test (1).csv` and `sample_submission (1).csv`. Oil price
and transactions are plausible external factors, but the project scope is a
single-series description of daily network sales; adding regressors would
turn a descriptive analysis into a modeling exercise (see
[Limitations](#limitations), [Recommendations](#recommendations--next-steps)).

Cleaning: nothing is dropped. Data integrity is asserted before analysis
(no duplicate `(date, store_nbr, family)` rows, no missing values, no
negative sales, no gaps in the daily calendar). The four Christmas closure
days are absent from the source data and zero-filled by the pipeline
(`daily_network_sales`, full-calendar index — a documented, auditable
policy); they are kept as zeros for the STL stages and replaced by
neighbour means **only** for the additive-vs-multiplicative comparison,
with the adjusted rows printed in notebook 04 so the policy is auditable.

## Exploratory Data Analysis

- Daily network sales are noisy at the daily scale, so a weekly rhythm is
  not visible in the raw series — smoothing is required before any trend
  claim (see [Methodology](#methodology)).
- The weekday profile shows a clear, stable weekly rhythm: Sunday is the peak
  day, Thursday the trough.
- Descriptive numbers only — no inference is drawn from EDA alone; all
  summary numbers below are recomputed by notebook 04.

![Daily network sales, 2013-01-01 .. 2017-08-15](assets/eda_daily_sales.png)

> [!NOTE]
> The EDA step exists to fix the analysis unit (a daily network series, one
> row per date) and to validate assumptions (missing days, closures,
> distribution of sales) before any decomposition runs. This is the
> "data before conclusions" golden standard.

## Methodology

All formulas below are implemented as tested `src/` functions
(`src/smoothing.py`, `src/decomposition.py`).

**Simple moving average (MA).** A window of `w` trailing days; the windows
7 / 14 / 30 probe the trend at daily, weekly and ~monthly horizons:

$$\text{MA}_w(y_t) = \frac{1}{w} \sum_{i=0}^{w-1} y_{t-i}$$

![Daily network sales smoothed: MA7 / MA14 / MA30](assets/smoothing_ma.png)

**Exponential moving average (EMA).** The same horizon as an MA of span `s`,
but recent observations weigh more — `alpha = 2 / (s+1)`:

$$\text{EMA}_t = \alpha\, y_t + (1-\alpha)\, \text{EMA}_{t-1}, \qquad \alpha = \frac{2}{s+1}$$

The two smooth curves answer the trend question consistently (notebooks 02
and 04): the level climbs strongly from 2013 through late 2016 and eases into
2017. MA vs EMA matter because their *lag profiles* differ at the same order
— MA cuts off hard after the window, EMA never fully forgets — which the
annual peaks expose:

![MA7 vs EMA7 on 2017-01-01 .. 2017-08-15: same order, different lag profile](assets/smoothing_ma_vs_ema.png)

**STL decomposition.** `statsmodels` STL with `period=7` (the weekly rhythm
measured in stage 1) and robust inner/outer loops, additive by construction:

$$y_t = T_t + S_t + R_t$$

**Seasonality and trend strength** (Wang, Smith & Hyndman, 2006): the share
of variance that each component explains over what the trend leaves behind:

$$F_s = \max\!\left(0,\ 1 - \frac{\mathrm{Var}(R_t)}{\mathrm{Var}(R_t + S_t)}\right), \qquad
F_t = \max\!\left(0,\ 1 - \frac{\mathrm{Var}(R_t)}{\mathrm{Var}(R_t + T_t)}\right)$$

**Additive vs multiplicative.** A multiplicative seasonal model is
approximated as STL on `log(y)` with components exp-mapped back, and the two
are compared by `resid_relative_std = resid.std() / series.mean()` (the same
scale-free residual metric on both branches). Decision rule (documented in
`src/decomposition.py`): pick the smaller `resid_relative_std`.

## Results

| Finding | Value | Evidence |
| ------- | ----- | -------- |
| Trend strength F_t | 0.793 | `seasonality_strength`/`trend_strength`, notebook 04 |
| Trend yearly means 2013 / 2017 | 385,121 / 846,530 | notebook 04 |
| Trend component peak | 1,010,730 around 2016-12 (Dec mean 931,604) | notebook 04 |
| Weekly rhythm (raw means) | Sunday 821,794 vs Thursday 503,173 (amplitude ~318,621) | notebook 04 |
| Seasonal band (STL) | Sunday +215,863 units above the weekday midpoint; Thursday −136,391 | notebook 04 |
| Seasonality strength F_s | 0.570 (network) | notebook 04 |
| National holidays at top-15 residual days | 14 of 15 | notebook 04 |
| Largest holiday effects | Navidad 2016 −1,321,021; New Year +1,262,485 (2017-01-01) | notebook 04 |
| Store 44 (type A, heavy) | F_s 0.633, F_t 0.711 | notebook 03 |
| Store 26 (type D/E, small) | F_s 0.171, F_t 0.306 (weak weekly seasonality) | notebook 03 |
| Model choice | multiplicative, but by only 0.94% (RRS 0.1311 vs 0.1323) — near-tie | notebook 04 §4 |

![Mean network sales by day of week — the evidence for period=7](assets/eda_weekday_profile.png)

The one day the series cannot explain by a holiday is **2017-04-01**
(1,463,084, the largest day in the whole range). Notebook 04 states it
honestly as a fact without a causal story rather than inventing one.

![The 15 largest STL residuals against national holidays](assets/seasonal_peaks_holidays.png)

![Additive vs multiplicative-proxy residual spread — the near-tie](assets/additive_vs_multiplicative.png)

> [!IMPORTANT]
> The model-choice row is a **near-tie, not a verdict**. No test was run on
> the 0.94% gap; both fits tell the same monthly-report story. "Multiplicative
> was picked" describes the documented decision rule, not a claim that one
> model is meaningfully better.

**Verdict.** The network grew strongly from 2013 through late 2016 and eased
into 2017; sales keep a stable weekly rhythm peaking on Sunday; the calendar
outliers are national holidays (14 of the 15 largest residual days), led by
the Christmas closures and New Year bursts. The full verdict is quoted
verbatim in notebook 04 and `reports/04_executive_summary.html`; it is
descriptive and exploratory, and it does not forecast or claim significance.

## Learning Resources

This repository is a teaching piece, and every analysis decision is
explained from first principles. Explanations live in `notes/` (in Russian,
with the English key term at every first mention):

| Note | Stage | Why it matters |
| ---- | ----- | -------------- |
| `notes/01_data_integrity_eda.md` | Integrity checks & EDA | data before conclusions; the analysis unit; closures |
| `notes/02_smoothing.md` | MA / EMA | why smoothing is needed, window vs span, lag profiles |
| `notes/03_stl_decomposition.md` | STL + holidays | trend vs seasonal vs residual; strengths F_s / F_t; holiday mapping |
| `notes/04_summary.md` | Executive summary | number-must-be-code discipline; honest near-tie reporting |
| `notes/05_readme_and_dod.md` | README & definition of done | README-vs-reality drift; verdict consistency; reproducible Quick Start |

The same "why it matters" framing is applied inside the code: every public
`src/` function carries a docstring covering purpose, parameters, return
value and the edge cases it guards.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

56 tests (18 EDA + 18 smoothing + 20 decomposition) cover the reusable
functions, including cross-checks against reference implementations:

- `tests/test_eda.py` — load contract dtypes, integrity assertions
  (duplicates, missing values, negative sales, calendar gaps), daily-network
  and store-level aggregation, STL store selection, promotion share.
- `tests/test_smoothing.py` — MA/EMA hand-computed cases, equivalence against
  `pandas` reference objects, warm-up behaviour, invalid-input rejection.
- `tests/test_decomposition.py` — STL component reconstruction and
  cross-validation against `statsmodels`, strength metrics on synthetic
  series, additive-vs-multiplicative decision rule, national-holiday filter.

## Limitations

- **One retailer, one series.** Findings describe Corporación Favorita's
  network in 2013–2017; they do not generalize to other retailers or years.
- **No forecasting.** STL is a descriptive decomposition of the past. The
  late-2016 trend peak followed by easing into 2017 describes the data; it is
  not evidence that future sales will continue easing. A forecast would
  require a separate, pre-registered modeling stage.
- **No statistical significance.** No test is run on any number in the
  report — the verdict is descriptive. In particular the 0.94%
  additive-vs-multiplicative gap is reported as a near-tie, not tested as a
  difference.
- **No store/region drill-down.** The scope is network-level description.
  Store-level analysis appears only as two sanity-check decompositions (store
  44 and store 26) whose only job is to show that the weekly pattern is not
  an artifact of aggregation.
- **Holiday weekends are not modeled.** National holidays explain the biggest
  one-off residual spikes, but the STL weekly band cannot absorb calendar
  dates — the holiday *effects* are visual explanations, not fitted effects.
- **The one unexplained day (2017-04-01)** is the largest day in the whole
  range and matches no national holiday. It is reported as a fact without a
  causal story.

## Recommendations / Next Steps

- **Operations:** capacity and staffing planning should lean on the stable
  Sunday peak (roughly 1.6× the Thursday trough) rather than on the one-off
  holiday spikes, which are one-day events.
- **Calendar awareness:** the Christmas closures (25 Dec 2013–2016) and the
  New Year burst are the two largest reproducible effects; a promotional or
  inventory calendar should treat them as distinct event types.
- **Store-level drill-down:** extend the same decomposition pipeline to store
  groups to see which types (A vs D/E) actually carry the network rhythm —
  the store 26 result suggests small daily-attended stores may not.
- **Forecasting stage:** a separate, pre-registered forecasting notebook
  (ARIMA-family or equivalent) that starts from the trend/seasonal structure
  mapped here and is evaluated by proper backtesting — kept out of this
  descriptive scope on purpose.
- **External regressors:** revisit `oil.csv` and `transactions.csv` as
  candidate regressors once a forecasting stage exists; they were consciously
  excluded from the descriptive scope.

## Support

- Found a bug or have a question: open a GitHub issue.
- General discussion about the methodology: GitHub Discussions.

## Contributing

Contributions are welcome. Please open an issue first to discuss the intended
change, keep the `src/` functions tested, and regenerate the notebooks with
the `build_*.py` scripts before submitting a pull request.

## License

[MIT](LICENSE)

## Acknowledgements

- Dataset: Store Sales — Time Series Forecasting, a Kaggle competition by
  Corporación Favorita.
  https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- STL decomposition and reference cross-validation: `statsmodels`
  (Cleveland et al. 1990, seasonal-trend decomposition by Loess).
- Seasonality/trend strength metrics: Wang, Smith & Hyndman (2006), "Characteristic-based clustering for time series data".
- Notebook rebuilds: `nbformat` / `nbclient`.
- README structure follows the "Ultimate README Guide" conventions used across
  this portfolio: badges that match reality, honest limitation sections, and
  figures that answer the question they illustrate.