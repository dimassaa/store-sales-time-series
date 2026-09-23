# Data

Source dataset: **Store Sales — Time Series Forecasting** (Kaggle competition,
Corporación Favorita).

- Files (all kept out of git, `data/*.csv` is git-ignored):
  - `train.csv` (~122 MB, ~3.0M rows) — daily unit sales per store and product
    family. Schema: `id`, `date`, `store_nbr`, `family`, `sales`,
    `onpromotion`.
  - `stores.csv` — 54 stores with `city`, `state`, `type` (A/B/C/D/E),
    `cluster`.
  - `holidays_events.csv` — national/regional/local holidays and transfers.
  - `transactions.csv` — daily transaction counts per store.
  - `oil.csv` — daily oil price (`dcoilwtico`), used as an economic indicator.
  - `test (1).csv`, `sample_submission (1).csv` — competition forecasting
    artifacts, **not used** in this descriptive project.
- Only `train.csv`, `stores.csv` and `holidays_events.csv` are used (see
  `plan.md`, fixed decisions).
- Date range used: 2013-01-01 .. 2017-08-15 (train split).

Original page:
`https://www.kaggle.com/competitions/store-sales-time-series-forecasting`

Re-download: create a Kaggle account, join the competition page, use the
"Data" tab download button (the files are otherwise not downloadable without
the terms-of-participation acceptance).