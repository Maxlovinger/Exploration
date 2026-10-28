# Exploration

Research code for my thesis: can a **Gaussian Dynamic Bayesian Network (GDBN)** learn how S&P 500 stocks affect each other from one day to the next, and can those links predict returns, volatility, or rankings?

Each stock's value today is modeled from every stock's value yesterday:

x_i(t) = b_i + Σ_j w_ij · x_j(t−1) + ε_i

A lasso sets most w_ij to zero, so the nonzero weights form the network (an arrow j → i means yesterday's j helps predict today's i).

## Data

All data comes from WRDS: CRSP (prices, returns, S&P 500 membership), Compustat (fundamentals), RavenPack (news sentiment and attention), ExecuComp (CEOs), OptionMetrics (implied volatility), I/B/E/S (analysts), Fama-French factors, and others.

**No data is in this repo.** WRDS data is licensed for academic use and can't be redistributed. Everything in `data/` is rebuilt by the scripts below.

Every feature is **point-in-time**: a value is only used from the day it became public (e.g., earnings from their report date, news after the 4 PM close counts for the next day). The universe is rebuilt each quarter (the 5 largest S&P 500 members in each sector), so there is no survivorship bias.

## Pipeline

1. **Explore** - `explore_*.py` print which WRDS tables and columns your account can access (names and date ranges only, no data rows)
2. **Pull** - `pull_pit.py`, `pull_pit_fallback.py`, `pull_fundamentals.py`, `pull_features.py`, `pull_extra.py` download raw data
3. **Build** - `build_pit.py` (point-in-time universe and daily panel), `build_fundamentals.py`, `build_features.py` (stock-day feature table), `build_extra_features.py` (options, analysts, short interest, etc.)
4. **Model** - the notebooks

Shared helpers: `gdbn_utils.py`, `pit_common.py`, `denoise_common.py`, `rp_news.py` (RavenPack daily aggregates), `name_match.py` (checks two company names are the same company).

## Notebooks

| # | Notebook | Question |
|---|---|---|
| 00 | `gdbn_walkthrough` | How the GDBN is built, learned, and used (toy examples, then 10 real stocks) |
| 01 | `sanity_check` | Does the dataset look right? |
| 02 | `baselines` | The bar to beat: zero, historical mean, AR(1) |
| 03 | `gdbn_lasso` | GDBN v1: can it predict next-day returns? |
| 04 | `gdbn_volatility` | GDBN v2: volatility, which persists and spreads |
| 05 | `rankings` | GDBN v3: which stocks beat the others tomorrow (long-short) |
| 06 | `pit_gdbn` | v4: point-in-time universe, walk-forward rankings |
| 07 | `pit_volatility` | v5: point-in-time volatility vs HAR |
| 08 | `pit_bayesian_returns` | v6: Bayesian confidence and trading costs |
| 09 | `pit_multiday_target` | v7: 5- and 22-day targets |
| 10 | `pit_slow_inputs` | v8: weekly and monthly inputs to cut turnover |
| 11 | `pit_fundamentals` | v9: do fundamentals and CEO data help? |
| 12 | `pit_weekly` | v10: fully weekly model |
| 13 | `factorvae` | FactorVAE (Duan et al., 2022) on the same data |
| 14-17 | `denoise_*` | Clean the data first (PCA, IPCA, conditional autoencoder, denoising autoencoder), then fit the GDBN |
| 18 | `denoise_comparison` | Which denoiser helps most? |
| 19 | `feature_blocks` | Which data sources actually help predict next-day returns? |

`archive_2015_2024/` has the first version, which used a fixed 2015-2024 universe.

## Splits

Split by date, never shuffled:

- **Train:** 2006 - 2020 (includes 2008 and 2020)
- **Validation:** 2021 - 2022 (one bull year, one bear year)
- **Test:** 2023 onward, used once at the end

## Run it

You need a WRDS account. The scripts expect to run from the thesis folder:

```bash
cd ~/Documents/Thesis
.venv/bin/python CoreFragment/explore_wrds.py   # saves your WRDS login to .pgpass
.venv/bin/python CoreFragment/pull_pit.py
.venv/bin/python CoreFragment/build_pit.py
# ...then the other pull_/build_ scripts, then the notebooks in order
```
