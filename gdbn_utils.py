"""Shared helpers for the GDBN notebooks: data loading, lagged pairs, baselines, metrics."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
RESULTS = DATA / "results"
SECTOR_ORDER = ["Tech", "Banks", "Energy", "Staples", "Healthcare", "Industrials",
                "Market", "Commodities/Rates"]
Z90 = 1.6449  # 90% two-sided interval for a normal distribution


def load_data():
    """Return (wide, split, info). wide: one row per trading day, 95 variables."""
    wide = pd.read_csv(DATA / "gdbn_wide.csv", index_col="date", parse_dates=True).sort_index()
    split = pd.read_csv(DATA / "splits" / "split_labels.csv", index_col="date", parse_dates=True)["split"]
    universe = pd.read_csv(DATA / "universe.csv").drop_duplicates("ticker")
    sector = dict(zip(universe["ticker"], universe["sector"]))

    # Order columns by sector, then type, so plots group nicely
    tickers = sorted(sector, key=lambda t: (SECTOR_ORDER.index(sector[t]), list(sector).index(t)))
    ret = [f"{t}_ret" for t in tickers]
    sent = [f"{t}_sent" for t in tickers if f"{t}_sent" in wide]
    attn = [f"{t}_attn" for t in tickers if f"{t}_attn" in wide]
    wide = wide[ret + sent + attn]

    info = {
        "RET": ret, "SENT": sent, "ATTN": attn,
        "STOCKS": [c[:-5] for c in sent],
        "SECTOR": sector,
        "ticker": lambda col: col.rsplit("_", 1)[0],
        "kind": lambda col: col.rsplit("_", 1)[1],
    }
    return wide, split, info


def make_pairs(wide, split):
    """Yesterday -> today pairs, built on the full series, labeled by the split of 'today'.

    Returns X (variables at t-1, indexed by t), Y (variables at t), and the split label of t.
    """
    X = wide.shift(1).iloc[1:]
    Y = wide.iloc[1:]
    return X, Y, split.loc[Y.index]


def standardizer(train_df):
    """Means and standard deviations from the TRAINING data only."""
    mu, sd = train_df.mean(), train_df.std().replace(0, 1.0)
    return mu, sd


# ----------------------------------------------------------------------------
# Baselines (predict next-day returns)
# ----------------------------------------------------------------------------
def baseline_predictions(X, Y, labels, ret_cols):
    """Predictions for each return series from three simple baselines.

    zero      tomorrow's return is 0
    hist_mean the series' average daily return in training
    ar1       each series from only its own previous day (OLS fit on training)
    Also returns sigma: each baseline's training residual standard deviation (for intervals).
    """
    tr = labels == "train"
    preds, sigmas = {}, {}

    preds["zero"] = pd.DataFrame(0.0, index=Y.index, columns=ret_cols)
    sigmas["zero"] = np.sqrt((Y.loc[tr, ret_cols] ** 2).mean())

    mu = Y.loc[tr, ret_cols].mean()
    preds["hist_mean"] = pd.DataFrame(np.tile(mu.values, (len(Y), 1)), index=Y.index, columns=ret_cols)
    sigmas["hist_mean"] = Y.loc[tr, ret_cols].std()

    ar = {}
    ar_sigma = {}
    for c in ret_cols:
        x, y = X.loc[tr, c].values, Y.loc[tr, c].values
        A = np.column_stack([np.ones_like(x), x])
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        ar[c] = a + b * X[c]
        ar_sigma[c] = np.std(y - (a + b * x), ddof=2)
    preds["ar1"] = pd.DataFrame(ar)[ret_cols]
    sigmas["ar1"] = pd.Series(ar_sigma)
    return preds, sigmas


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred, sigma=None, benchmark_mse=None):
    """Per-series metrics.

    mse          mean squared error
    r2_os        out-of-sample R^2 vs the benchmark (historical mean): 1 - MSE / MSE_benchmark
                 (positive = beats the historical mean)
    direction    share of days where the predicted sign matches the actual sign
                 (a prediction of exactly 0 counts as 'up')
    coverage90   share of days the actual value falls inside prediction +/- 1.645 sigma
    """
    err = y_true - y_pred
    out = pd.DataFrame({"mse": (err ** 2).mean()})
    if benchmark_mse is not None:
        out["r2_os"] = 1 - out["mse"] / benchmark_mse
    out["direction"] = ((y_pred >= 0) == (y_true >= 0)).mean()
    if sigma is not None:
        out["coverage90"] = (err.abs() <= Z90 * sigma).mean()
    return out


def summarize(per_series):
    """Average the per-series metrics into one row."""
    return per_series.mean()


def dm_test(err_a, err_b, lags=5):
    """Diebold-Mariano test: is model A's squared error different from model B's?

    err_a, err_b: DataFrames of forecast errors (days x series). The loss difference is
    averaged across series each day, then a HAC (Newey-West) t-test is run on the daily series.
    Returns (mean loss difference, t-statistic, two-sided p-value). Negative mean = A is better.
    Returns (0, nan, nan) if the two forecasts are identical.
    """
    from scipy.stats import norm

    d = ((err_a ** 2) - (err_b ** 2)).mean(axis=1).dropna().values
    if np.allclose(d, 0, atol=1e-12):
        return 0.0, np.nan, np.nan  # identical forecasts: nothing to test
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (dc[k:] @ dc[:-k]) / n
    t = d.mean() / np.sqrt(var / n)
    return d.mean(), t, 2 * (1 - norm.cdf(abs(t)))


def dm_report(name_a, name_b, err_a, err_b):
    """One readable line for a Diebold-Mariano comparison."""
    diff, t, p = dm_test(err_a, err_b)
    if np.isnan(t):
        return f"{name_a:>11} vs {name_b}: identical forecasts (no difference to test)"
    better = name_a if diff < 0 else name_b
    verdict = f"{better} is better, significant (p < 0.05)" if p < 0.05 else "difference not significant"
    return f"{name_a:>11} vs {name_b}: mean loss diff = {diff:+.2e}, t = {t:+.2f}, p = {p:.3f} -> {verdict}"


# ----------------------------------------------------------------------------
# Volatility version
# ----------------------------------------------------------------------------
VOL_FLOOR = 1e-4  # avoids log(0) on days with a return of exactly 0


def volatility_dataset(wide, info, floor=VOL_FLOOR):
    """Replace each return with daily volatility: vol = log(|return| + floor).

    Returns (data, VOL): data has 95 variables (35 vol + 30 sent + 30 attn); VOL lists the vol columns.
    The log makes volatility roughly bell-shaped, which suits a Gaussian model.
    """
    vol = np.log(wide[info["RET"]].abs() + floor)
    vol.columns = [c.replace("_ret", "_vol") for c in info["RET"]]
    data = pd.concat([vol, wide[info["SENT"]], wide[info["ATTN"]]], axis=1)
    return data, list(vol.columns)


def har_features(data, vol_cols):
    """HAR terms for each volatility series: its average over the past 5 days (week) and 22 days (month).

    Values on day t use only days up to t (shift them by one day before using as predictors).
    """
    week = data[vol_cols].rolling(5).mean().add_suffix("_week")
    month = data[vol_cols].rolling(22).mean().add_suffix("_month")
    return pd.concat([week, month], axis=1)
