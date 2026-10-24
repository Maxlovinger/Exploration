"""Shared code for the point-in-time walk-forward notebooks (09 multi-day target, 10 slow inputs).

Same setup as notebook 06: point-in-time universe (5 largest per sector each quarter), the GDBN
re-learned every quarter on the past 3 years (one lasso per stock), one shared λ chosen by rank IC
on 2006-2020, validation 2021-2022. Everything after EVAL_END is blanked out on load, so the test
years cannot be used by accident.
"""

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

from gdbn_utils import DATA

BUILT = DATA / "pit" / "built"
WINDOW = 756                          # training window: 3 years of trading days
LOOKBACK = 30                         # extra days before the window for lags and 22-day averages
TOP_FRAC = 0.20                       # long the top 20%, short the bottom 20%
COST_BPS = 5                          # trading cost per unit traded (bps)
ALPHAS = np.logspace(-0.7, -3, 20)    # shared λ grid (all inputs are z-scores), largest first
TUNE_END, EVAL_END = "2020-12-31", "2022-12-31"
ETFS = ["SPY", "USO", "GLD", "DBC", "TLT"]
PERIODS = {"2006–2020 walk-forward": (None, TUNE_END), "2021–2022 validation": ("2021-01-01", EVAL_END)}


def load(sent_kind="news"):
    """Point-in-time data, with every value after EVAL_END set to missing."""
    days = pd.DatetimeIndex(pd.read_csv(BUILT / "trading_days.csv", parse_dates=["date"])["date"])
    panel = pd.read_csv(BUILT / "panel.csv.gz", parse_dates=["date"],
                        usecols=["date", "permno", "ret", f"n_{sent_kind}", f"sent_{sent_kind}"])
    def blank_late(df):                                              # nothing after EVAL_END
        df = df.copy()
        df.loc[df.index > pd.Timestamp(EVAL_END)] = np.nan
        return df

    wide = lambda col: blank_late(panel.pivot(index="date", columns="permno", values=col).reindex(days))
    sel = pd.read_csv(BUILT / "universe.csv", parse_dates=["rebalance_date"]).query("selected")
    rebal = [pd.Timestamp(r) for r in sorted(sel["rebalance_date"].unique()) if r <= pd.Timestamp(EVAL_END)]
    etf = pd.read_csv(BUILT / "etf_returns.csv", index_col="date", parse_dates=True).reindex(days)[ETFS]
    return SimpleNamespace(
        days=days, RET=wide("ret"), SENT=wide(f"sent_{sent_kind}"), ATTN=np.log1p(wide(f"n_{sent_kind}")),
        ETF=blank_late(etf), rebal=rebal,
        U={r: list(g["permno"]) for r, g in sel.groupby("rebalance_date") if r in rebal})


# ----------------------------------------------------------------------------
# Inputs (values known at the close of day t; the walk-forward shifts them by one day)
# ----------------------------------------------------------------------------
def daily_features(D, span, U, EX):
    """Yesterday's values: excess return, sentiment, attention, ETF returns (as in notebook 06)."""
    return pd.concat([EX.add_suffix("|ret"), D.SENT.reindex(index=span, columns=U).add_suffix("|sent"),
                      D.ATTN.reindex(index=span, columns=U).add_suffix("|attn"), D.ETF.reindex(span)], axis=1)


def slow_features(D, span, U, EX):
    """Past-week (5-day) and past-month (22-day) averages of the same variables."""
    S, A, E = D.SENT.reindex(index=span, columns=U), D.ATTN.reindex(index=span, columns=U), D.ETF.reindex(span)
    parts = []
    for n, tag in [(5, "w"), (22, "m")]:
        avg = lambda df: df.rolling(n, min_periods=max(n // 2, 2)).mean()
        parts += [avg(EX).add_suffix(f"|ret_{tag}"), avg(S).add_suffix(f"|sent_{tag}"),
                  avg(A).add_suffix(f"|attn_{tag}"), avg(E).add_suffix(f"|{tag}")]
    return pd.concat(parts, axis=1)


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------
def walk_forward(D, features, horizon=1, label=""):
    """Each quarter: learn on the past WINDOW days, predict every day of the quarter, for every λ.

    Target for day t = the stock's excess return summed over days t ... t + horizon - 1.
    Training targets must END before the rebalance day (the last horizon-1 window days are dropped),
    so no training target overlaps the quarter being predicted.
    """
    out = []
    t0 = time.time()
    if label:
        print(f"{label}: started")
    for j, r in enumerate(D.rebal):
        nxt = D.rebal[j + 1] if j + 1 < len(D.rebal) else pd.Timestamp(EVAL_END) + pd.Timedelta(days=1)
        pos = D.days.searchsorted(r)
        if pos - WINDOW - LOOKBACK < 0:
            continue
        hold = D.days[pos: D.days.searchsorted(nxt)]
        hold = hold[hold <= pd.Timestamp(EVAL_END)]
        if len(hold) == 0:
            continue
        win = D.days[pos - WINDOW: pos - (horizon - 1)]
        span = D.days[pos - WINDOW - LOOKBACK: pos + len(hold)]
        U = D.U[r]
        R = D.RET.reindex(index=span, columns=U)
        EX = R.sub(R.mean(axis=1), axis=0)
        X = features(D, span, U, EX).shift(1)                       # information through day t-1
        Y = EX.rolling(horizon).sum().shift(-(horizon - 1)) if horizon > 1 else EX
        Xw, Xh = X.loc[win], X.loc[hold]
        mu, sd = Xw.mean(), Xw.std().replace(0, np.nan)
        Xwz, Xhz = ((Xw - mu) / sd).fillna(0).values, ((Xh - mu) / sd).fillna(0).values
        xm = Xwz.mean(axis=0)
        S = np.full((len(ALPHAS), len(hold), len(U)), np.nan)
        W = np.zeros((len(U), X.shape[1], len(ALPHAS)))
        for i, p in enumerate(U):
            y = Y.loc[win, p].values
            ok = ~np.isnan(y)
            if ok.sum() < 252:
                continue
            ys = y[ok].std()
            _, coefs, _ = lasso_path(Xwz[ok] - xm, (y[ok] - y[ok].mean()) / ys, alphas=ALPHAS)
            W[i] = coefs
            S[:, :, i] = ((Xhz - xm) @ coefs).T * ys
        out.append({"r": r, "hold": hold, "U": U, "S": S, "W": W, "cols": list(X.columns)})
        if label and (j + 1) % 10 == 0:
            print(f"  {label}: {j + 1} of {len(D.rebal)} quarters ({time.time() - t0:.0f}s)")
    if label:
        print(f"{label}: done in {time.time() - t0:.0f}s")
    return out


def scores(results, k):
    """date x permno score table at λ index k (NaN = not in that day's universe)."""
    return pd.concat([pd.DataFrame(q["S"][k], index=q["hold"], columns=q["U"]) for q in results]).sort_index()


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def forward_excess(D, score, h):
    """Each stock's excess return (vs that day's universe) summed over the next h days, starting the score's day."""
    total = None
    for k in range(h):
        fut = D.RET.shift(-k).reindex(index=score.index, columns=score.columns).where(score.notna())
        e = fut.sub(fut.mean(axis=1), axis=0)
        total = e if total is None else total + e
    return total


def rank_ic(score, target):
    ok = score.notna() & target.notna()
    ic = score.where(ok).rank(axis=1).corrwith(target.where(ok).rank(axis=1), axis=1)
    return ic[ok.sum(axis=1) >= 10]


def nw_t(x, lags=5):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n, xc = len(x), x - x.mean()
    var = xc @ xc / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (xc[k:] @ xc[:-k]) / n
    return x.mean() / np.sqrt(var / n) if var > 0 else np.nan


def choose_lambda(D, results, h):
    """λ index with the best average rank IC (vs the h-day forward excess return) over 2006-2020."""
    target = forward_excess(D, scores(results, 0), h)
    ics = [rank_ic(scores(results, k).loc[:TUNE_END], target.loc[:TUNE_END]).mean() for k in range(len(ALPHAS))]
    return int(np.nanargmax(ics)), np.array(ics)


def daily_portfolio(score):
    """+1/n on the top 20%, -1/n on the bottom 20% of each day's universe (random tie-break)."""
    ok = score.notna()
    rk = (score.rank(axis=1) + np.random.default_rng(1).uniform(0, 1e-9, score.shape)).where(ok).rank(axis=1)
    n = ok.sum(axis=1)
    top = np.maximum(np.floor(n * TOP_FRAC), 1)
    return (rk.gt(n - top, axis=0).astype(float).div(top, axis=0)
            - rk.le(top, axis=0).astype(float).div(top, axis=0)).where(ok, 0.0)


def held_weights(score, h):
    """Overlapping portfolios: each day's top/bottom 20% is held for h days; 1/h of the money in each."""
    return daily_portfolio(score).fillna(0.0).rolling(h, min_periods=1).sum() / h


def summary(D, w, start, end, lags=5):
    w = w.loc[start:end]
    r = D.RET.reindex(index=w.index, columns=w.columns).fillna(0.0)
    gross = (w * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    net = gross - turnover * COST_BPS / 1e4
    return pd.Series({"bps/day": gross.mean() * 1e4, "t": nw_t(gross, lags),
                      "Sharpe gross": gross.mean() / gross.std() * np.sqrt(252),
                      "Sharpe net": net.mean() / net.std() * np.sqrt(252),
                      "turnover/day": turnover.mean(), "breakeven bps": gross.mean() / turnover.mean() * 1e4})


def decay(D, score, start, end, lags=range(1, 11)):
    """Day-k IC (score vs the excess return k days later) and ranking stability (score vs the score k days later)."""
    S = score.loc[start:end]
    rows = {}
    for k in lags:
        fut = D.RET.shift(-(k - 1)).reindex(index=S.index, columns=S.columns).where(S.notna())
        ick = rank_ic(S, fut.sub(fut.mean(axis=1), axis=0))
        later = score.shift(-(k - 1)).reindex(index=S.index).where(S.notna())
        rows[k] = {"day-k IC": ick.mean(), "day-k SE": ick.std() / np.sqrt(len(ick)),
                   "ranking stability": rank_ic(S, later).mean() if k > 1 else 1.0}
    return pd.DataFrame(rows).T
