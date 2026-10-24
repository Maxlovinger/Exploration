"""Shared code for the DENOISING notebooks: 14 PCA, 15 IPCA, 16 conditional autoencoder, 17 denoising autoencoder,
and 18 (comparison).

Idea: clean the data first, then let the GDBN find the connections.
  - 14-16 split each day's excess returns into a SHARED part (a few market-wide factors) and a STOCK-SPECIFIC part
    (the residual). The GDBN then uses yesterday's residuals (plus the factors as market nodes) instead of raw returns.
  - 17 cleans each stock's FEATURE vector (returns, sentiment, attention, ...) through an autoencoder's waist.

All four start from the same feature table (features_daily.csv.gz, built by build_features.py).

No look-ahead:
  - The denoiser used for year Y is fit ONLY on days before Y (up to 5 years).
  - The GDBN's quarterly refits in year Y use residuals computed with that same denoiser (its training-window days
    are in-sample for the denoiser, but all of them are before the quarter being predicted).
  - Splitting day t's returns into factor part + residual only needs day t's returns (known at day t's close),
    and characteristics from day t-1.
  - Test years (2023+) are never built.
"""

import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pit_common as pc
from gdbn_utils import RESULTS

FEATS = ["exret_1d", "exret_5d", "exret_20d", "mom_12_1", "overnight", "intraday",
         "vol_1d", "range", "rvol_20", "beta_60", "ivol_60",
         "volume", "turnover_20", "amihud_20", "size",
         "sentiment", "has_news", "attention", "surprise",
         "days_since_earn", "sue", "bm", "gp", "lev", "ag"]
YEARS = range(2006, 2023)
FIT_YEARS = 5                 # denoiser for year Y is fit on up to the 5 years before Y
APPLY_BACK_YEARS = 4          # the GDBN's windows in year Y reach back about 3.2 years
OUT = Path(os.environ.get("DENOISE_OUT", RESULTS / "denoise"))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_cube():
    """The feature table as arrays: X[day, stock, feature], R = today's excess return, AVAIL / MEM masks."""
    t0 = time.time()
    df = pd.read_csv(pc.BUILT / "features_daily.csv.gz", usecols=["date", "permno", "in_sp500"] + FEATS,
                     parse_dates=["date"], dtype={f: "float32" for f in FEATS})
    df = df[df["date"] <= pc.EVAL_END]
    days = pd.DatetimeIndex(np.sort(df["date"].unique()))
    perm = np.sort(df["permno"].unique())
    di, pj = days.get_indexer(df["date"]), pd.Index(perm).get_indexer(df["permno"])
    X = np.full((len(days), len(perm), len(FEATS)), np.nan, dtype=np.float32)
    X[di, pj] = df[FEATS].values
    MEM = np.zeros((len(days), len(perm)), dtype=bool)
    MEM[di, pj] = df["in_sp500"].values > 0
    R = X[:, :, FEATS.index("exret_1d")].copy()
    AVAIL = np.isfinite(R)                      # stock has a return that day (members, and past/future members)
    print(f"Feature table: {len(days):,} days x {len(perm)} stocks x {len(FEATS)} features "
          f"({days[0].date()} to {days[-1].date()}; {time.time() - t0:.0f}s)")
    return SimpleNamespace(X=X, R=R, MEM=MEM, AVAIL=AVAIL, days=days, perm=perm)


def fit_days(cube, year):
    """Day indices used to FIT the denoiser for `year`: up to FIT_YEARS before it (past only)."""
    a = cube.days.searchsorted(pd.Timestamp(f"{year - FIT_YEARS}-01-01"))
    b = cube.days.searchsorted(pd.Timestamp(f"{year}-01-01"))
    return np.arange(max(a, 1), b)


def apply_days(cube, year):
    """Day indices the denoiser for `year` is APPLIED to: the GDBN's training windows in `year`, plus the year itself."""
    a = cube.days.searchsorted(pd.Timestamp(f"{year - APPLY_BACK_YEARS}-01-01"))
    b = cube.days.searchsorted(pd.Timestamp(f"{year + 1}-01-01"))
    return np.arange(max(a, 1), b)


def year_days(cube, year):
    """Day indices inside `year` (out-of-sample for that year's denoiser)."""
    return np.where(cube.days.year == year)[0]


def rank_chars(cube):
    """Characteristics ranked across stocks each day, scaled to [-0.5, 0.5] (0 = median or missing), as in
    Kelly-Pruitt-Su (2019) and Gu-Kelly-Xiu (2021). Ranked among stocks with a return that day."""
    t0 = time.time()
    Z = np.zeros_like(cube.X)
    for d in range(len(cube.days)):
        m = cube.AVAIL[d]
        if m.sum() < 10:
            continue
        r = pd.DataFrame(cube.X[d, m]).rank(pct=True).values - 0.5
        Z[d, m] = np.nan_to_num(r, nan=0.0)
    print(f"Ranked characteristics ({time.time() - t0:.0f}s)")
    return Z


def r2(actual, fitted):
    """Total R^2 = 1 - sum(residual^2) / sum(actual^2) (returns are excess returns, so no demeaning)."""
    ok = np.isfinite(actual) & np.isfinite(fitted)
    return 1 - ((actual[ok] - fitted[ok]) ** 2).sum() / (actual[ok] ** 2).sum()


def gdbn_mask(D, cube):
    """day x stock mask of the GDBN's 52-stock point-in-time universe (for R^2 on the GDBN's own stocks)."""
    mask = np.zeros(cube.R.shape, dtype=bool)
    col = pd.Index(cube.perm)
    for j, r in enumerate(D.rebal):
        nxt = D.rebal[j + 1] if j + 1 < len(D.rebal) else pd.Timestamp(pc.EVAL_END) + pd.Timedelta(days=1)
        rows = np.where((cube.days >= r) & (cube.days < nxt))[0]
        cols = col.get_indexer([p for p in D.U[r] if p in col])
        mask[np.ix_(rows, cols)] = True
    return mask


# ----------------------------------------------------------------------------
# GDBN on denoised data
# ----------------------------------------------------------------------------
def gdbn_features(by_year, kind="daily"):
    """Input function for pc.walk_forward. by_year[Y] holds the denoised series the GDBN uses in year Y:
    RET (date x permno, replaces the excess return), optional SENT / ATTN (replace sentiment / attention),
    optional FACT (date x K factor returns, added as market-wide nodes like the ETFs)."""
    def features(D, span, U, EX):
        V = by_year[span[-1].year]                              # every quarter's days are in one year
        Dv = SimpleNamespace(SENT=V.SENT if V.SENT is not None else D.SENT,
                             ATTN=V.ATTN if V.ATTN is not None else D.ATTN, ETF=D.ETF)
        ret = V.RET.reindex(index=span, columns=U)
        X = (pc.daily_features if kind == "daily" else pc.slow_features)(Dv, span, U, ret)
        if V.FACT is not None:
            F = V.FACT.reindex(span)
            if kind == "slow":
                F = pd.concat([F.rolling(5, min_periods=2).mean().add_suffix("|w"),
                               F.rolling(22, min_periods=11).mean().add_suffix("|m")], axis=1)
            X = pd.concat([X, F], axis=1)
        return X
    return features


def walk_forward(name, D, features, use_cache=False):
    """pc.walk_forward (1-day target), saved to OUT so notebook 18 can compare all versions.
    use_cache=True loads a saved run (used for the raw GDBN, which is the same in every notebook)."""
    path = OUT / f"wf_{name}.pkl"
    if use_cache and path.exists():
        print(f"{name}: loaded saved run ({path.name})")
        return pickle.load(open(path, "rb"))
    res = pc.walk_forward(D, features, 1, label=name)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


def raw_gdbn(D):
    """The two reference GDBNs: daily inputs (notebook 06) and slow inputs (notebook 10)."""
    return {"raw daily": walk_forward("raw_daily", D, pc.daily_features, use_cache=True),
            "raw slow": walk_forward("raw_slow", D, pc.slow_features, use_cache=True)}


def edges(q, k):
    W = q["W"][:, :, k]
    return {(q["U"][i], q["cols"][j]) for i, j in zip(*np.nonzero(W))}


def arrow_stability(res, k):
    """Average overlap (Jaccard) of the arrow sets of consecutive quarters: 1 = identical networks, 0 = nothing kept."""
    sims = []
    for a, b in zip(res[:-1], res[1:]):
        ea, eb = edges(a, k), edges(b, k)
        if ea or eb:
            sims.append(len(ea & eb) / len(ea | eb))
    return float(np.mean(sims))


def evaluate(D, runs):
    """One row per GDBN version: λ (best rank IC on 2006-2020), network size and stability, and results by period.

    Arrow stability depends on network size (denser networks overlap more by chance), so it is also reported at the
    SAME λ as the raw GDBN of the same kind (daily vs daily, slow vs slow): a like-for-like comparison."""
    ks = {name: pc.choose_lambda(D, res, 1)[0] for name, res in runs.items()}
    rows, scores = {}, {}
    for name, res in runs.items():
        k = ks[name]
        ref = "raw daily" if name.endswith("daily") else "raw slow"
        k_ref = ks.get(ref, k)
        s = pc.scores(res, k)
        scores[name] = s
        row = {"λ": pc.ALPHAS[k], "arrows/quarter": np.mean([(q["W"][:, :, k] != 0).sum() for q in res]),
               "arrow stability": arrow_stability(res, k),
               "arrows @ raw λ": np.mean([(q["W"][:, :, k_ref] != 0).sum() for q in res]),
               "stability @ raw λ": arrow_stability(res, k_ref)}
        target = pc.forward_excess(D, s, 1)
        for period, (a, b) in pc.PERIODS.items():
            ic = pc.rank_ic(s.loc[a:b], target.loc[a:b])
            sm = pc.summary(D, pc.daily_portfolio(s), a, b)
            tag = period.split()[0]
            row.update({f"IC {tag}": ic.mean(), f"IC t {tag}": pc.nw_t(ic), f"Sharpe gross {tag}": sm["Sharpe gross"],
                        f"breakeven {tag}": sm["breakeven bps"]})
        rows[name] = row
    return pd.DataFrame(rows).T, scores


def save(method, table, r2_table):
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"summary_{method}.csv")
    r2_table.to_csv(OUT / f"r2_{method}.csv")
    print(f"Saved to {OUT}")
