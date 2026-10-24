"""Build the stock-day FEATURE TABLE (one row per stock per day) for FactorVAE, the GDBN, and the TBGNN.

Run after pull_pit.py / build_pit.py, build_fundamentals.py, and pull_features.py (no WRDS needed here):

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/build_features.py

Every value in row (stock, day t) is known by the CLOSE of day t. The only future column is
target_next (the next trading day's excess return). Values are RAW (not z-scored): standardizing happens
inside each walk-forward training window. Features use each stock's full history (not only its
S&P 500 days), so a stock joining the index already has its past windows; in_sp500 marks the universe.
The test years (2023 onward) are NOT built.

Features
  Returns       exret_1d, exret_5d, exret_20d   excess return vs the S&P 500 members' average, summed
                mom_12_1                        return from 12 months ago to 1 month ago (skips the last month)
                overnight, intraday             close -> next open, and open -> close
  Risk          vol_1d                          log(|return| + 0.0001)
                range                           log(high / low)
                rvol_20                         standard deviation of the past 20 daily returns
                beta_60, ivol_60                60-day regression on the market: exposure and leftover volatility
  Trading       volume                          log(volume / its 20-day average)
                turnover_20                     log of the 20-day average of dollar volume / market cap
                amihud_20                       log of the 20-day average of |return| / dollar volume ($ millions)
                size                            log(market cap)
  News          sentiment, has_news             RavenPack sentiment (missing on no-news days), 1 if any story
                attention                       log(1 + stories)
                surprise                        log(1 + stories) - log(1 + the stock's average over the previous 60 days)
  Earnings      days_since_earn                 trading days since earnings became public (0 = first trading
                                                day after the report date), capped at 126
  Fundamentals  sue, bm, gp, lev, ag            from fundamentals_daily (each from its public date)
  Category      sector                          GICS sector code on that day (e.g. "45" = Information Technology)
  Target        target_next                     the next trading day's excess return

Output: CoreFragment/data/pit/built/features_daily.csv.gz
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIT = Path(__file__).resolve().parent / "data" / "pit"
BUILT = PIT / "built"
BUILD_END = "2022-12-31"            # test years (2023+) are not built
TUNE_END = "2020-12-31"             # sanity IC check uses 2006-2020 only
CAP_DAYS = 126                      # cap for days since earnings (about 6 months)
OPEN = pd.Timestamp("2100-01-01")
MANUAL_SECTOR = {11762: "20", 17700: "45"}      # same fill-ins as build_pit.py (no Compustat link)
OUT = Path(os.environ.get("FEATURES_OUT", BUILT / "features_daily.csv.gz"))
t0 = time.time()

# ----------------------------------------------------------------------------
# 1. Load the daily panel (all stocks ever in the S&P 500, with their full history)
# ----------------------------------------------------------------------------
days = pd.DatetimeIndex(pd.read_csv(BUILT / "trading_days.csv", parse_dates=["date"])["date"])
days = days[days <= BUILD_END]
panel = pd.read_csv(BUILT / "panel.csv.gz", parse_dates=["date"],
                    usecols=["permno", "date", "permco", "ret", "prc", "open", "high", "low", "cap", "volume",
                             "in_sp500", "n_news", "sent_news"])
panel = panel[panel["date"] <= BUILD_END]
panel["in_sp500"] = panel["in_sp500"].astype(str).str.lower().eq("true").astype(float)
permnos = np.sort(panel["permno"].unique())


def wide(col):
    return panel.pivot(index="date", columns="permno", values=col).reindex(index=days, columns=permnos)


R, P, O, H, L = wide("ret"), wide("prc").abs(), wide("open"), wide("high"), wide("low")
CAP, VOL, N, S, MEM = wide("cap"), wide("volume"), wide("n_news"), wide("sent_news"), wide("in_sp500").fillna(0) > 0
print(f"1. Panel: {len(panel):,} stock-days, {len(permnos)} stocks, {days[0].date()} to {days[-1].date()}; "
      f"S&P 500 members per day: {MEM.sum(axis=1).mean():.0f} on average ({time.time() - t0:.0f}s)")

# ----------------------------------------------------------------------------
# 2. Market series (for beta / idiosyncratic volatility)
# ----------------------------------------------------------------------------
mkt_file = PIT / "market_daily.csv"
if mkt_file.exists() and {"mktrf", "rf"} <= set(pd.read_csv(mkt_file, nrows=1).columns):
    mk = pd.read_csv(mkt_file, index_col="date", parse_dates=True).reindex(days)
    MKT, RF = mk["mktrf"], mk["rf"].fillna(0.0)
    mkt_source = "Fama-French market excess return (ff.factors_daily)"
else:
    etf = pd.read_csv(BUILT / "etf_returns.csv", index_col="date", parse_dates=True).reindex(days)
    MKT, RF = etf["SPY"], pd.Series(0.0, index=days)
    mkt_source = "SPY (market_daily.csv not found: run pull_features.py for the Fama-French market and rf)"
print(f"2. Market for beta: {mkt_source}")

# ----------------------------------------------------------------------------
# 3. Price, risk, and trading features
# ----------------------------------------------------------------------------
F = {}
EX = R.sub(R.where(MEM).mean(axis=1), axis=0)                        # excess return vs S&P 500 members' average
F["exret_1d"] = EX
F["exret_5d"] = EX.rolling(5, min_periods=4).sum()
F["exret_20d"] = EX.rolling(20, min_periods=15).sum()
F["mom_12_1"] = np.expm1(np.log1p(R).shift(21).rolling(231, min_periods=180).sum())   # days t-251 ... t-21

intraday = (P / O.where(O > 0) - 1)                                   # open -> close (same day, no adjustment needed)
F["intraday"] = intraday
F["overnight"] = (1 + R) / (1 + intraday) - 1                         # the rest of the total return (incl. splits/dividends)

F["vol_1d"] = np.log(R.abs() + 1e-4)
F["range"] = np.log(H.where(H > 0) / L.where(L > 0)).where(H >= L)
F["rvol_20"] = R.rolling(20, min_periods=15).std()

re = R.sub(RF, axis=0)                                                # stock return minus the risk-free rate
m = pd.DataFrame(np.repeat(MKT.values[:, None], len(permnos), axis=1), index=days, columns=permnos).where(re.notna())
re = re.where(m.notna())
roll = lambda x: x.rolling(60, min_periods=40).mean()
mean_r, mean_m = roll(re), roll(m)
cov = roll(re * m) - mean_r * mean_m
var_m = roll(m * m) - mean_m ** 2
var_r = roll(re * re) - mean_r ** 2
beta = cov / var_m.where(var_m > 0)
F["beta_60"] = beta
F["ivol_60"] = np.sqrt((var_r - beta * cov).clip(lower=0))           # residual std of the 60-day regression

dollar_vol = VOL * P                                                  # dollars traded
F["volume"] = np.log(VOL.where(VOL > 0) / VOL.rolling(20, min_periods=10).mean())
turn = (dollar_vol / (CAP * 1000)).rolling(20, min_periods=10).mean()   # cap is in $ thousands
F["turnover_20"] = np.log(turn.where(turn > 0))
amihud = (R.abs() / (dollar_vol.where(dollar_vol > 0) / 1e6)).rolling(20, min_periods=10).mean()
F["amihud_20"] = np.log(amihud.where(amihud > 0))
F["size"] = np.log(CAP.where(CAP > 0))
print(f"3. Price, risk, trading features ({time.time() - t0:.0f}s)")

# ----------------------------------------------------------------------------
# 4. News features (RavenPack daily aggregates, already in the panel)
# ----------------------------------------------------------------------------
F["sentiment"] = S.where(N > 0)
F["has_news"] = (N > 0).astype(float).where(N.notna())
F["attention"] = np.log1p(N)
F["surprise"] = np.log1p(N) - np.log1p(N.shift(1).rolling(60, min_periods=20).mean())
print(f"4. News features: stock-days with news (S&P members) {(N.where(MEM) > 0).sum().sum() / MEM.sum().sum():.0%}")

# ----------------------------------------------------------------------------
# 5. Days since earnings (Compustat report date -> first trading day after it -> permno via the CCM link)
# ----------------------------------------------------------------------------
fq = pd.read_csv(PIT / "fundamentals" / "fundq.csv", usecols=["gvkey", "rdq"], parse_dates=["rdq"], dtype={"gvkey": str})
fq = fq.dropna(subset=["rdq"]).drop_duplicates()
pos = days.searchsorted(fq["rdq"].values, side="right")
fq = fq[pos < len(days)].assign(usable=days[pos[pos < len(days)]])
ccm = pd.read_csv(PIT / "ccm_links.csv", parse_dates=["linkdt", "linkenddt"], dtype={"gvkey": str})
ccm = ccm.dropna(subset=["permno"]).assign(permno=lambda d: d["permno"].astype(int))
ev = fq.merge(ccm[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey")
ev = ev[(ev["usable"] >= ev["linkdt"]) & (ev["usable"] <= ev["linkenddt"].fillna(OPEN)) & ev["permno"].isin(permnos)]
ev = ev.drop_duplicates(["permno", "usable"])
E = np.zeros((len(days), len(permnos)), dtype=bool)
E[days.get_indexer(ev["usable"]), pd.Index(permnos).get_indexer(ev["permno"])] = True
idx = np.where(E, np.arange(len(days))[:, None], np.nan)
last = pd.DataFrame(idx, index=days, columns=permnos).ffill()
F["days_since_earn"] = last.rsub(np.arange(len(days)), axis=0).clip(upper=CAP_DAYS)
print(f"5. Earnings: {E.sum():,} report days matched to {int(E.any(axis=0).sum())} stocks ({time.time() - t0:.0f}s)")

# ----------------------------------------------------------------------------
# 6. Long table: one row per stock-day in the panel
# ----------------------------------------------------------------------------
out = panel[["date", "permno", "permco", "in_sp500"]].sort_values(["date", "permno"]).reset_index(drop=True)
di = days.get_indexer(out["date"])
pj = pd.Index(permnos).get_indexer(out["permno"])
for name, df in F.items():
    out[name] = df.values[di, pj]
out["target_next"] = EX.shift(-1).values[di, pj]                     # the ONLY future column
out.loc[out["date"] == days[-1], "target_next"] = np.nan              # needs a 2023 return: not built

fd = pd.read_csv(BUILT / "fundamentals_daily.csv.gz", parse_dates=["date"],
                 usecols=["permno", "date", "sue", "bm", "gp", "lev", "ag"])
fd = fd[fd["date"] <= BUILD_END]
fd["bm"] = fd["bm"] * 1000                                            # book equity ($M) / market cap ($000) -> ratio
out = out.merge(fd, on=["permno", "date"], how="left")

# Ticker (for reading only) from the names history
names = pd.read_csv(PIT / "names_history.csv", parse_dates=["start", "ending"])[["permno", "start", "ending", "ticker"]]
out = pd.merge_asof(out.sort_values("date"), names.sort_values("start"), left_on="date", right_on="start",
                    by="permno", direction="backward")
out.loc[out["date"] > out["ending"].fillna(OPEN), "ticker"] = np.nan
out = out.drop(columns=["start", "ending"])

# Sector on each day: permno -> gvkey (valid link) -> GICS sector (valid on that day)
gics = pd.read_csv(PIT / "gics_history.csv", parse_dates=["indfrom", "indthru"], dtype={"gvkey": str, "gsector": str})
gics = gics[gics["indtype"] == "GICS"]
lk = out[["date", "permno"]].merge(ccm[["permno", "gvkey", "linkdt", "linkenddt"]], on="permno")
lk = lk[(lk["date"] >= lk["linkdt"]) & (lk["date"] <= lk["linkenddt"].fillna(OPEN))].drop_duplicates(["date", "permno"])
lk = lk.merge(gics[["gvkey", "gsector", "indfrom", "indthru"]], on="gvkey")
lk = lk[(lk["date"] >= lk["indfrom"]) & (lk["date"] <= lk["indthru"].fillna(OPEN))].drop_duplicates(["date", "permno"])
out = out.merge(lk[["date", "permno", "gsector"]].rename(columns={"gsector": "sector"}), on=["date", "permno"], how="left")
manual = out["permno"].map(MANUAL_SECTOR)
out["sector"] = out["sector"].fillna(manual)
by_permco = out.dropna(subset=["sector"]).drop_duplicates(["date", "permco"]).set_index(["date", "permco"])["sector"]
fill = pd.Series(list(zip(out["date"], out["permco"]))).map(by_permco)       # second share class: same company's sector
out["sector"] = out["sector"].fillna(pd.Series(fill.values, index=out.index))

cols = ["date", "permno", "ticker", "sector", "in_sp500",
        "exret_1d", "exret_5d", "exret_20d", "mom_12_1", "overnight", "intraday",
        "vol_1d", "range", "rvol_20", "beta_60", "ivol_60",
        "volume", "turnover_20", "amihud_20", "size",
        "sentiment", "has_news", "attention", "surprise",
        "days_since_earn", "sue", "bm", "gp", "lev", "ag",
        "target_next"]
out = out.sort_values(["date", "permno"])[cols].reset_index(drop=True)
out = out.replace([np.inf, -np.inf], np.nan)
print(f"6. Long table: {len(out):,} rows x {len(cols)} columns ({time.time() - t0:.0f}s)")

# ----------------------------------------------------------------------------
# 7. Sanity checks
# ----------------------------------------------------------------------------
mem = out[out["in_sp500"] > 0]
feats = cols[5:-1]
print("\n7. Checks (S&P 500 member rows only)")
print(f"   Members per day: mean {mem.groupby('date').size().mean():.0f}, "
      f"min {mem.groupby('date').size().min()}, max {mem.groupby('date').size().max()}")
print(f"   Sector missing: {mem['sector'].isna().mean():.2%}; ticker missing: {mem['ticker'].isna().mean():.2%}")

miss = mem[feats].isna().mean()
desc = mem[feats].describe(percentiles=[0.05, 0.5, 0.95]).T[["5%", "50%", "95%"]]
desc.insert(0, "missing", miss)
print("\n   Missing share and spread of each feature:")
print(desc.round(4).to_string())

print(f"\n   Average beta of S&P 500 members: {mem['beta_60'].mean():.2f} (expect about 1.0)")
print(f"   Median |overnight| {mem['overnight'].abs().median():.4f}, median |intraday| {mem['intraday'].abs().median():.4f}")
print(f"   Median days since earnings: {mem['days_since_earn'].median():.0f} (expect about 30; reports are quarterly)")

aapl = out[(out["permno"] == 14593) & out["date"].between("2019-01-25", "2019-02-01")]
print("\n   Apple around its 29 Jan 2019 earnings (reported after the close -> day 0 is 30 Jan):")
print(aapl[["date", "ticker", "exret_1d", "overnight", "intraday", "volume", "attention", "surprise",
            "days_since_earn", "beta_60", "size"]].round(4).to_string(index=False))

# Look-ahead check: no input should be strongly correlated with tomorrow's return (|IC| > 0.1 would be suspicious)
dev = mem[mem["date"].between("2006-01-01", TUNE_END)]
ics = {}
for f in feats:
    sub = dev[["date", f, "target_next"]].dropna()
    r = sub.groupby("date")[[f, "target_next"]].rank()
    r = r - r.groupby(sub["date"]).transform("mean")                  # within-day demeaned ranks
    g = lambda v: v.groupby(sub["date"]).sum()
    ic = g(r[f] * r["target_next"]) / np.sqrt(g(r[f] ** 2) * g(r["target_next"] ** 2))
    ics[f] = ic.replace([np.inf, -np.inf], np.nan).mean()
ics = pd.Series(ics).sort_values()
print("\n   Rank IC of each feature with the NEXT day's excess return, 2006-2020 (look-ahead check; "
      "real signals are about 0.01-0.03):")
print(ics.round(4).to_string())
flag = ics[ics.abs() > 0.1]
print("   OK: no feature above 0.1" if flag.empty else f"   CHECK FOR LOOK-AHEAD: {list(flag.index)}")

out.to_csv(OUT, index=False, compression="gzip")
print(f"\nSaved {len(out):,} rows to {OUT} ({time.time() - t0:.0f}s)")
