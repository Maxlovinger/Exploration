"""Build DAILY fundamental and CEO features for the point-in-time stocks, each only from the day it was public.

Run after pull_fundamentals.py and build_pit.py (no WRDS needed):

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/build_fundamentals.py

Features (one value per stock per trading day, carried forward until the next update):

  Balance sheet / earnings (Compustat quarterly)
    bm         book-to-market: book equity / market cap (yesterday's)            value
    gp         gross profitability: last 4 quarters' (sales - cost of goods) / assets
    lev        leverage: (long-term + short-term debt) / assets
    ag         asset growth: assets vs 4 quarters earlier
    sue        earnings surprise: EPS vs the same quarter last year, divided by the
               usual size of that change over the previous 8 quarters        post-earnings drift
  CEO (ExecuComp)
    log_pay    log(1 + CEO total pay, $ thousands)
    eq_share   share of CEO pay in stock and options
    own        CEO ownership, % of shares (often missing)
    tenure     years since the current CEO took office
    new_ceo    1 if the CEO took office in the past 12 months

When each value becomes usable (no look-ahead):
    EPS-based (sue)         the trading day AFTER the earnings report date (rdq); if missing, quarter-end + 90 days
    balance-sheet based     the later of that day and quarter-end + 45 days (10-Q deadline for large firms)
    CEO pay (log_pay, eq_share, own)
                            the trading day after the first proxy statement (DEF 14A) filed 45-400 days after
                            fiscal year-end; if none, fiscal year-end + 150 days
    tenure, new_ceo         from the date the CEO took office (CEO changes are announced publicly)
    Values older than STALE_DAYS are dropped (company stopped reporting).

Output: data/pit/built/fundamentals_daily.csv.gz (permno, date, the 10 features)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PIT = Path(__file__).resolve().parent / "data" / "pit"
FUND = PIT / "fundamentals"
BUILT = PIT / "built"
OUT = Path(os.environ.get("FUND_BUILD_OUT", BUILT))
STALE_DAYS = 400
QUARTERLY = ["bm_be", "gp", "lev", "ag", "sue"]
CEO_PAY = ["log_pay", "eq_share", "own"]
OPEN = pd.Timestamp("2100-01-01")

days = pd.DatetimeIndex(pd.read_csv(BUILT / "trading_days.csv", parse_dates=["date"])["date"])


def next_trading_day(dates):
    """First trading day strictly after each date (NaT stays NaT)."""
    d = pd.to_datetime(dates)
    idx = days.searchsorted(d.fillna(OPEN).values, side="right")
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    ok = d.notna().values & (idx < len(days))
    out[ok] = days[idx[ok]]
    return out


# ----------------------------------------------------------------------------
# 1. Quarterly features
# ----------------------------------------------------------------------------
f = pd.read_csv(FUND / "fundq.csv", parse_dates=["datadate", "rdq"], dtype={"gvkey": str})
f = f.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last")
g = f.groupby("gvkey")
gross = (f["saleq"] - f["cogsq"])
f["gp"] = gross.groupby(f["gvkey"]).rolling(4, min_periods=4).sum().reset_index(level=0, drop=True) / f["atq"]
f["lev"] = (f["dlttq"].fillna(0) + f["dlcq"].fillna(0)) / f["atq"]
f.loc[f["dlttq"].isna() & f["dlcq"].isna(), "lev"] = np.nan
f["ag"] = f["atq"] / g["atq"].shift(4) - 1
d_eps = f["epspxq"] - g["epspxq"].shift(4)
scale = d_eps.groupby(f["gvkey"]).transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
f["sue"] = d_eps / scale.where(scale > 0)
f["bm_be"] = f["ceqq"]                                        # book equity; divided by market cap daily below
for c in ["gp", "lev", "ag", "sue"]:
    f[c] = f[c].replace([np.inf, -np.inf], np.nan)

report = f["rdq"].where(f["rdq"].notna(), f["datadate"] + pd.Timedelta(days=90))
f["avail_eps"] = next_trading_day(report)
f["avail_bs"] = pd.concat([f["avail_eps"], next_trading_day(f["datadate"] + pd.Timedelta(days=45))], axis=1).max(axis=1)
print(f"1. Quarterly rows: {len(f):,}; median days quarter-end -> usable: EPS {(f['avail_eps'] - f['datadate']).dt.days.median():.0f}, "
      f"balance sheet {(f['avail_bs'] - f['datadate']).dt.days.median():.0f}")

# ----------------------------------------------------------------------------
# 2. CEO features
# ----------------------------------------------------------------------------
ceo = pd.read_csv(FUND / "execucomp_ceo.csv", parse_dates=["becameceo", "leftofc"], dtype={"gvkey": str})
ceo = ceo.sort_values("becameceo").drop_duplicates(["gvkey", "year"], keep="last")     # CEO in office at year-end
ceo["log_pay"] = np.log1p(ceo["tdc1"].clip(lower=0))
ceo["eq_share"] = ((ceo["stock_awards_fv"].fillna(0) + ceo["option_awards_fv"].fillna(0)) / ceo["tdc1"].where(ceo["tdc1"] > 0)).clip(0, 1)
ceo["own"] = ceo["shrown_tot_pct"]

fy = f.drop_duplicates(["gvkey", "fyearq"], keep="last")[["gvkey", "fyearq", "fyr"]]
ceo = ceo.merge(fy, left_on=["gvkey", "year"], right_on=["gvkey", "fyearq"], how="left")
yr = (ceo["year"] + np.where(ceo["fyr"] <= 5, 1, 0)).fillna(2000).astype(int)
ceo["fye"] = pd.to_datetime(dict(year=yr, month=ceo["fyr"].fillna(12).astype(int), day=1)) + pd.offsets.MonthEnd(0)
ceo.loc[ceo["fyr"].isna(), "fye"] = pd.to_datetime(ceo["year"].astype(int).astype(str) + "-12-31")

cik = pd.read_csv(FUND / "company_cik.csv", dtype={"gvkey": str})
cik["cik"] = pd.to_numeric(cik["cik"], errors="coerce")
proxy = pd.read_csv(FUND / "proxy_dates.csv", parse_dates=["fdate"])
m = ceo[["gvkey", "year", "fye"]].merge(cik[["gvkey", "cik"]], on="gvkey").merge(proxy, on="cik")
m = m[(m["fdate"] >= m["fye"] + pd.Timedelta(days=45)) & (m["fdate"] <= m["fye"] + pd.Timedelta(days=400))]
first_proxy = m.sort_values("fdate").drop_duplicates(["gvkey", "year"]).set_index(["gvkey", "year"])["fdate"]
ceo["public"] = pd.Series(list(zip(ceo["gvkey"], ceo["year"]))).map(first_proxy).values
matched = ceo["public"].notna().mean()
ceo["public"] = ceo["public"].fillna(ceo["fye"] + pd.Timedelta(days=150))
ceo["avail_pay"] = next_trading_day(ceo["public"])
print(f"2. CEO company-years: {len(ceo):,}; pay dated by a proxy filing: {matched:.1%} (rest: fiscal year-end + 150 days)")

# CEO spells (who is CEO when): from the date they took office
spells = ceo.dropna(subset=["becameceo"]).drop_duplicates(["gvkey", "execid"])[["gvkey", "execid", "becameceo"]]

# ----------------------------------------------------------------------------
# 3. Daily panel: permno -> gvkey (valid link on the date) -> latest available values
# ----------------------------------------------------------------------------
panel = pd.read_csv(BUILT / "panel.csv.gz", parse_dates=["date"], usecols=["permno", "date", "cap"])
ccm = pd.read_csv(PIT / "ccm_links.csv", parse_dates=["linkdt", "linkenddt"], dtype={"gvkey": str})
link = panel.merge(ccm[["permno", "gvkey", "linkdt", "linkenddt"]], on="permno")
link = link[(link["date"] >= link["linkdt"]) & (link["date"] <= link["linkenddt"].fillna(OPEN))]
link = link.drop_duplicates(["permno", "date"])[["permno", "date", "gvkey", "cap"]].sort_values("date")


def asof(left, events, avail_col, cols):
    """For each (gvkey, date), the latest event row usable on or before that date (and not stale)."""
    ev = events.dropna(subset=[avail_col]).sort_values(avail_col)[["gvkey", avail_col] + cols].rename(columns={avail_col: "avail"})
    out = pd.merge_asof(left, ev, left_on="date", right_on="avail", by="gvkey", direction="backward")
    stale = (out["date"] - out["avail"]).dt.days > STALE_DAYS
    out.loc[stale, cols] = np.nan
    return out.drop(columns=["avail"])

daily = asof(link, f, "avail_bs", ["bm_be", "gp", "lev", "ag"])
daily = asof(daily, f, "avail_eps", ["sue"])
daily = asof(daily, ceo, "avail_pay", CEO_PAY)
sp = spells.sort_values("becameceo").rename(columns={"becameceo": "start"})
daily = pd.merge_asof(daily.sort_values("date"), sp.assign(start2=sp["start"]), left_on="date", right_on="start",
                      by="gvkey", direction="backward")
daily["tenure"] = (daily["date"] - daily["start2"]).dt.days / 365.25
daily["new_ceo"] = (daily["tenure"] < 1).astype(float).where(daily["tenure"].notna())

daily = daily.sort_values(["permno", "date"])
daily["bm"] = daily["bm_be"] / daily.groupby("permno")["cap"].shift(1)      # market cap known before the day
FEATURES = ["bm", "gp", "lev", "ag", "sue", "log_pay", "eq_share", "own", "tenure", "new_ceo"]
daily = daily[["permno", "date"] + FEATURES].replace([np.inf, -np.inf], np.nan)
OUT.mkdir(parents=True, exist_ok=True)
daily.to_csv(OUT / "fundamentals_daily.csv.gz", index=False, compression="gzip")

# ----------------------------------------------------------------------------
# 4. Checks: coverage on the selected universe's stock-days, by year
# ----------------------------------------------------------------------------
uni = pd.read_csv(BUILT / "universe.csv", parse_dates=["rebalance_date"]).query("selected")
uni["end"] = uni["rebalance_date"] + pd.offsets.QuarterBegin(1, startingMonth=1)
sel = daily.merge(uni[["permno", "rebalance_date", "end"]], on="permno")
sel = sel[(sel["date"] >= sel["rebalance_date"]) & (sel["date"] < sel["end"])]
cov = sel.groupby(sel["date"].dt.year)[FEATURES].apply(lambda d: d.notna().mean()).round(2)
print(f"3. Daily rows: {len(daily):,} ({daily['permno'].nunique()} stocks)")
print("\nShare of selected stock-days with each feature, by year:")
print(cov.to_string())

a = f[f["gvkey"] == "001690"].tail(4)            # Apple: when did its last quarters become usable?
if len(a):
    print("\nApple, last 4 quarters (quarter end -> earnings report -> usable for EPS / balance sheet):")
    for _, r in a.iterrows():
        print(f"   {r['datadate'].date()} -> {r['rdq'].date() if pd.notna(r['rdq']) else 'n/a'} -> "
              f"{r['avail_eps'].date()} / {r['avail_bs'].date()}")
print(f"\nSaved to {OUT}")
