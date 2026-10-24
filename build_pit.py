"""Build the point-in-time (PIT) universe and daily panel from the raw pulls in data/pit/.

No WRDS connection needed. Run after pull_pit.py and pull_pit_fallback.py:

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/build_pit.py

Rules (everything uses only information available at the time):
    - Membership: a stock is in the S&P 500 on a day if it is inside one of its membership spells.
    - Sector: historical GICS sector on that date (CRSP-Compustat link -> Compustat GICS history).
    - One share class per company (CRSP PERMCO): the larger one (e.g., GOOGL over GOOG).
    - News: RavenPack entity -> PERMNO only while the link is valid; stories at/after 4 PM ET count
      for the next trading day; several entities for one stock are combined (counts add up,
      sentiment is averaged weighted by story counts).
    - Delisting returns: already in the daily file (final row flagged delflg = 'Y'), not added again.
    - Universe snapshots: on the first trading day of each quarter, the K largest members in each
      sector by market cap on the PREVIOUS trading day, among stocks with at least MIN_HISTORY
      days of returns.

Outputs (in CoreFragment/data/pit/built/):
    panel.csv.gz       one row per (date, stock): returns, prices, market cap, in_sp500, news features
    universe.csv       one row per (rebalance date, candidate): sector, market-cap rank, selected (top K)
    etf_returns.csv    one row per date: SPY, USO, GLD, DBC, TLT returns
    trading_days.csv   the trading calendar
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

K = 5                    # stocks per sector
REBALANCE = "QS"         # quarterly (first trading day of each quarter)
START, END = "2006-01-01", "2025-12-31"
MIN_HISTORY = 504        # trading days of returns needed before a stock can be selected (about 2 years)

PIT = Path(__file__).resolve().parent / "data" / "pit"
OUT = Path(os.environ.get("PIT_BUILD_OUT", PIT / "built"))
OUT.mkdir(parents=True, exist_ok=True)

SECTOR_NAMES = {"10": "Energy", "15": "Materials", "20": "Industrials", "25": "Consumer Discretionary",
                "30": "Consumer Staples", "35": "Health Care", "40": "Financials", "45": "Information Technology",
                "50": "Communication Services", "55": "Utilities", "60": "Real Estate"}
# Stocks with no Compustat link (checked by hand); other missing ones are second share classes
MANUAL_SECTOR = {11762: "20",   # ETN, Eaton: Industrials
                 17700: "45"}   # DAY, Dayforce: Information Technology
OPEN = pd.Timestamp("2100-01-01")

# ----------------------------------------------------------------------------
# 1. Calendar, ETFs, membership, names
# ----------------------------------------------------------------------------
etf = pd.read_csv(PIT / "etf_daily.csv", parse_dates=["date"])
etf_wide = etf.pivot_table(index="date", columns="ticker", values="ret").sort_index()
days = etf_wide.index[etf_wide["SPY"].notna()]
pd.Series(days, name="date").to_csv(OUT / "trading_days.csv", index=False)
etf_wide.to_csv(OUT / "etf_returns.csv")

members = pd.read_csv(PIT / "sp500_membership.csv", parse_dates=["start", "ending"])
names = pd.read_csv(PIT / "names_history.csv", parse_dates=["start", "ending"])
permco_of = names.drop_duplicates("permno").set_index("permno")["permco"]
print(f"1. Trading days {days[0].date()} to {days[-1].date()} ({len(days):,}); "
      f"{members['permno'].nunique()} stocks ever in the S&P 500 in {START[:4]}-{END[:4]}")

# ----------------------------------------------------------------------------
# 2. Daily stock data + membership flag
# ----------------------------------------------------------------------------
daily = pd.read_csv(PIT / "daily_stock.csv.gz", parse_dates=["date"], low_memory=False,
                    usecols=["permno", "date", "ret", "prc", "open", "high", "low", "cap", "volume", "delflg"])
daily = daily[daily["date"].isin(days)]
daily["prc"] = daily["prc"].abs()
spells = members[["permno", "start", "ending"]]
flag = daily[["permno", "date"]].merge(spells, on="permno")
flag = flag[(flag["date"] >= flag["start"]) & (flag["date"] <= flag["ending"])][["permno", "date"]].drop_duplicates()
flag["in_sp500"] = True
daily = daily.merge(flag, on=["permno", "date"], how="left")
daily["in_sp500"] = daily["in_sp500"].eq(True)        # not a member on that day -> False
daily["permco"] = daily["permno"].map(permco_of)
per_day = daily[daily["in_sp500"] & (daily["date"] >= START)].groupby("date").size()
print(f"2. Daily rows: {len(daily):,}; S&P 500 members per day: min {per_day.min()}, max {per_day.max()}; "
      f"delisting rows (already include the delisting return): {(daily['delflg'] == 'Y').sum()}")

# ----------------------------------------------------------------------------
# 3. News: entity -> permno links valid over time, then daily features per stock
# ----------------------------------------------------------------------------
rp = pd.read_csv(PIT / "rp_entity_map.csv", parse_dates=["range_start", "range_end"])
link = names[["permno", "cusip8", "start", "ending"]].dropna(subset=["cusip8"]).merge(
    rp[["rp_entity_id", "cusip8", "range_start", "range_end"]], on="cusip8")
link["valid_from"] = link[["start", "range_start"]].max(axis=1)
link["valid_to"] = pd.concat([link["ending"], link["range_end"].fillna(OPEN)], axis=1).min(axis=1)
link = link[link["valid_from"] <= link["valid_to"]][["permno", "rp_entity_id", "valid_from", "valid_to"]]
fb = pd.read_csv(PIT / "rp_entity_map_fallback.csv", parse_dates=["valid_from", "valid_to"])
link = pd.concat([link, fb[["permno", "rp_entity_id", "valid_from", "valid_to"]]], ignore_index=True)

news = pd.concat([pd.read_csv(PIT / f, parse_dates=["eff_date"]) for f in ["news_daily.csv.gz", "news_daily_fallback.csv.gz"]
                  if (PIT / f).exists()], ignore_index=True)
idx = days.searchsorted(news["eff_date"])              # next trading day on/after the effective date
news = news[idx < len(days)].copy()
news["date"] = days[idx[idx < len(days)]]
news = news.merge(link, on="rp_entity_id")
news = news[(news["date"] >= news["valid_from"]) & (news["date"] <= news["valid_to"])]
news = news.drop_duplicates(["permno", "rp_entity_id", "date", "eff_date"])

for kind in ["all", "news"]:                           # sentiment weighted by story counts
    has = news[f"sent_{kind}"].notna()
    news[f"w_{kind}"] = news[f"n_{kind}"].where(has, 0)
    news[f"ws_{kind}"] = (news[f"sent_{kind}"] * news[f"n_{kind}"]).where(has, 0)
agg = news.groupby(["permno", "date"]).agg(n_all=("n_all", "sum"), n_news=("n_news", "sum"),
                                           w_all=("w_all", "sum"), ws_all=("ws_all", "sum"),
                                           w_news=("w_news", "sum"), ws_news=("ws_news", "sum")).reset_index()
agg["sent_all"] = (agg["ws_all"] / agg["w_all"]).where(agg["w_all"] > 0, 0.0)
agg["sent_news"] = (agg["ws_news"] / agg["w_news"]).where(agg["w_news"] > 0, 0.0)
daily = daily.merge(agg[["permno", "date", "n_all", "sent_all", "n_news", "sent_news"]], on=["permno", "date"], how="left")
for c in ["n_all", "n_news"]:
    daily[c] = daily[c].fillna(0).astype(int)
for c in ["sent_all", "sent_news"]:
    daily[c] = daily[c].fillna(0.0)
mem = daily[daily["in_sp500"] & (daily["date"] >= START)]
print(f"3. News linked for {link['permno'].nunique()} stocks; member-days with any story: {(mem['n_all'] > 0).mean():.0%} "
      f"(non-price news: {(mem['n_news'] > 0).mean():.0%})")

daily = daily.sort_values(["date", "permno"])
daily.to_csv(OUT / "panel.csv.gz", index=False, compression="gzip")

# ----------------------------------------------------------------------------
# 4. Sector on a date (historical GICS)
# ----------------------------------------------------------------------------
ccm = pd.read_csv(PIT / "ccm_links.csv", parse_dates=["linkdt", "linkenddt"], dtype={"gvkey": str})
gics = pd.read_csv(PIT / "gics_history.csv", parse_dates=["indfrom", "indthru"], dtype={"gvkey": str, "gsector": str})
gics = gics[gics["indtype"] == "GICS"]


def sectors_on(permnos, date):
    """GICS sector of each permno on `date` (manual fill-ins, then same-company share class)."""
    c = ccm[ccm["permno"].isin(permnos) & (ccm["linkdt"] <= date) & (ccm["linkenddt"].fillna(OPEN) >= date)]
    g = gics[(gics["indfrom"] <= date) & (gics["indthru"].fillna(OPEN) >= date)]
    s = c.merge(g[["gvkey", "gsector"]], on="gvkey").drop_duplicates("permno").set_index("permno")["gsector"]
    s = s.reindex(permnos)
    s = s.fillna(pd.Series(MANUAL_SECTOR, dtype=object).reindex(permnos))
    by_permco = pd.Series(s.values, index=permco_of.reindex(permnos).values).dropna()
    by_permco = by_permco[~by_permco.index.duplicated()]
    return s.fillna(pd.Series(permco_of.reindex(permnos).map(by_permco).values, index=permnos))


# ----------------------------------------------------------------------------
# 5. Universe snapshots: top K per sector, using only the previous day's market cap
# ----------------------------------------------------------------------------
history = daily[daily["ret"].notna()].groupby("permno")["date"].apply(lambda d: d.values)
rebal = sorted({days[days.searchsorted(q)] for q in pd.date_range(START, END, freq=REBALANCE)})
ticker_spells = names[["permno", "ticker", "start", "ending"]]
rows = []
for r in rebal:
    prev = days[days.searchsorted(r) - 1]
    d_prev = daily[(daily["date"] == prev) & daily["in_sp500"]].set_index("permno")
    on_r = set(daily.loc[(daily["date"] == r) & daily["in_sp500"], "permno"])
    cand = d_prev[d_prev.index.isin(on_r) & d_prev["cap"].notna()][["cap", "permco"]].copy()
    cand["history_days"] = [int(np.searchsorted(history.get(p, np.array([], dtype="datetime64[ns]")), np.datetime64(r)))
                            for p in cand.index]                 # days with a return before the rebalance date
    cand["sector"] = sectors_on(list(cand.index), r).values
    cand = cand[(cand["history_days"] >= MIN_HISTORY) & cand["sector"].notna()]
    cand = cand.sort_values("cap", ascending=False)
    cand["dup_class"] = cand["permco"].duplicated()              # smaller share class of the same company
    cand["rank_in_sector"] = cand[~cand["dup_class"]].groupby("sector")["cap"].rank(ascending=False, method="first")
    cand["selected"] = (~cand["dup_class"]) & (cand["rank_in_sector"] <= K)
    t = ticker_spells[(ticker_spells["start"] <= r) & (ticker_spells["ending"] >= r)].drop_duplicates("permno").set_index("permno")["ticker"]
    cand["ticker"] = t.reindex(cand.index).values
    cand["rebalance_date"], cand["cap_date"] = r, prev
    rows.append(cand.reset_index())
universe = pd.concat(rows, ignore_index=True)
universe["sector_name"] = universe["sector"].map(SECTOR_NAMES)
universe = universe[["rebalance_date", "cap_date", "permno", "ticker", "permco", "sector", "sector_name", "cap",
                     "rank_in_sector", "dup_class", "history_days", "selected"]]
universe.to_csv(OUT / "universe.csv", index=False)

# ----------------------------------------------------------------------------
# 6. Checks
# ----------------------------------------------------------------------------
sel = universe[universe["selected"]]
size = sel.groupby("rebalance_date").size()
assert (universe["cap_date"] < universe["rebalance_date"]).all(), "selection must use the previous day's market cap"
print(f"4. Rebalance dates: {len(rebal)} ({rebal[0].date()} to {rebal[-1].date()}); "
      f"stocks selected per quarter: min {size.min()}, max {size.max()}; distinct stocks ever selected: {sel['permno'].nunique()}")
print("   Sectors selected per year (count of sectors with K stocks at the first rebalance):")
first_of_year = sel[sel["rebalance_date"].dt.month == 1].groupby(sel["rebalance_date"].dt.year)["sector_name"].nunique()
print("   " + ", ".join(f"{y}: {n}" for y, n in first_of_year.items()))

print("\n5. Companies that failed or were rescued (were they in the universe, and what happened):")
famous = ["LEH", "AIG", "C", "FNM", "FRE", "BSC", "WB", "WM", "MER", "GM", "BAC", "SIVB"]
for tk in famous:
    q = sel[sel["ticker"] == tk]
    if q.empty:
        continue
    p = q["permno"].iloc[0]
    d = daily[daily["permno"] == p]
    worst = d.nsmallest(1, "ret")
    print(f"   {tk:>4}: selected {len(q)} quarters ({q['rebalance_date'].min().date()} to {q['rebalance_date'].max().date()}); "
          f"worst day {worst['date'].iloc[0].date()} {worst['ret'].iloc[0]:+.0%}; last data {d['date'].max().date()}"
          + (" (delisted)" if (d["delflg"] == "Y").any() else ""))

cov = daily.merge(sel[["rebalance_date", "permno"]], on="permno")
cov = cov[(cov["date"] >= cov["rebalance_date"]) & (cov["date"] < cov["rebalance_date"] + pd.offsets.QuarterBegin(1, startingMonth=1))]
cov = cov.drop_duplicates(["date", "permno"])
by_year = cov.groupby(cov["date"].dt.year).agg(any_news=("n_all", lambda s: (s > 0).mean()),
                                              non_price_news=("n_news", lambda s: (s > 0).mean()))
print("\n6. Share of selected stock-days with news, by year:")
print("   " + ", ".join(f"{y}: {r.any_news:.0%}" for y, r in by_year.iterrows()))
print(f"\nSaved to {OUT}")
