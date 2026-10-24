"""Pull the GDBN dataset from WRDS: CRSP daily returns + RavenPack daily news features.

Run in your own terminal (uses the .pgpass saved by explore_wrds.py):

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/pull_data.py

Outputs (in CoreFragment/data/):
    universe.csv       ticker -> PERMNO -> RavenPack entity (names/IDs only)
    master_long.csv    one row per (date, ticker): ret, news_count, news_sentiment
    gdbn_wide.csv      one row per date: {ticker}_ret, {ticker}_sent, {ticker}_attn

Only DAILY AGGREGATES of RavenPack data are saved (no story-level rows).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wrds

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
START, END = "2015-01-01", "2024-12-31"

STOCKS = {
    "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
    "Banks": ["JPM", "BAC", "GS", "MS", "WFC", "C"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY"],
    "Staples": ["PG", "KO", "PEP", "WMT", "COST"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
    "Industrials": ["CAT", "BA"],
}
MARKET_AND_COMMODITIES = {
    "Market": ["SPY"],
    "Commodities/Rates": ["USO", "GLD", "UNG", "TLT"],
}

MIN_RELEVANCE = 90        # keep stories where the company is a main subject (0-100)
CLOSE_HOUR_ET = 16        # news at/after 4:00 PM ET counts toward the next trading day

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

SECTOR = {t: s for s, ts in {**STOCKS, **MARKET_AND_COMMODITIES}.items() for t in ts}
ALL_TICKERS = list(SECTOR)
STOCK_TICKERS = [t for ts in STOCKS.values() for t in ts]


def sql_list(values):
    """Format a list of safe identifiers for a SQL IN (...) clause."""
    for v in values:
        if not re.fullmatch(r"[A-Za-z0-9.\-]+", str(v)):
            sys.exit(f"Unsafe value in SQL list: {v!r}")
    return ", ".join(f"'{v}'" for v in values)


import os
username = os.environ.get("WRDS_USERNAME")  # optional: skips the username prompt
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()

# ----------------------------------------------------------------------------
# 1. Tickers -> PERMNOs (identity as of the END date, then full history by PERMNO)
# ----------------------------------------------------------------------------
names = db.raw_sql(f"""
    SELECT permno, ticker, comnam, ncusip, namedt, nameenddt, shrcd
    FROM crsp.stocknames
    WHERE ticker IN ({sql_list(ALL_TICKERS)})
      AND namedt <= '{END}' AND nameenddt >= '{END}'
""", date_cols=["namedt", "nameenddt"])

names = names.sort_values("namedt").groupby("ticker").tail(1)
missing = sorted(set(ALL_TICKERS) - set(names["ticker"]))
if missing:
    print("WARNING: no CRSP match on", END, "for:", missing)

permno_of = dict(zip(names["ticker"], names["permno"].astype(int)))
ticker_of = {p: t for t, p in permno_of.items()}

# All historical 8-digit CUSIPs for each PERMNO (for linking to RavenPack)
cusips = db.raw_sql(f"""
    SELECT DISTINCT permno, ncusip FROM crsp.stocknames
    WHERE permno IN ({", ".join(str(p) for p in permno_of.values())}) AND ncusip IS NOT NULL
""")
cusips["permno"] = cusips["permno"].astype(int)

# ----------------------------------------------------------------------------
# 2. Daily returns from CRSP (dsf_v2)
# ----------------------------------------------------------------------------
rets = db.raw_sql(f"""
    SELECT permno, dlycaldt AS date, dlyret AS ret
    FROM crsp.dsf_v2
    WHERE permno IN ({", ".join(str(p) for p in permno_of.values())})
      AND dlycaldt BETWEEN '{START}' AND '{END}'
""", date_cols=["date"])
rets["permno"] = rets["permno"].astype(int)
rets["ticker"] = rets["permno"].map(ticker_of)
trading_days = pd.DatetimeIndex(sorted(rets["date"].unique()))
print(f"CRSP: {len(rets):,} return rows, {len(trading_days):,} trading days")

# ----------------------------------------------------------------------------
# 3. RavenPack entity IDs for the stocks (via CUSIP)
# ----------------------------------------------------------------------------
map_cols = db.describe_table("ravenpack_common", "wrds_rpa_all_mappings")["name"].tolist()
if "cusip" not in map_cols:
    sys.exit(f"Expected a 'cusip' column in wrds_rpa_all_mappings, found: {map_cols}")

stock_cusips = cusips[cusips["permno"].isin([permno_of[t] for t in STOCK_TICKERS if t in permno_of])]
rp_map = db.raw_sql(f"""
    SELECT DISTINCT rp_entity_id, {"entity_name" if "entity_name" in map_cols else "rp_entity_id AS entity_name"},
           substring(cusip, 1, 8) AS ncusip
    FROM ravenpack_common.wrds_rpa_all_mappings
    WHERE {"entity_type = 'COMP' AND" if "entity_type" in map_cols else ""}
      substring(cusip, 1, 8) IN ({sql_list(stock_cusips["ncusip"].unique())})
""")
rp_map = rp_map.merge(stock_cusips, on="ncusip")[["permno", "rp_entity_id", "entity_name"]].drop_duplicates()
rp_map["ticker"] = rp_map["permno"].map(ticker_of)

# Fallback: match by ticker for any stock the CUSIP link missed (e.g., XOM)
unmatched = sorted(set(STOCK_TICKERS) - set(rp_map["ticker"]))
if unmatched and "ticker" in map_cols:
    by_ticker = db.raw_sql(f"""
        SELECT DISTINCT rp_entity_id, {"entity_name" if "entity_name" in map_cols else "rp_entity_id AS entity_name"}, ticker
        FROM ravenpack_common.wrds_rpa_all_mappings
        WHERE {"entity_type = 'COMP' AND" if "entity_type" in map_cols else ""} ticker IN ({sql_list(unmatched)})
    """)
    if not by_ticker.empty:
        by_ticker["permno"] = by_ticker["ticker"].map(permno_of)
        print("Matched by ticker (check these names):")
        print(by_ticker[["ticker", "rp_entity_id", "entity_name"]].to_string(index=False))
        rp_map = pd.concat([rp_map, by_ticker[["permno", "rp_entity_id", "entity_name", "ticker"]]], ignore_index=True)

no_news_id = sorted(set(STOCK_TICKERS) - set(rp_map["ticker"]))
if no_news_id:
    print("WARNING: no RavenPack entity found for:", no_news_id)

# ----------------------------------------------------------------------------
# 4. RavenPack Dow Jones & PR news -> daily features (aggregated, per year)
# ----------------------------------------------------------------------------
entity_to_permno = dict(zip(rp_map["rp_entity_id"], rp_map["permno"]))
ids = sql_list(rp_map["rp_entity_id"].unique())
daily_parts = []

for year in range(int(START[:4]), int(END[:4]) + 1):
    stories = db.raw_sql(f"""
        SELECT timestamp_utc, rp_entity_id, rp_story_id, relevance, event_sentiment_score
        FROM ravenpack_dj.rpa_djpr_equities_{year}
        WHERE rp_entity_id IN ({ids}) AND relevance >= {MIN_RELEVANCE}
    """, date_cols=["timestamp_utc"])
    if stories.empty:
        continue

    # One row per (story, company): a story can contain several events
    stories = (stories.groupby(["rp_story_id", "rp_entity_id"], as_index=False)
                      .agg(timestamp_utc=("timestamp_utc", "min"),
                           sentiment=("event_sentiment_score", "mean")))

    # Assign each story to the trading day whose close it precedes (no look-ahead)
    ts = stories["timestamp_utc"]
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts
    et = ts.dt.tz_convert("America/New_York")
    effective = et.dt.normalize().dt.tz_localize(None) + pd.to_timedelta((et.dt.hour >= CLOSE_HOUR_ET).astype(int), unit="D")
    idx = trading_days.searchsorted(effective)  # next trading day on/after the effective date
    valid = idx < len(trading_days)
    stories = stories[valid].copy()
    stories["date"] = trading_days[idx[valid]]
    stories["permno"] = stories["rp_entity_id"].map(entity_to_permno)

    daily = (stories.groupby(["permno", "date"])
                    .agg(news_count=("rp_story_id", "nunique"), news_sentiment=("sentiment", "mean"))
                    .reset_index())
    daily_parts.append(daily)
    print(f"RavenPack {year}: {len(stories):,} stories -> {len(daily):,} stock-days with news")

news = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(
    columns=["permno", "date", "news_count", "news_sentiment"])
news = news.groupby(["permno", "date"], as_index=False).agg(
    news_count=("news_count", "sum"), news_sentiment=("news_sentiment", "mean"))

db.close()

# ----------------------------------------------------------------------------
# 5. Master dataset (long) and GDBN view (wide)
# ----------------------------------------------------------------------------
master = rets.merge(news, on=["permno", "date"], how="left")
master["news_count"] = master["news_count"].fillna(0).astype(int)
master["news_sentiment"] = master["news_sentiment"].fillna(0.0)  # no news = neutral
master["sector"] = master["ticker"].map(SECTOR)
master = master.sort_values(["date", "ticker"])[
    ["date", "ticker", "permno", "sector", "ret", "news_count", "news_sentiment"]]
master.to_csv(OUT / "master_long.csv", index=False)

wide_ret = master.pivot(index="date", columns="ticker", values="ret").add_suffix("_ret")
stock_rows = master[master["ticker"].isin(STOCK_TICKERS)]
wide_sent = stock_rows.pivot(index="date", columns="ticker", values="news_sentiment").add_suffix("_sent")
wide_attn = np.log1p(stock_rows.pivot(index="date", columns="ticker", values="news_count")).add_suffix("_attn")
wide = pd.concat([wide_ret, wide_sent, wide_attn], axis=1).sort_index()
wide.to_csv(OUT / "gdbn_wide.csv")

universe = (pd.DataFrame({"ticker": ALL_TICKERS})
            .assign(sector=lambda d: d["ticker"].map(SECTOR), permno=lambda d: d["ticker"].map(permno_of))
            .merge(rp_map[["ticker", "rp_entity_id", "entity_name"]], on="ticker", how="left"))
universe.to_csv(OUT / "universe.csv", index=False)

# ----------------------------------------------------------------------------
# 6. Summary (counts only; safe to share)
# ----------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"Dates: {wide.index.min().date()} to {wide.index.max().date()} ({len(wide):,} trading days)")
print(f"Series: {wide_ret.shape[1]} returns, {wide_sent.shape[1]} sentiment, {wide_attn.shape[1]} attention "
      f"= {wide.shape[1]} variables")
print(f"Missing returns: {int(wide_ret.isna().sum().sum())} cells")
coverage = (stock_rows.assign(has_news=stock_rows["news_count"] > 0)
            .groupby("ticker")["has_news"].mean().sort_values())
print("\nShare of trading days with at least one relevant story, per stock:")
print(coverage.round(2).to_string())
print("\nUniverse mapping (check the names look right):")
print(universe.to_string(index=False))
print(f"\nSaved to {OUT}")
