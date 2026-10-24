"""Pull the market-wide daily series needed for the stock feature table (beta, idiosyncratic volatility).

Everything else in the feature table was already pulled from WRDS by pull_pit.py (CRSP daily file with
open/high/low/volume/shares, GICS history, CRSP-Compustat links) and pull_fundamentals.py (Compustat
earnings dates and fundamentals), so only these are new:

    ff.factors_daily   Fama-French daily factors: market excess return (mktrf), smb, hml, momentum (umd),
                       and the daily risk-free rate (rf)
    crsp.dsi           CRSP daily market index returns: value-weighted (vwretd), equal-weighted (ewretd),
                       and the S&P 500 index return (sprtrn). Optional (a cross-check for the FF market).

Run (then run build_features.py):

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/pull_features.py

Output: CoreFragment/data/pit/market_daily.csv (one row per trading day)
"""

import os
from pathlib import Path

import pandas as pd
import wrds

START, END = "2003-01-01", "2025-12-31"
OUT = Path(__file__).resolve().parent / "data" / "pit" / "market_daily.csv"

SOURCES = {
    "ff": ("ff", "factors_daily", "date", ["mktrf", "smb", "hml", "umd", "rf"]),
    "crsp": ("crsp", "dsi", "date", ["vwretd", "ewretd", "sprtrn"]),
}

username = os.environ.get("WRDS_USERNAME")
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()

parts = []
for name, (lib, tab, dcol, wanted) in SOURCES.items():
    try:
        cols = set(db.describe_table(lib, tab)["name"].str.lower())
    except Exception as e:
        print(f"{lib}.{tab}: not available ({type(e).__name__}); skipped")
        continue
    use = [c for c in wanted if c in cols]
    missing = [c for c in wanted if c not in cols]
    if not use:
        print(f"{lib}.{tab}: none of {wanted} found (columns: {sorted(cols)[:20]} ...); skipped")
        continue
    df = db.raw_sql(f"SELECT {dcol} AS date, {', '.join(use)} FROM {lib}.{tab} "
                    f"WHERE {dcol} BETWEEN '{START}' AND '{END}' ORDER BY {dcol}", date_cols=["date"])
    df = df.set_index("date").astype(float)
    print(f"{lib}.{tab}: {len(df):,} days ({df.index.min().date()} to {df.index.max().date()}), "
          f"columns {use}" + (f"; not found: {missing}" if missing else ""))
    parts.append(df)

db.close()
if not parts:
    raise SystemExit("Nothing pulled. build_features.py will fall back to SPY (from etf_returns.csv) with rf = 0.")

market = pd.concat(parts, axis=1).sort_index()
OUT.parent.mkdir(parents=True, exist_ok=True)
market.to_csv(OUT)

# Sanity check: FF market (mktrf + rf) vs the CRSP value-weighted return should be almost identical
if {"mktrf", "rf", "vwretd"} <= set(market.columns):
    both = market[["mktrf", "rf", "vwretd"]].dropna()
    print(f"\nCheck: correlation of FF market (mktrf + rf) with CRSP vwretd = "
          f"{(both['mktrf'] + both['rf']).corr(both['vwretd']):.4f} (expect > 0.99)")
print(f"\nSaved {len(market):,} days to {OUT}")
