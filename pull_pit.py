"""Pull the RAW data for the point-in-time (PIT) universe from WRDS.

Table names were confirmed with explore_pit.py. Run:

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/pull_pit.py

This script only downloads. Building the point-in-time universe (who is in the index on
each date, sector at each date, delisting-adjusted returns, news matched to the right
company at each date) is a separate step (build_pit.py), so nothing here needs re-pulling
if a rule changes.

Outputs (in CoreFragment/data/pit/):
    sp500_membership.csv    PERMNO membership spells in the S&P 500 (start, end)
    names_history.csv       PERMNO -> ticker / CUSIP / company name over time
    daily_stock.csv.gz      daily return, price, market cap, ... for every PERMNO that was ever a
                            member 2006-2025 (from HISTORY_START, for the rolling windows)
    etf_daily.csv           daily returns for SPY, USO, GLD, DBC, TLT
    delistings.csv          delisting dates and delisting returns
    ccm_links.csv           PERMNO <-> GVKEY (CRSP-Compustat link history)
    gics_history.csv        GVKEY -> GICS sector over time
    rp_entity_map.csv       RavenPack entity <-> CUSIP
    news_daily.csv.gz       RavenPack DAILY AGGREGATES per entity (no story-level rows)

RavenPack: only daily aggregates are downloaded (aggregated on the WRDS server).
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import wrds

from rp_news import MIN_RELEVANCE, PRICE_GROUPS, daily_news_sql   # shared news rules

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
START, END = "2006-01-01", "2025-12-31"   # universe period
HISTORY_START = "2003-01-01"              # extra history so the first 3-year windows have data
ETFS = ["SPY", "USO", "GLD", "DBC", "TLT"]
CHUNK = 200                               # PERMNOs per daily-stock query

# Table names confirmed with explore_pit.py (Oct 16, 2026). All new-format (v2/CIZ) tables,
# since the legacy tables (dsp500list, stocknames, dsedelist) stop at Dec 2024.
CONFIRMED = True
CONFIG = {
    "sp500": ("crsp", "dsp500list_v2"),         # S&P 500 membership spells (to 2025-12-31)
    "names": ("crsp", "stocknames_v2"),         # names history: ticker, CUSIP, issuer (to 2025)
    "delist": ("crsp", "stkdelists"),           # delisting returns (to 2025-12-30)
    "ccm": ("crsp", "ccmxpf_lnkhist"),          # CRSP-Compustat link
    "gics": ("comp", "co_hgic"),                # historical GICS (1999 onward)
    "daily": ("crsp", "dsf_v2"),                # daily stock file (to 2025-12-31)
    "rp_map": ("ravenpack_common", "wrds_rpa_all_mappings"),   # has range_start / range_end
}

OUT = Path(__file__).resolve().parent / "data" / "pit"

if not CONFIRMED:
    sys.exit("Table names not confirmed yet: run explore_pit.py first, share its output, "
             "then set CONFIRMED = True.")
OUT.mkdir(parents=True, exist_ok=True)


def sql_list(values):
    """Format safe identifiers (tickers, CUSIPs, entity IDs) for a SQL IN (...) clause."""
    for v in values:
        if not re.fullmatch(r"[A-Za-z0-9.\-]+", str(v)):
            sys.exit(f"Unsafe value in SQL list: {v!r}")
    return ", ".join(f"'{v}'" for v in values)


def int_list(values):
    return ", ".join(str(int(v)) for v in values)


def columns(key):
    lib, table = CONFIG[key]
    return [c.lower() for c in db.describe_table(library=lib, table=table)["name"]]


def pick(cols, candidates, required=True, what=""):
    """First candidate column that exists (table formats differ between CRSP versions)."""
    for c in candidates:
        if c in cols:
            return c
    if required:
        sys.exit(f"None of {candidates} found for {what}. Columns: {cols}")
    return None


def table(key):
    return "{}.{}".format(*CONFIG[key])


username = os.environ.get("WRDS_USERNAME")
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()

# ----------------------------------------------------------------------------
# 1. S&P 500 membership spells
# ----------------------------------------------------------------------------
cols = columns("sp500")
c_start = pick(cols, ["start", "mbrstartdt", "startdt"], what="membership start")
c_end = pick(cols, ["ending", "mbrenddt", "enddt", "end"], what="membership end")
extra_m = "".join(f", {c}" for c in ["indno", "mbrflg", "indfam"] if c in cols)
members = db.raw_sql(f"""
    SELECT permno, {c_start} AS start, {c_end} AS ending{extra_m} FROM {table("sp500")}
    WHERE {c_start} <= '{END}' AND {c_end} >= '{START}'
""", date_cols=["start", "ending"])
members["permno"] = members["permno"].astype(int)
members.to_csv(OUT / "sp500_membership.csv", index=False)
PERMNOS = sorted(members["permno"].unique())
print(f"1. Membership: {len(members):,} spells, {len(PERMNOS):,} stocks; "
      f"latest end date {members['ending'].max().date()}")
if members["ending"].max() < pd.Timestamp(END) - pd.Timedelta(days=30):
    print(f"   NOTE: membership data ends {members['ending'].max().date()}, before {END}. "
          "Members on that date will be assumed to stay in the index afterward (flagged in build_pit.py).")

# ----------------------------------------------------------------------------
# 2. Names history (tickers and CUSIPs over time), new format
# ----------------------------------------------------------------------------
nc = columns("names")
keep = [c for c in ["sharetype", "securitytype", "securitysubtype", "primaryexch", "shareclass",
                    "issuertype", "usincflg", "siccd", "hdrcusip"] if c in nc]
names = db.raw_sql(f"""
    SELECT permno, permco, namedt AS start, nameenddt AS ending, ticker,
           substring(cusip, 1, 8) AS cusip8, issuernm AS name{"".join(", " + c for c in keep)}
    FROM {table("names")} WHERE permno IN ({int_list(PERMNOS)})
""", date_cols=["start", "ending"])
names["permno"] = names["permno"].astype(int)
names.to_csv(OUT / "names_history.csv", index=False)
print(f"2. Names history: {len(names):,} rows, {names['cusip8'].nunique():,} distinct CUSIPs, "
      f"latest name date {names['ending'].max().date()}")

# ETF PERMNOs (by ticker, as of the last names date)
last_names = str(names["ending"].max().date())
etf = db.raw_sql(f"""
    SELECT DISTINCT permno, ticker FROM {table("names")}
    WHERE ticker IN ({sql_list(ETFS)}) AND namedt <= '{last_names}' AND nameenddt >= '{last_names}'
""")
etf_permno = dict(zip(etf["ticker"], etf["permno"].astype(int)))
print(f"   ETFs found: {sorted(etf_permno)}" + (f"; MISSING: {sorted(set(ETFS) - set(etf_permno))}" if set(ETFS) - set(etf_permno) else ""))

# ----------------------------------------------------------------------------
# 3. Daily stock file (chunked; written as it goes)
# ----------------------------------------------------------------------------
dc = columns("daily")
optional = {"prc": ["dlyprc"], "open": ["dlyopen"], "high": ["dlyhigh"], "low": ["dlylow"],
            "cap": ["dlycap"], "shrout": ["shrout"], "volume": ["dlyvol"],
            "delflg": ["dlydelflg"], "prcflg": ["dlyprcflg"], "retmiss": ["dlyretmissflg"]}
extra = {k: pick(dc, v, required=False) for k, v in optional.items()}
select_extra = "".join(f", {c} AS {k}" for k, c in extra.items() if c)
print(f"3. Daily file columns used: ret{select_extra.replace(' AS', ' as')}")

daily_path = OUT / "daily_stock.csv.gz"
daily_path.unlink(missing_ok=True)
n_rows = 0
for i in range(0, len(PERMNOS), CHUNK):
    part = db.raw_sql(f"""
        SELECT permno, dlycaldt AS date, dlyret AS ret{select_extra}
        FROM {table("daily")}
        WHERE permno IN ({int_list(PERMNOS[i:i + CHUNK])}) AND dlycaldt BETWEEN '{HISTORY_START}' AND '{END}'
    """, date_cols=["date"])
    part.to_csv(daily_path, mode="a", header=(i == 0), index=False, compression="gzip")
    n_rows += len(part)
    print(f"   stocks {i + 1}-{min(i + CHUNK, len(PERMNOS))} of {len(PERMNOS)}: {n_rows:,} rows so far")

etf_daily = db.raw_sql(f"""
    SELECT permno, dlycaldt AS date, dlyret AS ret FROM {table("daily")}
    WHERE permno IN ({int_list(etf_permno.values())}) AND dlycaldt BETWEEN '{HISTORY_START}' AND '{END}'
""", date_cols=["date"])
etf_daily["ticker"] = etf_daily["permno"].map({p: t for t, p in etf_permno.items()})
etf_daily.to_csv(OUT / "etf_daily.csv", index=False)
print(f"   ETF rows: {len(etf_daily):,}; last trading day {etf_daily['date'].max().date()}")

# ----------------------------------------------------------------------------
# 4. Delistings
# ----------------------------------------------------------------------------
xc = columns("delist")
d_date = pick(xc, ["delistingdt", "dlstdt", "delistdt"], what="delisting date")
d_ret = pick(xc, ["delret", "dlret", "delistingret"], what="delisting return")
d_extra = "".join(f", {c}" for c in ["delactiontype", "delstatustype", "delreasontype", "delpaymenttype",
                                       "delretmisstype", "delpermno", "deldlydt"] if c in xc)
delist = db.raw_sql(f"""
    SELECT permno, {d_date} AS delist_date, {d_ret} AS delist_ret{d_extra}
    FROM {table("delist")} WHERE permno IN ({int_list(PERMNOS)})
""", date_cols=["delist_date"])
delist.to_csv(OUT / "delistings.csv", index=False)
print(f"4. Delistings: {len(delist):,} rows ({delist['delist_ret'].notna().sum():,} with a delisting return)")

# ----------------------------------------------------------------------------
# 5. CRSP-Compustat link and historical GICS
# ----------------------------------------------------------------------------
ccm = db.raw_sql(f"""
    SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim FROM {table("ccm")}
    WHERE lpermno IN ({int_list(PERMNOS)}) AND linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
""", date_cols=["linkdt", "linkenddt"])
ccm.to_csv(OUT / "ccm_links.csv", index=False)
gvkeys = sorted(ccm["gvkey"].dropna().unique())
gics = db.raw_sql(f"""
    SELECT gvkey, indtype, indfrom, indthru, gsector, ggroup, gind, gsubind FROM {table("gics")}
    WHERE gvkey IN ({sql_list(gvkeys)})
""", date_cols=["indfrom", "indthru"])
gics.to_csv(OUT / "gics_history.csv", index=False)
print(f"5. CCM links: {len(ccm):,} rows ({ccm['permno'].nunique():,} stocks linked); GICS history: {len(gics):,} rows")

# ----------------------------------------------------------------------------
# 6. RavenPack: entities by CUSIP, then daily aggregates (computed on the WRDS server)
# ----------------------------------------------------------------------------
mc = columns("rp_map")
range_cols = [c for c in ["range_start", "range_end"] if c in mc]
cusips = sorted(names["cusip8"].dropna().astype(str).str[:8].unique())
rp_parts = []
for i in range(0, len(cusips), 1000):
    rp_parts.append(db.raw_sql(f"""
        SELECT DISTINCT rp_entity_id, {"entity_name" if "entity_name" in mc else "rp_entity_id AS entity_name"},
               substring(cusip, 1, 8) AS cusip8{"".join(", " + c for c in range_cols)}
        FROM {table("rp_map")}
        WHERE {"entity_type = 'COMP' AND " if "entity_type" in mc else ""}substring(cusip, 1, 8) IN ({sql_list(cusips[i:i + 1000])})
    """))
rp_map = pd.concat(rp_parts, ignore_index=True).drop_duplicates()
rp_map.to_csv(OUT / "rp_entity_map.csv", index=False)
ENTITIES = sorted(rp_map["rp_entity_id"].unique())
print(f"6. RavenPack entities: {len(ENTITIES):,} (from {rp_map['cusip8'].nunique():,} CUSIPs)")

# Per story: average sentiment over all events, and over non-price events only.
# Effective date: the story's date in New York time, +1 day if at/after the 4 PM close
# (mapped to the next trading day in build_pit.py).
price = sql_list(PRICE_GROUPS)
news_path = OUT / "news_daily.csv.gz"
news_path.unlink(missing_ok=True)
first = True
for year in range(int(HISTORY_START[:4]), int(END[:4]) + 1):
    agg = db.raw_sql(daily_news_sql(year, sql_list(ENTITIES), price), date_cols=["eff_date"])
    agg.to_csv(news_path, mode="a", header=first, index=False, compression="gzip")
    first = False
    print(f"   RavenPack {year}: {len(agg):,} entity-days, {int(agg['n_all'].sum()):,} stories "
          f"({int(agg['n_news'].sum()):,} with non-price news)")

db.close()
print(f"\nSaved to {OUT}")
