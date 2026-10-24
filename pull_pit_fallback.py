"""RavenPack fallback for S&P 500 members that pull_pit.py could not match by CUSIP.

Some companies (e.g., XOM, HON, GM, Lehman) have no CUSIP link in RavenPack's mapping table.
Two extra passes, both requiring that the COMPANY NAMES AGREE (name_match.py), because tickers
are reused around the world (e.g., CCL is Carnival in the US but "CCL Products" in India):

    A. Ticker: same ticker as CRSP used at the time, mapping valid on overlapping dates, names agree.
    B. Name search (for stocks still unmatched): RavenPack entities with EXACTLY the same identifying
       words, in the same order, as the CRSP name ("Lehman Brothers Holdings Inc" = "LEHMAN BROTHERS
       HOLDINGS INC"; "City National Corp." is not "National City Corp."). Looser matches were mostly
       wrong companies. For US-incorporated companies, entities whose only ISINs are foreign are rejected.

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/pull_pit_fallback.py

Outputs (in CoreFragment/data/pit/):
    rp_entity_map_fallback.csv     permno <-> RavenPack entity, with the pass that matched it (check the names)
    news_daily_fallback.csv.gz     daily news aggregates for matched entities not already in news_daily.csv.gz
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import wrds

from name_match import exact_name, name_match, tokens
from rp_news import PRICE_GROUPS, daily_news_sql   # same news rules as pull_pit.py

HISTORY_START, END = "2003-01-01", "2025-12-31"
PIT = Path(__file__).resolve().parent / "data" / "pit"
OPEN_END = pd.Timestamp("2100-01-01")
# RavenPack entity names to drop after review: exact name matches that are a different company
MANUAL_EXCLUDE = {"Discovery Ltd."}   # South African insurer, not Discovery Inc (RavenPack lists that as Warner Bros. Discovery)


def sql_list(values):
    for v in values:
        if not re.fullmatch(r"[A-Za-z0-9.\-]+", str(v)):
            sys.exit(f"Unsafe value in SQL list: {v!r}")
    return ", ".join(f"'{v}'" for v in values)


def overlap(df):
    """Keep CRSP name spells and RavenPack mapping rows that are valid on overlapping dates."""
    r_end = df["range_end"].fillna(OPEN_END)
    r_start = df["range_start"].fillna(pd.Timestamp("1900-01-01"))
    df = df[(r_start <= df["ending"]) & (r_end >= df["start"])].copy()
    df["valid_from"] = pd.concat([df["start"], r_start.loc[df.index]], axis=1).max(axis=1)
    df["valid_to"] = pd.concat([df["ending"], r_end.loc[df.index]], axis=1).min(axis=1)
    return df


def summarize(pairs, label):
    return (pairs.groupby(["permno", "rp_entity_id"], as_index=False)
                 .agg(ticker=("ticker", "first"), crsp_name=("name", "last"), entity_name=("entity_name", "first"),
                      valid_from=("valid_from", "min"), valid_to=("valid_to", "max"))
                 .assign(match=label))


# ----------------------------------------------------------------------------
# 1. Which members have no RavenPack entity via CUSIP?
# ----------------------------------------------------------------------------
members = pd.read_csv(PIT / "sp500_membership.csv", parse_dates=["start", "ending"])
names = pd.read_csv(PIT / "names_history.csv", parse_dates=["start", "ending"])
rp_map = pd.read_csv(PIT / "rp_entity_map.csv")

by_cusip = names[["permno", "cusip8"]].dropna().merge(rp_map[["rp_entity_id", "cusip8"]].drop_duplicates(), on="cusip8")
unmatched = sorted(set(members["permno"]) - set(by_cusip["permno"]))
spells = names[names["permno"].isin(unmatched) & (names["ending"] >= HISTORY_START)][
    ["permno", "ticker", "name", "start", "ending", "usincflg"]]
print(f"1. Members without a CUSIP match: {len(unmatched)}")

username = os.environ.get("WRDS_USERNAME")
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()

# ----------------------------------------------------------------------------
# 2A. Ticker + overlapping dates + names agree
# ----------------------------------------------------------------------------
cand = db.raw_sql(f"""
    SELECT DISTINCT rp_entity_id, entity_name, ticker, range_start, range_end
    FROM ravenpack_common.wrds_rpa_all_mappings
    WHERE entity_type = 'COMP' AND ticker IN ({sql_list(sorted(spells["ticker"].dropna().unique()))})
""", date_cols=["range_start", "range_end"])
pairs = overlap(spells.dropna(subset=["ticker"]).merge(cand, on="ticker"))
pairs = pairs[[name_match(a, b) for a, b in zip(pairs["name"], pairs["entity_name"])]]
found_a = summarize(pairs, "A: ticker+dates+name")
print(f"2A. Ticker pass: {found_a['permno'].nunique()} stocks matched")

# ----------------------------------------------------------------------------
# 2B. Name search for the rest (names agree; US companies need a US ISIN if any ISIN exists)
# ----------------------------------------------------------------------------
rest = spells[~spells["permno"].isin(found_a["permno"])]
first_words = sorted({t[0] for t in rest["name"].map(tokens) if t and len(t[0]) >= 4})   # [a-z0-9] only
found_b = pd.DataFrame(columns=found_a.columns)
if first_words:
    cand_b = db.raw_sql(f"""
        SELECT DISTINCT rp_entity_id, entity_name, isin, range_start, range_end
        FROM ravenpack_common.wrds_rpa_all_mappings
        WHERE entity_type = 'COMP' AND lower(entity_name) ~ '{"|".join(first_words)}'
    """, date_cols=["range_start", "range_end"])
    isin_by_entity = cand_b.dropna(subset=["isin"]).groupby("rp_entity_id")["isin"].apply(lambda s: {x[:2] for x in s})
    ents = cand_b.groupby(["rp_entity_id", "entity_name"], as_index=False).agg(
        range_start=("range_start", "min"), range_end=("range_end", lambda s: pd.NaT if s.isna().any() else s.max()))
    pairs_b = rest.merge(ents, how="cross")
    # A stock-entity pair is confirmed if ANY of the stock's names matches exactly; it then applies to
    # all of that stock's name periods (e.g., "Apollo Group" before it became "Apollo Education Group")
    hit = pairs_b[[exact_name(a, b) for a, b in zip(pairs_b["name"], pairs_b["entity_name"])]]
    confirmed = hit[["permno", "rp_entity_id"]].drop_duplicates()
    pairs_b = overlap(pairs_b.merge(confirmed, on=["permno", "rp_entity_id"]))
    countries = pairs_b["rp_entity_id"].map(isin_by_entity)
    us_ok = [(u != "Y") or not isinstance(c, set) or ("US" in c) for u, c in zip(pairs_b["usincflg"], countries)]
    pairs_b = pairs_b[us_ok]
    found_b = summarize(pairs_b, "B: name search")
    print(f"2B. Name-search pass: {found_b['permno'].nunique()} more stocks matched")

fallback = pd.concat([found_a, found_b], ignore_index=True)
fallback = fallback[~fallback["entity_name"].isin(MANUAL_EXCLUDE)]
fallback.to_csv(PIT / "rp_entity_map_fallback.csv", index=False)

print("\nCheck that the names agree (CRSP name vs RavenPack name):")
print(fallback[["match", "ticker", "crsp_name", "entity_name", "valid_from", "valid_to"]]
      .assign(valid_from=lambda d: d["valid_from"].dt.date, valid_to=lambda d: d["valid_to"].dt.date)
      .sort_values(["match", "ticker"]).to_string(index=False))
still = sorted(set(unmatched) - set(fallback["permno"]))
if still:
    last = names[names["permno"].isin(still)].sort_values("ending").groupby("permno").tail(1)
    print(f"\nStill unmatched ({len(still)}; no news for these):", last[["ticker", "name"]].values.tolist())

# ----------------------------------------------------------------------------
# 3. Daily news for matched entities not already pulled
# ----------------------------------------------------------------------------
new_entities = sorted(set(fallback["rp_entity_id"]) - set(rp_map["rp_entity_id"]))
print(f"\n3. Entities to pull news for: {len(new_entities)}")
out = PIT / "news_daily_fallback.csv.gz"
out.unlink(missing_ok=True)
if new_entities:
    first = True
    for year in range(int(HISTORY_START[:4]), int(END[:4]) + 1):
        agg = db.raw_sql(daily_news_sql(year, sql_list(new_entities), sql_list(PRICE_GROUPS)), date_cols=["eff_date"])
        agg.to_csv(out, mode="a", header=first, index=False, compression="gzip")
        first = False
        print(f"   RavenPack {year}: {len(agg):,} entity-days, {int(agg['n_all'].sum()):,} stories")

db.close()
print(f"\nSaved to {PIT}")
