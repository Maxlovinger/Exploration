"""Check whether WRDS gives you Nasdaq-100 futures (NQ = E-mini, MNQ = Micro E-mini).

Prints ONLY library/table/column names and contract DESCRIPTIONS (names, codes, dates),
never prices or other data rows, so the output is safe to share.

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/explore_futures.py

Futures on WRDS usually come from one of these subscriptions:
    Refinitiv Datastream futures   daily, individual contracts + continuous series
    TickData                       intraday (tick / minute) futures
    OptionMetrics IvyDB Futures    daily futures (and options on futures)
Also checks CRSP for QQQ (the Nasdaq-100 ETF), a daily stand-in that is already in your data source.
"""

import os
import re

import wrds

username = os.environ.get("WRDS_USERNAME")  # optional: skips the username prompt
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()
libraries = sorted(db.list_libraries())

LIB_PATTERN = r"fut|tr_ds|^ds|datastream|refinitiv|tick|optionm|ivy|cme|trth"
TABLE_PATTERN = r"fut|contr|series|root|instrument|symbol|sec"
INFO_PATTERN = r"info|contr|series|desc|name|nmd|securd|root|master|ref|list|lookup"   # small metadata tables
DATA_PATTERN = r"prc|price|val|quote|trade|tick|daily|intraday|vol"                 # big data tables: skip
NAME_COLS = r"name|desc|contr|title|instr|underl"                                 # text columns describing contracts
CODE_COLS = r"ticker|symbol|root|code|mnem|ric|exchsym|sym"                       # short contract codes
NASDAQ_TEXT = "nasdaq.?100|nasdaq 100|e-?mini nasdaq|micro e-?mini nasdaq"
NQ_CODES = "^(NQ|MNQ|ENQ)"
MAX_ROWS = 5_000_000          # only search tables smaller than this (description tables, not daily prices)


def est_rows(lib, table):
    """Approximate row count from Postgres statistics (instant; no table scan)."""
    try:
        r = db.raw_sql(f"""SELECT c.reltuples::bigint AS n FROM pg_class c JOIN pg_namespace s ON s.oid = c.relnamespace
                           WHERE s.nspname = '{lib}' AND c.relname = '{table}'""")
        return int(r["n"].iloc[0]) if len(r) else None
    except Exception:
        return None

# ----------------------------------------------------------------------------
# 1. Which libraries could hold futures?
# ----------------------------------------------------------------------------
fut_libs = [lib for lib in libraries if re.search(LIB_PATTERN, lib, re.I)]
print("######## 1. Libraries that may hold futures ########")
print(fut_libs or "(none found: you may not have a futures subscription on WRDS)")

# ----------------------------------------------------------------------------
# 2. Futures-related tables and their columns
# ----------------------------------------------------------------------------
print("\n######## 2. Futures-related tables ########")
candidates = []
for lib in fut_libs:
    try:
        tables = sorted(db.list_tables(library=lib))
    except Exception as e:
        print(f"--- {lib}: cannot list tables ({type(e).__name__}): probably no access")
        continue
    hits = [t for t in tables if re.search(TABLE_PATTERN, t, re.I)]
    print(f"--- {lib}: {len(tables)} tables")
    for t in hits[:60]:
        n = est_rows(lib, t)
        size = f"{n:,} rows" if n is not None and n >= 0 else "size unknown"
        small = n is not None and 0 <= n < MAX_ROWS
        print(f"      {t:<40} {size:>20}{'' if small else '   (too big to search: skipped)'}")
        if small and re.search(INFO_PATTERN, t, re.I) and not re.search(DATA_PATTERN, t, re.I):
            candidates.append((lib, t))

# ----------------------------------------------------------------------------
# 3. Search contract descriptions for the Nasdaq-100 E-mini / Micro E-mini
# ----------------------------------------------------------------------------
print("\n######## 3. Nasdaq-100 futures contracts found (descriptions only) ########")
found_any = False
for lib, t in candidates[:40]:
    try:
        cols = db.describe_table(library=lib, table=t)[["name", "type"]]
    except Exception as e:
        print(f"  {lib}.{t}: cannot describe ({type(e).__name__})")
        continue
    print(f"  searching {lib}.{t} ...", flush=True)
    names = [c for c, ty in zip(cols["name"], cols["type"].astype(str)) if re.search(r"char|text|str", ty, re.I)]
    text_cols = [c for c in names if re.search(NAME_COLS, c, re.I)]
    code_cols = [c for c in names if re.search(CODE_COLS, c, re.I)]
    date_cols = [c for c, ty in zip(cols["name"], cols["type"].astype(str)) if re.search(r"date|time", ty, re.I)][:4]
    conds = [f"{c} ~* '{NASDAQ_TEXT}'" for c in text_cols] + [f"upper({c}) ~ '{NQ_CODES}'" for c in code_cols]
    if not conds:
        continue
    show = list(dict.fromkeys(text_cols[:3] + code_cols[:3] + date_cols))
    try:
        rows = db.raw_sql(f"SELECT DISTINCT {', '.join(show)} FROM {lib}.{t} WHERE {' OR '.join(conds)} LIMIT 25")
    except Exception as e:
        print(f"  {lib}.{t}: search failed ({type(e).__name__})")
        continue
    print(f"\n  {lib}.{t}  (columns: {', '.join(cols['name'])})")
    if rows.empty:
        print("    no Nasdaq-100 contracts")
    else:
        found_any = True
        print(rows.to_string(index=False, max_colwidth=60))
if not found_any:
    print("\n  No Nasdaq-100 (NQ / MNQ) contract descriptions found in the tables above.")

# ----------------------------------------------------------------------------
# 4. Daily stand-in already available: QQQ in CRSP
# ----------------------------------------------------------------------------
print("\n######## 4. QQQ (Nasdaq-100 ETF) in CRSP ########")
try:
    q = db.raw_sql("""
        SELECT DISTINCT permno, ticker, issuernm, namedt, nameenddt FROM crsp.stocknames_v2
        WHERE ticker = 'QQQ' ORDER BY namedt
    """)
    print(q.to_string(index=False) if not q.empty else "  QQQ not found in crsp.stocknames_v2")
    if not q.empty:
        p = int(q["permno"].iloc[-1])
        cov = db.raw_sql(f"SELECT min(dlycaldt) AS first, max(dlycaldt) AS last, count(*) AS days FROM crsp.dsf_v2 WHERE permno = {p}")
        print("  daily returns coverage:", cov.iloc[0].to_dict())
except Exception as e:
    print(f"  CRSP check failed ({type(e).__name__})")

db.close()
