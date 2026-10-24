"""Explore the WRDS tables needed for the point-in-time (PIT) universe.

Prints ONLY library, table, and column names, plus the date COVERAGE of a few key
tables (first/last date and row counts, no data rows), so the output is safe to share.

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/explore_pit.py

What we are looking for:
    1. S&P 500 membership history (which PERMNOs were in the index, and when)
    2. New-format (CIZ) CRSP names history (tickers / CUSIPs over time), since the
       legacy crsp.stocknames stopped updating after Dec 2024
    3. Delisting returns (e.g., Lehman's final return)
    4. CRSP-Compustat link (PERMNO <-> GVKEY)
    5. Historical GICS sectors (Compustat)
    6. Daily stock file columns (market cap, price, shares) and coverage
    7. RavenPack mapping columns (are CUSIP links dated?)
"""

import os
import re

import wrds

username = os.environ.get("WRDS_USERNAME")  # optional: skips the username prompt
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()
libraries = set(db.list_libraries())


def tables_matching(lib, pattern):
    if lib not in libraries:
        return []
    return sorted(t for t in db.list_tables(library=lib) if re.search(pattern, t, re.I))


def show_columns(lib, table):
    try:
        cols = db.describe_table(library=lib, table=table)[["name", "type"]]
        print(f"\n=== {lib}.{table} ===")
        print(", ".join(f"{n} ({str(t).split('(')[0]})" for n, t in zip(cols["name"], cols["type"])))
    except Exception as e:
        print(f"\n=== {lib}.{table}: not available ({type(e).__name__}) ===")


def coverage(label, sql):
    try:
        row = db.raw_sql(sql).iloc[0]
        print(f"  {label}: " + ", ".join(f"{k} = {v}" for k, v in row.items()))
    except Exception as e:
        print(f"  {label}: query failed ({type(e).__name__})")


crsp_libs = sorted(lib for lib in libraries if lib.startswith("crsp"))
print("CRSP libraries:", crsp_libs)

print("\n######## 1. S&P 500 membership ########")
for lib in crsp_libs:
    for t in tables_matching(lib, r"sp500|500list|indexcon|constit"):
        show_columns(lib, t)

print("\n######## 2. CRSP names history (legacy and CIZ) ########")
for lib in ["crsp"]:
    for t in tables_matching(lib, r"stocknames|securityinfo|secinfo|names_v2|stknames"):
        show_columns(lib, t)

print("\n######## 3. Delisting ########")
for lib in ["crsp"]:
    for t in tables_matching(lib, r"delist"):
        show_columns(lib, t)

print("\n######## 4. CRSP-Compustat link ########")
for lib in sorted({"crsp", "crsp_a_ccm"} & libraries):
    for t in tables_matching(lib, r"ccm|lnkhist|linktable"):
        show_columns(lib, t)

print("\n######## 5. Historical GICS (Compustat) ########")
for t in tables_matching("comp", r"^co_hgic$|^company$"):
    show_columns("comp", t)

print("\n######## 6. Daily stock file ########")
show_columns("crsp", "dsf_v2")

print("\n######## 7. RavenPack mapping ########")
show_columns("ravenpack_common", "wrds_rpa_all_mappings")

# ---------------------------------------------------------------------------
# Coverage (dates and counts only)
# ---------------------------------------------------------------------------
print("\n######## Coverage (dates and counts only) ########")
coverage("dsf_v2 (AAPL, permno 14593)",
         "SELECT min(dlycaldt) AS first, max(dlycaldt) AS last FROM crsp.dsf_v2 WHERE permno = 14593")
coverage("stocknames (legacy)", "SELECT max(nameenddt) AS last FROM crsp.stocknames")
for lib in crsp_libs:
    for t in tables_matching(lib, r"sp500|500list"):
        cols = db.describe_table(library=lib, table=t)["name"].str.lower().tolist()
        start = next((c for c in cols if c in ("start", "mbrstartdt", "startdt")), None)
        end = next((c for c in cols if c in ("ending", "mbrenddt", "enddt", "end")), None)
        if start and end and "permno" in cols:
            coverage(f"{lib}.{t}", f"SELECT min({start}) AS first_start, max({end}) AS last_end, "
                                   f"count(*) AS spells, count(DISTINCT permno) AS stocks FROM {lib}.{t}")
        else:
            print(f"  {lib}.{t}: columns {cols}")
for t in tables_matching("crsp", r"delist"):
    cols = db.describe_table(library="crsp", table=t)["name"].str.lower().tolist()
    dcol = next((c for c in cols if c in ("dlstdt", "delistingdt", "delistdt")), None)
    if dcol:
        coverage(f"crsp.{t}", f"SELECT min({dcol}) AS first, max({dcol}) AS last, count(*) AS rows FROM crsp.{t}")
if "co_hgic" in tables_matching("comp", r"^co_hgic$"):
    coverage("comp.co_hgic", "SELECT min(indfrom) AS first, max(indthru) AS last, count(DISTINCT gvkey) AS firms FROM comp.co_hgic")

db.close()
