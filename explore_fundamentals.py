"""Explore the WRDS tables for company fundamentals (balance sheet) and CEO data.

Prints ONLY library, table, and column names, plus date COVERAGE (first/last dates and counts,
no data rows), so the output is safe to share.

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/explore_fundamentals.py

What we are looking for:
    1. Compustat quarterly fundamentals (comp.fundq): balance sheet / income items, and the
       REPORT DATE (rdq) = when the numbers became public (needed to avoid look-ahead)
    2. ExecuComp (CEO pay, tenure, CEO changes, ownership)
    3. SEC filing dates (e.g., proxy statements, DEF 14A), to know when CEO pay became public
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


def show_columns(lib, table, keep=None):
    """Print column names; if `keep` is given, only those matching it (plus a count of the rest)."""
    try:
        cols = db.describe_table(library=lib, table=table)[["name", "type"]]
    except Exception as e:
        print(f"\n=== {lib}.{table}: not available ({type(e).__name__}) ===")
        return []
    names = list(cols["name"].str.lower())
    print(f"\n=== {lib}.{table} ({len(names)} columns) ===")
    if keep:
        shown = [n for n in names if re.fullmatch(keep, n)]
        print(", ".join(shown) + (f"  (+ {len(names) - len(shown)} other columns)" if len(names) > len(shown) else ""))
    else:
        print(", ".join(names))
    return names


def coverage(label, sql):
    try:
        row = db.raw_sql(sql).iloc[0]
        print(f"  {label}: " + ", ".join(f"{k} = {v}" for k, v in row.items()))
    except Exception as e:
        print(f"  {label}: query failed ({type(e).__name__})")


# ----------------------------------------------------------------------------
# 1. Compustat quarterly fundamentals
# ----------------------------------------------------------------------------
print("######## 1. Compustat quarterly fundamentals ########")
print("comp tables like fundq/funda:", tables_matching("comp", r"^fund[aq]$|^fundq|^funda"))
KEY_FUNDQ = (r"gvkey|datadate|rdq|fyearq|fqtr|fyr|indfmt|datafmt|popsrc|consol|curcdq|"
             r"atq|ltq|ceqq|seqq|cheq|dlttq|dlcq|actq|lctq|"            # balance sheet
             r"saleq|revtq|cogsq|xsgaq|oiadpq|niq|ibq|epspxq|"           # income statement
             r"oancfy|capxy|dvy|cshoq|prccq|pdateq|fdateq|apdedateq")   # cash flow, shares, price, dates
fundq_cols = show_columns("comp", "fundq", keep=KEY_FUNDQ)
if "rdq" in fundq_cols:
    coverage("fundq (standard filters, 2006-2025)",
             "SELECT min(datadate) AS first, max(datadate) AS last, count(DISTINCT gvkey) AS companies, "
             "count(*) AS rows, round(avg((rdq IS NOT NULL)::int), 3) AS share_with_report_date, "
             "round(avg((rdq - datadate))::numeric, 1) AS avg_days_quarter_end_to_report "
             "FROM comp.fundq WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' "
             "AND datadate BETWEEN '2003-01-01' AND '2025-12-31'")

# ----------------------------------------------------------------------------
# 2. ExecuComp (CEO data)
# ----------------------------------------------------------------------------
print("\n######## 2. ExecuComp ########")
exec_libs = sorted(lib for lib in libraries if "execucomp" in lib.lower() or lib.lower().startswith("comp_exec"))
print("ExecuComp libraries:", exec_libs or "(none found: you may not have ExecuComp access)")
for lib in exec_libs:
    tables = db.list_tables(library=lib)
    print(f"--- {lib}: {sorted(tables)}")
    for t in [t for t in tables if t.lower() in ("anncomp", "person", "codirfin", "coperol", "stgrttab")]:
        show_columns(lib, t)
    if "anncomp" in [t.lower() for t in tables]:
        coverage(f"{lib}.anncomp", f"SELECT min(year) AS first_year, max(year) AS last_year, count(DISTINCT gvkey) AS companies, "
                                   f"count(*) AS rows FROM {lib}.anncomp")
        coverage(f"{lib}.anncomp CEOs (ceoann = 'CEO'), 2006+",
                 f"SELECT count(*) AS ceo_rows, count(DISTINCT gvkey) AS companies FROM {lib}.anncomp "
                 f"WHERE ceoann = 'CEO' AND year >= 2006")

# ----------------------------------------------------------------------------
# 3. SEC filing dates (to date CEO pay by when the proxy statement was filed)
# ----------------------------------------------------------------------------
print("\n######## 3. SEC filings (filing dates) ########")
sec_libs = sorted(lib for lib in libraries if lib.lower().startswith("wrdssec") or lib.lower() in ("sec", "secsamp"))
print("SEC libraries:", sec_libs or "(none found)")
for lib in sec_libs:
    found = tables_matching(lib, r"forms|filings|index|wrds_forms")
    print(f"--- {lib}: tables like forms/filings/index: {found[:15]}")
    for t in found[:2]:
        show_columns(lib, t)

# Compustat-CRSP link already pulled (crsp.ccmxpf_lnkhist), so fundamentals join to our stocks by gvkey.
db.close()
