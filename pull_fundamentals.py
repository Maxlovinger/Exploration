"""Pull company fundamentals (Compustat) and CEO data (ExecuComp) for the point-in-time stocks.

Run after pull_pit.py (it uses the CRSP-Compustat links in data/pit/ccm_links.csv):

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/pull_fundamentals.py

Downloads raw items only, each with the date it became PUBLIC (building features is a separate step):
    fundq.csv            quarterly financials + report date (rdq), standard Compustat filters
    company_cik.csv      gvkey -> SEC CIK (to find proxy filing dates)
    execucomp_ceo.csv    one row per (company, fiscal year): the CEO's pay, ownership, start/leave dates
    proxy_dates.csv      DEF 14A (proxy statement) filing dates per CIK: when CEO pay became public

Outputs in CoreFragment/data/pit/fundamentals/
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import wrds

START, END = "2001-01-01", "2025-12-31"     # from 2001: 8 quarters of history for earnings surprise by 2003
PIT = Path(__file__).resolve().parent / "data" / "pit"
OUT = PIT / "fundamentals"
OUT.mkdir(parents=True, exist_ok=True)

FUNDQ_COLS = ["gvkey", "datadate", "rdq", "fyearq", "fqtr", "fyr",
              "atq", "ltq", "ceqq", "seqq", "cheq", "dlttq", "dlcq", "actq", "lctq",
              "saleq", "revtq", "cogsq", "xsgaq", "oiadpq", "niq", "ibq", "epspxq", "cshoq", "prccq"]
CEO_COLS = ["gvkey", "year", "execid", "exec_fullname", "tdc1", "salary", "bonus", "stock_awards_fv",
            "option_awards_fv", "shrown_tot_pct", "becameceo", "leftofc", "age"]


def sql_list(values):
    for v in values:
        if not re.fullmatch(r"[A-Za-z0-9.\-]+", str(v)):
            sys.exit(f"Unsafe value in SQL list: {v!r}")
    return ", ".join(f"'{v}'" for v in values)


ccm = pd.read_csv(PIT / "ccm_links.csv", dtype={"gvkey": str})
GVKEYS = sorted(ccm["gvkey"].dropna().unique())
print(f"Companies (gvkeys) linked to our {ccm['permno'].nunique()} stocks: {len(GVKEYS)}")

username = os.environ.get("WRDS_USERNAME")
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()

# ----------------------------------------------------------------------------
# 1. Quarterly fundamentals (standard filters: industrial format, standardized, domestic, consolidated)
# ----------------------------------------------------------------------------
fundq = db.raw_sql(f"""
    SELECT {", ".join(FUNDQ_COLS)} FROM comp.fundq
    WHERE gvkey IN ({sql_list(GVKEYS)}) AND datadate BETWEEN '{START}' AND '{END}'
      AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
""", date_cols=["datadate", "rdq"])
fundq.to_csv(OUT / "fundq.csv", index=False)
lag = (fundq["rdq"] - fundq["datadate"]).dt.days
print(f"1. fundq: {len(fundq):,} company-quarters, {fundq['gvkey'].nunique()} companies; "
      f"report date present: {fundq['rdq'].notna().mean():.1%}; days quarter-end -> report: median {lag.median():.0f}")

# ----------------------------------------------------------------------------
# 2. CIKs (for SEC filing dates)
# ----------------------------------------------------------------------------
cik = db.raw_sql(f"SELECT gvkey, cik, conm FROM comp.company WHERE gvkey IN ({sql_list(GVKEYS)})")
cik.to_csv(OUT / "company_cik.csv", index=False)
print(f"2. CIKs: {cik['cik'].notna().sum()} of {len(cik)} companies")

# ----------------------------------------------------------------------------
# 3. ExecuComp: the CEO of each company-year
# ----------------------------------------------------------------------------
ceo = db.raw_sql(f"""
    SELECT {", ".join(CEO_COLS)} FROM comp_execucomp.anncomp
    WHERE gvkey IN ({sql_list(GVKEYS)}) AND year BETWEEN {int(START[:4])} AND {int(END[:4])} AND ceoann = 'CEO'
""", date_cols=["becameceo", "leftofc"])
ceo.to_csv(OUT / "execucomp_ceo.csv", index=False)
print(f"3. ExecuComp CEOs: {len(ceo):,} company-years, {ceo['gvkey'].nunique()} companies, "
      f"years {int(ceo['year'].min())}-{int(ceo['year'].max())}")

# ----------------------------------------------------------------------------
# 4. Proxy statement (DEF 14A) filing dates: when CEO pay became public
# ----------------------------------------------------------------------------
ciks = sorted({int(c) for c in cik["cik"].dropna() if str(c).strip().isdigit()})   # numbers (no leading zeros)
parts = []
for i in range(0, len(ciks), 500):
    parts.append(db.raw_sql(f"""
        SELECT DISTINCT CAST(cik AS bigint) AS cik, fdate FROM wrdssec.dforms
        WHERE form = 'DEF 14A' AND fdate BETWEEN '{START}' AND '2026-12-31'
          AND CAST(cik AS bigint) IN ({", ".join(str(c) for c in ciks[i:i + 500])})
    """, date_cols=["fdate"]))
proxy = pd.concat(parts, ignore_index=True)
proxy.to_csv(OUT / "proxy_dates.csv", index=False)
print(f"4. Proxy (DEF 14A) filings: {len(proxy):,} for {proxy['cik'].nunique()} of {len(ciks)} CIKs")

db.close()
print(f"\nSaved to {OUT}")
