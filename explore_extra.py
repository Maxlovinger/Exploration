"""Check which EXTRA data sources this WRDS account can access, for new feature blocks and graph edges.

Prints ONLY library / table / column names, approximate table sizes, and date ranges (no data rows), so the
output is safe to share. Table sizes come from the database catalogue (pg_class), never by scanning, so
huge tables don't stall the script. Run in your own terminal:

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/explore_extra.py

What we are looking for (feature blocks, then graph edges):
  1. Options (OptionMetrics): implied-volatility surface, standardized options, option volume, CRSP link
  2. Analysts (I/B/E/S): consensus forecasts, forecast revisions, recommendations, CRSP link
  3. Short interest (Compustat) and securities lending (Markit)
  4. Institutional ownership (Thomson 13F)
  5. Insider trading (Thomson insiders)
  6. Customer-supplier links (Compustat segments; FactSet Revere) -> graph EDGES
  7. Events and text: Capital IQ key developments, earnings-call transcripts, SEC filings text
  8. Extras already in CRSP: bid/ask (spread) in the daily stock file
"""

import os
import re
import time
from collections import defaultdict

import wrds

username = os.environ.get("WRDS_USERNAME")
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()
LIBS = set(db.list_libraries())
found = {}                                    # category -> list of (library.table) found
t_start = time.time()


def libs_like(*keys):
    return sorted(l for l in LIBS if any(k in l.lower() for k in keys))


def tables(lib):
    try:
        return sorted(db.list_tables(library=lib))
    except Exception as e:
        print(f"   {lib}: cannot list tables ({type(e).__name__})")
        return []


def summarize(tabs, show=40):
    """Collapse yearly tables (vsurfd2015 ... vsurfd2023) into one line each."""
    groups = defaultdict(list)
    for t in tabs:
        m = re.fullmatch(r"(.*?)(\d{4})", t)
        groups[m.group(1) if m else t].append(m.group(2) if m else "")
    lines = []
    for base, years in sorted(groups.items()):
        years = sorted(y for y in years if y)
        lines.append(f"{base}{{year}} ({years[0]}-{years[-1]})" if years else base)
    print("   " + ", ".join(lines[:show]) + (f", ... (+{len(lines) - show} more)" if len(lines) > show else ""))


def rows(lib, tab):
    """Approximate row count from the catalogue (no scan)."""
    try:
        r = db.raw_sql(f"SELECT c.reltuples::bigint AS n FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                       f"WHERE n.nspname = '{lib}' AND c.relname = '{tab}'")
        return int(r["n"].iloc[0]) if len(r) else None
    except Exception:
        return None


def accessible(lib, tab):
    try:
        db.raw_sql(f"SELECT * FROM {lib}.{tab} LIMIT 1")
        return True
    except Exception:
        return False


def describe(lib, tab, keep=None, cat=None):
    """Columns (optionally only those matching `keep`), size, and whether we can read it."""
    try:
        cols = db.describe_table(library=lib, table=tab)["name"].str.lower().tolist()
    except Exception as e:
        print(f"   {lib}.{tab}: not available ({type(e).__name__})")
        return None
    ok = accessible(lib, tab)
    n = rows(lib, tab)
    shown = [c for c in cols if keep is None or re.search(keep, c)]
    print(f"\n   {lib}.{tab}: {'READABLE' if ok else 'NO READ PERMISSION'}; ~{n:,} rows" if n else
          f"\n   {lib}.{tab}: {'READABLE' if ok else 'NO READ PERMISSION'}")
    print(f"     columns ({len(cols)}): " + ", ".join(shown[:60]) + (" ..." if len(shown) > 60 else ""))
    if ok and cat:
        found.setdefault(cat, []).append(f"{lib}.{tab}")
    return cols if ok else None


def date_range(lib, tab, col, max_rows=5_000_000):
    """Min / max of a date column; skipped for big tables (a full scan could take very long)."""
    n = rows(lib, tab)
    if n is None or n > max_rows:
        print(f"     {col}: range skipped (large table: ~{n:,} rows)" if n else f"     {col}: range skipped (size unknown)")
        return
    try:
        r = db.raw_sql(f"SELECT MIN({col}) AS a, MAX({col}) AS b FROM {lib}.{tab}")
        print(f"     {col}: {r['a'].iloc[0]} to {r['b'].iloc[0]}")
    except Exception as e:
        print(f"     {col}: range not available ({type(e).__name__})")


def first_match(lib, patterns):
    tabs = tables(lib)
    for p in patterns:
        m = [t for t in tabs if re.fullmatch(p, t)]
        if m:
            return m
    return []


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# ----------------------------------------------------------------------------
section("1. OPTIONS (OptionMetrics)")
opt_libs = libs_like("optionm")
print("libraries:", opt_libs or "(none: no OptionMetrics access)")
for lib in opt_libs:
    tabs = tables(lib)
    print(f"\n-- {lib}: {len(tabs)} tables")
    summarize(tabs)
lib = "optionm" if "optionm" in LIBS else (opt_libs[0] if opt_libs else None)
if lib:
    for p, keep, cat in [(r"vsurfd\d{4}", r"secid|date|days|delta|impl|cp_flag", "options: IV surface"),
                         (r"stdopd\d{4}", r"secid|date|days|impl|cp_flag|forward|premium", "options: standardized"),
                         (r"opvold\d{4}", r"secid|date|cp_flag|volume|open_interest", "options: volume by stock"),
                         (r"opprcd\d{4}", r"secid|date|cp_flag|volume|open_interest|impl|delta|strike|exdate", "options: prices"),
                         (r"secnmd", None, "options: names"),
                         (r"securd", None, "options: security info")]:
        m = first_match(lib, [p])
        if m:
            years = [t for t in m if re.search(r"\d{4}", t)]
            print(f"\n   {p}: {len(m)} tables" + (f" ({years[0][-4:]}-{years[-1][-4:]})" if years else ""))
            if describe(lib, m[-1], keep, cat) is not None and "secnmd" not in p and "securd" not in p:
                date_range(lib, m[-1], "date")
print("\n-- CRSP <-> OptionMetrics link (secid <-> permno)")
for l in libs_like("wrdsapps", "link"):
    for t in tables(l):
        if "opcrsp" in t or ("optionm" in t and "crsp" in t):
            describe(l, t, None, "options: CRSP link")

# ----------------------------------------------------------------------------
section("2. ANALYSTS (I/B/E/S)")
ibes_libs = libs_like("ibes")
print("libraries:", ibes_libs or "(none: no I/B/E/S access)")
for lib in ibes_libs:
    tabs = tables(lib)
    print(f"\n-- {lib}: {len(tabs)} tables")
    summarize(tabs, show=60)
lib = "ibes" if "ibes" in LIBS else (ibes_libs[0] if ibes_libs else None)
if lib:
    for t, keep, cat, dcol in [("statsum_epsus", r"ticker|cusip|statpers|fpedats|fpi|measure|numest|numup|numdown|meanest|medest|stdev|actual", "analysts: consensus EPS", "statpers"),
                               ("det_epsus", r"ticker|cusip|analys|estimator|anndats|fpedats|fpi|value|revdats", "analysts: individual forecasts (EDGES: shared analysts)", None),
                               ("recddet", r"ticker|cusip|analys|estimid|anndats|ireccd|itext|revdats", "analysts: recommendations (detail)", None),
                               ("recdsum", r"ticker|cusip|statpers|meanrec|medrec|numrec|numup|numdown", "analysts: recommendations (consensus)", "statpers"),
                               ("act_epsus", r"ticker|cusip|anndats|pends|value", "analysts: actual EPS", None),
                               ("surpsum", r"ticker|anndats|surp|actual|suescore", "analysts: earnings surprise", None),
                               ("idsum", r"ticker|cusip|cname|sdates", "analysts: identifiers", None)]:
        if t in tables(lib):
            if describe(lib, t, keep, cat) is not None and dcol:
                date_range(lib, t, dcol)
print("\n-- CRSP <-> I/B/E/S link (ICLINK)")
for l in libs_like("wrdsapps", "link"):
    for t in tables(l):
        if "ibcrsp" in t or ("ibes" in t and "crsp" in t):
            describe(l, t, None, "analysts: CRSP link")

# ----------------------------------------------------------------------------
section("3. SHORT INTEREST (Compustat) and SECURITIES LENDING (Markit)")
for l in ["comp", "comp_na_daily_all"]:
    if l in LIBS:
        for t in tables(l):
            if "short" in t:
                if describe(l, t, None, "short interest") is not None:
                    date_range(l, t, "datadate")
mk = libs_like("markit")
print("\nMarkit libraries:", mk or "(none)")
for lib in mk:
    tabs = tables(lib)
    print(f"-- {lib}: {len(tabs)} tables")
    summarize(tabs)
    for t in tabs[:3]:
        describe(lib, t, r"date|cusip|isin|ticker|util|loan|fee|short|lend|borrow|dcbs", "securities lending")

# ----------------------------------------------------------------------------
section("4. INSTITUTIONAL OWNERSHIP (Thomson 13F)")
own_libs = libs_like("tfn", "tr_13f", "thomson", "13f")
print("libraries:", own_libs or "(none)")
for lib in own_libs:
    tabs = tables(lib)
    print(f"\n-- {lib}: {len(tabs)} tables")
    summarize(tabs)
    for t in tabs:
        if t in ("s34", "s34type1", "s34type3", "s12", "holdings"):
            describe(lib, t, r"fdate|rdate|cusip|mgrno|shares|shrout|prc|change|ticker", "institutional ownership")

# ----------------------------------------------------------------------------
section("5. INSIDER TRADING (Thomson insiders)")
ins_libs = libs_like("insider", "tfn", "tr_insiders")
for lib in ins_libs:
    for t in tables(lib):
        if t in ("table1", "table2", "insider1", "insider_table1") or "insider" in t:
            describe(lib, t, r"cusip|ticker|personid|role|trandate|fdate|acqdisp|trancode|shares|tprice|cleanse", "insider trading")

# ----------------------------------------------------------------------------
section("6. CUSTOMER-SUPPLIER LINKS (graph EDGES)")
seg_libs = libs_like("compseg", "segment", "comp_segments")
print("Compustat segments libraries:", seg_libs or "(none found by name)")
for lib in seg_libs + [l for l in ["comp"] if l in LIBS]:
    for t in tables(lib):
        if "cust" in t or (lib in seg_libs and t.startswith("seg")):
            describe(lib, t, None, "customer-supplier links")
rev = libs_like("factset", "revere")
print("\nFactSet / Revere libraries:", rev or "(none)")
for lib in rev:
    tabs = tables(lib)
    print(f"-- {lib}: {len(tabs)} tables")
    summarize(tabs)
    for t in tabs:
        if any(k in t for k in ["supply", "relation", "rbics", "ent_scr", "customer"]):
            describe(lib, t, None, "supply-chain links (FactSet)")

# ----------------------------------------------------------------------------
section("7. EVENTS AND TEXT (Capital IQ, transcripts, SEC filings)")
ciq = libs_like("ciq")
print("Capital IQ libraries:", ciq or "(none)")
for lib in ciq:
    tabs = tables(lib)
    print(f"-- {lib}: {len(tabs)} tables")
    summarize(tabs, show=30)
    for t in tabs:
        if t in ("wrds_keydev", "ciqkeydev", "wrds_transcript_detail", "ciqtranscript", "wrds_transcript_person"):
            describe(lib, t, None, "events / transcripts (Capital IQ)")
sec = libs_like("wrdssec", "secsamp", "sec_")
print("\nSEC filings libraries:", sec or "(none)")
for lib in sec:
    tabs = tables(lib)
    print(f"-- {lib}: {len(tabs)} tables")
    summarize(tabs, show=30)

# ----------------------------------------------------------------------------
section("8. EXTRAS ALREADY IN CRSP (bid / ask for spreads)")
try:
    cols = db.describe_table(library="crsp", table="dsf_v2")["name"].str.lower().tolist()
    print("   crsp.dsf_v2 bid/ask-type columns:", [c for c in cols if re.search(r"bid|ask|spread|numtrd|close|hilo", c)])
    found.setdefault("bid/ask spread (CRSP)", []).append("crsp.dsf_v2")
except Exception as e:
    print("   crsp.dsf_v2:", type(e).__name__)

# ----------------------------------------------------------------------------
section("SUMMARY: readable sources")
for cat, where in found.items():
    print(f"   {cat:55s} {', '.join(sorted(set(where)))[:150]}")
missing = [c for c in ["options: IV surface", "analysts: consensus EPS", "short interest", "institutional ownership",
                       "insider trading", "customer-supplier links", "events / transcripts (Capital IQ)"]
           if c not in found]
print("\n   Not found / not readable:", missing or "none")
print(f"\nDone in {time.time() - t_start:.0f}s")
db.close()
