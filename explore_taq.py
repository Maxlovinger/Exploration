"""Check whether you have TAQ (NYSE Trade and Quote: every trade and quote, to the millisecond).

Prints ONLY library/table/column names and date coverage, plus a yes/no for whether QQQ appears
on one recent day (no prices or trade data), so the output is safe to share.

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/explore_taq.py

TAQ covers US stocks and ETFs (including QQQ), not futures. On WRDS, daily millisecond TAQ is usually
in the library "taqmsec" with one table per trading day, e.g. ctm_20241231 (trades), cqm_20241231 (quotes).
"""

import os
import re

import wrds

username = os.environ.get("WRDS_USERNAME")
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()
libraries = sorted(db.list_libraries())

taq_libs = [lib for lib in libraries if "taq" in lib.lower()]
print("######## 1. TAQ libraries ########")
print(taq_libs or "(none found: no TAQ access on WRDS)")

print("\n######## 2. Tables and date coverage ########")
trade_tables = {}
for lib in taq_libs:
    try:
        tables = sorted(db.list_tables(library=lib))
    except Exception as e:
        print(f"--- {lib}: cannot list tables ({type(e).__name__}): probably no access")
        continue
    groups = {}
    for t in tables:
        m = re.fullmatch(r"([a-z_]+?)_?(\d{8})", t)
        if m:
            groups.setdefault(m.group(1), []).append(m.group(2))
    print(f"--- {lib}: {len(tables)} tables")
    for base, dates in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"      {base}_YYYYMMDD: {len(dates):,} days, {min(dates)} to {max(dates)}")
        if base in ("ctm", "ct", "taq_trades") or base.startswith("ctm"):
            trade_tables[lib] = (base, max(dates))
    others = [t for t in tables if not re.search(r"\d{8}$", t)]
    if others:
        print(f"      other tables: {others[:15]}{' ...' if len(others) > 15 else ''}")

print("\n######## 3. Trade table columns and QQQ check (yes/no only) ########")
for lib, (base, day) in trade_tables.items():
    table = f"{base}_{day}"
    try:
        cols = db.describe_table(library=lib, table=table)["name"].tolist()
        print(f"  {lib}.{table} columns: {cols}")
        sym = next((c for c in cols if c.lower() in ("sym_root", "symbol", "sym")), None)
        tcol = next((c for c in cols if c.lower() in ("time_m", "time", "time_m_nano")), None)
        if sym:
            hit = db.raw_sql(f"SELECT 1 AS found FROM {lib}.{table} WHERE {sym} = 'QQQ' LIMIT 1")
            print(f"  QQQ trades on {day}: {'yes' if len(hit) else 'no'}"
                  + (f"  (timestamps in column '{tcol}')" if tcol else ""))
    except Exception as e:
        print(f"  {lib}.{table}: check failed ({type(e).__name__})")

db.close()
