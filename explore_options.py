"""Check which WRDS options (derivatives) tables are available for a future GDBN v2.

Prints ONLY library, table, and column names (no data rows), so the output is
safe to share. Run it in your own terminal:

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/explore_options.py

What we are looking for:
    OptionMetrics volatility surface (vsurfd{year}) or standardized options (stdopd{year})
        -> each stock's 30-day at-the-money implied volatility and put/call skew
    A CRSP <-> OptionMetrics link table (secid <-> permno)
    CBOE VIX (market-wide fear)
"""

import os
import re
from collections import defaultdict

import wrds

username = os.environ.get("WRDS_USERNAME")  # optional: skips the username prompt
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()

libraries = db.list_libraries()


def summarize_tables(tables):
    """Collapse yearly tables (e.g., vsurfd2015 ... vsurfd2024) into one line each."""
    groups = defaultdict(list)
    for t in tables:
        m = re.fullmatch(r"(.*?)(\d{4})", t)
        groups[m.group(1) if m else t].append(m.group(2) if m else "")
    for base, years in sorted(groups.items()):
        years = sorted(y for y in years if y)
        print(f"   {base}" + (f"{{year}}  ({years[0]}-{years[-1]}, {len(years)} tables)" if years else ""))


print("\n=== Options / CBOE / link libraries ===")
opt_libs = sorted(lib for lib in libraries
                  if any(k in lib.lower() for k in ["optionm", "option", "cboe", "ivy"]))
link_libs = sorted(lib for lib in libraries if "link" in lib.lower() or lib.lower() == "wrdsapps")
print("options:", opt_libs or "(none found: you may not have OptionMetrics access)")
print("links:  ", link_libs or "(none found)")

for lib in opt_libs:
    tables = db.list_tables(library=lib)
    print(f"\n--- {lib}: {len(tables)} tables ---")
    summarize_tables(tables)

# Columns of the tables we would use (most recent year)
for lib in opt_libs:
    tables = sorted(db.list_tables(library=lib))
    for keyword in ["vsurfd", "stdopd", "secnmd", "optionmnames", "cboe"]:
        match = [t for t in tables if t.lower().startswith(keyword)]
        if match:
            t = match[-1]
            print(f"\n=== Columns of {lib}.{t} ===")
            print(db.describe_table(library=lib, table=t)[["name", "type"]].to_string(index=False))

# CRSP <-> OptionMetrics link (secid <-> permno)
for lib in link_libs:
    for t in sorted(db.list_tables(library=lib)):
        if "opcrsp" in t.lower() or ("optionm" in t.lower() and "crsp" in t.lower()):
            print(f"\n=== Link table {lib}.{t} ===")
            print(db.describe_table(library=lib, table=t)[["name", "type"]].to_string(index=False))

db.close()
