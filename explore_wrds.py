"""Explore which WRDS tables are available for the GDBN data pull.

Prints ONLY library, table, and column names (no data rows), so the output is
safe to share. Run it in your own terminal (it asks for your WRDS login):

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/explore_wrds.py
"""

import wrds

db = wrds.Connection()  # prompts for WRDS username/password (and Duo), can save a .pgpass

libraries = db.list_libraries()

print("\n=== RavenPack libraries ===")
rp_libs = sorted(lib for lib in libraries if "raven" in lib.lower() or lib.lower().startswith("rpa"))
print(rp_libs or "(none found: check the library names on the WRDS website)")

for lib in rp_libs:
    tables = db.list_tables(library=lib)
    print(f"\n--- {lib}: {len(tables)} tables ---")
    for t in sorted(tables)[:40]:
        print("  ", t)
    if len(tables) > 40:
        print(f"   ... and {len(tables) - 40} more")

# Columns of one equities table and the mapping table, if they exist
for lib in rp_libs:
    tables = sorted(db.list_tables(library=lib))
    for keyword in ["equities", "mapping"]:
        match = [t for t in tables if keyword in t.lower()]
        if match:
            t = match[-1]
            print(f"\n=== Columns of {lib}.{t} ===")
            print(db.describe_table(library=lib, table=t)[["name", "type"]].to_string(index=False))

print("\n=== CRSP daily stock file ===")
for lib, table in [("crsp", "dsf"), ("crsp", "dsf_v2"), ("crsp", "stocknames")]:
    try:
        cols = db.describe_table(library=lib, table=table)["name"].tolist()
        print(f"{lib}.{table}: {cols}")
    except Exception as e:
        print(f"{lib}.{table}: not available ({type(e).__name__})")

db.close()
