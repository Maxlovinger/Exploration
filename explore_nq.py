"""Details on the Nasdaq-100 E-mini (NQ, Datastream CEN) and Micro E-mini (MNQ, Datastream CMQ) futures.

Prints ONLY descriptions, codes, column names, and date coverage (no prices), so the output is safe to share.

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/explore_nq.py
"""

import os

import pandas as pd
import wrds

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 70)
LIB = "tr_ds_fut"
CONTRACTS = ["CEN", "CMQ"]          # Datastream contract IDs: CEN = E-mini Nasdaq-100 (NQ), CMQ = Micro (MNQ)
IDS = ", ".join(f"'{c}'" for c in CONTRACTS)

username = os.environ.get("WRDS_USERNAME")
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()

# 1. The contract classes
contr = db.raw_sql(f"SELECT contrcode, dscontrid, contrname, exchtickersymb FROM {LIB}.dsfutcontr WHERE dscontrid IN ({IDS})")
print("######## 1. Contracts ########")
print(contr.to_string(index=False))
codes = ", ".join(str(int(c)) for c in contr["contrcode"])

# 2. Individual contracts (one per expiry month): how many, and from when to when
info = db.raw_sql(f"""
    SELECT contrcode, count(*) AS contracts, min(startdate) AS first_start, max(lasttrddate) AS last_trade,
           min(dsmnem) AS example_code_first, max(dsmnem) AS example_code_last
    FROM {LIB}.wrds_contract_info WHERE contrcode IN ({codes}) GROUP BY contrcode
""")
print("\n######## 2. Individual contracts (expiry months) ########")
print(info.to_string(index=False))

# 3. Continuous series (stitched front-month series), by Datastream code prefix
series = db.raw_sql(f"""
    SELECT calcseriescode, dsmnem, calcseriesname, rollmethoddesc
    FROM {LIB}.wrds_cseries_info WHERE substring(dsmnem, 1, 3) IN ({IDS}) ORDER BY dsmnem
""")
print("\n######## 3. Continuous series ########")
print(series.to_string(index=False) if not series.empty else "  none found")

# 4. What the daily value tables contain (column names only)
print("\n######## 4. Daily value tables (columns) ########")
for t in ["wrds_fut_series", "wrds_fut_contract"]:
    cols = db.describe_table(library=LIB, table=t)["name"].tolist()
    print(f"  {t}: {cols}")

# 5. Date coverage of the main continuous series (one query per series; filtered by its code)
print("\n######## 5. Coverage of each continuous series (dates and counts only) ########")
date_col = next((c for c in db.describe_table(library=LIB, table="wrds_fut_series")["name"] if "date" in c.lower()), None)
for _, r in series.head(8).iterrows():
    try:
        cov = db.raw_sql(f"""SELECT min({date_col}) AS first, max({date_col}) AS last, count(*) AS rows
                             FROM {LIB}.wrds_fut_series WHERE calcseriescode = {int(r['calcseriescode'])}""")
        print(f"  {r['dsmnem']:<10} {r['calcseriesname'][:45]:<45} {cov.iloc[0].to_dict()}")
    except Exception as e:
        print(f"  {r['dsmnem']}: coverage query failed ({type(e).__name__})")

db.close()
