"""Turn the EXTRA raw data (pull_extra.py) into daily point-in-time features and a dated edge table.

Run after build_features.py and pull_extra.py (no WRDS needed):

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/build_extra_features.py

Steps for every source:
  1. MAP its ID to permno with a link valid ON THAT DATE (secid, I/B/E/S ticker, gvkey, CUSIP, CDS RED code, CIK).
  2. DATE each value by when it became PUBLIC (publication-lag rules below).
  3. CARRY it forward to every trading day, up to a maximum age (then it becomes missing).
  4. COMPUTE the features, then LEFT-JOIN them on (date, permno) to the existing feature table.

Publication rules (usable from ...):
  options, bid/ask                 the same day (computed from that day's closing quotes)
  Markit lending, CDS              the next trading day
  short interest                   10 trading days after the settlement date
  I/B/E/S consensus / recs         the next trading day after the statistics date
  I/B/E/S surprise, insiders,      the next trading day after the announcement / filing date
  events, 10-K text
  13F holdings                     45 days after the quarter end

Outputs (test years 2023+ are NOT built):
  data/pit/built/features_daily_v2.parquet   the old 31 columns + the new ones (same rows)
  data/pit/built/edges_all.parquet           dated links between stocks (supply chain, competitors, partners,
                                             shared analysts)
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIT = Path(__file__).resolve().parent / "data" / "pit"
BUILT, EXTRA = PIT / "built", PIT / "extra"
BUILD_END, TUNE_END = "2022-12-31", "2020-12-31"
OUT_FEATS = Path(os.environ.get("FEATURES_V2_OUT", BUILT / "features_daily_v2.parquet"))
OUT_EDGES = Path(os.environ.get("EDGES_OUT", BUILT / "edges_all.parquet"))
OPEN = pd.Timestamp("2100-01-01")
END = pd.Timestamp(BUILD_END)
t0 = time.time()


def log(msg):
    print(f"{msg} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------
# 0. Calendar, existing table, helpers
# ----------------------------------------------------------------------------
days = pd.DatetimeIndex(pd.read_csv(BUILT / "trading_days.csv", parse_dates=["date"])["date"])
days = days[days <= END]
base = pd.read_csv(BUILT / "features_daily.csv.gz", parse_dates=["date"], dtype={"sector": str, "ticker": str})
num = [c for c in base.columns if c not in ("date", "permno", "ticker", "sector")]
base[num] = base[num].astype("float32")
PERM = np.sort(base["permno"].unique())
JX = pd.Index(PERM)
di, pj = days.get_indexer(base["date"]), JX.get_indexer(base["permno"])
NEW = {}
log(f"0. Existing table: {len(base):,} rows x {base.shape[1]} columns, {len(PERM)} stocks")

panel = pd.read_csv(BUILT / "panel.csv.gz", parse_dates=["date"], usecols=["permno", "date", "prc", "cap", "volume"])
panel = panel[panel["date"] <= END]


def wide_from_panel(col):
    return panel.pivot(index="date", columns="permno", values=col).reindex(index=days, columns=PERM).values


PRC, CAP, VOL = np.abs(wide_from_panel("prc")), wide_from_panel("cap"), wide_from_panel("volume")
SHARES = CAP * 1000 / np.where(PRC > 0, PRC, np.nan)                 # cap is in $ thousands
ADV20 = pd.DataFrame(VOL).rolling(20, min_periods=10).mean().values
RVOL = np.full((len(days), len(PERM)), np.nan, dtype=np.float32)
RVOL[di, pj] = base["rvol_20"].values
del panel


def read_block(name, usecols=None):
    files = sorted((EXTRA / name).glob("*.csv.gz"))
    parts = [pd.read_csv(f, usecols=usecols, low_memory=False) for f in files]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols or [])


def next_td(dates, lag=1):
    """Index of the lag-th trading day strictly AFTER each date (len(days) = beyond the build)."""
    return days.searchsorted(pd.to_datetime(dates).values, side="right") + (lag - 1)


def on_or_after(dates):
    return days.searchsorted(pd.to_datetime(dates).values, side="left")


def last_on_or_before(dates):
    return days.searchsorted(pd.to_datetime(dates).values, side="right") - 1


def at(arr, dates, permnos):
    """Value of a day x stock array on the last trading day <= each date."""
    t, j = last_on_or_before(dates), JX.get_indexer(permnos)
    ok = (t >= 0) & (j >= 0)
    out = np.full(len(t), np.nan)
    out[ok] = arr[t[ok], j[ok]]
    return out


def to_wide(df, value, t_col="t"):
    """Place values at (trading-day index, stock); later rows overwrite earlier ones."""
    arr = np.full((len(days), len(PERM)), np.nan)
    j = JX.get_indexer(df["permno"])
    ok = (df[t_col].values >= 0) & (df[t_col].values < len(days)) & (j >= 0) & np.isfinite(df[value].values)
    arr[df[t_col].values[ok], j[ok]] = df[value].values[ok]
    return arr


def ffill(arr, limit):
    return pd.DataFrame(arr).ffill(limit=limit).values


def put(name, arr):
    NEW[name] = arr[di, pj].astype("float32")


# Identifier links ------------------------------------------------------------
names = pd.read_csv(PIT / "names_history.csv", parse_dates=["start", "ending"], dtype={"cusip8": str})
names = names[names["permno"].isin(PERM)].dropna(subset=["cusip8"])
ccm = pd.read_csv(PIT / "ccm_links.csv", parse_dates=["linkdt", "linkenddt"], dtype={"gvkey": str})
ccm = ccm.dropna(subset=["permno"]).assign(permno=lambda d: d["permno"].astype(int))
ccm = ccm[ccm["permno"].isin(PERM)]


def map_dated(df, key, links, link_key, date_col, lo, hi, extra_sort=None):
    """Join df to links on key and keep links valid on df[date_col] (lo <= date <= hi)."""
    m = df.merge(links, left_on=key, right_on=link_key, how="inner", suffixes=("", "_lk"))
    m = m[(m[date_col] >= m[lo]) & (m[date_col] <= m[hi].fillna(OPEN))]
    if extra_sort:
        m = m.sort_values(extra_sort)
    return m


def map_cusip8(df, col, date_col):
    return map_dated(df, col, names[["cusip8", "permno", "start", "ending"]], "cusip8", date_col, "start", "ending")


def map_gvkey(df, date_col, primary_only=False):
    lk = ccm[ccm["linkprim"] == "P"] if primary_only else ccm
    return map_dated(df, "gvkey", lk[["gvkey", "permno", "linkdt", "linkenddt"]], "gvkey", date_col, "linkdt", "linkenddt")


def coverage(name, arr):
    put(f"has_{name}", np.isfinite(arr).astype(float))


# ----------------------------------------------------------------------------
# 1. Options (OptionMetrics): implied volatility and option volume
# ----------------------------------------------------------------------------
lk_opt = pd.read_csv(EXTRA / "links" / "optionm_crsp.csv.gz", parse_dates=["sdate", "edate"])
iv = read_block("options_iv")
iv["date"] = pd.to_datetime(iv["date"])
iv = iv[iv["date"] <= END]
iv = map_dated(iv, "secid", lk_opt, "secid", "date", "sdate", "edate", ["score", "secid"])
iv = iv.drop_duplicates(["permno", "date"], keep="first")              # best link score per stock-day
iv["t"] = on_or_after(iv["date"])
iv["atm"] = (iv["iv_c50"] + iv["iv_p50"]) / 2
iv["spread"] = iv["iv_c50"] - iv["iv_p50"]
iv["skew"] = iv["iv_p25"] - iv["iv_c50"]
IVA = ffill(to_wide(iv, "atm"), 5)
put("iv_atm", IVA)
put("iv_spread", ffill(to_wide(iv, "spread"), 5))
put("iv_skew", ffill(to_wide(iv, "skew"), 5))
put("iv_chg_5d", IVA - np.vstack([np.full((5, len(PERM)), np.nan), IVA[:-5]]))
put("iv_rv", IVA - RVOL * np.sqrt(252))
coverage("options", IVA)
del iv

ov = read_block("options_volume")
ov["date"] = pd.to_datetime(ov["date"])
ov = ov[ov["date"] <= END]
ov = map_dated(ov, "secid", lk_opt, "secid", "date", "sdate", "edate", ["score", "secid"])
ov = ov.drop_duplicates(["permno", "date"], keep="first")
ov["t"] = on_or_after(ov["date"])
CV5 = pd.DataFrame(to_wide(ov, "call_vol")).rolling(5, min_periods=3).sum().values
PV5 = pd.DataFrame(to_wide(ov, "put_vol")).rolling(5, min_periods=3).sum().values
VOL5 = pd.DataFrame(VOL).rolling(5, min_periods=3).sum().values
put("pc_volume", np.log((PV5 + 1) / (CV5 + 1)))
put("os_volume", np.log(((CV5 + PV5) * 100 + 1) / (VOL5 + 1)))
del ov, CV5, PV5
log("1. Options")

# ----------------------------------------------------------------------------
# 2. Bid-ask spread (CRSP closing quotes)
# ----------------------------------------------------------------------------
qt = read_block("crsp_quotes")
qt["date"] = pd.to_datetime(qt["date"])
qt = qt[(qt["date"] <= END) & (qt["bid"] > 0) & (qt["ask"] >= qt["bid"])]
qt["spread"] = (qt["ask"] - qt["bid"]) / ((qt["ask"] + qt["bid"]) / 2)
qt["t"] = on_or_after(qt["date"])
SPR = to_wide(qt, "spread")
put("spread", SPR)
put("spread_20", pd.DataFrame(SPR).rolling(20, min_periods=10).mean().values)
del qt
log("2. Bid-ask spread")

# ----------------------------------------------------------------------------
# 3. Short selling: Markit lending (daily) and Compustat short interest (twice a month)
# ----------------------------------------------------------------------------
ln = read_block("lending", ["datadate", "cusip", "utilisation", "dcbs", "indicativefee", "lendablequantity"])
ln["datadate"] = pd.to_datetime(ln["datadate"])
ln = ln[ln["datadate"] <= END]
ln["cusip8"] = ln["cusip"].astype(str).str[:8]
# Markit can have several records per stock and day; keep the main one (most lendable shares)
ln = map_cusip8(ln, "cusip8", "datadate").sort_values("lendablequantity", na_position="first")
ln = ln.drop_duplicates(["permno", "datadate"], keep="last")
ln["t"] = next_td(ln["datadate"])
UT = ffill(to_wide(ln, "utilisation"), 10)
put("lend_util", UT)
put("lend_cost", ffill(to_wide(ln, "dcbs"), 10))
put("lend_fee", ffill(to_wide(ln, "indicativefee"), 10))
put("lend_util_chg", UT - np.vstack([np.full((5, len(PERM)), np.nan), UT[:-5]]))
coverage("lending", UT)
del ln

si = pd.concat([pd.read_csv(EXTRA / "short_interest" / "current.csv.gz").assign(src=0),    # prefer the current table
                pd.read_csv(EXTRA / "short_interest" / "legacy.csv.gz").assign(src=1)], ignore_index=True)
si["datadate"] = pd.to_datetime(si["datadate"])
si["gvkey"] = si["gvkey"].astype(str).str.zfill(6)
si["iid"] = si["iid"].astype(str)
si = si[si["datadate"] <= END].sort_values("src").drop_duplicates(["gvkey", "iid", "datadate"])
si = si.sort_values(["gvkey", "datadate", "iid"]).drop_duplicates(["gvkey", "datadate"], keep="first")   # primary issue
si = map_gvkey(si, "datadate", primary_only=True).drop_duplicates(["permno", "datadate"])
si["si_ratio"] = (si["shortint"] / at(SHARES, si["datadate"], si["permno"])).clip(0, 1.5)
si["si_dtc"] = si["shortint"] / at(ADV20, si["datadate"], si["permno"])
si = si.sort_values(["permno", "datadate"])
gap = si.groupby("permno")["datadate"].diff().dt.days
si["si_change"] = si.groupby("permno")["si_ratio"].diff().where(gap <= 31)
si["t"] = next_td(si["datadate"], lag=10)
SIR = ffill(to_wide(si, "si_ratio"), 31)
put("si_ratio", SIR)
put("si_days_to_cover", ffill(to_wide(si, "si_dtc"), 31))
put("si_change", ffill(to_wide(si, "si_change"), 31))
coverage("short", SIR)
del si
log("3. Short selling")

# ----------------------------------------------------------------------------
# 4. Analysts (I/B/E/S)
# ----------------------------------------------------------------------------
lk_ib = pd.read_csv(EXTRA / "links" / "ibes_crsp.csv.gz", parse_dates=["sdate", "edate"])

ic = read_block("ibes_consensus")
ic["statpers"], ic["fpedats"] = pd.to_datetime(ic["statpers"]), pd.to_datetime(ic["fpedats"])
ic = ic[ic["statpers"] <= END]
ic = map_dated(ic, "ticker", lk_ib, "ticker", "statpers", "sdate", "edate", ["score"]).drop_duplicates(["permno", "statpers"])
ic = ic.sort_values(["permno", "fpedats", "statpers"])
g = ic.groupby(["permno", "fpedats"])
prev, gap = g["meanest"].shift(1), g["statpers"].diff().dt.days
ic["rev"] = ((ic["meanest"] - prev) / np.maximum(prev.abs(), 0.05)).where(gap <= 45).clip(-1, 1)   # unit-free (split-safe)
ic["disp"] = (ic["stdev"] / np.maximum(ic["meanest"].abs(), 0.05)).clip(0, 5)
ic["cov"] = np.log1p(ic["numest"])
ic["updown"] = (ic["numup"] - ic["numdown"]) / ic["numest"].where(ic["numest"] > 0)
ic["t"] = next_td(ic["statpers"])
ic = ic.sort_values("statpers")
COV = ffill(to_wide(ic, "cov"), 31)
put("an_coverage", COV)
put("an_revision", ffill(to_wide(ic, "rev"), 31))
put("an_dispersion", ffill(to_wide(ic, "disp"), 31))
put("an_updown", ffill(to_wide(ic, "updown"), 31))
coverage("analysts", COV)
del ic

rc = read_block("ibes_recs")
rc["statpers"] = pd.to_datetime(rc["statpers"])
rc = rc[rc["statpers"] <= END]
rc = map_dated(rc, "ticker", lk_ib, "ticker", "statpers", "sdate", "edate", ["score"]).drop_duplicates(["permno", "statpers"])
rc = rc.sort_values(["permno", "statpers"])
gap = rc.groupby("permno")["statpers"].diff().dt.days
rc["chg"] = rc.groupby("permno")["meanrec"].diff().where(gap <= 45)
rc["t"] = next_td(rc["statpers"])
put("rec_mean", ffill(to_wide(rc, "meanrec"), 31))
put("rec_change", ffill(to_wide(rc, "chg"), 31))
del rc

su = read_block("ibes_surprise")
su["anndats"] = pd.to_datetime(su["anndats"])
su = su[(su["anndats"] <= END) & su["suescore"].notna()]
su = map_dated(su, "ticker", lk_ib, "ticker", "anndats", "sdate", "edate", ["score"]).drop_duplicates(["permno", "anndats"])
su["t"] = next_td(su["anndats"])
put("ibes_sue", ffill(to_wide(su.sort_values("anndats"), "suescore"), 63))
del su
log("4. Analysts")

# ----------------------------------------------------------------------------
# 5. Institutional ownership (13F)
# ----------------------------------------------------------------------------
io = read_block("inst_own")
io["rdate"] = pd.to_datetime(io["rdate"])
io = io[io["rdate"] <= END]
io = map_cusip8(io, "cusip", "rdate").sort_values("inst_shares").drop_duplicates(["permno", "rdate"], keep="last")
io["ratio"] = (io["inst_shares"] / at(SHARES, io["rdate"], io["permno"])).clip(0, 1.5)
io["n"] = np.log1p(io["n_inst"])
io = io.sort_values(["permno", "rdate"])
gap = io.groupby("permno")["rdate"].diff().dt.days
io["chg"] = io.groupby("permno")["ratio"].diff().where(gap <= 100)
io["t"] = on_or_after(io["rdate"] + pd.Timedelta(days=45))
IOR = ffill(to_wide(io, "ratio"), 93)
put("io_ratio", IOR)
put("io_n", ffill(to_wide(io, "n"), 93))
put("io_change", ffill(to_wide(io, "chg"), 93))
coverage("inst", IOR)
del io
log("5. Institutional ownership")

# ----------------------------------------------------------------------------
# 6. Insider trades (Form 4 filings): open-market purchases (P) and sales (S)
# ----------------------------------------------------------------------------
it = read_block("insiders", ["cusip6", "cusip2", "personid", "fdate", "trancode"])
it["fdate"] = pd.to_datetime(it["fdate"])
it = it[it["fdate"] <= END]
it["cusip8"] = it["cusip6"].astype(str).str.zfill(6) + it["cusip2"].astype(str).str.zfill(2)
it = map_cusip8(it, "cusip8", "fdate").drop_duplicates(["permno", "personid", "fdate", "trancode"])
it["t"] = next_td(it["fdate"])
first_filing = it["fdate"].min()
cnt = it.groupby(["t", "permno", "trancode"]).size().unstack(fill_value=0).reset_index()
B = np.zeros((len(days), len(PERM)))
S = np.zeros((len(days), len(PERM)))
for col, arr in [("P", B), ("S", S)]:
    if col in cnt:
        w = to_wide(cnt.rename(columns={col: "v"}), "v")
        arr[np.isfinite(w)] = w[np.isfinite(w)]
B90 = pd.DataFrame(B).rolling(63).sum().values
S90 = pd.DataFrame(S).rolling(63).sum().values
before = days < first_filing + pd.Timedelta(days=91)                  # no coverage yet: missing, not zero
for a in (B90, S90):
    a[before] = np.nan
put("ins_buys_90", np.log1p(B90))
put("ins_sells_90", np.log1p(S90))
put("ins_net_90", (B90 - S90) / (B90 + S90 + 1))
coverage("insiders", B90)
del it, B, S
log(f"6. Insiders (coverage from {first_filing.date()})")

# ----------------------------------------------------------------------------
# 7. Corporate events (Capital IQ key developments)
# ----------------------------------------------------------------------------
kd = read_block("keydev")
kd["announcedate"] = pd.to_datetime(kd["announcedate"])
kd["gvkey"] = kd["gvkey"].astype(str).str.zfill(6)
kd = kd[kd["announcedate"] <= END]
SCHEDULED = {"Conferences", "Company Conference Presentations", "Earnings Calls", "Earnings Release Date",
             "Annual General Meeting", "Shareholder/Analyst Calls", "Special Calls", "Analyst/Investor Day",
             "Announcements of Earnings", "Index Constituent Adds", "Index Constituent Drops"}
CATS = {"guidance": lambda e: e.startswith("Corporate Guidance"),
        "buyback": lambda e: "Buyback" in e,
        "mna": lambda e: "M&A" in e,
        "mgmt": lambda e: e.startswith("Executive"),
        "lawsuit": lambda e: e.startswith("Lawsuits")}            # (dividend events are in a separate Capital IQ table)
types = kd.groupby("eventtype")["n"].sum().sort_values(ascending=False)
print("   Top event types and how they are used:")
for e, n in types.head(25).items():
    use = [c for c, f in CATS.items() if f(e)]
    print(f"     {n:>8,}  {e:45s} -> {'excluded (scheduled)' if e in SCHEDULED else ('ev_' + use[0] if use else 'counted in ev_count')}")
kd = map_gvkey(kd, "announcedate")
kd["t"] = next_td(kd["announcedate"])
kd_unsched = kd[~kd["eventtype"].isin(SCHEDULED)]
cnt = kd_unsched.groupby(["t", "permno"])["n"].sum().rename("v").reset_index()
EV = to_wide(cnt, "v")
EV[~np.isfinite(EV)] = 0
put("ev_count", np.log1p(EV))
for cat, f in CATS.items():
    sub = kd[kd["eventtype"].map(f)]
    w = to_wide(sub.groupby(["t", "permno"])["n"].sum().rename("v").reset_index(), "v")
    w = np.isfinite(w).astype(float)
    put(f"ev_{cat}", pd.DataFrame(w).rolling(5, min_periods=1).max().values)
del kd, kd_unsched
log("7. Corporate events")

# ----------------------------------------------------------------------------
# 8. Credit (Markit CDS, 5-year senior USD)
# ----------------------------------------------------------------------------
cd = read_block("cds", ["date", "redcode", "docclause", "parspread"])
cd["date"] = pd.to_datetime(cd["date"])
cd = cd[(cd["date"] <= END) & (cd["parspread"] > 0)]
pri = {"XR14": 0, "XR": 1, "MR14": 2, "MR": 3, "CR14": 4, "CR": 5, "MM14": 6, "MM": 7}
cd["p"] = cd["docclause"].map(pri).fillna(9)
cd = cd.sort_values("p").drop_duplicates(["redcode", "date"])
red = pd.read_csv(EXTRA / "links" / "markit_red.csv.gz", dtype={"entity_cusip": str})[["redcode", "entity_cusip"]].drop_duplicates()
n6 = names.assign(cusip6=names["cusip8"].str[:6])[["cusip6", "permno", "start", "ending"]]
cd = cd.merge(red, on="redcode")
cd = map_dated(cd, "entity_cusip", n6, "cusip6", "date", "start", "ending").drop_duplicates(["permno", "date"])
cd["lspread"] = np.log(cd["parspread"] * 1e4)                          # log of the spread in basis points
cd["t"] = next_td(cd["date"])
CDS = ffill(to_wide(cd, "lspread"), 10)
put("cds_5y", CDS)
put("cds_chg_5d", CDS - np.vstack([np.full((5, len(PERM)), np.nan), CDS[:-5]]))
coverage("cds", CDS)
del cd
log("8. CDS")

# ----------------------------------------------------------------------------
# 9. 10-K text similarity to the previous year's 10-K ("Lazy Prices")
# ----------------------------------------------------------------------------
tx = read_block("sec_text", ["form", "fdate", "cik", "cosine", "lookback"])
tx["fdate"] = pd.to_datetime(tx["fdate"])
tx = tx[(tx["form"] == "10-K") & (tx["lookback"] == 1) & tx["cosine"].notna() & (tx["fdate"] <= END)]
cik = pd.read_csv(PIT / "fundamentals" / "company_cik.csv", dtype={"gvkey": str})
cik["cik"] = pd.to_numeric(cik["cik"], errors="coerce")
tx = tx.merge(cik[["cik", "gvkey"]].dropna(), on="cik")
tx = map_gvkey(tx, "fdate").sort_values("fdate").drop_duplicates(["permno", "fdate"], keep="last")
tx["t"] = next_td(tx["fdate"])
TXT = ffill(to_wide(tx, "cosine"), 275)
put("txt_similarity", TXT)
coverage("text", TXT)
del tx
log("9. 10-K text similarity")

# ----------------------------------------------------------------------------
# 10. Write the extended table
# ----------------------------------------------------------------------------
for k in list(NEW):
    NEW[k][~np.isfinite(NEW[k])] = np.nan
new = pd.DataFrame(NEW)
cols_old = [c for c in base.columns if c != "target_next"]
out = pd.concat([base[cols_old].reset_index(drop=True), new, base[["target_next"]].reset_index(drop=True)], axis=1)
OUT_FEATS.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_FEATS, index=False)
log(f"10. Saved {len(out):,} rows x {out.shape[1]} columns ({len(NEW)} new) to {OUT_FEATS}")

# ----------------------------------------------------------------------------
# 11. Edge table
# ----------------------------------------------------------------------------
c2p = names[["cusip8", "permno"]].drop_duplicates()
rel = pd.read_csv(EXTRA / "supply_chain" / "relationships.csv.gz", parse_dates=["start_"])
rel["end_"] = pd.to_datetime(rel["end_"], errors="coerce")
rel.loc[rel["end_"] >= pd.Timestamp("2099-01-01"), "end_"] = pd.NaT
rel["s8"], rel["t8"] = rel["source_cusip"].str[:8], rel["target_cusip"].str[:8]
rel = rel.merge(c2p.rename(columns={"cusip8": "s8", "permno": "sp"}), on="s8").merge(
    c2p.rename(columns={"cusip8": "t8", "permno": "tp"}), on="t8")
rel = rel[rel["sp"] != rel["tp"]]
rt = pd.read_csv(EXTRA / "supply_chain" / "relationship_type.csv.gz").set_index("id")["name"].to_dict()
sup = rel["rel_type"].isin(["CUSTOMER", "SUPPLIER"])
e_sup = pd.DataFrame({
    # CUSTOMER row: the target is a customer of the source -> source supplies target. SUPPLIER row: the reverse.
    "source_permno": np.where(rel.loc[sup, "rel_type"] == "CUSTOMER", rel.loc[sup, "sp"], rel.loc[sup, "tp"]),
    "target_permno": np.where(rel.loc[sup, "rel_type"] == "CUSTOMER", rel.loc[sup, "tp"], rel.loc[sup, "sp"]),
    "type": "supplier_of", "detail": "", "directed": 1,
    "start": rel.loc[sup, "start_"].values, "end": rel.loc[sup, "end_"].values,
    "weight": rel.loc[sup, "revenue_percent"].values})
oth = ~sup
e_oth = pd.DataFrame({
    "source_permno": rel.loc[oth, "sp"].values, "target_permno": rel.loc[oth, "tp"].values,
    "type": np.where(rel.loc[oth, "rel_type"] == "COMPETITOR", "competitor", "partner"),
    "detail": rel.loc[oth, "rel_type"].map(rt).values, "directed": 0,
    "start": rel.loc[oth, "start_"].values, "end": rel.loc[oth, "end_"].values, "weight": 1.0})
e_rev = pd.concat([e_sup, e_oth], ignore_index=True).assign(dataset="FactSet Revere")
e_rev = e_rev.drop_duplicates(["source_permno", "target_permno", "type", "detail", "start"])

# Shared analysts: two stocks covered by the same analyst in year y -> a link valid in year y + 1 (known after y)
ao_parts = []
for f in sorted((EXTRA / "ibes_analyst_overlap").glob("*.csv.gz")):
    y = int(f.name.split(".")[0])
    if y + 1 > END.year:
        continue
    a = pd.read_csv(f)
    a = a.merge(lk_ib, on="ticker")
    a = a[(a["sdate"] <= pd.Timestamp(f"{y}-12-31")) & (a["edate"].fillna(OPEN) >= pd.Timestamp(f"{y}-01-01"))]
    a = a[a["permno"].isin(PERM)][["analys", "permno"]].drop_duplicates()
    pairs = a.merge(a, on="analys")
    pairs = pairs[pairs["permno_x"] < pairs["permno_y"]]
    shared = pairs.groupby(["permno_x", "permno_y"])["analys"].nunique().rename("weight").reset_index()
    ao_parts.append(pd.DataFrame({
        "source_permno": shared["permno_x"], "target_permno": shared["permno_y"], "type": "shared_analyst", "detail": "",
        "directed": 0, "start": pd.Timestamp(f"{y + 1}-01-01"), "end": pd.Timestamp(f"{y + 1}-12-31"),
        "weight": shared["weight"].astype(float), "dataset": "I/B/E/S"}))
e_an = pd.concat(ao_parts, ignore_index=True)

edges = pd.concat([e_rev, e_an], ignore_index=True)
edges["known_from"] = edges["start"]
edges = edges[edges["known_from"] <= END]
edges = edges[["source_permno", "target_permno", "type", "detail", "directed", "start", "end", "known_from", "weight", "dataset"]]
edges.to_parquet(OUT_EDGES, index=False)
log(f"11. Saved {len(edges):,} edges to {OUT_EDGES}")

# ----------------------------------------------------------------------------
# 12. Checks
# ----------------------------------------------------------------------------
mem = out[out["in_sp500"] > 0]
newcols = list(NEW)
print("\n12. Checks (S&P 500 member rows)")
cov = pd.DataFrame({y: mem.loc[mem["date"].dt.year == y, newcols].notna().mean() for y in [2004, 2008, 2014, 2020, 2022]})
print("   Share of member stock-days with a value, by year:")
print(cov.round(2).to_string())

aapl = out[(out["permno"] == 14593) & out["date"].between("2019-01-28", "2019-02-01")]
print("\n   Apple around its 29 Jan 2019 earnings:")
print(aapl[["date", "iv_atm", "iv_skew", "pc_volume", "spread", "lend_util", "si_ratio", "an_coverage", "an_revision",
            "rec_mean", "io_ratio", "ins_net_90", "ev_count", "cds_5y", "txt_similarity"]].round(4).to_string(index=False))

dev = mem[mem["date"].between("2006-01-01", TUNE_END)]
ics = {}
for f in newcols:
    if f.startswith("has_"):
        continue
    sub = dev[["date", f, "target_next"]].dropna()
    if len(sub) < 10000:
        continue
    r = sub.groupby("date")[[f, "target_next"]].rank()
    r = r - r.groupby(sub["date"]).transform("mean")
    gs = lambda v: v.groupby(sub["date"]).sum()
    ic = gs(r[f] * r["target_next"]) / np.sqrt(gs(r[f] ** 2) * gs(r["target_next"] ** 2))
    ics[f] = ic.replace([np.inf, -np.inf], np.nan).mean()
ics = pd.Series(ics).sort_values()
print("\n   Rank IC of each new feature with the NEXT day's excess return, 2006-2020 (look-ahead check):")
print(ics.round(4).to_string())
flag = ics[ics.abs() > 0.1]
print("   OK: no feature above 0.1" if flag.empty else f"   CHECK FOR LOOK-AHEAD: {list(flag.index)}")

print("\n   Edges by type:")
print(edges.groupby(["dataset", "type"]).size().to_string())
for d in ["2008-06-30", "2015-06-30", "2022-06-30"]:
    d = pd.Timestamp(d)
    members = set(out.loc[(out["date"] == days[days.searchsorted(d) - 1]) & (out["in_sp500"] > 0), "permno"])
    act = edges[(edges["start"] <= d) & (edges["end"].fillna(OPEN) >= d)
                & edges["source_permno"].isin(members) & edges["target_permno"].isin(members)]
    print(f"   Active links among S&P 500 members on {d.date()}: " + ", ".join(f"{k} {v:,}" for k, v in act["type"].value_counts().items()))
log("Done")
