"""Pull the EXTRA data sources (options, analysts, short interest, securities lending, institutional ownership,
insiders, corporate events, supply-chain links, industry codes, filing text, CDS, bid/ask) for every stock that was
ever in the point-in-time S&P 500 panel. Tables and columns were confirmed with explore_extra.py (Oct 23, 2026).

Run (takes a while; every block saves as it goes and SKIPS files that already exist, so it can be stopped and re-run):

    cd ~/Documents/Thesis
    .venv/bin/python CoreFragment/pull_extra.py              # all blocks
    .venv/bin/python CoreFragment/pull_extra.py options_iv   # only some blocks (names below)

Tip: `export WRDS_USERNAME=maxlovinger` (e.g. in ~/.zshrc) skips the username prompt.

Big tables are filtered and aggregated ON THE WRDS SERVER, so only small results are downloaded
(e.g. the OptionMetrics volatility surface has ~350M rows per year; we keep 4 numbers per stock per day).
RAW data only: turning it into daily point-in-time features is a separate step (build_extra_features.py).

Blocks (output in CoreFragment/data/pit/extra/<block>/):
  links                 OptionMetrics secid <-> permno, I/B/E/S ticker <-> permno, Markit RED entity <-> CUSIP
  crsp_quotes           closing bid / ask / number of trades (CRSP daily file)
  options_iv            30-day implied volatility: ATM call / put, 25-delta call / put   (OptionMetrics vsurfd)
  options_volume        daily call / put volume and open interest                      (OptionMetrics opvold)
  ibes_consensus        monthly EPS consensus (FY1): mean, dispersion, # estimates, ups / downs
  ibes_recs             monthly consensus recommendation, ups / downs
  ibes_surprise         I/B/E/S earnings surprise (SUE score) by announcement date
  ibes_analyst_overlap  which analysts cover which stocks each year (-> "shared analyst" graph EDGES)
  short_interest        Compustat short interest (twice a month)
  lending               Markit securities lending: utilisation, cost of borrow (DCBS), fee, shares on loan
  inst_own              13F institutional holdings, aggregated per stock and quarter
  insiders              insider open-market purchases / sales (Thomson insiders, Form 4)
  keydev                Capital IQ key developments (corporate events), counted per stock, day, and event type
  supply_chain          FactSet Revere relationships (customer, supplier, competitor, partner) -> graph EDGES
  rbics                 FactSet RBICS industry codes (much finer than GICS)
  seg_customer          Compustat segment customers (major customers named in 10-Ks) -> graph EDGES
  sec_text              SEC filing text similarity ("Lazy Prices": how much a 10-K / 10-Q changed)
  cds                   Markit CDS 5-year spreads (credit risk)
"""

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import wrds

START, END = "2002-01-01", "2025-12-31"
YEARS = range(int(START[:4]), int(END[:4]) + 1)
PIT = Path(__file__).resolve().parent / "data" / "pit"
OUT = PIT / "extra"
CHUNK = 200

# ----------------------------------------------------------------------------
# Identifiers of every stock in the panel
# ----------------------------------------------------------------------------
panel_permnos = pd.read_csv(PIT / "built" / "panel.csv.gz", usecols=["permno"])["permno"].unique()
PERMNOS = sorted(int(p) for p in panel_permnos)
names = pd.read_csv(PIT / "names_history.csv", dtype={"cusip8": str, "ticker": str})
names = names[names["permno"].isin(PERMNOS)]
CUSIP8 = sorted(c for c in names["cusip8"].dropna().unique() if re.fullmatch(r"[0-9A-Z]{8}", c))
CUSIP6 = sorted({c[:6] for c in CUSIP8})
ccm = pd.read_csv(PIT / "ccm_links.csv", dtype={"gvkey": str})
GVKEYS = sorted(ccm.loc[ccm["permno"].isin(PERMNOS), "gvkey"].dropna().unique())
cik = pd.read_csv(PIT / "fundamentals" / "company_cik.csv", dtype={"gvkey": str})
CIKS = sorted(int(c) for c in pd.to_numeric(cik.loc[cik["gvkey"].isin(GVKEYS), "cik"], errors="coerce").dropna().unique())
print(f"Universe: {len(PERMNOS)} permnos, {len(CUSIP8)} CUSIPs, {len(GVKEYS)} gvkeys, {len(CIKS)} CIKs")


def q(values):
    for v in values:
        if not re.fullmatch(r"[A-Za-z0-9.\-]+", str(v)):
            sys.exit(f"Unsafe value in SQL list: {v!r}")
    return ", ".join(f"'{v}'" for v in values)


def ints(values):
    return ", ".join(str(int(v)) for v in values)


username = os.environ.get("WRDS_USERNAME")
db = wrds.Connection(wrds_username=username) if username else wrds.Connection()


def cols(lib, tab):
    return db.describe_table(library=lib, table=tab)["name"].str.lower().tolist()


def save(block, name, sql, date_cols=None):
    """Run `sql` and save to extra/<block>/<name>.csv.gz, unless that file already exists."""
    path = OUT / block / f"{name}.csv.gz"
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = db.raw_sql(sql, date_cols=date_cols)
    df.to_csv(path, index=False, compression="gzip")
    print(f"   {block}/{name}: {len(df):,} rows ({time.time() - t0:.0f}s)")
    return df


def load(block, name):
    p = OUT / block / f"{name}.csv.gz"
    return pd.read_csv(p) if p.exists() else None


# ----------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------
def links():
    save("links", "optionm_crsp", f"""
        SELECT secid, sdate, edate, permno, score FROM wrdsapps_link_crsp_optionm.opcrsphist
        WHERE permno IN ({ints(PERMNOS)})""")
    save("links", "ibes_crsp", f"""
        SELECT ticker, permno, ncusip, sdate, edate, score FROM wrdsapps_link_crsp_ibes.ibcrsphist
        WHERE permno IN ({ints(PERMNOS)})""")
    save("links", "markit_red", f"SELECT * FROM markit_red.redent WHERE entity_cusip IN ({q(CUSIP6)})")


def crsp_quotes():
    for i in range(0, len(PERMNOS), CHUNK):
        save("crsp_quotes", f"part{i // CHUNK:02d}", f"""
            SELECT permno, dlycaldt AS date, dlybid AS bid, dlyask AS ask, dlynumtrd AS ntrades
            FROM crsp.dsf_v2 WHERE permno IN ({ints(PERMNOS[i:i + CHUNK])}) AND dlycaldt BETWEEN '{START}' AND '{END}'""")


def secids():
    lk = load("links", "optionm_crsp")
    return sorted(lk["secid"].astype(int).unique()) if lk is not None else []


def options_iv():
    """One file per year; years that are not done yet are pulled one QUARTER at a time (the later yearly tables
    are much bigger and a whole-year query can time out)."""
    s = secids()
    for y in YEARS:
        if (OUT / "options_iv" / f"{y}.csv.gz").exists():
            continue
        for qn, (a, b) in enumerate([("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")], 1):
            save("options_iv", f"{y}q{qn}", f"""
            SELECT secid, date,
                   MAX(CASE WHEN cp_flag = 'C' AND delta = 50  THEN impl_volatility END) AS iv_c50,
                   MAX(CASE WHEN cp_flag = 'P' AND delta = -50 THEN impl_volatility END) AS iv_p50,
                   MAX(CASE WHEN cp_flag = 'C' AND delta = 25  THEN impl_volatility END) AS iv_c25,
                   MAX(CASE WHEN cp_flag = 'P' AND delta = -25 THEN impl_volatility END) AS iv_p25
            FROM optionm.vsurfd{y}
            WHERE days = 30 AND delta IN (50, -50, 25, -25) AND secid IN ({ints(s)})
              AND date BETWEEN '{y}-{a}' AND '{y}-{b}'
            GROUP BY secid, date""")


def options_volume():
    s = secids()
    c = cols("optionm", "opvold")
    print(f"   optionm.opvold columns: {c}")
    if not {"secid", "date", "volume"} <= set(c):
        print("   options_volume: expected columns not found; skipped")
        return
    oi = "open_interest" in c
    if "cp_flag" in c:
        sel = ("SUM(CASE WHEN cp_flag = 'C' THEN volume END) AS call_vol, SUM(CASE WHEN cp_flag = 'P' THEN volume END) AS put_vol"
               + (", SUM(CASE WHEN cp_flag = 'C' THEN open_interest END) AS call_oi, SUM(CASE WHEN cp_flag = 'P' THEN open_interest END) AS put_oi" if oi else ""))
    else:
        sel = "SUM(volume) AS volume" + (", SUM(open_interest) AS open_interest" if oi else "")
    for y in YEARS:
        save("options_volume", str(y), f"""
            SELECT secid, date, {sel} FROM optionm.opvold
            WHERE secid IN ({ints(s)}) AND date BETWEEN '{y}-01-01' AND '{y}-12-31' GROUP BY secid, date""")


def ibes_tickers():
    lk = load("links", "ibes_crsp")
    return sorted(t for t in lk["ticker"].dropna().astype(str).unique() if re.fullmatch(r"[A-Za-z0-9.\-]+", t)) if lk is not None else []


def ibes_consensus():
    t = ibes_tickers()
    for y in YEARS:
        save("ibes_consensus", str(y), f"""
            SELECT ticker, statpers, fpedats, numest, numup, numdown, meanest, medest, stdev, actual
            FROM ibes.statsum_epsus
            WHERE measure = 'EPS' AND fpi = '1' AND ticker IN ({q(t)}) AND statpers BETWEEN '{y}-01-01' AND '{y}-12-31'""")


def ibes_recs():
    save("ibes_recs", "all", f"""
        SELECT ticker, statpers, meanrec, medrec, numrec, numup, numdown FROM ibes.recdsum
        WHERE ticker IN ({q(ibes_tickers())}) AND statpers BETWEEN '{START}' AND '{END}'""")


def ibes_surprise():
    c = cols("ibes", "surpsum")
    keep = [x for x in ["ticker", "anndats", "pyear", "pmon", "measure", "fiscalp", "actual", "surpmean", "surpstdev", "suescore"] if x in c]
    where = (" AND measure = 'EPS'" if "measure" in c else "") + (" AND fiscalp = 'QTR'" if "fiscalp" in c else "")
    save("ibes_surprise", "all", f"""
        SELECT {', '.join(keep)} FROM ibes.surpsum
        WHERE ticker IN ({q(ibes_tickers())}) AND anndats BETWEEN '{START}' AND '{END}'{where}""")


def ibes_analyst_overlap():
    t = ibes_tickers()
    for y in YEARS:
        save("ibes_analyst_overlap", str(y), f"""
            SELECT ticker, analys, COUNT(*) AS n_forecasts FROM ibes.det_epsus
            WHERE ticker IN ({q(t)}) AND fpi IN ('1', '2') AND analys IS NOT NULL
              AND anndats BETWEEN '{y}-01-01' AND '{y}-12-31'
            GROUP BY ticker, analys""")


def short_interest():
    save("short_interest", "current", f"""
        SELECT gvkey, iid, datadate, shortint, shortintadj, splitadjdate FROM comp.sec_shortint
        WHERE gvkey IN ({q(GVKEYS)}) AND datadate BETWEEN '{START}' AND '{END}'""")
    save("short_interest", "legacy", f"""
        SELECT gvkey, iid, datadate, shortint, shortintadj, splitadjdate FROM comp.sec_shortint_legacy
        WHERE gvkey IN ({q(GVKEYS)}) AND datadate BETWEEN '{START}' AND '{END}'""")


def lending():
    lib = "markit_msf_analytics_eqty_amer"
    tabs = set(db.list_tables(library=lib))
    want = ["datadate", "isin", "cusip", "utilisation", "activeutilisation", "dcbs", "indicativefee",
            "shortloanquantity", "quantityonloan", "lendablequantity", "lenderconcentration"]
    for y in YEARS:
        t = f"amereqty{y}"
        if t not in tabs:
            continue
        c = cols(lib, t)
        save("lending", str(y), f"""
            SELECT {', '.join(x for x in want if x in c)} FROM {lib}.{t}
            WHERE substr(cusip, 1, 8) IN ({q(CUSIP8)})""")


def inst_own():
    for y in YEARS:
        save("inst_own", str(y), f"""
            WITH h AS (
                SELECT DISTINCT ON (mgrno, rdate, cusip) mgrno, rdate, cusip, shares
                FROM tfn.s34
                WHERE cusip IN ({q(CUSIP8)}) AND rdate BETWEEN '{y}-01-01' AND '{y}-12-31'
                ORDER BY mgrno, rdate, cusip, fdate)
            SELECT rdate, cusip, SUM(shares) AS inst_shares, COUNT(*) AS n_inst FROM h GROUP BY rdate, cusip""")


def insiders():
    for y in YEARS:
        save("insiders", str(y), f"""
            SELECT cusip6, cusip2, personid, fdate, trandate, trancode, acqdisp, shares, tprice, rolecode1, cleanse
            FROM tfn.table1
            WHERE cusip6 IN ({q(CUSIP6)}) AND trancode IN ('P', 'S') AND cleanse IN ('R', 'H')
              AND fdate BETWEEN '{y}-01-01' AND '{y}-12-31'""")


def keydev():
    for y in YEARS:
        save("keydev", str(y), f"""
            SELECT gvkey, announcedate, keydeveventtypeid, eventtype, objectroletype, COUNT(*) AS n
            FROM ciq.wrds_keydev
            WHERE gvkey IN ({q(GVKEYS)}) AND announcedate BETWEEN '{y}-01-01' AND '{y}-12-31'
            GROUP BY gvkey, announcedate, keydeveventtypeid, eventtype, objectroletype""")


def supply_chain():
    lib = "factset_revere_supply_chain"
    save("supply_chain", "relationships", f"""
        SELECT start_, end_, rel_type, source_cusip, target_cusip, source_ticker, target_ticker,
               source_name, target_name, revenue_percent, percent_estimated
        FROM {lib}.wrds_relationship
        WHERE substr(source_cusip, 1, 8) IN ({q(CUSIP8)}) AND substr(target_cusip, 1, 8) IN ({q(CUSIP8)})""")
    save("supply_chain", "relationship_type", f"SELECT * FROM {lib}.relationship_type")


def rbics():
    c = cols("factset_revere_common", "company_cusip")
    print(f"   factset_revere_common.company_cusip columns: {c}")
    ccol = next((x for x in c if "cusip" in x), None)
    if ccol is None:
        print("   rbics: no CUSIP column found; skipped")
        return
    m = save("rbics", "company_cusip", f"""
        SELECT * FROM factset_revere_common.company_cusip WHERE substr({ccol}, 1, 8) IN ({q(CUSIP8)})""")
    m = m if m is not None else load("rbics", "company_cusip")
    idcol = next((x for x in m.columns if x in ("company_id", "id")), None)
    ids = sorted(m[idcol].dropna().astype(int).unique()) if idcol else []
    print(f"   rbics: {len(ids)} FactSet company ids")
    if ids:
        save("rbics", "focus_l6", f"SELECT * FROM factset_revere_ind_class.company_rbics2_focus_l6 WHERE company_id IN ({ints(ids)})")
        save("rbics", "structure_l6", "SELECT * FROM factset_revere_ind_class.rbics2_structure_l6")


def seg_customer():
    save("seg_customer", "all", f"""
        SELECT gvkey, cid, cnms, ctype, gareac, gareat, salecs, sid, stype, srcdate
        FROM comp_segments_hist_daily.wrds_seg_customer WHERE gvkey IN ({q(GVKEYS)})""")


def sec_text():
    c = cols("wrdssec_bow", "bow_cossim")
    print(f"   wrdssec_bow.bow_cossim columns: {c}")
    ccol = next((x for x in c if x == "cik"), None)
    if ccol is None:
        print("   sec_text: no CIK column; skipped (send me the columns above)")
        return
    save("sec_text", "cossim", f"SELECT * FROM wrdssec_bow.bow_cossim WHERE CAST({ccol} AS bigint) IN ({ints(CIKS)})")


def cds():
    """Markit CDS is in LONG format: one row per entity, date, and tenor, with the spread in `parspread`.
    Keep the 5-year tenor, US dollars, senior unsecured debt."""
    red = load("links", "markit_red")
    if red is None or "redcode" not in red.columns:
        print("   cds: no RED codes (run the links block first); skipped")
        return
    codes = sorted(red["redcode"].dropna().astype(str).unique())
    lib = "markit_cds"
    tabs = set(db.list_tables(library=lib))
    for y in YEARS:
        t = f"cds{y}"
        if t not in tabs:
            continue
        c = cols(lib, t)
        need = {"date", "redcode", "tenor", "parspread"}
        if not need <= set(c):
            print(f"   {lib}.{t}: missing {sorted(need - set(c))}; skipped")
            continue
        keep = [x for x in ["date", "ticker", "redcode", "tier", "currency", "docclause", "parspread", "compositedepth5y"] if x in c]
        where = (" AND currency = 'USD'" if "currency" in c else "") + (" AND tier = 'SNRFOR'" if "tier" in c else "")
        save("cds", str(y), f"""
            SELECT {', '.join(keep)} FROM {lib}.{t}
            WHERE redcode IN ({q(codes)}) AND tenor = '5Y'{where}""")


BLOCKS = {f.__name__: f for f in [links, crsp_quotes, options_iv, options_volume, ibes_consensus, ibes_recs,
                                    ibes_surprise, ibes_analyst_overlap, short_interest, lending, inst_own, insiders,
                                    keydev, supply_chain, rbics, seg_customer, sec_text, cds]}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(BLOCKS)
    if "links" not in wanted and not (OUT / "links").exists():
        wanted = ["links"] + wanted                     # most blocks need the link tables
    t_all = time.time()
    for name in wanted:
        if name not in BLOCKS:
            print(f"Unknown block {name!r}. Choose from: {', '.join(BLOCKS)}")
            continue
        print(f"\n== {name} ==")
        t0 = time.time()
        try:
            BLOCKS[name]()
            print(f"   {name}: done ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"   {name}: FAILED ({type(e).__name__}: {str(e)[:300]}). Other blocks continue; re-run this block later.")
            try:
                db.connection.rollback()                 # clear the failed transaction so later blocks can run
            except Exception:
                pass
    db.close()
    print(f"\nAll done in {(time.time() - t_all) / 60:.1f} min. Files in {OUT}")
