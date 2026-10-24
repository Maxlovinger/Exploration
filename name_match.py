"""Do two company names refer to the same company? Used to check RavenPack matches.

    name_match("EXXON MOBIL CORP", "ExxonMobil Holdings Corp.")  -> True
    name_match("CARNIVAL CORP", "CCL Products (India) Ltd.")     -> False
"""

import re

# Legal-form and filler words that do not identify a company
STOP = {"inc", "corp", "corporation", "co", "company", "companies", "cos", "ltd", "limited", "plc", "holdings",
        "holding", "group", "the", "new", "de", "sa", "ag", "nv", "llc", "lp", "trust", "international", "intl",
        "incorporated", "kgaa", "se", "spa", "pcl", "tbk", "psc", "asa", "as", "bhd", "oyj", "ab"}
# First words too common to identify a company on their own ("National City" vs "National Cement")
GENERIC = {"national", "american", "general", "first", "united", "global", "china", "great", "southern",
           "northern", "western", "eastern", "pacific", "royal", "central", "union", "state", "capital", "energy",
           "bank", "financial", "air", "home", "health", "us", "usa", "north", "south", "east", "west", "world"}


def tokens(name):
    """Lowercase identifying words; runs of single letters are joined ("B M C" -> "bmc")."""
    s = str(name).lower().replace("&", " ").replace("'", "").replace("!", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    out, buf = [], ""
    for w in s.split():
        if len(w) == 1:
            buf += w
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(w)
    if buf:
        out.append(buf)
    return [w for w in out if w not in STOP]


def name_match(a, b):
    """True if the two names plausibly refer to the same company."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    sa, sb = set(ta), set(tb)
    if len(sa & sb) / len(sa | sb) >= 0.5:                        # most words shared
        return True
    ca, cb = "".join(ta), "".join(tb)
    if min(len(ca), len(cb)) >= 6 and (ca.startswith(cb) or cb.startswith(ca)):   # "exxon mobil" vs "exxonmobil"
        return True
    return ta[0] == tb[0] and ta[0] not in GENERIC and len(ta[0]) >= 4             # same distinctive first word


def exact_name(a, b):
    """Strict check: the same identifying words in the same order ("National City" is not "City National")."""
    ta, tb = tokens(a), tokens(b)
    return bool(ta) and "".join(ta) == "".join(tb)
