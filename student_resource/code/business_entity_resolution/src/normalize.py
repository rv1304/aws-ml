"""
Layer 2 core: per-field-type normalization.  DIFFERENT rule per field type.
- name    : semantic/fuzzy   -> cleanco legal-suffix strip + unidecode translit + canonical forms
- address : structured       -> libpostal expand/parse (if available) else dict fallback; extract pin/number/state
- country : categorical       -> passthrough (open set, France-safe; never hard-coded/one-hot)
- id      : structural        -> handled by ingest (prefix router); never fuzzy-matched
No external DB / geocoding API / internet augmentation (fair-play). libpostal is an OFFLINE
statistical normalizer (OSM-trained model) not a lookup service; flagged in README.
"""
from __future__ import annotations
import re
from unidecode import unidecode

# ---- optional best-in-class libs (graceful fallback if not installed) ----
try:
    from cleanco import basename as _cleanco_basename        # strips legal suffixes, multi-country
    _HAS_CLEANCO = True
except Exception:
    _HAS_CLEANCO = False

try:
    from postal.expand import expand_address as _pexpand      # libpostal
    from postal.parser import parse_address as _pparse
    _HAS_POSTAL = True
except Exception:
    _HAS_POSTAL = False

# ---------------------------------------------------------------- dictionaries
NAME_ABBR = {
    "corp": "corporation", "inc": "incorporated", "co": "company", "ltd": "limited",
    "pvt": "private", "intl": "international", "assoc": "associates", "bros": "brothers",
    "mfg": "manufacturing", "svcs": "services", "svc": "service", "grp": "group",
    "ent": "enterprises", "ind": "industries", "&": "and",
}
LEGAL_SUFFIX = {"corporation", "incorporated", "company", "limited", "private", "llp",
                "llc", "plc", "gmbh", "srl", "sarl", "pte", "bv", "ag", "sa", "inc",
                "ltd", "pvt", "co", "corp",
                # French legal entities (#4, unseen FR test)
                "sas", "sasu", "sci", "eurl", "snc", "sca", "scs", "scop", "gie"}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court", "pl": "place",
    "sq": "square", "hwy": "highway", "pkwy": "parkway", "ter": "terrace",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "rm": "room",
    "bldg": "building",
    "opp": "opposite", "nr": "near", "nearby": "near", "no": "number", "num": "number",
    "sec": "sector", "ph": "phase", "blk": "block", "kh": "khasra", "po": "postoffice",
    # French street types (#4)
    "bd": "boulevard", "bld": "boulevard", "imp": "impasse", "all": "allee",
    "rte": "route", "che": "chemin", "sq.": "square",
}
# business-type words -> canonical type. Distinct types at same address are distinct
# entities (ATM vs Bank, Clinic vs Hospital) (#15).
ENTITY_TYPES = {
    "atm": "atm", "bank": "bank", "hospital": "hospital", "clinic": "clinic",
    "pharmacy": "pharmacy", "chemist": "pharmacy", "medical": "clinic",
    "hotel": "hotel", "lodge": "hotel", "restaurant": "restaurant", "cafe": "cafe",
    "school": "school", "college": "college", "university": "college",
    "store": "store", "mart": "store", "supermarket": "store", "mall": "mall",
    "temple": "worship", "church": "worship", "mosque": "worship",
    "petrol": "fuel", "fuel": "fuel", "station": "station", "office": "office",
    "factory": "factory", "warehouse": "warehouse", "showroom": "showroom",
    "salon": "salon", "gym": "gym", "hostel": "hostel",
}
# French postal artifacts to strip (no entity signal) (#4)
_fr_postal = re.compile(r"\b(cedex|bp|cs|tsa)\b\s*\d*", re.IGNORECASE)
# French ordinal building subdivisions -> glue to preceding number: "12 bis" -> "12bis" (#4)
_fr_ordinal = re.compile(r"\b(\d+)\s+(bis|ter|quater|quinquies)\b", re.IGNORECASE)
US_STATES = {
    "al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california","co":"colorado",
    "ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho",
    "il":"illinois","in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana",
    "me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan","mn":"minnesota",
    "ms":"mississippi","mo":"missouri","mt":"montana","ne":"nebraska","nv":"nevada",
    "nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york","nc":"north carolina",
    "nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania",
    "ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee","tx":"texas",
    "ut":"utah","vt":"vermont","va":"virginia","wa":"washington","wv":"west virginia",
    "wi":"wisconsin","wy":"wyoming",
}
IN_STATES = {
    "ap":"andhra pradesh","ar":"arunachal pradesh","as":"assam","br":"bihar","cg":"chhattisgarh",
    "ga":"goa","gj":"gujarat","hr":"haryana","hp":"himachal pradesh","jh":"jharkhand",
    "ka":"karnataka","kl":"kerala","mp":"madhya pradesh","mh":"maharashtra","mn":"manipur",
    "ml":"meghalaya","mz":"mizoram","nl":"nagaland","od":"odisha","pb":"punjab","rj":"rajasthan",
    "sk":"sikkim","tn":"tamil nadu","tg":"telangana","ts":"telangana","tr":"tripura",
    "up":"uttar pradesh","uk":"uttarakhand","wb":"west bengal","dl":"delhi",
    "jk":"jammu and kashmir","ch":"chandigarh",
}
# country-scoped state maps so a US 'or'=oregon does not collide with India 'or'=odisha etc.
STATE_BY_COUNTRY = {"US": US_STATES, "India": {**US_STATES, **IN_STATES}}

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]", re.UNICODE)
_repeat = re.compile(r"(.)\1{2,}")
_pin_in = re.compile(r"\b(\d{6})\b")
_zip_us = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_num = re.compile(r"\b\d+[a-z]?\b")
_tld = re.compile(r"\.(com|net|org|in|co|io|biz|info|us|co\.in|org\.in)\b", re.IGNORECASE)
_phone = re.compile(r"\b\d{7,}\b")          # phone-like long digit runs (not house/pin)
_domain_tokens = {"com", "net", "org", "io", "biz", "info", "www"}


def _basic(s: str) -> str:
    s = unidecode(s or "").lower()
    s = s.replace("&", " and ")
    s = _punct.sub(" ", s)
    s = _repeat.sub(r"\1\1", s)            # squeeze translit doubling (limittedd -> limitedd)
    return _ws.sub(" ", s).strip()


# ------------------------------------------------------------------ NAME field
def norm_name(raw: str) -> dict:
    core_src = _tld.sub(" ", raw or "")          # strip domain TLDs (maurewilliams.com -> maurewilliams)
    if _HAS_CLEANCO:
        try:
            core_src = _cleanco_basename(core_src) or core_src   # drop legal suffixes across countries
        except Exception:
            pass
    b_full = _basic(raw)
    b_core = _basic(core_src)
    toks_full = [NAME_ABBR.get(t, t) for t in b_full.split() if t not in _domain_tokens]
    core = [t for t in (NAME_ABBR.get(t, t) for t in b_core.split())
            if t not in LEGAL_SUFFIX and t not in _domain_tokens]
    suffixes = sorted({t for t in toks_full if t in LEGAL_SUFFIX})
    core_str = " ".join(core) if core else b_full
    types = sorted({ENTITY_TYPES[t] for t in toks_full if t in ENTITY_TYPES})
    return {
        "name_norm": " ".join(toks_full),
        "name_core": core_str,
        "name_sorted": " ".join(sorted(core_str.split())),
        "name_acronym": "".join(w[0] for w in core_str.split() if w),
        "name_suffix": " ".join(suffixes),
        "name_nospace": core_str.replace(" ", ""),
        "name_type": " ".join(types),          # business type(s) for distinct-entity guard (#15)
    }


# --------------------------------------------------------------- ADDRESS field
def _canon_state(tok: str, country: str) -> str:
    return STATE_BY_COUNTRY.get(country, US_STATES).get(tok, tok)


def norm_address(raw: str, country: str = "US") -> dict:
    if _HAS_POSTAL and raw:
        try:
            expanded = _pexpand(raw)
            base = expanded[0] if expanded else raw
        except Exception:
            base = raw
    else:
        base = raw
    base = _fr_ordinal.sub(r"\1\2", base or "")   # "12 bis" -> "12bis" (distinct adjacent buildings) (#4)
    base = _fr_postal.sub(" ", base)              # strip CEDEX/BP/CS artifacts (#4)
    b = _basic(base).replace(" null ", " ").replace("null", " ")
    pin = ""
    m = _pin_in.search(b) or _zip_us.search(b)
    if m:
        pin = m.group(1)
    b = _phone.sub(" ", b)                        # drop phone-like long digit runs (denoise)
    toks = [_canon_state(ADDR_ABBR.get(t, t), country) for t in b.split()]
    txt = " ".join(toks)
    nums = {n for n in _num.findall(txt) if len(n) <= 6}   # house/plot numbers only
    comps = {}
    if _HAS_POSTAL and raw:
        try:
            for val, key in _pparse(raw):
                comps.setdefault(key, unidecode(val).lower())
        except Exception:
            pass
    return {
        "addr_norm": txt,
        "addr_pin": pin or comps.get("postcode", ""),
        "addr_nums": nums,
        "addr_tokens": {t for t in txt.split() if len(t) > 1},
        "addr_city": comps.get("city", ""),
        "addr_state": comps.get("state", ""),
    }


def blocking_key(name_core: str, k: int = 8) -> str:
    return name_core.replace(" ", "")[:k]
