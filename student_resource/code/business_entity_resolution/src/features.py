"""
Layer 3b (part 1): pairwise FEATURES.  Per-field comparators (different metric per field type).
Expanded feature set (v2) for higher F_0.5:
  name  : string(6 variants) + phonetic + set/char-jaccard + first/last tok + containment + digits
  addr  : token-set/ratio + jaccard + pin + number overlap/exact + state + empty flags
  cross : embedding cosine + block score + token-count diffs
"""
from __future__ import annotations
import numpy as np
from rapidfuzz import fuzz, distance
import jellyfish

FEATURE_NAMES = [
    "blk_score",
    # name string
    "name_token_set", "name_token_sort", "name_ratio", "name_partial",
    "name_wratio", "name_partial_token_sort", "name_jw", "name_lev_norm",
    "name_nospace_ratio",
    # name structure
    "name_acronym_eq", "name_first_tok_eq", "name_last_tok_eq",
    "name_token_jaccard", "name_char3_jaccard", "name_containment",
    "name_metaphone_eq", "name_soundex_ratio", "name_digits_eq",
    "name_tok_diff", "name_len_ratio", "suffix_eq",
    # address
    "addr_token_set", "addr_ratio", "addr_jaccard", "addr_pin_eq",
    "addr_num_overlap", "addr_num_exact", "addr_both_pin", "addr_empty_either",
    "addr_both_empty", "addr_num_disjoint",
    # distinct-entity guards
    "type_conflict", "chain_branch_trap",
    # cross-modal
    "emb_cos",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _char3(s: str) -> set:
    s = s.replace(" ", "")
    return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def _safe(fn, *a, default=0.0):
    try:
        return fn(*a)
    except Exception:
        return default


def _row(rec1, rec2, blk_score, emb_cos):
    n1, n2 = rec1["name_core"], rec2["name_core"]
    ns1, ns2 = rec1["name_sorted"], rec2["name_sorted"]
    a1, a2 = rec1["addr_norm"], rec2["addr_norm"]
    t1, t2 = rec1["addr_tokens"], rec2["addr_tokens"]
    p1, p2 = rec1["addr_pin"], rec2["addr_pin"]
    nm1, nm2 = rec1["addr_nums"], rec2["addr_nums"]
    tok1, tok2 = n1.split(), n2.split()
    set1, set2 = set(tok1), set(tok2)
    d1 = {c for c in n1 if c.isdigit()}
    d2 = {c for c in n2 if c.isdigit()}
    maxlen = max(len(n1), len(n2)) or 1
    smaller, larger = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
    ty1 = set(rec1.get("name_type", "").split())
    ty2 = set(rec2.get("name_type", "").split())
    contained = bool(smaller and smaller in larger and n1 != n2)
    addr_jac = _jaccard(t1, t2)
    return [
        blk_score,
        fuzz.token_set_ratio(n1, n2) / 100.0,
        fuzz.token_sort_ratio(ns1, ns2) / 100.0,
        fuzz.ratio(n1, n2) / 100.0,
        fuzz.partial_ratio(n1, n2) / 100.0,
        fuzz.WRatio(n1, n2) / 100.0,
        fuzz.partial_token_sort_ratio(n1, n2) / 100.0,
        distance.JaroWinkler.similarity(n1, n2),
        1.0 - distance.Levenshtein.normalized_distance(n1, n2),
        fuzz.ratio(rec1["name_nospace"], rec2["name_nospace"]) / 100.0,
        1.0 if rec1["name_acronym"] and rec1["name_acronym"] == rec2["name_acronym"] else 0.0,
        1.0 if tok1 and tok2 and tok1[0] == tok2[0] else 0.0,
        1.0 if tok1 and tok2 and tok1[-1] == tok2[-1] else 0.0,
        _jaccard(set1, set2),
        _jaccard(_char3(n1), _char3(n2)),
        1.0 if smaller and smaller in larger else 0.0,
        1.0 if _safe(jellyfish.metaphone, n1, default="x") == _safe(jellyfish.metaphone, n2, default="y") else 0.0,
        fuzz.ratio(_safe(jellyfish.soundex, n1, default=""), _safe(jellyfish.soundex, n2, default="")) / 100.0,
        1.0 if d1 == d2 else 0.0,
        abs(len(tok1) - len(tok2)),
        min(len(n1), len(n2)) / maxlen,
        1.0 if rec1["name_suffix"] and rec1["name_suffix"] == rec2["name_suffix"] else 0.0,
        fuzz.token_set_ratio(a1, a2) / 100.0,
        fuzz.ratio(a1, a2) / 100.0,
        addr_jac,
        1.0 if p1 and p2 and p1 == p2 else 0.0,
        _jaccard(nm1, nm2),
        1.0 if (nm1 & nm2) else 0.0,
        1.0 if p1 and p2 else 0.0,
        1.0 if (not a1 or not a2) else 0.0,
        1.0 if (not a1 and not a2) else 0.0,
        1.0 if (nm1 and nm2 and not (nm1 & nm2)) else 0.0,      # #8 disjoint house numbers -> distinct
        1.0 if (ty1 and ty2 and not (ty1 & ty2)) else 0.0,      # #15 ATM vs Bank etc -> distinct
        1.0 if (contained and a1 and a2 and addr_jac < 0.5) else 0.0,  # #11 chain/branch trap
        float(emb_cos) if emb_cos is not None else -1.0,
    ]


def build_features(candidates: dict, rec_lookup: dict, emb_lookup=None, gt: dict | None = None):
    """
    candidates : dict[s1_id] -> list[(right_id, blk_score)]
    rec_lookup : dict[entity_id] -> record dict
    emb_lookup : dict[entity_id] -> np.array (optional)
    gt         : dict[s1_id] -> set(right ids) (optional, for labels)
    Returns X (float32), pairs list[(s1,right)], y (int8|None)
    """
    X, pairs, y = [], [], ([] if gt is not None else None)
    for sid, cands in candidates.items():
        r1 = rec_lookup.get(sid)
        if r1 is None:
            continue
        e1 = emb_lookup.get(sid) if emb_lookup is not None else None
        pos = gt.get(sid, set()) if gt is not None else None
        for rid, blk in cands:
            r2 = rec_lookup.get(rid)
            if r2 is None:
                continue
            emb_cos = None
            if e1 is not None and emb_lookup is not None and rid in emb_lookup:
                emb_cos = float(np.dot(e1, emb_lookup[rid]))
            X.append(_row(r1, r2, blk, emb_cos))
            pairs.append((sid, rid))
            if y is not None:
                y.append(1 if rid in pos else 0)
    Xa = np.asarray(X, dtype="float32") if X else np.zeros((0, len(FEATURE_NAMES)), "float32")
    ya = np.asarray(y, dtype="int8") if y is not None else None
    return Xa, pairs, ya
