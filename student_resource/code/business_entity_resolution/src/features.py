"""
Layer 3b (part 1): pairwise FEATURES.  Per-field comparators (different metric per field type).
Name  -> string + phonetic + set overlap.
Addr  -> token jaccard + pin/number equality + string.
Plus embedding cosine + block score.  Output feeds the matcher.
"""
from __future__ import annotations
import numpy as np
from rapidfuzz import fuzz, distance
import jellyfish

FEATURE_NAMES = [
    "blk_score",
    "name_token_set", "name_token_sort", "name_ratio", "name_partial",
    "name_jw", "name_lev_norm", "name_acronym_eq", "name_len_ratio",
    "name_nospace_ratio", "suffix_eq",
    "addr_token_set", "addr_ratio", "addr_jaccard", "addr_pin_eq",
    "addr_num_overlap", "addr_empty_either",
    "emb_cos",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _row(rec1, rec2, blk_score, emb_cos):
    n1, n2 = rec1["name_core"], rec2["name_core"]
    ns1, ns2 = rec1["name_sorted"], rec2["name_sorted"]
    a1, a2 = rec1["addr_norm"], rec2["addr_norm"]
    t1, t2 = rec1["addr_tokens"], rec2["addr_tokens"]
    p1, p2 = rec1["addr_pin"], rec2["addr_pin"]
    nm1, nm2 = rec1["addr_nums"], rec2["addr_nums"]
    len_ratio = (min(len(n1), len(n2)) / max(len(n1), len(n2))) if max(len(n1), len(n2)) else 1.0
    return [
        blk_score,
        fuzz.token_set_ratio(n1, n2) / 100.0,
        fuzz.token_sort_ratio(ns1, ns2) / 100.0,
        fuzz.ratio(n1, n2) / 100.0,
        fuzz.partial_ratio(n1, n2) / 100.0,
        distance.JaroWinkler.similarity(n1, n2),
        1.0 - distance.Levenshtein.normalized_distance(n1, n2),
        1.0 if rec1["name_acronym"] and rec1["name_acronym"] == rec2["name_acronym"] else 0.0,
        len_ratio,
        fuzz.ratio(rec1["name_nospace"], rec2["name_nospace"]) / 100.0,
        1.0 if rec1["name_suffix"] and rec1["name_suffix"] == rec2["name_suffix"] else 0.0,
        fuzz.token_set_ratio(a1, a2) / 100.0,
        fuzz.ratio(a1, a2) / 100.0,
        _jaccard(t1, t2),
        1.0 if p1 and p2 and p1 == p2 else 0.0,
        _jaccard(nm1, nm2),
        1.0 if (not a1 or not a2) else 0.0,
        float(emb_cos) if emb_cos is not None else -1.0,
    ]


def build_features(candidates: dict, rec_lookup: dict, emb_lookup=None, gt: dict | None = None):
    """
    candidates : dict[s1_id] -> list[(right_id, blk_score)]
    rec_lookup : dict[entity_id] -> record dict (normalized fields)
    emb_lookup : dict[entity_id] -> np.array  (optional, for emb cosine)
    gt         : dict[s1_id] -> set(right ids)  (optional, for labels)
    Returns X (np.float32), pairs (list of (s1_id,right_id)), y (np.int8 or None)
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
