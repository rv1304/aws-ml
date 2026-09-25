"""
Layer 3c: RESOLVE / decide.  Turn scored pairs into final matches under precision guards.
- threshold (tuned for F_0.5)
- HARD VETO: drop a candidate whose address hard-contradicts (both PINs present & different)
  unless the name is near-identical -> kills chained false merges.
- GOLDEN CONSTRAINT: Source 1 is deduplicated, so a right record maps to AT MOST ONE S1.
  If several S1 claim the same right id, keep only the highest-scoring S1.
"""
from __future__ import annotations
from collections import defaultdict
from rapidfuzz import fuzz


def _veto(r1, r2) -> bool:
    p1, p2 = r1.get("addr_pin", ""), r2.get("addr_pin", "")
    if p1 and p2 and p1 != p2:
        # allow only if name is essentially identical (same brand, data-entry pin error)
        if fuzz.token_set_ratio(r1["name_core"], r2["name_core"]) < 95:
            return True
    return False


def resolve(pairs, scores, threshold, rec_lookup, enforce_one_s1=True):
    """
    pairs  : list[(s1_id, right_id)]
    scores : aligned probabilities
    Returns dict[s1_id] -> set(right_id).
    """
    # 1) threshold + veto ; track best (s1,score) per right for the one-S1 constraint
    kept = []            # (s1, right, score)
    for (sid, rid), sc in zip(pairs, scores):
        if sc < threshold:
            continue
        r1, r2 = rec_lookup.get(sid), rec_lookup.get(rid)
        if r1 is None or r2 is None or _veto(r1, r2):
            continue
        kept.append((sid, rid, float(sc)))

    if enforce_one_s1:
        best_owner = {}          # right_id -> (score, s1_id)
        for sid, rid, sc in kept:
            cur = best_owner.get(rid)
            if cur is None or sc > cur[0]:
                best_owner[rid] = (sc, sid)
        out = defaultdict(set)
        for rid, (sc, sid) in best_owner.items():
            out[sid].add(rid)
        return dict(out)

    out = defaultdict(set)
    for sid, rid, sc in kept:
        out[sid].add(rid)
    return dict(out)


def to_submission_rows(all_s1_ids, match_map):
    """Every S1 gets exactly one row; empty when no match. Returns list[(s1_id, csv_ids)]."""
    rows = []
    for sid in all_s1_ids:
        ids = sorted(match_map.get(sid, set()))
        rows.append((sid, ",".join(ids)))
    return rows
