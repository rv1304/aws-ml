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
    # entity-type conflict (ATM vs Bank, Store vs Warehouse) -> distinct business.
    # Checked FIRST and NOT overridable by name: same-brand different-type is the exact trap
    # (an ATM and its parent branch always share the brand name).
    ty1 = set(r1.get("name_type", "").split())
    ty2 = set(r2.get("name_type", "").split())
    if ty1 and ty2 and not (ty1 & ty2):
        return True
    name_sim = fuzz.token_set_ratio(r1["name_core"], r2["name_core"])
    # near-identical names override the remaining location vetoes (brand w/ dirty address /
    # data-entry error) -> avoids splitting a real match on a typo'd PIN.
    if name_sim >= 95:
        return False
    # contradicting PINs -> different location
    p1, p2 = r1.get("addr_pin", ""), r2.get("addr_pin", "")
    if p1 and p2 and p1 != p2:
        return True
    # disjoint house numbers with a decent-but-not-identical name -> distinct branch/chain trap
    nm1, nm2 = r1.get("addr_nums", set()), r2.get("addr_nums", set())
    if nm1 and nm2 and not (nm1 & nm2) and name_sim < 90:
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
        scored = defaultdict(list)   # s1_id -> [(rid, score)] for triangle prune
        for rid, (sc, sid) in best_owner.items():
            out[sid].add(rid)
            scored[sid].append((rid, sc))
        return _triangle_prune(dict(out), scored, rec_lookup)

    out = defaultdict(set)
    scored = defaultdict(list)
    for sid, rid, sc in kept:
        out[sid].add(rid)
        scored[sid].append((rid, sc))
    return _triangle_prune(dict(out), scored, rec_lookup)


def _triangle_prune(out, scored, rec_lookup, tau: int = 85):
    """
    Triangle-inequality pruning (#2). If an S1 owns >1 right, the rights should also be
    similar to EACH OTHER (they're the same real business). Anchor = highest-score right;
    drop any co-member whose name is dissimilar to the anchor (< tau) -> kills false merges
    where two distinct businesses both got attached to one S1.
    """
    for sid, rids in list(out.items()):
        if len(rids) < 2:
            continue
        members = sorted(scored[sid], key=lambda x: -x[1])   # by score desc
        anchor = members[0][0]
        a_rec = rec_lookup.get(anchor)
        if a_rec is None:
            continue
        keep = {anchor}
        for rid, _sc in members[1:]:
            r = rec_lookup.get(rid)
            if r is None:
                continue
            if fuzz.token_set_ratio(a_rec["name_core"], r["name_core"]) >= tau:
                keep.add(rid)
        out[sid] = keep
    return out


def to_submission_rows(all_s1_ids, match_map):
    """Every S1 gets exactly one row; empty when no match. Returns list[(s1_id, csv_ids)]."""
    rows = []
    for sid in all_s1_ids:
        ids = sorted(match_map.get(sid, set()))
        rows.append((sid, ",".join(ids)))
    return rows
