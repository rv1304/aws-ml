"""
F_0.5 scorer — EXACTLY the competition metric.
Per S1 entity: precision & recall of predicted match set vs truth, then F_0.5,
then MACRO-average over all S1 entities. Singletons: empty-correct = 1.0, false-merge = 0.0.
"""
from __future__ import annotations

BETA2 = 0.25   # beta^2 for beta=0.5


def f_beta_entity(pred: set, truth: set) -> float:
    if not truth and not pred:
        return 1.0
    if not pred:                      # missed everything (truth non-empty)
        return 0.0
    if not truth:                     # predicted on a true singleton -> false merge
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(truth)
    denom = BETA2 * precision + recall
    return (1 + BETA2) * precision * recall / denom if denom else 0.0


def macro_f05(pred_map: dict, truth_map: dict) -> float:
    """pred_map/truth_map: dict[s1_id] -> set(right ids). Averaged over truth_map keys."""
    if not truth_map:
        return 0.0
    total = 0.0
    for sid, truth in truth_map.items():
        pred = pred_map.get(sid, set())
        total += f_beta_entity(pred, truth)
    return total / len(truth_map)


def candidate_recall(candidates: dict, truth_map: dict):
    """
    HARD CEILING. Fraction of true (s1,right) pairs that blocking actually retrieved.
    Matcher can never recover a pair missing from candidates -> this bounds max recall,
    therefore bounds max macro-F0.5. Returns (micro_recall, macro_f05_ceiling).
    """
    cand_map = {sid: {rid for rid, _ in lst} for sid, lst in candidates.items()}
    tp = tot = 0
    ceil_total = 0.0
    for sid, truth in truth_map.items():
        got = cand_map.get(sid, set()) & truth
        tp += len(got)
        tot += len(truth)
        # best achievable: predict exactly the retrieved truth (perfect precision) -> its F0.5
        ceil_total += f_beta_entity(got, truth)
    micro = tp / tot if tot else 1.0
    macro_ceil = ceil_total / len(truth_map) if truth_map else 1.0
    return micro, macro_ceil


def tune_threshold(pair_scores, pairs, truth_map, grid=None):
    """
    pair_scores: np.array probs; pairs: list[(s1,right)] aligned.
    Returns (best_threshold, best_f05). Searches grid for max macro-F0.5.
    """
    import numpy as np
    from collections import defaultdict
    if grid is None:
        grid = np.round(np.arange(0.10, 0.96, 0.02), 3)
    by_s1 = defaultdict(list)
    for (sid, rid), sc in zip(pairs, pair_scores):
        by_s1[sid].append((rid, float(sc)))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        pred = {sid: {rid for rid, sc in lst if sc >= t} for sid, lst in by_s1.items()}
        f = macro_f05(pred, truth_map)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def tune_threshold_by_country(pair_scores, pairs, truth_map, country_of, grid=None,
                              min_s1=200, unseen_margin=0.10):
    """
    Country-specific decision boundaries (#3). Tune theta independently per country
    (US=structured -> lower; India=noisy -> stricter). Countries with too few S1 fall back
    to the global threshold. Returns (thr_by_country dict, global_thr, global_f05).
    unseen (e.g. French test) countries get global_thr + unseen_margin -> counteract
    uncalibrated overconfidence on distribution shift.
    """
    from collections import defaultdict
    g_t, g_f = tune_threshold(pair_scores, pairs, truth_map, grid)
    # split by country
    idx_by_c = defaultdict(list)
    for i, (sid, _rid) in enumerate(pairs):
        idx_by_c[country_of.get(sid, "")].append(i)
    thr = {}
    ps = list(pair_scores)
    for c, idxs in idx_by_c.items():
        s1_here = {pairs[i][0] for i in idxs}
        if len(s1_here) < min_s1:
            thr[c] = g_t
            continue
        sub_scores = [ps[i] for i in idxs]
        sub_pairs = [pairs[i] for i in idxs]
        sub_truth = {sid: truth_map[sid] for sid in s1_here if sid in truth_map}
        t_c, _ = tune_threshold(sub_scores, sub_pairs, sub_truth, grid)
        thr[c] = t_c
    return thr, g_t, g_f
