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
