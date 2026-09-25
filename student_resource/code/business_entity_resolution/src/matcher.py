"""
Layer 3b (part 2): the MATCHER.
Primary: LightGBM on pair features (fast, strong, handles missing/-1).
Optional deep leg: cross-encoder (Ditto-style) reads both records -> semantic match prob;
blended with the GBDT prob.  Both are MIT/Apache, <=8B (rule-compliant).
"""
from __future__ import annotations
import numpy as np
import lightgbm as lgb

from .config import CFG


class Matcher:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        pos = max(int(y.sum()), 1)
        neg = max(len(y) - pos, 1)
        params = dict(
            objective="binary", metric="average_precision",
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            min_child_samples=50, scale_pos_weight=neg / pos,   # precision-aware
            n_jobs=CFG.n_jobs, random_state=CFG.seed, verbosity=-1,
        )
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype="float32")
        return self.model.predict_proba(X)[:, 1].astype("float32")

    def save(self, path):
        self.model.booster_.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)
        return self


# ------------------------------------------------ optional cross-encoder blend
def cross_encoder_scores(pairs, rec_lookup) -> np.ndarray:
    """Deep semantic re-scoring. Returns prob per pair; falls back to zeros if unavailable."""
    if not CFG.use_cross_encoder or not pairs:
        return np.zeros(len(pairs), dtype="float32")
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(CFG.cross_encoder, device=CFG.device)
    except Exception as e:
        print(f"[cross-encoder disabled: {e}]")
        return np.zeros(len(pairs), dtype="float32")

    def _ser(rid):
        r = rec_lookup[rid]
        return f"{r['name_core']} [SEP] {r['addr_norm']}"

    texts = [(_ser(a), _ser(b)) for a, b in pairs]
    scores = ce.predict(texts, batch_size=CFG.embed_batch, show_progress_bar=True)
    scores = np.asarray(scores, dtype="float32")
    # squash to 0..1 if the model outputs logits
    if scores.min() < 0 or scores.max() > 1:
        scores = 1 / (1 + np.exp(-scores))
    return scores


def blend(gbdt_p: np.ndarray, ce_p: np.ndarray, w: float = 0.5) -> np.ndarray:
    if ce_p is None or not len(ce_p) or float(np.abs(ce_p).sum()) == 0.0:
        return gbdt_p
    return (1 - w) * gbdt_p + w * ce_p
