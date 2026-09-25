"""
Layer 3b (part 2): the MATCHER.
Ensemble of gradient-boosted trees (LightGBM + XGBoost + CatBoost) averaged for diversity,
all precision-aware (F_0.5 weights false-positives 2x).  Optional cross-encoder deep leg blended.
Graceful fallback to LightGBM-only if xgboost/catboost unavailable.
"""
from __future__ import annotations
import numpy as np
import lightgbm as lgb

from .config import CFG


class Matcher:
    def __init__(self, ensemble: bool = True):
        self.ensemble = ensemble
        self.models = []

    def train(self, X, y):
        pos = max(int(y.sum()), 1)
        neg = max(len(y) - pos, 1)
        spw = neg / pos
        self.models = []

        lgbm = lgb.LGBMClassifier(
            objective="binary", n_estimators=800, learning_rate=0.03, num_leaves=95,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            min_child_samples=40, scale_pos_weight=spw,
            n_jobs=CFG.n_jobs, random_state=CFG.seed, verbosity=-1,
        )
        lgbm.fit(X, y)
        self.models.append(("lgbm", lgbm))

        if self.ensemble:
            try:
                import xgboost as xgb
                xgbm = xgb.XGBClassifier(
                    n_estimators=700, learning_rate=0.03, max_depth=8,
                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                    eval_metric="aucpr", n_jobs=CFG.n_jobs, random_state=CFG.seed,
                    tree_method="hist",
                )
                xgbm.fit(X, y)
                self.models.append(("xgb", xgbm))
            except Exception as e:
                print(f"[xgb skipped: {e}]")
            try:
                from catboost import CatBoostClassifier
                cat = CatBoostClassifier(
                    iterations=700, learning_rate=0.03, depth=8,
                    scale_pos_weight=spw, random_seed=CFG.seed,
                    thread_count=CFG.n_jobs, verbose=False,
                )
                cat.fit(X, y)
                self.models.append(("cat", cat))
            except Exception as e:
                print(f"[catboost skipped: {e}]")
        return self

    def predict(self, X) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype="float32")
        preds = [m.predict_proba(X)[:, 1] for _, m in self.models]
        return np.mean(preds, axis=0).astype("float32")

    # single-model save/load for the primary lgbm (full ensemble is retrained per run)
    def save(self, path):
        self.models[0][1].booster_.save_model(path)


# ------------------------------------------------ optional cross-encoder blend
def cross_encoder_scores(pairs, rec_lookup) -> np.ndarray:
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
    scores = np.asarray(ce.predict(texts, batch_size=CFG.embed_batch, show_progress_bar=True), dtype="float32")
    if scores.min() < 0 or scores.max() > 1:
        scores = 1 / (1 + np.exp(-scores))
    return scores


def blend(gbdt_p, ce_p, w: float = 0.4):
    if ce_p is None or not len(ce_p) or float(np.abs(ce_p).sum()) == 0.0:
        return gbdt_p
    return (1 - w) * gbdt_p + w * ce_p
