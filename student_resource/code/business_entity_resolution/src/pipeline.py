"""
Orchestrator.  Two entry modes:
  dev  : train-holdout -> reports true macro F_0.5 (so you know your score before uploading)
  full : train matcher on train, run test -> writes matching_results.tsv + candidate_pairs.tsv

Memory strategy: process ONE COUNTRY at a time (matches are same-country in this data),
free between countries. Embeddings/cross-encoder optional via config (disable on low-RAM).
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CFG
from . import ingest, block, features, matcher, resolve, score
from .embed import Embedder

REC_FIELDS = ["name_core", "name_sorted", "name_acronym", "name_suffix", "name_nospace",
              "name_type", "addr_norm", "addr_pin", "addr_nums", "addr_tokens"]


# ------------------------------------------------------------------- helpers
def _rec_dicts(df: pd.DataFrame) -> dict:
    ids = df["entity_id"].to_numpy()
    cols = {f: df[f].to_numpy() for f in REC_FIELDS}
    return {ids[i]: {f: cols[f][i] for f in REC_FIELDS} for i in range(len(ids))}


def _embed(df: pd.DataFrame, key: str, embedder: Embedder):
    if not CFG.use_embeddings:
        return None
    arr = embedder.encode(df["blk_text"].tolist(), cache_key=key)
    ids = df["entity_id"].to_numpy()
    return dict(zip(ids, np.asarray(arr)))


def _countries(s1: pd.DataFrame):
    return list(pd.unique(s1["country"]))


def _slice_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    if CFG.same_country_only:
        return df[df["country"] == country].reset_index(drop=True)
    return df


# --------------------------------------------------- candidate+feature+score
def _candidates_and_features(s1c, rightc, embedder, tag, gt=None):
    emb_s1 = _embed(s1c, f"{tag}_s1", embedder)
    emb_r = _embed(rightc, f"{tag}_right", embedder)
    emb_s1_arr = np.stack([emb_s1[i] for i in s1c["entity_id"]]) if emb_s1 else None
    emb_r_arr = np.stack([emb_r[i] for i in rightc["entity_id"]]) if emb_r else None
    cands = block.generate_candidates(s1c, rightc, emb_s1_arr, emb_r_arr)
    rec = _rec_dicts(s1c); rec.update(_rec_dicts(rightc))
    emb_lookup = None
    if emb_s1 and emb_r:
        emb_lookup = {**emb_s1, **emb_r}
    X, pairs, y = features.build_features(cands, rec, emb_lookup, gt)
    return cands, X, pairs, y, rec


# --------------------------------------------------------------------- DEV
def run_dev(sample_s1: int = 20000, sample_right: int | None = None):
    print(f"[dev] device={CFG.device} embeddings={CFG.use_embeddings}")
    # load RAW (normalize only the small slices we actually use -> fast dev)
    s1 = ingest.load_side("source1", "train")
    right = ingest.load_side("right", "train")
    gt = ingest.load_ground_truth("train")

    rng = np.random.RandomState(CFG.seed)
    ids = s1["entity_id"].to_numpy()
    perm = rng.permutation(len(ids))
    n_val = int(len(ids) * CFG.val_frac)
    val_ids = set(ids[perm[:n_val]])
    fit_ids = set(ids[perm[n_val:]])
    # subsample for a fast, runnable dev estimate
    if sample_s1:
        fit_ids = set(list(fit_ids)[:sample_s1])
        val_ids = set(list(val_ids)[:max(sample_s1 // 4, 1)])
    # guarantee the true matches of sampled S1 are in the right pool (honest recall for matcher)
    need_right = set()
    for sid in fit_ids | val_ids:
        need_right |= gt.get(sid, set())

    embedder = Embedder()
    Xtr, ytr, Xva, pva = [], [], [], []
    truth_val = {}
    val_cands = {}   # for blocking-recall ceiling

    for c in _countries(s1):
        rc = _slice_country(right, c)
        if sample_right and len(rc) > sample_right:
            forced = rc[rc["entity_id"].isin(need_right)]
            extra = rc[~rc["entity_id"].isin(need_right)].sample(
                min(sample_right, len(rc)), random_state=CFG.seed)
            rc = pd.concat([forced, extra]).drop_duplicates("entity_id").reset_index(drop=True)
        rc = ingest.add_normalized(rc)
        s1c_raw = _slice_country(s1, c)
        fit_c = ingest.add_normalized(s1c_raw[s1c_raw["entity_id"].isin(fit_ids)].reset_index(drop=True))
        val_c = ingest.add_normalized(s1c_raw[s1c_raw["entity_id"].isin(val_ids)].reset_index(drop=True))
        if len(fit_c):
            _, X, pr, y, _ = _candidates_and_features(fit_c, rc, embedder, f"trfit_{c}", gt)
            if len(X):
                Xtr.append(X); ytr.append(y)
        if len(val_c):
            cands, X, pr, y, _ = _candidates_and_features(val_c, rc, embedder, f"trval_{c}", gt)
            if len(X):
                Xva.append(X); pva.extend(pr)
            val_cands.update(cands)
            for sid in val_c["entity_id"]:
                truth_val[sid] = gt.get(sid, set())
        del rc; gc.collect()

    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    print(f"[dev] train pairs={len(ytr)} pos_rate={ytr.mean():.3f}")
    # HARD CEILING: what blocking retrieved. Matcher cannot beat this.
    micro_r, macro_ceil = score.candidate_recall(val_cands, truth_val)
    print(f"[dev] blocking recall={micro_r:.4f}  macro F_0.5 CEILING={macro_ceil:.4f}  "
          f"<- max reachable; if this < target, fix blocking, not matcher")
    mdl = matcher.Matcher().train(Xtr, ytr)
    Xva = np.vstack(Xva) if Xva else np.zeros((0, Xtr.shape[1]), "float32")
    pscore = mdl.predict(Xva)
    t, f = score.tune_threshold(pscore, pva, truth_val)
    gap = macro_ceil - f
    print(f"[dev] BEST threshold={t}  macro F_0.5={f:.4f}  (val S1={len(truth_val)})  "
          f"gap-to-ceiling={gap:.4f}  ({'matcher-bound' if gap > 0.005 else 'blocking-bound'})")
    return t, f


# --------------------------------------------------------------------- FULL
def run_full(train_sample: int = 200000):
    print(f"[full] device={CFG.device} embeddings={CFG.use_embeddings} cross_encoder={CFG.use_cross_encoder}")
    embedder = Embedder()

    # ---- 1) train matcher on TRAIN (sampled S1) ----
    s1 = ingest.add_normalized(ingest.load_side("source1", "train"))
    right = ingest.add_normalized(ingest.load_side("right", "train"))
    gt = ingest.load_ground_truth("train")
    rng = np.random.RandomState(CFG.seed)
    keep = set(s1["entity_id"].sample(min(train_sample, len(s1)), random_state=CFG.seed))
    hold = set(s1["entity_id"]) - keep
    hold = set(list(hold)[:max(len(keep) // 5, 1)])

    Xtr, ytr, Xho, pho = [], [], [], []
    truth_ho = {}
    country_of = {}      # s1_id -> country, for per-country threshold tuning (#3)
    for c in _countries(s1):
        rc = _slice_country(right, c)
        s1c = _slice_country(s1, c)
        fit_c = s1c[s1c["entity_id"].isin(keep)].reset_index(drop=True)
        ho_c = s1c[s1c["entity_id"].isin(hold)].reset_index(drop=True)
        if len(fit_c):
            _, X, pr, y, _ = _candidates_and_features(fit_c, rc, embedder, f"ftr_{c}", gt)
            if len(X): Xtr.append(X); ytr.append(y)
        if len(ho_c):
            _, X, pr, y, _ = _candidates_and_features(ho_c, rc, embedder, f"fho_{c}", gt)
            if len(X): Xho.append(X); pho.extend(pr)
            for sid in ho_c["entity_id"]:
                truth_ho[sid] = gt.get(sid, set())
                country_of[sid] = c
        del rc; gc.collect()
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    mdl = matcher.Matcher().train(Xtr, ytr)
    mdl.save(CFG.w("matcher_lgbm.txt"))
    thr = CFG.threshold
    thr_by_c, unseen_margin = {}, 0.10
    if Xho:
        thr_by_c, thr, f = score.tune_threshold_by_country(
            mdl.predict(np.vstack(Xho)), pho, truth_ho, country_of)
        print(f"[full] global threshold={thr}  holdout F_0.5={f:.4f}")
        print(f"[full] per-country thresholds={ {k: round(v,3) for k,v in thr_by_c.items()} }")
    del s1, right, gt; gc.collect()

    # ---- 2) run TEST ----
    s1t = ingest.add_normalized(ingest.load_side("source1", "test"))
    rightt = ingest.add_normalized(ingest.load_side("right", "test"))
    all_ids = s1t["entity_id"].tolist()
    match_map, cand_map = {}, {}
    for c in _countries(s1t):
        s1c = _slice_country(s1t, c)
        rc = _slice_country(rightt, c)
        cands, X, pairs, _, rec = _candidates_and_features(s1c, rc, embedder, f"test_{c}")
        p = mdl.predict(X)
        if CFG.use_cross_encoder:
            # #14 selective rescoring: only run the heavy cross-encoder on BORDERLINE pairs
            # (gbdt prob in [0.55,0.80]) where it acts as tie-breaker. Confident pairs untouched.
            band = [i for i, s in enumerate(p) if 0.55 <= s <= 0.80]
            if band:
                sub_pairs = [pairs[i] for i in band]
                ce = matcher.cross_encoder_scores(sub_pairs, rec)
                blended = matcher.blend(p[band], ce)
                for k, i in enumerate(band):
                    p[i] = blended[k]
        # #3 per-country theta; unseen country (e.g. French) -> stricter (global + margin)
        thr_c = thr_by_c.get(c, min(thr + unseen_margin, 0.95))
        print(f"[full] country={c} threshold={thr_c:.3f}{' (unseen->strict)' if c not in thr_by_c else ''}")
        mm = resolve.resolve(pairs, p, thr_c, rec, CFG.enforce_one_s1)
        match_map.update(mm)
        for sid, lst in cands.items():
            cand_map[sid] = {rid for rid, _ in lst}
        del rc, X, pairs, rec; gc.collect()

    _write(CFG.o("matching_results.tsv"), "matched_entity_ids",
           resolve.to_submission_rows(all_ids, match_map))
    _write(CFG.o("candidate_pairs.tsv"), "candidate_entity_ids",
           resolve.to_submission_rows(all_ids, cand_map))
    print(f"[full] wrote outputs for {len(all_ids)} test S1 entities -> {CFG.out_dir}/")


def _write(path, col2, rows):
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col2}\n")
        for sid, ids in rows:
            f.write(f"{sid}\t{ids}\n")
