"""
Layer 3a: BLOCKING / candidate generation.  Multi-blocker UNION, per country.
Blockers (complementary):
  1. embedding ANN (LaBSE)      -> cross-language + semantic + typo tolerant
  2. char-ngram TF-IDF ANN      -> lexical overlap embeddings can miss
  3. phonetic-key block         -> heavy typos / transliteration
  4. exact-key blocks           -> nospace-name / sorted-tokens / name+pin
                                   (zero-recall-loss for clean dupes; ANN can rank them out)
Output = candidate_pairs: dict[s1_id] -> list[(right_id, score)].
This exact set is written to candidate_pairs.tsv (the last stage before matching).
Recall ceiling vs candidate size is the blocking-prize tradeoff -> measured in pipeline.
"""
from __future__ import annotations
import numpy as np
import faiss
from collections import defaultdict
import jellyfish

from .config import CFG


# ------------------------------------------------------------------ faiss ANN
def _build_index(vecs: np.ndarray):
    n, d = vecs.shape
    vecs = np.ascontiguousarray(vecs, dtype="float32")
    if n > CFG.ivf_threshold:
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, CFG.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs[np.random.RandomState(CFG.seed).choice(n, min(n, 200_000), replace=False)])
        index.add(vecs)
        index.nprobe = CFG.ivf_nprobe
    else:
        index = faiss.IndexFlatIP(d)
        index.add(vecs)
    return index


def _ann(query: np.ndarray, right: np.ndarray, k: int):
    """Return (idx[n,k], sim[n,k]) top-k right rows for each query row (cosine via IP on normalized)."""
    index = _build_index(right)
    q = np.ascontiguousarray(query, dtype="float32")
    sim, idx = index.search(q, min(k, right.shape[0]))
    return idx, sim


# ------------------------------------------------------- char TF-IDF -> dense
def _tfidf_svd(right_texts, query_texts, dim=128):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=CFG.tfidf_ngram,
                          min_df=2, max_features=CFG.tfidf_max_features)
    Xr = vec.fit_transform(right_texts)
    svd = TruncatedSVD(n_components=min(dim, Xr.shape[1] - 1), random_state=CFG.seed)
    Rr = normalize(svd.fit_transform(Xr)).astype("float32")
    Xq = vec.transform(query_texts)
    Rq = normalize(svd.transform(Xq)).astype("float32")
    return Rr, Rq


# ------------------------------------------------------------- phonetic block
def _phonetic_key(name_core: str) -> str:
    toks = name_core.split()
    if not toks:
        return ""
    try:
        return jellyfish.metaphone(" ".join(toks[:2]))
    except Exception:
        return toks[0][:4]


# ------------------------------------------------------------------- assemble
def generate_candidates(s1: "pd.DataFrame", right: "pd.DataFrame",
                        emb_s1=None, emb_right=None, k: int | None = None) -> dict:
    """
    s1, right : frames with columns entity_id, name_core, addr_norm, blk_text
    emb_*     : optional precomputed embeddings aligned to frame order
    Returns dict[s1_id] -> list[(right_id, score)] (union of blockers, dedup, best score kept).
    """
    k = k or CFG.top_k
    s1_ids = s1["entity_id"].to_numpy()
    r_ids = right["entity_id"].to_numpy()
    cand = defaultdict(dict)   # s1_id -> {right_id: score}

    def _merge(idx, sim, weight):
        for i in range(idx.shape[0]):
            sid = s1_ids[i]
            row = cand[sid]
            for j, s in zip(idx[i], sim[i]):
                if j < 0:
                    continue
                rid = r_ids[j]
                v = float(s) * weight
                if v > row.get(rid, -1):
                    row[rid] = v

    # 1) embedding ANN
    if CFG.use_embeddings and emb_s1 is not None and emb_right is not None:
        idx, sim = _ann(emb_s1, emb_right, k)
        _merge(idx, sim, 1.0)

    # 2) char TF-IDF ANN
    Rr, Rq = _tfidf_svd(right["blk_text"].tolist(), s1["blk_text"].tolist())
    idx, sim = _ann(Rq, Rr, k)
    _merge(idx, sim, 0.9)

    # 3) phonetic-key exact block
    buckets = defaultdict(list)
    for j, nm in enumerate(right["name_core"].to_numpy()):
        buckets[_phonetic_key(nm)].append(j)
    for i, nm in enumerate(s1["name_core"].to_numpy()):
        key = _phonetic_key(nm)
        if not key:
            continue
        row = cand[s1_ids[i]]
        for j in buckets.get(key, [])[:k]:
            rid = r_ids[j]
            row.setdefault(rid, 0.5)

    # 4) exact-key blocks -> guarantee clean dupes survive (ANN can rank them out of top-k).
    #    high seed score so resolve/matcher see them as strong candidates.
    def _exact_block(key_fn, seed_score):
        b = defaultdict(list)
        for j in range(len(r_ids)):
            key = key_fn(right, j)
            if key:
                b[key].append(j)
        for i in range(len(s1_ids)):
            key = key_fn(s1, i)
            if not key:
                continue
            row = cand[s1_ids[i]]
            for j in b.get(key, [])[:k]:
                rid = r_ids[j]
                if seed_score > row.get(rid, -1):
                    row[rid] = seed_score

    nosp_s1, nosp_r = s1.get("name_nospace"), right.get("name_nospace")
    if nosp_s1 is not None and nosp_r is not None:
        nv1, nv2 = nosp_s1.to_numpy(), nosp_r.to_numpy()
        _exact_block(lambda df, j: (nv1 if df is s1 else nv2)[j] or "", 0.99)

    nc_s1, nc_r = s1["name_core"].to_numpy(), right["name_core"].to_numpy()
    _exact_block(lambda df, j: " ".join(sorted((nc_s1 if df is s1 else nc_r)[j].split())), 0.95)

    pin_s1, pin_r = s1.get("addr_pin"), right.get("addr_pin")
    if pin_s1 is not None and pin_r is not None:
        pv1, pv2 = pin_s1.to_numpy(), pin_r.to_numpy()

        def _name_pin(df, j):
            nc = (nc_s1 if df is s1 else nc_r)[j]
            pv = (pv1 if df is s1 else pv2)[j]
            toks = nc.split()
            return f"{toks[0]}|{pv}" if toks and pv else ""

        _exact_block(_name_pin, 0.9)

    # finalize: sort by score desc, cap
    out = {}
    cap = 3 * k
    for sid in s1_ids:
        items = sorted(cand.get(sid, {}).items(), key=lambda x: -x[1])[:cap]
        out[sid] = items
    return out
