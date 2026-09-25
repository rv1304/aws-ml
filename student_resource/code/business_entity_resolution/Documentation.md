# ML Challenge 2026: Business Entity Resolution — Methodology

**Team Name:** [Your Team Name]
**Team Members:** [List]
**Submission Date:** [Date]

---

## 1. Executive Summary
A two-stage entity-resolution pipeline (multi-blocker candidate generation → learned pairwise
matcher → constrained graph resolution). Core innovations: (a) per-field-type comparators,
(b) language-agnostic LaBSE embeddings to link business names across scripts (Devanagari/Tamil/
Kannada/Latin/French) without transliteration, (c) a precision-first resolver that exploits the
"Source-1 is deduplicated" constraint to forbid many-S1 clusters, tuned directly for macro F_0.5.

## 2. Methodology

### 2.1 Problem Analysis (EDA findings)
- 2.21 M S1 train entities; avg 3.67 matches/S1 (range 1–11); **5.6 % singletons**.
- Matches are **same-country** in practice → country is a strong hard block. Test adds **France**
  (unseen in train) → country handled as an **open set**, never hard-coded.
- Dominant noise: transliteration (Indic scripts vs Latin), typos, legal-suffix variance
  (Ltd/Limited), address abbreviation/reordering, missing/empty addresses, state abbrev vs full.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier + Graph resolution (Hybrid).
**Core Innovation:** cross-language embedding blocking + precision-first resolver with the
one-S1-per-entity constraint, optimized for F_0.5.

## 3. Candidate Generation (Blocking)
Per-country union of complementary blockers, capped per S1:
- **Blocking keys:** char-ngram TF-IDF (SVD-reduced) ANN; **LaBSE embedding** ANN (cross-language);
  Metaphone phonetic-key buckets.
- **Recall protection:** union of three independent signals so a true match missed by one is caught
  by another; recall-ceiling vs candidate-size measured on a held-out split to trade off the
  blocking prize.
- `candidate_pairs.tsv` is the exact set fed to the matcher (last stage before scoring).

## 4. Matching Model
**Features (per-field comparators):**
- Name: token-set / token-sort / ratio / partial (rapidfuzz), Jaro-Winkler, normalized Levenshtein,
  acronym equality, no-space ratio, legal-suffix equality.
- Address: token-set/ratio, token Jaccard, PIN/ZIP equality, house-number overlap, empty-address flag.
- Cross-modal: LaBSE embedding cosine; blocking score.
**Model:** LightGBM (precision-aware via `scale_pos_weight`), trained on blocking-derived
**hard negatives**; optional cross-encoder (Ditto-style) deep re-scoring blended in.

## 5. Decision & Resolution
- Threshold chosen to **maximize macro F_0.5** on a train holdout (precision-favored, ≈0.9).
- **Hard veto:** drop a candidate whose PIN hard-contradicts unless names are near-identical.
- **Golden constraint:** Source-1 is deduplicated ⇒ each right record is assigned to at most one S1
  (highest-scoring). Prevents mega-cluster false merges that F_0.5 punishes 2×.
- Singletons: predict empty when nothing clears threshold (scores 1.0).

## 6. Validation
Custom scorer replicating the exact macro-F_0.5 (per-entity, singletons included). Split by S1
entity (no pair leakage). Threshold and model choices selected on this split, not the public LB.

## 7. Reproducibility & Fair Play
- Fully offline; no external DB/API/geocoding/internet augmentation.
- Models: LaBSE (Apache-2.0), multilingual-e5 (MIT), cross-encoder (MIT/Apache), all ≤8B params.
- `libpostal` optional (offline OSM-trained normalizer, not a lookup service); pipeline runs without it.
- End-to-end run instructions in `README.md`; dependencies pinned in `requirements.txt`.
