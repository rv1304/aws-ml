# Business Entity Resolution — Amazon ML Challenge 2026

Match every **Source-1** business to its true records in **Source-2 / Source-3** across
3 noisy, multilingual, no-shared-id sources. Metric: **macro F_0.5** (precision-heavy) + a
**blocking prize** (smaller candidate set per S1 ranks higher).

## Pipeline (3 layers)
```
L1 ingest    src/ingest.py      load TSVs, provenance by id prefix
L2 enrich    src/normalize.py   PER-FIELD normalization + src/embed.py cross-lang embeddings
             src/embed.py
L3 resolve   src/block.py       candidate generation (multi-blocker union)
             src/features.py    per-field pair comparators
             src/matcher.py     LightGBM (+ optional cross-encoder blend)
             src/resolve.py     threshold + hard-veto + 1-S1-per-entity constraint
             src/score.py       exact F_0.5 macro scorer + threshold tuning
orchestrator src/pipeline.py    dev / full modes
entry        run.py
```

### Per-field-type handling (different metric per field)
- **entity_id** — structural: prefix routes source; never fuzzy-matched.
- **business_name** — fuzzy + semantic: `cleanco` legal-suffix strip, `unidecode` transliteration,
  char-ngram TF-IDF, rapidfuzz (Jaro-Winkler/token-set/Levenshtein), Metaphone phonetic, LaBSE embedding.
- **business_address** — structured: `libpostal` expand/parse (optional) else dict fallback;
  regex-extract PIN/ZIP, house/plot number, state canonicalization; token Jaccard.
- **country** — categorical **open set** (France appears only in test): soft block key + flag,
  never hard-coded / one-hot to {US, India}.

### Cross-language linking
`LaBSE` (Apache-2.0) maps the same business across scripts (Devanagari/Tamil/Kannada/Latin/French)
to nearby vectors **without transliteration** — the primary India/France ↔ English link.
Transliteration (`unidecode`) is a cheap secondary signal.

### Safe transitive resolution (avoids the mega-cluster trap)
Blind union-find chains unrelated businesses through one false edge → precision death under F_0.5.
Instead: highest-scoring-neighbor + **hard vetoes** (contradicting PIN/city) + the **golden constraint**
(Source-1 is deduplicated → a right record maps to **at most one S1**; highest-scoring S1 wins).

## Install
```bash
pip install -r requirements.txt
# GPU box (4060 / AWS): install CUDA torch + faiss-gpu instead of faiss-cpu
# OPTIONAL best address parsing: build libpostal C lib, then `pip install postal`  (see note below)
```

## Run
```bash
# know your score first: true macro F_0.5 on a train holdout
python run.py dev --sample 20000

# full pipeline -> output/matching_results.tsv + output/candidate_pairs.tsv
python run.py full --train-sample 200000

# validate format before uploading
python ../../utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir ../../dataset/test
```

### Env toggles (see `src/config.py`)
| var | default | meaning |
|-----|---------|---------|
| `DATA_ROOT` | `dataset` | dataset dir (train/ test/) |
| `OUT_DIR` | `output` | where results are written |
| `WORK_DIR` | `artifacts` | embedding / model cache |
| `USE_EMBEDDINGS` | `1` | LaBSE cross-language blocking (set `0` on low-RAM/CPU) |
| `USE_CROSS_ENCODER` | `1` | deep semantic re-scoring (set `0` to skip) |
| `EMBED_MODEL` | LaBSE | swap to `paraphrase-multilingual-MiniLM-L12-v2` for speed |
| `EMBED_DIM_REDUCE` | `0` | PCA-reduce embeddings (e.g. `128`) to fit low RAM |
| `N_JOBS` | all cores | parallelism |

## Hardware guidance
- **Full-scale (test ≈ 11.7M rows) wants GPU + RAM.** Best on AWS (g5/L4) or the RTX 4060 laptop.
- **16 GB RAM**: process per-country (already done); keep `USE_EMBEDDINGS=1` only with `EMBED_DIM_REDUCE=128`.
- **≤10 GB RAM / CPU-only**: run with `USE_EMBEDDINGS=0 USE_CROSS_ENCODER=0` (TF-IDF+phonetic blocking still strong).
- **macOS**: `run.py` auto-sets `OMP_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK=TRUE` to avoid a faiss/OpenMP segfault.

## Fair-play / licenses
- No external DB / API / geocoding / internet augmentation. All offline on provided data.
- Models: LaBSE (Apache-2.0), multilingual-e5 (MIT), cross-encoder roberta/deberta (MIT/Apache), all ≤8B.
- **libpostal note:** optional; it is an OFFLINE OpenStreetMap-trained *normalizer*, not a lookup/geocode
  service. If in doubt about the rules, run without it (dict fallback) — pipeline works either way.
