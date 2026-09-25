# Amazon ML Challenge 2026 — Entity Resolution — MASTER PLAN

Goal: max F_0.5 on private leaderboard + smallest candidate set (blocking prize).
Reality: 100% impossible (noisy data, macro F_0.5). Aim top-of-board via discipline, not magic.

---

## 0. THE PROBLEM IN ONE LINE
For each Source-1 business, find its true twins in Source-2 + Source-3. Match only when sure.

## 1. METRIC = LAW (F_0.5, macro per S1 entity)
- Precision weighted 2x over recall. **False merge = 2x worse than miss.**
- Per-entity averaged. One bad entity = a 0.0 dragging mean. Consistency > heroics.
- Singleton (no true match) → predict empty → **score 1.0**. Never force a match.
- Strategy: **high threshold, conservative matching, protect precision.**

## 2. HARD CONSTRAINTS (rules — break = DQ or reject)
- Final model: **MIT or Apache-2.0 license, <= 8B params.** (No Llama/Qwen non-commercial. Use permissive ones.)
- NO external data/APIs: no geocoding, no business registries, no internet lookup. Provided data ONLY.
- Read TSV with `sep="\t"` always. (commas live inside fields.)
- Country = OPEN SET. Test has France (not in train). Do NOT hardcode {US, India}. No one-hot on country values.
- Output: every test S1 = exactly one row. Empty allowed. S2-/S3- IDs only. No dup IDs in a list. Matches subset of candidates.
- Validate with `utils/validate_submission.py` before every upload.

## 3. ARCHITECTURE — 5 STAGES
```
LOAD -> NORMALIZE -> BLOCK (candidate gen) -> MATCH (pair classify) -> DECIDE (threshold) -> FORMAT + VALIDATE
```
Two-stage ER (blocking + matching) is the proven SOTA shape (Magellan, Ditto, DeepMatcher).

---

## 4. BEST LIBRARIES (all permissive license, offline-capable)
| Job | Lib | Why |
|-----|-----|-----|
| ER scaffolding | `recordlinkage` | blocking+compare+classify framework |
| ER framework | `py_entitymatching` (Magellan) | end-to-end ER, feature auto-gen |
| Fast string sim | `rapidfuzz` | Jaro-Winkler, Levenshtein, token ratios, C-fast |
| Phonetic | `jellyfish` | Soundex, Metaphone, NYSIIS for names |
| Transliteration | `unidecode` | India/France accents -> ascii |
| Vectorize | `scikit-learn` TfidfVectorizer | char n-gram TF-IDF |
| ANN search | `faiss` (or `hnswlib`) | fast nearest-neighbor blocking on GPU |
| Embeddings | `sentence-transformers` | dense semantic blocking (pick Apache model) |
| Matcher model | `lightgbm` + `xgboost` + `catboost` | pair classifier ensemble |
| Deep ER (SOTA) | Ditto-style cross-encoder | fine-tune small transformer on pairs |
| Dedup util | `networkx` | connected-components for match grouping |

Embedding/model picks that satisfy license+8B:
- `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, tiny, fast) — blocking + features
- `BAAI/bge-small-en-v1.5` / `bge-base` (MIT) — stronger embeddings
- `microsoft/deberta-v3-base` (MIT) — cross-encoder fine-tune for matcher
- `intfloat/e5-base-v2` (MIT) — multilingual-ish embeddings (helps France)
- multilingual: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache) — India/France scripts
CHECK each license before final submission. Log them in the doc.

---

## 5. STAGE-BY-STAGE (every technique)

### STAGE A — LOAD (bulletproof)
- `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)` — keep everything string, no NaN surprises.
- Source of a record = its id prefix (S1-/S2-/S3-) + which file. Add a `source` col yourself.
- Concat S2+S3 into one "right" pool (matches can come from either).

### STAGE B — NORMALIZE (biggest quiet score lever)
Build a strong cleaner. Apply to name + address separately.
Name normalization:
- lowercase, `unidecode`, strip punctuation (keep meaningful digits)
- `&` -> `and`, collapse whitespace
- expand legal/abbrev dict: corp->corporation, pvt->private, ltd->limited, inc->incorporated, co->company, llp, plc, gmbh, srl...
- extract + strip legal suffix into a separate field (suffix match = weak signal, not identity)
- handle DBA / "trade name" splits
- token-sort (sorted tokens) variant for word-order transposition
- keep an "initials/acronym" form (e.g., "H D F C")
Address normalization:
- expand: rd->road, st->street, ave->avenue, blvd, ln, apt, fl, near, opp, sec/sector...
- extract structured bits WITHOUT external geocode: PIN/ZIP regex, house/plot number, unit number, city/state tokens
- normalize transliteration variants; strip landmark noise ("near sbi atm") into separate landmark field
- per-country light rules (India PIN 6-digit, US ZIP 5) but do it by REGEX PATTERN not by country-name hardcode (France-safe)
Make normalization language/script tolerant (India Hindi/regional, France accents).

### STAGE C — BLOCK / CANDIDATE GENERATION (recall ceiling + prize)
This sets your MAX possible score. If true twin not in candidates, unreachable. But bloated set loses the blocking prize (smaller = ranked higher). Optimize recall-per-size.

Multi-blocker UNION (run several, merge candidate sets, dedup):
1. **Char n-gram TF-IDF + ANN**: char 3-5 grams on name -> TF-IDF -> cosine top-K via faiss. Catches typos/order.
2. **Dense embedding ANN**: sentence-transformer embed "name || address" -> faiss top-K. Catches semantic/translit.
3. **Token / MinHash-LSH blocking**: shared rare tokens (`datasketch` MinHashLSH). Cheap high-recall.
4. **Phonetic block**: Soundex/Metaphone key on name -> group. Catches heavy typos/transliteration.
5. **Address/number block**: exact PIN + street-number buckets.
Then:
- soft-block by country (same-country strongly preferred) but keep small cross-country tail (France unseen — don't over-trust).
- Rank union by max similarity, keep top-K per S1 (tune K, e.g. 20-50). Then a cheap pre-filter cut to keep candidate list tight.
- **Measure on validation: recall ceiling (fraction of true matches captured) vs avg candidates/S1.** Push size down while recall stays ~0.98+. Plot the curve. This is your blocking prize.
- candidate_pairs.tsv = EXACT set fed to matcher (the last stage before scoring), one row per S1.

### STAGE D — MATCH (pair classifier)
For each (S1, candidate) pair build a rich feature vector:
String features (name, then address, each):
- rapidfuzz: ratio, partial_ratio, token_sort_ratio, token_set_ratio, WRatio
- Jaro-Winkler, Levenshtein distance + normalized
- Jaccard on token sets, on char n-gram sets
- TF-IDF cosine (name), TF-IDF cosine (address)
- longest common substring / prefix
- phonetic-key equality (Soundex/Metaphone)
Structured features:
- PIN/ZIP exact match, street-number match, city token overlap, state match
- country match flag (feature, NOT filter)
- legal-suffix match, acronym match, length ratios, token-count diffs
- embedding cosine (name, address, combined)
- "one is substring of other" flags
Model (ensemble):
- Primary: **LightGBM** on features (fast, strong, handles missing).
- Add XGBoost + CatBoost, average probs (diversity).
- SOTA leg: **Ditto-style cross-encoder** — serialize pair as
  `"COL name VAL <n1> COL addr VAL <a1> [SEP] COL name VAL <n2> COL addr VAL <a2>"`,
  fine-tune deberta-v3-base (MIT) for binary match. Blend its prob with GBDT.
Training data:
- Positives = ground-truth pairs.
- Negatives = **hard negatives** = candidates from blocking that are NOT true matches (realistic!). Do NOT use random negatives — they're too easy and inflate CV.
- Balance / weight classes; F_0.5 cares about precision so weight false-positives.

### STAGE E — DECIDE (tune to F_0.5)
- Get match probability per pair. Threshold to maximize **F_0.5 on validation** (higher than F1 optimum — precision favored).
- Per S1: keep ALL candidates >= threshold (many-to-many allowed). If none clear -> empty (singleton).
- Optional precision guards:
  - require agreement of 2 signals (e.g., strong name AND non-contradicting address) for a match.
  - if country mismatch, raise threshold (rare true cross-country).
  - de-conflict: if a single S2/S3 record maps strongly to multiple S1, keep only best (S1 is deduped reference; a right-record usually belongs to one S1) — improves precision.
- Consider per-country / per-source threshold tuning (but keep France on a sane default since unseen).

### STAGE F — FORMAT + VALIDATE
- One row per test S1 (include ALL, even empties).
- comma-join ids, no dup, S2-/S3- only, must exist in test files.
- matches subset of candidates (assert!).
- Run `python3 utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test` -> must PASS before upload.

---

## 6. VALIDATION (private LB decides — this is how you avoid crashing)
- Write THEIR exact scorer first: per-entity precision/recall -> F_0.5 -> macro-average, singletons scored (empty-correct=1.0, false-merge=0.0).
- Hold out a validation split BY S1 ENTITY (not by pair) so no leakage. Keep the split's true matches from ground truth.
- Every change: score on validation. Trust it over public LB (public = subset, noisy).
- Watch validation-vs-public gap; if diverge, your CV split is wrong — fix before trusting anything.
- Final submission choice: best VALIDATION F_0.5, not best public. Submit a safe blend too.

## 7. ALL EDGE CASES (checklist — each one a real score leak)
- [ ] France (unseen country) rows: must appear, don't crash, don't hardcode country.
- [ ] Empty/partial address, missing PIN/state: features must handle blank gracefully (keep_default_na=False).
- [ ] Business with MANY true matches: don't cap to top-1; keep all above threshold.
- [ ] Singletons: don't force matches; empty is correct + full credit.
- [ ] Transliteration (Hindi/regional/French accents): unidecode + multilingual embeddings.
- [ ] Word-order transposition: token_sort/token_set ratios.
- [ ] `&` vs "and", punctuation, casing: normalized away.
- [ ] Legal suffix diff (Ltd vs Limited vs blank): expanded, suffix not treated as identity.
- [ ] Same name, different city/business (chains, franchises): address must disambiguate -> precision guard.
- [ ] One S2 record claimed by 2 S1: de-conflict to best.
- [ ] Duplicate ids in output / missing S1 rows / non-existent ids: validator catches — run it.
- [ ] TSV read without sep -> single column: always sep="\t".
- [ ] matched id not in candidate set: pipeline bug — assert subset.
- [ ] Very large right-pool: ANN + blocking scales; don't do full O(n*m).

## 8. ALL OPTIMIZATION TECHNIQUES (squeeze every point)
- Deeper normalization + bigger abbrev dict (cheapest gains).
- Multi-blocker union -> raise recall ceiling; then trim -> blocking prize.
- Hard-negative mining from blocking for a realistic matcher.
- Ensemble GBDT (LGB+XGB+CAT) + cross-encoder blend.
- Threshold tuned to F_0.5 (per-country if it helps on val).
- Precision guards / two-signal rule / de-confliction.
- Pseudo-labeling: high-confidence test matches added to train, retrain matcher (careful, verify on val).
- Feature importance pruning -> keep strong features, less overfit.
- Calibrate probabilities (isotonic) so threshold is stable across public/private.
- Seed-averaging + k-fold matcher for stability (macro metric loves stability).
- Post-process sanity pass: drop matches that violate hard structured contradictions (different PIN AND different city AND weak name).

## 9. AWS USAGE (credits) — you have them, use day 0
- Launch a GPU instance (g5/g4dn) for: faiss-GPU blocking, embedding extraction, cross-encoder fine-tune.
- Big-RAM instance if right-pool huge (blocking in memory).
- Set up env FIRST (day 0), preload models offline (no internet lookup during matching — respects fair-play).
- Store data + checkpoints on the instance / EBS. Snapshot so you don't lose work.
- Keep a CPU box for fast GBDT iteration; GPU only for embeddings/transformer.

## 10. TIMELINE (72h, starts 25 Sep)
- H0-3: setup AWS+env, load data, EDA (look at rows by hand, noise patterns, label stats). Write F_0.5 scorer + val split.
- H3-8: normalize + simplest blocker + GBDT baseline -> first VALID submission. Confirm val tracks public.
- H8-30: strengthen blocking (multi-blocker, measure recall/size), rich features, tune matcher.
- H30-50: cross-encoder fine-tune + ensemble; threshold tuning to F_0.5; precision guards.
- H50-64: pseudo-label, calibration, de-confliction, edge-case sweep, blocking trim for prize.
- H64-72: lock final by VALIDATION score, package zip, fill Documentation_template.md, run validator, sleep before deadline.

## 11. TEAM SPLIT (2-4)
- A: pipeline + val scorer + submission + validator (owns correctness)
- B: blocking (recall/size optimization, the prize)
- C: matcher features + GBDT/cross-encoder tuning
- D: EDA + normalization dict + edge cases + methodology doc

## 12. FINAL PACKAGE
```
<team>_submission.zip
  output/matching_results.tsv
  output/candidate_pairs.tsv
  code/business_entity_resolution/src/...
  code/business_entity_resolution/README.md      (exact run steps)
  code/business_entity_resolution/requirements.txt (pinned)
  Documentation_template.md  (methodology: blocking strategy, model, features, licenses used)
```
Judges review top teams' zips by hand: must reproduce end-to-end, clean, licenses stated.

## 13. WHERE RANKS ARE WON (top 5)
1. Precision discipline under F_0.5 (don't over-match; respect singletons).
2. Blocking recall-vs-size (uncapped ceiling + tiny set = both leaderboard & prize).
3. Normalization depth (India/France/typos/abbrev).
4. Trustworthy validation (private LB survivor).
5. Clean reproducible package + strong methodology doc (finale + audit).
```

===================================================================
# PART II — PRODUCTION ARCHITECTURE (v2, the real design)
===================================================================
Framed as a 3-layer data pipeline (like a medallion / production ER system:
Splink, Dedupe, Zingg, AWS Entity Resolution all follow this shape).
Each layer has ONE job. Data flows one direction. Everything reproducible.

## LAYER 1 — SOURCE -> DB  (Ingest / "Bronze")
Job: get raw records into a clean, typed, provenance-tagged store. No matching yet.
- Parse all 6 TSVs with sep="\t", dtype=str, keep_default_na=False.
- Router by id prefix: S1-/S2-/S3- -> source tag. (ID is structural, never fuzzy-matched.)
- Build ONE unified record table: [entity_id, source, business_name, business_address, country, raw_*].
  Keep raw originals untouched (audit + reproduce).
- Right-pool = S2 UNION S3 (matches come from both).
- Sanity stats: counts per source, per country, null/blank rates, name/addr length dist,
  duplicate raw strings. This is your EDA gate before touching anything.
Output: canonical_records table (typed, immutable raw + working copy).

## LAYER 2 — DB -> ENRICHMENT  (Normalize + Featurize / "Silver")
Job: turn messy strings into comparable signals. THIS is where per-field-type rules live.
Do NOT apply one rule to all fields. Each field type gets its own treatment:

### PER-FIELD-TYPE HANDLING (core of your request)
| Field | Type | Rules / signals produced |
|-------|------|--------------------------|
| entity_id | STRUCTURAL / exact | prefix->source router; NEVER fuzzy. used for join/provenance/dedup only |
| business_name | SEMANTIC + FUZZY | lowercase, unidecode, &->and, strip punct; expand legal/abbrev dict; strip+store legal suffix separately; token-sort form; acronym/initials form; produce: char3-5 TF-IDF vec, word TF-IDF, phonetic key (Metaphone/Soundex), dense embedding (semantic/DBA/translation) |
| business_address | STRUCTURED PARSE + component compare | expand abbrev (Rd->Road...); regex-extract PIN/ZIP, house/plot no., unit, street name, city, state; landmark ("near X") -> separate weak field; produce per-component tokens + embedding. NO external geocode (fair-play) |
| country | CATEGORICAL open-set | soft block key + feature flag ONLY. never hardcode {US,India}; France-safe. no fixed one-hot |
| numerics (PIN/plot) | EXACT / edit-dist-1 | exact match or off-by-one; strong disambiguator |

Rule: a comparator is chosen PER FIELD. IDs -> equality. Names -> string+phonetic+semantic.
Addresses -> component-wise. Country -> categorical. Never a global similarity.

### SIGNAL TOOLKIT (best per job — your "use the best thing")
- String distance: rapidfuzz (Jaro-Winkler, Levenshtein, token_sort/set ratio) -> typos, word-order.
- Bag of words / TF-IDF: char + word n-grams (sklearn) -> robust lexical overlap, blocking.
- Phonetic: jellyfish Metaphone/Soundex -> heavy typos + transliteration.
- Embeddings (semantic "LLM finds similar meaning"): sentence-transformers (bge / e5 / MiniLM,
  MIT/Apache) -> "Kumar Traders" ~ "Kumar Trading Co", cross-language, DBA/translation.
- Cross-encoder (deep semantic, the strongest LLM leg): Ditto-style, fine-tune deberta-v3-base
  (MIT, <8B) reading BOTH records at once -> best pairwise judge. THIS is the "LLM decides same-or-not".
Decision: use ALL as features. TF-IDF/phonetic/embeddings drive BLOCKING; full set + cross-encoder
drive MATCHING. No single technique — they're complementary. (embeddings alone = too many false
merges under F_0.5; string alone = misses semantic; combine.)

Output: feature/enrichment store — per record: normalized fields, blocking keys
(tfidf vec, phonetic key, embedding, address components). Cache to disk (parquet) so
Layer 3 iterates fast without recompute.

## LAYER 3 — CUSTOMER -> FINAL  (Match + Resolve + Serve / "Gold")
Job: produce the two deliverables the "customer" (submission) consumes.
"Customer" = the scored output: matching_results.tsv + candidate_pairs.tsv.

Sub-steps:
### 3a. BLOCK (candidate generation) — lazy, multi-blocker union
- Blocking is a lazy iterator, NOT a full cross-join (industry pattern; scales to millions).
- Union of: char-TFIDF ANN (faiss) + embedding ANN + MinHash-LSH token block (datasketch)
  + phonetic-key block + PIN/street-number block.
- Soft-block by country; small cross-country tail (France).
- candidate_pairs.tsv = this exact set (last stage before model). Measure recall ceiling vs size.

### 3b. MATCH (pairwise score)
- Fellegi-Sunter probabilistic model via SPLINK (DuckDB backend, offline, no external data):
  EM learns per-field match weights -> interpretable, calibrated match probability.
  (This is how UK gov / enterprise link records. Great for methodology doc + a strong baseline.)
- PLUS LightGBM/XGBoost/CatBoost on the full feature vector (hard-negative trained).
- PLUS cross-encoder (deberta) deep semantic leg.
- Blend probabilities (Splink + GBDT + cross-encoder). Calibrate (isotonic).

### 3c. RESOLVE — GRAPH / TRANSITIVE CLOSURE (done SAFELY)
Build graph: nodes=records, edges=matched pairs (weight=prob above threshold).
Naive transitive closure (blind Union-Find) is DANGEROUS: one false edge chains unrelated
businesses into a mega-cluster -> precision collapses -> death under F_0.5.
Use CONSTRAINED resolution (matches enterprise "self-serve ER" lessons):
1. Highest-scoring-neighbor first: each record links to its single best neighbor above
   threshold; edges do NOT blindly propagate.
2. HARD VETOES: never merge across a hard contradiction (different PIN AND different city,
   or country conflict with weak name). Veto blocks the chain.
3. GOLDEN CONSTRAINT (this problem): Source 1 is DEDUPLICATED -> a resolved entity may contain
   AT MOST ONE S1. If a cluster pulls in 2+ S1 records, SPLIT it (they are distinct by definition).
   Powerful precision guard unique to this task — enforce it.
4. Controlled transitive use for RECALL only: if S2-x matches S1-a, and S3-y matches S2-x,
   propose S3-y as a candidate for S1-a — but RE-SCORE S1-a vs S3-y directly through the matcher
   before accepting. Propagate candidates, never propagate decisions.
Connected components / Union-Find (networkx or scipy) with the above guards.

### 3d. DECIDE (threshold to F_0.5)
- Threshold tuned to maximize F_0.5 on validation (higher than F1 optimum -> precision favored).
- Per S1: keep all resolved S2/S3 (many-to-many). None -> empty (singleton = 1.0).
- Two-signal precision guard: require name-strong AND address-non-contradicting for a match.

### 3e. SERVE (format + validate)
- matching_results.tsv + candidate_pairs.tsv, one row per test S1, all rules enforced.
- assert matches subset of candidates; assert at-most-one-S1 constraint held.
- run utils/validate_submission.py -> PASS before upload.

## DATA FLOW (one direction, reproducible)
raw TSV
  -> [L1 ingest] canonical_records
  -> [L2 enrich] normalized fields + blocking keys + embeddings (cached parquet)
  -> [L3a block] candidate_pairs
  -> [L3b match] scored pairs (Splink + GBDT + cross-encoder blend)
  -> [L3c resolve] constrained graph clustering (veto + 1-S1-per-entity)
  -> [L3d decide] threshold to F_0.5
  -> [L3e serve] matching_results.tsv + candidate_pairs.tsv -> validate -> submit

## INDUSTRY TECH REFERENCED (cite in methodology doc)
- Fellegi-Sunter model + EM weight learning -> Splink (DuckDB/Spark, offline).
- Active-learning FS -> Dedupe.
- Deep ER (LM pairwise) -> Ditto / DeepMatcher (cross-encoder).
- Blocking: MinHash-LSH, sorted-neighborhood, canopy, ANN (faiss).
- Clustering: connected components / Union-Find with hard-veto + highest-neighbor
  (avoids mega-cluster false-merge, the known enterprise failure mode).
- Managed refs (AWS Entity Resolution, Zingg, Senzing) = same shape; we build our own
  because rules forbid external resolution services.
NOTE fair-play: everything runs OFFLINE on provided data only. No external DB/API/geocode.
