"""Central config. Everything tunable lives here."""
from __future__ import annotations
import os
from dataclasses import dataclass, field


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class Config:
    # ---- paths (relative to dataset root; override via env DATA_ROOT) ----
    data_root: str = os.environ.get("DATA_ROOT", "dataset")
    work_dir: str = os.environ.get("WORK_DIR", "artifacts")   # caches: embeddings, indexes, models
    out_dir: str = os.environ.get("OUT_DIR", "output")

    # ---- compute ----
    device: str = field(default_factory=_detect_device)
    n_jobs: int = int(os.environ.get("N_JOBS", os.cpu_count() or 4))
    embed_batch: int = int(os.environ.get("EMBED_BATCH", 256))
    chunk_rows: int = 500_000            # TSV read chunk (memory bound)

    # ---- models (all MIT/Apache, <=8B  -> respects competition rules) ----
    embed_model: str = os.environ.get("EMBED_MODEL", "sentence-transformers/LaBSE")  # language-agnostic
    embed_model_light: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    cross_encoder: str = os.environ.get("CROSS_ENCODER", "cross-encoder/stsb-roberta-base")  # or fine-tune deberta
    use_embeddings: bool = os.environ.get("USE_EMBEDDINGS", "1") == "1"
    use_cross_encoder: bool = os.environ.get("USE_CROSS_ENCODER", "1") == "1"
    embed_dim_reduce: int = 0            # 0 = keep full dim; else PCA to this many dims (memory on low-RAM)

    # ---- blocking ----
    top_k: int = int(os.environ.get("TOP_K", 25))     # candidates kept per S1 (per blocker), then unioned
    tfidf_ngram: tuple = (2, 4)
    tfidf_max_features: int = 2 ** 20
    ivf_threshold: int = 1_500_000       # right-pool size above which faiss uses IVF instead of Flat
    ivf_nlist: int = 4096
    ivf_nprobe: int = 32
    same_country_only: bool = True       # matches are same-country in this data; keep tiny cross tail off by default

    # ---- matcher ----
    matcher: str = os.environ.get("MATCHER", "lgbm")   # lgbm | blend (lgbm+cross-encoder)
    neg_per_pos: int = 3                 # hard-negative sampling ratio for training

    # ---- decision ----
    threshold: float = float(os.environ.get("THRESHOLD", 0.5))  # overwritten by F0.5 tuning
    enforce_one_s1: bool = True          # golden constraint: S1 is deduped -> right record maps to <=1 S1

    # ---- dev / validation ----
    val_frac: float = 0.1
    seed: int = 42

    def p(self, *parts) -> str:
        return os.path.join(self.data_root, *parts)

    def w(self, *parts) -> str:
        os.makedirs(self.work_dir, exist_ok=True)
        return os.path.join(self.work_dir, *parts)

    def o(self, *parts) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        return os.path.join(self.out_dir, *parts)


CFG = Config()
