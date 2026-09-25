"""
Embeddings for cross-language linking (the key to India/France <-> English).
Uses LaBSE (language-agnostic, Apache-2.0) by default -> same meaning across scripts
maps to nearby vectors WITHOUT transliteration. Falls back to a lighter multilingual
model if requested. GPU strongly recommended (4060 / AWS). Cached to disk (memmap).
"""
from __future__ import annotations
import os
import numpy as np
from .config import CFG


class Embedder:
    def __init__(self, model_name: str | None = None, light: bool = False):
        self.model_name = model_name or (CFG.embed_model_light if light else CFG.embed_model)
        self._model = None

    def _lazy(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=CFG.device)
        return self._model

    def encode(self, texts, cache_key: str | None = None) -> np.ndarray:
        """Return float32 L2-normalized embeddings. Cached by key if given."""
        if cache_key:
            path = CFG.w(f"emb_{cache_key}.npy")
            if os.path.exists(path):
                return np.load(path, mmap_mode="r")
        model = self._lazy()
        emb = model.encode(
            list(texts),
            batch_size=CFG.embed_batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        if CFG.embed_dim_reduce and emb.shape[1] > CFG.embed_dim_reduce:
            emb = _pca_reduce(emb, CFG.embed_dim_reduce)
        if cache_key:
            np.save(CFG.w(f"emb_{cache_key}.npy"), emb)
        return emb


def _pca_reduce(x: np.ndarray, dim: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    p = PCA(n_components=dim, random_state=CFG.seed)
    out = p.fit_transform(x).astype("float32")
    # renormalize for cosine/IP
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n
