"""Semantic signals: embedding-outlier novelty and relevance-to-profile.

The embedding backend is pluggable. With ``sentence-transformers`` installed you
get real semantic vectors; without it we fall back to dependency-free signed
feature hashing, which still supports outlier/relevance ranking (less semantic,
but the tool always runs).
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

_TOKEN_RE = re.compile(r"[a-z][a-z0-9+\-]{1,}", re.I)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder:
    """Encodes texts into L2-normalized row vectors."""

    def __init__(self, backend: str = "auto", model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.requested_backend = backend
        self.model_name = model
        self.backend, self._model = self._resolve(backend, model)

    @staticmethod
    def _resolve(backend: str, model: str):
        if backend in ("auto", "sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer

                return "sentence-transformers", SentenceTransformer(model)
            except Exception:
                if backend == "sentence-transformers":
                    # explicit request failed — degrade rather than crash
                    pass
        return "hashing", None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "sentence-transformers":
            vecs = np.asarray(
                self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            return vecs
        vecs = _hash_embed(texts)
        return _l2_normalize(vecs)


def _hash_embed(texts: list[str], dim: int = 1024) -> np.ndarray:
    """Signed feature hashing (the 'hashing trick'), deterministic across runs."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for tok in _tokenize(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=5).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            out[row, idx] += sign
    return out


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _rank_unit(values: np.ndarray) -> np.ndarray:
    """Map values to [0,1] by percentile rank (robust to outliers/scale)."""
    n = len(values)
    if n <= 1:
        return np.full(n, 0.5, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return ranks / (n - 1)


def outlier_scores(matrix: np.ndarray, k: int = 8) -> np.ndarray:
    """Per-row novelty as semantic isolation: distance to k nearest neighbors.

    Higher = more isolated in embedding space = more likely an unusual/novel idea
    relative to the batch. Returned as percentile ranks in [0,1].
    """
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.full(1, 0.5, dtype=np.float32)
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)  # exclude self
    kk = min(k, n - 1)
    # top-kk neighbor similarities per row
    nn = np.sort(sims, axis=1)[:, -kk:]
    mean_nn_sim = nn.mean(axis=1)
    isolation = 1.0 - mean_nn_sim  # larger distance => more novel
    return _rank_unit(isolation)


def keyword_score(text: str, keywords: list[str], cap: int = 4) -> float:
    low = text.lower()
    hits = sum(1 for kw in keywords if kw in low)
    return min(1.0, hits / cap) if cap else 0.0


def relevance_scores(
    matrix: np.ndarray,
    profile_vec: np.ndarray,
    texts: list[str],
    keywords: list[str],
) -> np.ndarray:
    """How relevant each paper is to the interest profile, in [0,1].

    Blend of cosine similarity to the profile embedding and keyword overlap, so
    it stays meaningful under both the semantic and hashing backends.
    """
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    cos = matrix @ profile_vec.reshape(-1)
    # MiniLM topical cosine typically lands ~0.05..0.5; map that band to [0,1].
    semantic = np.clip((cos - 0.05) / 0.45, 0.0, 1.0)
    kw = np.array([keyword_score(t, keywords) for t in texts], dtype=np.float32)
    return np.clip(0.65 * semantic + 0.35 * kw, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=4)
def get_embedder(backend: str, model: str) -> Embedder:
    return Embedder(backend=backend, model=model)
