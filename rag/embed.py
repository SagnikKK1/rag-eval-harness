"""Embedding wrapper around the sentence-transformers model.

Kept identical to the original Sparkathon pipeline (``all-MiniLM-L6-v2``, 384-d)
for continuity. Embeddings are L2-normalized so a FAISS inner-product index
yields cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import RagConfig


@lru_cache(maxsize=4)
def get_model(name: str) -> SentenceTransformer:
    """Load (and cache) a SentenceTransformer by name."""
    return SentenceTransformer(name)


class Embedder:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig()
        self.model = get_model(self.config.embed_model)

    @property
    def dim(self) -> int:
        # Method was renamed across sentence-transformers versions; support both.
        getter = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        return getter()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings (cosine-ready)."""
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
