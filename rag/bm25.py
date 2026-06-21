"""BM25 sparse index over the same chunks as the dense FAISS index.

Provides the lexical signal for hybrid retrieval (dense + sparse fused via RRF in
``retrieve.py``). Built/persisted alongside the FAISS index from the same
``chunks.jsonl``, so row identity is the chunk's ``chunk_id``.
"""
from __future__ import annotations

import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import Chunk, load_chunks
from .config import RagConfig

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Index:
    def __init__(self, bm25: BM25Okapi, chunks: list[Chunk]):
        self.bm25 = bm25
        self.chunks = chunks

    def search(self, query: str, n: int) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        n = min(n, len(self.chunks))
        top = np.argsort(scores)[::-1][:n]
        return [(self.chunks[i], float(scores[i])) for i in top]


def build_bm25(config: RagConfig | None = None) -> Bm25Index:
    config = config or RagConfig()
    chunks = load_chunks(config)
    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
    config.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with config.bm25_path.open("wb") as f:
        pickle.dump(bm25, f)
    return Bm25Index(bm25, chunks)


def load_bm25(config: RagConfig | None = None) -> Bm25Index:
    config = config or RagConfig()
    if not config.bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not built yet: {config.bm25_path}. Run build_bm25(config) first."
        )
    with config.bm25_path.open("rb") as f:
        bm25 = pickle.load(f)
    return Bm25Index(bm25, load_chunks(config))
