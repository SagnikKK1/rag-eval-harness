"""Pure retrieval interface: query -> ranked chunks. No DB, no side effects.

Flow: embed query -> dense FAISS search for ``top_n`` candidates -> (optional)
cross-encoder rerank -> top ``k``. With the reranker off, it is a plain dense
top-k. Returns lightweight ``RetrievedChunk`` records carrying the provenance
``source_ids`` the eval harness scores against.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import RagConfig
from .embed import Embedder
from .index import Chunk, ChunkIndex, load_index
from .policy import effective_rerank_policy
from .rerank import rerank


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source_ids: list[str]
    score: float


def to_retrieved(chunk: Chunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id, text=chunk.text, source_ids=chunk.source_ids, score=score
    )


def rrf_fuse(
    ranked_lists: list[list[tuple[Chunk, float]]], rrf_k: int, n: int
) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion: score(c) = sum 1/(rrf_k + rank). Scale-free across signals."""
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunk_by_id[chunk.chunk_id] = chunk
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(chunk_by_id[cid], score) for cid, score in ordered]


class Retriever:
    def __init__(self, index: ChunkIndex | None = None, config: RagConfig | None = None):
        self.config = config or (index.config if index else RagConfig())
        self.index = index or load_index(self.config)
        self.embedder = Embedder(self.config)
        self._bm25 = None  # lazily loaded only for sparse/hybrid modes

    @property
    def bm25(self):
        if self._bm25 is None:
            from .bm25 import load_bm25

            self._bm25 = load_bm25(self.config)
        return self._bm25

    def candidates(self, query: str, n: int) -> list[tuple[Chunk, float]]:
        """First-stage retrieval per ``retrieval_mode`` — input to reranking / CRAG."""
        mode = self.config.retrieval_mode
        if mode == "dense":
            return self.index.search(self.embedder.encode_one(query), n)
        if mode == "sparse":
            return self.bm25.search(query, n)
        if mode == "hybrid":
            dense = self.index.search(self.embedder.encode_one(query), n)
            sparse = self.bm25.search(query, n)
            return rrf_fuse([dense, sparse], self.config.rrf_k, n)
        raise ValueError(f"Unknown retrieval_mode: {mode!r}")

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = self.config.k if k is None else k
        # Fetch a wider candidate pool unless reranking is disabled outright.
        n = k if effective_rerank_policy(self.config) == "never" else max(self.config.top_n, k)
        ranked = rerank(query, self.candidates(query, n), self.config)[:k]
        return [to_retrieved(c, score) for c, score in ranked]
