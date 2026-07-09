"""FAISS index build/load over the chunk corpus.

Replaces the original 58-summary ``IndexFlatL2`` with a cosine-similarity
``IndexFlatIP`` over the fixed-size chunks. The index row order matches the
chunk order in ``config.chunks_path``; we persist the chunks alongside so the
row id maps back to ``chunk_id`` / ``text`` / ``source_ids``.
"""
from __future__ import annotations

import faiss
import numpy as np

from .chunk import Chunk, build_chunks, load_chunks
from .config import RagConfig
from .embed import Embedder


class ChunkIndex:
    """A FAISS cosine index plus the parallel list of chunks it was built from."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk], config: RagConfig):
        self.index = index
        self.chunks = chunks
        self.config = config

    @property
    def size(self) -> int:
        return self.index.ntotal

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, score) pairs for a single normalized query vector."""
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        out: list[tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            out.append((self.chunks[idx], float(score)))
        return out


def build_index(config: RagConfig | None = None, rebuild_chunks: bool = True) -> ChunkIndex:
    """Build chunks (optionally) + embeddings + FAISS index and persist the index."""
    config = config or RagConfig()
    chunks = build_chunks(config) if rebuild_chunks else load_chunks(config)

    embedder = Embedder(config)
    vectors = embedder.encode([c.text for c in chunks])

    index = faiss.IndexFlatIP(embedder.dim)
    index.add(vectors)

    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.index_path))
    return ChunkIndex(index, chunks, config)


def load_index(config: RagConfig | None = None) -> ChunkIndex:
    """Load a previously built index + its chunks."""
    config = config or RagConfig()
    if not config.index_path.exists():
        raise FileNotFoundError(
            f"Index not built yet: {config.index_path}. Run build_index(config) first."
        )
    index = faiss.read_index(str(config.index_path))
    chunks = load_chunks(config)
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"Index/chunks mismatch: index has {index.ntotal} vectors, "
            f"{len(chunks)} chunks on disk. Rebuild with build_index(config)."
        )
    return ChunkIndex(index, chunks, config)
