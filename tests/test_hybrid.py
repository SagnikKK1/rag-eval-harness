"""Hybrid retrieval pieces: RRF fusion + BM25 search (key-free, no models)."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from rag.bm25 import Bm25Index, _tokenize
from rag.chunk import Chunk
from rag.retrieve import rrf_fuse


def _chunk(cid: str, text: str = "") -> Chunk:
    return Chunk(chunk_id=cid, text=text, source_ids=[cid], doc_id="d0")


def test_rrf_rewards_agreement_across_lists():
    a, b, c = _chunk("A"), _chunk("B"), _chunk("C")
    dense = [(a, 0.9), (b, 0.8), (c, 0.1)]   # A best
    sparse = [(b, 5.0), (a, 4.0), (c, 0.0)]  # B best, A second
    fused = rrf_fuse([dense, sparse], rrf_k=60, n=3)
    ids = [ch.chunk_id for ch, _ in fused]
    # A is rank 0 then 1; B is rank 1 then 0 -> tie, but A and B both beat C.
    assert ids[2] == "C"
    assert set(ids[:2]) == {"A", "B"}


def test_rrf_truncates_to_n():
    chunks = [(_chunk(f"c{i}"), 1.0) for i in range(10)]
    assert len(rrf_fuse([chunks], rrf_k=60, n=3)) == 3


def test_bm25_ranks_lexical_match_first():
    chunks = [
        _chunk("c0", "the battery lasts all day on this phone"),
        _chunk("c1", "the camera takes great photos at night"),
        _chunk("c2", "the display is bright and smooth"),
    ]
    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
    index = Bm25Index(bm25, chunks)
    # Use discriminating terms unique to c1 (avoid stopwords like "the"/"is",
    # whose BM25 IDF goes negative when they appear in every doc).
    results = index.search("camera photos", n=3)
    assert results[0][0].chunk_id == "c1"


def test_bm25_search_respects_n():
    chunks = [_chunk(f"c{i}", f"word{i} common") for i in range(5)]
    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
    assert len(Bm25Index(bm25, chunks).search("common", n=2)) == 2
