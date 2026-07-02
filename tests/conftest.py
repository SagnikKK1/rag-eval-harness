"""Shared test fixtures."""
from __future__ import annotations

import pytest

import rag.chunk as chunk_mod
from rag import ingest
from rag.config import RagConfig
from rag.index import build_index
from rag.retrieve import Retriever

_RAW = """\
2025-06-10T05:04:26Z
The battery life on the Edge 50 Fusion is excellent, easily a full day.

2025-06-10T05:57:28Z
Camera quality is good in daylight but struggles a bit at night.

2025-06-10T05:52:49Z
The 144Hz display is bright and very smooth to scroll.

2025-06-10T06:00:00Z
Charging is fast, 20 to 80 percent in about half an hour.

2025-06-10T06:05:00Z
Build quality feels premium with the vegan leather back.
"""


@pytest.fixture()
def tiny_retriever(tmp_path):
    """A Retriever over a tiny in-tmp corpus/index — key-free, no model downloads beyond MiniLM."""
    raw = tmp_path / "sample.txt"
    raw.write_text(_RAW, encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    ingest.build_corpus(raw_files=[raw], out_path=processed / "corpus.jsonl")

    orig = chunk_mod.load_corpus
    chunk_mod.load_corpus = lambda path=None: orig(processed / "corpus.jsonl")
    try:
        cfg = RagConfig(chunk_size=128, chunk_overlap=16, k=3, processed_dir=processed,
                        rerank_policy="never", grader="threshold")
        index = build_index(cfg)
    finally:
        chunk_mod.load_corpus = orig
    return Retriever(index=index, config=cfg)
