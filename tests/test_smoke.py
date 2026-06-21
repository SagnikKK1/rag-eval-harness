"""Smoke tests for the eval-ready RAG pipeline.

These must pass with NO API keys set: the whole retrieval path (ingest -> chunk ->
embed -> index -> retrieve) is LLM-free. Generation is intentionally not exercised
here. A small temp corpus keeps the test fast and self-contained.
"""
from __future__ import annotations

import os
from dataclasses import replace

import pytest

from rag.config import RagConfig
from rag.index import build_index
from rag.retrieve import Retriever

RAW_SAMPLE = """\
2025-06-10T05:04:26Z
The battery life on the Edge 50 Fusion is excellent, easily a full day.

2025-06-10T05:57:28Z
Camera quality is good in daylight but struggles a bit at night.

2025-06-10T05:52:49Z
The display is bright and the 144Hz refresh rate feels very smooth.

2025-06-10T06:00:00Z
Charging is fast, goes from 20 to 80 percent in about half an hour.

2025-06-10T06:05:00Z
Build quality feels premium with the vegan leather back.
"""


@pytest.fixture()
def config(tmp_path):
    raw = tmp_path / "sample.txt"
    raw.write_text(RAW_SAMPLE, encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    # Rebuild the corpus from the temp raw file into the temp processed dir.
    from rag import ingest

    ingest.build_corpus(raw_files=[raw], out_path=processed / "corpus.jsonl")
    cfg = RagConfig(chunk_size=128, chunk_overlap=16, k=3, processed_dir=processed)
    return cfg, processed / "corpus.jsonl"


def test_index_builds_and_retrieves(config):
    cfg, corpus_path = config
    # build_index -> build_chunks reads the temp corpus
    import rag.chunk as chunk_mod

    orig = chunk_mod.load_corpus
    chunk_mod.load_corpus = lambda path=None: orig(corpus_path)
    try:
        index = build_index(cfg)
    finally:
        chunk_mod.load_corpus = orig

    assert index.size > 0
    assert index.size == len(index.chunks)

    retriever = Retriever(index=index, config=cfg)
    results = retriever.retrieve("How is the battery life?", k=3)

    assert len(results) == 3
    for r in results:
        assert r.chunk_id and isinstance(r.text, str) and r.text
        assert isinstance(r.source_ids, list)
        assert isinstance(r.score, float)

    # The battery comment should surface near the top for a battery query.
    joined = " ".join(r.text.lower() for r in results)
    assert "battery" in joined


def test_retrieval_needs_no_api_key(config):
    cfg, corpus_path = config
    for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL"):
        assert os.environ.get(key) in (None, ""), f"{key} unexpectedly set in test env"

    import rag.chunk as chunk_mod

    orig = chunk_mod.load_corpus
    chunk_mod.load_corpus = lambda path=None: orig(corpus_path)
    try:
        index = build_index(replace(cfg))
    finally:
        chunk_mod.load_corpus = orig

    results = Retriever(index=index, config=cfg).retrieve("camera at night", k=2)
    assert len(results) == 2


def test_modules_import_without_keys():
    # Importing the generation/retrieval modules must not require any key.
    import rag.generate  # noqa: F401
    import rag.llm  # noqa: F401
    import rag.retrieve  # noqa: F401
