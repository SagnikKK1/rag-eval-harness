"""FastAPI demo tests — key-free, with a tiny injected pipeline (no real index)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as api
import rag.chunk as chunk_mod
from rag import ingest
from rag.config import RagConfig
from rag.index import build_index
from rag.pipeline import RagPipeline
from rag.retrieve import Retriever

RAW = """\
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
def client(tmp_path, monkeypatch):
    raw = tmp_path / "sample.txt"
    raw.write_text(RAW, encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    ingest.build_corpus(raw_files=[raw], out_path=processed / "corpus.jsonl")

    orig = chunk_mod.load_corpus
    chunk_mod.load_corpus = lambda path=None: orig(processed / "corpus.jsonl")
    try:
        base = RagConfig(chunk_size=128, chunk_overlap=16, k=3, processed_dir=processed)
        index = build_index(base)
    finally:
        chunk_mod.load_corpus = orig

    def fake_get_pipeline(mode, rerank_policy, retrieval_mode):
        cfg = RagConfig(
            chunk_size=128, chunk_overlap=16, k=3, processed_dir=processed,
            mode=mode, rerank_policy="never", retrieval_mode="dense", grader="threshold",
        )
        return RagPipeline(config=cfg, retriever=Retriever(index=index, config=cfg))

    monkeypatch.setattr(api, "get_pipeline", fake_get_pipeline)
    return TestClient(api.app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_retrieve(client):
    r = client.post("/retrieve", json={"query": "how is the battery life", "k": 3})
    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert 0 < len(chunks) <= 3
    assert {"chunk_id", "text", "source_ids", "score"} <= set(chunks[0])


def test_crag_returns_trace(client):
    r = client.post("/crag", json={"query": "battery life", "generate": False})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["refused"], bool)
    assert isinstance(body["trace"], list) and len(body["trace"]) >= 1
    assert {"round_idx", "action", "reranked", "correction"} <= set(body["trace"][0])
