"""FastAPI demo for the RAG pipeline.

Endpoints: /healthz, /retrieve, /answer (plain), /crag (answer + correction trace).
Pipelines are cached per config. Retrieval and the CRAG trace work with no API
key; /answer (and /crag?generate=true) need ANTHROPIC_API_KEY for generation.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel

from rag.config import RagConfig
from rag.generate import generate
from rag.pipeline import REFUSAL, RagPipeline

app = FastAPI(title="RAG Eval Harness", description="Adaptive reranking + Corrective-RAG demo")

_cache: dict[tuple, RagPipeline] = {}


def get_pipeline(mode: str, rerank_policy: str, retrieval_mode: str) -> RagPipeline:
    """Cached pipeline per config (tests override this to inject a tiny index)."""
    key = (mode, rerank_policy, retrieval_mode)
    if key not in _cache:
        _cache[key] = RagPipeline(
            RagConfig(mode=mode, rerank_policy=rerank_policy, retrieval_mode=retrieval_mode, k=5)
        )
    return _cache[key]


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    retrieval_mode: str = "dense"
    rerank_policy: str = "never"


class AskRequest(BaseModel):
    query: str
    rerank_policy: str = "auto"
    retrieval_mode: str = "dense"
    generate: bool = False


def _chunk_dicts(contexts) -> list[dict]:
    return [
        {"chunk_id": c.chunk_id, "text": c.text, "source_ids": c.source_ids, "score": c.score}
        for c in contexts
    ]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    pipe = get_pipeline("plain", req.rerank_policy, req.retrieval_mode)
    return {"query": req.query, "chunks": _chunk_dicts(pipe.retrieve(req.query, k=req.k))}


@app.post("/answer")
def answer(req: AskRequest) -> dict:
    pipe = get_pipeline("plain", req.rerank_policy, req.retrieval_mode)
    result = pipe.answer(req.query)
    return {"query": req.query, "answer": result.answer, "chunks": _chunk_dicts(result.contexts)}


@app.post("/crag")
def crag(req: AskRequest) -> dict:
    pipe = get_pipeline("crag", req.rerank_policy, req.retrieval_mode)
    result = pipe.corrective(req.query)
    payload = {
        "query": req.query,
        "refused": result.refused,
        "trace": [asdict(r) for r in result.trace],
        "chunks": _chunk_dicts(result.contexts),
    }
    if req.generate:
        payload["answer"] = (
            REFUSAL if result.refused else generate(req.query, result.contexts, pipe.config)
        )
    return payload
