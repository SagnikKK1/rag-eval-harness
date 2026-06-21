"""Configurable fixed-size chunking with source provenance.

Cleaned comments are concatenated into pseudo-documents, then split with a
``RecursiveCharacterTextSplitter`` whose ``chunk_size``/``chunk_overlap`` come
from ``RagConfig``. Each chunk records the ``source_ids`` of the comments it
overlaps (via character offsets), so the Part 2 golden set can mark exactly
which chunks are ground truth for a question -> meaningful recall@k / MRR.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RagConfig
from .ingest import Comment, load_corpus

_SEP = "\n"
_COMMENTS_PER_DOC = 40  # group size for pseudo-documents (kept > chunk_size in chars)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_ids: list[str]
    doc_id: str


def _pseudo_documents(comments: list[Comment], group: int = _COMMENTS_PER_DOC):
    """Yield (doc_id, doc_text, spans) where spans = [(start, end, source_id), ...]."""
    for d, start in enumerate(range(0, len(comments), group)):
        block = comments[start : start + group]
        spans: list[tuple[int, int, str]] = []
        parts: list[str] = []
        cursor = 0
        for c in block:
            spans.append((cursor, cursor + len(c.text), c.source_id))
            parts.append(c.text)
            cursor += len(c.text) + len(_SEP)
        yield f"doc{d:04d}", _SEP.join(parts), spans


def _overlapping_source_ids(chunk_start: int, chunk_end: int, spans) -> list[str]:
    return [sid for (s, e, sid) in spans if s < chunk_end and e > chunk_start]


def build_chunks(config: RagConfig | None = None, corpus_path: Path | None = None) -> list[Chunk]:
    """Chunk the corpus per ``config`` and persist to ``config.chunks_path``."""
    config = config or RagConfig()
    comments = load_corpus(corpus_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
    )

    chunks: list[Chunk] = []
    n = 0
    for doc_id, doc_text, spans in _pseudo_documents(comments):
        for doc in splitter.create_documents([doc_text]):
            start = doc.metadata["start_index"]
            text = doc.page_content
            source_ids = _overlapping_source_ids(start, start + len(text), spans)
            chunks.append(
                Chunk(chunk_id=f"c{n:05d}", text=text, source_ids=source_ids, doc_id=doc_id)
            )
            n += 1

    out_path = config.chunks_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
    return chunks


def load_chunks(config: RagConfig | None = None) -> list[Chunk]:
    config = config or RagConfig()
    path = config.chunks_path
    if not path.exists():
        raise FileNotFoundError(f"Chunks not built yet: {path}. Run build_chunks(config) first.")
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


if __name__ == "__main__":
    cfg = RagConfig()
    built = build_chunks(cfg)
    print(f"Built {len(built)} chunks ({cfg.tag}) -> {cfg.chunks_path}")
