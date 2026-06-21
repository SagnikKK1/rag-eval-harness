"""CLI: build the cleaned corpus, chunk it, and build the FAISS index.

    python scripts/build_index.py [--chunk-size 512] [--chunk-overlap 64]

Retrieval-only — needs no API key.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.bm25 import build_bm25  # noqa: E402
from rag.config import CORPUS_PATH, RagConfig  # noqa: E402
from rag.index import build_index  # noqa: E402
from rag.ingest import build_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG chunk index.")
    parser.add_argument("--chunk-size", type=int, default=RagConfig.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=RagConfig.chunk_overlap)
    parser.add_argument("--rebuild-corpus", action="store_true", help="Re-clean the raw corpus.")
    args = parser.parse_args()

    config = RagConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if args.rebuild_corpus or not CORPUS_PATH.exists():
        comments = build_corpus()
        print(f"Corpus: {len(comments)} comments -> {CORPUS_PATH}")

    index = build_index(config)
    print(f"Chunks: {len(index.chunks)} ({config.tag})")
    print(f"Index:  {index.size} vectors -> {config.index_path}")

    build_bm25(config)
    print(f"BM25:   {len(index.chunks)} docs -> {config.bm25_path}")


if __name__ == "__main__":
    main()
