"""Central configuration for the RAG pipeline.

A single ``RagConfig`` object carries every knob the eval harness (Part 2) will
sweep: chunk size/overlap, retrieval depth, reranking, and the generation
provider/model. Artifact paths encode the chunk settings so different A/B
configurations write to distinct files instead of clobbering each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw corpus files ported from the Sparkathon project (public scraped reviews).
RAW_CORPUS_FILES = [
    RAW_DIR / "motorolaedge50fusion_comments.txt",
    RAW_DIR / "motorolaedge50fusion_reddit_comments.txt",
]

# Cleaned, normalized corpus (one record per comment). Stable across chunk settings.
CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"


@dataclass(frozen=True)
class RagConfig:
    # --- chunking (the primary A/B lever in Part 2) ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- embedding / index ---
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- retrieval ---
    k: int = 5              # final number of chunks returned
    top_n: int = 20         # candidates fetched before (optional) reranking
    retrieval_mode: str = "dense"  # "dense" | "sparse" (BM25) | "hybrid" (dense+BM25 via RRF)
    rrf_k: int = 60         # Reciprocal Rank Fusion constant for hybrid

    # --- reranker + adaptive gating ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # When to run the cross-encoder: "never" | "always" | "auto" | "llm".
    # "auto" gates on dense-retrieval confidence; "llm" asks a cheap classifier.
    rerank_policy: str = "never"
    use_reranker: bool = False  # backward-compat alias: True -> rerank_policy="always"
    # "auto" heuristic: skip reranking only when retrieval looks confident, i.e.
    # top cosine score >= auto_min_top_score AND (top1 - mean(tail)) >= auto_min_margin.
    auto_min_top_score: float = 0.55
    auto_min_margin: float = 0.05

    # --- pipeline mode ---
    mode: str = "plain"     # "plain" | "crag" (corrective loop) | "agent" (tool-using)
    crag_engine: str = "langgraph"  # "langgraph" (StateGraph) | "plain" (hand-rolled loop)
    max_tool_steps: int = 6  # max tool-call rounds for the tool-using agent (mode="agent")

    # --- CRAG (corrective RAG) ---
    grader: str = "threshold"          # "threshold" (free, on rerank/dense score) | "llm"
    # Cross-encoder logits (ms-marco) are unbounded (~[-11, +11]); calibrate in P2.
    rerank_relevant_threshold: float = 0.0    # score >= -> chunk is relevant ("CORRECT")
    rerank_incorrect_threshold: float = -4.0  # all scores <= -> "INCORRECT" (re-retrieve)
    # Dense-cosine fallback thresholds (used when a round was not reranked).
    dense_relevant_threshold: float = 0.45
    dense_incorrect_threshold: float = 0.25
    max_correction_rounds: int = 2     # extra retrieve+grade rounds after the first
    reformulate: bool = True           # rewrite the query between correction rounds

    # --- abstention (threshold-refusal baseline for non-CRAG arms) ---
    # Refuse when the top-1 dense cosine is below this. Uncalibrated default —
    # NOT tuned on the golden refuse items (that would be test-set leakage).
    refusal_min_top_score: float = 0.35

    # --- generation ---
    provider: str = "anthropic"               # "anthropic" | "openrouter"
    model: str = "claude-haiku-4-5-20251001"  # generation model; sonnet is the quality upgrade
    temperature: float = 0.0
    max_tokens: int = 512

    # --- artifact locations (chunk-setting-scoped so A/B runs don't collide) ---
    processed_dir: Path = field(default=PROCESSED_DIR)

    @property
    def tag(self) -> str:
        return f"cs{self.chunk_size}_co{self.chunk_overlap}"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / f"chunks_{self.tag}.jsonl"

    @property
    def index_path(self) -> Path:
        return self.processed_dir / f"index_{self.tag}.faiss"

    @property
    def bm25_path(self) -> Path:
        return self.processed_dir / f"bm25_{self.tag}.pkl"
