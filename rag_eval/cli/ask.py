"""CLI: query the RAG pipeline (replaces the original query.py __main__).

    python scripts/ask.py "How is the battery life?"                  # plain retrieval
    python scripts/ask.py "How is the battery life?" --rerank-policy auto
    python scripts/ask.py "Does it support wireless charging?" --mode crag --generate
    python scripts/ask.py "What's the battery like and 60/200 as a percent?" --mode agent

Retrieval (and the CRAG loop with the threshold grader) run with no key;
generation, the LLM grader, LLM reformulation, and the tool agent need ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse

from rag_eval.config import RagConfig  # noqa: E402
from rag_eval.generate import generate  # noqa: E402
from rag_eval.pipeline import REFUSAL, RagPipeline  # noqa: E402


def _print_chunks(contexts) -> None:
    for i, c in enumerate(contexts, start=1):
        print(f"[S{i}] score={c.score:.3f}  sources={c.source_ids}")
        print(f"     {c.text[:200]}{'...' if len(c.text) > 200 else ''}\n")


def _print_trace(trace) -> None:
    print("CRAG trace:\n" + "-" * 60)
    for r in trace:
        rr = "reranked" if r.reranked else "dense"
        corr = f" -> {r.correction}" if r.correction else ""
        print(
            f"  round {r.round_idx}: {r.action:9s} [{rr}, {r.n_candidates} cand, "
            f"kept {len(r.kept_chunk_ids)}]{corr}\n            q={r.query!r}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("query", nargs="?", help="The question to ask.")
    parser.add_argument("--mode", default="plain", choices=["plain", "crag", "agent"])
    parser.add_argument("--chunk-size", type=int, default=RagConfig.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=RagConfig.chunk_overlap)
    parser.add_argument("-k", type=int, default=RagConfig.k, help="Number of chunks to keep.")
    parser.add_argument(
        "--rerank-policy", default=None, choices=["never", "always", "auto", "llm"],
        help="Default: 'auto' in crag mode, else 'never'.",
    )
    parser.add_argument("--max-rounds", type=int, default=RagConfig.max_correction_rounds)
    parser.add_argument("--generate", action="store_true", help="Also generate a grounded answer.")
    parser.add_argument("--provider", default=RagConfig.provider,
                        choices=["anthropic", "openrouter"])
    args = parser.parse_args()

    query = args.query or input("\nYour question: ").strip()
    if not query:
        print("No question entered. Exiting.")
        return

    policy = args.rerank_policy or ("auto" if args.mode == "crag" else "never")
    config = RagConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k=args.k,
        mode=args.mode,
        rerank_policy=policy,
        max_correction_rounds=args.max_rounds,
        provider=args.provider,
    )
    pipeline = RagPipeline(config=config)

    if config.mode == "agent":
        result = pipeline.tool_agent(query)
        print(f"\nTool agent for: {query!r}  ({result.steps} step(s))\n" + "-" * 60)
        for tc in result.trace:
            print(f"  step {tc.step}: {tc.name}({tc.tool_input})")
        if result.contexts:
            print(f"\nChunks gathered via search_corpus ({len(result.contexts)}):\n" + "-" * 60)
            _print_chunks(result.contexts)
        print("=" * 60 + "\nAnswer:\n" + result.answer)
        return

    if config.mode == "crag":
        crag = pipeline.corrective(query)
        print(f"\nCRAG for: {query!r}  (rerank-policy={policy})\n")
        _print_trace(crag.trace)
        if crag.refused:
            print("No relevant chunks found -> would refuse.\n")
        else:
            print(f"Final {len(crag.contexts)} chunks:\n" + "-" * 60)
            _print_chunks(crag.contexts)
        if args.generate:
            answer = REFUSAL if crag.refused else generate(query, crag.contexts, config)
            print("=" * 60 + "\nAnswer:\n" + answer)
        return

    contexts = pipeline.retrieve(query)
    print(f"\nTop {len(contexts)} chunks for: {query!r}  (rerank-policy={policy})\n" + "-" * 60)
    _print_chunks(contexts)
    if args.generate:
        print("=" * 60 + "\nAnswer:\n" + pipeline.answer(query).answer)


if __name__ == "__main__":
    main()
