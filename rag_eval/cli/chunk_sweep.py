"""Chunking A/B: sweep chunk size/overlap and measure retrieval quality.

    python scripts/chunk_sweep.py            # 256/32, 512/64, 1024/128 × {baseline, reranked}

Ground truth is comment-level (chunk-config independent), so the same golden set
scores every chunking fairly: a retrieved chunk is a hit iff it contains a gold
comment. Builds any missing index on the fly (no API key needed). Appends the
table to eval/results/chunk_sweep.md.
"""
from __future__ import annotations

from pathlib import Path

from rag_eval.bm25 import build_bm25  # noqa: E402
from rag_eval.config import RagConfig  # noqa: E402
from rag_eval.evals.experiment import EVAL_K, run_arm  # noqa: E402
from rag_eval.evals.golden.schema import load_golden  # noqa: E402
from rag_eval.index import build_index  # noqa: E402

SWEEP = [(256, 32), (512, 64), (1024, 128)]
RESULTS = Path.cwd() / "results" / "chunk_sweep.md"


def _ensure_artifacts(cfg: RagConfig) -> None:
    if not cfg.index_path.exists() or not cfg.chunks_path.exists():
        print(f"  building index for {cfg.tag} ...")
        build_index(cfg)
    if not cfg.bm25_path.exists():
        build_bm25(cfg)


def main() -> None:
    golden = load_golden()
    rows = ["| chunks (size/overlap) | arm | recall@1 | recall@5 | recall@10 | mrr |",
            "| --- | --- | --- | --- | --- | --- |"]
    for cs, co in SWEEP:
        base_kw = dict(chunk_size=cs, chunk_overlap=co, k=EVAL_K, mode="plain")
        _ensure_artifacts(RagConfig(**base_kw))
        for arm_name, policy in (("baseline", "never"), ("reranked", "always")):
            cfg = RagConfig(rerank_policy=policy, **base_kw)
            r = run_arm(f"{arm_name}@{cs}/{co}", cfg, golden, with_generation=False)["retrieval"]
            rows.append(
                f"| {cs}/{co} | {arm_name} | {r['recall_at_1']:.3f} | {r['recall_at_5']:.3f} "
                f"| {r['recall_at_10']:.3f} | {r['mrr']:.3f} |"
            )
            print(rows[-1])
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("### Chunking sweep\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
