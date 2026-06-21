"""A/B experiment runner: sweep retrieval strategies over the golden set.

Each arm is just a different ``RagConfig`` (no per-arm code), exercised through
``RagPipeline``. Retrieval metrics are computed key-free; generation metrics
(Ragas faithfulness / answer-relevancy) run only when ``with_generation`` and an
Anthropic key are available, otherwise they are reported as skipped.
"""
from __future__ import annotations

from rag.config import RagConfig
from rag.pipeline import RagPipeline

from .golden.schema import GoldenItem, load_golden
from .retrieval_metrics import evaluate_retrieval, hit_ranks, refusal_accuracy

EVAL_K = 10  # retrieve a depth-10 ranking so we can score recall@{1,3,5,10}


def arms() -> dict[str, RagConfig]:
    """The A/B arms: baseline dense, +rerank always, +rerank adaptive, hybrid, CRAG."""
    base = dict(k=EVAL_K)
    return {
        "baseline": RagConfig(retrieval_mode="dense", rerank_policy="never", mode="plain", **base),
        "reranked": RagConfig(retrieval_mode="dense", rerank_policy="always", mode="plain", **base),
        "adaptive": RagConfig(retrieval_mode="dense", rerank_policy="auto", mode="plain", **base),
        "hybrid": RagConfig(retrieval_mode="hybrid", rerank_policy="never", mode="plain", **base),
        "crag": RagConfig(retrieval_mode="dense", rerank_policy="auto", mode="crag", **base),
    }


def _retrieved_for(pipe: RagPipeline, item: GoldenItem):
    """Return (source_id_sets_per_rank, refused) for one item under the arm's config."""
    if pipe.config.mode == "crag":
        res = pipe.corrective(item.question)
        return [set(c.source_ids) for c in res.contexts], res.refused
    retrieved = pipe.retrieve(item.question, k=EVAL_K)
    return [set(c.source_ids) for c in retrieved], len(retrieved) == 0


def run_arm(name: str, config: RagConfig, golden: list[GoldenItem], with_generation: bool) -> dict:
    pipe = RagPipeline(config)
    answerable_hits: list[list[bool]] = []
    refuse_flags: list[bool] = []
    gen_samples: list[dict] = []

    for item in golden:
        source_id_sets, refused = _retrieved_for(pipe, item)
        if item.is_refuse:
            refuse_flags.append(refused)
        else:
            answerable_hits.append(hit_ranks(source_id_sets, set(item.gold_source_ids)))

    retrieval = evaluate_retrieval(answerable_hits)
    n_ref, refuse_acc = refusal_accuracy(refuse_flags)

    generation = None
    if with_generation:
        from .ragas_eval import evaluate_generation, ragas_available

        if ragas_available():
            for item in golden:
                if item.is_refuse:
                    continue
                result = pipe.answer(item.question, expand=False)
                gen_samples.append({
                    "question": item.question,
                    "answer": result.answer,
                    "contexts": [c.text for c in result.contexts],
                })
            generation = evaluate_generation(gen_samples, config)

    return {
        "arm": name,
        "retrieval": retrieval.as_dict(),
        "refusal": {"n": n_ref, "accuracy": refuse_acc},
        "generation": generation,
    }


def run_experiment(
    arm_names: list[str] | None = None, with_generation: bool = False
) -> list[dict]:
    golden = load_golden()
    selected = arms()
    if arm_names:
        selected = {k: v for k, v in selected.items() if k in arm_names}
    return [run_arm(name, cfg, golden, with_generation) for name, cfg in selected.items()]
