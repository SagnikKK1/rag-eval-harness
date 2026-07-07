"""A/B experiment runner: sweep retrieval strategies over the golden set.

Each arm is just a different ``RagConfig`` (no per-arm code), exercised through
``RagPipeline``. Per arm it records:

- retrieval metrics (recall@k, MRR) plus **per-item** hit/reciprocal-rank arrays
  so ``eval/stats.py`` can compute paired significance vs the baseline;
- an honest abstention pair: refusal accuracy on not-in-corpus items AND the
  false-refusal rate on answerable items. Non-CRAG arms get a real abstention
  mechanism (threshold on top-1 dense cosine) instead of the old
  ``len(retrieved)==0`` which was structurally impossible for top-k retrieval;
- mean retrieval latency (ms/query) and the cross-encoder invocation rate — the
  evidence behind the "adaptive reranking saves cross-encoder calls" claim.

Generation metrics (Ragas) run only ``with_generation`` + an Anthropic key.
"""
from __future__ import annotations

import time

from rag import rerank as rerank_mod
from rag.config import RagConfig
from rag.pipeline import RagPipeline

from .golden.schema import GoldenItem, load_golden
from .retrieval_metrics import evaluate_retrieval, hit_ranks, refusal_accuracy

EVAL_K = 10  # retrieve a depth-10 ranking so we can score recall@{1,3,5,10}


def arms() -> dict[str, RagConfig]:
    """The A/B arms. `sparse` (BM25-only) is the lexical control: the golden set's
    gold labels are keyword-resolved, so this arm exposes how much of any arm's
    score is explainable by lexical matching alone."""
    base = dict(k=EVAL_K)
    return {
        "baseline": RagConfig(retrieval_mode="dense", rerank_policy="never", mode="plain", **base),
        "sparse": RagConfig(retrieval_mode="sparse", rerank_policy="never", mode="plain", **base),
        "hybrid": RagConfig(retrieval_mode="hybrid", rerank_policy="never", mode="plain", **base),
        "reranked": RagConfig(retrieval_mode="dense", rerank_policy="always", mode="plain", **base),
        "adaptive": RagConfig(retrieval_mode="dense", rerank_policy="auto", mode="plain", **base),
        "crag": RagConfig(retrieval_mode="dense", rerank_policy="auto", mode="crag", **base),
    }


def _run_item(pipe: RagPipeline, item: GoldenItem, config: RagConfig):
    """Return (source_id_sets_per_rank, refused, elapsed_seconds) for one item."""
    t0 = time.perf_counter()
    if config.mode == "crag":
        res = pipe.corrective(item.question)
        sets, refused = [set(c.source_ids) for c in res.contexts], res.refused
    else:
        retrieved = pipe.retrieve(item.question, k=EVAL_K)
        sets = [set(c.source_ids) for c in retrieved]
        # Threshold abstention: refuse when top-1 dense confidence is low.
        refused = pipe.retriever.dense_top_score(item.question) < config.refusal_min_top_score
    return sets, refused, time.perf_counter() - t0


def _n_relevant_chunks(pipe: RagPipeline, gold: set[str]) -> int:
    """True number of corpus chunks containing >=1 gold comment (for ideal DCG)."""
    return sum(1 for c in pipe.retriever.index.chunks if set(c.source_ids) & gold)


def _p95(xs: list[float]) -> float:
    return sorted(xs)[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else 0.0


def run_arm(name: str, config: RagConfig, golden: list[GoldenItem], with_generation: bool) -> dict:
    pipe = RagPipeline(config)
    answerable_hits: list[list[bool]] = []
    n_relevants: list[int] = []
    item_types: list[str] = []          # parallel to answerable_hits (per-type breakdown)
    refuse_flags: list[bool] = []       # on refuse items: did it (correctly) refuse?
    false_refusals: list[bool] = []     # on answerable items: did it (wrongly) refuse?
    latencies: list[float] = []
    gen_samples: list[dict] = []

    ce_calls_before = rerank_mod.CE_STATS["calls"]
    for item in golden:
        sets, refused, elapsed = _run_item(pipe, item, config)
        latencies.append(elapsed)
        if item.is_refuse:
            refuse_flags.append(refused)
        else:
            false_refusals.append(refused)
            gold = set(item.gold_source_ids)
            answerable_hits.append(hit_ranks(sets, gold))
            n_relevants.append(_n_relevant_chunks(pipe, gold))
            item_types.append(item.type)
    ce_calls = rerank_mod.CE_STATS["calls"] - ce_calls_before

    retrieval = evaluate_retrieval(answerable_hits, n_relevants)
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
        "per_item_hits": answerable_hits,   # for paired stats vs baseline
        "per_item_types": item_types,       # for the per-type breakdown
        "refusal": {
            "n": n_ref,
            "accuracy": refuse_acc,
            "false_refusal_rate": (
                sum(false_refusals) / len(false_refusals) if false_refusals else 0.0
            ),
        },
        "latency_ms": 1000.0 * sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p95_ms": 1000.0 * _p95(latencies),
        "ce_calls_per_query": ce_calls / len(golden) if golden else 0.0,
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
