"""CLI: run the A/B eval over the golden set; print + persist the results table.

    python scripts/run_eval.py                 # all arms, retrieval metrics (no key)
    python scripts/run_eval.py --generate      # + Ragas faithfulness (needs ANTHROPIC_API_KEY)
    python scripts/run_eval.py --gate          # CI: fail if any gated arm regresses

Writes eval/results/results.csv and eval/results/results.md, stamped with run
metadata (git SHA, timestamp, N, models) and a paired-significance table vs the
baseline arm (McNemar on recall@5, bootstrap CI on MRR).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.experiment import EVAL_K, run_arm, run_experiment  # noqa: E402
from eval.experiment import arms as all_arms  # noqa: E402
from eval.golden.schema import load_golden  # noqa: E402
from eval.stats import compare_arms  # noqa: E402
from rag.config import RagConfig  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"
THRESHOLDS = Path(__file__).resolve().parent.parent / "eval" / "thresholds.json"

COLUMNS = [
    "arm", "recall@1", "recall@3", "recall@5", "recall@10", "mrr",
    "refusal_acc", "false_refusal", "latency_ms", "ce_calls/q",
    "faithfulness", "answer_relevancy",
]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _metadata(n_answerable: int, n_refuse: int) -> str:
    cfg = RagConfig()
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"run: {ts} · git {_git_sha()} · N={n_answerable} answerable + {n_refuse} refuse · "
        f"depth {EVAL_K} · embed `{cfg.embed_model}` · reranker `{cfg.reranker_model}` · "
        f"chunks {cfg.chunk_size}/{cfg.chunk_overlap}"
    )


def _row(result: dict) -> dict:
    r = result["retrieval"]
    gen = result["generation"]
    gen_cell = lambda key: (  # noqa: E731
        f"{gen[key]:.3f}" if gen and not gen.get("skipped") else "—"
    )
    return {
        "arm": result["arm"],
        "recall@1": f"{r['recall_at_1']:.3f}",
        "recall@3": f"{r['recall_at_3']:.3f}",
        "recall@5": f"{r['recall_at_5']:.3f}",
        "recall@10": f"{r['recall_at_10']:.3f}",
        "mrr": f"{r['mrr']:.3f}",
        "refusal_acc": f"{result['refusal']['accuracy']:.3f}",
        "false_refusal": f"{result['refusal']['false_refusal_rate']:.3f}",
        "latency_ms": f"{result['latency_ms']:.0f}",
        "ce_calls/q": f"{result['ce_calls_per_query']:.2f}",
        "faithfulness": gen_cell("faithfulness"),
        "answer_relevancy": gen_cell("answer_relevancy"),
    }


def _significance_rows(results: list[dict]) -> list[str]:
    base = next((r for r in results if r["arm"] == "baseline"), None)
    if base is None:
        return ["(no baseline arm in this run — significance table skipped)"]
    lines = [
        "| arm vs baseline | Δrecall@5 (won/lost) | McNemar p | ΔMRR [95% CI] | significant? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        if r["arm"] == "baseline":
            continue
        cmp = compare_arms(base["per_item_hits"], r["per_item_hits"], k=5)
        b, c = cmp["discordant"]
        lo, hi = cmp["mrr_ci95"]
        lines.append(
            f"| {r['arm']} | +{b}/−{c} | {cmp['recall_at_k_p']:.3f} "
            f"| {cmp['mrr_delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
            f"| {'yes' if cmp['significant'] else 'no'} |"
        )
    return lines


def _markdown(rows: list[dict], results: list[dict], meta: str, generated: bool) -> str:
    head = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = "\n".join("| " + " | ".join(r[c] for c in COLUMNS) + " |" for r in rows)
    sig = "\n".join(_significance_rows(results))
    gen_note = (
        "Generation metrics via Ragas (Anthropic judge)."
        if generated else
        "_Generation columns pending ANTHROPIC_API_KEY (`--generate`)._"
    )
    return (
        f"### RAG A/B results\n\n{meta}\n\n{head}\n{sep}\n{body}\n\n"
        f"Refusal: `refusal_acc` = correct refusals on not-in-corpus items; "
        f"`false_refusal` = wrong refusals on answerable items. Non-CRAG arms "
        f"refuse when top-1 dense cosine < {RagConfig().refusal_min_top_score} "
        f"(uncalibrated default, not tuned on the golden set).\n\n"
        f"#### Paired significance vs baseline\n\n{sig}\n\n"
        f"At this N, treat non-significant deltas as directional only.\n\n{gen_note}\n"
    )


def _write(rows: list[dict], md: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "results.md").write_text(md, encoding="utf-8")


def run_gate() -> int:
    cfg = json.loads(THRESHOLDS.read_text())
    golden = load_golden()
    failed = False
    for arm_name, thr in cfg["arms"].items():
        result = run_arm(arm_name, all_arms()[arm_name], golden, with_generation=False)
        r = result["retrieval"]
        checks = {"recall@5": (r["recall_at_5"], thr["recall@5"]), "mrr": (r["mrr"], thr["mrr"])}
        for k, (got, want) in checks.items():
            ok = got >= want
            failed |= not ok
            print(f"  [{'ok' if ok else 'FAIL'}] {arm_name} {k} = {got:.3f} (threshold {want})")
    print("GATE FAILED." if failed else "GATE PASSED.")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG A/B eval.")
    parser.add_argument("--arms", nargs="*", default=None, help="Subset of arms to run.")
    parser.add_argument("--generate", action="store_true", help="Also run Ragas generation eval.")
    parser.add_argument("--gate", action="store_true", help="CI gate: exit 1 on regression.")
    args = parser.parse_args()

    if args.gate:
        sys.exit(run_gate())

    golden = load_golden()
    n_refuse = sum(1 for g in golden if g.is_refuse)
    results = run_experiment(args.arms, with_generation=args.generate)
    rows = [_row(r) for r in results]
    generated = any(r["generation"] and not r["generation"].get("skipped") for r in results)
    md = _markdown(rows, results, _metadata(len(golden) - n_refuse, n_refuse), generated)
    _write(rows, md)
    print(md)
    print(f"Wrote {RESULTS_DIR/'results.csv'} and {RESULTS_DIR/'results.md'}")


if __name__ == "__main__":
    main()
