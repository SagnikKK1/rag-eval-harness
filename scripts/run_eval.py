"""CLI: run the A/B eval over the golden set; print + persist the results table.

    python scripts/run_eval.py                 # all arms, retrieval metrics (no key)
    python scripts/run_eval.py --generate       # + Ragas faithfulness (needs ANTHROPIC_API_KEY)
    python scripts/run_eval.py --gate           # CI: fail if the gate arm regresses

Writes eval/results/results.csv and eval/results/results.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.experiment import arms as all_arms  # noqa: E402
from eval.experiment import run_arm, run_experiment  # noqa: E402
from eval.golden.schema import load_golden  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"
THRESHOLDS = Path(__file__).resolve().parent.parent / "eval" / "thresholds.json"

COLUMNS = ["arm", "recall@1", "recall@3", "recall@5", "recall@10", "mrr", "refusal_acc",
           "faithfulness", "answer_relevancy"]


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
        "faithfulness": gen_cell("faithfulness"),
        "answer_relevancy": gen_cell("answer_relevancy"),
    }


def _markdown(rows: list[dict], n_answerable: int, n_refuse: int, generated: bool) -> str:
    head = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = "\n".join("| " + " | ".join(r[c] for c in COLUMNS) + " |" for r in rows)
    note = (
        "Generation metrics via Ragas (Anthropic judge)."
        if generated else
        "_Generation columns skipped — set ANTHROPIC_API_KEY and rerun with --generate._"
    )
    return (
        f"### RAG A/B results\n\n"
        f"Golden set: {n_answerable} answerable + {n_refuse} not-in-corpus items.\n\n"
        f"{head}\n{sep}\n{body}\n\n{note}\n"
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
    arm = cfg["arm"]
    result = run_arm(arm, all_arms()[arm], load_golden(), with_generation=False)
    r = result["retrieval"]
    checks = {"recall@5": (r["recall_at_5"], cfg["recall@5"]), "mrr": (r["mrr"], cfg["mrr"])}
    failed = {k: v for k, v in checks.items() if v[0] < v[1]}
    for k, (got, thr) in checks.items():
        status = "FAIL" if k in failed else "ok"
        print(f"  [{status}] {arm} {k} = {got:.3f} (threshold {thr})")
    if failed:
        print(f"GATE FAILED on arm '{arm}'.")
        return 1
    print(f"GATE PASSED on arm '{arm}'.")
    return 0


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
    md = _markdown(rows, len(golden) - n_refuse, n_refuse, generated)
    _write(rows, md)
    print(md)
    print(f"Wrote {RESULTS_DIR/'results.csv'} and {RESULTS_DIR/'results.md'}")


if __name__ == "__main__":
    main()
