# RAG Evaluation Harness (with adaptive reranking + Corrective-RAG)

A reusable **RAG evaluation harness with a CI regression gate**, an agentic
**Corrective-RAG (CRAG)** pipeline built on **LangGraph**, adaptive cross-encoder
reranking, and hybrid retrieval — measured with a versioned golden set over a real
corpus of Motorola Edge 50 Fusion user reviews (public scraped YouTube/Reddit
comments, ported from the Sparkathon project).

The harness produces a before→after A/B table across retrieval strategies; the CI
gate fails the build on retrieval regressions.

## Results (A/B over the golden set)

> **Measurement history, disclosed:** an earlier version of this table showed much
> larger deltas (e.g. reranker recall@1 0.45→0.70). An internal audit found the
> ground-truth labeling inflated them: gold ids were resolved by *chunk-level
> substring* matching, which (a) marked every comment sharing a chunk with one
> matching comment as gold (up to 8× spillover), and (b) matched substrings like
> "sd" inside unrelated words. Gold is now resolved at the **comment level with
> word-boundary regexes**, refusal is compared fairly (every arm gets an abstention
> mechanism), and all deltas carry **paired significance tests**. The corrected
> numbers below are smaller and honest.

Golden set: **99 answerable + 14 not-in-corpus** items (factoid / co-occurrence /
paraphrase / refuse, incl. *near-domain* refuses like "Samsung washing machine
price"). Retrieve depth-10; a retrieved chunk is a hit iff it contains a gold
comment. `sparse` (BM25-only) is the **lexical control** — gold labels are
keyword-resolved, so this arm bounds how much score keyword matching alone
explains. Reproduce with `python scripts/run_eval.py`.

| arm | recall@1 | recall@5 | MRR | nDCG@10 | refusal_acc | false_refusal | latency ms/q | CE calls/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (dense) | 0.545 | 0.798 | 0.660 | 0.438 | 0.714 | 0.081 | 30 | 0 |
| sparse (BM25 control) | 0.566 | 0.818 | 0.669 | 0.423 | 0.714 | 0.081 | 13 | 0 |
| hybrid (RRF) | 0.586 | 0.859 | 0.703 | 0.476 | 0.714 | 0.081 | 24 | 0 |
| **reranked (always)** | **0.778** | **0.919** | **0.844** | **0.554** | 0.714 | 0.081 | 262 | 1.00 |
| reranked (adaptive) | 0.707 | 0.879 | 0.791 | 0.508 | 0.714 | 0.081 | 152 | 0.67 |
| CRAG (agentic) | 0.636 | 0.697 | 0.667 | 0.312 | 0.714 | 0.202 | 748 | 1.65 |

**Paired significance vs baseline** (McNemar on recall@5; paired bootstrap 95% CI
on ΔMRR; N=99) — full tables incl. per-type breakdown and the risk–coverage sweep
in [results/results.md](results/results.md):

- **Cross-encoder reranking: ΔMRR +0.184, 95% CI [+0.117, +0.257]; recall@5
  McNemar p=0.002 (+13/−1) — clearly significant.** Cost: ~9× baseline latency.
- **Adaptive reranking: ΔMRR +0.131, 95% CI [+0.071, +0.194]; p=0.021 —
  significant**, at ~58% of the always-rerank latency, invoking the cross-encoder
  on 67% of queries.
- **Per-type breakdown is the semantic-retrieval story**: on de-lexicalized
  paraphrase items (n=22), BM25 collapses to recall@5 **0.27** vs dense 0.55 and
  reranked **0.73** — keyword matching can't find evidence worded differently.
- **CRAG's abstention has a now-significant cost**: recall@5 −10pts vs baseline
  (+4/−14 discordant, p=0.031) with 20% false refusals — its threshold grader
  discards true positives. Reported as a real trade-off, not hidden.
- **Hybrid** is the best no-CE arm (ΔMRR +0.043 [−0.013, +0.101]) but still
  short of significance.
- **Chunking sweep** ([results/chunk_sweep.md](results/chunk_sweep.md)):
  reranking wins at every chunk size (ΔMRR +0.12–0.18); 512/64+reranked is the
  best MRR cell (0.844), 1024/128+reranked the best recall@10 (0.960).
- **Generation metrics** (faithfulness / answer-relevancy) are **pending an
  Anthropic API key** — see below.

## Pipeline modes & the reranker

- **Plain** (`mode=plain`): dense / sparse / hybrid retrieval, optional rerank.
- **CRAG** (`mode=crag`): retrieve → grade chunks → if weak, reformulate the query
  (LLM) or widen retrieval → re-retrieve (bounded rounds) → answer, or honestly
  refuse. Built as a **LangGraph `StateGraph`** (default) with a dependency-free
  plain loop as an equivalent fallback; both are tested to behave identically.

  ```
  START → retrieve → grade ──(CORRECT / last round)──→ finalize → END
                       └────────(weak)────────→ transform → retrieve   (loop)
  ```
  (`rag_eval.graph.crag_graph_mermaid()` renders the live diagram.)
- **Tool-using agent** (`mode=agent`): an Anthropic **tool-calling loop** where the
  model decides when to call tools — `search_corpus(query, k)` (the core RAG tool,
  callable repeatedly with reformulated queries), `fetch_chunk(chunk_id)`, and
  `calculator(expression)` — then writes a grounded, cited answer. Returns the full
  tool-call trace + the chunks gathered via `search_corpus`. Key-gated (LLM-driven);
  the agent loop is unit-tested with a mocked client. (`rag_eval/tools.py`, `rag_eval/tool_agent.py`.)
- **Adaptive reranking** (`rerank_policy=auto`): the cross-encoder
  (`ms-marco-MiniLM-L-6-v2`) runs *only when dense retrieval looks uncertain*
  (low top score / tight score cluster). Also: `never`, `always`, `llm`.
- **Hybrid retrieval** (`retrieval_mode=hybrid`): dense + BM25 fused via Reciprocal
  Rank Fusion, composing with reranking and CRAG.

## Eval harness

- **Golden set** `rag_eval/evals/golden/golden_set.jsonl` — questions + reference answers +
  ground-truth `source_ids`. Questions are hand-authored; gold ids are resolved
  at the **comment level** with word-boundary, stem-tolerant regexes
  (`rag_eval/evals/golden/build_seed.py` — reproducible, spillover-free, fails loudly on
  unresolved items). Expansion toward 150–300 items (LLM-drafted, hand-verified)
  is planned once the key is available.
- **Metrics** `rag_eval/evals/retrieval_metrics.py` — recall@k, MRR, nDCG@10 (ideal DCG uses
  each item's true relevant-chunk count), refusal accuracy (deterministic, no key).
  `rag_eval/evals/stats.py` — paired McNemar + bootstrap CIs.
  `rag_eval/evals/ragas_eval.py` — Ragas faithfulness / answer-relevancy (key-gated; see note).
- **Runner** `rag_eval/evals/experiment.py` + `rag-eval` — sweeps 6 arms by
  varying `RagConfig`; measures quality + latency + CE invocation rate; stamps
  results with git SHA/timestamp/N/models. `rag-chunk-sweep` runs the
  chunk-size A/B (gold is chunk-config independent by construction).

## CI regression gate

`.github/workflows/ci.yml` runs `pytest` + `python scripts/run_eval.py --gate` on
every PR; the gate fails if the **baseline or reranked** arm's `recall@5`/`MRR`
drop below `rag_eval/evals/thresholds.json` (deterministic, no LLM; the cross-encoder is
cached via actions/cache — a reranker-path regression can no longer merge green).
`.github/workflows/nightly.yml` runs the Ragas generation eval, guarded to no-op
unless the `ANTHROPIC_API_KEY` secret is set.

## Layout

```
rag_eval/         the installable package
  config.py ingest.py chunk.py embed.py index.py    # corpus -> chunks -> FAISS
  bm25.py policy.py rerank.py retrieve.py           # sparse + adaptive rerank + hybrid
  grade.py reformulate.py agent.py graph.py         # CRAG (LangGraph + plain loop)
  tools.py tool_agent.py                            # tool-using agent (Anthropic tool-calling)
  llm.py generate.py pipeline.py                    # generation + entry point
  cli/                                              # rag-ask, rag-build-index, rag-eval, rag-chunk-sweep
  evals/                                            # golden set, metrics, stats, experiment, thresholds
app/              FastAPI API (+ /agent) + Streamlit UI (the demo, not installed)
scripts/          thin shims so `python scripts/x.py` keeps working
results/          committed A/B results (SHA-stamped)
tests/            63+ keyless tests
```

## Install & use as a library

```bash
pip install -e .                      # or: pip install -e ".[demo,dev]"
export RAG_EVAL_DATA=/path/to/data    # dir containing raw/ corpus files (defaults to ./data)
```

```python
from rag_eval import RagConfig, RagPipeline

pipe = RagPipeline.build(RagConfig())            # corpus -> chunks -> FAISS + BM25
chunks = pipe.retrieve("battery life?", k=5)     # no API key needed
crag = pipe.corrective("wireless charging?")     # CRAG loop + trace, no key
answer = pipe.answer("battery life?")            # generation (needs ANTHROPIC_API_KEY)

from rag_eval.evals.experiment import run_experiment
results = run_experiment()                       # the full A/B harness
```

## CLI usage

```bash
rag-build-index                                                # build FAISS + BM25 (no key)
rag-ask "How is the battery life?"                             # plain retrieval
rag-ask "wireless charging support" --mode crag                # CRAG trace
rag-ask "battery life, and 60/200 as a percent?" --mode agent  # tool agent (needs key)
rag-eval                                                       # A/B table + significance (no key)
rag-eval --gate                                                # CI gate (baseline + reranked)
rag-chunk-sweep                                                # chunk-size A/B (no key)
python -m pytest -q                                            # 63 tests, no key

# (python scripts/ask.py etc. still work from a source checkout)

# Demo
uvicorn app.main:app --reload          # API at /docs
streamlit run app/ui.py                # interactive UI with the CRAG trace
```

## Models

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (adaptively gated).
- **Generation / judge:** Anthropic `claude-haiku-4-5-20251001` (pinned;
  `claude-sonnet-4-6` as the quality upgrade).

## Generation eval — pending key & dependency note

Ragas faithfulness/answer-relevancy and the LLM grader/reformulation need
`ANTHROPIC_API_KEY`. Ragas is an **optional `[eval]` extra**: current `ragas`
(0.4.x) is import-incompatible with the `langchain-core`/`langgraph` 1.x stack the
CRAG graph uses, so run the generation eval in an isolated env (or with a matching
ragas pin). The retrieval A/B + CI gate are fully functional without it.

## Limitations & honesty

- **N=99 answerable items** — reranking deltas clear paired significance
  decisively (p=0.002); hybrid remains directional. Next: LLM-drafted +
  hand-verified expansion toward 150–300 and full human gold verification.
- **Gold labels are keyword-derived** (word-boundary, comment-level, reproducible —
  but still lexical). The BM25-only control arm quantifies the resulting bias and
  the de-lexicalized paraphrase items probe semantic retrieval; full hand
  verification of every gold comment is planned with the golden-set expansion.
- **"Co-occurrence" items are single-comment compound questions**, not true
  cross-document multi-hop (named honestly).
- **CRAG's measured numbers ran keyless**: LLM query-reformulation fell back to
  pool-widening in every round, so the reported CRAG arm exercises the
  grade→widen→re-retrieve loop, not LLM rewriting.
- The abstention threshold (dense cosine < 0.35) is an **uncalibrated default**,
  deliberately not tuned on the golden refuse items.
- LLM-as-judge is noisy; the judge model + N are reported with any generation
  numbers. "Built/evaluated", not "production".
- Corpus is public user-generated review content ported from the Sparkathon
  project; FactSet's internal systems are **not** used here.
