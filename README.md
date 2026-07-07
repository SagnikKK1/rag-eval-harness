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

Golden set: **52 answerable + 14 not-in-corpus** items (factoid / co-occurrence /
paraphrase / refuse, incl. *near-domain* refuses like "Samsung washing machine
price"). Retrieve depth-10; a retrieved chunk is a hit iff it contains a gold
comment. `sparse` (BM25-only) is the **lexical control** — gold labels are
keyword-resolved, so this arm bounds how much score keyword matching alone
explains. Reproduce with `python scripts/run_eval.py`.

| arm | recall@1 | recall@5 | MRR | nDCG@10 | refusal_acc | false_refusal | latency ms/q | CE calls/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (dense) | 0.577 | 0.865 | 0.690 | 0.471 | 0.714 | 0.058 | 26 | 0 |
| sparse (BM25 control) | 0.635 | 0.846 | 0.724 | 0.451 | 0.714 | 0.058 | 13 | 0 |
| hybrid (RRF) | 0.654 | 0.923 | 0.775 | 0.511 | 0.714 | 0.058 | 23 | 0 |
| **reranked (always)** | **0.750** | 0.904 | **0.821** | **0.577** | 0.714 | 0.058 | 298 | 1.00 |
| reranked (adaptive) | 0.692 | 0.885 | 0.783 | 0.542 | 0.714 | 0.058 | 166 | 0.71 |
| CRAG (agentic) | 0.673 | 0.750 | 0.712 | 0.333 | 0.714 | 0.154 | 835 | 1.71 |

**Paired significance vs baseline** (McNemar on recall@5; paired bootstrap 95% CI
on ΔMRR; N=52) — full tables incl. per-type breakdown and the risk–coverage sweep
in [eval/results/results.md](eval/results/results.md):

- **Cross-encoder reranking: ΔMRR +0.131, 95% CI [+0.037, +0.227] — significant.**
  Cost: ~11× baseline latency (26→298 ms/q).
- **Adaptive reranking: ΔMRR +0.093, 95% CI [+0.011, +0.177] — significant**, at
  ~56% of the always-rerank latency, invoking the cross-encoder on 71% of queries.
- **Per-type breakdown is the semantic-retrieval story**: on de-lexicalized
  paraphrase items, BM25 collapses (recall@5 0.42) while dense holds 0.75 —
  keyword matching can't find evidence that's worded differently.
- **Hybrid** trends strongest among no-CE arms (recall@3 0.90, ΔMRR +0.085
  [−0.009, +0.184]) but doesn't clear significance yet.
- **Abstention risk–coverage** (threshold sweep in results.md): at 0.35, refusal
  0.714 with 5.8% false refusals; 0.45 buys 0.857 refusal at 25% false refusals.
  CRAG matches threshold refusal while tripling false refusals (0.154) and paying
  recall@10 0.90→0.75 — its grader discards true positives.
- **Chunking sweep** ([eval/results/chunk_sweep.md](eval/results/chunk_sweep.md)):
  reranking wins at every chunk size (MRR +0.11–0.12 at each of 256/32, 512/64,
  1024/128); 1024/128+reranked is the best cell (MRR 0.861, recall@10 0.981).
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
  (`rag.graph.crag_graph_mermaid()` renders the live diagram.)
- **Tool-using agent** (`mode=agent`): an Anthropic **tool-calling loop** where the
  model decides when to call tools — `search_corpus(query, k)` (the core RAG tool,
  callable repeatedly with reformulated queries), `fetch_chunk(chunk_id)`, and
  `calculator(expression)` — then writes a grounded, cited answer. Returns the full
  tool-call trace + the chunks gathered via `search_corpus`. Key-gated (LLM-driven);
  the agent loop is unit-tested with a mocked client. (`rag/tools.py`, `rag/tool_agent.py`.)
- **Adaptive reranking** (`rerank_policy=auto`): the cross-encoder
  (`ms-marco-MiniLM-L-6-v2`) runs *only when dense retrieval looks uncertain*
  (low top score / tight score cluster). Also: `never`, `always`, `llm`.
- **Hybrid retrieval** (`retrieval_mode=hybrid`): dense + BM25 fused via Reciprocal
  Rank Fusion, composing with reranking and CRAG.

## Eval harness

- **Golden set** `eval/golden/golden_set.jsonl` — questions + reference answers +
  ground-truth `source_ids`. Questions are hand-authored; gold ids are resolved
  at the **comment level** with word-boundary, stem-tolerant regexes
  (`eval/golden/build_seed.py` — reproducible, spillover-free, fails loudly on
  unresolved items). Expansion toward 150–300 items (LLM-drafted, hand-verified)
  is planned once the key is available.
- **Metrics** `eval/retrieval_metrics.py` — recall@k, MRR, nDCG@10 (ideal DCG uses
  each item's true relevant-chunk count), refusal accuracy (deterministic, no key).
  `eval/stats.py` — paired McNemar + bootstrap CIs.
  `eval/ragas_eval.py` — Ragas faithfulness / answer-relevancy (key-gated; see note).
- **Runner** `eval/experiment.py` + `scripts/run_eval.py` — sweeps 6 arms by
  varying `RagConfig`; measures quality + latency + CE invocation rate; stamps
  results with git SHA/timestamp/N/models. `scripts/chunk_sweep.py` runs the
  chunk-size A/B (gold is chunk-config independent by construction).

## CI regression gate

`.github/workflows/ci.yml` runs `pytest` + `python scripts/run_eval.py --gate` on
every PR; the gate fails if the **baseline or reranked** arm's `recall@5`/`MRR`
drop below `eval/thresholds.json` (deterministic, no LLM; the cross-encoder is
cached via actions/cache — a reranker-path regression can no longer merge green).
`.github/workflows/nightly.yml` runs the Ragas generation eval, guarded to no-op
unless the `ANTHROPIC_API_KEY` secret is set.

## Layout

```
rag/        retrieval + generation pipeline
  config.py ingest.py chunk.py embed.py index.py     # corpus -> chunks -> FAISS
  bm25.py policy.py rerank.py retrieve.py            # sparse + adaptive rerank + hybrid
  grade.py reformulate.py agent.py graph.py          # CRAG (LangGraph + plain loop)
  tools.py tool_agent.py                             # tool-using agent (Anthropic tool-calling)
  llm.py generate.py pipeline.py                     # generation + entry point
eval/       golden set, metrics, ragas (key-gated), experiment runner, thresholds
app/        FastAPI API (+ /agent) + Streamlit UI (the demo)
scripts/    build_index.py  ask.py  run_eval.py
tests/      keyless: smoke, policy, rerank, CRAG (both engines), hybrid, metrics, api, graph, tools, tool_agent
```

## Setup

```bash
conda create -p ./.conda-env python=3.11 -y      # or any Python 3.11 venv
./.conda-env/bin/python -m pip install -r requirements.txt
cp .env.example .env                              # add ANTHROPIC_API_KEY for generation
python scripts/build_index.py                     # build FAISS + BM25 (no key)
```

## Usage

```bash
python scripts/ask.py "How is the battery life?"                       # plain retrieval
python scripts/ask.py "battery drain after android 15" --rerank-policy always
python scripts/ask.py "wireless charging support" --mode crag          # CRAG trace
python scripts/ask.py "battery life, and 60/200 as a percent?" --mode agent  # tool agent (needs key)
python scripts/run_eval.py                                             # A/B table + significance (no key)
python scripts/run_eval.py --gate                                      # CI gate (baseline + reranked)
python scripts/chunk_sweep.py                                          # chunk-size A/B (no key)
python -m pytest -q                                                    # 59 tests, no key

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

- **N=52 answerable items** — enough for the reranking deltas to clear paired
  significance, but CIs are still wide; hybrid/CRAG deltas remain directional.
  Expansion toward 150–300 (LLM-drafted, hand-verified) is the next step.
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
