# RAG Evaluation Harness (with adaptive reranking + Corrective-RAG)

A reusable **RAG evaluation harness with a CI regression gate**, an agentic
**Corrective-RAG (CRAG)** pipeline built on **LangGraph**, adaptive cross-encoder
reranking, and hybrid retrieval — measured with a versioned golden set over a real
corpus of Motorola Edge 50 Fusion user reviews (public scraped YouTube/Reddit
comments, ported from the Sparkathon project).

The harness produces a before→after A/B table across retrieval strategies; the CI
gate fails the build on retrieval regressions.

## Results (A/B over the golden set)

Golden set: **20 answerable + 6 not-in-corpus** items (factoid / multi-hop /
paraphrase / refuse). Retrieve depth-10; a chunk is a hit iff its `source_ids`
intersect the item's ground-truth `source_ids`. Reproduce with
`python scripts/run_eval.py`.

| arm | recall@1 | recall@3 | recall@5 | recall@10 | MRR | refusal_acc |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (dense top-k) | 0.450 | 0.700 | 0.750 | 0.900 | 0.591 | 0.00 |
| + reranker (always) | **0.700** | 0.750 | **0.900** | 0.900 | **0.762** | 0.00 |
| + reranker (adaptive) | 0.650 | 0.700 | 0.850 | 0.900 | 0.720 | 0.00 |
| hybrid (dense+BM25 RRF) | 0.450 | **0.900** | **0.900** | 0.900 | 0.658 | 0.00 |
| CRAG (agentic) | **0.700** | 0.750 | 0.750 | 0.800 | 0.732 | **0.67** |

**Reading it honestly (N=20, small — deltas are indicative, not significant):**
- The **cross-encoder reranker** is the biggest retrieval win: **recall@1 0.45→0.70**
  and **MRR 0.59→0.76**.
- **Adaptive** reranking recovers most of that gain while only invoking the
  cross-encoder on low-confidence queries (a latency/cost saving).
- **Hybrid** retrieval most improves mid-rank recall (**recall@3 0.70→0.90**).
- **CRAG** is the only arm that handles not-in-corpus questions — **refusal
  accuracy 0.00→0.67** (the threshold grader still false-accepts some; an LLM
  grader, once the key is available, should improve this).
- **Generation metrics** (Ragas faithfulness / answer-relevancy) are **pending an
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
  ground-truth `source_ids`. The seed is hand-authored (questions) with gold ids
  resolved by content match (`eval/golden/build_seed.py`, reproducible, no
  retriever leakage). `eval/build_golden.py` (LLM-assisted, key-gated) expands it
  toward 150–300 later.
- **Metrics** `eval/retrieval_metrics.py` — recall@k, MRR, refusal accuracy
  (deterministic, no key). `eval/ragas_eval.py` — Ragas faithfulness /
  answer-relevancy (key-gated; see note).
- **Runner** `eval/experiment.py` + `scripts/run_eval.py` — sweeps the 5 arms by
  varying `RagConfig`; writes `eval/results/results.{csv,md}`.

## CI regression gate

`.github/workflows/ci.yml` runs `pytest` + `python scripts/run_eval.py --gate` on
every PR; the gate fails if the deterministic **baseline** arm's `recall@5`/`MRR`
drop below `eval/thresholds.json` (no LLM / no cross-encoder download → fast,
reproducible). `.github/workflows/nightly.yml` runs the Ragas generation eval,
guarded to no-op unless the `ANTHROPIC_API_KEY` secret is set.

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
python scripts/run_eval.py                                             # A/B table (no key)
python scripts/run_eval.py --gate                                      # CI gate
python -m pytest -q                                                    # 41 tests, no key

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

- N=20 answerable golden items (seed) — deltas are indicative; expand to 150–300
  before quoting significance.
- Opinion corpus: several chunks legitimately support a theme, so recall is a
  "found ≥1 supporting chunk" hit-rate; gold sets were tightened to stay
  discriminative.
- LLM-as-judge is noisy; the judge model + N are reported with any generation
  numbers. "Built/evaluated", not "production".
- Corpus is public user-generated review content ported from the Sparkathon
  project; FactSet's internal systems are **not** used here.
