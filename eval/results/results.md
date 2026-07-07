### RAG A/B results

run: 2026-07-07 15:21 UTC · git bbfd2d3 · N=52 answerable + 14 refuse · depth 10 · embed `sentence-transformers/all-MiniLM-L6-v2` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · chunks 512/64

| arm | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 | refusal_acc | false_refusal | latency_ms | p95_ms | ce_calls/q | faithfulness | answer_relevancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.577 | 0.788 | 0.865 | 0.904 | 0.690 | 0.471 | 0.714 | 0.058 | 35 | 85 | 0.00 | — | — |
| sparse | 0.635 | 0.769 | 0.846 | 0.904 | 0.724 | 0.451 | 0.714 | 0.058 | 25 | 49 | 0.00 | — | — |
| hybrid | 0.654 | 0.904 | 0.923 | 0.942 | 0.775 | 0.511 | 0.714 | 0.058 | 39 | 67 | 0.00 | — | — |
| reranked | 0.750 | 0.885 | 0.904 | 0.923 | 0.821 | 0.577 | 0.714 | 0.058 | 352 | 616 | 1.00 | — | — |
| adaptive | 0.692 | 0.865 | 0.885 | 0.904 | 0.783 | 0.542 | 0.714 | 0.058 | 178 | 326 | 0.71 | — | — |
| crag | 0.673 | 0.750 | 0.750 | 0.750 | 0.712 | 0.333 | 0.714 | 0.154 | 799 | 1861 | 1.71 | — | — |

Refusal: `refusal_acc` = correct refusals on not-in-corpus items; `false_refusal` = wrong refusals on answerable items. Non-CRAG arms refuse when top-1 dense cosine < 0.35 (uncalibrated default, not tuned on the golden set).

#### Paired significance vs baseline

| arm vs baseline | Δrecall@5 (won/lost) | McNemar p | ΔMRR [95% CI] | significant? |
| --- | --- | --- | --- | --- |
| sparse | +4/−5 | 1.000 | +0.034 [-0.098, +0.164] | no |
| hybrid | +4/−1 | 0.375 | +0.085 [-0.009, +0.184] | no |
| reranked | +3/−1 | 0.625 | +0.131 [+0.037, +0.227] | yes |
| adaptive | +2/−1 | 1.000 | +0.093 [+0.011, +0.177] | yes |
| crag | +3/−9 | 0.146 | +0.022 [-0.108, +0.146] | no |

At this N, treat non-significant deltas as directional only.

#### Per-type breakdown

| arm | cooccurrence r@5 / MRR (n) | factoid r@5 / MRR (n) | paraphrase r@5 / MRR (n) |
| --- | --- | --- | --- |
| baseline | 0.83 / 0.57 (12) | 0.93 / 0.83 (28) | 0.75 / 0.49 (12) |
| sparse | 1.00 / 0.81 (12) | 0.96 / 0.86 (28) | 0.42 / 0.33 (12) |
| hybrid | 1.00 / 0.78 (12) | 1.00 / 0.90 (28) | 0.67 / 0.47 (12) |
| reranked | 1.00 / 0.96 (12) | 0.93 / 0.89 (28) | 0.75 / 0.53 (12) |
| adaptive | 0.92 / 0.83 (12) | 0.93 / 0.87 (28) | 0.75 / 0.53 (12) |
| crag | 0.92 / 0.83 (12) | 0.82 / 0.80 (28) | 0.42 / 0.38 (12) |

#### Abstention risk–coverage (dense-confidence threshold sweep)

| threshold | refusal_acc (refuse items) | false_refusal (answerable) |
| --- | --- | --- |
| 0.25 | 0.429 | 0.000 |
| 0.30 | 0.571 | 0.000 |
| 0.35 | 0.714 | 0.058 |
| 0.40 | 0.786 | 0.077 |
| 0.45 | 0.857 | 0.250 |
| 0.50 | 0.929 | 0.481 |

_Generation columns pending ANTHROPIC_API_KEY (`--generate`)._
