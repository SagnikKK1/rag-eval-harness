"""Paired significance statistics for the A/B table (stdlib only, seeded).

Two complementary treatments, both *paired* (same golden items across arms):

- ``mcnemar_exact`` — for binary per-item outcomes (hit@k): exact two-sided
  binomial test on the discordant pairs. The right test for "arm B fixed b items
  and broke c items relative to arm A".
- ``paired_bootstrap_ci`` — for real-valued per-item scores (reciprocal ranks):
  percentile CI of the mean difference over resampled item indices.

With N≈20 most deltas will NOT clear these bars — that is the point: the table
should say so instead of bolding noise.
"""
from __future__ import annotations

import random
from math import comb

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260707  # fixed for reproducibility


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value. b = items only arm-B got right,
    c = items only arm-A got right. Returns 1.0 when there are no discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided binomial(n, 0.5) tail probability
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    a: list[float],
    b: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for mean(b) - mean(a), resampling paired items."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples required")
    rng = random.Random(seed)
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return lo, hi


def _rr(hits: list[bool]) -> float:
    for i, h in enumerate(hits):
        if h:
            return 1.0 / (i + 1)
    return 0.0


def _hit_at(hits: list[bool], k: int) -> bool:
    return any(hits[:k])


def compare_arms(base_hits: list[list[bool]], arm_hits: list[list[bool]], k: int = 5) -> dict:
    """Paired comparison of one arm vs the baseline from per-item hit arrays."""
    if len(base_hits) != len(arm_hits):
        raise ValueError("arms were run on different item sets")
    b = sum(1 for x, y in zip(base_hits, arm_hits) if not _hit_at(x, k) and _hit_at(y, k))
    c = sum(1 for x, y in zip(base_hits, arm_hits) if _hit_at(x, k) and not _hit_at(y, k))
    base_rr = [_rr(h) for h in base_hits]
    arm_rr = [_rr(h) for h in arm_hits]
    lo, hi = paired_bootstrap_ci(base_rr, arm_rr)
    return {
        "recall_at_k_p": mcnemar_exact(b, c),
        "discordant": (b, c),
        "mrr_delta": sum(arm_rr) / len(arm_rr) - sum(base_rr) / len(base_rr),
        "mrr_ci95": (lo, hi),
        "significant": mcnemar_exact(b, c) < 0.05 or (lo > 0 or hi < 0),
    }
