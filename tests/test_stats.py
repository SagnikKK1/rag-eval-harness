"""Statistics module: McNemar exact test + paired bootstrap CI."""
from __future__ import annotations

import pytest

from eval.stats import compare_arms, mcnemar_exact, paired_bootstrap_ci


def test_mcnemar_no_discordance_is_1():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_symmetry_and_known_values():
    assert mcnemar_exact(5, 0) == pytest.approx(2 * (1 / 32))  # 0.0625
    assert mcnemar_exact(0, 5) == mcnemar_exact(5, 0)
    assert mcnemar_exact(3, 3) == 1.0  # perfectly balanced discordance


def test_mcnemar_significant_when_one_sided_enough():
    assert mcnemar_exact(8, 0) < 0.05
    assert mcnemar_exact(5, 0) > 0.05  # the audit's point: 5 flips isn't significant


def test_bootstrap_ci_zero_diff():
    a = [0.5] * 20
    lo, hi = paired_bootstrap_ci(a, a)
    assert lo == 0.0 and hi == 0.0


def test_bootstrap_ci_detects_large_consistent_diff():
    a = [0.0] * 20
    b = [1.0] * 20
    lo, hi = paired_bootstrap_ci(a, b)
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_reproducible():
    a = [0.1, 0.5, 0.9, 0.2, 0.7] * 4
    b = [x + (0.3 if i % 2 else -0.1) for i, x in enumerate(a)]
    assert paired_bootstrap_ci(a, b) == paired_bootstrap_ci(a, b)


def test_compare_arms_end_to_end():
    base = [[True] + [False] * 9] * 10 + [[False] * 10] * 10   # 10 hits@1, 10 misses
    arm = [[True] + [False] * 9] * 20                          # all hit@1
    cmp = compare_arms(base, arm, k=5)
    assert cmp["discordant"] == (10, 0)
    assert cmp["recall_at_k_p"] < 0.05
    assert cmp["significant"] is True
    assert cmp["mrr_delta"] == pytest.approx(0.5)


def test_compare_arms_rejects_unpaired():
    with pytest.raises(ValueError):
        compare_arms([[True]], [[True], [False]])
