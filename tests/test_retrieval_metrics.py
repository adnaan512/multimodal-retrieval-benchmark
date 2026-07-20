"""Unit tests for Recall@K math: R@1=1.0 when ground truth is top-1, R@K monotonic in K."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.evaluator import median_rank, recall_at_k  # noqa: E402
from src.models import RetrievalResult  # noqa: E402


def _make_result(retrieved, ground_truth, direction="text_to_image"):
    return RetrievalResult(
        query_id=0, query_text="test query", query_image_index=None,
        retrieved_indices=retrieved,
        similarity_scores=[1.0 - 0.01 * i for i in range(len(retrieved))],
        ground_truth_indices=ground_truth,
        direction=direction,
    )


def test_recall_at_1_perfect_when_top1_correct():
    results = [_make_result([5, 1, 2, 3], [5])]
    assert recall_at_k(results, 1) == 1.0


def test_recall_at_1_zero_when_top1_wrong():
    results = [_make_result([1, 5, 2, 3], [5])]
    assert recall_at_k(results, 1) == 0.0


def test_recall_at_5_hit_when_gt_in_top5_not_top1():
    results = [_make_result([1, 2, 3, 4, 5], [5])]
    assert recall_at_k(results, 5) == 1.0
    assert recall_at_k(results, 1) == 0.0


def test_recall_at_k_monotonically_increases_with_k():
    results = [_make_result([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [7])]
    r1 = recall_at_k(results, 1)
    r5 = recall_at_k(results, 5)
    r10 = recall_at_k(results, 10)
    assert r1 <= r5 <= r10
    assert r1 == 0.0
    assert r5 == 0.0
    assert r10 == 1.0


def test_recall_averages_over_multiple_queries():
    results = [
        _make_result([1, 2, 3], [1]),   # hit @1
        _make_result([2, 1, 3], [1]),   # miss @1, hit @2
        _make_result([2, 3, 4], [1]),   # total miss (gt not present)
    ]
    assert recall_at_k(results, 1) == 1 / 3
    assert abs(recall_at_k(results, 2) - 2 / 3) < 1e-9


def test_multiple_ground_truth_indices_image_to_text():
    """image->text: ground truth is ANY of the 5 captions for that image."""
    results = [_make_result([10, 11, 12], [11, 20, 21, 22, 23], direction="image_to_text")]
    assert recall_at_k(results, 1) == 0.0
    assert recall_at_k(results, 2) == 1.0


def test_median_rank_of_ground_truth():
    results = [
        _make_result([1, 2, 3], [1]),   # rank 1
        _make_result([2, 1, 3], [1]),   # rank 2
        _make_result([2, 3, 1], [1]),   # rank 3
    ]
    assert median_rank(results) == 2


def test_median_rank_none_when_no_hits():
    results = [_make_result([1, 2, 3], [99])]
    assert median_rank(results) is None


def test_empty_results_recall_is_zero():
    assert recall_at_k([], 1) == 0.0
