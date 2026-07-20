"""
Recall@K and median-rank computation, with per-category breakdown and
backbone comparison support (RQ1).

Standard Information Retrieval definitions used throughout:

    Recall@K = (1 / |Q|) × Σ_{q ∈ Q} 𝟙[ground_truth(q) ∈ top-K(q)]

    Median Rank = median over all queries of the 1-indexed position of the
                  first correct answer in the ranked list.

Both metrics are computed independently for Text→Image and Image→Text
retrieval directions.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence

from src.models import QueryAnalysis, RecallMetrics, RetrievalResult


def recall_at_k(results: List[RetrievalResult], k: int) -> float:
    """Compute Recall@K across a list of retrieval results.

    A query is counted as a hit if *any* of its ground-truth indices appears
    within the first ``k`` positions of its ranked candidate list.

    Args:
        results: List of ``RetrievalResult`` objects, one per query.
        k: Cutoff rank (1-indexed, inclusive).

    Returns:
        Fraction of queries that are hits at rank ``k``, in ``[0.0, 1.0]``.
        Returns ``0.0`` for an empty list.
    """
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.hit_at_k(k))
    return hits / len(results)


def median_rank(results: List[RetrievalResult]) -> Optional[float]:
    """Compute the median 1-indexed rank of the first correct answer.

    Queries where the ground truth does not appear anywhere in the ranked
    list are excluded from the median computation (they would require knowing
    the full corpus size to assign a meaningful rank).

    Args:
        results: List of ``RetrievalResult`` objects.

    Returns:
        Median rank as a float, or ``None`` if no query has a ground-truth
        answer in its ranked list.
    """
    ranks = [r.rank_of_ground_truth() for r in results]
    ranks = [r for r in ranks if r is not None]
    if not ranks:
        return None
    return statistics.median(ranks)


def per_category_recall_at_1(
    results: List[RetrievalResult],
    categories: Sequence[QueryAnalysis],
) -> Dict[str, dict]:
    """Compute Recall@1 broken down by query semantic category.

    Args:
        results: List of retrieval results, one per query. Must have the
            same length as ``categories``.
        categories: Sequence of ``QueryAnalysis`` objects providing the
            category label for each query.

    Returns:
        Mapping of category name → ``{"r@1": float, "n": int}``.
        ``"r@1"`` is rounded to 4 decimal places.
    """
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for i, r in enumerate(results):
        cat = categories[i].category if i < len(categories) else "n/a"
        buckets[cat].append(r)
    out = {}
    for cat, items in buckets.items():
        out[cat] = {"r@1": round(recall_at_k(items, 1), 4), "n": len(items)}
    return out


def evaluate(
    results: List[RetrievalResult],
    direction: str,
    backbone: str,
    categories: Optional[Sequence[QueryAnalysis]] = None,
) -> RecallMetrics:
    """Aggregate retrieval results into a ``RecallMetrics`` object.

    Args:
        results: List of ``RetrievalResult`` objects for one direction.
        direction: Either ``"text_to_image"`` or ``"image_to_text"``.
        backbone: Short backbone name (e.g., ``"vit-l-14"``), stored on
            the returned metrics for labelling purposes.
        categories: Optional sequence of ``QueryAnalysis`` objects. When
            provided, per-category Recall@1 is included in the output.

    Returns:
        A ``RecallMetrics`` instance with R@1, R@5, R@10, median rank, and
        (optionally) per-category breakdown.
    """
    per_cat = per_category_recall_at_1(results, categories) if categories is not None else {}
    return RecallMetrics(
        direction=direction,
        backbone=backbone,
        num_queries=len(results),
        recall_at_1=recall_at_k(results, 1),
        recall_at_5=recall_at_k(results, 5),
        recall_at_10=recall_at_k(results, 10),
        median_rank=median_rank(results),
        per_category=per_cat,
    )


def compare_backbones(metrics_by_backbone: Dict[str, List[RecallMetrics]]) -> List[dict]:
    """Flatten per-backbone metrics into a list of summary rows for tabulation.

    Args:
        metrics_by_backbone: Mapping of backbone name → list of
            ``RecallMetrics`` (one per retrieval direction).

    Returns:
        List of ``summary_row()`` dicts suitable for a README table or
        HTML report, ordered by backbone then direction.
    """
    rows = []
    for _backbone, metrics_list in metrics_by_backbone.items():
        for m in metrics_list:
            rows.append(m.summary_row())
    return rows
