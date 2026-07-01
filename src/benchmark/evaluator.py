"""
Recall@K and median-rank computation, with per-category breakdown and
backbone comparison support (RQ1).
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence

from src.models import QueryAnalysis, RecallMetrics, RetrievalResult


def recall_at_k(results: List[RetrievalResult], k: int) -> float:
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.hit_at_k(k))
    return hits / len(results)


def median_rank(results: List[RetrievalResult]) -> Optional[float]:
    ranks = [r.rank_of_ground_truth() for r in results]
    ranks = [r for r in ranks if r is not None]
    if not ranks:
        return None
    return statistics.median(ranks)


def per_category_recall_at_1(results: List[RetrievalResult], categories: Sequence[QueryAnalysis]) -> Dict[str, dict]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, r in enumerate(results):
        cat = categories[i].category if i < len(categories) else "n/a"
        buckets[cat].append(r)
    out = {}
    for cat, items in buckets.items():
        out[cat] = {"r@1": round(recall_at_k(items, 1), 4), "n": len(items)}
    return out


def evaluate(results: List[RetrievalResult], direction: str, backbone: str,
             categories: Optional[Sequence[QueryAnalysis]] = None) -> RecallMetrics:
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
    """
    Flatten {backbone: [RecallMetrics for each direction]} into a list of
    summary rows suitable for a README/report table (ViT-B/32 vs ViT-L/14).
    """
    rows = []
    for backbone, metrics_list in metrics_by_backbone.items():
        for m in metrics_list:
            rows.append(m.summary_row())
    return rows
