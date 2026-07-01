"""
Analyze Recall@K failures: which queries miss, do misses cluster by
category (RQ2/RQ3), and what are the hardest individual queries.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from src.models import FailureCase, QueryAnalysis, RetrievalResult


def find_failures(results: List[RetrievalResult], categories: Optional[Sequence[QueryAnalysis]] = None,
                   k: int = 10) -> List[FailureCase]:
    """
    A failure = ground truth not present anywhere in the top-K retrieved
    candidates. categories[i].category is used to tag failure i's query
    type when available (text_to_image queries); image_to_text queries are
    tagged "n/a" since captions, not images, carry the semantic category.
    """
    failures = []
    for i, r in enumerate(results):
        top_k = r.retrieved_indices[:k]
        hit = any(idx in r.ground_truth_indices for idx in top_k)
        if hit:
            continue
        category = categories[i].category if categories is not None and i < len(categories) else "n/a"
        top1_idx = r.retrieved_indices[0] if r.retrieved_indices else -1
        top1_score = r.similarity_scores[0] if r.similarity_scores else 0.0
        failures.append(FailureCase(
            query_id=r.query_id,
            query_text=r.query_text,
            query_image_index=r.query_image_index,
            category=category,
            top1_index=top1_idx,
            top1_score=top1_score,
            ground_truth_indices=r.ground_truth_indices,
            ground_truth_score=None,
            direction=r.direction,
        ))
    return failures


def failure_rate_by_category(results: List[RetrievalResult], categories: Sequence[QueryAnalysis],
                              k: int = 1) -> Dict[str, Dict[str, float]]:
    """Per-category failure rate = failures / total queries in that category."""
    totals = defaultdict(int)
    fails = defaultdict(int)
    for i, r in enumerate(results):
        cat = categories[i].category if i < len(categories) else "n/a"
        totals[cat] += 1
        top_k = r.retrieved_indices[:k]
        if not any(idx in r.ground_truth_indices for idx in top_k):
            fails[cat] += 1

    out = {}
    for cat, total in totals.items():
        rate = fails[cat] / total if total else 0.0
        out[cat] = {"failures": fails[cat], "total": total, "failure_rate": round(rate, 4)}
    return out


def hardest_queries(results: List[RetrievalResult], categories: Optional[Sequence[QueryAnalysis]] = None,
                     top_n: int = 10) -> List[FailureCase]:
    """
    Rank *all* queries (not just outright failures) by similarity to their
    ground truth (approximated by the best score among ground-truth
    indices that appear anywhere in the ranked list; queries where the
    ground truth never appears among retrieved candidates are treated as
    score -1, i.e. hardest). Returns the top_n hardest.
    """
    scored = []
    for i, r in enumerate(results):
        gt_score = None
        for idx, score in zip(r.retrieved_indices, r.similarity_scores):
            if idx in r.ground_truth_indices:
                gt_score = score
                break
        effective_score = gt_score if gt_score is not None else -1.0
        category = categories[i].category if categories is not None and i < len(categories) else "n/a"
        top1_idx = r.retrieved_indices[0] if r.retrieved_indices else -1
        top1_score = r.similarity_scores[0] if r.similarity_scores else 0.0
        scored.append((effective_score, FailureCase(
            query_id=r.query_id,
            query_text=r.query_text,
            query_image_index=r.query_image_index,
            category=category,
            top1_index=top1_idx,
            top1_score=top1_score,
            ground_truth_indices=r.ground_truth_indices,
            ground_truth_score=gt_score,
            direction=r.direction,
        )))
    scored.sort(key=lambda pair: pair[0])
    return [case for _, case in scored[:top_n]]


def semantic_failure_summary(failure_rates: Dict[str, Dict[str, float]]) -> str:
    """
    Produce a one-line, honest finding string from category failure rates,
    e.g. "spatial_relation queries fail 3.2x more often than object_centric
    queries (42.0% vs 13.1%)". Used directly in the HTML report and README.
    """
    if not failure_rates:
        return "No failure data available."
    ranked = sorted(failure_rates.items(), key=lambda kv: kv[1]["failure_rate"], reverse=True)
    worst_cat, worst_stats = ranked[0]
    best_cat, best_stats = ranked[-1]
    if best_stats["failure_rate"] == 0:
        multiplier_str = "N/A (best category has 0% failures)"
    else:
        multiplier = worst_stats["failure_rate"] / best_stats["failure_rate"]
        multiplier_str = f"{multiplier:.1f}x"
    return (
        f"{worst_cat} queries fail most often ({worst_stats['failure_rate']*100:.1f}%), "
        f"{multiplier_str} the rate of {best_cat} queries ({best_stats['failure_rate']*100:.1f}%)."
    )
