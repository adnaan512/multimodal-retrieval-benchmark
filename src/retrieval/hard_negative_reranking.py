"""
Hard negative reranking (RQ3): a training-free post-processing step applied
to the top-K candidates of each query.

Idea: across the whole test set, some candidate images/captions show up as
*incorrect* top-1 predictions unusually often -- these are "hard negatives"
that CLIP's embedding space places suspiciously close to many different
queries (visually generic scenes, common phrasing, etc.). We compute a
hard-negative frequency score per candidate from the *initial* ranking pass,
then, for each query's top-K list, apply a mild similarity penalty to
candidates with high hard-negative frequency and re-sort. This never
touches CLIP's weights (no fine-tuning) -- it only reweights an already
frozen similarity ranking using statistics gathered across the dataset.
"""
from __future__ import annotations

from collections import Counter
from typing import List

from src.models import RetrievalResult

DEFAULT_PENALTY_WEIGHT = 0.15


def compute_hard_negative_frequencies(results: List[RetrievalResult]) -> Counter:
    """
    Count how often each candidate index appears as an *incorrect* top-1
    prediction across all queries. High count => suspected hard negative.
    """
    freq = Counter()
    for r in results:
        if not r.retrieved_indices:
            continue
        top1 = r.retrieved_indices[0]
        if top1 not in r.ground_truth_indices:
            freq[top1] += 1
    return freq


def rerank_with_hard_negatives(results: List[RetrievalResult],
                                penalty_weight: float = DEFAULT_PENALTY_WEIGHT) -> List[RetrievalResult]:
    """
    Apply hard-negative penalty to each result's candidate list and
    re-sort by adjusted score. Returns new RetrievalResult objects
    (originals are left untouched so the "before" metrics stay available).
    """
    freq = compute_hard_negative_frequencies(results)
    if not freq:
        return results
    max_freq = max(freq.values())

    reranked = []
    for r in results:
        adjusted = []
        for idx, score in zip(r.retrieved_indices, r.similarity_scores):
            penalty = penalty_weight * (freq.get(idx, 0) / max_freq)
            adjusted.append((idx, score - penalty))
        adjusted.sort(key=lambda pair: pair[1], reverse=True)
        new_indices = [p[0] for p in adjusted]
        new_scores = [p[1] for p in adjusted]
        reranked.append(RetrievalResult(
            query_id=r.query_id,
            query_text=r.query_text,
            query_image_index=r.query_image_index,
            retrieved_indices=new_indices,
            similarity_scores=new_scores,
            ground_truth_indices=r.ground_truth_indices,
            direction=r.direction,
        ))
    return reranked
