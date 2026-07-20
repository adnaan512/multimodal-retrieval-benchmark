"""
Hard-negative reranking (RQ3): a training-free post-processing step applied
to the top-K candidates of each query.

Algorithm:
    Across the whole test set, some candidate images/captions appear as
    *incorrect* top-1 predictions unusually often — these are "hard negatives"
    that CLIP's embedding space places suspiciously close to many different
    queries (visually generic scenes, common phrasing, etc.). We compute a
    hard-negative frequency score per candidate from the *initial* ranking
    pass, then, for each query's top-K list, apply a mild similarity penalty
    to candidates with high hard-negative frequency and re-sort.

Properties:
    - **Training-free:** no CLIP parameter is updated; only the frozen
      similarity ranking is reweighted using dataset-level statistics.
    - **O(Q·K) overhead:** negligible compared to the O(Q·N) retrieval pass.
    - **Non-destructive:** original ``RetrievalResult`` objects are never
      mutated; new objects are returned.

Reference:
    Inspired by hard-negative mining at *training* time (Faghri et al.,
    VSE++, BMVC 2018). This project applies the same intuition at *test*
    time, with no labelled triplets or model updates.

Warning:
    The frequency table is computed on the same test split used for
    evaluation. This introduces mild in-distribution optimism. Treat the
    measured improvement as an upper bound on generalisation performance.
"""
from __future__ import annotations

from collections import Counter
from typing import List

from src.models import RetrievalResult

DEFAULT_PENALTY_WEIGHT: float = 0.15
"""Scalar weight applied to the normalised hard-negative frequency penalty.

Higher values penalise suspected hard negatives more aggressively.
The optimal value is dataset-dependent and was not tuned via cross-validation
in this project (see ``docs/RESEARCH.md`` Limitations).
"""


def compute_hard_negative_frequencies(results: List[RetrievalResult]) -> Counter:
    """Count how often each candidate appears as an incorrect top-1 prediction.

    A candidate is counted as a hard negative for a query when it is ranked
    first *and* it is not in the query's ground-truth set. High counts indicate
    candidates that CLIP's embedding space places close to many unrelated queries
    (e.g., visually generic scenes or common phrasing).

    Args:
        results: List of retrieval results from a first-pass ranking.
            Results with empty ``retrieved_indices`` are skipped.

    Returns:
        A ``Counter`` mapping candidate index → number of queries for which
        it was an incorrect top-1 prediction. Returns an empty ``Counter``
        if all queries have empty ranked lists.
    """
    freq: Counter = Counter()
    for r in results:
        if not r.retrieved_indices:
            continue
        top1 = r.retrieved_indices[0]
        if top1 not in r.ground_truth_indices:
            freq[top1] += 1
    return freq


def rerank_with_hard_negatives(
    results: List[RetrievalResult],
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
) -> List[RetrievalResult]:
    """Apply a hard-negative frequency penalty to each result and re-rank.

    For each query's top-K candidate list, the adjusted score is::

        adjusted_score(c) = similarity(c) - penalty_weight × freq(c) / max_freq

    where ``freq(c)`` is the number of queries for which candidate ``c`` was
    an incorrect top-1 prediction, and ``max_freq`` is the maximum such count
    across all candidates. Candidates are then re-sorted by ``adjusted_score``
    in descending order.

    Args:
        results: List of ``RetrievalResult`` objects from a first-pass ranking.
            The originals are **not** mutated.
        penalty_weight: Scaling factor for the hard-negative penalty in
            ``[0, 1]``. Defaults to ``DEFAULT_PENALTY_WEIGHT`` (0.15).

    Returns:
        A new list of ``RetrievalResult`` objects with reranked
        ``retrieved_indices`` and ``similarity_scores`` (adjusted values).
        If no hard negatives were found (all top-1 predictions are correct),
        the original list is returned unchanged.
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
