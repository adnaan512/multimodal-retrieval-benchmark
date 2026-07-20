"""
Core data structures shared across the retrieval pipeline.

These dataclasses form the contract between modules: the encoder produces
embeddings, the index/retrieval modules produce RetrievalResult objects,
the analysis modules consume those to produce QueryAnalysis / FailureCase
objects, and the evaluator aggregates everything into RecallMetrics.

Design principle: all inter-module data passes through these typed
dataclasses. No module accepts or returns raw dicts except as final
serialisation targets (e.g., ``RecallMetrics.summary_row()``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievalResult:
    """The result of a single query (text→image or image→text).

    Attributes:
        query_id: Zero-based integer identifying the query within the batch.
        query_text: The raw caption string for text→image queries; ``None``
            for image→text queries.
        query_image_index: Row index into the image embedding matrix for
            image→text queries; ``None`` for text→image queries.
        retrieved_indices: Top-K candidate indices ranked by descending
            similarity score. Length equals the ``k`` passed to the
            retrieval function.
        similarity_scores: Cosine similarity scores corresponding to
            ``retrieved_indices``, in the same order.
        ground_truth_indices: Valid correct-answer indices.
            - Text→Image: a single-element list containing the one correct image index.
            - Image→Text: a list of all caption indices that describe the query image
              (typically 5 for Flickr30K).
        direction: Either ``"text_to_image"`` or ``"image_to_text"``.
    """

    query_id: int
    query_text: Optional[str]
    query_image_index: Optional[int]
    retrieved_indices: List[int]
    similarity_scores: List[float]
    ground_truth_indices: List[int]
    direction: str

    def rank_of_ground_truth(self) -> Optional[int]:
        """Return the 1-indexed rank of the first correct answer in the ranked list.

        Returns:
            The 1-indexed position of the first ground-truth index found in
            ``retrieved_indices``, or ``None`` if no ground-truth index appears.
        """
        for rank, idx in enumerate(self.retrieved_indices, start=1):
            if idx in self.ground_truth_indices:
                return rank
        return None

    def hit_at_k(self, k: int) -> bool:
        """Return True if a ground-truth answer appears in the top-K results.

        Args:
            k: Cutoff rank (1-indexed, inclusive).

        Returns:
            ``True`` if ``rank_of_ground_truth() <= k``, ``False`` otherwise.
        """
        rank = self.rank_of_ground_truth()
        return rank is not None and rank <= k


@dataclass
class QueryAnalysis:
    """Rule-based semantic categorisation of a single caption/query.

    Attributes:
        query_id: Index of this query in the caption list (assigned by
            ``categorize_captions`` after construction).
        text: The raw caption string.
        category: One of ``"object_centric"``, ``"scene_centric"``,
            ``"attribute_centric"``, or ``"spatial_relation"``.
        matched_pattern: The specific keyword or regex match that triggered
            the assigned category; ``None`` for the ``object_centric`` fallback.
    """

    query_id: int
    text: str
    category: str
    matched_pattern: Optional[str] = None


@dataclass
class FailureCase:
    """A Recall@K miss, with enough diagnostic context to inspect why.

    Attributes:
        query_id: Index of the failing query.
        query_text: Caption text (text→image queries); ``None`` otherwise.
        query_image_index: Image index (image→text queries); ``None`` otherwise.
        category: Semantic category of the query (from ``QueryAnalysis``).
        top1_index: The (incorrect) candidate returned at rank 1.
        top1_score: Cosine similarity of the rank-1 candidate.
        ground_truth_indices: The correct answer(s) that were not retrieved.
        ground_truth_score: Cosine similarity of the correct answer if it
            appears anywhere in the ranked list; ``None`` if it does not.
        direction: ``"text_to_image"`` or ``"image_to_text"``.
    """

    query_id: int
    query_text: Optional[str]
    query_image_index: Optional[int]
    category: str
    top1_index: int
    top1_score: float
    ground_truth_indices: List[int]
    ground_truth_score: Optional[float]
    direction: str


@dataclass
class RecallMetrics:
    """Aggregate retrieval metrics for one direction + backbone combination.

    Attributes:
        direction: ``"text_to_image"`` or ``"image_to_text"``.
        backbone: Name of the CLIP backbone used (e.g., ``"vit-l-14"``).
        num_queries: Total number of queries evaluated.
        recall_at_1: Fraction of queries with a hit at rank 1 (R@1).
        recall_at_5: Fraction of queries with a hit at rank 5 (R@5).
        recall_at_10: Fraction of queries with a hit at rank 10 (R@10).
        median_rank: Median 1-indexed rank of the first correct answer across
            all queries; ``None`` if no query has a ground-truth answer in
            the ranked list.
        per_category: Mapping of category name → ``{"r@1": float, "n": int}``,
            populated when ``categories`` is passed to ``evaluate()``.
    """

    direction: str
    backbone: str
    num_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    median_rank: Optional[float]
    per_category: dict = field(default_factory=dict)

    def summary_row(self) -> dict:
        """Serialise to a flat dict suitable for a table row or CSV export.

        Returns:
            Dictionary with keys: ``direction``, ``backbone``, ``n``,
            ``R@1``, ``R@5``, ``R@10``, ``median_rank``.
        """
        return {
            "direction": self.direction,
            "backbone": self.backbone,
            "n": self.num_queries,
            "R@1": round(self.recall_at_1, 4),
            "R@5": round(self.recall_at_5, 4),
            "R@10": round(self.recall_at_10, 4),
            "median_rank": self.median_rank,
        }
