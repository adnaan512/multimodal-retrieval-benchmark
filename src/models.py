"""
Core data structures shared across the retrieval pipeline.

These dataclasses form the contract between modules: the encoder produces
embeddings, the index/retrieval modules produce RetrievalResult objects,
the analysis modules consume those to produce QueryAnalysis / FailureCase
objects, and the evaluator aggregates everything into RecallMetrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievalResult:
    """The result of a single query (text->image or image->text)."""

    query_id: int
    query_text: Optional[str]              # set for text->image queries
    query_image_index: Optional[int]       # set for image->text queries
    retrieved_indices: List[int]           # top-K candidate indices, ranked
    similarity_scores: List[float]         # cosine similarities, same order
    ground_truth_indices: List[int]        # valid correct answer(s)
    direction: str                         # "text_to_image" | "image_to_text"

    def rank_of_ground_truth(self) -> Optional[int]:
        """1-indexed rank of the *first* correct answer found, else None."""
        for rank, idx in enumerate(self.retrieved_indices, start=1):
            if idx in self.ground_truth_indices:
                return rank
        return None

    def hit_at_k(self, k: int) -> bool:
        rank = self.rank_of_ground_truth()
        return rank is not None and rank <= k


@dataclass
class QueryAnalysis:
    """Rule-based semantic categorization of a single caption/query."""

    query_id: int
    text: str
    category: str  # object_centric | scene_centric | attribute_centric | spatial_relation
    matched_pattern: Optional[str] = None


@dataclass
class FailureCase:
    """A Recall@1 (or Recall@K) miss, with enough context to inspect why."""

    query_id: int
    query_text: Optional[str]
    query_image_index: Optional[int]
    category: str
    top1_index: int
    top1_score: float
    ground_truth_indices: List[int]
    ground_truth_score: Optional[float]  # similarity to the true match, if computable
    direction: str


@dataclass
class RecallMetrics:
    """Aggregate retrieval metrics for one direction (and optionally backbone)."""

    direction: str
    backbone: str
    num_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    median_rank: Optional[float]
    per_category: dict = field(default_factory=dict)  # category -> {"r@1":.., "n":..}

    def summary_row(self) -> dict:
        return {
            "direction": self.direction,
            "backbone": self.backbone,
            "n": self.num_queries,
            "R@1": round(self.recall_at_1, 4),
            "R@5": round(self.recall_at_5, 4),
            "R@10": round(self.recall_at_10, 4),
            "median_rank": self.median_rank,
        }
