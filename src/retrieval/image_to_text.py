"""
Image -> Text retrieval: for each image in the test set, find the top-K
most similar captions. Ground truth for image i is the set of 5 captions
written to describe it.
"""
from __future__ import annotations

from typing import List, Sequence

from src.indexing.embedding_index import retrieve_batch
from src.models import RetrievalResult


def run_image_to_text(
    image_embeddings,
    text_embeddings,
    image_to_caption_indices: Sequence[Sequence[int]],
    k: int = 10,
) -> List[RetrievalResult]:
    """
    image_embeddings: (N, 512) matrix, one row per image
    text_embeddings: (Q, 512) matrix, one row per caption
    image_to_caption_indices[i]: list of caption indices (usually 5) that describe image i
    """
    order, scores = retrieve_batch(image_embeddings, text_embeddings, k=k)
    results = []
    for i in range(image_embeddings.shape[0]):
        results.append(RetrievalResult(
            query_id=i,
            query_text=None,
            query_image_index=i,
            retrieved_indices=order[i].tolist(),
            similarity_scores=scores[i].tolist(),
            ground_truth_indices=list(image_to_caption_indices[i]),
            direction="image_to_text",
        ))
    return results
