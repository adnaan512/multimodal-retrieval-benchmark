"""
Text -> Image retrieval: for each caption in the test set, find the top-K
most similar images. Ground truth for caption i is the single image it
was written to describe.
"""
from __future__ import annotations

from typing import List, Sequence

from src.indexing.embedding_index import retrieve_batch
from src.models import RetrievalResult


def run_text_to_image(text_embeddings, image_embeddings,
                       caption_to_image_index: Sequence[int], k: int = 10) -> List[RetrievalResult]:
    """
    text_embeddings: (Q, 512) matrix, one row per caption (Q = N_images * 5)
    image_embeddings: (N, 512) matrix, one row per image
    caption_to_image_index[i]: index into image_embeddings that caption i belongs to
    """
    order, scores = retrieve_batch(text_embeddings, image_embeddings, k=k)
    results = []
    for i in range(text_embeddings.shape[0]):
        results.append(RetrievalResult(
            query_id=i,
            query_text=None,  # filled in by caller if caption text is available
            query_image_index=None,
            retrieved_indices=order[i].tolist(),
            similarity_scores=scores[i].tolist(),
            ground_truth_indices=[caption_to_image_index[i]],
            direction="text_to_image",
        ))
    return results
