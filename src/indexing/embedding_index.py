"""
Build and persist the image/text embedding matrices used for retrieval.

Design decision (see README): embeddings are L2-normalized before being
stored, so that a plain dot product between query and index rows is
mathematically identical to cosine similarity -- no separate norm step is
needed at query time, which keeps retrieve() a single matrix multiply.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np

IMAGE_EMB_FILENAME = "image_embeddings.npy"
TEXT_EMB_FILENAME = "text_embeddings.npy"


def build_image_index(encoder, image_identifiers: Sequence[str], is_mock: bool) -> np.ndarray:
    """Encode all images into an (N, 512) L2-normalized matrix."""
    if is_mock:
        return encoder.encode_images_batch(image_identifiers)
    # Real encoder path: image_identifiers are file paths, decode with PIL.
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in image_identifiers]
    return encoder.encode_images_batch(images)


def build_text_index(encoder, captions: Sequence[str]) -> np.ndarray:
    """Encode all captions into an (N*5, 512) L2-normalized matrix."""
    return encoder.encode_texts_batch(captions)


def save_index(matrix: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, matrix)


def load_index(path: str) -> np.ndarray:
    return np.load(path)


def load_or_build_indices(
    encoder,
    image_identifiers: Sequence[str],
    captions: Sequence[str],
    cache_dir: str,
    is_mock: bool,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core caching pattern used throughout the project: never recompute
    embeddings if a cached .npy already exists on disk for this
    (backbone, dataset) combination. cache_dir should already encode the
    backbone name (e.g. .cache/vit-b-32/) so switching --backbone doesn't
    silently reuse the wrong embeddings.
    """
    img_path = os.path.join(cache_dir, IMAGE_EMB_FILENAME)
    txt_path = os.path.join(cache_dir, TEXT_EMB_FILENAME)

    if not force_rebuild and os.path.isfile(img_path) and os.path.isfile(txt_path):
        return load_index(img_path), load_index(txt_path)

    image_matrix = build_image_index(encoder, image_identifiers, is_mock)
    text_matrix = build_text_index(encoder, captions)
    save_index(image_matrix, img_path)
    save_index(text_matrix, txt_path)
    return image_matrix, text_matrix


def retrieve(query_embedding: np.ndarray, index_matrix: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
    """
    Rank all rows of index_matrix by cosine similarity to query_embedding
    (both assumed L2-normalized) and return the top-K (indices, scores).
    """
    scores = index_matrix @ query_embedding  # (N,)
    order = np.argsort(scores)[::-1][:k]
    return order.tolist(), scores[order].tolist()


def retrieve_batch(query_matrix: np.ndarray, index_matrix: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized retrieve() over many queries at once: (Q, 512) x (N, 512) -> (Q, N)."""
    scores = query_matrix @ index_matrix.T  # (Q, N)
    order = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    top_scores = np.take_along_axis(scores, order, axis=1)
    return order, top_scores
