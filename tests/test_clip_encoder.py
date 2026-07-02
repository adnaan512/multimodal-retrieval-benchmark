"""Unit tests for the mock CLIP encoder: normalization, similarity range, reproducibility."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbone.clip_encoder import EMBED_DIM, MockCLIPEncoder


@pytest.fixture
def encoder():
    return MockCLIPEncoder()


def test_encode_text_is_l2_normalized(encoder):
    vec = encoder.encode_text("a dog running in a park")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5


def test_encode_image_is_l2_normalized(encoder):
    vec = encoder.encode_image("mock_image_0001.jpg")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5


def test_embedding_dimension(encoder):
    vec = encoder.encode_text("a red car parked outside")
    assert vec.shape == (EMBED_DIM,)


def test_cosine_similarity_range(encoder):
    """Cosine similarity between two L2-normalized vectors must lie in [-1, 1]."""
    v1 = encoder.encode_text("a cat sitting on a mat")
    v2 = encoder.encode_text("a dog running in a field")
    sim = float(np.dot(v1, v2))
    assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6


def test_self_similarity_is_one(encoder):
    v1 = encoder.encode_text("a boat on the lake")
    sim = float(np.dot(v1, v1))
    assert abs(sim - 1.0) < 1e-5


def test_mock_encoder_reproducibility_text(encoder):
    v1 = encoder.encode_text("a child playing in a garden")
    v2 = encoder.encode_text("a child playing in a garden")
    assert np.allclose(v1, v2)


def test_mock_encoder_reproducibility_image(encoder):
    v1 = encoder.encode_image("mock_image_0042.jpg")
    v2 = encoder.encode_image("mock_image_0042.jpg")
    assert np.allclose(v1, v2)


def test_mock_encoder_different_inputs_differ(encoder):
    v1 = encoder.encode_text("a dog in a park")
    v2 = encoder.encode_text("a cat in a kitchen")
    assert not np.allclose(v1, v2)


def test_batch_encoding_matches_individual(encoder):
    texts = ["a dog running", "a cat sleeping", "a bird flying"]
    batch = encoder.encode_texts_batch(texts)
    individual = np.stack([encoder.encode_text(t) for t in texts])
    assert np.allclose(batch, individual)
