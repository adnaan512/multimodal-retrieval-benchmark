"""
Shared pytest fixture: 50 image-caption pairs from the mock dataset,
always using MockCLIPEncoder. No downloads needed for CI.
"""
from __future__ import annotations

from src.data.dataset_loader import build_mock_dataset

N_MOCK_ITEMS = 50


def get_mock_samples(n_items: int = N_MOCK_ITEMS):
    return build_mock_dataset(n_items=n_items, seed=7)
