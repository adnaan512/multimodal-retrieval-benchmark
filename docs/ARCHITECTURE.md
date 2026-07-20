# Module Architecture

This document describes the role, design, and data contracts of each module in the `multimodal-retrieval-benchmark` codebase. It is intended as a reference for contributors and reviewers.

---

## Overview

The pipeline is a strict directed acyclic graph (DAG) with no circular imports:

```
dataset_loader  ──►  clip_encoder  ──►  embedding_index
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                              text_to_image  image_to_text
                                    └──────────┼──────────┘
                                               │
                                        query_categorizer
                                               │
                                         evaluator
                                               │
                                       failure_analyzer
                                               │
                                   hard_negative_reranking
                                               │
                                       report_generator
```

All inter-module data is passed as typed dataclasses defined in `src/models.py`. No module imports from a "downstream" module — each layer only imports from `src/models.py` or its own upstream layer.

---

## Module Reference

### `src/models.py` — Core Dataclasses

**Role:** Single source of truth for all shared data structures. Every other module imports from here; nothing imports *into* `models.py`.

| Dataclass | Purpose |
|-----------|---------|
| `RetrievalResult` | One query's top-K ranked candidates, scores, and ground-truth indices |
| `QueryAnalysis` | Category label assigned to a single caption |
| `FailureCase` | A Recall@K miss with diagnostic context |
| `RecallMetrics` | Aggregate R@1/5/10 and median rank for one direction + backbone |

Key invariants:
- `RetrievalResult.retrieved_indices` and `.similarity_scores` are always the same length and ordered by descending score.
- `RecallMetrics.per_category` maps category name → `{"r@1": float, "n": int}`.

---

### `src/data/dataset_loader.py` — Dataset Loading

**Role:** Produce a list of `Sample` objects (image path + 5 captions) from Flickr30K or a synthetic mock.

| Function | Description |
|----------|-------------|
| `load_flickr30k_test_split(data_dir, split_size, seed)` | Parses the token file, deterministically selects `split_size` images |
| `build_mock_dataset(n_items, seed)` | Fully synthetic; all 4 query categories represented |

**Flexibility:** `_parse_token_file` handles multiple caption file formats (`results_20130124.token`, `captions.txt`, `captions.csv`) to support various Kaggle mirrors of Flickr30K.

**Test split protocol:** No official split file ships with the Kaggle mirror. We sort all image IDs, apply a seeded shuffle (`seed=42`), and take the first `split_size=1000`. This is deterministic and reproducible but *not* the Faghri et al. (2018) split.

---

### `src/backbone/clip_encoder.py` — Encoders

**Role:** Produce L2-normalised embedding vectors from images or text. All downstream code is encoder-agnostic.

| Class | Description |
|-------|-------------|
| `CLIPEncoder` | Real HuggingFace CLIP (ViT-B/32 or ViT-L/14), frozen weights, batched |
| `MockCLIPEncoder` | SHA-256 hash → seeded RNG → 512-dim vector; fully deterministic |

**Interface (both classes):**

```python
encode_image(identifier) -> np.ndarray  # shape (512,), L2-norm = 1.0
encode_text(text)        -> np.ndarray  # shape (512,), L2-norm = 1.0
encode_images_batch(ids) -> np.ndarray  # shape (N, 512)
encode_texts_batch(txts) -> np.ndarray  # shape (N, 512)
```

**Normalisation:** `_l2_normalize` divides by the L2 norm, clipping zero norms to `1e-12`. This makes dot-product retrieval mathematically identical to cosine similarity.

**CLIP weights are always frozen:** `requires_grad = False` is set on all parameters after loading. This is intentional — the benchmark measures zero-shot capability, not fine-tuned performance.

---

### `src/indexing/embedding_index.py` — Embedding Cache & Retrieval Primitives

**Role:** Encode the full dataset once and persist the result; provide low-level retrieval functions.

| Function | Description |
|----------|-------------|
| `load_or_build_indices(encoder, ...)` | Cache-aware entry point; returns `(image_matrix, text_matrix)` |
| `retrieve(query_embedding, index_matrix, k)` | Single-query top-K via `argsort` |
| `retrieve_batch(query_matrix, index_matrix, k)` | Vectorised; `(Q, D) @ (N, D).T` = `(Q, N)` scores |

**Cache naming:** The cache directory encodes the backbone name (e.g., `.cache/vit-b-32/`). Switching `--backbone` never silently reuses the wrong embeddings.

---

### `src/retrieval/text_to_image.py` and `image_to_text.py`

**Role:** Direction-specific wrappers around `retrieve_batch` that construct `RetrievalResult` objects with correct `ground_truth_indices`.

- **Text → Image:** Each caption's ground truth is the *one* image it was written for.
- **Image → Text:** Each image's ground truth is *any* of its 5 associated captions.

---

### `src/analysis/query_categorizer.py` — Rule-Based Caption Categorisation

**Role:** Assign each caption to exactly one of four semantic categories.

Priority order (highest specificity first):

1. `spatial_relation` — spatial preposition phrases (e.g., *"to the left of"*)
2. `attribute_centric` — colour/size/age adjectives (e.g., *"red"*, *"tall"*)
3. `scene_centric` — place words (e.g., *"beach"*, *"kitchen"*)
4. `object_centric` — subject-verb pattern, or default fallback

Spatial relations are checked first so that a sentence containing both a spatial phrase and a colour word is not miscategorised away from the most informative dimension.

---

### `src/analysis/failure_analyzer.py` — Failure Analysis

**Role:** Consume `RetrievalResult` lists and produce structured failure statistics.

| Function | Description |
|----------|-------------|
| `find_failures(results, categories, k)` | Returns `FailureCase` for every query whose ground truth is not in top-K |
| `failure_rate_by_category(results, categories, k)` | `failures / total` per category |
| `hardest_queries(results, categories, top_n)` | Ranked by ground-truth similarity (lowest = hardest) |
| `semantic_failure_summary(failure_rates)` | One-line finding: worst vs best category with multiplier |

---

### `src/benchmark/evaluator.py` — Metric Aggregation

**Role:** Compute standard IR metrics over lists of `RetrievalResult`.

| Function | Description |
|----------|-------------|
| `recall_at_k(results, k)` | Fraction of queries hitting ground truth in top-K |
| `median_rank(results)` | Median 1-indexed rank of first correct answer |
| `evaluate(results, direction, backbone, categories)` | Full `RecallMetrics` object |

---

### `src/retrieval/hard_negative_reranking.py` — Training-Free Reranking

**Role:** Post-process top-K candidate lists by penalising globally frequent wrong predictions.

**Algorithm:**
1. Count how often each candidate appears as an *incorrect* top-1 across the full query set.
2. Normalise counts by the maximum observed frequency.
3. For each query's top-K list, subtract `penalty_weight × normalised_frequency` from each candidate's score.
4. Re-sort by adjusted score.

This is O(Q·K) overhead after the O(Q·N) retrieval pass. No model weights are touched.

---

### `src/reporting/report_generator.py` — HTML Report

**Role:** Produce a self-contained dark-themed HTML file summarising a full run.

Sections generated:
- Summary cards (best R@1, R@5, R@10, total queries)
- Direction & backbone comparison table
- Per-category breakdown table
- Semantic finding callout box
- Top failure case cards
- Hard-negative reranking impact table
- Run configuration list

The output is a single `.html` file with all CSS inlined — no external assets or CDN calls required.

---

## Adding a New Module

1. Place it in the appropriate `src/` subdirectory.
2. Import only from `src/models.py` and upstream modules (never from downstream).
3. Accept and return typed dataclasses — no raw dicts between modules.
4. Add a `MockCLIPEncoder`-compatible test in `tests/` that works with no external downloads.
5. Document the new module in this file and update `README.md`.
