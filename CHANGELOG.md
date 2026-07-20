# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-07-09

### Added

#### Core Pipeline
- End-to-end zero-shot cross-modal retrieval pipeline for Flickr30K
- `CLIPEncoder` wrapping HuggingFace `transformers` for ViT-B/32 and ViT-L/14 backbones
- `MockCLIPEncoder` for deterministic, download-free testing and CI
- L2-normalised embedding matrices with `.npy` disk caching (keyed by backbone)
- Vectorised `retrieve_batch()` using matrix multiplication (dot product ≡ cosine similarity)

#### Evaluation (RQ1)
- `recall_at_k()` and `median_rank()` for standard IR metrics
- Backbone comparison mode (`--compare-backbones`) running ViT-B/32 and ViT-L/14 under identical protocol
- Bidirectional evaluation: Text→Image and Image→Text

#### Query Analysis (RQ2)
- Rule-based `categorize_captions()` assigning each caption to one of four categories:
  `spatial_relation` · `attribute_centric` · `scene_centric` · `object_centric`
- `failure_rate_by_category()` computing per-category Recall@1 failure rates
- `hardest_queries()` ranking all queries by ground-truth similarity score
- `semantic_failure_summary()` producing a one-line, data-driven finding string

#### Reranking (RQ3)
- `compute_hard_negative_frequencies()` building a frequency table of globally wrong top-1 predictions
- `rerank_with_hard_negatives()` applying a normalised penalty and re-sorting candidates

#### Reporting
- Self-contained dark-themed HTML dashboard via `generate_report()`
- Metric summary cards, direction/backbone table, per-category breakdown, failure examples

#### Infrastructure
- CLI entry point (`main.py`) with `--mode`, `--backbone`, `--compare-backbones`, `--k`, `--dry-run`
- GitHub Actions CI testing Python 3.10 and 3.11 in mock mode (no ML dependencies)
- Pytest suite covering encoder normalization, cosine similarity range, recall math, median rank
- Offline demo (`examples/run_demo.py`) exercising the full pipeline with synthetic data
- Kaggle notebook (`notebooks/flickr30k_benchmark.ipynb`) for T4 GPU execution
- `CITATION.cff` with machine-readable citation metadata
- `docs/RESEARCH.md` with full methodology, split protocol, and design decisions

[1.0.0]: https://github.com/adnaan512/multimodal-retrieval-benchmark/releases/tag/v1.0.0
