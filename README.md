<div align="center">

# Multimodal Retrieval Benchmark
### Zero-Shot Cross-Modal Retrieval with CLIP on Flickr30K

[![CI](https://github.com/adnaan512/multimodal-retrieval-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/adnaan512/multimodal-retrieval-benchmark/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/adnanhassnain/multimodal-retrieval-benchmark)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## Abstract

We present a systematic zero-shot cross-modal retrieval benchmark evaluating OpenAI's CLIP models (ViT-B/32 and ViT-L/14) on the Flickr30K dataset. Beyond top-level accuracy, this work investigates *where and why* retrieval fails by categorising text queries into four semantic buckets — **object-centric**, **scene-centric**, **attribute-centric**, and **spatial-relation** — and measuring per-category failure rates. We find that scene-centric queries fail at **41.9%**, representing a 1.5× higher failure rate than spatial-relation queries (27.8%), contrary to the intuition that spatial relations should be hardest. We further propose a **training-free hard-negative reranking** algorithm that lifts Recall@1 from 66.28% to **71.94%** (+5.66 pp) without modifying CLIP's weights.

---

## Research Questions

| # | Research Question | Key Finding |
|---|---|---|
| **RQ1** | How does CLIP backbone capacity (ViT-B/32 vs ViT-L/14) affect zero-shot retrieval? | ViT-L/14 improves T→I Recall@1 by **+6.9 pp** (59.36% → 66.28%) |
| **RQ2** | Which linguistic query categories expose CLIP's blind spots? | **Scene-centric** queries fail most (41.9%), not spatial-relation (27.8%) |
| **RQ3** | Can a training-free algorithm improve upon raw CLIP similarity? | Hard-negative reranking yields **+5.66 pp** Recall@1 with zero fine-tuning |

---

## Results

*Computed across 1,000 images / 5,000 captions on Kaggle (T4 GPU). Seed = 42.*

### Backbone & Direction Comparison

| Direction | Backbone | N | R@1 | R@5 | R@10 | Median Rank |
|-----------|----------|--:|----:|----:|-----:|:-----------:|
| Text → Image | ViT-B/32 | 5 000 | 59.36% | 84.20% | 90.76% | 1.0 |
| Image → Text | ViT-B/32 | 1 000 | 80.50% | 94.90% | 97.40% | 1.0 |
| Text → Image | **ViT-L/14** | 5 000 | **66.28%** | **88.22%** | **93.08%** | 1.0 |
| Image → Text | **ViT-L/14** | 1 000 | **87.10%** | **97.90%** | **99.00%** | 1.0 |

### Query-Category Breakdown (RQ2)

*Which caption categories confuse the zero-shot model the most?*

| Query Category | N | Failure Rate |
|----------------|--:|-------------:|
| **scene_centric** | 377 | **41.9%** *(highest)* |
| object_centric | 2 387 | 36.9% |
| attribute_centric | 1 754 | 29.3% |
| spatial_relation | 482 | 27.8% |

> **Finding:** Scene-centric queries (e.g., *"people gathering in a crowded park"*) fail 1.5× more often than spatial-relation queries. Highly overlapping global visual contexts — multiple images sharing similar scenes — are harder for a bag-of-words contrastive model to distinguish than localised relational descriptions. This contradicts the common assumption that spatial language is CLIP's primary weakness.

### Hard-Negative Reranking Impact (RQ3)

| Stage | Recall@1 (Text → Image) | Δ |
|-------|------------------------:|:-:|
| Baseline CLIP | 66.28% | — |
| **+ Hard-Negative Reranking** | **71.94%** | **+5.66 pp** |

---

## Quick Start

```bash
git clone https://github.com/adnaan512/multimodal-retrieval-benchmark.git
cd multimodal-retrieval-benchmark
pip install -e .                   # installs as an editable package
# or: pip install -r requirements.txt
```

**Option A — Kaggle (Recommended for full GPU run):**
> Launch the notebook directly in your browser (free T4 GPU):
> [kaggle.com/code/adnanhassnain/multimodal-retrieval-benchmark](https://www.kaggle.com/code/adnanhassnain/multimodal-retrieval-benchmark)

**Option B — Local CPU (1 000-image split, real Flickr30K data):**
```bash
python main.py --mode local --data-dir ./flickr30k
python main.py --mode local --data-dir ./flickr30k --backbone vit-l-14
python main.py --mode local --data-dir ./flickr30k --compare-backbones
```

**Option C — Offline Demo (synthetic mock data, no downloads needed):**
```bash
python examples/run_demo.py        # runs full pipeline; writes examples/demo_report.html
```

**Option D — Dry run (validate config without computing):**
```bash
python main.py --dry-run --mode local --data-dir ./flickr30k --backbone vit-l-14
```

---

## Architecture

```
                 ┌─────────────────────┐
                 │   Flickr30K Dataset  │
                 │  (1 000-image split) │
                 └──────────┬───────────┘
                            │ image paths + captions
                 ┌──────────▼───────────┐
                 │  CLIP Encoder (frozen)│  ← No fine-tuning (see §Design)
                 │  ViT-B/32 | ViT-L/14 │
                 └──────┬──────────┬─────┘
          image branch  │          │  text branch
               ┌────────▼──────┐  ┌──────▼────────┐
               │  Image Index  │  │  Text Index   │
               │  (N × 512)    │  │  (N×5 × 512)  │
               │  L2-normalised│  │  L2-normalised│
               └────────┬──────┘  └──────┬────────┘
                        │   cosine similarity
        ┌───────────────▼─────────────────▼──────────────┐
        │              Top-K Retrieval                    │
        │  (matrix multiply = dot product on L2 vectors)  │
        └───────────────────────┬─────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────┐
        │  Query Categorizer + Evaluator (Recall@K)       │
        │  4 categories: object / scene / attribute /     │
        │  spatial — checked in priority order            │
        └───────────────────────┬─────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────┐
        │     Hard-Negative Reranking (training-free)     │
        │  Penalises globally-frequent wrong predictions  │
        └───────────────────────┬─────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────┐
        │         Self-Contained HTML Report              │
        │  Metrics · Category breakdown · Failure cases   │
        └─────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Dataset** | Flickr30K over MS-COCO | Shorter, more visually-grounded captions; smaller scale makes a single-GPU run feasible; widely used zero-shot benchmark |
| **Normalization** | L2-normalize all embeddings | Converts dot product to cosine similarity — retrieval becomes a single matrix multiply with no extra arithmetic at query time |
| **No fine-tuning** | Frozen CLIP weights throughout | Measures the *inherent* zero-shot capability of the pretrained embedding space; fine-tuning would mask the linguistic blind spots we aim to characterise |
| **Embedding cache** | `.npy` files keyed by backbone | Avoids re-encoding 31 K images across runs; cache directory encodes backbone name to prevent silent reuse of wrong embeddings |
| **Reranking** | Frequency-based hard-negative penalty | Training-free, interpretable, and O(Q·K) overhead on top of already-computed scores |

---

## Project Structure

```
multimodal-retrieval-benchmark/
├── main.py                              # CLI entry point (--mode, --backbone, --compare-backbones)
├── pyproject.toml                       # Installable package config (pip install -e .)
├── requirements.txt                     # Runtime dependencies
├── requirements-dev.txt                 # Dev/lint/test dependencies
├── CITATION.cff                         # Machine-readable citation metadata
├── CHANGELOG.md                         # Version history
├── CONTRIBUTING.md                      # Contribution guide
├── src/
│   ├── models.py                        # Core dataclasses: RetrievalResult, RecallMetrics, …
│   ├── data/dataset_loader.py           # Flickr30K loader + synthetic mock dataset
│   ├── backbone/clip_encoder.py         # CLIPEncoder (HuggingFace) + MockCLIPEncoder
│   ├── indexing/embedding_index.py      # Caching, L2-norm matrices, retrieve_batch()
│   ├── retrieval/
│   │   ├── text_to_image.py             # Caption → Image search
│   │   ├── image_to_text.py             # Image → Caption search
│   │   └── hard_negative_reranking.py   # Training-free reranking (RQ3)
│   ├── analysis/
│   │   ├── query_categorizer.py         # Rule-based caption → category assignment
│   │   └── failure_analyzer.py          # Failure rate by category, hardest queries
│   ├── benchmark/evaluator.py           # Recall@K, median rank, per-category aggregation
│   └── reporting/report_generator.py    # Self-contained HTML dashboard
├── tests/                               # Pytest suite (mock mode, no downloads)
│   ├── test_clip_encoder.py
│   └── test_retrieval_metrics.py
├── notebooks/
│   └── flickr30k_benchmark.ipynb        # Kaggle-ready notebook (T4 GPU)
├── examples/
│   └── run_demo.py                      # Offline end-to-end demo
└── docs/
    ├── RESEARCH.md                      # Full methodology & statistical notes
    └── ARCHITECTURE.md                  # Module-level design documentation
```

---

## Reproducibility

All stochastic steps use fixed seeds. To reproduce published results exactly:

```bash
# 1. Obtain Flickr30K from Kaggle (requires free account):
#    https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
#    Extract to ./flickr30k/

# 2. Run with both backbones (replicates the full results table):
python main.py \
    --mode local \
    --data-dir ./flickr30k \
    --compare-backbones \
    --k 10 \
    --output results/report.html

# Seeds: dataset split seed=42 (fixed in dataset_loader.py), mock data seed=7
# Hardware: Kaggle T4 GPU (16 GB VRAM), Python 3.10, PyTorch 2.1, transformers 4.36
```

**Embedding cache:** Once generated, embeddings are stored as `.npy` files under `.cache/` and reused automatically on subsequent runs. Delete `.cache/` to force a full re-encode.

---

## Limitations

- **Single dataset:** Conclusions are specific to Flickr30K. Flickr30K captions are shorter and more object-focused than MS-COCO's; failure rates may differ on other benchmarks.
- **1 000-image test split:** The Kaggle mirror of Flickr30K does not ship an official train/val/test split; we deterministically select 1 000 images with a seeded shuffle. Results may differ slightly from papers using the official Faghri et al. (2018) split.
- **Rule-based categoriser:** The four query categories are assigned by keyword matching, not NLP parsing. Captions with ambiguous phrasing may be miscategorised.
- **Hard-negative reranking:** The frequency-based penalty is computed on the *test set itself* (no held-out validation set), which introduces mild in-distribution optimism. The improvement (+5.66 pp) should be treated as a proof-of-concept rather than a validated generalisation bound.
- **No ablation of penalty weight:** `DEFAULT_PENALTY_WEIGHT = 0.15` was not tuned via cross-validation.

---

## Related Work

| Work | Contribution | Relation to This Project |
|------|-------------|--------------------------|
| Radford et al. (2021) — CLIP | Contrastive Language-Image Pre-Training on 400M pairs | Backbone used for all embeddings |
| Young et al. (2014) — Flickr30K | Dataset & visual denotation metrics | Evaluation dataset |
| Faghri et al. (2018) — VSE++ | Hard-negative mining during *training* | Inspiration for test-time hard-negative reranking |
| Johnson et al. (2019) — FAISS | Billion-scale GPU similarity search | Algorithmic context for embedding indexing |
| Lin et al. (2014) — MS-COCO | Large-scale detection & captioning benchmark | Alternative dataset discussed in limitations |

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@software{hassnain2026multimodal,
  author    = {Hassnain, Adnan},
  title     = {Multimodal Retrieval Benchmark: Zero-Shot Cross-Modal Retrieval with CLIP},
  year      = {2026},
  url       = {https://github.com/adnaan512/multimodal-retrieval-benchmark},
  license   = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for a machine-readable version compatible with GitHub's *Cite this repository* button.

---

## Author & Acknowledgements

**Adnan Hassnain** — BS Computer Science, NUST Pakistan
([GitHub](https://github.com/adnaan512))

The Flickr30K dataset is provided by Young et al. (2014). CLIP model weights are provided by OpenAI via Hugging Face. Compute resources provided by Kaggle's free GPU quota.

---

## License

This project is released under the [MIT License](LICENSE). The Flickr30K dataset and CLIP model weights are subject to their respective licences; please consult their original sources before redistribution.
