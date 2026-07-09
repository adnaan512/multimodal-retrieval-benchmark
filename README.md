# Multimodal Retrieval Benchmark: Zero-Shot CLIP Evaluation

This project implements a zero-shot cross-modal retrieval benchmark using OpenAI's pre-trained CLIP models (ViT-B/32 and ViT-L/14) on the Flickr30K dataset. 

Instead of simply reporting top-level accuracy, this project systematically breaks down **where and why retrieval fails** across different types of language (e.g., spatial relations vs. object-centric queries) and implements a training-free reranking algorithm to improve performance.

## Key Features
- **End-to-End Pipeline**: Handles data loading, embedding generation, caching, cosine similarity indexing, and evaluation.
- **Deep Failure Analysis**: Automatically categorizes text queries into four semantic buckets (`object_centric`, `scene_centric`, `attribute_centric`, `spatial_relation`) to identify the models' linguistic blind spots.
- **Hard-Negative Reranking**: A custom, training-free algorithm that penalizes frequently mis-retrieved "hard negatives" to boost accuracy.
- **Automated Reporting**: Generates a comprehensive, dark-themed HTML dashboard with metric tables and visual failure examples.

---

##  Benchmark Results (Flickr30K)
*Computed across 1,000 images and 5,000 captions on Kaggle (T4 GPU).*

### Recall Metrics & Backbone Comparison
| Direction | Backbone | N | R@1 | R@5 | R@10 | Median Rank |
|-----------|----------|---|-----|-----|------|-------------|
| Text $\rightarrow$ Image | ViT-B/32 | 5000 | 59.36% | 84.20% | 90.76% | 1.0 |
| Image $\rightarrow$ Text | ViT-B/32 | 1000 | 80.50% | 94.90% | 97.40% | 1.0 |
| Text $\rightarrow$ Image | **ViT-L/14** | 5000 | **66.28%** | **88.22%** | **93.08%** | 1.0 |
| Image $\rightarrow$ Text | **ViT-L/14** | 1000 | **87.10%** | **97.90%** | **99.00%** | 1.0 |

### Query Category Breakdown
*Which kinds of captions confuse the zero-shot model the most?*

| Query Category | N | Failure Rate |
|----------------|---|--------------|
| **scene_centric** (high failure) | 377 | 41.9% |
| object_centric | 2387 | 36.9% |
| attribute_centric | 1754 | 29.3% |
| spatial_relation | 482 | 27.8% |

**Scientific Finding:** `scene_centric` queries fail the most often (41.9%), occurring at 1.5x the rate of `spatial_relation` queries (27.8%). This highlights that highly overlapping global visual contexts (e.g., "people in a park") are much harder for contrastive bag-of-words models to differentiate than localized relational attributes.

### Hard Negative Reranking Impact
Applying our training-free hard-negative penalty reranking dramatically improves retrieval without any fine-tuning:

| Stage | Recall@1 (Text $\rightarrow$ Image) |
|-------|------------------------------------|
| Before Reranking | 66.28% |
| **After Hard-Negative Reranking** | **71.94%** |

---

##  Quick Start

```bash
git clone https://github.com/adnaan512/multimodal-retrieval-benchmark.git
cd multimodal-retrieval-benchmark
pip install -r requirements.txt
```

**Option 1  Kaggle (Recommended for full dataset processing):**
* Run directly in your browser with a free GPU: [Kaggle Notebook](https://www.kaggle.com/code/adnanhassnain/multimodal-retrieval-benchmark)

**Option 2 Local CPU (Standard 1K split):**
```bash
python main.py --mode local --data-dir ./flickr30k
```

**Option 3  Offline Demo (Mock data for testing):**
```bash
python examples/run_demo.py
```

---

## Architecture & Design Decisions

```
                 ┌─────────────────────┐
                 │   Flickr30K Dataset  │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ CLIP Encoder (frozen)│
                 │ ViT-B/32 or ViT-L/14 │
                 └──────┬──────────┬─────┘
           image branch │          │ text branch
                 ┌──────▼──────┐  ┌──────▼──────┐
                 │ image index │  │ text index  │
                 │ (L2-norm)   │  │ (L2-norm)   │
                 └──────┬──────┘  └──────┬──────┘
                        │ cosine similarity
        ┌───────────────▼─────────────────▼───────────────┐
        │                 Top-K Retrieval                 │
        └───────────────────────┬─────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────┐
        │  Query Categorizer + Evaluator (Recall Metrics) │
        └───────────────────────┬─────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────┐
        │        Hard Negative Reranking Algorithm        │
        └───────────────────────┬─────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────┐
        │             Interactive HTML Report             │
        └─────────────────────────────────────────────────┘
```

1. **Why Flickr30K over MS-COCO?** Flickr30K is smaller, has shorter, visually-grounded captions, and is widely used for zero-shot benchmarks, making it practical for a single GPU environment.
2. **L2-Normalization**: Embeddings are L2-normalized before retrieval. This makes dot-product mathematically identical to cosine similarity, enabling lightning-fast matrix multiplications without extra math at query time.
3. **No Fine-Tuning**: The goal is to evaluate the *zero-shot* capabilities of the pretrained embedding space. Fine-tuning would mask the model's inherent linguistic blind spots.

---

##  Project Structure

```
multimodal-retrieval-benchmark/
├── main.py                              # CLI entry point
├── src/
│   ├── data/dataset_loader.py           # Flickr30K loading & mock data
│   ├── backbone/clip_encoder.py         # HuggingFace CLIP & embedding generation
│   ├── indexing/embedding_index.py      # Caching and similarity matrices
│   ├── retrieval/                       # Cross-modal search & reranking logic
│   ├── analysis/                        # Query categorization & failure analytics
│   ├── benchmark/evaluator.py           # Recall metrics computation
│   └── reporting/report_generator.py    # HTML dashboard generation
├── tests/                               # Pytest suite
└── notebooks/flickr30k_benchmark.ipynb  # Kaggle-ready notebook
```

## Author
Adnan Hassnain  BS CS, NUST Pakistan
