# multimodal-retrieval-benchmark

Zero-shot cross-modal retrieval with pretrained CLIP: retrieve images from
text queries and text from image queries, with systematic analysis of
*where and why* retrieval fails across different query types.

## Abstract

CLIP (Radford et al., 2021) is trained purely with a contrastive
image-text objective and never sees a retrieval label, yet the resulting
shared embedding space supports competitive zero-shot retrieval out of the
box. On MS-COCO, published zero-shot CLIP (ViT-B backbone) results report
roughly 58% Recall@1 for image-to-text and 38% Recall@1 for text-to-image
retrieval on the 5K test set — clearly behind supervised, retrieval-trained
systems, but remarkable for a model with no retrieval-specific training at
all. Flickr30K, with its shorter, more focused captions, is reported to be
considerably easier for the same zero-shot approach: published figures put
CLIP zero-shot image-to-text Recall@1 around 88% and text-to-image Recall@1
around 69% on the standard 1K test split. This project reproduces that
protocol end-to-end (`main.py --mode local`) and goes further: it breaks
retrieval performance down by **query category** (object-centric,
scene-centric, attribute-centric, spatial-relation) to test whether CLIP's
zero-shot gap is uniform or concentrated in specific kinds of language, and
tests whether a training-free hard-negative reranking step can close part
of that gap.

## Research Questions

| ID | Question |
|---|---|
| RQ1 | Does zero-shot CLIP retrieval achieve competitive Recall@K against supervised baselines on Flickr30K? |
| RQ2 | Which query categories (object-centric, scene-centric, attribute-centric, spatial-relation) show the largest retrieval gaps? |
| RQ3 | Do retrieval failures cluster by semantic category — and can hard negative mining improve Recall@1 without any fine-tuning? |

## Benchmark Results (Flickr30K)

*These results were computed across 1,000 images and 5,000 captions on Kaggle (T4 GPU).*

### Recall Metrics & Backbone Comparison (RQ1)

| Direction | Backbone | N | R@1 | R@5 | R@10 | Median Rank |
|-----------|----------|---|-----|-----|------|-------------|
| Text $\rightarrow$ Image | ViT-B/32 | 5000 | 59.36% | 84.20% | 90.76% | 1.0 |
| Image $\rightarrow$ Text | ViT-B/32 | 1000 | 80.50% | 94.90% | 97.40% | 1.0 |
| Text $\rightarrow$ Image | **ViT-L/14** | 5000 | **66.28%** | **88.22%** | **93.08%** | 1.0 |
| Image $\rightarrow$ Text | **ViT-L/14** | 1000 | **87.10%** | **97.90%** | **99.00%** | 1.0 |

### Query Category Breakdown (RQ2)

*Which kinds of captions confuse the zero-shot model the most?*

| Query Category | N | Failure Rate |
|----------------|---|--------------|
| **scene_centric** (high failure) | 377 | 41.9% |
| object_centric | 2387 | 36.9% |
| attribute_centric | 1754 | 29.3% |
| spatial_relation | 482 | 27.8% |

**Key Finding:** `scene_centric` queries fail the most often (41.9%), occurring at 1.5x the rate of `spatial_relation` queries (27.8%). This highlights that highly overlapping global visual contexts (e.g., "people in a park") are much harder for contrastive bag-of-words models to differentiate than localized relational attributes.

### Hard Negative Reranking Impact (RQ3)

Applying our training-free hard-negative penalty reranking dramatically improves retrieval:

| Stage | Recall@1 (Text $\rightarrow$ Image) |
|-------|------------------------------------|
| Before Reranking | 66.28% |
| **After Hard-Negative Reranking** | **71.94%** |
## Architecture

```
                 ┌─────────────────────┐
                 │   Flickr30K / mock   │
                 │  images + captions   │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │   CLIP encoder (frozen)│
                 │  ViT-B/32 or ViT-L/14 │
                 └──────┬──────────┬─────┘
                         │          │
             image branch│          │text branch
                         ▼          ▼
              ┌──────────────┐  ┌──────────────┐
              │ image index   │  │ text index    │
              │ (N, 512)      │  │ (N*5, 512)    │
              │ L2-normalized │  │ L2-normalized │
              └──────┬────────┘  └───────┬───────┘
                     │  cosine similarity  │
                     │   (dot product)     │
        ┌────────────▼──────┐   ┌─────────▼────────────┐
        │ image → text top-K │   │ text → image top-K    │
        └────────────┬───────┘   └─────────┬─────────────┘
                     │                      │
                     ▼                      ▼
              ┌─────────────────────────────────┐
              │  query_categorizer + evaluator    │
              │  Recall@1/5/10, median rank,       │
              │  per-category breakdown            │
              └──────────────┬──────────────────────┘
                              │
                     ┌────────▼─────────┐
                     │ hard_negative_    │
                     │ reranking (RQ3)   │
                     └────────┬──────────┘
                              │
                     ┌────────▼─────────┐
                     │  dark HTML report  │
                     └───────────────────┘
```

## Quick Start

```bash
git clone https://github.com/adnaan512/multimodal-retrieval-benchmark.git
cd multimodal-retrieval-benchmark
pip install -r requirements.txt   # only needed for real CLIP / --mode local

# Option 1 — Demo (no downloads, ~30 seconds, 50 synthetic image-caption pairs):
python examples/run_demo.py

# Option 2 — Flickr30K test split (1000 images, local PC, CPU):
python main.py --mode local --data-dir ./flickr30k

# Option 3 — Full Flickr30K, both backbones (Kaggle recommended, T4 GPU):
# Run directly in browser: https://www.kaggle.com/code/adnanhassnain/multimodal-retrieval-benchmark
```

`main.py` also supports `--backbone {vit-b-32,vit-l-14}`,
`--compare-backbones`, `--k`, `--cache-dir`, `--output`, and `--dry-run` (validate
configuration without computing anything). Embeddings are cached to
`.cache/<backbone>/*.npy` and are never recomputed once built
(`load_or_build_indices`).

## Environment Notes

| | Kaggle (Option A, recommended for full dataset) | Local PC (Option B) |
|---|---|---|
| RAM | 30GB free tier | 16GB+ recommended |
| GPU | Optional T4 | None required (CPU) |
| Dataset | Full Flickr30K, 31,783 images (~4GB) | Standard 1,000-image test split only |
| CLIP inference speed | ~2ms/image on T4 (63K images in ~2 min) | ~150ms/image on CPU (25 min for 1,000 images) |
| Notebook | `notebooks/flickr30k_benchmark.ipynb` | `python main.py --mode local --data-dir ./flickr30k` |

## Main Results

### Reference: published zero-shot CLIP figures (literature)

These are the zero-shot CLIP numbers reported in prior work on the standard
1K Flickr30K test split, included here as the baseline this project's
protocol is designed to reproduce — run `python main.py --mode local
--compare-backbones` on the real dataset to populate the table below with
your own measured numbers.

| Backbone | Direction | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| CLIP (reported, literature) | Image → Text | ~88.0% | ~98.7% | ~99.4% |
| CLIP (reported, literature) | Text → Image | ~68.7% | ~90.6% | ~95.2% |

### This run (`python examples/run_demo.py`, 50-item mock dataset, MockCLIPEncoder)

The mock encoder produces *random but deterministic* embeddings (no
learned semantics), so these numbers are expected to be near chance —
they exist purely to validate that the full pipeline (indexing, both
retrieval directions, evaluation, failure analysis, reranking, reporting)
runs correctly end-to-end without downloads:

| Backbone | Direction | N | R@1 | R@5 | R@10 | Median Rank |
|---|---|---|---|---|---|---|
| mock | Text → Image | 250 | 2.4% | 10.4% | 20.8% | 5.5 |
| mock | Image → Text | 50 | 2.0% | 16.0% | 20.0% | 4.0 |

Run against real Flickr30K data with a real CLIP backbone to populate this
table with meaningful zero-shot numbers (RQ1).

## Query Category Breakdown (RQ2)

`src/analysis/query_categorizer.py` assigns each caption to exactly one of
four categories (spatial_relation checked first, since it's the category
we most want to isolate; see `docs/RESEARCH.md` for the full priority
order and rationale). `src/analysis/failure_analyzer.py` then computes a
failure rate per category from actual retrieval results — nothing here is
hard-coded, the report always reflects the measured rates from whichever
run produced it.

**Spatial relations are the category this project's hypothesis singles
out as hardest, and the finding is reported honestly whichever way the
numbers land.** The reasoning: "the cat is to the left of the dog" and
"the dog is to the left of the cat" share nearly identical bag-of-words
content but describe completely different images. CLIP's contrastive,
sentence-level training objective gives it no explicit signal to encode
word order as spatial layout, so its text encoder is expected to struggle
to separate such pairs. On the full Flickr30K test split, expect
`spatial_relation` failure rate to be the highest of the four categories
by a noticeable margin. (On the tiny 50-item random mock dataset used for
CI/demo, category sample sizes are too small and the mock encoder has no
real semantics, so this pattern is not expected to appear there — see the
sample run above.)

## Top Failure Examples

Each HTML report (`report.html` / `examples/demo_report.html`) includes
the 5 hardest queries — the ones whose ground-truth match has the lowest
similarity score among retrieved candidates (or never appears in the
retrieved list at all) — printed with the query text, the incorrect
top-1 result, and the true ground-truth index(es), so failures can be
inspected individually rather than only summarized as a rate.

## Hard Negative Reranking Impact (RQ3)

`src/retrieval/hard_negative_reranking.py` implements a fully
training-free reranking step: it counts how often each candidate shows up
as an *incorrect* top-1 prediction across the whole query set, then
penalizes high-frequency "hard negative" candidates in every query's
top-K list and re-sorts. No CLIP weight is ever updated.

| Stage | R@1 (this demo run) |
|---|---|
| Before reranking | 2.40% |
| After hard-negative reranking | 3.60% |

The relative direction of this effect (reranking helps vs. hurts, and by
how much) should be re-checked on the real dataset — the demo run above
uses the mock encoder and exists only to prove the reranking code path
executes correctly.

## Key Design Decisions

**Decision 1 — Why Flickr30K over MS-COCO.** Flickr30K is smaller
(31,783 vs ~123,000 images), has shorter and more visually-grounded
captions, and is a standard, widely-used retrieval benchmark with a
well-established zero-shot CLIP reference point — making it a practical
choice for a project meant to run on a single Kaggle notebook or a local
PC rather than requiring a large-scale cluster.

**Decision 2 — Why L2-normalize embeddings before the dot product.** For
L2-normalized vectors, the dot product is mathematically identical to
cosine similarity, and normalizing removes any dependence on embedding
magnitude (which can vary across images/captions and carries no semantic
meaning in CLIP's space). This lets `retrieve()` stay a single matrix
multiply with no extra normalization step at query time.

**Decision 3 — Why not fine-tune CLIP.** The research questions here are
specifically about *zero-shot* capability — what the pretrained embedding
space already supports without any task-specific adaptation. Fine-tuning
would answer a different question (how much retrieval-specific training
improves things) and would make it impossible to attribute the observed
failure patterns to CLIP's pretraining objective itself.

## Limitations

- English captions only — no multilingual evaluation.
- Static images only — no video.
- No audio modality.
- Query categorization is rule-based (regex/keyword matching), not a
  learned classifier, and captions are assigned exactly one category even
  when multiple apply (see priority order in `docs/RESEARCH.md`).
- The 1,000-image Flickr30K test split used here is not an officially
  distributed split file (the Kaggle mirror does not ship one) — it is a
  deterministic seeded selection over sorted image IDs, so exact numbers
  may differ slightly from papers using the original Karpathy split.

## Project Structure

```
multimodal-retrieval-benchmark/
├── main.py                              CLI entry point
├── src/
│   ├── models.py                        Shared dataclasses
│   ├── data/dataset_loader.py           Flickr30K + mock dataset loaders
│   ├── backbone/clip_encoder.py         CLIP ViT-B/32, ViT-L/14, MockCLIPEncoder
│   ├── indexing/embedding_index.py      Build/cache/retrieve embedding matrices
│   ├── retrieval/
│   │   ├── text_to_image.py             Text → image queries
│   │   ├── image_to_text.py             Image → text queries
│   │   └── hard_negative_reranking.py   Training-free reranking (RQ3)
│   ├── analysis/
│   │   ├── query_categorizer.py         Rule-based caption categorization (RQ2)
│   │   └── failure_analyzer.py          Failure clustering, hardest queries
│   ├── benchmark/evaluator.py           Recall@K, median rank, backbone comparison
│   └── reporting/report_generator.py    Dark HTML report
├── tests/                               Unit tests + mock fixtures
├── examples/run_demo.py                 50-item offline demo
├── docs/RESEARCH.md                     Full methodology
├── notebooks/flickr30k_benchmark.ipynb  Full-dataset Kaggle run
├── .github/workflows/ci.yml             CI: mock mode, Python 3.10/3.11
├── requirements.txt / requirements-dev.txt
└── CITATION.cff
```

## Installation

```bash
pip install -r requirements.txt        # torch, transformers, Pillow, numpy, faiss-cpu
pip install -r requirements-dev.txt    # + pytest, flake8, pytest-cov
```

The demo (`examples/run_demo.py`) and the unit tests only need `numpy` —
they use `MockCLIPEncoder` and never import `torch`/`transformers`.

## Testing

```bash
pytest tests/ -v --cov=src
```

Covers: L2-normalization correctness, cosine similarity range `[-1, 1]`,
mock-encoder reproducibility, Recall@1 = 1.0 when ground truth is ranked
first, and Recall@K monotonically increasing in K.

## References

- Radford, A. et al. (2021). *Learning Transferable Visual Models From
  Natural Language Supervision.* ICML.
- Young, P. et al. (2014). *From Image Descriptions to Visual
  Denotations: New Similarity Metrics for Semantic Inference over Event
  Descriptions.* TACL.
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity
  search with GPUs.* IEEE Transactions on Big Data.
- Karpathy, A., & Fei-Fei, L. (2015). *Deep Visual-Semantic Alignments
  for Generating Image Descriptions.* CVPR.

## Author

Adnan Hassnain — BS CS, NUST Pakistan

## License

MIT
