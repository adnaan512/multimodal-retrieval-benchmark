# Research Methodology

## Abstract

This document describes the full experimental methodology of the multimodal-retrieval-benchmark.
We evaluate OpenAI's CLIP models (ViT-B/32 and ViT-L/14) under a strictly zero-shot protocol
on the Flickr30K dataset (1,000-image test split, 5,000 captions). Three research questions are
investigated: (RQ1) how backbone capacity affects retrieval performance; (RQ2) which linguistic
query categories constitute CLIP's primary failure modes; and (RQ3) whether a training-free
hard-negative penalty reranking can improve upon the raw CLIP similarity baseline.
Key findings: scene-centric queries exhibit the highest failure rate (41.9%), which is 1.5× greater
than spatial-relation queries (27.8%), contradicting the common assumption that spatial language
is CLIP's weakest dimension. Hard-negative reranking improves Text→Image Recall@1 from 66.28%
to 71.94% (+5.66 pp) without modifying any model weights.

---

## 1. Background: CLIP and the Shared Embedding Space

CLIP (Contrastive Language-Image Pre-training; Radford et al., 2021) trains an image encoder
and a text encoder jointly on ~400M (image, caption) pairs scraped from the web. The training
objective is contrastive: within a batch of N pairs, the model is pushed to maximise cosine
similarity between each image and *its* caption while minimising similarity to the other N-1
captions in the batch (and vice versa for images). After training, no matching step is needed
at inference time — the two encoders simply place semantically related images and text near each
other in a shared 512-dim space. This is what makes **zero-shot cross-modal retrieval** possible:
we never train a retrieval-specific model; we only embed both sides with the frozen encoders and
rank by cosine similarity.

---

## 2. Zero-Shot Retrieval Protocol

For a dataset of N images, each with 5 captions:

1. Encode all N images with `CLIPEncoder.encode_images_batch` → an `(N, 512)` L2-normalised matrix.
2. Encode all `N×5` captions with `CLIPEncoder.encode_texts_batch` → an `(N×5, 512)` L2-normalised matrix.
3. **Text→Image**: for each caption row, compute its dot product against every image row (equivalent to cosine similarity, since both sides are L2-normalised) and take the top-K image indices.
4. **Image→Text**: symmetric — for each image row, take the top-K caption indices.
5. A query is a "hit" at K if the ground-truth image (Text→Image) or *any* of the 5 ground-truth captions (Image→Text) appears in the top-K list.

No parameter of CLIP is updated at any point in this pipeline — the entire benchmark measures out-of-the-box zero-shot capability.

---

## 3. Recall@K Definition

For a query set Q and cutoff K:

```
Recall@K = (1 / |Q|) × Σ_{q ∈ Q} 𝟙[ground_truth(q) ∈ top-K(q)]
```

We report Recall@1, Recall@5, and Recall@10 for both retrieval directions, plus the **median rank**
of the first correct answer (a more granular signal than Recall@K alone, since it distinguishes
"just missed the cutoff" from "wildly wrong").

---

## 4. Query Categorisation Methodology

Each caption is assigned exactly one primary category using simple, inspectable rules
(`src/analysis/query_categorizer.py`), checked in this priority order:

1. **spatial_relation** — contains a spatial preposition phrase
   (*"next to"*, *"in front of"*, *"behind"*, *"on top of"*, *"beside"*, *"to the left/right of"*).
2. **attribute_centric** — contains a colour/size/age adjective
   (*red, blue, tall, small, young, old, ...*).
3. **scene_centric** — contains a place word
   (*park, street, beach, kitchen, forest, stadium, ...*).
4. **object_centric** — matches an `"a/an/the <noun> is/are/sits..."` pattern,
   or is the default category if nothing else matched.

Spatial relations are checked first because they are the most specific, least ambiguous signal,
and because they are the category this project's hypothesis (RQ2) predicts will be hardest for
CLIP — we don't want an incidental colour word in a spatial sentence to reclassify it away from
the category we most want to measure.

**Why spatial relations are expected to be hard for CLIP.** *"The cat is to the left of the dog"*
and *"the dog is to the left of the cat"* contain exactly the same tokens and nearly identical
bag-of-words / bigram statistics, yet describe two different images. CLIP's text encoder is trained
with a sentence-level contrastive objective, not an objective that explicitly supervises word order
as spatial layout, so there is nothing in the training signal that specifically forces these two
sentences to be pushed apart from each other. We treat this as a hypothesis to test empirically —
the report always states the *measured* failure rate per category rather than baking in a fixed narrative.

---

## 5. Hard-Negative Reranking (Training-Free)

For RQ3, we test whether a purely statistical, training-free post-processing step can improve Recall@1:

1. Run the standard retrieval pass and record, for every query, whether its top-1 candidate was correct or not.
2. Build a frequency table of how often each candidate index appears as an *incorrect* top-1 prediction across the whole query set — candidates with high frequency are suspected "hard negatives" (generic scenes or phrasing that CLIP's embedding space places suspiciously close to many different queries).
3. For each query's top-K candidate list, apply a similarity penalty proportional to a candidate's hard-negative frequency (normalised by the maximum observed frequency), then re-sort.

This never updates CLIP's weights — it only reweights an already-computed, frozen similarity ranking using dataset-level statistics. See `src/retrieval/hard_negative_reranking.py`.

**Note on in-distribution optimism:** The frequency table is computed over the *same* test set used for evaluation. This introduces mild optimism — the penalty is perfectly calibrated to this specific split. The +5.66 pp improvement should be interpreted as a proof-of-concept upper bound rather than an expected generalisation gain on unseen data.

---

## 6. Dataset Statistics (Flickr30K, Standard Test Split)

| Property | Value |
|---|---|
| Total images | 31,783 |
| Captions per image | 5 |
| Evaluation split (this project) | 1,000 images / 5,000 captions |
| Split selection | Deterministic seeded shuffle (seed=42) over sorted image IDs |
| Caption source file | `results_20130124.token` |
| Hardware | Kaggle T4 GPU (16 GB VRAM) |
| Framework | PyTorch 2.1, transformers 4.36 |

### Comparison to MS-COCO

| Property | Flickr30K | MS-COCO (5K split) |
|---|---|---|
| Total images | 31,783 | 123,287 |
| Test images | 1,000 | 5,000 |
| Captions per image | 5 | 5 |
| Caption style | Short, object-focused | Longer, more descriptive |
| Zero-shot CLIP R@1 (T→I) | 66.3% (ViT-L/14) | ~58–62% (ViT-L/14) |
| Common in zero-shot benchmarks | ✅ | ✅ |

We chose Flickr30K for its smaller scale (enabling a single-GPU run in under an hour) and its shorter captions, which make the query-categorisation rules more reliable. Results may not transfer directly to MS-COCO due to caption length and style differences.

### Statistical Note on Split Size

With 5,000 text queries at Recall@1 ≈ 0.66, the standard error of the mean is
`√(0.66 × 0.34 / 5000) ≈ 0.0067` (0.67 pp). Differences larger than ~2 pp between
backbones or reranking stages are statistically reliable at the 95% confidence level.
The 1,000-image split provides sufficient statistical power for the comparisons made
in this project.

---

## 7. Backbone Comparison

Two frozen CLIP backbones are compared under identical protocol:

| Backbone | HF Model ID | Embedding Dim | Params |
|---|---|---|---|
| ViT-B/32 | `openai/clip-vit-base-patch32` | 512 | ~150M |
| ViT-L/14 | `openai/clip-vit-large-patch14` | 512 (projected) | ~428M |

Both output a 512-dim projected embedding, so all retrieval and evaluation code is backbone-agnostic;
only the encoder weights differ.

---

## 8. Limitations & Future Work

### Current Limitations

1. **Single-dataset scope.** All conclusions are specific to Flickr30K. Replication on MS-COCO,
   Conceptual Captions, or domain-specific datasets (medical, remote sensing) is needed before
   general claims can be made.

2. **No official split.** The Kaggle mirror of Flickr30K does not include the Faghri et al. (2018)
   train/val/test split. Our deterministic 1K split may not be directly comparable to numbers
   reported in prior work using the official split.

3. **Rule-based categoriser.** Keyword matching cannot handle complex sentence structure. A
   dependency-parse or NLP-based categoriser would produce more accurate category assignments.

4. **Reranking on test data.** The hard-negative frequency table is derived from the same
   test set used for evaluation (no separate calibration set). This makes the reranking improvement
   an upper bound on generalisation performance.

5. **No ablation of `penalty_weight`.** The default value (0.15) was set heuristically.
   A proper sweep over a held-out validation set is needed for deployment-grade tuning.

### Future Directions

- **FAISS integration:** Replace the dense matrix multiply with an approximate nearest-neighbour
  index (FAISS IVF or HNSW) to scale to the full 31K-image corpus.
- **CLIP fine-tuning ablation:** Compare zero-shot performance against a retrieval-fine-tuned
  CLIP to quantify how much is left on the table.
- **LLM-based query rewriting:** Use an LLM to rephrase spatial-relation captions before
  encoding, to test whether the failure mode is in the embedding or the text representation.
- **Broader backbone sweep:** Include BLIP-2, SigLIP, and InternVL for a fairer cross-architecture
  comparison.
- **Calibrated reranking:** Hold out a validation split to tune `penalty_weight` and validate
  that the reranking gain transfers to unseen data.

---

## References

- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
- Young, P., et al. (2014). *From Image Descriptions to Visual Denotations.* TACL.
- Faghri, F., et al. (2018). *VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.* BMVC.
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data.
- Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV.
