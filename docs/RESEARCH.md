# Research Methodology

## 1. Background: CLIP and the shared embedding space

CLIP (Contrastive Language-Image Pre-training; Radford et al., 2021) trains
an image encoder and a text encoder jointly on ~400M (image, caption) pairs
scraped from the web. The training objective is contrastive: within a batch
of N pairs, the model is pushed to maximize cosine similarity between each
image and *its* caption while minimizing similarity to the other N-1
captions in the batch (and vice versa for images). After training, no
matching step is needed at inference time — the two encoders simply place
semantically related images and text near each other in a shared 512-dim
space. This is what makes **zero-shot cross-modal retrieval** possible: we
never train a retrieval-specific model, we only embed both sides with the
frozen encoders and rank by cosine similarity.

## 2. Zero-shot retrieval protocol

For a dataset of N images, each with 5 captions:

1. Encode all N images with `CLIPEncoder.encode_images_batch` → an
   `(N, 512)` L2-normalized matrix.
2. Encode all `N*5` captions with `CLIPEncoder.encode_texts_batch` → an
   `(N*5, 512)` L2-normalized matrix.
3. **Text→Image**: for each caption row, compute its dot product against
   every image row (equivalent to cosine similarity, since both sides are
   L2-normalized) and take the top-K image indices.
4. **Image→Text**: symmetric — for each image row, take the top-K caption
   indices.
5. A query is a "hit" at K if the ground-truth image (text→image) or *any*
   of the 5 ground-truth captions (image→text) appears in the top-K list.

No parameter of CLIP is updated at any point in this pipeline (see Decision
3 in the README) — the entire benchmark measures out-of-the-box zero-shot
capability.

## 3. Recall@K definition

For a query set Q and cutoff K:

```
Recall@K = (1 / |Q|) * sum_{q in Q} [ ground_truth(q) appears in top-K(q) ]
```

We report Recall@1, Recall@5, and Recall@10 for both retrieval directions,
plus the **median rank** of the first correct answer (a more granular
signal than Recall@K alone, since it distinguishes "just missed the
cutoff" from "wildly wrong").

## 4. Query categorization methodology

Each caption is assigned exactly one primary category using simple,
inspectable rules (`src/analysis/query_categorizer.py`), checked in this
priority order:

1. **spatial_relation** — contains a spatial preposition phrase
   ("next to", "in front of", "behind", "on top of", "beside", "to the
   left/right of").
2. **attribute_centric** — contains a color/size/age adjective (red, blue,
   tall, small, young, old, ...).
3. **scene_centric** — contains a place word (park, street, beach,
   kitchen, forest, stadium, ...).
4. **object_centric** — matches an `"a/an/the <noun> is/are/sits..."`
   pattern, or is the default category if nothing else matched.

Spatial relations are checked first because they are the most specific,
least ambiguous signal, and because they are the category this project's
hypothesis (RQ2) predicts will be hardest for CLIP — we don't want an
incidental color word in a spatial sentence to reclassify it away from the
category we most want to measure.

**Why spatial relations are expected to be hard for CLIP.** "the cat is to
the left of the dog" and "the dog is to the left of the cat" contain
exactly the same tokens and nearly identical bag-of-words / bigram
statistics, yet describe two different images. CLIP's text encoder is
trained with a sentence-level contrastive objective, not an objective that
explicitly supervises word order as spatial layout, so there is nothing in
the training signal that specifically forces these two sentences to be
pushed apart from each other. We treat this as a hypothesis to test
empirically (see `failure_analyzer.py` and the results tables), not as an
assumed conclusion — the report always states the *measured* failure rate
per category rather than baking in a fixed narrative.

## 5. Hard negative reranking (training-free)

For RQ3, we test whether a purely statistical, training-free
post-processing step can improve Recall@1:

1. Run the standard retrieval pass and record, for every query, whether
   its top-1 candidate was correct or not.
2. Build a frequency table of how often each candidate index appears as an
   *incorrect* top-1 prediction across the whole query set — candidates
   with high frequency are suspected "hard negatives" (generic scenes or
   phrasing that CLIP's embedding space places suspiciously close to many
   different queries).
3. For each query's top-K candidate list, apply a similarity penalty
   proportional to a candidate's hard-negative frequency (normalized by
   the maximum observed frequency), then re-sort.

This never updates CLIP's weights — it only reweights an already-computed,
frozen similarity ranking using dataset-level statistics. See
`src/retrieval/hard_negative_reranking.py`.

## 6. Dataset statistics (Flickr30K, standard test split)

| Property | Value |
|---|---|
| Total images | 31,783 |
| Captions per image | 5 |
| Evaluation split (this project) | 1,000 images / 5,000 captions |
| Split selection | deterministic seeded shuffle over sorted image IDs (no official split file ships with the Kaggle mirror) |
| Caption source file | `results_20130124.token` |

## 7. Backbone comparison

Two frozen CLIP backbones are compared under identical protocol:

| Backbone | HF model id | Embedding dim |
|---|---|---|
| ViT-B/32 | `openai/clip-vit-base-patch32` | 512 |
| ViT-L/14 | `openai/clip-vit-large-patch14` | 512 (projected) |

Both output a 512-dim projected embedding, so retrieval code is
backbone-agnostic; only the encoder weights differ.
