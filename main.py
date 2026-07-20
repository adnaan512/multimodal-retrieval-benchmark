#!/usr/bin/env python3
"""
CLI entry point for the multimodal-retrieval-benchmark project.

Usage:
    python main.py --mode demo
    python main.py --mode local --data-dir ./flickr30k
    python main.py --mode local --data-dir ./flickr30k --backbone vit-l-14
    python main.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from src.analysis.failure_analyzer import (
    failure_rate_by_category,
    find_failures,
    hardest_queries,
    semantic_failure_summary,
)
from src.analysis.query_categorizer import categorize_captions
from src.backbone.clip_encoder import build_encoder
from src.benchmark.evaluator import evaluate
from src.data.dataset_loader import build_mock_dataset, load_flickr30k_test_split
from src.indexing.embedding_index import load_or_build_indices
from src.reporting.report_generator import generate_report
from src.retrieval.hard_negative_reranking import rerank_with_hard_negatives
from src.retrieval.image_to_text import run_image_to_text
from src.retrieval.text_to_image import run_text_to_image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Zero-shot cross-modal retrieval benchmark using CLIP on Flickr30K.",
    )
    p.add_argument(
        "--mode", choices=["demo", "local"], default="demo",
        help="'demo' runs on a 50-item synthetic mock dataset (no downloads). "
             "'local' runs on the real Flickr30K test split (requires --data-dir).",
    )
    p.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to extracted Flickr30K dataset (required for --mode local).",
    )
    p.add_argument(
        "--backbone", choices=["vit-b-32", "vit-l-14"], default="vit-b-32",
        help="CLIP backbone to use (ignored in demo mode, which always uses the mock encoder).",
    )
    p.add_argument(
        "--compare-backbones", action="store_true",
        help="Run both vit-b-32 and vit-l-14 and compare (local mode only).",
    )
    p.add_argument("--k", type=int, default=10, help="Top-K for retrieval / Recall@K.")
    p.add_argument("--cache-dir", type=str, default=".cache", help="Where to cache embeddings.")
    p.add_argument("--output", type=str, default="report.html", help="Path to write the HTML report.")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print the planned run without computing anything.",
    )
    return p


def _caption_to_image_index(n_images: int, captions_per_image: int = 5):
    return [i // captions_per_image for i in range(n_images * captions_per_image)]


def _image_to_caption_indices(n_images: int, captions_per_image: int = 5):
    return [list(range(i * captions_per_image, (i + 1) * captions_per_image)) for i in range(n_images)]


def run_pipeline(backbone: str, mock: bool, image_ids, captions, cache_dir: str, k: int):
    encoder = build_encoder(backbone=backbone, mock=mock)
    n_images = len(image_ids)
    backbone_cache = os.path.join(cache_dir, "mock" if mock else backbone)

    image_matrix, text_matrix = load_or_build_indices(
        encoder, image_ids, captions, cache_dir=backbone_cache, is_mock=mock,
    )

    cap2img = _caption_to_image_index(n_images)
    img2caps = _image_to_caption_indices(n_images)

    t2i_results = run_text_to_image(text_matrix, image_matrix, cap2img, k=k)
    for r, caption in zip(t2i_results, captions):
        r.query_text = caption

    i2t_results = run_image_to_text(image_matrix, text_matrix, img2caps, k=k)

    categories = categorize_captions(captions)

    t2i_metrics = evaluate(t2i_results, direction="text_to_image", backbone=backbone, categories=categories)
    i2t_metrics = evaluate(i2t_results, direction="image_to_text", backbone=backbone)

    return {
        "encoder": encoder,
        "image_matrix": image_matrix,
        "text_matrix": text_matrix,
        "t2i_results": t2i_results,
        "i2t_results": i2t_results,
        "categories": categories,
        "t2i_metrics": t2i_metrics,
        "i2t_metrics": i2t_metrics,
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "local" and not args.data_dir:
        parser.error("--mode local requires --data-dir /path/to/flickr30k")

    print(f"[main] mode={args.mode} backbone={args.backbone} k={args.k}")

    if args.dry_run:
        print("[main] --dry-run: configuration is valid, exiting without computation.")
        print(f"  mode          = {args.mode}")
        print(f"  data_dir      = {args.data_dir}")
        print(f"  backbone(s)   = {['vit-b-32', 'vit-l-14'] if args.compare_backbones else [args.backbone]}")
        print(f"  k             = {args.k}")
        print(f"  cache_dir     = {args.cache_dir}")
        print(f"  output        = {args.output}")
        return 0

    start = time.time()

    if args.mode == "demo":
        samples = build_mock_dataset(n_items=50)
        mock = True
        backbones_to_run = ["mock"]
    else:
        samples = load_flickr30k_test_split(args.data_dir)
        mock = False
        backbones_to_run = ["vit-b-32", "vit-l-14"] if args.compare_backbones else [args.backbone]

    image_ids = [s.image_path for s in samples]
    captions = [cap for s in samples for cap in s.captions]
    print(f"[main] loaded {len(samples)} images / {len(captions)} captions")

    all_metrics_rows = []
    run_state = None
    for bb in backbones_to_run:
        print(f"[main] running backbone={bb} ...")
        state = run_pipeline(bb, mock, image_ids, captions, args.cache_dir, args.k)
        all_metrics_rows.append(state["t2i_metrics"].summary_row())
        all_metrics_rows.append(state["i2t_metrics"].summary_row())
        run_state = state  # keep the last one for failure analysis / reranking demo
        print(f"  text_to_image: {state['t2i_metrics'].summary_row()}")
        print(f"  image_to_text: {state['i2t_metrics'].summary_row()}")

    # Failure analysis on the last backbone's text->image results
    categories = run_state["categories"]
    t2i_results = run_state["t2i_results"]
    per_category = failure_rate_by_category(t2i_results, categories, k=1)
    _ = find_failures(t2i_results, categories, k=args.k)  # reserved for future reporting use
    top_failures = hardest_queries(t2i_results, categories, top_n=5)
    finding = semantic_failure_summary(per_category)
    print(f"[main] {finding}")

    # Hard negative reranking (RQ3)
    before_metrics = evaluate(t2i_results, direction="text_to_image", backbone="reranking-before")
    reranked = rerank_with_hard_negatives(t2i_results)
    after_metrics = evaluate(reranked, direction="text_to_image", backbone="reranking-after")
    print(f"[main] hard-negative reranking: R@1 {before_metrics.recall_at_1:.4f} -> {after_metrics.recall_at_1:.4f}")

    report_path = generate_report(
        output_path=args.output,
        metrics_rows=all_metrics_rows,
        per_category=per_category,
        failure_summary=finding,
        top_failures=top_failures,
        hard_negative_before={"R@1": round(before_metrics.recall_at_1, 4)},
        hard_negative_after={"R@1": round(after_metrics.recall_at_1, 4)},
        run_config={
            "mode": args.mode,
            "backbone(s)": ", ".join(backbones_to_run),
            "k": args.k,
            "n_images": len(samples),
            "n_captions": len(captions),
            "elapsed_seconds": round(time.time() - start, 2),
        },
    )
    print(f"[main] report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
