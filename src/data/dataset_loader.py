"""
Flickr30K dataset loader (and a synthetic mock loader for CI / offline demo).

Flickr30K layout expected under --data-dir:
    <data-dir>/flickr30k-images/*.jpg          (31,783 images)
    <data-dir>/results_20130124.token          (5 captions per image, tab-separated:
                                                 "<image>.jpg#<caption_idx>\t<caption text>")

We only ever evaluate on the standard 1000-image test split (see
docs/RESEARCH.md for the split protocol). Loading is lazy: this module
returns file paths and caption strings, not decoded images -- decoding
happens inside the CLIP encoder so that batching stays in one place.
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

TOKEN_FILE = "results_20130124.token"
IMAGE_DIR = "flickr30k-images"
TEST_SPLIT_SIZE = 1000
CAPTIONS_PER_IMAGE = 5


@dataclass
class Sample:
    image_id: str            # e.g. "1000092795.jpg"
    image_path: str           # full path on disk
    captions: List[str]       # 5 captions for this image


def _parse_token_file(token_path: str) -> dict:
    """Parse results_20130124.token or captions.txt into {image_id: [captions...]}."""
    captions_by_image: dict = {}
    
    is_csv_format = token_path.endswith(".txt") or token_path.endswith(".csv")
    if is_csv_format:
        import csv
        with open(token_path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            delimiter = "|" if "|" in header else ","
            f.seek(0)
            
            reader = csv.reader(f, delimiter=delimiter)
            # check if first row is header
            first_row = next(reader, None)
            if first_row and not (first_row[0].endswith('.jpg') or first_row[0].endswith('.png')):
                # It was a header, skip it
                pass
            else:
                # It wasn't a header, process it
                if first_row and len(first_row) >= 2:
                    image_id = first_row[0].strip()
                    caption = first_row[-1].strip()
                    captions_by_image.setdefault(image_id, []).append(caption)

            for row in reader:
                if len(row) >= 2:
                    image_id = row[0].strip()
                    caption = row[-1].strip()  # the comment is usually the last column
                    captions_by_image.setdefault(image_id, []).append(caption)
        
        if len(captions_by_image) > 0:
            return captions_by_image

    # Original parsing logic for results_20130124.token
    with open(token_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                key, caption = line.split("\t", 1)
                image_id = key.split("#")[0]
            else:
                parts = line.split(" ", 1)
                if len(parts) == 2 and "#" in parts[0]:
                    image_id = parts[0].split("#")[0]
                    caption = parts[1]
                elif len(parts) == 2 and "," in parts[0]:
                    image_id = parts[0].split(",")[0]
                    caption = parts[1]
                else:
                    continue
            captions_by_image.setdefault(image_id, []).append(caption)
    return captions_by_image


def load_flickr30k_test_split(data_dir: str, split_size: int = TEST_SPLIT_SIZE,
                               seed: int = 42) -> List[Sample]:
    """
    Load the standard Flickr30K test split (default 1000 images).

    Requires data_dir/results_20130124.token and data_dir/flickr30k-images/.
    Selection of the 1000 test images is deterministic (sorted image ids,
    seeded shuffle) since the Kaggle mirror does not ship an official
    split file.
    """
    possible_token_files = [
        os.path.join(data_dir, "results_20130124.token"),
        os.path.join(data_dir, "captions.txt"),
        os.path.join(data_dir, "captions.csv"),
        os.path.join(data_dir, "images", "captions.txt"),
        os.path.join(data_dir, "Images", "captions.txt"),
    ]
    token_path = None
    for p in possible_token_files:
        if os.path.isfile(p):
            token_path = p
            break
            
    if not token_path:
        raise FileNotFoundError(
            f"Caption file not found in {data_dir}. Looked for results_20130124.token or captions.txt."
        )

    possible_image_dirs = [
        "flickr30k-images",
        "Images",
        "images",
        "flickr30k_images",
        "images/flickr30k_images",
        "Images/flickr30k_images"
    ]
    image_dir = None
    for d in possible_image_dirs:
        if os.path.isdir(os.path.join(data_dir, d)):
            image_dir = os.path.join(data_dir, d)
            break
            
    if not image_dir:
        raise FileNotFoundError(f"Image directory not found in {data_dir}. Expected one of {possible_image_dirs}")

    captions_by_image = _parse_token_file(token_path)
    image_ids = sorted(captions_by_image.keys())

    rng = random.Random(seed)
    rng.shuffle(image_ids)
    test_ids = sorted(image_ids[:split_size])

    samples = []
    for image_id in test_ids:
        caps = captions_by_image[image_id][:CAPTIONS_PER_IMAGE]
        if len(caps) < CAPTIONS_PER_IMAGE:
            continue  # skip malformed entries missing captions
        samples.append(Sample(
            image_id=image_id,
            image_path=os.path.join(image_dir, image_id),
            captions=caps,
        ))
    return samples


# ---------------------------------------------------------------------------
# Mock dataset: fully synthetic, no downloads, deterministic, used by
# tests/, examples/run_demo.py and CI.
# ---------------------------------------------------------------------------

_SCENES = ["park", "street", "beach", "kitchen", "forest", "stadium", "office", "garden"]
_OBJECTS = ["dog", "cyclist", "child", "chef", "surfer", "musician", "climber", "vendor"]
_ATTRS = ["red", "blue", "green", "tall", "small", "large", "young", "old"]
_SPATIAL = ["next to", "in front of", "behind", "on top of", "beside"]


def _mock_caption(idx: int, rng: random.Random) -> str:
    """Generate one synthetic caption exercising all 4 query categories."""
    kind = idx % 4
    obj = rng.choice(_OBJECTS)
    obj2 = rng.choice(_OBJECTS)
    scene = rng.choice(_SCENES)
    attr = rng.choice(_ATTRS)
    spatial = rng.choice(_SPATIAL)
    if kind == 0:
        return f"A {obj} is running near the water"
    elif kind == 1:
        return f"A {obj} playing in a {scene}"
    elif kind == 2:
        return f"A {attr} {obj} standing outside"
    else:
        return f"A {obj} {spatial} a {obj2} in a {scene}"


def build_mock_dataset(n_items: int = 50, seed: int = 7) -> List[Sample]:
    """
    Build a fully synthetic dataset of n_items image-caption groups.

    No actual image files are read: image_path is a synthetic identifier
    (e.g. 'mock_image_0007.jpg') that the MockCLIPEncoder hashes directly,
    so this works completely offline (used in CI and examples/run_demo.py).
    """
    rng = random.Random(seed)
    samples = []
    for i in range(n_items):
        image_id = f"mock_image_{i:04d}.jpg"
        captions = [_mock_caption(i * CAPTIONS_PER_IMAGE + j, rng) for j in range(CAPTIONS_PER_IMAGE)]
        samples.append(Sample(image_id=image_id, image_path=image_id, captions=captions))
    return samples


def deterministic_hash_seed(text: str) -> int:
    """Utility: stable integer seed derived from text (used by mock encoder too)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
