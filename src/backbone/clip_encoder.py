"""
CLIP encoder wrapper: turns images and text into a shared 512-dim embedding
space, L2-normalized so that dot product == cosine similarity.

Why this works (background for RQ1):
CLIP (Radford et al., 2021) is trained contrastively on ~400M image-text
pairs scraped from the web. During training, the model sees a batch of
(image, caption) pairs and is pushed to make the embedding of "a dog
running in a park" close to the embedding of the *matching* photo, and far
from the embeddings of the other captions/images in the batch. After
training, no matching is needed anymore -- a caption's embedding will
simply land close to any photo whose content matches it, even for images
and captions never seen during training. Zero-shot cross-modal retrieval
(what this project measures) is a direct consequence of that shared
semantic space: no retrieval-specific head or fine-tuning is required, we
just embed both sides and rank by cosine similarity.

Two interchangeable encoders are provided:
  - CLIPEncoder      : real HuggingFace CLIP (ViT-B/32 or ViT-L/14), frozen.
  - MockCLIPEncoder  : deterministic sha256-hash-based vectors, so that
                        tests/CI/the offline demo never need to download
                        model weights or images.
Both implement the same interface: encode_image(...) / encode_text(...) /
encode_images_batch(...) / encode_texts_batch(...), all returning
L2-normalized numpy arrays of shape (512,) or (N, 512).
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Union

import numpy as np

EMBED_DIM = 512
BATCH_SIZE = 64

SUPPORTED_BACKBONES = {
    "vit-b-32": "openai/clip-vit-base-patch32",
    "vit-l-14": "openai/clip-vit-large-patch14",
}


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    return x / norm


class CLIPEncoder:
    """Real CLIP encoder (HuggingFace transformers). Weights are frozen --
    this project never fine-tunes CLIP; see docs/RESEARCH.md Decision 3."""

    def __init__(self, backbone: str = "vit-b-32", device: str = "cpu"):
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(SUPPORTED_BACKBONES)}")
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            raise ImportError(
                "transformers/torch are required for CLIPEncoder. Install requirements.txt, "
                "or use --mode demo / MockCLIPEncoder which needs no ML dependencies."
            ) from e

        self.torch = torch
        self.backbone = backbone
        self.device = device
        model_name = SUPPORTED_BACKBONES[backbone]
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # frozen, always
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, pil_image) -> np.ndarray:
        return self.encode_images_batch([pil_image])[0]

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts_batch([text])[0]

    def encode_images_batch(self, pil_images: Sequence) -> np.ndarray:
        out = []
        for i in range(0, len(pil_images), BATCH_SIZE):
            chunk = pil_images[i:i + BATCH_SIZE]
            inputs = self.processor(images=list(chunk), return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            
            # Handle newer transformers versions returning BaseModelOutputWithPooling
            if not isinstance(feats, self.torch.Tensor):
                feats = getattr(feats, "image_embeds", getattr(feats, "pooler_output", feats[0]))

            out.append(feats.cpu().numpy())
        arr = np.concatenate(out, axis=0)
        return _l2_normalize(arr)

    def encode_texts_batch(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i:i + BATCH_SIZE]
            inputs = self.processor(text=list(chunk), return_tensors="pt",
                                     padding=True, truncation=True).to(self.device)
            with self.torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            
            # Handle newer transformers versions returning BaseModelOutputWithPooling
            if not isinstance(feats, self.torch.Tensor):
                feats = getattr(feats, "text_embeds", getattr(feats, "pooler_output", feats[0]))

            out.append(feats.cpu().numpy())
        arr = np.concatenate(out, axis=0)
        return _l2_normalize(arr)


class MockCLIPEncoder:
    """
    Deterministic stand-in for CLIP: hashes input text/identifiers with
    sha256 and expands the digest into a 512-dim vector via a seeded RNG.
    Same input always -> same vector (reproducibility, tested in
    tests/test_clip_encoder.py). This lets the full pipeline (indexing,
    retrieval, evaluation, reporting) run with zero downloads and zero GPU,
    which is what powers examples/run_demo.py and CI.
    """

    def __init__(self, backbone: str = "mock", embed_dim: int = EMBED_DIM):
        self.backbone = backbone
        self.embed_dim = embed_dim

    def _hash_vector(self, key: str) -> np.ndarray:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.RandomState(seed % (2**32))
        vec = rng.normal(size=self.embed_dim).astype(np.float32)
        return vec

    def encode_image(self, image_identifier: str) -> np.ndarray:
        """For the mock encoder, 'image' is just its identifier/path string
        (no real image is decoded)."""
        return _l2_normalize(self._hash_vector(f"IMG::{image_identifier}")[None, :])[0]

    def encode_text(self, text: str) -> np.ndarray:
        return _l2_normalize(self._hash_vector(f"TXT::{text}")[None, :])[0]

    def encode_images_batch(self, image_identifiers: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode_image(i) for i in image_identifiers], axis=0)

    def encode_texts_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode_text(t) for t in texts], axis=0)


def build_encoder(backbone: str, mock: bool = False, device: str = "cpu"):
    """Factory used by main.py: returns a real or mock encoder."""
    if mock:
        return MockCLIPEncoder(backbone=backbone)
    return CLIPEncoder(backbone=backbone, device=device)
