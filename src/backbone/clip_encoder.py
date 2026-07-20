"""
CLIP encoder wrapper: turns images and text into a shared 512-dim embedding
space, L2-normalized so that dot product == cosine similarity.

Background (RQ1):
    CLIP (Radford et al., 2021) is trained contrastively on ~400M image-text
    pairs scraped from the web. During training, the model sees a batch of
    (image, caption) pairs and is pushed to make the embedding of "a dog
    running in a park" close to the embedding of the *matching* photo, and
    far from the embeddings of the other captions/images in the batch. After
    training, no matching is needed at inference time — a caption's embedding
    will simply land close to any photo whose content matches it, even for
    images and captions never seen during training. Zero-shot cross-modal
    retrieval (what this project measures) is a direct consequence of that
    shared semantic space: no retrieval-specific head or fine-tuning is
    required; we just embed both sides and rank by cosine similarity.

Two interchangeable encoders are provided:
    - ``CLIPEncoder``     : real HuggingFace CLIP (ViT-B/32 or ViT-L/14), frozen.
    - ``MockCLIPEncoder`` : deterministic SHA-256-hash-based vectors, so that
                            tests / CI / the offline demo never need to download
                            model weights or images.

Both implement the same interface::

    encode_image(...)        -> np.ndarray  shape (D,)
    encode_text(...)         -> np.ndarray  shape (D,)
    encode_images_batch(...) -> np.ndarray  shape (N, D)
    encode_texts_batch(...)  -> np.ndarray  shape (N, D)

All returned arrays are L2-normalized (unit norm) and have dtype ``float32``.
``D`` is always ``EMBED_DIM = 512`` for both CLIP backbones (ViT-L/14 uses a
linear projection to 512 dims internally).
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

EMBED_DIM: int = 512
"""Embedding dimensionality shared by both CLIP backbones and the mock encoder."""

BATCH_SIZE: int = 64
"""Maximum number of images or captions processed in a single forward pass."""

SUPPORTED_BACKBONES: dict[str, str] = {
    "vit-b-32": "openai/clip-vit-base-patch32",
    "vit-l-14": "openai/clip-vit-large-patch14",
}
"""Map of short backbone names to their HuggingFace model identifiers."""


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize ``x`` along its last axis.

    Clips zero norms to ``1e-12`` to avoid division-by-zero on degenerate inputs.

    Args:
        x: Array of shape ``(D,)`` or ``(N, D)``.

    Returns:
        Array of the same shape with unit L2 norm along the last axis.
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    return x / norm


class CLIPEncoder:
    """Real CLIP encoder backed by HuggingFace ``transformers``.

    Weights are always frozen — this project never fine-tunes CLIP; see
    ``docs/RESEARCH.md`` Decision 3. Use ``MockCLIPEncoder`` in tests and CI
    to avoid downloading model weights.

    Args:
        backbone: Short backbone identifier; must be a key in
            ``SUPPORTED_BACKBONES``. Defaults to ``"vit-b-32"``.
        device: PyTorch device string (e.g., ``"cpu"``, ``"cuda"``).
            Defaults to ``"cpu"``.

    Raises:
        ValueError: If ``backbone`` is not in ``SUPPORTED_BACKBONES``.
        ImportError: If ``torch`` or ``transformers`` are not installed.
    """

    def __init__(self, backbone: str = "vit-b-32", device: str = "cpu") -> None:
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
        """Encode a single PIL image to a 512-dim L2-normalised vector.

        Args:
            pil_image: A ``PIL.Image.Image`` object (RGB mode recommended).

        Returns:
            L2-normalised embedding of shape ``(512,)``, dtype ``float32``.
        """
        return self.encode_images_batch([pil_image])[0]

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string to a 512-dim L2-normalised vector.

        Args:
            text: The input caption or query string.

        Returns:
            L2-normalised embedding of shape ``(512,)``, dtype ``float32``.
        """
        return self.encode_texts_batch([text])[0]

    def encode_images_batch(self, pil_images: Sequence) -> np.ndarray:
        """Encode a sequence of PIL images in batches.

        Args:
            pil_images: Sequence of ``PIL.Image.Image`` objects.

        Returns:
            L2-normalised embedding matrix of shape ``(N, 512)``, dtype
            ``float32``, where ``N = len(pil_images)``.
        """
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
        """Encode a sequence of text strings in batches.

        Args:
            texts: Sequence of caption or query strings.

        Returns:
            L2-normalised embedding matrix of shape ``(N, 512)``, dtype
            ``float32``, where ``N = len(texts)``.
        """
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i:i + BATCH_SIZE]
            inputs = self.processor(
                text=list(chunk), return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with self.torch.no_grad():
                feats = self.model.get_text_features(**inputs)

            # Handle newer transformers versions returning BaseModelOutputWithPooling
            if not isinstance(feats, self.torch.Tensor):
                feats = getattr(feats, "text_embeds", getattr(feats, "pooler_output", feats[0]))

            out.append(feats.cpu().numpy())
        arr = np.concatenate(out, axis=0)
        return _l2_normalize(arr)


class MockCLIPEncoder:
    """Deterministic stand-in for ``CLIPEncoder`` used in tests and CI.

    Maps each input to a 512-dim vector by expanding a SHA-256 digest through
    a seeded NumPy RNG. The same input *always* produces the same vector
    (reproducibility is verified in ``tests/test_clip_encoder.py``).

    This lets the full pipeline (indexing, retrieval, evaluation, reporting)
    run with **zero downloads and zero GPU**, which is what powers
    ``examples/run_demo.py`` and the GitHub Actions CI workflow.

    Args:
        backbone: Identifier stored on ``self.backbone``; does not affect
            embedding computation. Defaults to ``"mock"``.
        embed_dim: Output dimensionality. Defaults to ``EMBED_DIM`` (512).
    """

    def __init__(self, backbone: str = "mock", embed_dim: int = EMBED_DIM) -> None:
        self.backbone = backbone
        self.embed_dim = embed_dim

    def _hash_vector(self, key: str) -> np.ndarray:
        """Derive a ``float32`` vector from the SHA-256 digest of ``key``.

        Args:
            key: Any string; typically prefixed with ``"IMG::"`` or
                ``"TXT::"`` to ensure image/text vectors differ.

        Returns:
            Random-normal vector of shape ``(embed_dim,)`` seeded
            deterministically from the first 8 bytes of the digest.
        """
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.RandomState(seed % (2**32))
        vec = rng.normal(size=self.embed_dim).astype(np.float32)
        return vec

    def encode_image(self, image_identifier: str) -> np.ndarray:
        """Encode an image identifier to a L2-normalised mock embedding.

        No actual image file is read — ``image_identifier`` is hashed directly.

        Args:
            image_identifier: A string such as a file path or image ID
                (e.g., ``"mock_image_0007.jpg"``).

        Returns:
            L2-normalised embedding of shape ``(embed_dim,)``, dtype ``float32``.
        """
        return _l2_normalize(self._hash_vector(f"IMG::{image_identifier}")[None, :])[0]

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string to a L2-normalised mock embedding.

        Args:
            text: Any caption or query string.

        Returns:
            L2-normalised embedding of shape ``(embed_dim,)``, dtype ``float32``.
        """
        return _l2_normalize(self._hash_vector(f"TXT::{text}")[None, :])[0]

    def encode_images_batch(self, image_identifiers: Sequence[str]) -> np.ndarray:
        """Encode a sequence of image identifiers.

        Args:
            image_identifiers: Sequence of image path strings or IDs.

        Returns:
            L2-normalised matrix of shape ``(N, embed_dim)``.
        """
        return np.stack([self.encode_image(i) for i in image_identifiers], axis=0)

    def encode_texts_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of text strings.

        Args:
            texts: Sequence of caption or query strings.

        Returns:
            L2-normalised matrix of shape ``(N, embed_dim)``.
        """
        return np.stack([self.encode_text(t) for t in texts], axis=0)


def build_encoder(backbone: str, mock: bool = False, device: str = "cpu"):
    """Factory function used by ``main.py`` to obtain a real or mock encoder.

    Args:
        backbone: CLIP backbone identifier (e.g., ``"vit-b-32"``). Ignored
            when ``mock=True``.
        mock: If ``True``, return a ``MockCLIPEncoder`` (no downloads, no GPU).
        device: PyTorch device string forwarded to ``CLIPEncoder``.

    Returns:
        A ``CLIPEncoder`` or ``MockCLIPEncoder`` instance with the standard
        encode interface.
    """
    if mock:
        return MockCLIPEncoder(backbone=backbone)
    return CLIPEncoder(backbone=backbone, device=device)
