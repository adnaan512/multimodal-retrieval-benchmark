"""
Rule-based caption categorization into 4 query types (RQ2):
  object_centric    -- "a/an/the <noun> is/are/sits ..."
  scene_centric     -- mentions a place word (park, street, beach, ...)
  attribute_centric -- mentions a color/size/age adjective
  spatial_relation   -- mentions a spatial preposition phrase

Why spatial relations are hard for CLIP (see README/RESEARCH.md):
"the cat is to the left of the dog" and "the dog is to the left of the
cat" share nearly identical bag-of-words content -- same tokens, same
unigram/bigram statistics -- but describe two different images. CLIP's
text encoder is a transformer trained contrastively at the sentence level,
not explicitly supervised to encode word *order* as spatial layout, so
embeddings for such sentence pairs end up closer together than their
completely different visual referents warrant. This project measures that
gap empirically in failure_analyzer.py rather than assuming it.
"""
from __future__ import annotations

import re
from typing import List

from src.models import QueryAnalysis

_OBJECT_PATTERN = re.compile(
    r"\b(a|an|the)\s+\w+\s+(is|are|sits|sitting|stands|standing|runs|running)\b",
    re.IGNORECASE,
)

_SCENE_WORDS = ["park", "street", "beach", "kitchen", "forest", "stadium", "office", "garden", "field", "city"]
_ATTR_WORDS = ["red", "blue", "green", "yellow", "black", "white", "tall", "small", "large",
               "young", "old", "big", "little"]
_SPATIAL_PHRASES = [
    "next to", "in front of", "behind", "on top of", "beside",
    "to the left of", "to the right of",
]

CATEGORIES = ("spatial_relation", "attribute_centric", "scene_centric", "object_centric")


def categorize_caption(caption: str) -> QueryAnalysis:
    """
    Categorize a single caption. Checked in priority order spatial >
    attribute > scene > object, because spatial-relation phrasing is the
    most specific signal and should not be masked by an incidental color
    word or place mention in the same sentence. Falls back to
    object_centric if nothing else matches (per spec default).
    """
    lower = caption.lower()

    for phrase in _SPATIAL_PHRASES:
        if phrase in lower:
            return QueryAnalysis(query_id=-1, text=caption, category="spatial_relation", matched_pattern=phrase)

    for word in _ATTR_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return QueryAnalysis(query_id=-1, text=caption, category="attribute_centric", matched_pattern=word)

    for word in _SCENE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return QueryAnalysis(query_id=-1, text=caption, category="scene_centric", matched_pattern=word)

    match = _OBJECT_PATTERN.search(lower)
    if match:
        return QueryAnalysis(query_id=-1, text=caption, category="object_centric", matched_pattern=match.group(0))

    return QueryAnalysis(query_id=-1, text=caption, category="object_centric", matched_pattern=None)


def categorize_captions(captions: List[str]) -> List[QueryAnalysis]:
    analyses = []
    for i, cap in enumerate(captions):
        qa = categorize_caption(cap)
        qa.query_id = i
        analyses.append(qa)
    return analyses
