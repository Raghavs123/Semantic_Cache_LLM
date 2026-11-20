"""Turn text into unit-length vectors, so a dot product equals the cosine similarity."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim); every row has length 1."""
        ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    """Real embedding model (downloads once from Hugging Face, then runs locally on CPU).

    `paraphrase-multilingual-MiniLM-L12-v2` (384 dims) handles English and Hinglish;
    `all-MiniLM-L6-v2` is smaller and English-only.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import: keep it lazy

        self.name = model_name
        # E5 models are trained with a "query: " prefix; for question-vs-question similarity
        # the model card says to put "query: " on both sides.
        self.prefix = "query: " if "e5" in model_name.lower() else ""
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self.embed(["dimension probe"]).shape[1])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            [self.prefix + t for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder:
    """Tiny offline embedder (character 3-grams hashed into buckets).

    It needs no download, so unit tests and CI run in seconds. It only knows spelling,
    not meaning: never use it for the real evaluation.
    """

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hash-3gram-{dim}"
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {re.sub(r'[^a-z0-9 ]', ' ', text.lower())}  "
            for i in range(len(padded) - 2):
                out[row, zlib.crc32(padded[i : i + 3].encode()) % self.dim] += 1.0
        return l2_normalize(out)


def build_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(model_name)
