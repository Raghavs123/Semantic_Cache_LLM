"""Exact (brute-force) search with NumPy. Fast enough for tens of thousands of entries."""

from __future__ import annotations

import numpy as np

from faqcache.stores.base import CacheEntry, Candidate


class InMemoryVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._entries: dict[str, CacheEntry] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._matrix: np.ndarray | None = None  # rebuilt lazily after a change
        self._ids: list[str] = []

    def add(self, entry: CacheEntry, vector: np.ndarray) -> None:
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vector.shape}")
        self._entries[entry.id] = entry
        self._vectors[entry.id] = vector.astype(np.float32)
        self._matrix = None

    def remove(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)
        if self._vectors.pop(entry_id, None) is not None:
            self._matrix = None

    def search(self, vector: np.ndarray, k: int) -> list[Candidate]:
        if not self._entries:
            return []
        if self._matrix is None:
            self._ids = list(self._vectors)
            self._matrix = np.stack([self._vectors[i] for i in self._ids])
        scores = self._matrix @ vector.astype(np.float32)
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Candidate(self._entries[self._ids[i]], float(scores[i])) for i in top]

    def get(self, entry_id: str) -> CacheEntry | None:
        return self._entries.get(entry_id)

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)
