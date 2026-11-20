"""FAISS inner-product index. With unit vectors, inner product == cosine similarity.

IndexFlatIP is exact search. For millions of entries switch to an approximate index
(IndexHNSWFlat or IndexIVFFlat) and re-measure the hit rate: approximate search can miss
the true nearest neighbour.
"""

from __future__ import annotations

import numpy as np

from faqcache.stores.base import CacheEntry, Candidate


class FaissVectorStore:
    def __init__(self, dim: int) -> None:
        import faiss  # optional dependency: pip install "semantic-faq-cache[faiss]"

        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._entries: dict[str, CacheEntry] = {}
        self._to_int: dict[str, int] = {}
        self._to_str: dict[int, str] = {}
        self._next = 0

    def add(self, entry: CacheEntry, vector: np.ndarray) -> None:
        if entry.id in self._to_int:
            self.remove(entry.id)
        num = self._next
        self._next += 1
        self._index.add_with_ids(
            vector.reshape(1, -1).astype(np.float32), np.array([num], dtype=np.int64)
        )
        self._entries[entry.id] = entry
        self._to_int[entry.id] = num
        self._to_str[num] = entry.id

    def remove(self, entry_id: str) -> None:
        num = self._to_int.pop(entry_id, None)
        if num is None:
            return
        self._index.remove_ids(np.array([num], dtype=np.int64))
        self._to_str.pop(num, None)
        self._entries.pop(entry_id, None)

    def search(self, vector: np.ndarray, k: int) -> list[Candidate]:
        total = self._index.ntotal
        if total == 0:
            return []
        scores, ids = self._index.search(vector.reshape(1, -1).astype(np.float32), min(k, total))
        out = []
        for score, num in zip(scores[0], ids[0], strict=True):
            if num == -1:
                continue
            out.append(Candidate(self._entries[self._to_str[int(num)]], float(score)))
        return out

    def get(self, entry_id: str) -> CacheEntry | None:
        return self._entries.get(entry_id)

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)
