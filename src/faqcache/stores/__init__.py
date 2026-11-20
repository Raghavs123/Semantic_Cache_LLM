"""Vector stores: where cached question vectors live and how the nearest one is found."""

from __future__ import annotations

from faqcache.stores.base import CacheEntry, Candidate, VectorStore
from faqcache.stores.memory import InMemoryVectorStore


def build_store(kind: str, dim: int) -> VectorStore:
    if kind == "faiss":
        from faqcache.stores.faiss_store import FaissVectorStore

        return FaissVectorStore(dim)
    return InMemoryVectorStore(dim)


__all__ = ["CacheEntry", "Candidate", "InMemoryVectorStore", "VectorStore", "build_store"]
