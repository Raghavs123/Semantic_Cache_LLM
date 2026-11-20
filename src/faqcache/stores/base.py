from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class CacheEntry:
    id: str
    question: str
    answer: str
    created_at: float
    expires_at: float | None  # unix time; None = never
    source: Literal["faq", "llm"]
    faq_id: str | None = None

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


@dataclass(frozen=True)
class Candidate:
    entry: CacheEntry
    score: float  # cosine similarity in [-1, 1]


class VectorStore(Protocol):
    dim: int

    def add(self, entry: CacheEntry, vector: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> list[Candidate]:
        """Return up to k candidates, best (highest score) first."""
        ...

    def remove(self, entry_id: str) -> None: ...

    def get(self, entry_id: str) -> CacheEntry | None: ...

    def entries(self) -> list[CacheEntry]: ...

    def __len__(self) -> int: ...
