"""Load and validate the FAQ file and the labelled test set."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

NONE_LABEL = "none"


class FAQ(BaseModel):
    id: str = Field(pattern=r"^F\d{2}$")
    question: str
    answer: str
    category: str
    # Extra ways to ask the same thing. Each one is cached as its own entry (same answer),
    # which lifts the hit rate for phrasings - e.g. Hinglish - that the model scores low.
    variants: list[str] = Field(default_factory=list)

    def phrasings(self, include_variants: bool = True) -> list[str]:
        return [self.question, *self.variants] if include_variants else [self.question]


class TestItem(BaseModel):
    __test__ = False  # tell pytest this is not a test class

    id: str
    question: str
    label: str  # an FAQ id such as "F02", or "none" when the cache must NOT answer
    lang: Literal["en", "hinglish"]
    kind: Literal["paraphrase", "out_of_scope", "off_topic", "hard_negative", "negation_flip"]
    split: Literal["dev", "test"] = "dev"


def load_faqs(path: Path) -> list[FAQ]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    faqs = [FAQ.model_validate(r) for r in raw]
    ids = [f.id for f in faqs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate FAQ ids")
    return faqs


def load_test_set(path: Path, faq_ids: set[str] | None = None, seed: int = 42) -> list[TestItem]:
    items = [
        TestItem.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if faq_ids is not None:
        bad = [i.id for i in items if i.label != NONE_LABEL and i.label not in faq_ids]
        if bad:
            raise ValueError(f"test items point to unknown FAQ ids: {bad}")
    return assign_splits(items, seed)


def assign_splits(items: list[TestItem], seed: int = 42) -> list[TestItem]:
    """Stratified 50/50 dev/test split (same share of 'should hit' and 'should miss' in each).

    We tune the threshold on *dev* and report it on *test*, so the headline number is not
    measured on the same questions it was tuned on.
    """
    rng = random.Random(seed)
    out: list[TestItem] = []
    for positive in (True, False):
        group = [i for i in items if (i.label != NONE_LABEL) == positive]
        order = list(range(len(group)))
        rng.shuffle(order)
        half = len(group) // 2
        for rank, idx in enumerate(order):
            split = "dev" if rank < half else "test"
            out.append(group[idx].model_copy(update={"split": split}))
    by_id = {i.id: i for i in out}
    return [by_id[i.id] for i in items]
