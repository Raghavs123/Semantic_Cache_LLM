"""Safety rules: when must the cache NOT be used?

1. Negation guard  - "refund milega" vs "refund nahi milega" look alike to an embedding model
                     but mean opposite things. If exactly one side has a negation word, skip.
2. User-data guard - never cache (or serve from cache) text that holds one person's data:
                     order ids, phone numbers, e-mails, names.
3. TTL policy      - answers about prices, fees, offers or policies expire after N days.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ----------------------------------------------------------------------------- negation
NEGATION_WORDS = frozenset(
    {
        # English
        "not", "no", "never", "none", "nothing", "without", "cannot", "cant", "dont", "doesnt",
        "didnt", "wont", "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt",
        "shouldnt", "wouldnt", "couldnt",
        # Hinglish / Hindi in Latin script
        "nahi", "nahin", "nhi", "nai", "mat", "bina",
    }
)  # fmt: skip
# Phrases that contain a negation word but are not a negation of the question's meaning.
NEGATION_EXCEPTIONS = ("no-cost", "no cost", "no problem", "no worries")

_WORD = re.compile(r"[a-z0-9']+")


def _clean(text: str) -> str:
    text = text.lower().replace("’", "'")
    for phrase in NEGATION_EXCEPTIONS:
        text = text.replace(phrase, " ")
    return text


def has_negation(text: str) -> bool:
    for tok in _WORD.findall(_clean(text)):
        if tok in NEGATION_WORDS or tok.endswith("n't") or tok.replace("'", "") in NEGATION_WORDS:
            return True
    return False


def negation_mismatch(a: str, b: str) -> bool:
    """True when exactly one of the two questions is negated."""
    return has_negation(a) != has_negation(b)


# ----------------------------------------------------------------------------- user data
_USER_DATA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("order_id", re.compile(r"\b(?:ord|order)[\s#:-]*[a-z]{0,3}\d{5,}\b", re.IGNORECASE)),
    ("ref_number", re.compile(r"#\s?\d{4,}")),
    ("phone", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("long_number", re.compile(r"(?<!\d)\d{8,}(?!\d)")),
    ("name", re.compile(r"\b(?:my name is|i am called|mera naam)\s+[a-z]+", re.IGNORECASE)),
    ("greeting_name", re.compile(r"\b(?:[Dd]ear|[Hh]i|[Hh]ey)\s+[A-Z][a-z]+,")),
)


def find_user_data(text: str) -> str | None:
    """Return the kind of personal data found (e.g. 'order_id'), or None."""
    for kind, pattern in _USER_DATA_PATTERNS:
        if pattern.search(text):
            return kind
    return None


def redact(text: str) -> str:
    """Mask personal data before it is written to a log file."""
    for kind, pattern in _USER_DATA_PATTERNS:
        text = pattern.sub(f"[{kind}]", text)
    return text


# ----------------------------------------------------------------------------- TTL
_VOLATILE = re.compile(
    r"\b(price|prices|pricing|cost|costs|fee|fees|charge|charges|charged|offer|offers|"
    r"discount|discounts|sale|coupon|coupons|emi|interest|policy|policies|kitna|kitne|kitni)\b",
    re.IGNORECASE,
)


def is_volatile(text: str) -> bool:
    return bool(_VOLATILE.search(text))


@dataclass(frozen=True)
class TTLPolicy:
    """How long a cached answer may live, in seconds (None = forever)."""

    volatile_ttl_s: float | None = 7 * 86_400
    default_ttl_s: float | None = None

    def ttl_for(self, question: str, answer: str) -> float | None:
        if is_volatile(question) or is_volatile(answer):
            return self.volatile_ttl_s
        return self.default_ttl_s
