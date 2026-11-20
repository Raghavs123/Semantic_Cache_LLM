from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cacheable: bool = True  # False for refusals, truncated or error answers


class LLMClient(Protocol):
    name: str

    def generate(self, question: str, system: str) -> LLMResponse: ...


def build_system_prompt(faqs) -> str:
    """System prompt = rules + the FAQ list (kept identical across calls so it can be cached)."""
    lines = [
        "You are the customer-support assistant of DemoMart, an online store in India.",
        "Answer in one to three short sentences, using the FAQ below when it applies.",
        "Reply in the customer's style: English, or Hinglish if they wrote in Hinglish.",
        "Never invent order details. If the question needs the customer's own order or "
        "account data, ask them to use the 'Help' page while signed in.",
        "If the question is not about shopping on DemoMart, say politely that you can only "
        "help with DemoMart orders, payments, delivery, returns and accounts.",
        "",
        "FAQ:",
    ]
    lines += [f"Q: {f.question}\nA: {f.answer}" for f in faqs]
    return "\n".join(lines)
