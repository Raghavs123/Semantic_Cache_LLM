"""Claude via the official Anthropic SDK.   pip install -e ".[anthropic]"

Not run while building this repo (no API key on the build machine) - run it yourself with
ANTHROPIC_API_KEY set (or after `ant auth login`).

Choices:
* model defaults to claude-opus-5-5; set FAQ_LLM_MODEL=claude-sonnet-5-5 or claude-haiku-5-5
  for a cheaper bot. No temperature/top_p: current models reject sampling parameters.
* effort "low": FAQ answers are short and simple, so we ask for little thinking.
* The system prompt (rules + all FAQs) is the same on every call, so it is marked with
  cache_control for prompt caching. (This is the LLM provider's *prompt* cache; our project
  is a *response* cache. They stack.)
* Server-side refusal fallback is enabled; a refused answer is returned but never cached.
"""

from __future__ import annotations

import anthropic

from faqcache.llm.base import LLMResponse


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low") -> None:
        self.model = model
        self.effort = effort
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY or an `ant` profile

    def generate(self, question: str, system: str) -> LLMResponse:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=1024,  # deliberately short answers
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": question}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        ok = response.stop_reason == "end_turn" and bool(text)
        if response.stop_reason == "refusal":
            text = "Sorry, I can't help with that here. Please contact DemoMart customer care."
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            cacheable=ok,
        )
