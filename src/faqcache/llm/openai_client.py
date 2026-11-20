"""OpenAI Chat Completions.   pip install -e ".[openai]"   (template - not run in this repo)

Needs OPENAI_API_KEY and FAQ_LLM_MODEL (pick a current model from OpenAI's model list).
"""

from __future__ import annotations

from openai import OpenAI

from faqcache.llm.base import LLMResponse


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = OpenAI()

    def generate(self, question: str, system: str) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        usage = resp.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=resp.model,
            cacheable=choice.finish_reason == "stop" and bool(text),
        )
