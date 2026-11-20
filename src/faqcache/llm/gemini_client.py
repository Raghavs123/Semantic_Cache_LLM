"""Google Gemini via the google-genai SDK.   pip install -e ".[gemini]"   (template - not run)

Needs GEMINI_API_KEY and FAQ_LLM_MODEL (pick a current model from Google's model list).
"""

from __future__ import annotations

from google import genai
from google.genai import types

from faqcache.llm.base import LLMResponse


class GeminiLLM:
    name = "gemini"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = genai.Client()

    def generate(self, question: str, system: str) -> LLMResponse:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        text = (resp.text or "").strip()
        meta = resp.usage_metadata
        return LLMResponse(
            text=text,
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=(meta.candidates_token_count or 0) if meta else 0,
            model=self.model,
            cacheable=bool(text),
        )
