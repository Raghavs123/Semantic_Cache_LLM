"""Free local model through Ollama's REST API.   pip install -e ".[ollama]"   (template - not run)

1. Install Ollama from https://ollama.com and pull a small model, e.g. `ollama pull llama3.2`
2. Set FAQ_LLM_PROVIDER=ollama and FAQ_LLM_MODEL=llama3.2
"""

from __future__ import annotations

import httpx

from faqcache.llm.base import LLMResponse


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.model = model
        self._http = httpx.Client(base_url=host, timeout=120.0)

    def generate(self, question: str, system: str) -> LLMResponse:
        r = self._http.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ],
            },
        )
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "").strip()
        return LLMResponse(
            text=text,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            model=self.model,
            cacheable=data.get("done_reason", "stop") == "stop" and bool(text),
        )
