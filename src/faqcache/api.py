"""HTTP API.   Run:  uvicorn faqcache.api:app --port 8000   then open http://localhost:8000/docs"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from faqcache import __version__
from faqcache.config import get_settings
from faqcache.factory import build_bot
from faqcache.service import FAQBot, answer_dict


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["refund kab milega?"])
    tenant: str | None = Field(default=None, pattern=r"^[a-z0-9_-]{1,40}$")


class AskResponse(BaseModel):
    answer: str
    source: str
    reason: str
    similarity: float | None
    matched_question: str | None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    stored: bool
    tenant: str
    lookup_ms: float
    llm_ms: float


def create_app(bot: FAQBot | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bot = bot or build_bot(get_settings())
        app.state.bot.cache_for()  # build + seed the default tenant at start-up, not on 1st call
        yield

    app = FastAPI(title="Semantic FAQ Cache", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> dict:
        # plain `def`: FastAPI runs it in a thread pool, so blocking model calls are fine
        return answer_dict(request.app.state.bot.ask(body.question, body.tenant))

    @app.get("/v1/stats")
    def stats(request: Request) -> dict:
        b: FAQBot = request.app.state.bot
        return {**b.logger.stats(), "cache_entries": len(b.cache_for()), "provider": b.llm.name}

    return app


app = create_app()
