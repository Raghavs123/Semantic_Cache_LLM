from fastapi.testclient import TestClient

from faqcache.api import create_app


def test_api_ask_and_stats(bot):
    with TestClient(create_app(bot)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        r = client.post("/v1/ask", json={"question": "How can I track my order?"})
        assert r.status_code == 200 and r.json()["source"] == "cache"
        r = client.post("/v1/ask", json={"question": "Do you sell medicines?"})
        assert r.json()["source"] == "llm"
        stats = client.get("/v1/stats").json()
        assert stats["requests"] == 2 and stats["hits"] == 1


def test_api_validates_input(bot):
    with TestClient(create_app(bot)) as client:
        assert client.post("/v1/ask", json={"question": ""}).status_code == 422
        bad = {"question": "hi", "tenant": "Bad Tenant!"}
        assert client.post("/v1/ask", json=bad).status_code == 422
