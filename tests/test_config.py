from faqcache.config import Settings


def test_env_overrides_and_empty_values(monkeypatch):
    monkeypatch.setenv("FAQ_THRESHOLD", "0.9")
    monkeypatch.setenv("FAQ_LOG_PATH", "")
    monkeypatch.setenv("FAQ_DEFAULT_TTL_DAYS", "none")
    s = Settings(_env_file=None)
    assert s.threshold == 0.9
    assert s.log_path is None  # empty value disables logging
    assert s.default_ttl_s is None
    assert s.volatile_ttl_s == 7 * 86_400
