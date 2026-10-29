import httpx
import pytest

from pharmaeval.adapters import AnthropicAdapter, ConfigError, MockAdapter, OpenAIAdapter, make_adapter, post_with_retry
from pharmaeval.dataset import load_dataset
from pharmaeval.runner import ResponseCache, run_model

ITEMS = load_dataset("data/sample_questions.jsonl")


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_adapter_request_and_parsing():
    seen = {}

    def handler(request):
        seen["headers"], seen["body"] = request.headers, request.read().decode()
        return httpx.Response(200, json={"content": [{"type": "text", "text": "He"}, {"type": "text", "text": "llo"}]})

    adapter = AnthropicAdapter("some-model", api_key="k", system="be brief", client=client(handler))
    assert adapter.generate("hi") == "Hello"
    assert seen["headers"]["x-api-key"] == "k" and "anthropic-version" in seen["headers"]
    assert '"system":"be brief"' in seen["body"] and '"temperature":0.0' in seen["body"]


def test_openai_adapter_request_and_parsing(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://local.test/v1/")
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "B"}}]})

    adapter = OpenAIAdapter("m", api_key="k", client=client(handler))
    assert adapter.generate("q") == "B"
    assert seen["url"] == "http://local.test/v1/chat/completions" and seen["auth"] == "Bearer k"


def test_retry_on_rate_limit_then_success():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else httpx.Response(200, json={"ok": True})

    assert post_with_retry(client(handler), "http://x", {}, {}, sleep=sleeps.append) == {"ok": True}
    assert len(calls) == 3 and sleeps == [1, 2]


def test_retry_gives_up_and_raises():
    with pytest.raises(httpx.HTTPStatusError):
        post_with_retry(client(lambda r: httpx.Response(503)), "http://x", {}, {}, retries=1, sleep=lambda s: None)


def test_non_retryable_error_fails_immediately():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    with pytest.raises(httpx.HTTPStatusError):
        post_with_retry(client(handler), "http://x", {}, {}, sleep=lambda s: None)
    assert len(calls) == 1


def test_make_adapter_validation(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        make_adapter("anthropic:m", ITEMS)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make_adapter("openai:m", ITEMS)
    with pytest.raises(ConfigError, match="provider:model"):
        make_adapter("anthropic", ITEMS)
    with pytest.raises(ConfigError, match="unknown provider"):
        make_adapter("foo:bar", ITEMS)
    with pytest.raises(ConfigError, match="mock"):
        make_adapter("mock:genius", ITEMS)


def test_perfect_and_careless_mock_models():
    perfect = run_model(MockAdapter("perfect", ITEMS), ITEMS)
    careless = run_model(MockAdapter("careless", ITEMS), ITEMS)
    assert all(r.score.passed for r in perfect.results)
    assert 0 < sum(r.score.passed for r in careless.results) < len(ITEMS) / 2
    assert {r.score.failure for r in careless.results} >= {"wrong_choice", "missing_key_fact", "hallucinated_drug"}


def test_adapter_errors_are_recorded_not_raised():
    class Broken:
        name = "broken"

        def generate(self, prompt):
            raise ConnectionError("network down")

    run = run_model(Broken(), ITEMS[:3])
    assert run.errors == 3 and "ConnectionError" in run.results[0].score.detail


def test_cache_avoids_repeat_calls_and_persists(tmp_path):
    calls = []

    class Counting:
        name = "counting"

        def generate(self, prompt):
            calls.append(prompt)
            return "A"

    path = tmp_path / "cache.jsonl"
    run_model(Counting(), ITEMS[:4], cache=ResponseCache(path))
    assert len(calls) == 4
    run_model(Counting(), ITEMS[:4], cache=ResponseCache(path))  # new object: loads from disk
    assert len(calls) == 4


def test_delay_between_live_calls_only():
    sleeps = []
    run_model(MockAdapter("perfect", ITEMS), ITEMS[:3], delay=0.5, sleep=sleeps.append)
    assert sleeps == [0.5, 0.5]
