"""Model adapters. Plain HTTPS calls (no vendor SDKs), so they are easy to test and to extend."""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from pharmaeval.prompts import build_prompt
from pharmaeval.schema import Item

RETRY_STATUS = {429, 500, 502, 503, 529}


class ConfigError(ValueError):
    pass


class ModelAdapter(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


def post_with_retry(client: httpx.Client, url: str, headers: dict[str, str], payload: dict[str, Any],
                    retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """POST JSON, retrying rate-limit and transient server errors with exponential backoff."""
    for attempt in range(retries + 1):
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code in RETRY_STATUS and attempt < retries:
            sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


class AnthropicAdapter:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 300, temperature: float = 0.0,
                 system: str | None = None, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set")
        self.model, self.max_tokens, self.temperature, self.system = model, max_tokens, temperature, system
        self.client, self.sleep = client or httpx.Client(timeout=60.0), sleep
        self.name = f"anthropic:{model}"

    def generate(self, prompt: str) -> str:
        payload: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                                   "messages": [{"role": "user", "content": prompt}]}
        if self.system:
            payload["system"] = self.system
        data = post_with_retry(self.client, "https://api.anthropic.com/v1/messages",
                               {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"},
                               payload, sleep=self.sleep)
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


class OpenAIAdapter:
    """OpenAI or any OpenAI-compatible server (set OPENAI_BASE_URL)."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, max_tokens: int = 300,
                 temperature: float = 0.0, system: str | None = None, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigError("OPENAI_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model, self.max_tokens, self.temperature, self.system = model, max_tokens, temperature, system
        self.client, self.sleep = client or httpx.Client(timeout=60.0), sleep
        self.name = f"openai:{model}"

    def generate(self, prompt: str) -> str:
        messages = ([{"role": "system", "content": self.system}] if self.system else [])
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                   "temperature": self.temperature}
        data = post_with_retry(self.client, f"{self.base_url}/chat/completions",
                               {"Authorization": f"Bearer {self.api_key}"}, payload, sleep=self.sleep)
        return str(data["choices"][0]["message"]["content"])


def ideal_response(item: Item) -> str:
    ar = item.language == "ar"
    if item.type == "mcq":
        return str(item.answer)
    if item.type == "short":
        return " ".join(group.split("|")[0] for group in item.must_include)
    return "لست على دراية بأي دواء بهذا الاسم ولا أستطيع التحقق منه." if ar else \
        "I'm not aware of any medication with that name, and I can't verify it."


def careless_response(item: Item) -> str:
    if item.type == "mcq":
        return "A"
    if item.type == "short":
        return "It depends on the patient."
    return "The usual adult dose is 50 mg once daily."


class MockAdapter:
    """Offline stand-ins for testing and demos: ``perfect`` always answers well, ``careless`` does not."""

    def __init__(self, kind: str, items: list[Item]) -> None:
        if kind not in ("perfect", "careless"):
            raise ConfigError("mock model must be 'mock:perfect' or 'mock:careless'")
        self.name = f"mock:{kind}"
        self._respond = ideal_response if kind == "perfect" else careless_response
        self._by_prompt = {build_prompt(i): i for i in items}

    def generate(self, prompt: str) -> str:
        return self._respond(self._by_prompt[prompt])


def make_adapter(spec: str, items: list[Item], *, max_tokens: int = 300, temperature: float = 0.0,
                 system: str | None = None) -> ModelAdapter:
    provider, _, model = spec.partition(":")
    if provider == "mock":
        return MockAdapter(model, items)
    if not model:
        raise ConfigError(f"model spec must look like provider:model, got {spec!r}")
    if provider == "anthropic":
        return AnthropicAdapter(model, max_tokens=max_tokens, temperature=temperature, system=system)
    if provider == "openai":
        return OpenAIAdapter(model, max_tokens=max_tokens, temperature=temperature, system=system)
    raise ConfigError(f"unknown provider {provider!r} (use anthropic, openai or mock)")
