from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pharmaeval.adapters import ModelAdapter
from pharmaeval.prompts import build_prompt
from pharmaeval.schema import Item
from pharmaeval.scoring import Score, score_item


@dataclass
class ItemResult:
    item: Item
    reply: str
    score: Score


@dataclass
class ModelRun:
    model: str
    results: list[ItemResult]

    @property
    def errors(self) -> int:
        return sum(r.score.failure == "error" for r in self.results)


class ResponseCache:
    """Append-only JSONL cache so re-running a benchmark does not pay for the same answers twice."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._data[row["key"]] = row["reply"]

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._data.get(self.key(model, prompt))

    def put(self, model: str, prompt: str, reply: str) -> None:
        key = self.key(model, prompt)
        self._data[key] = reply
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "model": model, "reply": reply}, ensure_ascii=False) + "\n")


def run_model(adapter: ModelAdapter, items: list[Item], delay: float = 0.0, cache: ResponseCache | None = None,
              sleep: Callable[[float], None] = time.sleep) -> ModelRun:
    results: list[ItemResult] = []
    for item in items:
        prompt = build_prompt(item)
        reply = cache.get(adapter.name, prompt) if cache else None
        if reply is None:
            if delay and results:
                sleep(delay)
            try:
                reply = adapter.generate(prompt)
            except Exception as exc:  # one failed call must not discard the rest of a paid run
                results.append(ItemResult(item, "", Score(False, "error", f"{type(exc).__name__}: {exc}"[:160])))
                continue
            if cache:
                cache.put(adapter.name, prompt, reply)
        results.append(ItemResult(item, reply, score_item(item, reply)))
    return ModelRun(adapter.name, results)
