from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from pharmaeval.schema import Item


class DatasetError(ValueError):
    pass


def read_items(path: str | Path) -> tuple[list[Item], list[str]]:
    """Parse a JSONL file. Returns (valid items, problems with line numbers)."""
    items: list[Item] = []
    problems: list[str] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(Item.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            problems.append(f"line {lineno}: invalid JSON ({exc.msg})")
        except ValidationError as exc:
            err = exc.errors()[0]
            where = ".".join(str(p) for p in err["loc"]) or "item"
            problems.append(f"line {lineno}: {where}: {err['msg']}")
    return items, problems


def check_dataset(items: list[Item]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            problems.append(f"duplicate id: {item.id}")
        seen.add(item.id)
    pairs: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.pair_id:
            pairs[item.pair_id].append(item)
    for pair_id, group in pairs.items():
        langs = sorted(i.language for i in group)
        if langs != ["ar", "en"]:
            problems.append(f"pair {pair_id}: needs exactly one ar and one en item, found {langs}")
        elif len({i.type for i in group}) > 1 or len({i.category for i in group}) > 1:
            problems.append(f"pair {pair_id}: twins must share type and category")
    return problems


def load_dataset(path: str | Path) -> list[Item]:
    items, problems = read_items(path)
    problems += check_dataset(items)
    if problems:
        raise DatasetError("\n".join(problems))
    if not items:
        raise DatasetError("dataset is empty")
    return items
