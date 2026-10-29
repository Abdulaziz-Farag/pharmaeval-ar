from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

from pharmaeval.runner import ItemResult, ModelRun
from pharmaeval.schema import Item

MAX_FAILURES_LISTED = 15


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval: small benchmarks need honest error bars."""
    if n == 0:
        return 0.0, 0.0
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def rate(results: list[ItemResult]) -> float | None:
    return sum(r.score.passed for r in results) / len(results) if results else None


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def paired_gap(results: list[ItemResult]) -> float | None:
    """Arabic minus English pass rate (percentage points) over items that have a twin in the other language."""
    groups: dict[str, list[ItemResult]] = defaultdict(list)
    for r in results:
        if r.item.pair_id:
            groups[r.item.pair_id].append(r)
    ar = [r for g in groups.values() if len(g) == 2 for r in g if r.item.language == "ar"]
    en = [r for g in groups.values() if len(g) == 2 for r in g if r.item.language == "en"]
    if not ar or not en:
        return None
    return ((rate(ar) or 0) - (rate(en) or 0)) * 100


def _fence(text: str, limit: int = 200) -> str:
    return " ".join(text.replace("```", "'''").split())[:limit]


def _cell(runs: list[ModelRun], pick) -> list[str]:
    return [pct(rate([r for r in run.results if pick(r.item)])) for run in runs]


def describe_dataset(items: list[Item]) -> str:
    langs, types = Counter(i.language for i in items), Counter(i.type for i in items)
    return (f"{len(items)} items (ar {langs['ar']}, en {langs['en']}) · mcq {types['mcq']}, "
            f"short {types['short']}, trap {types['trap']}")


def render_markdown(runs: list[ModelRun], items: list[Item]) -> str:
    out = ["# PharmaEval-AR Report", "", f"Dataset: {describe_dataset(items)}", "", "## Overall", "",
           "| Model | Pass rate (95% CI) | MCQ | Short | Trap | Arabic | English | AR-EN gap | Errors |",
           "|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        passed = sum(r.score.passed for r in run.results)
        lo, hi = wilson(passed, len(run.results))
        gap = paired_gap(run.results)
        by_type = [pct(rate([r for r in run.results if r.item.type == t])) for t in ("mcq", "short", "trap")]
        by_lang = [pct(rate([r for r in run.results if r.item.language == lang])) for lang in ("ar", "en")]
        out.append(f"| `{run.model}` | {pct(passed / len(run.results))} ({lo * 100:.0f}-{hi * 100:.0f}%) | "
                   f"{' | '.join(by_type)} | {' | '.join(by_lang)} | "
                   f"{'-' if gap is None else f'{gap:+.0f} pts'} | {run.errors} |")

    out += ["", "## By category", "", "| Category | " + " | ".join(f"`{r.model}`" for r in runs) + " |",
            "|---|" + "---|" * len(runs)]
    for cat in sorted({i.category for i in items}):
        out.append(f"| {cat} | " + " | ".join(_cell(runs, lambda it, c=cat: it.category == c)) + " |")

    failures = sorted({r.score.failure for run in runs for r in run.results if r.score.failure})
    if failures:
        out += ["", "## Failure types", "", "| Type | " + " | ".join(f"`{r.model}`" for r in runs) + " |",
                "|---|" + "---|" * len(runs)]
        for f in failures:
            out.append(f"| {f} | " + " | ".join(str(sum(r.score.failure == f for r in run.results)) for run in runs) + " |")

    for run in runs:
        failed = [r for r in run.results if not r.score.passed]
        if not failed:
            continue
        out += ["", f"## Failures: `{run.model}`", ""]
        for r in failed[:MAX_FAILURES_LISTED]:
            out += [f"- `{r.item.id}` ({r.score.failure}) {_fence(r.score.detail, 120)}", "  ```",
                    f"  {_fence(r.reply)}", "  ```"]
        if len(failed) > MAX_FAILURES_LISTED:
            out.append(f"- … and {len(failed) - MAX_FAILURES_LISTED} more (see the JSON output)")

    out += ["", "## Method and limitations", "",
            "- Deterministic scoring: MCQ by letter; short answers by required key facts; fictional-drug traps pass only "
            "if the model says it cannot verify the drug.",
            "- Keyword rubrics can miss correct answers phrased unusually, so review the failures list.",
            "- Small datasets give wide confidence intervals; compare models only when intervals do not overlap.",
            "- This measures knowledge on the dataset. It is not a safety certification and not medical advice.", ""]
    return "\n".join(out)


def render_json(runs: list[ModelRun], items: list[Item]) -> str:
    data = {
        "dataset": describe_dataset(items),
        "models": [
            {"model": run.model, "pass_rate": rate(run.results), "errors": run.errors,
             "results": [{"id": r.item.id, "passed": r.score.passed, "failure": r.score.failure,
                          "detail": r.score.detail, "reply": r.reply[:500]} for r in run.results]}
            for run in runs
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)
