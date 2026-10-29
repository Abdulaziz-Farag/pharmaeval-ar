from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from pharmaeval import __version__
from pharmaeval.adapters import ConfigError, make_adapter
from pharmaeval.dataset import DatasetError, check_dataset, load_dataset, read_items
from pharmaeval.report import pct, rate, render_json, render_markdown
from pharmaeval.runner import ResponseCache, run_model

DEFAULT_DATA = "data/sample_questions.jsonl"


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        items, problems = read_items(args.data)
    except OSError as exc:
        print(f"Cannot read dataset: {exc}", file=sys.stderr)
        return 2
    problems += check_dataset(items)
    for p in problems:
        print(f"ERROR {p}")
    langs, types, cats = (Counter(getattr(i, f) for i in items) for f in ("language", "type", "category"))
    print(f"{len(items)} valid items | languages {dict(langs)} | types {dict(types)} | categories {dict(cats)}")
    print("OK" if not problems else f"{len(problems)} problem(s) found")
    return 1 if problems else 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        items = load_dataset(args.data)
    except (OSError, DatasetError) as exc:
        print(f"Dataset error:\n{exc}", file=sys.stderr)
        return 2
    if args.limit:
        items = items[: args.limit]
    cache = ResponseCache(args.cache) if args.cache else None
    runs = []
    try:
        for spec in args.model:
            adapter = make_adapter(spec, items, max_tokens=args.max_tokens, temperature=args.temperature,
                                   system=args.system)
            run = run_model(adapter, items, delay=args.delay, cache=cache)
            runs.append(run)
            print(f"{run.model}: pass rate {pct(rate(run.results))} ({run.errors} errors)")
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).write_text(render_markdown(runs, items), encoding="utf-8")
        print(f"Markdown report: {args.out}")
    if args.json:
        Path(args.json).write_text(render_json(runs, items), encoding="utf-8")
        print(f"JSON results: {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pharmaeval", description="Arabic pharmacy benchmark for LLMs.")
    parser.add_argument("--version", action="version", version=f"pharmaeval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    val = sub.add_parser("validate", help="check a dataset file for schema and pairing problems")
    val.add_argument("data", nargs="?", default=DEFAULT_DATA)
    val.set_defaults(func=_cmd_validate)

    run = sub.add_parser("run", help="evaluate one or more models")
    run.add_argument("--data", default=DEFAULT_DATA)
    run.add_argument("--model", action="append", required=True,
                     help="provider:model, e.g. anthropic:<model>, openai:<model>, mock:perfect, mock:careless")
    run.add_argument("--out", help="write a Markdown report")
    run.add_argument("--json", help="write detailed JSON results")
    run.add_argument("--cache", help="JSONL file that caches model replies between runs")
    run.add_argument("--delay", type=float, default=0.0, help="seconds between API calls")
    run.add_argument("--limit", type=int, help="only use the first N items (cheap smoke test)")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=300)
    run.add_argument("--system", help="optional system prompt")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
