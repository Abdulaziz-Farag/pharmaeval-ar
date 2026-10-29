# PharmaEval-AR

[![CI](https://github.com/Abdulaziz-Farag/pharmaeval-ar/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdulaziz-Farag/pharmaeval-ar/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An evaluation harness that measures how well LLMs know **pharmacy** in **Arabic and English**, including a test that
most benchmarks skip: asking about a **drug that does not exist** to see whether the model admits it cannot verify it
or invents a dose.

Created by a pharmacist who writes code. The harness is the easy part; the value of a benchmark is the quality of its
questions, so the dataset format is built for expert-written, reviewed items.

> **Status: v0.1.** `data/sample_questions.jsonl` has 12 *sample* items (6 Arabic/English twins) that demonstrate the
> format. They are **not** a benchmark. Replace them with your own reviewed questions (target: 100+) before drawing
> conclusions or publishing results.

## Quick start (offline, no API key)

```bash
pip install -e ".[dev]"
pharmaeval validate
pharmaeval run --model mock:perfect --model mock:careless --out report.md
```

```text
mock:perfect: pass rate 100% (0 errors)
mock:careless: pass rate 17% (0 errors)
Markdown report: /tmp/pe_report.md
```

Real models (calls the provider API, costs money, use `--limit 5` first and `--cache cache.jsonl` to avoid paying twice):

```bash
export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY (OPENAI_BASE_URL for compatible servers)
pharmaeval run --model anthropic:<model-name> --model openai:<model-name> --cache cache.jsonl --out report.md
```

## What the report shows

- Pass rate with a **95% Wilson confidence interval** (small datasets need honest error bars)
- Breakdown by question type, category and language
- **AR-EN gap**: pass-rate difference on paired Arabic/English twins of the same question
- Failure types (`wrong_choice`, `missing_key_fact`, `forbidden_claim`, `hallucinated_drug`, `unparseable`) and the failing replies

Example (`mock:careless`, which picks "A" and fabricates doses):

| Model | Pass rate (95% CI) | MCQ | Short | Trap | Arabic | English | AR-EN gap | Errors |
|---|---|---|---|---|---|---|---|---|
| `mock:careless` | 17% (5-45%) | 33% | 0% | 0% | 17% | 17% | +0 pts | 0 |

## Question types

| Type | Passes when |
|---|---|
| `mcq` | the model answers with the correct letter (A-D) |
| `short` | the reply contains every required key fact (alternatives separated by `\|`, Arabic-normalized) and no forbidden claim |
| `trap` | the model says it is unfamiliar with / cannot verify a **fictional** drug instead of inventing details |

## Writing your own questions

One JSON object per line (see the sample file):

```json
{"id": "interaction-001-ar", "type": "mcq", "category": "interactions", "language": "ar", "difficulty": "medium",
 "question": "...", "choices": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "C",
 "pair_id": "interaction-001", "source": "BNF 2025, section ...", "author": "your name"}
```

`pharmaeval validate my_questions.jsonl` checks the schema, unique ids, answer keys, that fictional drug names appear
in their question, and that every `pair_id` has exactly one Arabic and one English twin of the same type and category.
Always record a real `source`, vary the position of the correct answer, and **verify that fictional names do not exist**.

## Design decisions

- **Deterministic scoring.** Same reply, same score. No LLM judge in v0.1, so results are reproducible and auditable.
- **Arabic-aware matching.** Replies and rubrics are normalized (diacritics, tatweel, hamza/alef, ya/ta-marbuta, invisible characters).
- **Hallucination traps** test calibration, not just recall: a confident invented dose is the dangerous failure.
- **No vendor SDKs.** Plain HTTPS adapters with retry/backoff on 429/5xx, easy to extend and to test offline.
- **Cost control.** `--limit`, a response cache, and per-call error capture so one failure never discards a paid run.
- **Tested.** Scoring edge cases, dataset validation, adapters (mocked HTTP), caching, reports and CLI.

## Limitations

- Keyword rubrics can reject a correct answer phrased unusually; read the failure list.
- Trap detection looks for admission phrases in English and Arabic, so it can be fooled by unusual wording.
- Adapters were tested with mocked HTTP only. Provider-specific parameters (for example newer reasoning models that
  reject `temperature` or `max_tokens`) may need small adjustments.
- Measures knowledge on the dataset only. It is not a clinical safety certification and not medical advice.

## Roadmap

- [ ] Optional LLM-as-judge for free-text answers, with agreement checks against human grading
- [ ] Dose-calculation items with numeric tolerance scoring
- [ ] Gulf / Levantine / Egyptian dialect question variants
- [ ] Publish a reviewed dataset (100+ items) on Hugging Face

## Development

```bash
pip install -e ".[dev]" && ruff check . && pytest
```

## License

MIT (code). Dataset licensing should be chosen by the question author(s).
