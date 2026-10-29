import json

import pytest
from pydantic import ValidationError

from pharmaeval.dataset import DatasetError, check_dataset, load_dataset, read_items
from pharmaeval.schema import Item

BASE = {"id": "x1", "category": "c", "language": "en", "source": "ref", "author": "me"}
MCQ = {**BASE, "type": "mcq", "question": "Which one?", "choices": {"A": "a", "B": "b"}, "answer": "B"}


def test_valid_items():
    assert Item.model_validate(MCQ).answer == "B"
    short = {**BASE, "type": "short", "question": "Question?", "must_include": ["x|y"]}
    assert Item.model_validate(short)
    trap = {**BASE, "type": "trap", "question": "Dose of Foobarex?", "fictional_name": "foobarex"}
    assert Item.model_validate(trap)


@pytest.mark.parametrize("patch", [
    {"answer": "C"},
    {"choices": {"A": "a", "C": "c"}},
    {"choices": {"A": "a"}},
    {"type": "short", "must_include": []},
    {"type": "trap", "fictional_name": "Nothere"},
    {"type": "trap"},
    {"language": "fr"},
])
def test_invalid_items_are_rejected(patch):
    with pytest.raises(ValidationError):
        Item.model_validate({**MCQ, **patch})


def write(tmp_path, rows, raw=None):
    path = tmp_path / "d.jsonl"
    path.write_text(raw if raw is not None else "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def test_sample_dataset_is_valid():
    items = load_dataset("data/sample_questions.jsonl")
    assert len(items) == 12 and {i.language for i in items} == {"ar", "en"}
    assert len({i.answer for i in items if i.type == "mcq"}) > 1  # answer positions are not all the same letter


def test_parse_errors_report_line_numbers(tmp_path):
    bad = json.dumps(MCQ) + "\nnot json\n" + json.dumps({**MCQ, "id": "x2", "answer": "Z"}) + "\n"
    items, problems = read_items(write(tmp_path, [], raw=bad))
    assert len(items) == 1
    assert problems[0].startswith("line 2: invalid JSON") and problems[1].startswith("line 3: ")


def test_duplicate_ids_and_bad_pairs():
    a = Item.model_validate({**MCQ, "pair_id": "p"})
    problems = check_dataset([a, a])
    assert any("duplicate id" in p for p in problems) and any("exactly one ar and one en" in p for p in problems)
    ar = Item.model_validate({**MCQ, "id": "x2", "language": "ar", "category": "other", "pair_id": "p"})
    assert any("share type and category" in p for p in check_dataset([a, ar]))


def test_load_dataset_raises_on_problems_and_empty(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(write(tmp_path, [], raw="oops\n"))
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(write(tmp_path, [], raw="\n"))
