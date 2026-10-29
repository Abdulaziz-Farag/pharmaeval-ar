import json

import pytest

from pharmaeval.adapters import MockAdapter
from pharmaeval.cli import main
from pharmaeval.dataset import load_dataset
from pharmaeval.report import paired_gap, render_json, render_markdown, wilson
from pharmaeval.runner import ItemResult, ModelRun, run_model
from pharmaeval.scoring import Score

ITEMS = load_dataset("data/sample_questions.jsonl")


def runs():
    return [run_model(MockAdapter(k, ITEMS), ITEMS) for k in ("perfect", "careless")]


def test_wilson_interval_behaves():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(10, 10)
    assert 0.7 < lo < 1.0 and hi == pytest.approx(1.0)
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi and (hi - lo) > 0.4  # tiny samples have wide intervals


def test_paired_gap_measures_arabic_minus_english():
    def result(item, passed):
        return ItemResult(item, "", Score(passed))

    ar = next(i for i in ITEMS if i.pair_id == "p2" and i.language == "ar")
    en = next(i for i in ITEMS if i.pair_id == "p2" and i.language == "en")
    assert paired_gap([result(ar, False), result(en, True)]) == -100
    assert paired_gap([result(ar, True), result(en, True)]) == 0
    assert paired_gap([result(ar, True)]) is None


def test_markdown_report_contents():
    md = render_markdown(runs(), ITEMS)
    for text in ("# PharmaEval-AR Report", "`mock:perfect`", "`mock:careless`", "100%", "AR-EN gap", "## By category",
                 "hallucinated_drug", "Method and limitations", "12 items (ar 6, en 6)"):
        assert text in md


def test_markdown_neutralizes_model_output():
    item = ITEMS[0]
    bad = ModelRun("evil", [ItemResult(item, "```\n# pwned\n```", Score(False, "unparseable", "```x"))])
    md = render_markdown([bad], ITEMS)
    assert "```\n# pwned" not in md and "'''" in md


def test_failure_list_is_capped():
    many = [ItemResult(ITEMS[0], "x", Score(False, "wrong_choice", "d")) for _ in range(30)]
    assert "and 15 more" in render_markdown([ModelRun("m", many)], ITEMS)


def test_json_report():
    data = json.loads(render_json(runs(), ITEMS))
    assert data["models"][0]["pass_rate"] == 1.0 and len(data["models"][1]["results"]) == 12


def test_cli_validate(tmp_path, capsys):
    assert main(["validate"]) == 0 and "OK" in capsys.readouterr().out
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    assert main(["validate", str(bad)]) == 1 and "ERROR line 1" in capsys.readouterr().out
    assert main(["validate", str(tmp_path / "missing.jsonl")]) == 2


def test_cli_run_with_mock_models_writes_reports(tmp_path, capsys):
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    code = main(["run", "--model", "mock:perfect", "--model", "mock:careless", "--out", str(md), "--json", str(js)])
    assert code == 0
    assert "mock:perfect: pass rate 100%" in capsys.readouterr().out
    assert md.read_text().startswith("# PharmaEval-AR Report") and json.loads(js.read_text())["models"]


def test_cli_run_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["run", "--model", "anthropic:m"]) == 2
    assert main(["run", "--model", "mock:perfect", "--data", str(tmp_path / "nope.jsonl")]) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_limit(capsys):
    main(["run", "--model", "mock:careless", "--limit", "2"])
    assert "pass rate" in capsys.readouterr().out
