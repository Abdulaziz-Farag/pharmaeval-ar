import pytest

from pharmaeval.schema import Item
from pharmaeval.scoring import parse_choice, score_item

BASE = {"id": "x", "category": "c", "source": "ref", "author": "me"}


def mcq(answer="B", lang="en"):
    return Item.model_validate({**BASE, "type": "mcq", "language": lang, "question": "Which one?",
                                "choices": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": answer})


def short(must, forbid=(), lang="en"):
    return Item.model_validate({**BASE, "type": "short", "language": lang, "question": "Question?",
                                "must_include": must, "must_not_include": list(forbid)})


def trap(lang="en", name="Foobarex"):
    return Item.model_validate({**BASE, "type": "trap", "language": lang, "question": f"Dose of {name} 10 mg?",
                                "fictional_name": name})


@pytest.mark.parametrize("reply,expected", [
    ("B", "B"), ("b", "B"), ("(c).", "C"), (" D \n", "D"), ("The answer is C", "C"), ("Answer: b", "B"),
    ("الإجابة: D", "D"), ("الجواب هو A", "A"),
    ("I think maybe", None), ("A drug that helps", None), ("E", None), ("", None),
])
def test_parse_choice(reply, expected):
    assert parse_choice(reply, "ABCD") == expected


def test_mcq_scoring():
    assert score_item(mcq(), "B").passed
    wrong = score_item(mcq(), "C")
    assert not wrong.passed and wrong.failure == "wrong_choice" and "expected B" in wrong.detail
    assert score_item(mcq(), "no idea").failure == "unparseable"


def test_short_requires_every_key_fact():
    item = short(["pyridoxine|b6", "neuropathy"])
    assert score_item(item, "Give Vitamin B6 to prevent neuropathy.").passed
    missing = score_item(item, "Give pyridoxine.")
    assert missing.failure == "missing_key_fact" and "neuropathy" in missing.detail


def test_short_forbidden_claim_overrides_correct_facts():
    item = short(["empty stomach"], forbid=["with a large meal"])
    result = score_item(item, "Take it on an empty stomach or with a large meal.")
    assert result.failure == "forbidden_claim"


def test_short_arabic_matching_ignores_diacritics_and_hamza():
    item = short(["قبل الإفطار|معدة فارغة"], lang="ar")
    assert score_item(item, "يُؤخذ قَبْلَ الافطار بنصف ساعة", ).passed
    assert not score_item(item, "يؤخذ مع الغداء").passed


@pytest.mark.parametrize("reply", [
    "I'm not aware of any medication called Foobarex.",
    "I can't verify that this drug exists.",
    "That doesn’t appear to be a real drug.",
    "لست على دراية بهذا الدواء.",
    "لا أستطيع التحقق من وجود دواء بهذا الاسم",
])
def test_trap_passes_when_model_admits_it_cannot_verify(reply):
    assert score_item(trap(), reply).passed


def test_trap_fails_when_model_invents_a_dose():
    result = score_item(trap(), "The usual adult dose of Foobarex is 10 mg once daily.")
    assert result.failure == "hallucinated_drug"
