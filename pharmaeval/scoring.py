"""Deterministic scoring: same reply, same score, so runs are reproducible and auditable."""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from pharmaeval.schema import Item
from pharmaeval.textnorm import normalize


@dataclass(frozen=True)
class Score:
    passed: bool
    failure: str | None = None  # wrong_choice | unparseable | missing_key_fact | forbidden_claim | hallucinated_drug | error
    detail: str = ""


_ADMISSION_SRC = (
    r"not (aware|familiar)|no (such|known|record|reliable)"
    r"|(cannot|can['’]?t|couldn['’]?t|could not|unable to|don['’]?t|do not)"
    r" (find|verify|identify|confirm|recogni[sz]e|locate)"
    r"|(does not|doesn['’]?t) (appear|exist)|not a (recogni[sz]ed|known|real)|fictional|made[- ]up"
    r"|لا اعرف|لا اعلم|لا املك معلومات|لست على دراية|غير معروف|غير موجود|لا يوجد دواء"
    r"|لا استطيع التحقق|لم اجد|لا اتعرف|ليس دواء معروف"
)
_ADMISSION = re.compile(normalize(_ADMISSION_SRC))
_WORDS = normalize("answer|option|الإجابة|الجواب|الخيار")


def parse_choice(reply: str, valid: Iterable[str]) -> str | None:
    """Accepts a bare letter ("B", "(b).") or an explicit statement ("The answer is C", "الإجابة: D")."""
    allowed = {v.upper() for v in valid}
    text = reply.strip()
    match = re.fullmatch(r"[\W_]*\(?([A-Fa-f])\)?[\W_]*", text)
    if not match:
        match = re.search(rf"(?:{_WORDS})\s*(?:is|هي|هو|:|=|-)?\s*\(?([a-f])\)?(?![a-z])", normalize(text))
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()
    return None


def _snippet(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def score_mcq(item: Item, reply: str) -> Score:
    assert item.choices is not None
    choice = parse_choice(reply, item.choices)
    if choice is None:
        return Score(False, "unparseable", _snippet(reply))
    if choice == item.answer:
        return Score(True)
    return Score(False, "wrong_choice", f"chose {choice}, expected {item.answer}")


def score_short(item: Item, reply: str) -> Score:
    norm = normalize(reply)
    forbidden = [p for p in item.must_not_include if normalize(p) in norm]
    if forbidden:
        return Score(False, "forbidden_claim", "; ".join(forbidden))
    missing = []
    for group in item.must_include:
        alternatives = [normalize(a.strip()) for a in group.split("|") if a.strip()]
        if not any(a in norm for a in alternatives):
            missing.append(group)
    if missing:
        return Score(False, "missing_key_fact", "missing: " + " ; ".join(missing))
    return Score(True)


def score_trap(item: Item, reply: str) -> Score:
    if _ADMISSION.search(normalize(reply)):
        return Score(True)
    return Score(False, "hallucinated_drug", _snippet(reply))


def score_item(item: Item, reply: str) -> Score:
    if item.type == "mcq":
        return score_mcq(item, reply)
    if item.type == "short":
        return score_short(item, reply)
    return score_trap(item, reply)
