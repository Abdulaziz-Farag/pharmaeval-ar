from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from pharmaeval.textnorm import normalize


class Item(BaseModel):
    """One evaluation question. ``mcq``: pick a letter. ``short``: must mention key facts.
    ``trap``: asks about a fictional drug; the correct behaviour is to say it cannot be verified."""

    id: str = Field(min_length=1)
    type: Literal["mcq", "short", "trap"]
    category: str = Field(min_length=1)
    language: Literal["ar", "en"]
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question: str = Field(min_length=5)
    choices: dict[str, str] | None = None
    answer: str | None = None
    must_include: list[str] = Field(default_factory=list)  # each entry: alternatives separated by "|"
    must_not_include: list[str] = Field(default_factory=list)
    fictional_name: str | None = None
    pair_id: str | None = None  # links an Arabic item to its English twin
    source: str = Field(min_length=3)
    author: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_type_fields(self) -> Item:
        if self.type == "mcq":
            if not self.choices or not 2 <= len(self.choices) <= 6:
                raise ValueError("mcq needs 2 to 6 choices")
            expected = list("ABCDEF"[: len(self.choices)])
            if list(self.choices) != expected:
                raise ValueError(f"choice keys must be {expected}")
            if self.answer not in self.choices:
                raise ValueError("answer must be one of the choice keys")
        elif self.type == "short":
            if not self.must_include or any(not g.strip() for g in self.must_include):
                raise ValueError("short items need a non-empty must_include list")
        else:
            if not self.fictional_name:
                raise ValueError("trap items need fictional_name")
            if normalize(self.fictional_name) not in normalize(self.question):
                raise ValueError("fictional_name must appear in the question")
        return self
