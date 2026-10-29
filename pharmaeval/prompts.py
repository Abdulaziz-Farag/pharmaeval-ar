from __future__ import annotations

from pharmaeval.schema import Item


def build_prompt(item: Item) -> str:
    ar = item.language == "ar"
    if item.type == "mcq":
        assert item.choices is not None
        options = [f"{key}) {text}" for key, text in item.choices.items()]
        hint = ("أجب بحرف الخيار الصحيح فقط (مثال: A)." if ar
                else "Answer with only the letter of the correct option (e.g. A).")
        return "\n".join([item.question, "", *options, "", hint])
    if item.type == "short":
        return f"{item.question}\n\n{'أجب باختصار.' if ar else 'Answer concisely.'}"
    return item.question  # trap: ask naturally, with no hint that the drug may not exist
