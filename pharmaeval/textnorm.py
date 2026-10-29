from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})


def normalize(text: str) -> str:
    """Canonical form for matching: strips tatweel, diacritics, invisible chars; folds Arabic letter variants."""
    text = unicodedata.normalize("NFKC", text).replace("\u0640", "")
    text = _DIACRITICS.sub("", _INVISIBLE.sub("", text))
    return text.translate(_FOLD).casefold()
