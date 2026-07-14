from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_PROMPT_PREFIX = re.compile(r"^\s*[qa]\s*:\s*")
_OCR_SEPARATORS = frozenset("|｜丨¦‖")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _PROMPT_PREFIX.sub("", normalized, count=1)
    return "".join(
        character
        for character in normalized
        if character.isalnum() and character not in _OCR_SEPARATORS
    )


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    source_id: str
    question: str
    answer: str
    normalized_question: str


@dataclass(frozen=True, slots=True)
class QuestionMatch:
    score: float
    runner_up_score: float
    record: QuestionRecord


@dataclass(frozen=True, slots=True)
class OptionMatch:
    score: float
    runner_up_score: float
    option_index: int
