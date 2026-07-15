from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


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


class QuestionOrigin(StrEnum):
    OFFICIAL = "official"
    LOCAL_SUPPLEMENT = "local_supplement"
    LOCAL_OVERRIDE = "local_override"


@dataclass(frozen=True, slots=True)
class QuestionRecordMetadata:
    origin: QuestionOrigin = QuestionOrigin.OFFICIAL
    local_id: str | None = None
    target_source_id: str | None = None
    answer_aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QuestionMatch:
    score: float
    runner_up_score: float
    record: QuestionRecord
    alternative_records: tuple[QuestionRecord, ...] = ()
    query_variant: str = "canonical"

    @property
    def candidate_records(self) -> tuple[QuestionRecord, ...]:
        return (self.record, *self.alternative_records)


@dataclass(frozen=True, slots=True)
class RankedQuestionCandidate:
    score: float
    normalized_question: str
    records: tuple[QuestionRecord, ...]
    query_variant: str = "canonical"


@dataclass(frozen=True, slots=True)
class OptionMatch:
    score: float
    runner_up_score: float
    option_index: int
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionOptionCandidate:
    record: QuestionRecord
    option: OptionMatch
