from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz import fuzz, process

from xyq_quiz.knowledge.models import (
    OptionMatch,
    QuestionMatch,
    normalize_text,
)
from xyq_quiz.knowledge.store import QuestionBank


# The upstream quiz bank occasionally contains a visually similar wrong form
# while the in-game option uses the correct historical character. Apply these
# aliases only when mapping answers to options so question indexing is untouched.
_ANSWER_CHARACTER_ALIASES = str.maketrans({"狄": "逖"})


def _normalize_answer_choice(text: str) -> str:
    return normalize_text(text).translate(_ANSWER_CHARACTER_ALIASES)


class QuestionMatcher:
    def __init__(
        self,
        bank: QuestionBank,
        question_score: int,
        question_gap: int,
        option_score: int,
    ) -> None:
        self._bank = bank
        self._question_score = question_score
        self._question_gap = question_gap
        self._option_score = option_score

    def match_question(self, text: str) -> QuestionMatch | None:
        normalized = normalize_text(text)
        if not normalized:
            return None

        ranked = process.extract(
            normalized,
            self._bank.normalized_questions,
            scorer=fuzz.ratio,
            limit=2,
            score_cutoff=0,
        )
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

        exact_record = self._bank.exact.get(normalized)
        if exact_record is not None:
            return QuestionMatch(100.0, runner_up_score, exact_record)

        _, score, record_index = ranked[0]
        return QuestionMatch(
            score,
            runner_up_score,
            self._bank.records[record_index],
        )

    def map_answer(
        self,
        answer: str,
        options: Sequence[str],
    ) -> OptionMatch | None:
        normalized_answer = _normalize_answer_choice(answer)
        if not normalized_answer or not options:
            return None

        normalized_options = [_normalize_answer_choice(option) for option in options]
        ranked = process.extract(
            normalized_answer,
            normalized_options,
            scorer=fuzz.ratio,
            limit=2,
            score_cutoff=0,
        )
        _, score, option_index = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if score < self._option_score or score == runner_up_score:
            return None

        return OptionMatch(score, runner_up_score, option_index)

    def is_high_confidence(
        self,
        question: QuestionMatch | None,
        option: OptionMatch | None,
    ) -> bool:
        if question is None or option is None:
            return False

        return (
            question.score >= self._question_score
            and question.score - question.runner_up_score >= self._question_gap
            and option.score >= self._option_score
            and option.score > option.runner_up_score
        )


__all__ = ["QuestionMatcher", "normalize_text"]
