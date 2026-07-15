from __future__ import annotations

from collections.abc import Sequence
import re

from rapidfuzz import fuzz, process

from xyq_quiz.knowledge.models import (
    OptionMatch,
    QuestionMatch,
    QuestionOptionCandidate,
    RankedQuestionCandidate,
    normalize_text,
)
from xyq_quiz.knowledge.store import QuestionBank


# The upstream quiz bank occasionally contains a visually similar wrong form
# while the in-game option uses the correct historical character. Apply these
# aliases only when mapping answers to options so question indexing is untouched.
_ANSWER_CHARACTER_ALIASES = str.maketrans({"狄": "逖"})
_QUESTION_BODY_PATTERNS = (
    re.compile(
        r"第\s*\d+\s*关\s*[:：].{0,80}?题目\s*[:：]\s*(?P<body>.+)$",
        re.DOTALL,
    ),
    re.compile(
        r"^\s*(?:科举大赛|科举|乡试|会试|殿试).{0,80}?"
        r"题目\s*[:：]\s*(?P<body>.+)$",
        re.DOTALL,
    ),
)
_MINIMUM_PARTIAL_QUERY_LENGTH = 6
_MINIMUM_PARTIAL_COVERAGE = 0.55


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
        # The normal OCR path usually produces the exact normalized bank key.
        # Avoid two full RapidFuzz scans in that overwhelmingly common case;
        # tolerant ranking remains available for damaged/truncated OCR text.
        exact_candidates: list[tuple[str, str]] = []
        for query_variant, normalized in _query_variants(text):
            if normalized in self._bank.groups and all(
                existing != normalized for existing, _ in exact_candidates
            ):
                exact_candidates.append((normalized, query_variant))
        if exact_candidates:
            normalized, query_variant = exact_candidates[0]
            records = self._bank.records_for(normalized)
            runner_up_score = 100.0 if len(exact_candidates) > 1 else 0.0
            return QuestionMatch(
                100.0,
                runner_up_score,
                records[0],
                records[1:],
                query_variant,
            )

        ranked = self.rank_question_candidates(text, limit=2)
        if not ranked:
            return None

        candidate = ranked[0]
        runner_up_score = ranked[1].score if len(ranked) > 1 else 0.0
        return QuestionMatch(
            candidate.score,
            runner_up_score,
            candidate.records[0],
            candidate.records[1:],
            candidate.query_variant,
        )

    def rank_question_candidates(
        self,
        text: str,
        *,
        limit: int = 5,
        score_cutoff: float = 0.0,
    ) -> tuple[RankedQuestionCandidate, ...]:
        """Rank unique question texts while retaining every answer record.

        Partial matching is deliberately capped below the configured high-confidence
        threshold. It can therefore recover one low-confidence candidate without
        silently turning a truncated OCR string into a high-confidence answer.
        """

        if limit <= 0:
            return ()
        variants = _query_variants(text)
        if not variants or not self._bank.normalized_questions:
            return ()

        choices = self._bank.normalized_questions
        pool_size = min(len(choices), max(32, limit * 8))
        scores: dict[int, tuple[float, str]] = {}

        for query_variant, normalized in variants:
            ratio_ranked = process.extract(
                normalized,
                choices,
                scorer=fuzz.ratio,
                limit=pool_size,
                score_cutoff=0,
            )
            for _, score, index in ratio_ranked:
                _keep_best_score(scores, index, score, query_variant)

            if len(normalized) < _MINIMUM_PARTIAL_QUERY_LENGTH:
                continue
            partial_ranked = process.extract(
                normalized,
                choices,
                scorer=fuzz.partial_ratio,
                limit=pool_size,
                score_cutoff=0,
            )
            for candidate, partial_score, index in partial_ranked:
                coverage = min(len(normalized), len(candidate)) / max(
                    len(normalized),
                    len(candidate),
                )
                if coverage < _MINIMUM_PARTIAL_COVERAGE:
                    continue
                tolerant_score = partial_score * (0.65 + 0.35 * coverage)
                tolerant_score = min(
                    tolerant_score,
                    max(0.0, float(self._question_score) - 0.01),
                )
                _keep_best_score(
                    scores,
                    index,
                    tolerant_score,
                    f"{query_variant}_partial",
                )

        ranked_indices = sorted(scores, key=lambda index: (-scores[index][0], index))
        candidates: list[RankedQuestionCandidate] = []
        for index in ranked_indices:
            score, query_variant = scores[index]
            if score < score_cutoff:
                continue
            normalized_question = choices[index]
            records = self._bank.records_for(normalized_question)
            candidates.append(
                RankedQuestionCandidate(
                    score=score,
                    normalized_question=normalized_question,
                    records=records,
                    query_variant=query_variant,
                )
            )
            if len(candidates) >= limit:
                break
        return tuple(candidates)

    def unique_question_candidate(
        self,
        text: str,
        *,
        score_cutoff: float,
        minimum_gap: float = 0.0,
    ) -> QuestionMatch | None:
        """Return a uniquely ranked candidate, including below the red threshold."""

        ranked = self.rank_question_candidates(
            text,
            limit=2,
            score_cutoff=score_cutoff,
        )
        if not ranked:
            return None
        runner_up_score = ranked[1].score if len(ranked) > 1 else 0.0
        if ranked[0].score - runner_up_score < minimum_gap:
            return None
        candidate = ranked[0]
        return QuestionMatch(
            candidate.score,
            runner_up_score,
            candidate.records[0],
            candidate.records[1:],
            candidate.query_variant,
        )

    def map_answer(
        self,
        answer: str,
        options: Sequence[str],
    ) -> OptionMatch | None:
        match = self._rank_answer(answer, options)
        if match is None:
            return None
        if match.score < self._option_score or match.score == match.runner_up_score:
            return None
        return match

    def rank_option_candidates(
        self,
        question: QuestionMatch,
        options: Sequence[str],
        *,
        score_cutoff: float | None = None,
    ) -> tuple[QuestionOptionCandidate, ...]:
        """Map every answer attached to one question group onto current options."""

        cutoff = self._option_score if score_cutoff is None else score_cutoff
        candidates: list[QuestionOptionCandidate] = []
        for record in question.candidate_records:
            mapped = self._map_record_answer(record.source_id, record.answer, options, cutoff)
            if mapped is None:
                continue
            candidates.append(QuestionOptionCandidate(record=record, option=mapped))
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.option.score,
                    candidate.option.option_index,
                    candidate.record.source_id,
                ),
            )
        )

    def unique_option_candidate(
        self,
        question: QuestionMatch,
        options: Sequence[str],
        *,
        score_cutoff: float | None = None,
        minimum_gap: float = 0.0,
    ) -> OptionMatch | None:
        """Resolve all same-question answers only when they select one option."""

        candidates = tuple(
            candidate
            for candidate in self.rank_option_candidates(
                question,
                options,
                score_cutoff=score_cutoff,
            )
            if candidate.option.score - candidate.option.runner_up_score >= minimum_gap
        )
        if not candidates:
            return None
        option_indexes = {candidate.option.option_index for candidate in candidates}
        if len(option_indexes) != 1:
            return None
        return candidates[0].option

    def is_high_confidence(
        self,
        question: QuestionMatch | None,
        option: OptionMatch | None,
    ) -> bool:
        if question is None or option is None:
            return False

        distinct_answers = {
            _normalize_answer_choice(record.answer)
            for record in question.candidate_records
        }
        if len(distinct_answers) > 1 and option.source_id not in {
            record.source_id for record in question.candidate_records
        }:
            return False

        return (
            question.score >= self._question_score
            and question.score - question.runner_up_score >= self._question_gap
            and option.score >= self._option_score
            and option.score > option.runner_up_score
        )

    def _rank_answer(
        self,
        answer: str,
        options: Sequence[str],
        *,
        source_id: str | None = None,
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
        if not ranked:
            return None
        _, score, option_index = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        return OptionMatch(score, runner_up_score, option_index, source_id)

    def _map_record_answer(
        self,
        source_id: str,
        answer: str,
        options: Sequence[str],
        score_cutoff: float,
    ) -> OptionMatch | None:
        metadata = self._bank.metadata_for(source_id)
        full_match = self._rank_answer(answer, options, source_id=source_id)
        if (
            full_match is not None
            and full_match.score >= score_cutoff
            and full_match.score > full_match.runner_up_score
        ):
            return full_match

        explicit_matches = [
            match
            for alias in metadata.answer_aliases
            if (
                match := self._rank_answer(alias, options, source_id=source_id)
            )
            is not None
            and match.score >= score_cutoff
            and match.score > match.runner_up_score
        ]
        if explicit_matches:
            if len({match.option_index for match in explicit_matches}) != 1:
                return None
            return max(explicit_matches, key=lambda match: match.score)

        # Compatibility fallback for legacy rows that encoded aliases with an
        # ASCII comma. The complete answer is always tried first so compound
        # answers such as "3,36" and English sentences remain matchable.
        legacy_aliases = [part.strip() for part in answer.split(",") if part.strip()]
        if len(legacy_aliases) < 2:
            return None
        legacy_matches = [
            match
            for alias in legacy_aliases
            if (
                match := self._rank_answer(alias, options, source_id=source_id)
            )
            is not None
            and match.score >= score_cutoff
            and match.score > match.runner_up_score
        ]
        if not legacy_matches or len(
            {match.option_index for match in legacy_matches}
        ) != 1:
            return None
        return max(legacy_matches, key=lambda match: match.score)


def _query_variants(text: str) -> tuple[tuple[str, str], ...]:
    variants: list[tuple[str, str]] = []
    canonical = normalize_text(text)
    if canonical:
        variants.append(("canonical", canonical))
    for pattern in _QUESTION_BODY_PATTERNS:
        match = pattern.search(text[:400])
        if match is None:
            continue
        body = normalize_text(match.group("body"))
        if body and body != canonical:
            variants.append(("prompt_body", body))
        break
    return tuple(variants)


def _keep_best_score(
    scores: dict[int, tuple[float, str]],
    index: int,
    score: float,
    query_variant: str,
) -> None:
    current = scores.get(index)
    if current is None or score > current[0]:
        scores[index] = (score, query_variant)


__all__ = ["QuestionMatcher", "normalize_text"]
