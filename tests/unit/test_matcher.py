from __future__ import annotations

from pathlib import Path

import pytest

from xyq_quiz.knowledge.matcher import QuestionMatcher, normalize_text
from xyq_quiz.knowledge.store import QuestionBank


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "questions-small.json"


@pytest.fixture
def bank() -> QuestionBank:
    return QuestionBank.load(FIXTURE_PATH)


@pytest.fixture
def bank_with_near_duplicates(bank: QuestionBank) -> QuestionBank:
    return bank


def test_normalize_removes_prompt_punctuation_and_whitespace() -> None:
    assert normalize_text("Q： 梦幻西游中，有多少个种族？ ") == "梦幻西游中有多少个种族"


def test_normalize_applies_nfkc_casefold_and_removes_ocr_separators() -> None:
    assert normalize_text("Ａ： ＸＹＱ丨Test｜１２３") == "xyqtest123"


def test_exact_question_and_unique_option_are_high_confidence(
    bank: QuestionBank,
) -> None:
    matcher = QuestionMatcher(
        bank,
        question_score=92,
        question_gap=5,
        option_score=90,
    )

    question = matcher.match_question("梦幻西游中有多少个种族？")
    assert question is not None
    option = matcher.map_answer(question.record.answer, ["2", "3", "4", "5"])

    assert question.score == 100
    assert option is not None
    assert option.option_index == 1
    assert matcher.is_high_confidence(question, option)


def test_close_question_candidates_are_uncertain(
    bank_with_near_duplicates: QuestionBank,
) -> None:
    matcher = QuestionMatcher(bank_with_near_duplicates, 92, 5, 90)

    match = matcher.match_question("等级到多少能参加科举第二阶段")
    option = matcher.map_answer("30", ["20", "30", "40", "50"])

    assert match is not None
    assert match.score - match.runner_up_score < 5
    assert option is not None
    assert option.score >= 90
    assert matcher.is_high_confidence(match, option) is False


def test_duplicate_answer_options_never_map_uniquely(bank: QuestionBank) -> None:
    matcher = QuestionMatcher(bank, 92, 5, 90)

    assert matcher.map_answer("长安", ["长安", "长安", "建邺", "傲来"]) is None


def test_upstream_historical_name_typo_maps_to_correct_game_option(
    bank: QuestionBank,
) -> None:
    matcher = QuestionMatcher(bank, 92, 5, 90)

    option = matcher.map_answer("祖狄", ["赵括", "勾践", "李广", "祖逖"])

    assert option is not None
    assert option.option_index == 3
    assert option.score == 100


def test_answer_character_alias_still_rejects_ambiguous_options(
    bank: QuestionBank,
) -> None:
    matcher = QuestionMatcher(bank, 92, 5, 90)

    assert matcher.map_answer("祖狄", ["祖狄", "祖逖", "李广", "勾践"]) is None


def test_question_below_score_threshold_is_not_high_confidence(
    bank: QuestionBank,
) -> None:
    matcher = QuestionMatcher(bank, 92, 5, 90)

    question = matcher.match_question("完全无关的文字")
    option = matcher.map_answer("3", ["2", "3", "4", "5"])

    assert question is not None
    assert question.score < 92
    assert option is not None
    assert not matcher.is_high_confidence(question, option)


def test_option_below_score_threshold_does_not_map(bank: QuestionBank) -> None:
    matcher = QuestionMatcher(bank, 92, 5, 90)

    assert matcher.map_answer("长安", ["建邺", "傲来", "宝象", "西梁"]) is None


def test_empty_question_or_options_do_not_match(bank: QuestionBank) -> None:
    matcher = QuestionMatcher(bank, 92, 5, 90)

    assert matcher.match_question("？ ") is None
    assert matcher.map_answer("3", []) is None
