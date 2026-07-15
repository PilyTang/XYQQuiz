from __future__ import annotations

from pathlib import Path

import pytest

from xyq_quiz.knowledge.matcher import QuestionMatcher, normalize_text
from xyq_quiz.knowledge.models import QuestionRecord, QuestionRecordMetadata
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


def test_duplicate_question_rows_are_one_ranked_candidate() -> None:
    bank = QuestionBank(
        [
            QuestionRecord("1", "重复题目", "答案甲", "重复题目"),
            QuestionRecord("2", "重复题目？", "答案乙", "重复题目"),
            QuestionRecord("3", "完全不同的问题", "答案丙", "完全不同的问题"),
        ]
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)

    match = matcher.match_question("重复题目")

    assert match is not None
    assert match.score == 100
    assert match.runner_up_score < 100
    assert [record.source_id for record in match.candidate_records] == ["1", "2"]
    assert len(matcher.rank_question_candidates("重复题目")) == 2


def test_same_question_different_answers_are_resolved_by_current_options() -> None:
    bank = QuestionBank(
        [
            QuestionRecord("1", "骨精灵可以加入哪个门派", "魔王寨", "骨精灵可以加入哪个门派"),
            QuestionRecord("2", "骨精灵可以加入哪个门派", "阴曹地府", "骨精灵可以加入哪个门派"),
            QuestionRecord("3", "另一题", "其他", "另一题"),
        ]
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)
    question = matcher.match_question("骨精灵可以加入哪个门派")
    assert question is not None

    resolved = matcher.unique_option_candidate(
        question,
        ["化生寺", "阴曹地府", "方寸山", "女儿村"],
    )

    assert resolved is not None
    assert resolved.option_index == 1
    assert resolved.source_id == "2"
    assert matcher.is_high_confidence(question, resolved)


def test_same_question_answers_mapping_to_different_options_stay_uncertain() -> None:
    bank = QuestionBank(
        [
            QuestionRecord("1", "骨精灵可以加入哪个门派", "魔王寨", "骨精灵可以加入哪个门派"),
            QuestionRecord("2", "骨精灵可以加入哪个门派", "阴曹地府", "骨精灵可以加入哪个门派"),
        ]
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)
    question = matcher.match_question("骨精灵可以加入哪个门派")
    assert question is not None

    assert matcher.unique_option_candidate(
        question,
        ["魔王寨", "阴曹地府", "方寸山", "女儿村"],
    ) is None
    legacy_option = matcher.map_answer("魔王寨", ["魔王寨", "化生寺", "方寸山", "女儿村"])
    assert legacy_option is not None
    assert matcher.is_high_confidence(question, legacy_option) is False


def test_duplicate_rows_with_same_answer_can_still_be_high_confidence() -> None:
    bank = QuestionBank(
        [
            QuestionRecord("1", "十面埋伏由哪种乐器演奏", "琵琶", "十面埋伏由哪种乐器演奏"),
            QuestionRecord("2", "十面埋伏由哪种乐器演奏？", "琵琶", "十面埋伏由哪种乐器演奏"),
            QuestionRecord("3", "另一道题", "古琴", "另一道题"),
        ]
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)
    question = matcher.match_question("十面埋伏由哪种乐器演奏")
    assert question is not None
    option = matcher.map_answer("琵琶", ["古琴", "琵琶", "二胡", "笛子"])

    assert option is not None
    assert matcher.is_high_confidence(question, option)


def test_clear_exam_prompt_is_removed_on_query_side_only() -> None:
    bank = QuestionBank(
        [
            QuestionRecord(
                "1",
                "哪个时期国家设立五经博士",
                "汉武帝时期",
                "哪个时期国家设立五经博士",
            )
        ]
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)

    match = matcher.match_question(
        "科举大赛殿试部分第2关：这一关考的是历史知识。题目：哪个时期，国家设立五经博士？"
    )

    assert match is not None
    assert match.score == 100
    assert match.query_variant == "prompt_body"


def test_truncated_ocr_can_be_returned_as_unique_low_confidence_candidate() -> None:
    bank = QuestionBank(
        [
            QuestionRecord("1", "梦幻西游中有多少个种族", "3", "梦幻西游中有多少个种族"),
            QuestionRecord("2", "长安城中有多少座桥", "4", "长安城中有多少座桥"),
        ]
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)

    candidate = matcher.unique_question_candidate(
        "梦幻西游中有多少",
        score_cutoff=75,
        minimum_gap=5,
    )

    assert candidate is not None
    assert candidate.record.source_id == "1"
    assert 75 <= candidate.score < 92
    assert candidate.query_variant.endswith("_partial")


def test_short_partial_query_never_uses_tolerant_partial_matching() -> None:
    bank = QuestionBank(
        [QuestionRecord("1", "梦幻西游中有多少个种族", "3", "梦幻西游中有多少个种族")]
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)

    assert matcher.unique_question_candidate("多少", score_cutoff=70) is None


def test_complete_ascii_comma_answer_is_tried_before_legacy_aliases() -> None:
    bank = QuestionBank(
        [QuestionRecord("1", "奥运口号", "新北京,新奥运", "奥运口号")]
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)
    question = matcher.match_question("奥运口号")
    assert question is not None

    option = matcher.unique_option_candidate(
        question,
        ["同一个世界", "新北京,新奥运", "更快更高更强", "绿色奥运"],
    )

    assert option is not None
    assert option.option_index == 1
    assert option.score == 100


def test_complete_answer_wins_before_explicit_local_aliases() -> None:
    record = QuestionRecord("local:1", "奥运口号", "新北京,新奥运", "奥运口号")
    bank = QuestionBank(
        [record],
        record_metadata={
            record.source_id: QuestionRecordMetadata(answer_aliases=("新北京",)),
        },
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)
    question = matcher.match_question("奥运口号")
    assert question is not None

    option = matcher.unique_option_candidate(
        question,
        ["新北京", "新北京,新奥运", "同一个世界", "绿色奥运"],
    )

    assert option is not None
    assert option.option_index == 1
    assert option.score == 100


def test_explicit_local_alias_is_used_only_when_complete_answer_does_not_match() -> None:
    record = QuestionRecord("local:1", "称谓题", "孙悟空", "称谓题")
    bank = QuestionBank(
        [record],
        record_metadata={
            record.source_id: QuestionRecordMetadata(answer_aliases=("齐天大圣",)),
        },
    )
    matcher = QuestionMatcher(bank, 92, 5, 90)
    question = matcher.match_question("称谓题")
    assert question is not None

    option = matcher.unique_option_candidate(
        question,
        ["斗战胜佛", "齐天大圣", "天蓬元帅", "卷帘大将"],
    )

    assert option is not None
    assert option.option_index == 1
    assert option.score == 100
