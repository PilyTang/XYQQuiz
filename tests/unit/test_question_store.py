from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from xyq_quiz.knowledge.store import QuestionBank


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "questions-small.json"


def _write_bank(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_builds_immutable_question_records() -> None:
    bank = QuestionBank.load(FIXTURE_PATH)

    assert bank.count == 3
    assert bank.records[0].source_id == "1"
    assert bank.records[0].normalized_question == "梦幻西游中有多少个种族"
    with pytest.raises(FrozenInstanceError):
        bank.records[0].answer = "4"  # type: ignore[misc]


def test_normalized_question_candidates_cannot_be_modified() -> None:
    bank = QuestionBank.load(FIXTURE_PATH)

    with pytest.raises(TypeError):
        bank.normalized_questions[0] = "破坏索引对应"  # type: ignore[index]


def test_load_rejects_empty_question_list(tmp_path: Path) -> None:
    path = _write_bank(tmp_path / "questions.json", [])

    with pytest.raises(ValueError, match="empty"):
        QuestionBank.load(path)


def test_load_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    rows = [
        {
            "source_id": "1",
            "question": "题目一",
            "answer": "答案一",
            "normalized_question": "题目一",
        },
        {
            "source_id": "1",
            "question": "题目二",
            "answer": "答案二",
            "normalized_question": "题目二",
        },
    ]
    path = _write_bank(tmp_path / "questions.json", rows)

    with pytest.raises(ValueError, match="duplicate source_id"):
        QuestionBank.load(path)


@pytest.mark.parametrize(("field", "value"), [("question", "  "), ("answer", "")])
def test_load_rejects_blank_question_or_answer(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    row = {
        "source_id": "1",
        "question": "有效题目",
        "answer": "有效答案",
        "normalized_question": "有效题目",
    }
    row[field] = value
    path = _write_bank(tmp_path / "questions.json", [row])

    with pytest.raises(ValueError, match=f"blank {field}"):
        QuestionBank.load(path)


def test_load_rejects_mismatched_stored_normalization(tmp_path: Path) -> None:
    rows = [
        {
            "source_id": "1",
            "question": "梦幻西游中，有多少个种族？",
            "answer": "3",
            "normalized_question": "错误归一化",
        }
    ]
    path = _write_bank(tmp_path / "questions.json", rows)

    with pytest.raises(ValueError, match="normalized_question mismatch"):
        QuestionBank.load(path)
