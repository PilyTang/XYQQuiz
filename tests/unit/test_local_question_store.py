from __future__ import annotations

import json
from pathlib import Path

import pytest

from xyq_quiz.knowledge.knowledge import (
    load_knowledge_snapshot,
    merge_knowledge,
)
from xyq_quiz.knowledge.local import (
    LocalQuestionConflictError,
    LocalQuestionDocument,
    LocalQuestionMode,
    LocalQuestionRecord,
    LocalQuestionSchemaError,
    LocalQuestionStore,
    LocalQuestionWriteConflictError,
    parse_local_question_document,
)
from xyq_quiz.knowledge.models import QuestionOrigin, QuestionRecord
from xyq_quiz.knowledge.store import QuestionBank


def _record(
    record_id: str = "local-1",
    *,
    question: str = "本地补充题",
    answer: str = "本地答案",
    mode: LocalQuestionMode = LocalQuestionMode.SUPPLEMENT,
    target_source_id: str | None = None,
    aliases: tuple[str, ...] = (),
) -> LocalQuestionRecord:
    return LocalQuestionRecord(
        id=record_id,
        question=question,
        answer=answer,
        mode=mode,
        target_source_id=target_source_id,
        answer_aliases=aliases,
    )


def _official_bank() -> QuestionBank:
    return QuestionBank(
        [
            QuestionRecord("1", "官方题目一", "官方答案一", "官方题目一"),
            QuestionRecord("2", "官方题目二", "官方答案二", "官方题目二"),
        ]
    )


def test_parse_local_document_computes_normalization_and_keeps_aliases() -> None:
    document = parse_local_question_document(
        {
            "schema_version": 1,
            "records": [
                {
                    "id": "local-1",
                    "question": " 本地补充题？ ",
                    "answer": "答案甲",
                    "answer_aliases": ["别名甲", "别名乙"],
                }
            ],
        }
    )

    assert document.records[0].normalized_question == "本地补充题"
    assert document.records[0].answer_aliases == ("别名甲", "别名乙")


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"schema_version": 99, "records": []},
        {"schema_version": 1, "records": [], "unknown": True},
        {
            "schema_version": 1,
            "records": [
                {"id": "same", "question": "题一", "answer": "答一"},
                {"id": "same", "question": "题二", "answer": "答二"},
            ],
        },
        {
            "schema_version": 1,
            "records": [
                {
                    "id": "bad",
                    "question": "题",
                    "answer": "答",
                    "mode": "override",
                }
            ],
        },
        {
            "schema_version": 1,
            "records": [
                {
                    "id": "bad",
                    "question": "题",
                    "answer": "答",
                    "normalized_question": "题",
                }
            ],
        },
    ],
)
def test_invalid_local_documents_are_rejected(value: object) -> None:
    with pytest.raises(LocalQuestionSchemaError):
        parse_local_question_document(value)


def test_duplicate_enabled_override_targets_are_detected_and_quarantined() -> None:
    document = parse_local_question_document(
        {
            "schema_version": 1,
            "records": [
                {
                    "id": "override-a",
                    "mode": "override",
                    "target_source_id": "1",
                    "question": "官方题目一",
                    "answer": "修正一",
                },
                {
                    "id": "override-b",
                    "mode": "override",
                    "target_source_id": "1",
                    "question": "官方题目一",
                    "answer": "修正二",
                },
            ],
        }
    )

    assert document.conflicts[0].code == "duplicate_override_target"
    assert document.active_records == ()


def test_merge_revalidates_programmatic_documents_and_quarantines_conflicts() -> None:
    document = LocalQuestionDocument(
        (
            _record(
                "override-a",
                question="官方题目一",
                answer="修正一",
                mode=LocalQuestionMode.OVERRIDE,
                target_source_id="1",
            ),
            _record(
                "override-b",
                question="官方题目一",
                answer="修正二",
                mode=LocalQuestionMode.OVERRIDE,
                target_source_id="1",
            ),
        )
    )

    snapshot = merge_knowledge(_official_bank(), document)

    assert snapshot.local_conflicts[0].code == "duplicate_override_target"
    assert snapshot.local_document.active_records == ()
    assert snapshot.bank.exact["官方题目一"].answer == "官方答案一"


def test_store_round_trip_is_atomic_and_supports_optimistic_hash(tmp_path: Path) -> None:
    path = tmp_path / "user-data" / "questions.json"
    store = LocalQuestionStore(path)

    first = store.write(LocalQuestionDocument((_record(),)))
    loaded = store.load()

    assert loaded.records == first.records
    assert loaded.sha256 == first.sha256
    assert json.loads(path.read_text("utf-8"))["schema_version"] == 1
    assert not list(path.parent.glob(".questions.json.*.tmp"))

    updated = store.upsert(
        _record("local-2", question="第二题", answer="第二答"),
        expected_sha256=first.sha256,
    )
    assert len(updated.records) == 2
    with pytest.raises(LocalQuestionWriteConflictError):
        store.delete("local-1", expected_sha256=first.sha256)
    assert len(store.load().records) == 2


def test_store_never_overwrites_a_damaged_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_bytes(b"{ damaged")
    original = path.read_bytes()
    store = LocalQuestionStore(path)

    safe = store.load_safe()
    assert safe.error is not None
    assert safe.document.records == ()
    with pytest.raises(LocalQuestionSchemaError):
        store.upsert(_record())
    assert path.read_bytes() == original


def test_store_refuses_to_publish_conflicted_document(tmp_path: Path) -> None:
    store = LocalQuestionStore(tmp_path / "questions.json")
    conflict = LocalQuestionDocument(
        (
            _record(
                "a",
                mode=LocalQuestionMode.OVERRIDE,
                target_source_id="1",
            ),
            _record(
                "b",
                mode=LocalQuestionMode.OVERRIDE,
                target_source_id="1",
            ),
        )
    )

    with pytest.raises(LocalQuestionConflictError):
        store.write(conflict)
    assert not store.path.exists()


def test_merge_applies_targeted_override_and_supplement_with_provenance() -> None:
    document = LocalQuestionDocument(
        (
            _record(
                "override-1",
                question="官方题目一？",
                answer="本地修正答案",
                mode=LocalQuestionMode.OVERRIDE,
                target_source_id="1",
                aliases=("修正别名",),
            ),
            _record("supplement-1", question="新增题", answer="新增答"),
        )
    )

    snapshot = merge_knowledge(_official_bank(), document)

    assert snapshot.bank.count == 3
    assert snapshot.bank.exact["官方题目一"].answer == "本地修正答案"
    override = snapshot.bank.exact["官方题目一"]
    supplement = snapshot.bank.exact["新增题"]
    assert (
        snapshot.bank.metadata_for(override.source_id).origin
        is QuestionOrigin.LOCAL_OVERRIDE
    )
    assert snapshot.bank.metadata_for(override.source_id).target_source_id == "1"
    assert snapshot.bank.metadata_for(override.source_id).answer_aliases == ("修正别名",)
    assert (
        snapshot.bank.metadata_for(supplement.source_id).origin
        is QuestionOrigin.LOCAL_SUPPLEMENT
    )
    assert snapshot.active_local_count == 2


def test_stale_or_missing_override_never_silently_shadows_official_data() -> None:
    document = LocalQuestionDocument(
        (
            _record(
                "stale",
                question="已经变掉的题目",
                answer="不应应用",
                mode=LocalQuestionMode.OVERRIDE,
                target_source_id="1",
            ),
            _record(
                "missing",
                question="不存在",
                answer="不应应用",
                mode=LocalQuestionMode.OVERRIDE,
                target_source_id="404",
            ),
        )
    )

    snapshot = merge_knowledge(_official_bank(), document)

    assert snapshot.bank.exact["官方题目一"].answer == "官方答案一"
    assert {issue.code for issue in snapshot.issues} == {
        "stale_override_question",
        "missing_override_target",
    }
    assert snapshot.active_local_count == 0


def test_redundant_supplement_is_reported_without_duplicating_bank_row() -> None:
    snapshot = merge_knowledge(
        _official_bank(),
        LocalQuestionDocument(
            (_record(question="官方题目一？", answer="官方答案一"),)
        ),
    )

    assert snapshot.bank.count == 2
    assert snapshot.issues[0].code == "redundant_supplement"


def test_corrupt_local_file_fails_open_to_official_bank(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text("not json", encoding="utf-8")

    snapshot = load_knowledge_snapshot(_official_bank(), LocalQuestionStore(path))

    assert snapshot.bank.records == _official_bank().records
    assert snapshot.local_error is not None
