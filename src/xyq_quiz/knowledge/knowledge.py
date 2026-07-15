from __future__ import annotations

from dataclasses import dataclass

from xyq_quiz.knowledge.local import (
    LocalQuestionConflict,
    LocalQuestionDocument,
    LocalQuestionLoad,
    LocalQuestionMode,
    LocalQuestionStore,
    validate_local_question_document,
)
from xyq_quiz.knowledge.models import (
    QuestionOrigin,
    QuestionRecord,
    QuestionRecordMetadata,
    normalize_text,
)
from xyq_quiz.knowledge.store import QuestionBank


@dataclass(frozen=True, slots=True)
class KnowledgeIssue:
    code: str
    local_ids: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    bank: QuestionBank
    official_bank: QuestionBank
    local_document: LocalQuestionDocument
    local_error: str | None = None
    issues: tuple[KnowledgeIssue, ...] = ()

    @property
    def local_conflicts(self) -> tuple[LocalQuestionConflict, ...]:
        return self.local_document.conflicts

    @property
    def active_local_count(self) -> int:
        return sum(
            self.bank.metadata_for(record.source_id).origin
            in {QuestionOrigin.LOCAL_SUPPLEMENT, QuestionOrigin.LOCAL_OVERRIDE}
            for record in self.bank.records
        )


def load_knowledge_snapshot(
    official_bank: QuestionBank,
    local_store: LocalQuestionStore,
) -> KnowledgeSnapshot:
    loaded = local_store.load_safe()
    return merge_knowledge(official_bank, loaded)


def merge_knowledge(
    official_bank: QuestionBank,
    local: LocalQuestionLoad | LocalQuestionDocument,
    *,
    local_error: str | None = None,
) -> KnowledgeSnapshot:
    if isinstance(local, LocalQuestionLoad):
        document = validate_local_question_document(local.document)
        effective_error = local.error
    else:
        document = validate_local_question_document(local)
        effective_error = local_error

    issues: list[KnowledgeIssue] = []
    official_by_source_id = {
        record.source_id: record for record in official_bank.records
    }
    overrides = {
        record.target_source_id: record
        for record in document.active_records
        if record.mode is LocalQuestionMode.OVERRIDE
        and record.target_source_id is not None
    }

    records: list[QuestionRecord] = []
    metadata: dict[str, QuestionRecordMetadata] = {}
    applied_local_ids: set[str] = set()

    for official_record in official_bank.records:
        override = overrides.get(official_record.source_id)
        if override is None:
            records.append(official_record)
            metadata[official_record.source_id] = official_bank.metadata_for(
                official_record.source_id
            )
            continue
        if override.normalized_question != official_record.normalized_question:
            issues.append(
                KnowledgeIssue(
                    code="stale_override_question",
                    local_ids=(override.id,),
                    message=(
                        f"local override {override.id} no longer matches official "
                        f"source_id {official_record.source_id}"
                    ),
                )
            )
            records.append(official_record)
            metadata[official_record.source_id] = official_bank.metadata_for(
                official_record.source_id
            )
            continue

        local_source_id = _local_source_id(override.id)
        records.append(
            QuestionRecord(
                source_id=local_source_id,
                question=override.question,
                answer=override.answer,
                normalized_question=override.normalized_question,
            )
        )
        metadata[local_source_id] = QuestionRecordMetadata(
            origin=QuestionOrigin.LOCAL_OVERRIDE,
            local_id=override.id,
            target_source_id=official_record.source_id,
            answer_aliases=override.answer_aliases,
        )
        applied_local_ids.add(override.id)

    for override in overrides.values():
        if override.id in applied_local_ids:
            continue
        if override.target_source_id not in official_by_source_id:
            issues.append(
                KnowledgeIssue(
                    code="missing_override_target",
                    local_ids=(override.id,),
                    message=(
                        f"local override {override.id} targets missing official "
                        f"source_id {override.target_source_id}"
                    ),
                )
            )

    existing_question_answers = {
        (record.normalized_question, normalize_text(record.answer))
        for record in records
    }
    for supplement in document.active_records:
        if supplement.mode is not LocalQuestionMode.SUPPLEMENT:
            continue
        key = (supplement.normalized_question, normalize_text(supplement.answer))
        if key in existing_question_answers:
            issues.append(
                KnowledgeIssue(
                    code="redundant_supplement",
                    local_ids=(supplement.id,),
                    message=f"local supplement {supplement.id} duplicates an active record",
                )
            )
            continue
        existing_question_answers.add(key)
        source_id = _local_source_id(supplement.id)
        records.append(
            QuestionRecord(
                source_id=source_id,
                question=supplement.question,
                answer=supplement.answer,
                normalized_question=supplement.normalized_question,
            )
        )
        metadata[source_id] = QuestionRecordMetadata(
            origin=QuestionOrigin.LOCAL_SUPPLEMENT,
            local_id=supplement.id,
            answer_aliases=supplement.answer_aliases,
        )

    return KnowledgeSnapshot(
        bank=QuestionBank(records, record_metadata=metadata),
        official_bank=official_bank,
        local_document=document,
        local_error=effective_error,
        issues=tuple(issues),
    )


def _local_source_id(local_id: str) -> str:
    return f"local:{local_id}"


__all__ = [
    "KnowledgeIssue",
    "KnowledgeSnapshot",
    "load_knowledge_snapshot",
    "merge_knowledge",
]
