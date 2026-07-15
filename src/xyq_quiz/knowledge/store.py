from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Self

from xyq_quiz.knowledge.models import (
    QuestionRecord,
    QuestionRecordMetadata,
    normalize_text,
)


class QuestionBank:
    def __init__(
        self,
        records: list[QuestionRecord],
        *,
        record_metadata: Mapping[str, QuestionRecordMetadata] | None = None,
    ) -> None:
        self.records = tuple(records)
        source_ids: set[str] = set()
        exact: dict[str, QuestionRecord] = {}
        groups: dict[str, list[QuestionRecord]] = {}
        for record in records:
            if record.source_id in source_ids:
                raise ValueError(f"duplicate source_id: {record.source_id}")
            source_ids.add(record.source_id)
            exact.setdefault(record.normalized_question, record)
            groups.setdefault(record.normalized_question, []).append(record)
        self.exact = MappingProxyType(exact)
        self.groups = MappingProxyType(
            {key: tuple(group) for key, group in groups.items()}
        )
        self.normalized_questions = tuple(groups)

        supplied_metadata = dict(record_metadata or {})
        unknown_source_ids = set(supplied_metadata) - source_ids
        if unknown_source_ids:
            unknown = sorted(unknown_source_ids)[0]
            raise ValueError(f"metadata references unknown source_id: {unknown}")
        self.record_metadata = MappingProxyType(
            {
                source_id: supplied_metadata.get(
                    source_id,
                    QuestionRecordMetadata(),
                )
                for source_id in source_ids
            }
        )

    @property
    def count(self) -> int:
        return len(self.records)

    def records_for(self, normalized_question: str) -> tuple[QuestionRecord, ...]:
        return self.groups.get(normalized_question, ())

    def metadata_for(self, source_id: str) -> QuestionRecordMetadata:
        return self.record_metadata.get(source_id, QuestionRecordMetadata())

    @classmethod
    def load(cls, path: Path) -> Self:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("question bank must be a non-empty list")

        records: list[QuestionRecord] = []
        source_ids: set[str] = set()
        for index, raw_record in enumerate(data):
            record = cls._parse_record(raw_record, index)
            if record.source_id in source_ids:
                raise ValueError(f"duplicate source_id: {record.source_id}")
            source_ids.add(record.source_id)
            records.append(record)

        return cls(records)

    @staticmethod
    def _parse_record(raw_record: Any, index: int) -> QuestionRecord:
        if not isinstance(raw_record, dict):
            raise ValueError(f"question record {index} must be an object")

        fields: dict[str, str] = {}
        for name in ("source_id", "question", "answer", "normalized_question"):
            value = raw_record.get(name)
            if not isinstance(value, str):
                raise ValueError(f"question record {index} has invalid {name}")
            fields[name] = value

        if not fields["source_id"].strip():
            raise ValueError(f"question record {index} has blank source_id")
        if not normalize_text(fields["question"]):
            raise ValueError(f"question record {index} has blank question")
        if not normalize_text(fields["answer"]):
            raise ValueError(f"question record {index} has blank answer")

        expected_normalization = normalize_text(fields["question"])
        if fields["normalized_question"] != expected_normalization:
            raise ValueError(
                f"question record {index} normalized_question mismatch"
            )

        return QuestionRecord(**fields)
