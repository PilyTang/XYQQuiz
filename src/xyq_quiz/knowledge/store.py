from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from xyq_quiz.knowledge.models import QuestionRecord, normalize_text


class QuestionBank:
    def __init__(self, records: list[QuestionRecord]) -> None:
        self.records = tuple(records)
        exact: dict[str, QuestionRecord] = {}
        for record in records:
            exact.setdefault(record.normalized_question, record)
        self.exact = MappingProxyType(exact)
        self.normalized_questions = tuple(
            record.normalized_question for record in records
        )

    @property
    def count(self) -> int:
        return len(self.records)

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
