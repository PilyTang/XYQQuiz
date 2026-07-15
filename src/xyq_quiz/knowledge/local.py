from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Self
from uuid import uuid4

from xyq_quiz.knowledge.models import normalize_text


LOCAL_QUESTION_SCHEMA_VERSION = 1
_MAXIMUM_FILE_BYTES = 5_000_000
_MAXIMUM_RECORDS = 5_000
_MAXIMUM_TEXT_LENGTH = 2_000
_MAXIMUM_ALIASES = 32
_LOCAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class LocalQuestionError(ValueError):
    pass


class LocalQuestionSchemaError(LocalQuestionError):
    pass


class LocalQuestionConflictError(LocalQuestionError):
    pass


class LocalQuestionWriteConflictError(LocalQuestionError):
    pass


class LocalQuestionMode(StrEnum):
    SUPPLEMENT = "supplement"
    OVERRIDE = "override"


@dataclass(frozen=True, slots=True)
class LocalQuestionRecord:
    id: str
    question: str
    answer: str
    mode: LocalQuestionMode = LocalQuestionMode.SUPPLEMENT
    enabled: bool = True
    target_source_id: str | None = None
    answer_aliases: tuple[str, ...] = ()

    @property
    def normalized_question(self) -> str:
        return normalize_text(self.question)


@dataclass(frozen=True, slots=True)
class LocalQuestionConflict:
    code: str
    record_ids: tuple[str, ...]
    message: str
    target_source_id: str | None = None


@dataclass(frozen=True, slots=True)
class LocalQuestionDocument:
    records: tuple[LocalQuestionRecord, ...] = ()
    conflicts: tuple[LocalQuestionConflict, ...] = ()
    sha256: str | None = None

    @classmethod
    def empty(cls) -> Self:
        return cls()

    @property
    def active_records(self) -> tuple[LocalQuestionRecord, ...]:
        conflicted_ids = {
            record_id
            for conflict in self.conflicts
            for record_id in conflict.record_ids
        }
        return tuple(
            record
            for record in self.records
            if record.enabled and record.id not in conflicted_ids
        )


@dataclass(frozen=True, slots=True)
class LocalQuestionLoad:
    document: LocalQuestionDocument
    error: str | None = None


class LocalQuestionStore:
    """Validated, optimistic and atomic storage for user-maintained questions."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> LocalQuestionDocument:
        with self._lock:
            return self._load_unlocked()

    def load_safe(self) -> LocalQuestionLoad:
        try:
            return LocalQuestionLoad(self.load())
        except (LocalQuestionError, OSError, UnicodeError) as error:
            return LocalQuestionLoad(LocalQuestionDocument.empty(), str(error))

    def write(
        self,
        document: LocalQuestionDocument,
        *,
        expected_sha256: str | None = None,
    ) -> LocalQuestionDocument:
        with self._lock:
            current = self._load_unlocked()
            return self._write_unlocked(
                document,
                current=current,
                expected_sha256=expected_sha256,
            )

    def upsert(
        self,
        record: LocalQuestionRecord,
        *,
        expected_sha256: str | None = None,
    ) -> LocalQuestionDocument:
        with self._lock:
            current = self._load_unlocked()
            records = list(current.records)
            for index, existing in enumerate(records):
                if existing.id == record.id:
                    records[index] = record
                    break
            else:
                records.append(record)
            return self._write_unlocked(
                LocalQuestionDocument(tuple(records)),
                current=current,
                expected_sha256=expected_sha256,
            )

    def delete(
        self,
        record_id: str,
        *,
        expected_sha256: str | None = None,
    ) -> LocalQuestionDocument:
        with self._lock:
            current = self._load_unlocked()
            records = tuple(record for record in current.records if record.id != record_id)
            if len(records) == len(current.records):
                raise KeyError(record_id)
            return self._write_unlocked(
                LocalQuestionDocument(records),
                current=current,
                expected_sha256=expected_sha256,
            )

    def _load_unlocked(self) -> LocalQuestionDocument:
        if not self.path.exists():
            return LocalQuestionDocument.empty()
        if not self.path.is_file():
            raise LocalQuestionSchemaError(
                f"local question path is not a file: {self.path}"
            )
        payload = self.path.read_bytes()
        if len(payload) > _MAXIMUM_FILE_BYTES:
            raise LocalQuestionSchemaError("local question file is too large")
        digest = hashlib.sha256(payload).hexdigest()
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LocalQuestionSchemaError(
                f"local question file is invalid: {error}"
            ) from error
        return replace(parse_local_question_document(value), sha256=digest)

    def _write_unlocked(
        self,
        document: LocalQuestionDocument,
        *,
        current: LocalQuestionDocument,
        expected_sha256: str | None,
    ) -> LocalQuestionDocument:
        if expected_sha256 is not None and current.sha256 != expected_sha256:
            raise LocalQuestionWriteConflictError(
                "local question file changed since it was loaded"
            )

        validated = validate_local_question_document(document)
        if validated.conflicts:
            raise LocalQuestionConflictError(validated.conflicts[0].message)

        payload = _document_bytes(validated)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.parent / f".{self.path.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return replace(
            validated,
            sha256=hashlib.sha256(payload).hexdigest(),
        )


def parse_local_question_document(value: Any) -> LocalQuestionDocument:
    if not isinstance(value, dict):
        raise LocalQuestionSchemaError("local question document must be an object")
    unexpected = set(value) - {"schema_version", "records"}
    if unexpected:
        raise LocalQuestionSchemaError(
            f"local question document has unknown field: {sorted(unexpected)[0]}"
        )
    version = value.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise LocalQuestionSchemaError("local question schema_version must be an integer")
    if version != LOCAL_QUESTION_SCHEMA_VERSION:
        raise LocalQuestionSchemaError(
            f"unsupported local question schema_version: {version}"
        )
    raw_records = value.get("records")
    if not isinstance(raw_records, list):
        raise LocalQuestionSchemaError("local question records must be a list")
    if len(raw_records) > _MAXIMUM_RECORDS:
        raise LocalQuestionSchemaError("local question document has too many records")

    records: list[LocalQuestionRecord] = []
    seen_ids: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        record = _parse_record(raw_record, index)
        if record.id in seen_ids:
            raise LocalQuestionSchemaError(f"duplicate local question id: {record.id}")
        seen_ids.add(record.id)
        records.append(record)
    return LocalQuestionDocument(
        records=tuple(records),
        conflicts=_detect_conflicts(records),
    )


def validate_local_question_document(
    document: LocalQuestionDocument,
) -> LocalQuestionDocument:
    """Rebuild derived conflict state from a programmatic document.

    Callers may construct ``LocalQuestionDocument`` directly, so its cached
    ``conflicts`` tuple cannot be trusted at merge or write boundaries.
    """

    if not isinstance(document, LocalQuestionDocument):
        raise LocalQuestionSchemaError("local question document has invalid type")
    try:
        validated = parse_local_question_document(_document_value(document))
    except (AttributeError, TypeError, ValueError) as error:
        if isinstance(error, LocalQuestionError):
            raise
        raise LocalQuestionSchemaError(
            f"local question document is invalid: {error}"
        ) from error
    return replace(validated, sha256=document.sha256)


def _parse_record(value: Any, index: int) -> LocalQuestionRecord:
    if not isinstance(value, dict):
        raise LocalQuestionSchemaError(f"local question record {index} must be an object")
    allowed = {
        "id",
        "question",
        "answer",
        "mode",
        "enabled",
        "target_source_id",
        "answer_aliases",
    }
    unexpected = set(value) - allowed
    if unexpected:
        raise LocalQuestionSchemaError(
            f"local question record {index} has unknown field: {sorted(unexpected)[0]}"
        )

    record_id = _required_text(value.get("id"), index, "id", maximum=128)
    if _LOCAL_ID.fullmatch(record_id) is None:
        raise LocalQuestionSchemaError(
            f"local question record {index} has invalid id"
        )
    question = _required_text(value.get("question"), index, "question")
    answer = _required_text(value.get("answer"), index, "answer")
    if not normalize_text(question):
        raise LocalQuestionSchemaError(
            f"local question record {index} has blank question"
        )
    if not normalize_text(answer):
        raise LocalQuestionSchemaError(
            f"local question record {index} has blank answer"
        )

    raw_mode = value.get("mode", LocalQuestionMode.SUPPLEMENT.value)
    try:
        mode = LocalQuestionMode(raw_mode)
    except (TypeError, ValueError) as error:
        raise LocalQuestionSchemaError(
            f"local question record {index} has invalid mode"
        ) from error
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise LocalQuestionSchemaError(
            f"local question record {index} enabled must be a boolean"
        )

    target = value.get("target_source_id")
    if target is not None:
        target = _required_text(target, index, "target_source_id", maximum=128)
    if mode is LocalQuestionMode.OVERRIDE and target is None:
        raise LocalQuestionSchemaError(
            f"local question record {index} override requires target_source_id"
        )
    if mode is LocalQuestionMode.SUPPLEMENT and target is not None:
        raise LocalQuestionSchemaError(
            f"local question record {index} supplement cannot target a source_id"
        )

    raw_aliases = value.get("answer_aliases", [])
    if not isinstance(raw_aliases, list) or len(raw_aliases) > _MAXIMUM_ALIASES:
        raise LocalQuestionSchemaError(
            f"local question record {index} has invalid answer_aliases"
        )
    aliases: list[str] = []
    normalized_aliases: set[str] = set()
    for alias_index, raw_alias in enumerate(raw_aliases):
        alias = _required_text(
            raw_alias,
            index,
            f"answer_aliases[{alias_index}]",
            maximum=500,
        )
        normalized_alias = normalize_text(alias)
        if not normalized_alias:
            raise LocalQuestionSchemaError(
                f"local question record {index} has blank answer alias"
            )
        if normalized_alias in normalized_aliases:
            raise LocalQuestionSchemaError(
                f"local question record {index} has duplicate answer alias"
            )
        normalized_aliases.add(normalized_alias)
        aliases.append(alias)

    return LocalQuestionRecord(
        id=record_id,
        question=question,
        answer=answer,
        mode=mode,
        enabled=enabled,
        target_source_id=target,
        answer_aliases=tuple(aliases),
    )


def _required_text(
    value: Any,
    index: int,
    name: str,
    *,
    maximum: int = _MAXIMUM_TEXT_LENGTH,
) -> str:
    if not isinstance(value, str):
        raise LocalQuestionSchemaError(
            f"local question record {index} has invalid {name}"
        )
    stripped = value.strip()
    if not stripped:
        raise LocalQuestionSchemaError(
            f"local question record {index} has blank {name}"
        )
    if len(stripped) > maximum:
        raise LocalQuestionSchemaError(
            f"local question record {index} {name} is too long"
        )
    return stripped


def _detect_conflicts(
    records: list[LocalQuestionRecord],
) -> tuple[LocalQuestionConflict, ...]:
    overrides: dict[str, list[str]] = {}
    for record in records:
        if (
            record.enabled
            and record.mode is LocalQuestionMode.OVERRIDE
            and record.target_source_id is not None
        ):
            overrides.setdefault(record.target_source_id, []).append(record.id)

    conflicts = [
        LocalQuestionConflict(
            code="duplicate_override_target",
            record_ids=tuple(record_ids),
            target_source_id=target,
            message=(
                f"multiple local questions override official source_id {target}: "
                f"{', '.join(record_ids)}"
            ),
        )
        for target, record_ids in overrides.items()
        if len(record_ids) > 1
    ]
    return tuple(conflicts)


def _document_value(document: LocalQuestionDocument) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for record in document.records:
        value: dict[str, Any] = {
            "id": record.id,
            "enabled": record.enabled,
            "mode": record.mode.value,
            "question": record.question,
            "answer": record.answer,
            "answer_aliases": list(record.answer_aliases),
        }
        if record.target_source_id is not None:
            value["target_source_id"] = record.target_source_id
        records.append(value)
    return {
        "schema_version": LOCAL_QUESTION_SCHEMA_VERSION,
        "records": records,
    }


def _document_bytes(document: LocalQuestionDocument) -> bytes:
    return (
        json.dumps(
            _document_value(document),
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "LOCAL_QUESTION_SCHEMA_VERSION",
    "LocalQuestionConflict",
    "LocalQuestionConflictError",
    "LocalQuestionDocument",
    "LocalQuestionError",
    "LocalQuestionLoad",
    "LocalQuestionMode",
    "LocalQuestionRecord",
    "LocalQuestionSchemaError",
    "LocalQuestionStore",
    "LocalQuestionWriteConflictError",
    "parse_local_question_document",
    "validate_local_question_document",
]
