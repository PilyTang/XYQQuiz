from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import httpx
import pyjson5

from xyq_quiz.runtime.portable import (
    STATE_SCHEMA_VERSION,
    migrate_metadata_document,
    migrate_pointer_document,
)

from xyq_quiz.knowledge.models import QuestionRecord, normalize_text
from xyq_quiz.knowledge.store import QuestionBank


DEFAULT_SOURCE_URL = "https://w.163.com/h5/xyq/dtk/"
MINIMUM_RECORD_COUNT = 2000
MAXIMUM_DUPLICATE_RATE = 0.05


class UpdaterParseError(ValueError):
    """Raised when the official page no longer has the expected structure."""


class UpdateValidationError(ValueError):
    """Raised when a fetched question bank is unsafe to publish."""


class UpdateInProgressError(RuntimeError):
    """Raised when the same updater instance is already publishing."""


@dataclass(frozen=True, slots=True)
class UpdateResult:
    generation_id: str
    source_url: str
    chunk_url: str
    module_id: int
    record_count: int
    raw_record_count: int
    published_record_count: int
    filtered_ids: tuple[str, ...]
    normalized_duplicate_rate: float
    sha256: str


@dataclass(frozen=True, slots=True)
class _ExtractedRecords:
    records: list[QuestionRecord]
    raw_record_count: int
    filtered_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CurrentGeneration:
    generation_id: str
    question_bank: QuestionBank
    metadata: dict[str, Any]
    question_path: Path
    metadata_path: Path


class _ScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "script":
            return
        source = dict(attrs).get("src")
        if source is not None:
            self.sources.append(source)


def _app_script_source(index_html: str) -> str:
    parser = _ScriptParser()
    parser.feed(index_html)
    matches = [
        source
        for source in parser.sources
        if re.search(r"(?:^|/)app(?:\.[^/?]+)?\.js$", urlparse(source).path)
    ]
    if len(matches) != 1:
        raise UpdaterParseError(
            f"expected exactly one app.*.js script, found {len(matches)}"
        )
    return matches[0]


def _balanced_end(
    text: str,
    start: int,
    opening: str,
    closing: str,
) -> int:
    if start >= len(text) or text[start] != opening:
        raise UpdaterParseError(f"expected {opening!r} at offset {start}")

    depth = 0
    quote: str | None = None
    escaped = False
    index = start
    while index < len(text):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue

        if character in "'\"`":
            quote = character
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1

    raise UpdaterParseError(f"unterminated {opening!r} starting at offset {start}")


def _object_regions(text: str) -> list[tuple[int, int]]:
    stack: list[int] = []
    regions: list[tuple[int, int]] = []
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in "'\"`":
            quote = character
        elif character == "{":
            stack.append(index)
        elif character == "}" and stack:
            regions.append((stack.pop(), index + 1))
    return regions


def _route_object_source(app_js: str, route_path: str) -> str:
    route_matches = list(
        re.finditer(
            rf"(?:['\"]path['\"]|\bpath)\s*:\s*(['\"]){re.escape(route_path)}\1",
            app_js,
        )
    )
    regions = _object_regions(app_js)
    route_regions: set[tuple[int, int]] = set()
    for route_match in route_matches:
        enclosing = [
            region
            for region in regions
            if region[0] < route_match.start() and route_match.end() < region[1]
        ]
        if enclosing:
            route_regions.add(min(enclosing, key=lambda region: region[1] - region[0]))
    if len(route_regions) != 1:
        raise UpdaterParseError(
            f"expected exactly one route object for {route_path}, "
            f"found {len(route_regions)}"
        )
    start, end = route_regions.pop()
    return app_js[start:end]


def _expression_end(source: str, start: int) -> int:
    depths = {"(": 0, "[": 0, "{": 0}
    closing_to_opening = {")": "(", "]": "[", "}": "{"}
    quote: str | None = None
    escaped = False
    for index in range(start, len(source)):
        character = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in "'\"`":
            quote = character
        elif character in depths:
            depths[character] += 1
        elif character in closing_to_opening:
            opening = closing_to_opening[character]
            if depths[opening] == 0:
                return index
            depths[opening] -= 1
        elif character == "," and all(depth == 0 for depth in depths.values()):
            return index
    return len(source)


def _component_expression(route_source: str, route_path: str) -> str:
    component_match = re.search(
        r"(?:['\"]component['\"]|\bcomponent)\s*:",
        route_source,
    )
    if component_match is None:
        raise UpdaterParseError(f"{route_path} route has no component loader")
    end = _expression_end(route_source, component_match.end())
    return route_source[component_match.end() : end]


def _webpack_map_values(script_expression: str, chunk_id: int) -> list[str]:
    values: list[tuple[int, int, str]] = []
    for start, end in _object_regions(script_expression):
        candidate = script_expression[start:end]
        try:
            # JavaScript object literals allow numeric property names while
            # pyjson5 requires JSON5-compatible quoted names.
            compatible = re.sub(
                r"(?P<prefix>[{,]\s*)(?P<key>\d+)(?P<suffix>\s*:)",
                r'\g<prefix>"\g<key>"\g<suffix>',
                candidate,
            )
            decoded = pyjson5.decode(compatible)
        except (TypeError, pyjson5.Json5DecoderException):
            continue
        if not isinstance(decoded, dict):
            continue
        value = decoded.get(str(chunk_id), decoded.get(chunk_id))
        if isinstance(value, str):
            values.append((end - start, start, value))

    # Nested objects may contain the same mapping. Keep the smallest region for
    # each source position/value, which is the actual manifest object.
    values.sort()
    selected: list[tuple[int, str]] = []
    for _length, start, value in values:
        if any(
            outer_start <= start and outer_value == value
            for outer_start, outer_value in selected
        ):
            continue
        selected.append((start, value))
    return [value for _, value in sorted(selected)]


def resolve_keju_chunk(
    index_html: str,
    app_js: str,
    base_url: str,
) -> tuple[str, int]:
    """Resolve the dynamic Keju bundle URL and its Webpack module id."""

    _app_script_source(index_html)

    route_source = _route_object_source(app_js, "/keju")
    component_source = _component_expression(route_source, "/keju")

    bind_pattern = re.compile(
        r"(?P<runtime>[$\w]+)\.bind\(\s*(?P=runtime)\s*,\s*(?P<module>\d+)\s*\)"
    )
    bind_match = bind_pattern.search(component_source)
    if bind_match is None:
        raise UpdaterParseError("the /keju component loader has no module binding")

    runtime = bind_match.group("runtime")
    route_loader = component_source[: bind_match.start()]
    chunk_matches = list(
        re.finditer(rf"\b{re.escape(runtime)}\.e\(\s*(\d+)\s*\)", route_loader)
    )
    if not chunk_matches:
        raise UpdaterParseError("could not find the /keju dynamic chunk")
    chunk_id = int(chunk_matches[-1].group(1))
    module_id = int(bind_match.group("module"))

    manifest_match = re.search(rf"\b{re.escape(runtime)}\.u\s*=", app_js)
    if manifest_match is None:
        raise UpdaterParseError("could not find the Webpack script resolver")
    js_suffix = re.search(r"(['\"])\.js\1", app_js[manifest_match.end() :])
    if js_suffix is None:
        raise UpdaterParseError("could not find the Webpack script suffix")
    expression_end = manifest_match.end() + js_suffix.end()
    manifest_values = _webpack_map_values(
        app_js[manifest_match.end() : expression_end],
        chunk_id,
    )
    if len(manifest_values) < 2:
        raise UpdaterParseError(
            f"could not resolve name and hash maps for chunk {chunk_id}"
        )
    chunk_name, chunk_hash = manifest_values[-2:]

    chunk_path = f"static/js/{chunk_name}.{chunk_hash}.js"
    return urljoin(base_url, chunk_path), module_id


def extract_keju_records(chunk_js: str, module_id: int) -> list[QuestionRecord]:
    """Extract exact Id/Q/A objects from one resolved Webpack module."""

    return _extract_keju_records_with_audit(chunk_js, module_id).records


def _extract_keju_records_with_audit(
    chunk_js: str,
    module_id: int,
) -> _ExtractedRecords:

    module_match = re.search(
        rf"(?<![$\w])(?:['\"]?{module_id}['\"]?)\s*:",
        chunk_js,
    )
    if module_match is None:
        raise UpdaterParseError(f"could not find Webpack module {module_id}")
    module_start = chunk_js.find("{", module_match.end())
    if module_start < 0:
        raise UpdaterParseError(f"module {module_id} has no function body")
    module_end = _balanced_end(chunk_js, module_start, "{", "}")
    module_source = chunk_js[module_start:module_end]

    array_match = re.search(r"\bconst\s+u\s*=\s*\[", module_source)
    if array_match is None:
        raise UpdaterParseError(f"module {module_id} has no const u array")
    array_start = array_match.end() - 1
    array_end = _balanced_end(module_source, array_start, "[", "]")
    try:
        decoded = pyjson5.decode(module_source[array_start:array_end])
    except (TypeError, pyjson5.Json5DecoderException) as error:
        raise UpdaterParseError("could not decode the Keju question array") from error
    if not isinstance(decoded, list):
        raise UpdaterParseError("the Keju question value is not an array")

    records: list[QuestionRecord] = []
    raw_record_count = 0
    filtered_ids: list[str] = []
    for raw_record in decoded:
        if not isinstance(raw_record, dict) or set(raw_record) != {"Id", "Q", "A"}:
            continue
        raw_record_count += 1
        source_id = raw_record["Id"]
        question = raw_record["Q"]
        answer = raw_record["A"]
        if not isinstance(source_id, (str, int)):
            raise UpdaterParseError("question Id must be a string or integer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise UpdaterParseError("question Q and A must be strings")
        source_id_text = str(source_id)
        question_is_blank = not question.strip()
        answer_is_blank = not answer.strip()
        if question_is_blank and answer_is_blank:
            filtered_ids.append(source_id_text)
            continue
        if question_is_blank:
            raise UpdateValidationError(
                f"record {source_id_text} has an empty question"
            )
        if answer_is_blank:
            raise UpdateValidationError(f"record {source_id_text} has an empty answer")
        records.append(
            QuestionRecord(
                source_id=source_id_text,
                question=question,
                answer=answer,
                normalized_question=normalize_text(question),
            )
        )
    return _ExtractedRecords(
        records=records,
        raw_record_count=raw_record_count,
        filtered_ids=tuple(filtered_ids),
    )


def load_current_generation(data_dir: Path) -> CurrentGeneration:
    """Load a question bank and metadata through one captured generation pointer."""

    data_dir = Path(data_dir)
    pointer = migrate_pointer_document(
        json.loads((data_dir / "current.json").read_text(encoding="utf-8"))
    )
    generation_id = pointer.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", generation_id)
        or generation_id in {".", ".."}
        or Path(generation_id).name != generation_id
    ):
        raise ValueError("current.json has an invalid generation_id")

    generation_dir = data_dir / "generations" / generation_id
    question_path = generation_dir / "keju_questions.json"
    metadata_path = generation_dir / "metadata.json"
    question_bank = QuestionBank.load(question_path)
    metadata = migrate_metadata_document(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    if metadata.get("generation_id") != generation_id:
        raise ValueError("generation metadata does not match current.json")
    digest = hashlib.sha256(question_path.read_bytes()).hexdigest()
    if metadata.get("sha256") != digest:
        raise ValueError("generation question-bank SHA-256 mismatch")
    if metadata.get("published_record_count") != question_bank.count:
        raise ValueError("generation metadata record count mismatch")
    return CurrentGeneration(
        generation_id=generation_id,
        question_bank=question_bank,
        metadata=metadata,
        question_path=question_path,
        metadata_path=metadata_path,
    )


class QuestionBankUpdater:
    def __init__(
        self,
        data_dir: Path,
        *,
        source_url: str = DEFAULT_SOURCE_URL,
        transport: httpx.BaseTransport | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.source_url = source_url
        self.transport = transport
        self._fault_injector = fault_injector
        self._update_lock = threading.Lock()

    def update(self) -> UpdateResult:
        if not self._update_lock.acquire(blocking=False):
            raise UpdateInProgressError("question-bank update is already in progress")
        try:
            return self._update_locked()
        finally:
            self._update_lock.release()

    def _update_locked(self) -> UpdateResult:
        generations_dir = self.data_dir / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_safe_temporaries(generations_dir)

        generation_id = self._new_generation_id()
        staging_dir = generations_dir / f".tmp-{generation_id}"
        generation_dir = generations_dir / generation_id
        pointer_tmp = self.data_dir / f".current-{uuid.uuid4().hex}.tmp"
        staging_dir.mkdir()

        try:
            chunk_js, chunk_url, module_id = self._fetch()
            extracted = _extract_keju_records_with_audit(chunk_js, module_id)
            records = extracted.records
            duplicate_rate = self._validate(records)

            question_path = staging_dir / "keju_questions.json"
            question_bytes = self._question_bytes(records)
            self._write_synced(question_path, question_bytes, "question")
            try:
                QuestionBank.load(question_path)
            except (OSError, ValueError) as error:
                raise UpdateValidationError(
                    f"temporary question bank failed to load: {error}"
                ) from error

            digest = hashlib.sha256(question_bytes).hexdigest()
            metadata = {
                "schema_version": STATE_SCHEMA_VERSION,
                "generation_id": generation_id,
                "source_url": self.source_url,
                "chunk_url": chunk_url,
                "module_id": module_id,
                "record_count": len(records),
                "raw_record_count": extracted.raw_record_count,
                "published_record_count": len(records),
                "filtered_ids": list(extracted.filtered_ids),
                "unique_source_id_count": len({r.source_id for r in records}),
                "unique_normalized_question_count": len(
                    {r.normalized_question for r in records}
                ),
                "normalized_duplicate_rate": duplicate_rate,
                "sha256": digest,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path = staging_dir / "metadata.json"
            self._write_synced(
                metadata_path,
                self._json_bytes(metadata),
                "metadata",
            )
            self._validate_staged_generation(
                generation_id,
                question_path,
                metadata_path,
            )

            self._checkpoint("generation_publish")
            os.replace(staging_dir, generation_dir)

            pointer_bytes = self._json_bytes(
                {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "generation_id": generation_id,
                }
            )
            self._write_synced(pointer_tmp, pointer_bytes, "current")
            self._checkpoint("current_replace")
            os.replace(pointer_tmp, self.data_dir / "current.json")

            return UpdateResult(
                generation_id=generation_id,
                source_url=self.source_url,
                chunk_url=chunk_url,
                module_id=module_id,
                record_count=len(records),
                raw_record_count=extracted.raw_record_count,
                published_record_count=len(records),
                filtered_ids=extracted.filtered_ids,
                normalized_duplicate_rate=duplicate_rate,
                sha256=digest,
            )
        finally:
            pointer_tmp.unlink(missing_ok=True)
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def _fetch(self) -> tuple[str, str, int]:
        headers = {"User-Agent": "XYQQuiz/0.1"}
        with httpx.Client(
            timeout=15,
            follow_redirects=True,
            headers=headers,
            transport=self.transport,
        ) as client:
            index_response = client.get(self.source_url)
            index_response.raise_for_status()
            index_html = index_response.text

            app_url = urljoin(self.source_url, _app_script_source(index_html))
            app_response = client.get(app_url)
            app_response.raise_for_status()
            app_js = app_response.text

            chunk_url, module_id = resolve_keju_chunk(
                index_html,
                app_js,
                self.source_url,
            )
            chunk_response = client.get(chunk_url)
            chunk_response.raise_for_status()
            return chunk_response.text, chunk_url, module_id

    @staticmethod
    def _validate(records: list[QuestionRecord]) -> float:
        if len(records) < MINIMUM_RECORD_COUNT:
            raise UpdateValidationError(
                f"question bank must contain at least {MINIMUM_RECORD_COUNT} records; "
                f"got {len(records)}"
            )

        source_ids = [record.source_id for record in records]
        if len(set(source_ids)) != len(source_ids):
            raise UpdateValidationError("question bank must have unique source IDs")

        for index, record in enumerate(records):
            if not record.source_id.strip():
                raise UpdateValidationError(f"record {index} has an empty source ID")
            if not record.question.strip() or not record.normalized_question:
                raise UpdateValidationError(f"record {index} has an empty question")
            if not record.answer.strip() or not normalize_text(record.answer):
                raise UpdateValidationError(f"record {index} has an empty answer")

        unique_questions = len({record.normalized_question for record in records})
        duplicate_rate = (len(records) - unique_questions) / len(records)
        if duplicate_rate >= MAXIMUM_DUPLICATE_RATE:
            raise UpdateValidationError(
                "normalized-question duplicate rate must be below 5%; "
                f"got {duplicate_rate:.2%}"
            )
        return duplicate_rate

    @classmethod
    def _question_bytes(cls, records: list[QuestionRecord]) -> bytes:
        return cls._json_bytes([asdict(record) for record in records])

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
        ).encode("utf-8")

    def _write_synced(self, path: Path, payload: bytes, stage: str) -> None:
        with path.open("wb") as handle:
            self._checkpoint(f"{stage}_write")
            handle.write(payload)
            handle.flush()
            self._checkpoint(f"{stage}_fsync")
            os.fsync(handle.fileno())

    def _checkpoint(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    @staticmethod
    def _new_generation_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{timestamp}-{uuid.uuid4().hex}"

    def _cleanup_safe_temporaries(self, generations_dir: Path) -> None:
        for child in generations_dir.iterdir():
            if not child.name.startswith(".tmp-"):
                continue
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child)
        for pointer_tmp in self.data_dir.glob(".current-*.tmp"):
            if pointer_tmp.is_file() or pointer_tmp.is_symlink():
                pointer_tmp.unlink(missing_ok=True)

    @staticmethod
    def _validate_staged_generation(
        generation_id: str,
        question_path: Path,
        metadata_path: Path,
    ) -> None:
        bank = QuestionBank.load(question_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise UpdateValidationError("temporary metadata must contain an object")
        if metadata.get("generation_id") != generation_id:
            raise UpdateValidationError("temporary metadata generation mismatch")
        if metadata.get("published_record_count") != bank.count:
            raise UpdateValidationError("temporary metadata record count mismatch")
        digest = hashlib.sha256(question_path.read_bytes()).hexdigest()
        if metadata.get("sha256") != digest:
            raise UpdateValidationError("temporary metadata SHA-256 mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update the local Keju question bank from NetEase."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    result = QuestionBankUpdater(args.data_dir).update()
    print(f"generation_id={result.generation_id}")
    print(f"source_url={result.source_url}")
    print(f"chunk_url={result.chunk_url}")
    print(f"module_id={result.module_id}")
    print(f"record_count={result.record_count}")
    print(f"raw_record_count={result.raw_record_count}")
    print(f"published_record_count={result.published_record_count}")
    print(f"filtered_ids={','.join(result.filtered_ids)}")
    print(f"normalized_duplicate_rate={result.normalized_duplicate_rate:.6f}")
    print(f"sha256={result.sha256}")


if __name__ == "__main__":
    main()
