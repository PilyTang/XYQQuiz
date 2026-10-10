from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

from xyq_quiz.knowledge.models import normalize_text
from xyq_quiz.knowledge.store import QuestionBank


def load_question_supplements(
    data_dir: Path,
) -> tuple[QuestionBank, dict[str, Any]] | None:
    """Read independently stored, audited text additions to the upstream bank."""
    root = Path(data_dir) / "supplements"
    if not root.exists():
        return None
    return _load_supplement_root(root)


def _load_supplement_root(root: Path) -> tuple[QuestionBank, dict[str, Any]]:
    root = root.resolve()
    generation_id = None
    if (root / "current.json").exists():
        pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
            raise ValueError("question supplement pointer schema is invalid")
        generation_id = pointer.get("generation_id")
        if not isinstance(generation_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", generation_id,
        ):
            raise ValueError("question supplement generation_id is invalid")
        directory = (root / "generations" / generation_id).resolve()
        if directory.parent != root / "generations":
            raise ValueError("question supplement generation path escapes its root")
        root = directory
    path = root / "text_questions.json"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("question supplement metadata schema is invalid")
    if generation_id is not None and metadata.get("generation_id") != generation_id:
        raise ValueError("question supplement metadata generation_id mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise ValueError("question supplement SHA-256 mismatch")
    bank = QuestionBank.load(path)
    if metadata.get("record_count") != bank.count:
        raise ValueError("question supplement record count mismatch")
    return bank, metadata


def initialize_question_supplements(data_dir: Path, default_data_dir: Path) -> int:
    """Seed or merge bundled text additions before loading any runtime bank.

    Legacy flat files stay readable. Incremental upgrades publish a complete
    immutable generation and then switch only the supplement pointer.
    """
    from xyq_quiz.runtime.paths import _fsync_file, _publish_seed

    data_dir, default_data_dir = Path(data_dir).resolve(), Path(default_data_dir).resolve()
    if data_dir == default_data_dir:
        return 0
    incoming = load_question_supplements(default_data_dir)
    if incoming is None:
        return 0
    incoming_bank, incoming_metadata = incoming
    target = data_dir / "supplements"
    if not target.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        staged = data_dir / (".question-supplements-" + uuid4().hex)
        try:
            shutil.copytree(default_data_dir / "supplements", staged)
            _load_supplement_root(staged)
            for path in staged.rglob("*"):
                if path.is_file():
                    _fsync_file(path)
            _publish_seed(staged, target)
            if not staged.exists():
                return incoming_bank.count
        finally:
            if staged.exists():
                assert staged.resolve().parent == data_dir
                shutil.rmtree(staged)
        # A competing initializer can publish first; merge into its valid bank.

    existing_bank, existing_metadata = _load_supplement_root(target)
    merged = merge_question_supplements(existing_bank, incoming_bank)
    added = merged.count - existing_bank.count
    if not added:
        return 0
    payload = (json.dumps([asdict(r) for r in merged.records], ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    generation_id = "merged-" + digest[:24]
    metadata = dict(incoming_metadata, generation_id=generation_id,
                    record_count=merged.count, sha256=digest,
                    merged_from=[existing_metadata, incoming_metadata])
    generations = (target / "generations").resolve()
    if generations.parent != target.resolve():
        raise ValueError("question supplement generation path escapes its root")
    generations.mkdir(exist_ok=True)
    staged = generations / (".seed-" + uuid4().hex)
    pointer = target / (".current-" + uuid4().hex + ".tmp")
    try:
        staged.mkdir()
        _write_synced(staged / "text_questions.json", payload)
        _write_synced(staged / "metadata.json", _json_bytes(metadata))
        _load_supplement_root(staged)
        _publish_seed(staged, generations / generation_id)
        _, published_metadata = _load_supplement_root(generations / generation_id)
        if (published_metadata.get("generation_id") != generation_id
                or published_metadata.get("sha256") != digest):
            raise ValueError("question supplement published generation mismatch")
        if _load_supplement_root(target)[1]["sha256"] != existing_metadata["sha256"]:
            raise ValueError("question supplements changed during initialization")
        _write_synced(pointer, _json_bytes(dict(schema_version=1, generation_id=generation_id)))
        os.replace(pointer, target / "current.json")
    finally:
        if staged.exists():
            assert staged.resolve().parent == generations
            shutil.rmtree(staged)
        pointer.unlink(missing_ok=True)
    return added


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_synced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def merge_question_supplements(
    official: QuestionBank, supplement: QuestionBank,
) -> QuestionBank:
    """Keep upstream answers and only add distinct question/answer pairs.

    If upstream later includes a supplement, it takes precedence. Different
    answers remain alternatives for the existing conservative option matcher.
    """
    records = list(official.records)
    metadata = dict(official.record_metadata)
    seen = {
        (record.normalized_question, normalize_text(record.answer))
        for record in records
    }
    for record in supplement.records:
        key = (record.normalized_question, normalize_text(record.answer))
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
        metadata[record.source_id] = supplement.metadata_for(record.source_id)
    return QuestionBank(records, record_metadata=metadata)
