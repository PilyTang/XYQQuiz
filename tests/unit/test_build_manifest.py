from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = runpy.run_path(str(ROOT / "scripts" / "generate_build_manifest.py"))
load_question_bank_manifest = SCRIPT["load_question_bank_manifest"]


def _write_question_bank(package_root: Path, *, digest: str | None = None) -> Path:
    generation_id = "generation-1"
    data = package_root / "_internal" / "defaults" / "data"
    generation = data / "generations" / generation_id
    generation.mkdir(parents=True)
    payload = b'[{"question":"q","answer":"a"}]'
    actual_digest = hashlib.sha256(payload).hexdigest()
    (data / "current.json").write_text(
        json.dumps({"generation_id": generation_id}),
        encoding="utf-8",
    )
    (generation / "keju_questions.json").write_bytes(payload)
    (generation / "metadata.json").write_text(
        json.dumps(
            {
                "generation_id": generation_id,
                "source_url": "https://example.invalid/questions",
                "updated_at": "2026-07-14T00:00:00+00:00",
                "published_record_count": 1,
                "sha256": digest or actual_digest,
            }
        ),
        encoding="utf-8",
    )
    return generation


def test_question_bank_build_metadata_is_verified_and_exported(tmp_path: Path) -> None:
    _write_question_bank(tmp_path)

    result = load_question_bank_manifest(tmp_path)

    assert result["generation_id"] == "generation-1"
    assert result["record_count"] == 1
    assert result["source_url"] == "https://example.invalid/questions"
    assert len(result["sha256"]) == 64


def test_question_bank_build_metadata_rejects_hash_mismatch(tmp_path: Path) -> None:
    _write_question_bank(tmp_path, digest="0" * 64)

    with pytest.raises(ValueError, match="SHA-256"):
        load_question_bank_manifest(tmp_path)
