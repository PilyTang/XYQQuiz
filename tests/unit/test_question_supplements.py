from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from xyq_quiz.knowledge.models import QuestionRecord, normalize_text
from xyq_quiz.knowledge.store import QuestionBank
from xyq_quiz.knowledge.supplements import (
    initialize_question_supplements,
    load_question_supplements,
    merge_question_supplements,
)
from xyq_quiz.knowledge.updater import load_current_generation


def _row(source_id: str, question: str, answer: str) -> dict[str, str]:
    return dict(source_id=source_id, question=question, answer=answer,
                normalized_question=normalize_text(question))


def _write_supplements(data_dir: Path, rows: list[dict[str, str]]) -> Path:
    root = data_dir / "supplements"
    root.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(rows, ensure_ascii=False).encode("utf-8")
    (root / "text_questions.json").write_bytes(raw)
    (root / "metadata.json").write_text(json.dumps(dict(
        schema_version=1, generation_id="tools-fixture",
        record_count=len(rows), sha256=hashlib.sha256(raw).hexdigest(),
    )), encoding="utf-8")
    return root


def test_merge_keeps_upstream_ids_deduplicates_pairs_and_preserves_answer_alternatives():
    upstream = QuestionBank([QuestionRecord(**_row("1", "题目一？", "甲"))])
    extra = QuestionBank([QuestionRecord(**row) for row in [
        _row("tools:1", "题目一", "甲"),
        _row("tools:2", "题目一", "乙"),
        _row("tools:3", "题目二", "丙"),
    ]])
    merged = merge_question_supplements(upstream, extra)
    assert [r.source_id for r in merged.records] == ["1", "tools:2", "tools:3"]
    assert [r.answer for r in merged.records_for("题目一")] == ["甲", "乙"]
    assert upstream.count == 1


def test_supplements_cannot_silently_replace_an_upstream_source_id():
    upstream = QuestionBank([QuestionRecord(**_row("1", "题目一", "甲"))])
    extra = QuestionBank([QuestionRecord(**_row("1", "题目二", "乙"))])
    with pytest.raises(ValueError, match="duplicate source_id"):
        merge_question_supplements(upstream, extra)


@pytest.mark.parametrize("damage", ["hash", "count", "normalization"])
def test_corrupt_supplements_are_rejected(tmp_path: Path, damage: str):
    root = _write_supplements(tmp_path, [_row("tools:1", "题目一", "甲")])
    metadata = json.loads((root / "metadata.json").read_text("utf-8"))
    if damage == "hash":
        (root / "text_questions.json").write_bytes(b"[]")
    elif damage == "count":
        metadata["record_count"] = 2
    else:
        raw = json.dumps([dict(_row("tools:1", "题目一", "甲"),
                               normalized_question="wrong")]).encode()
        (root / "text_questions.json").write_bytes(raw)
        metadata["sha256"] = hashlib.sha256(raw).hexdigest()
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_question_supplements(tmp_path)


def test_missing_supplements_keep_existing_bank_behavior(tmp_path: Path):
    assert load_question_supplements(tmp_path) is None


def test_bundled_tools_additions_load_and_keep_the_original_snapshot_auditable():
    root = Path(__file__).parents[2] / "data"
    current = load_current_generation(root)
    assert current.question_bank.count == 3166
    assert current.metadata["published_record_count"] == 2586
    assert current.metadata["supplement_record_count"] == 580
    assert current.metadata["effective_record_count"] == 3166
    assert QuestionBank.load(current.question_path).count == 2586
    assert current.question_bank.records_for("四川哪个地级市被称为南国灯城")[0].answer == "自贡"


def test_upgrade_seeds_missing_supplements_and_repeated_startup_does_not_write(tmp_path):
    source, target = tmp_path / "bundled", tmp_path / "old-data"
    _write_supplements(source, [_row("tools:1", "补充问题", "补充答案")])
    target.mkdir()
    (target / "current.json").write_bytes(b"original official pointer")
    assert initialize_question_supplements(target, source) == 1
    bank, _ = load_question_supplements(target)
    assert bank.records[0].answer == "补充答案"
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert initialize_question_supplements(target, source) == 0
    assert {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()} == before
    assert (target / "current.json").read_bytes() == b"original official pointer"


def test_incremental_upgrade_keeps_custom_records_alternative_answers_and_legacy_files(tmp_path):
    source, target = tmp_path / "bundled", tmp_path / "old-data"
    _write_supplements(source, [_row("tools:1", "已有问题", "甲"), _row("tools:2", "新增问题", "丙")])
    existing = _write_supplements(target, [
        _row("custom:1", "已有问题？", "甲"),
        _row("custom:2", "已有问题", "乙"),
        _row("custom:3", "自有问题", "自有答案"),
    ])
    legacy = {p.name: p.read_bytes() for p in existing.iterdir()}
    assert initialize_question_supplements(target, source) == 1
    bank, metadata = load_question_supplements(target)
    assert bank.count == 4
    assert [r.answer for r in bank.records_for("已有问题")] == ["甲", "乙"]
    assert bank.records_for("自有问题")[0].answer == "自有答案"
    assert all((existing / name).read_bytes() == raw for name, raw in legacy.items())
    first_pointer = (existing / "current.json").read_bytes()
    assert initialize_question_supplements(target, source) == 0
    assert (existing / "current.json").read_bytes() == first_pointer
    _write_supplements(source, [
        _row("tools:1", "已有问题", "甲"), _row("tools:2", "新增问题", "丙"),
        _row("tools:3", "下一版问题", "丁"),
    ])
    assert initialize_question_supplements(target, source) == 1
    next_bank, _ = load_question_supplements(target)
    assert next_bank.count == 5 and set(bank.records).issubset(set(next_bank.records))
    assert (existing / "generations" / metadata["generation_id"]).is_dir()


def test_pointer_publish_failure_keeps_old_supplements_readable_and_retry_succeeds(tmp_path, monkeypatch):
    import xyq_quiz.knowledge.supplements as module
    source, target = tmp_path / "bundled", tmp_path / "old-data"
    _write_supplements(source, [_row("tools:1", "新问题", "新答案")])
    existing = _write_supplements(target, [_row("custom:1", "旧问题", "旧答案")])
    before = {p.name: p.read_bytes() for p in existing.iterdir()}
    replace = module.os.replace
    def fail_publish(src, dst):
        if Path(dst) == existing / "current.json":
            raise OSError("simulated pointer publication failure")
        return replace(src, dst)
    monkeypatch.setattr(module.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publication failure"):
        initialize_question_supplements(target, source)
    assert load_question_supplements(target)[0].count == 1
    assert all((existing / name).read_bytes() == raw for name, raw in before.items())
    assert not (existing / "current.json").exists()
    assert not list(existing.glob(".current-*.tmp"))
    assert not list((existing / "generations").glob(".seed-*"))
    monkeypatch.setattr(module.os, "replace", replace)
    assert initialize_question_supplements(target, source) == 1
    assert load_question_supplements(target)[0].count == 2


@pytest.mark.parametrize("damaged", ["bundled", "existing"])
def test_corruption_blocks_upgrade_without_overwriting_existing_files(tmp_path, damaged):
    source, target = tmp_path / "bundled", tmp_path / "old-data"
    incoming = _write_supplements(source, [_row("tools:1", "新问题", "新答案")])
    existing = _write_supplements(target, [_row("custom:1", "旧问题", "旧答案")])
    root = incoming if damaged == "bundled" else existing
    (root / "text_questions.json").write_bytes(b"broken")
    before = {p.name: p.read_bytes() for p in existing.iterdir()}
    with pytest.raises(ValueError, match="SHA-256"):
        initialize_question_supplements(target, source)
    assert {p.name: p.read_bytes() for p in existing.iterdir()} == before


def test_generation_pointer_cannot_escape_the_supplement_directory(tmp_path):
    root = _write_supplements(tmp_path, [_row("tools:1", "有效问题", "有效答案")])
    (root / "current.json").write_text(json.dumps(dict(schema_version=1, generation_id="../../outside")))
    with pytest.raises(ValueError, match="generation_id"):
        load_question_supplements(tmp_path)


def test_supplement_source_id_collision_keeps_existing_bank(tmp_path):
    source, target = tmp_path / "bundled", tmp_path / "old-data"
    _write_supplements(source, [_row("collision", "新问题", "新答案")])
    existing = _write_supplements(target, [_row("collision", "旧问题", "旧答案")])
    before = {p.name: p.read_bytes() for p in existing.iterdir()}
    with pytest.raises(ValueError, match="duplicate source_id"):
        initialize_question_supplements(target, source)
    assert {p.name: p.read_bytes() for p in existing.iterdir()} == before


def test_seed_publication_failure_leaves_the_old_data_directory_intact(tmp_path, monkeypatch):
    import xyq_quiz.runtime.paths as paths_module
    source, target = tmp_path / "bundled", tmp_path / "old-data"
    _write_supplements(source, [_row("tools:1", "新问题", "新答案")])
    target.mkdir()
    (target / "current.json").write_bytes(b"old official pointer")
    def fail_publish(*args):
        raise PermissionError("simulated sharing denial")
    monkeypatch.setattr(paths_module, "_publish_seed", fail_publish)
    with pytest.raises(PermissionError):
        initialize_question_supplements(target, source)
    assert (target / "current.json").read_bytes() == b"old official pointer"
    assert not (target / "supplements").exists()
    assert not list(target.glob(".question-supplements-*"))
