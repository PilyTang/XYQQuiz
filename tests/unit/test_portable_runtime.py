from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from xyq_quiz.config import AppConfig
from xyq_quiz.runtime.paths import RuntimePaths, initialize_portable_state
from xyq_quiz.runtime.portable import (
    STATE_SCHEMA_VERSION,
    StateSchemaError,
    migrate_config_document,
    migrate_metadata_document,
    migrate_portable_state,
    migrate_pointer_document,
)


def test_frozen_paths_use_executable_root_and_internal_bundle(tmp_path: Path) -> None:
    app_root = tmp_path / "便携 目录"
    paths = RuntimePaths.discover(
        executable=app_root / "XYQQuiz.exe",
        bundle_root=app_root / "_internal",
        frozen=True,
    )

    assert paths.app_root == app_root.resolve()
    assert paths.resource_root == (app_root / "_internal").resolve()
    assert paths.default_config_path == app_root / "_internal" / "defaults" / "config.json"
    assert paths.config_path == app_root / "config.json"
    assert paths.data_dir == app_root / "data"


def test_source_paths_do_not_depend_on_current_working_directory(tmp_path: Path) -> None:
    module = tmp_path / "repo" / "src" / "xyq_quiz" / "runtime" / "paths.py"
    paths = RuntimePaths.discover(module_file=module, frozen=False)

    assert paths.app_root == (tmp_path / "repo").resolve()
    assert paths.default_config_path == (tmp_path / "repo" / "config.example.json").resolve()


def test_missing_portable_state_is_seeded_without_overwriting_existing_state(
    tmp_path: Path,
) -> None:
    defaults = tmp_path / "_internal" / "defaults"
    defaults.mkdir(parents=True)
    (defaults / "config.json").write_text('{"schema_version":1}', encoding="utf-8")
    (defaults / "data").mkdir()
    (defaults / "data" / "current.json").write_text(
        '{"schema_version":1,"generation_id":"seed"}',
        encoding="utf-8",
    )
    paths = RuntimePaths.discover(
        executable=tmp_path / "XYQQuiz.exe",
        bundle_root=tmp_path / "_internal",
        frozen=True,
    )

    initialize_portable_state(paths)
    assert paths.config_path.read_text("utf-8") == '{"schema_version":1}'
    assert (paths.data_dir / "current.json").is_file()
    assert paths.logs_dir.is_dir()
    assert paths.diagnostics_dir.is_dir()

    paths.config_path.write_text("user-config", encoding="utf-8")
    (paths.data_dir / "current.json").write_text("user-data", encoding="utf-8")
    initialize_portable_state(paths)

    assert paths.config_path.read_text("utf-8") == "user-config"
    assert (paths.data_dir / "current.json").read_text("utf-8") == "user-data"


def test_legacy_documents_migrate_to_current_schema_without_mutating_input() -> None:
    config = {"web": {"host": "localhost", "port": 8877}}
    pointer = {"generation_id": "g1"}
    metadata = {"generation_id": "g1"}

    migrated_config = migrate_config_document(config)
    migrated_pointer = migrate_pointer_document(pointer)
    migrated_metadata = migrate_metadata_document(metadata)

    assert "schema_version" not in config
    assert migrated_config["schema_version"] == STATE_SCHEMA_VERSION
    assert migrated_config["web"]["host"] == "127.0.0.1"
    assert migrated_pointer["schema_version"] == STATE_SCHEMA_VERSION
    assert migrated_metadata["schema_version"] == STATE_SCHEMA_VERSION


def test_future_schema_and_non_loopback_legacy_host_are_rejected() -> None:
    with pytest.raises(StateSchemaError, match="高于程序支持"):
        migrate_pointer_document({"schema_version": 99, "generation_id": "future"})
    with pytest.raises(StateSchemaError, match="不是安全的本机地址"):
        migrate_config_document({"web": {"host": "0.0.0.0"}})


def test_app_config_loads_legacy_localhost_as_fixed_loopback(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"web": {"host": "localhost", "port": 8877}}),
        encoding="utf-8",
    )

    config = AppConfig.load(path)

    assert config.schema_version == STATE_SCHEMA_VERSION
    assert config.web.host == "127.0.0.1"
    assert config.web.port == 8877


def test_portable_state_migration_validates_staging_and_retains_backup(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"web":{"host":"localhost"}}', encoding="utf-8")
    generation = tmp_path / "data" / "generations" / "g1"
    generation.mkdir(parents=True)
    question_bytes = json.dumps(
        [
            {
                "source_id": "1",
                "question": "问题",
                "answer": "答案",
                "normalized_question": "问题",
            }
        ],
        ensure_ascii=False,
    ).encode("utf-8")
    (generation / "keju_questions.json").write_bytes(question_bytes)
    (generation / "metadata.json").write_text(
        json.dumps(
            {
                "generation_id": "g1",
                "published_record_count": 1,
                "sha256": hashlib.sha256(question_bytes).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "data" / "current.json").write_text(
        '{"generation_id":"g1"}',
        encoding="utf-8",
    )

    backup = migrate_portable_state(config_path, tmp_path / "data")

    assert backup is not None and backup.is_dir()
    assert json.loads(config_path.read_text("utf-8"))["schema_version"] == 1
    assert json.loads((tmp_path / "data" / "current.json").read_text("utf-8"))[
        "schema_version"
    ] == 1
    assert json.loads((generation / "metadata.json").read_text("utf-8"))[
        "schema_version"
    ] == 1
    assert json.loads((backup / "config.json").read_text("utf-8"))["web"][
        "host"
    ] == "localhost"


def test_future_portable_schema_fails_before_creating_backup(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"schema_version":99}', encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "current.json").write_text(
        '{"schema_version":1,"generation_id":"g1"}', encoding="utf-8"
    )

    with pytest.raises(StateSchemaError, match="高于程序支持"):
        migrate_portable_state(config, data)

    assert not (tmp_path / "state-backups").exists()
    assert json.loads(config.read_text("utf-8"))["schema_version"] == 99
