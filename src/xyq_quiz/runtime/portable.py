from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable
from uuid import uuid4


STATE_SCHEMA_VERSION = 1


class StateSchemaError(ValueError):
    pass


def migrate_config_document(value: Any) -> dict[str, Any]:
    document = _object(value, "config.json")
    version = _version(document, "config.json")
    if version == 0:
        web = document.get("web")
        if web is None:
            web = {}
            document["web"] = web
        if not isinstance(web, dict):
            raise StateSchemaError("config.json web must contain an object")
        host = web.get("host", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost"}:
            raise StateSchemaError("旧配置的 web.host 不是安全的本机地址")
        web["host"] = "127.0.0.1"
        document["schema_version"] = STATE_SCHEMA_VERSION
    _require_current(document, "config.json")
    web = document.get("web")
    if isinstance(web, dict):
        host = web.get("host", "127.0.0.1")
        if host != "127.0.0.1":
            raise StateSchemaError("web.host 只允许 127.0.0.1")
        web["host"] = "127.0.0.1"
    return document


def migrate_pointer_document(value: Any) -> dict[str, Any]:
    document = _object(value, "current.json")
    version = _version(document, "current.json")
    if version == 0:
        document["schema_version"] = STATE_SCHEMA_VERSION
    _require_current(document, "current.json")
    return document


def migrate_metadata_document(value: Any) -> dict[str, Any]:
    document = _object(value, "metadata.json")
    version = _version(document, "metadata.json")
    if version == 0:
        document["schema_version"] = STATE_SCHEMA_VERSION
    _require_current(document, "metadata.json")
    return document


def migrate_portable_state(config_path: Path, data_dir: Path) -> Path | None:
    """Migrate portable config/data through validated same-volume staging.

    Returns the retained backup directory when a migration was committed.
    """

    config_path = Path(config_path)
    data_dir = Path(data_dir)
    app_root = config_path.parent
    if data_dir.parent != app_root:
        raise ValueError("portable config and data must share one app root")

    raw_config = _read_json(config_path)
    migrated_config = migrate_config_document(raw_config)
    raw_pointer = _read_json(data_dir / "current.json")
    migrated_pointer = migrate_pointer_document(raw_pointer)

    metadata_updates: dict[Path, dict[str, Any]] = {}
    generations_dir = data_dir / "generations"
    if generations_dir.is_dir():
        for metadata_path in generations_dir.glob("*/metadata.json"):
            metadata_updates[metadata_path.relative_to(data_dir)] = (
                migrate_metadata_document(_read_json(metadata_path))
            )

    changed = migrated_config != raw_config or migrated_pointer != raw_pointer
    if not changed:
        changed = any(
            document != _read_json(data_dir / relative)
            for relative, document in metadata_updates.items()
        )
    if not changed:
        return None

    staging = app_root / f".state-migration-{uuid4().hex}.tmp"
    backup_parent = app_root / "state-backups"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_parent / f"state-{stamp}-{uuid4().hex[:8]}"
    try:
        staging.mkdir()
        staged_config = staging / "config.json"
        staged_data = staging / "data"
        shutil.copy2(config_path, staged_config)
        shutil.copytree(data_dir, staged_data)
        _write_json_synced(staged_config, migrated_config)
        _write_json_synced(staged_data / "current.json", migrated_pointer)
        for relative, document in metadata_updates.items():
            _write_json_synced(staged_data / relative, document)

        from xyq_quiz.config import AppConfig
        from xyq_quiz.knowledge.updater import load_current_generation

        AppConfig.load(staged_config)
        load_current_generation(staged_data)

        backup.mkdir(parents=True)
        shutil.copy2(config_path, backup / "config.json")
        os.replace(data_dir, backup / "data")
        try:
            os.replace(staged_config, config_path)
            os.replace(staged_data, data_dir)
        except Exception:
            if not data_dir.exists() and (backup / "data").exists():
                os.replace(backup / "data", data_dir)
            if (backup / "config.json").is_file():
                restore = app_root / f".config-restore-{uuid4().hex}.tmp"
                shutil.copy2(backup / "config.json", restore)
                os.replace(restore, config_path)
            raise
        return backup
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StateSchemaError(f"{name} must contain an object")
    return deepcopy(value)


def _version(document: dict[str, Any], name: str) -> int:
    version = document.get("schema_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise StateSchemaError(f"{name} has an invalid schema_version")
    if version > STATE_SCHEMA_VERSION:
        raise StateSchemaError(
            f"{name} schema_version {version} 高于程序支持的 {STATE_SCHEMA_VERSION}"
        )
    return version


def _require_current(document: dict[str, Any], name: str) -> None:
    if document.get("schema_version") != STATE_SCHEMA_VERSION:
        raise StateSchemaError(f"{name} 无法迁移到当前 schema")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_synced(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "STATE_SCHEMA_VERSION",
    "StateSchemaError",
    "migrate_config_document",
    "migrate_metadata_document",
    "migrate_portable_state",
    "migrate_pointer_document",
]
