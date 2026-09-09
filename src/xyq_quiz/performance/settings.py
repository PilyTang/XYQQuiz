from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from xyq_quiz.config import AppConfig, PerformanceConfig
from xyq_quiz.runtime.portable import migrate_config_document


@dataclass(frozen=True, slots=True)
class SavedPerformanceSettings:
    ocr_backend: str
    preview_backend: str
    low_resource_mode: bool = False
    resource_profile_origin: str = "manual"
    resource_profile_reason: str = ""


def runtime_performance_config(config: AppConfig) -> AppConfig:
    """Cap runtime work without rewriting the user's normal-mode preferences."""
    if not config.performance.low_resource_mode:
        return config
    return config.model_copy(update={
        "capture": config.capture.model_copy(update={"preview_fps": min(10,config.capture.preview_fps)}),
        "recognition": config.recognition.model_copy(update={"scan_fps": min(5,config.recognition.scan_fps)}),
    })


def save_performance_settings(
    config_path: Path,
    *,
    ocr_backend: str,
    preview_backend: str,
    fallback_config: AppConfig | None = None,
    low_resource_mode: bool | None = None,
) -> SavedPerformanceSettings:
    validated = PerformanceConfig.model_validate(
        {
            "ocr_backend": ocr_backend,
            "preview_backend": preview_backend,
        }
    )
    path = Path(config_path).resolve()
    if path.is_file():
        document = migrate_config_document(
            json.loads(path.read_text(encoding="utf-8"))
        )
    elif fallback_config is not None:
        document = fallback_config.model_dump(mode="json")
    else:
        document = AppConfig().model_dump(mode="json")

    previous = PerformanceConfig.model_validate(document.get("performance", {}))
    manual_choice = low_resource_mode is not None
    if low_resource_mode is None:
        low_resource_mode = document.get("performance", {}).get("low_resource_mode", False)
    if not isinstance(low_resource_mode, bool):
        raise ValueError("low_resource_mode 必须是布尔值")
    validated.low_resource_mode = low_resource_mode
    validated.resource_profile_origin = "manual" if manual_choice else previous.resource_profile_origin
    validated.resource_profile_reason = "" if manual_choice else previous.resource_profile_reason
    document["performance"] = validated.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, document)
    return SavedPerformanceSettings(
        validated.ocr_backend,
        validated.preview_backend,
        validated.low_resource_mode,
        validated.resource_profile_origin,
        validated.resource_profile_reason,
    )


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    payload = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = ["SavedPerformanceSettings", "save_performance_settings"]
