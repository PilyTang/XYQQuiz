from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
from numpy.typing import NDArray
import numpy as np

from xyq_quiz import __version__
from xyq_quiz.capture.models import CapturedFrame
from xyq_quiz.runtime.state import RuntimeSnapshot


_CROP_NAMES = (
    "question.png",
    "option-a.png",
    "option-b.png",
    "option-c.png",
    "option-d.png",
)
_SENSITIVE_NAME = re.compile(
    r"token|key$|secret|password|authorization|cookie",
    re.IGNORECASE,
)
_SENSITIVE_LOG_VALUE = re.compile(
    r"(?i)\b(token|authorization|cookie|password|secret|api[_-]?key)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_REDACTED = "[REDACTED]"


@dataclass(frozen=True, slots=True)
class DiagnosticSnapshot:
    """A caller-assembled view of data already held by running services."""

    frame: CapturedFrame | None
    runtime: RuntimeSnapshot
    crops: tuple[NDArray[np.uint8], ...] = ()
    config: Any = None
    metadata: Any = None


class DiagnosticUnavailable(RuntimeError):
    pass


class DiagnosticWriter:
    def __init__(
        self,
        directory: Path,
        *,
        log_path: Path | None = None,
        log_tail_bytes: int = 200_000,
        app_root: Path | None = None,
    ) -> None:
        if log_tail_bytes < 0:
            raise ValueError("log_tail_bytes must not be negative")
        self.directory = Path(directory)
        self.log_path = Path(log_path) if log_path is not None else None
        self.log_tail_bytes = log_tail_bytes
        self.app_root = Path(app_root or self.directory.parent).resolve()

    def write(self, snapshot: DiagnosticSnapshot) -> Path:
        if snapshot.frame is None:
            raise DiagnosticUnavailable("当前没有可用画面，无法导出诊断包")
        if len(snapshot.crops) not in {0, 4, 5}:
            raise ValueError(
                "diagnostic crops must contain exactly 0, 4, or 5 images"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.directory / f"xyq-quiz-diagnostics-{stamp}-{uuid4().hex[:8]}.zip"
        temporary = path.with_suffix(".tmp")
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("frame.jpg", _encode_image(snapshot.frame.bgr, ".jpg"))
                for name, crop in zip(_CROP_NAMES, snapshot.crops, strict=False):
                    archive.writestr(name, _encode_image(crop, ".png"))
                archive.writestr("state.json", _json_bytes(snapshot.runtime))
                archive.writestr(
                    "config.json",
                    _json_bytes(_redact(snapshot.config if snapshot.config is not None else {})),
                )
                archive.writestr(
                    "metadata.json",
                    _json_bytes(snapshot.metadata if snapshot.metadata is not None else {}),
                )
                archive.writestr("app.log", self._read_log_tail())
            temporary.replace(path)
            return path
        finally:
            temporary.unlink(missing_ok=True)

    def _read_log_tail(self) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        size = self.log_path.stat().st_size
        with self.log_path.open("rb") as handle:
            handle.seek(max(0, size - self.log_tail_bytes))
            text = handle.read().decode("utf-8", errors="replace")
        return _sanitize_log_text(
            text,
            {
                str(self.app_root): "<APP_ROOT>",
                str(Path.home()): "<USERPROFILE>",
            },
        )


class EnvironmentDiagnosticWriter:
    """Write a support bundle that never contains game pixels or question rows."""

    def __init__(
        self,
        directory: Path,
        *,
        app_root: Path,
        resource_root: Path,
        log_path: Path | None = None,
        log_tail_bytes: int = 200_000,
    ) -> None:
        self.directory = Path(directory)
        self.app_root = Path(app_root).resolve()
        self.resource_root = Path(resource_root).resolve()
        self.log_path = Path(log_path) if log_path is not None else None
        self.log_tail_bytes = log_tail_bytes

    def write(self, config: Any) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.directory / f"xyq-quiz-environment-{stamp}-{uuid4().hex[:8]}.zip"
        temporary = path.with_suffix(".tmp")
        replacements = {
            str(self.app_root): "<APP_ROOT>",
            str(Path.home()): "<USERPROFILE>",
        }
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr(
                    "system.json",
                    _json_bytes(
                        {
                            "app_version": __version__,
                            "platform": platform.platform(),
                            "windows_version": platform.version(),
                            "architecture": platform.machine(),
                            "python": sys.version,
                            "frozen": bool(getattr(sys, "frozen", False)),
                        }
                    ),
                )
                archive.writestr(
                    "config.json",
                    _json_bytes(_replace_paths(_redact(config), replacements)),
                )
                archive.writestr(
                    "build-manifest-summary.json",
                    _json_bytes(self._manifest_summary()),
                )
                archive.writestr(
                    "app.log",
                    _sanitize_log_text(self._read_log_tail(), replacements),
                )
            temporary.replace(path)
            return path
        finally:
            temporary.unlink(missing_ok=True)

    def _manifest_summary(self) -> dict[str, Any]:
        path = self.resource_root / "build-manifest.json"
        if not path.is_file():
            return {"available": False}
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {"available": False, "error": str(error)}
        if not isinstance(manifest, dict):
            return {"available": False, "error": "manifest is not an object"}
        files = manifest.get("files")
        return {
            "available": True,
            "schema_version": manifest.get("schema_version"),
            "app_version": manifest.get("app_version"),
            "git_commit": manifest.get("git_commit"),
            "target": manifest.get("target"),
            "signed": manifest.get("signed"),
            "question_bank": manifest.get("question_bank", {}),
            "dependencies": manifest.get("dependencies", {}),
            "file_count": len(files) if isinstance(files, list) else None,
        }

    def _read_log_tail(self) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        size = self.log_path.stat().st_size
        with self.log_path.open("rb") as handle:
            handle.seek(max(0, size - self.log_tail_bytes))
            return handle.read().decode("utf-8", errors="replace")


def _encode_image(image: NDArray[np.uint8], suffix: str) -> bytes:
    encoded, payload = cv2.imencode(suffix, image)
    if not encoded:
        raise RuntimeError(f"failed to encode diagnostic image as {suffix}")
    return payload.tobytes()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_compatible(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_compatible(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _redact(value: Any) -> Any:
    compatible = _json_compatible(value)
    if isinstance(compatible, dict):
        return {
            key: (_REDACTED if _SENSITIVE_NAME.search(key) else _redact(item))
            for key, item in compatible.items()
        }
    if isinstance(compatible, list):
        return [_redact(item) for item in compatible]
    return compatible


def _replace_paths(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if isinstance(value, str):
        return _replace_text(value, replacements)
    return value


def _replace_text(value: str, replacements: Mapping[str, str]) -> str:
    result = value
    for source, replacement in replacements.items():
        if source:
            result = result.replace(source, replacement)
            result = result.replace(source.replace("\\", "/"), replacement)
    return result


def _sanitize_log_text(value: str, replacements: Mapping[str, str]) -> str:
    result = _replace_text(value, replacements)
    return _SENSITIVE_LOG_VALUE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        result,
    )


__all__ = [
    "DiagnosticSnapshot",
    "DiagnosticUnavailable",
    "DiagnosticWriter",
    "EnvironmentDiagnosticWriter",
]
