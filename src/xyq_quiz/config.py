from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from xyq_quiz.runtime.portable import STATE_SCHEMA_VERSION, migrate_config_document


class WindowConfig(BaseModel):
    process_names: list[str] = Field(default_factory=lambda: ["mhtab.exe"])
    class_names: list[str] = Field(default_factory=lambda: ["MHXYMainFrame"])


class CaptureConfig(BaseModel):
    preview_fps: int = Field(default=30, ge=1, le=60)
    black_frame_count: int = Field(default=10, gt=0)


class MatchConfig(BaseModel):
    question_score: int = Field(default=92, ge=0, le=100)
    question_gap: int = Field(default=5, ge=0, le=100)
    option_score: int = Field(default=90, ge=0, le=100)


class RecognitionConfig(BaseModel):
    ocr_workers: int = 1

    @field_validator("ocr_workers", mode="before")
    @classmethod
    def only_one_ocr_worker_is_supported(cls, value: object) -> object:
        if not isinstance(value, int) or isinstance(value, bool) or value != 1:
            raise ValueError(
                "ocr_workers 现在只支持 1；请改为 1 或删除该配置项"
            )
        return value


class WebConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def host_must_be_local(cls, value: str) -> str:
        if value != "127.0.0.1":
            raise ValueError("web host must be 127.0.0.1")
        return value


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = STATE_SCHEMA_VERSION
    window: WindowConfig = Field(default_factory=WindowConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    data_dir: Path = Path("data")
    layout_path: Path = Path("data/layouts/keju-default.json")
    layout_paths: list[Path] | None = None
    diagnostics_dir: Path = Path("diagnostics")
    log_path: Path = Path("logs/app.log")

    @classmethod
    def load(cls, path: Path | None) -> Self:
        if path is None:
            return cls()

        config_path = path.resolve()
        data = migrate_config_document(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        config = cls.model_validate(data)
        base_dir = config_path.parent

        updates = {
            field_name: cls._resolve_relative(getattr(config, field_name), base_dir)
            for field_name in (
                "data_dir",
                "layout_path",
                "diagnostics_dir",
                "log_path",
            )
        }
        if config.layout_paths is not None:
            updates["layout_paths"] = [
                cls._resolve_relative(item, base_dir)
                for item in config.layout_paths
            ]
        return config.model_copy(update=updates)

    @property
    def effective_layout_paths(self) -> tuple[Path, ...]:
        paths = self.layout_paths if self.layout_paths is not None else [self.layout_path]
        return tuple(paths)

    @staticmethod
    def _resolve_relative(value: Path, base_dir: Path) -> Path:
        return value if value.is_absolute() else (base_dir / value).resolve()
