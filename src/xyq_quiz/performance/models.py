from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BackendCapability(StrEnum):
    OCR = "ocr"
    PREVIEW = "preview"


@dataclass(frozen=True, slots=True)
class GraphicsAdapter:
    device_id: int
    name: str
    vendor_id: int
    device_id_hex: int
    dedicated_video_memory: int
    is_software: bool = False


@dataclass(frozen=True, slots=True)
class BackendOption:
    value: str
    label: str
    capability: BackendCapability
    available: bool
    selectable: bool
    reason: str | None = None
    device_id: int | None = None


@dataclass(frozen=True, slots=True)
class BackendRuntimeState:
    requested: str
    effective: str
    label: str
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    ocr: BackendRuntimeState
    preview: BackendRuntimeState
    pending_ocr: str
    pending_preview: str
    ocr_options: tuple[BackendOption, ...]
    preview_options: tuple[BackendOption, ...]
    probing: bool = False
    benchmark_status: str = "idle"
    canvas_fps: float | None = None
