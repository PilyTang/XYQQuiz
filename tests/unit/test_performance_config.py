from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from xyq_quiz.config import AppConfig, PerformanceConfig
from xyq_quiz.performance.models import GraphicsAdapter
from xyq_quiz.performance.settings import save_performance_settings


def test_performance_config_defaults_to_independent_auto_backends() -> None:
    config = AppConfig()

    assert config.performance.ocr_backend == "auto"
    assert config.performance.preview_backend == "auto"


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        ("ocr_backend", "directml:0", "directml:0"),
        ("ocr_backend", "directml:12", "directml:12"),
        ("preview_backend", "windows_hardware:auto", "windows_hardware:auto"),
        ("preview_backend", "windows_hardware:0", "windows_hardware:auto"),
        ("preview_backend", "windows_hardware:2", "windows_hardware:auto"),
    ),
)
def test_performance_config_accepts_and_migrates_backend_selection(
    field: str,
    value: str,
    expected: str,
) -> None:
    config = PerformanceConfig.model_validate({field: value})

    assert getattr(config, field) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ocr_backend", "gpu"),
        ("ocr_backend", "directml:-1"),
        ("ocr_backend", "windows_hardware:0"),
        ("preview_backend", "directml:0"),
        ("preview_backend", "windows_hardware:any-device"),
    ),
)
def test_performance_config_rejects_ambiguous_or_cross_capability_selection(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        PerformanceConfig.model_validate({field: value})


def test_save_performance_settings_preserves_other_config_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    document = AppConfig().model_dump(mode="json")
    document["match"]["question_score"] = 87
    path.write_text(json.dumps(document), encoding="utf-8")

    saved = save_performance_settings(
        path,
        ocr_backend="directml:0",
        preview_backend="cpu",
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert saved.ocr_backend == "directml:0"
    assert saved.preview_backend == "cpu"
    assert persisted["performance"] == {
        "ocr_backend": "directml:0",
        "preview_backend": "cpu",
    }
    assert persisted["match"]["question_score"] == 87


def test_graphics_adapter_keeps_directml_device_order() -> None:
    adapter = GraphicsAdapter(
        device_id=1,
        name="Example GPU",
        vendor_id=0x10DE,
        device_id_hex=0x2C02,
        dedicated_video_memory=16 * 1024**3,
    )

    assert adapter.device_id == 1
    assert adapter.name == "Example GPU"
