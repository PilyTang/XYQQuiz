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
        "low_resource_mode": False,
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


def test_low_mode_caps_preview_independently_and_preserves_normal_settings(tmp_path):
    from xyq_quiz.performance.settings import runtime_performance_config
    config = AppConfig(performance=PerformanceConfig(low_resource_mode=True))
    runtime = runtime_performance_config(config)
    assert (runtime.capture.preview_fps,runtime.recognition.scan_fps) == (10,5)
    assert (config.capture.preview_fps,config.recognition.scan_fps) == (30,15)
    assert runtime.performance.ocr_backend == runtime.performance.preview_backend == "auto"
    saved = save_performance_settings(tmp_path/'config.json',ocr_backend='auto',preview_backend='auto',fallback_config=config)
    assert saved.low_resource_mode
    saved = save_performance_settings(tmp_path/'config.json',ocr_backend='cpu',preview_backend='cpu')
    assert saved.low_resource_mode  # Older clients/migrations preserve the switch.
    save_performance_settings(tmp_path/'config.json',ocr_backend='auto',preview_backend='auto',low_resource_mode=False)
    restored = runtime_performance_config(AppConfig.load(tmp_path/'config.json'))
    assert (restored.capture.preview_fps,restored.recognition.scan_fps) == (30,15)


def test_low_mode_does_not_increase_user_selected_lower_rates():
    from xyq_quiz.performance.settings import runtime_performance_config
    config = AppConfig.model_validate({'capture':{'preview_fps':5},'recognition':{'scan_fps':2},'performance':{'low_resource_mode':True}})
    runtime = runtime_performance_config(config)
    assert (runtime.capture.preview_fps,runtime.recognition.scan_fps) == (5,2)
