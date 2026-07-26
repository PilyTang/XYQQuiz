from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xyq_quiz.config import AppConfig, PerformanceConfig
from xyq_quiz.performance.controller import PerformanceController
from xyq_quiz.performance.models import GraphicsAdapter
from xyq_quiz.recognition.models import OCRText


_ADAPTERS = (
    GraphicsAdapter(0, "NVIDIA Test GPU", 0x10DE, 1, 16 * 1024**3),
    GraphicsAdapter(1, "AMD Test GPU", 0x1002, 2, 2 * 1024**3),
)


class FakeEngine:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def recognize(self, _image: np.ndarray) -> OCRText:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return OCRText("ok", 1.0, 1.0)

    def recognize_region(self, *_args: object, **_kwargs: object) -> OCRText:
        return self.recognize(np.zeros((1, 1, 3), dtype=np.uint8))

    def execution_providers(self) -> tuple[tuple[str, ...], ...]:
        return (("DmlExecutionProvider", "CPUExecutionProvider"),) * 3


def _controller(
    tmp_path: Path,
    preferences: PerformanceConfig | None = None,
    *,
    provider_ok: bool = True,
    dml_factory=None,
    desktop_mode: bool = True,
) -> PerformanceController:
    config = AppConfig(
        performance=preferences or PerformanceConfig(),
    )
    return PerformanceController(
        config.performance,
        config_path=tmp_path / "config.json",
        fallback_config=config,
        desktop_mode=desktop_mode,
        adapter_provider=lambda: _ADAPTERS,
        dml_provider_probe=lambda: (
            (True, None)
            if provider_ok
            else (False, "DML provider missing")
        ),
        dml_engine_factory=dml_factory or (lambda _device_id: FakeEngine()),
        cpu_engine_factory=FakeEngine,
    )


def test_auto_starts_on_cpu_and_dml_is_hidden_until_selftest(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    before = controller.snapshot()
    assert before.ocr.requested == "auto"
    assert before.ocr.effective == "cpu"
    assert not next(
        option for option in before.ocr_options if option.value == "directml:0"
    ).selectable

    controller.start()
    assert controller.wait_for_probe()

    after = controller.snapshot()
    assert next(
        option for option in after.ocr_options if option.value == "directml:0"
    ).selectable


def test_saved_unavailable_backend_is_hidden_and_falls_back(
    tmp_path: Path,
) -> None:
    controller = _controller(
        tmp_path,
        PerformanceConfig(ocr_backend="directml:9"),
    )

    snapshot = controller.snapshot()
    assert not any(
        option.value == "directml:9" for option in snapshot.ocr_options
    )
    assert snapshot.ocr.effective == "cpu"
    assert "不存在" in (snapshot.ocr.fallback_reason or "")


def test_directml_runtime_failure_permanently_falls_back_to_cpu(
    tmp_path: Path,
) -> None:
    failing = FakeEngine(error=RuntimeError("device removed"))
    cpu = FakeEngine()
    config = AppConfig(
        performance=PerformanceConfig(ocr_backend="directml:0")
    )
    controller = PerformanceController(
        config.performance,
        config_path=tmp_path / "config.json",
        fallback_config=config,
        desktop_mode=True,
        adapter_provider=lambda: _ADAPTERS,
        dml_provider_probe=lambda: (True, None),
        dml_engine_factory=lambda _device_id: failing,
        cpu_engine_factory=lambda: cpu,
    )
    engine = controller.create_ocr_engine()

    assert engine.recognize(np.zeros((8, 8, 3), dtype=np.uint8)).text == "ok"
    assert engine.recognize(np.zeros((8, 8, 3), dtype=np.uint8)).text == "ok"

    assert failing.calls == 1
    assert cpu.calls == 2
    snapshot = controller.snapshot()
    assert snapshot.ocr.requested == "directml:0"
    assert snapshot.ocr.effective == "cpu"
    assert "device removed" in (snapshot.ocr.fallback_reason or "")


def test_save_rejects_backend_that_has_not_passed_selftest(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    with pytest.raises(ValueError, match="当前不可选择"):
        controller.save(ocr_backend="directml:0", preview_backend="cpu")


def test_save_persists_independent_pending_choices(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.start()
    assert controller.wait_for_probe()

    saved = controller.save(
        ocr_backend="directml:1",
        preview_backend="cpu",
    )

    persisted = json.loads((tmp_path / "config.json").read_text("utf-8"))
    assert saved.ocr_backend == "directml:1"
    assert controller.snapshot().pending_ocr == "directml:1"
    assert controller.snapshot().ocr.effective == "cpu"
    assert persisted["performance"] == {
        "ocr_backend": "directml:1",
        "preview_backend": "cpu",
    }


def test_external_browser_hides_unavailable_saved_hardware_preview(
    tmp_path: Path,
) -> None:
    controller = _controller(
        tmp_path,
        PerformanceConfig(preview_backend="windows_hardware:0"),
        desktop_mode=False,
    )

    snapshot = controller.snapshot()
    assert snapshot.preview.effective == "cpu"
    assert not any(
        item.value == "windows_hardware:0"
        for item in snapshot.preview_options
    )
    assert "外部浏览器" in (snapshot.preview.fallback_reason or "")


def test_canvas_fps_is_reported_in_snapshot(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    controller.record_canvas_fps(29.84)

    assert controller.snapshot().canvas_fps == 29.8
    with pytest.raises(ValueError):
        controller.record_canvas_fps(float("nan"))


def test_preview_exposes_one_monitor_automatic_option_for_all_adapters(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "native-preview.exe"
    helper.write_bytes(b"test")
    duplicate_adapters = (
        GraphicsAdapter(0, "NVIDIA GPU", 0x10DE, 0x2C02, 16 * 1024**3),
        GraphicsAdapter(1, "AMD GPU", 0x1002, 0x13C0, 2 * 1024**3),
        GraphicsAdapter(2, "NVIDIA GPU", 0x10DE, 0x2C02, 16 * 1024**3),
    )
    config = AppConfig(
        performance=PerformanceConfig(
            ocr_backend="directml:0",
            preview_backend="windows_hardware:0",
        )
    )

    def preview_probe(_helper: Path, device_id: int) -> object:
        if device_id == 0:
            raise RuntimeError("duplicate DXGI entry cannot capture")
        return object()

    controller = PerformanceController(
        config.performance,
        config_path=tmp_path / "config.json",
        fallback_config=config,
        desktop_mode=True,
        adapter_provider=lambda: duplicate_adapters,
        dml_provider_probe=lambda: (True, None),
        dml_engine_factory=lambda _device_id: FakeEngine(),
        cpu_engine_factory=FakeEngine,
        native_preview_helper=helper,
        native_preview_probe=preview_probe,
    )
    controller.start()
    assert controller.wait_for_probe()

    snapshot = controller.snapshot()
    nvidia_ocr = [
        item for item in snapshot.ocr_options if "NVIDIA GPU" in item.label
    ]
    hardware_preview = [
        item
        for item in snapshot.preview_options
        if item.value == "windows_hardware:auto"
    ]

    assert len(nvidia_ocr) == 1
    assert len(hardware_preview) == 1
    assert hardware_preview[0].selectable
    assert "自动选择显示器显卡" in hardware_preview[0].label
    assert snapshot.pending_preview == "windows_hardware:auto"


def test_live_preview_failure_preserves_saved_choice_and_reports_runtime_fallback(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "native-preview.exe"
    helper.write_bytes(b"test")
    config = AppConfig(
        performance=PerformanceConfig(
            ocr_backend="directml:0",
            preview_backend="windows_hardware:auto",
        )
    )
    controller = PerformanceController(
        config.performance,
        config_path=tmp_path / "config.json",
        fallback_config=config,
        desktop_mode=True,
        adapter_provider=lambda: _ADAPTERS,
        dml_provider_probe=lambda: (True, None),
        dml_engine_factory=lambda _device_id: FakeEngine(),
        cpu_engine_factory=FakeEngine,
        native_preview_helper=helper,
        native_preview_probe=lambda _helper, _device_id: object(),
    )
    controller._verified_preview.update({0, 1})
    assert any(
        item.value == "windows_hardware:auto"
        for item in controller.snapshot().preview_options
    )

    controller.mark_preview_failure(-1, "五秒内没有返回首个视频帧")

    snapshot = controller.snapshot()
    assert snapshot.preview.effective == "cpu"
    assert snapshot.pending_preview == "windows_hardware:auto"
    assert any(
        item.value == "windows_hardware:auto"
        for item in snapshot.preview_options
    )
