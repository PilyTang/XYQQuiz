from __future__ import annotations

import threading
from pathlib import Path

import pytest

from xyq_quiz.capture import native as native_module
from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturePhase, CaptureStatus, Rect, WindowTarget
from xyq_quiz.capture.native import NativePreviewError, ResilientNativeCaptureService
from xyq_quiz.capture.video import LatestVideoHub
from xyq_quiz.capture.wgc import WGCCaptureStats
from xyq_quiz.config import AppConfig


class _FakeCaptureService:
    instances: list[_FakeCaptureService] = []

    def __init__(
        self,
        _config: AppConfig,
        _hub: LatestFrameHub,
        *,
        capture_fps: int | None = None,
    ) -> None:
        self.capture_fps = capture_fps
        self.started = 0
        self.stopped = 0
        self.instances.append(self)

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def status(self) -> CaptureStatus:
        return CaptureStatus(CapturePhase.CAPTURING)

    def capture_stats(self) -> WGCCaptureStats:
        return WGCCaptureStats(0, None, 0, self.started > self.stopped, 1)


class _FailingNativeSession:
    def __init__(self, **_kwargs: object) -> None:
        self.stopped = 0

    def start(self) -> None:
        raise NativePreviewError("synthetic native failure")

    def stop(self) -> None:
        self.stopped += 1


def test_native_failure_replaces_low_rate_ocr_with_full_rate_cpu_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = WindowTarget(
        hwnd=1,
        title="game",
        process_id=2,
        process_name="game.exe",
        class_name="GameWindow",
        rect=Rect(0, 0, 1280, 720),
    )
    failure = threading.Event()
    reasons: list[str] = []
    _FakeCaptureService.instances = []
    monkeypatch.setattr(native_module, "CaptureService", _FakeCaptureService)
    monkeypatch.setattr(native_module, "NativePreviewSession", _FailingNativeSession)
    monkeypatch.setattr(native_module, "enumerate_windows", lambda: [target])

    helper = tmp_path / "XYQPreviewHelper.exe"
    helper.write_bytes(b"MZ")
    service = ResilientNativeCaptureService(
        AppConfig.model_validate(
            {
                "window": {
                    "process_names": ["game.exe"],
                    "class_names": ["GameWindow"],
                }
            }
        ),
        LatestFrameHub(),
        LatestVideoHub(),
        adapter_id=0,
        helper_path=helper,
        on_failure=lambda reason: (reasons.append(reason), failure.set()),
    )

    service.start()
    assert failure.wait(1)
    assert len(_FakeCaptureService.instances) == 2
    low_rate, fallback = _FakeCaptureService.instances
    assert low_rate.capture_fps == 5
    assert low_rate.started == 1
    assert low_rate.stopped == 1
    assert fallback.capture_fps is None
    assert fallback.started == 1
    assert service.native_preview is False
    assert service.status().phase is CapturePhase.CAPTURING
    assert "synthetic native failure" in reasons[0]

    service.stop()
    assert fallback.stopped == 1
