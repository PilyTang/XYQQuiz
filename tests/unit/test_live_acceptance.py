from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import xyq_quiz.acceptance.live as live_module

from xyq_quiz.acceptance.live import (
    LiveAcceptanceError,
    LiveAcceptanceReport,
    ensure_websocket_backend,
    main,
    _collect_report,
    _observed_freshness_threshold_ms,
    _target_presence_at_end,
)
from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import (
    CapturedFrame,
    CapturePhase,
    CaptureStatus,
    Rect,
    WindowTarget,
)
from xyq_quiz.capture.wgc import WGCCaptureStats
from xyq_quiz.runtime.state import RuntimePhase, RuntimeStore


def _report(*, nonblack: int = 1, black: int = 0) -> LiveAcceptanceReport:
    return LiveAcceptanceReport(
        seconds=10.0,
        target={"hwnd": 123, "process_name": "mhtab.exe"},
        native_frame_arrivals=12,
        hub_frame_arrivals=10,
        content_changes=1,
        capture_arrival_hz=1.2,
        nonblack_frames=nonblack,
        black_frames=black,
        frame_dimensions=((2072, 1662),),
        mean_bgr_min=103.0 if nonblack else 0.0,
        mean_bgr_max=103.0 if nonblack else 0.0,
        runtime_phases=("MONITORING",),
        layout_seen=False,
        overlay_seen=False,
        final_overlay=None,
        clear_messages=("dialog_missing",),
        http_ok=True,
        frames_websocket_ok=True,
        state_websocket_ok=True,
        rss_start_mb=100.0,
        rss_end_mb=101.0,
        rss_delta_mb=1.0,
        process_cpu_percent=3.0,
        capture_phases=("CAPTURING",),
        capture_final_phase="CAPTURING",
        capture_running=True,
        capture_hwnd=123,
        capture_status_hwnd=123,
        final_frame_age_ms=20.0,
        freshness_threshold_ms=100.0,
        capture_status_event_gap=False,
        target_present_at_end=True,
        target_enumeration_error=None,
        sampling_elapsed_ns=10_000_000_000,
    )


def _required_args() -> list[str]:
    return [
        "--seconds", "10",
        "--process-name", "mhtab.exe",
        "--class-name", "MHXYMainFrame",
    ]


def test_websocket_backend_check_fails_before_live_start_when_missing() -> None:
    try:
        ensure_websocket_backend(find_spec=lambda _name: None)
    except LiveAcceptanceError as error:
        assert "websockets" in str(error)
    else:
        raise AssertionError("missing backend must fail before WGC starts")


def test_cli_reports_window_not_found_without_false_pass(capsys) -> None:
    def fail(**_kwargs: object) -> LiveAcceptanceReport:
        raise LiveAcceptanceError("未找到符合规则的目标窗口")

    assert main(_required_args(), runner=fail) == 2
    output = capsys.readouterr()
    assert "未找到" in output.err
    assert "通过" not in output.out


def test_cli_rejects_black_capture_and_still_emits_machine_readable_report(
    capsys,
) -> None:
    assert main(_required_args(), runner=lambda **_kwargs: _report(nonblack=0, black=4)) == 2

    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["nonblack_frames"] == 0
    assert payload["negative_sample_pass"] is False
    assert payload["fps_target_met"] is None
    assert "黑屏" in output.err


def test_cli_reports_startup_timeout_and_runs_no_fake_fps_claim(capsys) -> None:
    def timeout(**_kwargs: object) -> LiveAcceptanceReport:
        raise LiveAcceptanceError("等待首帧超时")

    assert main(_required_args(), runner=timeout) == 2
    output = capsys.readouterr()
    assert "超时" in output.err
    assert "25" not in output.out


def test_static_negative_screen_passes_only_the_negative_sample_boundary(capsys) -> None:
    assert main(_required_args(), runner=lambda **_kwargs: _report()) == 0

    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["negative_sample_pass"] is True
    assert payload["positive_ocr_validated"] is False
    assert payload["fps_target_met"] is None
    assert "正题/OCR 未验收" in output.err
    assert "25-30 FPS 已达标" not in output.err


def test_negative_pass_rejects_window_disappearance_after_first_frame() -> None:
    report = replace(
        _report(),
        capture_phases=("CAPTURING", "WAITING_FOR_WINDOW"),
        capture_final_phase="WAITING_FOR_WINDOW",
        capture_running=False,
        capture_hwnd=None,
    )

    assert report.negative_sample_pass is False


def test_negative_pass_rejects_capture_that_stops_after_first_frame() -> None:
    report = replace(_report(), capture_running=False)

    assert report.negative_sample_pass is False


def test_negative_pass_rejects_capture_switching_to_another_hwnd() -> None:
    report = replace(_report(), capture_hwnd=456)

    assert report.negative_sample_pass is False


def test_negative_pass_rejects_status_switching_to_another_hwnd() -> None:
    report = replace(_report(), capture_status_hwnd=456)

    assert report.negative_sample_pass is False


def test_negative_pass_rejects_stale_final_frame() -> None:
    report = replace(_report(), final_frame_age_ms=99_999.0)

    assert report.negative_sample_pass is False


def test_negative_pass_requires_two_native_and_one_hub_arrivals() -> None:
    assert replace(_report(), native_frame_arrivals=0).negative_sample_pass is False
    assert replace(_report(), native_frame_arrivals=1).negative_sample_pass is False
    assert replace(_report(), hub_frame_arrivals=0).negative_sample_pass is False


def test_observed_freshness_threshold_has_frequency_based_bounds() -> None:
    assert _observed_freshness_threshold_ms(10.0, 0) is None
    assert _observed_freshness_threshold_ms(10.0, 300) == 500.0
    assert _observed_freshness_threshold_ms(3.0, 5) == 1800.0
    assert _observed_freshness_threshold_ms(10.0, 2) == 2000.0


def test_negative_pass_rejects_missing_target_or_enumeration_failure() -> None:
    assert replace(_report(), target_present_at_end=False).negative_sample_pass is False
    assert replace(
        _report(),
        target_present_at_end=False,
        target_enumeration_error="EnumWindows failed",
    ).negative_sample_pass is False


def test_end_target_enumeration_is_explicit_for_present_missing_and_error() -> None:
    target = WindowTarget(
        hwnd=123,
        title="game",
        process_id=1,
        process_name="mhtab.exe",
        class_name="MHXYMainFrame",
        rect=Rect(0, 0, 100, 80),
    )
    other = replace(target, hwnd=456)

    assert _target_presence_at_end(target, lambda: [other, target]) == (True, None)
    assert _target_presence_at_end(target, lambda: [other]) == (False, None)

    def fail() -> list[WindowTarget]:
        raise OSError("EnumWindows failed")

    present, error = _target_presence_at_end(target, fail)
    assert present is False
    assert error == "OSError: EnumWindows failed"


def test_negative_pass_rejects_capture_status_history_gap() -> None:
    assert replace(_report(), capture_status_event_gap=True).negative_sample_pass is False


def test_negative_pass_rejects_transient_capture_error_even_if_final_state_recovers() -> None:
    report = replace(
        _report(),
        capture_phases=("CAPTURING", "ERROR", "CAPTURING"),
    )

    assert report.negative_sample_pass is False


def test_hub_arrivals_count_only_frames_newer_than_sampling_start() -> None:
    target = WindowTarget(
        hwnd=123,
        title="game",
        process_id=1,
        process_name="mhtab.exe",
        class_name="MHXYMainFrame",
        rect=Rect(0, 0, 100, 80),
    )
    hub = LatestFrameHub()
    hub.publish(CapturedFrame.create(10, 10, np.full((8, 10, 3), 80, np.uint8)))
    runtime = RuntimeStore()
    runtime.set_phase(RuntimePhase.MONITORING)

    class Capture:
        stats_calls = 0
        after_cursor: int | None = None

        def status(self) -> CaptureStatus:
            return CaptureStatus(CapturePhase.CAPTURING, target)

        def capture_stats(self) -> WGCCaptureStats:
            self.stats_calls += 1
            return WGCCaptureStats(
                frame_count=10 if self.stats_calls == 1 else 12,
                frame_age_ms=20.0,
                content_change_count=1,
                running=True,
                hwnd=123,
            )

        def status_event_cursor(self) -> int:
            return 42

        def status_sampling_start(self):
            return SimpleNamespace(
                cursor=40,
                current_status=CaptureStatus(CapturePhase.CAPTURING, target),
            )

        def status_events_after(self, cursor: int):
            self.after_cursor = cursor
            events = ()
            if cursor == 40:
                events = (
                    SimpleNamespace(
                        sequence=41,
                        status=CaptureStatus(CapturePhase.ERROR, target, "boom"),
                    ),
                    SimpleNamespace(
                        sequence=42,
                        status=CaptureStatus(CapturePhase.CAPTURING, target),
                    ),
                )
            return SimpleNamespace(events=events, gap=False)

    services = SimpleNamespace(capture=Capture(), runtime=runtime, hub=hub)

    class WebMonitor:
        began: tuple[float, int] | None = None
        frozen = False

        def begin_sample(self, *, sample_started_at: float, minimum_frame_id: int) -> None:
            self.began = (sample_started_at, minimum_frame_id)

        def freeze_sample(self) -> None:
            self.frozen = True

        def finish_sample(self, *, sample_ended_at: float):
            assert self.frozen is True
            assert sample_ended_at == 0.03
            return SimpleNamespace(
                preview=SimpleNamespace(
                    packet_count=2,
                    unique_frames=2,
                    frame_ids=(11, 12),
                    first_received_s=0.01,
                    last_received_s=0.03,
                    preview_hz=50.0,
                    duplicate_frames=0,
                    out_of_order_frames=0,
                    invalid_packets=0,
                    dropped_observations=0,
                ),
                frames_connected=True,
                state_connected=True,
                frames_disconnected=False,
                state_disconnected=False,
                state_packets=1,
            )

    web_monitor = WebMonitor()

    now = 0.0
    next_frame_id = 11

    def clock_ns() -> int:
        return round(now * 1_000_000_000)

    def sleep(interval: float) -> None:
        nonlocal now, next_frame_id
        now += interval
        if next_frame_id <= 12:
            hub.publish(
                CapturedFrame.create(
                    next_frame_id,
                    next_frame_id,
                    np.full((8, 10, 3), 80, np.uint8),
                )
            )
            next_frame_id += 1

    report = _collect_report(
        services,
        target,
        0.03,
        http_ok=True,
        frames_websocket_ok=True,
        state_websocket_ok=True,
        web_monitor=web_monitor,
        clock_ns=clock_ns,
        sleeper=sleep,
        window_enumerator=lambda: [target],
    )

    assert report.hub_frame_arrivals == 2
    assert report.hub_arrival_hz == 66.667
    assert report.sampling_elapsed_ns == 30_000_000
    assert report.nonblack_frames == 2
    assert report.preview_ws_packet_count == 2
    assert report.preview_ws_unique_frames == 2
    assert report.preview_ws_frame_ids == (11, 12)
    assert report.preview_ws_hz == 50.0
    assert report.state_ws_packets == 1
    assert web_monitor.began == (0.0, 10)
    assert services.capture.after_cursor == 40
    assert report.capture_status_event_gap is False
    assert report.capture_phases == ("CAPTURING", "ERROR", "CAPTURING")
    assert report.negative_sample_pass is False


def test_sampling_boundaries_exclude_baseline_gap_and_post_end_native_callbacks() -> None:
    target = WindowTarget(
        hwnd=123,
        title="game",
        process_id=1,
        process_name="mhtab.exe",
        class_name="MHXYMainFrame",
        rect=Rect(0, 0, 100, 80),
    )
    hub = LatestFrameHub()
    hub.publish(CapturedFrame.create(10, -1, np.full((8, 10, 3), 80, np.uint8)))
    runtime = RuntimeStore()
    runtime.set_phase(RuntimePhase.MONITORING)
    now_ns = 0

    class Capture:
        frame_count = 50
        content_count = 10
        stats_calls = 0

        def status(self) -> CaptureStatus:
            return CaptureStatus(CapturePhase.CAPTURING, target)

        def status_sampling_start(self):
            # This callback lands after start_ns but before the start baseline;
            # the conservative ordered boundary must exclude it.
            self.frame_count += 1
            self.content_count += 1
            hub.publish(
                CapturedFrame.create(11, 1, np.full((8, 10, 3), 80, np.uint8))
            )
            return SimpleNamespace(
                cursor=0,
                current_status=CaptureStatus(CapturePhase.CAPTURING, target),
            )

        def capture_stats(self) -> WGCCaptureStats:
            self.stats_calls += 1
            snapshot = WGCCaptureStats(
                frame_count=self.frame_count,
                frame_age_ms=20.0,
                content_change_count=self.content_count,
                running=True,
                hwnd=123,
            )
            if self.stats_calls == 2:
                # These callbacks occur after the end snapshot and must not count.
                self.frame_count += 100
                self.content_count += 100
            return snapshot

        def status_events_after(self, _cursor: int):
            return SimpleNamespace(events=(), gap=False)

    capture = Capture()

    def clock_ns() -> int:
        return now_ns

    def sleep(_interval: float) -> None:
        nonlocal now_ns
        now_ns = 10_000_000_000
        capture.frame_count += 249
        capture.content_count += 200
        hub.publish(
            CapturedFrame.create(
                12,
                9_000_000_000,
                np.full((8, 10, 3), 80, np.uint8),
            )
        )

    report = _collect_report(
        SimpleNamespace(capture=capture, runtime=runtime, hub=hub),
        target,
        10.0,
        http_ok=True,
        frames_websocket_ok=True,
        state_websocket_ok=True,
        clock_ns=clock_ns,
        sleeper=sleep,
        window_enumerator=lambda: [target],
    )

    assert report.sampling_elapsed_ns == 10_000_000_000
    assert report.native_frame_arrivals == 249
    assert report.content_changes == 200
    assert report.capture_arrival_hz == 24.9
    assert report.hub_frame_arrivals == 1
    assert capture.frame_count == 400
    assert report.fps_target_met is None


def _dynamic_report(**changes: object) -> LiveAcceptanceReport:
    report = replace(
        _report(),
        seconds=10.0,
        native_frame_arrivals=332,
        hub_frame_arrivals=273,
        content_changes=300,
        capture_arrival_hz=33.2,
        hub_arrival_hz=27.3,
        preview_ws_packet_count=273,
        preview_ws_unique_frames=273,
        preview_ws_first_received_s=0.02,
        preview_ws_last_received_s=9.98,
        preview_ws_hz=27.309,
        preview_ws_duplicate_frames=0,
        preview_ws_out_of_order_frames=0,
        preview_ws_invalid_packets=0,
        preview_ws_disconnected=False,
        state_ws_packets=1,
        state_ws_disconnected=False,
        sampling_elapsed_ns=10_000_000_000,
    )
    return replace(report, **changes)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"sampling_elapsed_ns": 9_999_600_000, "preview_ws_disconnected": True}, None),
        ({"preview_ws_unique_frames": 0, "preview_ws_hz": None}, False),
        ({"preview_ws_unique_frames": 1, "preview_ws_hz": None}, False),
        ({"preview_ws_disconnected": True}, False),
        ({"preview_ws_invalid_packets": 1}, False),
        ({"preview_ws_dropped_observations": 1}, False),
        ({"frames_websocket_ok": False}, False),
        ({"state_websocket_ok": False}, False),
        ({"content_changes": 100, "preview_ws_disconnected": True}, None),
        ({}, True),
    ],
)
def test_fps_gate_semantics_table(
    changes: dict[str, object],
    expected: bool | None,
) -> None:
    assert replace(_dynamic_report(), **changes).fps_target_met is expected


def test_fps_gate_raw_25_hz_and_content_ratio_boundaries() -> None:
    exact = replace(
        _dynamic_report(),
        native_frame_arrivals=250,
        hub_frame_arrivals=250,
        content_changes=200,
        sampling_elapsed_ns=10_000_000_000,
        preview_ws_hz=25.0,
    )
    assert exact.fps_target_met is True
    assert replace(exact, native_frame_arrivals=249).fps_target_met is None
    assert replace(exact, content_changes=199).fps_target_met is None
    rounded_up = replace(
        exact,
        native_frame_arrivals=10_000,
        content_changes=7_996,
    )
    assert rounded_up.content_change_ratio == 0.8
    assert rounded_up.fps_target_met is None


def test_preview_packet_collector_excludes_duplicate_out_of_order_and_invalid_packets() -> None:
    collector = live_module._PreviewPacketCollector(max_packets=8)

    collector.observe((10).to_bytes(8, "big") + b"\xff\xd8jpeg", 1.0)
    collector.observe((10).to_bytes(8, "big") + b"\xff\xd8duplicate", 1.1)
    collector.observe((9).to_bytes(8, "big") + b"\xff\xd8old", 1.2)
    collector.observe((11).to_bytes(8, "big") + b"not-jpeg", 1.3)
    collector.observe((12).to_bytes(8, "big") + b"\xff\xd8jpeg", 2.0)

    sample = collector.snapshot(sample_started_at=0.5)
    assert sample.packet_count == 5
    assert sample.unique_frames == 2
    assert sample.frame_ids == (10, 12)
    assert sample.duplicate_frames == 1
    assert sample.out_of_order_frames == 1
    assert sample.invalid_packets == 1
    assert sample.preview_hz == 1.0


def test_preview_packet_collector_is_bounded_and_reports_overflow() -> None:
    collector = live_module._PreviewPacketCollector(max_packets=2)
    for frame_id in (1, 2, 3):
        collector.observe(frame_id.to_bytes(8, "big") + b"\xff\xd8jpeg", float(frame_id))

    sample = collector.snapshot(sample_started_at=0.0)
    assert sample.frame_ids == (2, 3)
    assert sample.dropped_observations == 1


def test_preview_packet_collector_filters_packets_outside_closed_sample_window() -> None:
    collector = live_module._PreviewPacketCollector(max_packets=8)
    collector.observe((8).to_bytes(8, "big") + b"not-jpeg", 0.9)
    collector.observe((9).to_bytes(8, "big") + b"\xff\xd8jpeg", 1.0)
    collector.observe((10).to_bytes(8, "big") + b"\xff\xd8jpeg", 2.0)
    collector.observe((11).to_bytes(8, "big") + b"not-jpeg", 2.1)

    sample = collector.snapshot(sample_started_at=1.0, sample_ended_at=2.0)
    assert sample.packet_count == 2
    assert sample.frame_ids == (9, 10)
    assert sample.invalid_packets == 0
    assert sample.preview_hz == 1.0


def test_fps_gate_needs_ten_raw_seconds_then_fails_closed_on_too_few_frames() -> None:
    assert replace(
        _dynamic_report(),
        seconds=10.0,
        sampling_elapsed_ns=9_999_600_000,
    ).fps_target_met is None
    assert replace(
        _dynamic_report(),
        seconds=10.0,
        sampling_elapsed_ns=10_000_000_000,
    ).fps_target_met is True
    assert replace(
        _dynamic_report(),
        preview_ws_packet_count=0,
        preview_ws_unique_frames=0,
        preview_ws_first_received_s=None,
        preview_ws_last_received_s=None,
        preview_ws_hz=None,
    ).fps_target_met is False
    assert replace(
        _dynamic_report(),
        preview_ws_packet_count=1,
        preview_ws_unique_frames=1,
        preview_ws_first_received_s=1.0,
        preview_ws_last_received_s=1.0,
        preview_ws_hz=None,
    ).fps_target_met is False


def test_fps_gate_requires_dynamic_native_evidence() -> None:
    assert replace(_dynamic_report(), content_changes=100).fps_target_met is None
    assert replace(
        _dynamic_report(),
        native_frame_arrivals=249,
        capture_arrival_hz=24.9,
    ).fps_target_met is None
    assert replace(
        _dynamic_report(),
        content_changes=100,
        preview_ws_unique_frames=0,
        preview_ws_hz=None,
        preview_ws_disconnected=True,
        preview_ws_invalid_packets=1,
    ).fps_target_met is None


def test_fps_gate_accepts_dynamic_native_33_and_hub_preview_27() -> None:
    report = _dynamic_report()

    assert report.fps_target_met is True
    assert report.to_json()["fps_target_met"] is True
    assert report.to_json()["content_change_ratio"] == 0.904
    assert report.to_json()["preview_ws_jpeg_valid"] is True


def test_fps_gate_allows_explicit_one_hz_scheduler_upper_tolerance() -> None:
    assert replace(
        _dynamic_report(),
        hub_frame_arrivals=310,
        hub_arrival_hz=31.0,
        preview_ws_hz=31.0,
    ).fps_target_met is True
    assert replace(
        _dynamic_report(),
        hub_frame_arrivals=311,
        hub_arrival_hz=31.1,
    ).fps_target_met is False


def test_fps_gate_fails_when_preview_encoding_delivery_is_below_25_hz() -> None:
    report = replace(
        _dynamic_report(),
        preview_ws_unique_frames=200,
        preview_ws_hz=20.0,
    )

    assert report.fps_target_met is False
    assert report.negative_sample_pass is True


def test_fps_gate_fails_closed_on_preview_disconnect_or_protocol_error() -> None:
    assert replace(_dynamic_report(), preview_ws_disconnected=True).fps_target_met is False
    assert replace(_dynamic_report(), preview_ws_invalid_packets=1).fps_target_met is False
    assert replace(_dynamic_report(), preview_ws_duplicate_frames=1).fps_target_met is False
    assert replace(_dynamic_report(), preview_ws_out_of_order_frames=1).fps_target_met is False


def test_cli_reports_dynamic_preview_fps_gate_without_claiming_positive_ocr(capsys) -> None:
    assert main(_required_args(), runner=lambda **_kwargs: _dynamic_report()) == 0

    output = capsys.readouterr()
    assert "预览链路 FPS 门禁通过" in output.err
    assert "正题/OCR 未验收" in output.err
    assert "FPS 未作达标结论" not in output.err


def test_cli_returns_nonzero_when_dynamic_preview_delivery_is_too_slow(capsys) -> None:
    report = replace(_dynamic_report(), preview_ws_hz=20.0)

    assert main(_required_args(), runner=lambda **_kwargs: report) == 2
    assert "预览链路 FPS 门禁未通过" in capsys.readouterr().err
