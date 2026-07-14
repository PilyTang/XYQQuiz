from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any
from urllib.request import urlopen

import numpy as np
import uvicorn

from xyq_quiz.acceptance.benchmark import _process_rss_mb
from xyq_quiz.capture.models import CapturePhase, WindowTarget
from xyq_quiz.capture.service import _is_black_frame
from xyq_quiz.capture.windowing import enumerate_windows, select_window
from xyq_quiz.config import AppConfig, WindowConfig
from xyq_quiz.launcher import build_services
from xyq_quiz.runtime.state import RuntimePhase
from xyq_quiz.web.app import create_app


class LiveAcceptanceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _PreviewPacketSample:
    packet_count: int
    unique_frames: int
    frame_ids: tuple[int, ...]
    first_received_s: float | None
    last_received_s: float | None
    preview_hz: float | None
    duplicate_frames: int
    out_of_order_frames: int
    invalid_packets: int
    dropped_observations: int


class _PreviewPacketCollector:
    """Thread-safe, bounded accounting for production preview packets."""

    def __init__(self, *, max_packets: int = 4096) -> None:
        if max_packets <= 0:
            raise ValueError("max_packets must be positive")
        self._packets: deque[tuple[object, float]] = deque(maxlen=max_packets)
        self._lock = threading.Lock()
        self._dropped_observations = 0

    def observe(self, packet: object, received_at: float) -> None:
        with self._lock:
            if len(self._packets) == self._packets.maxlen:
                self._dropped_observations += 1
            self._packets.append((packet, received_at))

    def snapshot(
        self,
        *,
        sample_started_at: float,
        sample_ended_at: float | None = None,
    ) -> _PreviewPacketSample:
        with self._lock:
            packets = tuple(
                (packet, received_at)
                for packet, received_at in self._packets
                if received_at >= sample_started_at
                and (sample_ended_at is None or received_at <= sample_ended_at)
            )
            dropped_observations = self._dropped_observations
        observations: list[tuple[int, float]] = []
        duplicate_frames = 0
        out_of_order_frames = 0
        invalid_packets = 0
        last_frame_id: int | None = None
        for packet, received_at in packets:
            if (
                not isinstance(packet, bytes)
                or len(packet) <= 10
                or packet[8:10] != b"\xff\xd8"
            ):
                invalid_packets += 1
                continue
            frame_id = int.from_bytes(packet[:8], "big")
            if last_frame_id is not None:
                if frame_id == last_frame_id:
                    duplicate_frames += 1
                    continue
                if frame_id < last_frame_id:
                    out_of_order_frames += 1
                    continue
            observations.append((frame_id, received_at))
            last_frame_id = frame_id
        first_s = (
            None if not observations else observations[0][1] - sample_started_at
        )
        last_s = (
            None if not observations else observations[-1][1] - sample_started_at
        )
        preview_hz = None
        if len(observations) >= 2 and last_s is not None and first_s is not None:
            interval = last_s - first_s
            if interval > 0:
                preview_hz = round((len(observations) - 1) / interval, 3)
        return _PreviewPacketSample(
            packet_count=len(packets),
            unique_frames=len(observations),
            frame_ids=tuple(frame_id for frame_id, _received_at in observations),
            first_received_s=None if first_s is None else round(first_s, 6),
            last_received_s=None if last_s is None else round(last_s, 6),
            preview_hz=preview_hz,
            duplicate_frames=duplicate_frames,
            out_of_order_frames=out_of_order_frames,
            invalid_packets=invalid_packets,
            dropped_observations=dropped_observations,
        )


@dataclass(frozen=True, slots=True)
class _LoopbackWebSample:
    preview: _PreviewPacketSample
    frames_connected: bool
    state_connected: bool
    frames_disconnected: bool
    state_disconnected: bool
    state_packets: int


class _LoopbackWebMonitor:
    """Keep both production WebSockets open throughout a live sample."""

    def __init__(self, port: int, *, startup_timeout: float) -> None:
        self._port = port
        self._startup_timeout = startup_timeout
        self._stop_event = threading.Event()
        self._sampling = threading.Event()
        self._frames_ready = threading.Event()
        self._state_ready = threading.Event()
        self._lock = threading.Lock()
        self._collector = _PreviewPacketCollector()
        self._sample_started_at = 0.0
        self._minimum_frame_id = -1
        self._state_packets = 0
        self._frames_disconnected = False
        self._state_disconnected = False
        self._errors: list[str] = []
        self._connections: list[Any] = []
        self._threads = (
            threading.Thread(
                target=self._receive_frames,
                name="xyq-live-frames-websocket",
            ),
            threading.Thread(
                target=self._receive_state,
                name="xyq-live-state-websocket",
            ),
        )

    def __enter__(self) -> _LoopbackWebMonitor:
        for thread in self._threads:
            thread.start()
        deadline = time.monotonic() + self._startup_timeout
        while not (self._frames_ready.is_set() and self._state_ready.is_set()):
            if self._errors:
                self._stop()
                raise LiveAcceptanceError(
                    "本地 WebSocket 持续采样启动失败：" + "; ".join(self._errors)
                )
            if time.monotonic() >= deadline:
                self._stop()
                raise LiveAcceptanceError("等待本地 WebSocket 首包超时")
            time.sleep(0.01)
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop()

    @property
    def active_threads(self) -> int:
        return sum(thread.is_alive() for thread in self._threads)

    def begin_sample(self, *, sample_started_at: float, minimum_frame_id: int) -> None:
        with self._lock:
            self._collector = _PreviewPacketCollector()
            self._sample_started_at = sample_started_at
            self._minimum_frame_id = minimum_frame_id
        self._sampling.set()

    def freeze_sample(self) -> None:
        self._sampling.clear()

    def finish_sample(self, *, sample_ended_at: float) -> _LoopbackWebSample:
        self.freeze_sample()
        with self._lock:
            collector = self._collector
            sample_started_at = self._sample_started_at
            state_packets = self._state_packets
            frames_disconnected = self._frames_disconnected
            state_disconnected = self._state_disconnected
        return _LoopbackWebSample(
            preview=collector.snapshot(
                sample_started_at=sample_started_at,
                sample_ended_at=sample_ended_at,
            ),
            frames_connected=self._frames_ready.is_set(),
            state_connected=self._state_ready.is_set(),
            frames_disconnected=frames_disconnected,
            state_disconnected=state_disconnected,
            state_packets=state_packets,
        )

    def _receive_frames(self) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        try:
            with connect(
                f"ws://127.0.0.1:{self._port}/ws/frames",
                open_timeout=self._startup_timeout,
                close_timeout=1,
            ) as websocket:
                self._register_connection(websocket)
                while not self._stop_event.is_set():
                    try:
                        packet = websocket.recv(timeout=0.25)
                    except TimeoutError:
                        continue
                    valid = (
                        isinstance(packet, bytes)
                        and len(packet) > 10
                        and packet[8:10] == b"\xff\xd8"
                    )
                    if valid:
                        self._frames_ready.set()
                    if self._sampling.is_set():
                        with self._lock:
                            minimum_frame_id = self._minimum_frame_id
                            collector = self._collector
                        if (
                            not valid
                            or int.from_bytes(packet[:8], "big") > minimum_frame_id
                        ):
                            collector.observe(packet, time.perf_counter())
        except ConnectionClosed:
            if not self._stop_event.is_set():
                with self._lock:
                    self._frames_disconnected = True
        except Exception as error:
            self._record_error("frames", error)

    def _receive_state(self) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        try:
            with connect(
                f"ws://127.0.0.1:{self._port}/ws/state",
                open_timeout=self._startup_timeout,
                close_timeout=1,
            ) as websocket:
                self._register_connection(websocket)
                while not self._stop_event.is_set():
                    try:
                        raw_state = websocket.recv(timeout=0.25)
                    except TimeoutError:
                        continue
                    state = json.loads(raw_state)
                    if not all(key in state for key in ("phase", "capture", "overlay")):
                        raise ValueError("state WebSocket 缺少生产协议字段")
                    with self._lock:
                        self._state_packets += 1
                    self._state_ready.set()
        except ConnectionClosed:
            if not self._stop_event.is_set():
                with self._lock:
                    self._state_disconnected = True
        except Exception as error:
            self._record_error("state", error)

    def _register_connection(self, websocket: Any) -> None:
        with self._lock:
            self._connections.append(websocket)

    def _record_error(self, channel: str, error: BaseException) -> None:
        if self._stop_event.is_set():
            return
        with self._lock:
            self._errors.append(f"{channel}: {type(error).__name__}: {error}")
            if channel == "frames":
                self._frames_disconnected = True
            else:
                self._state_disconnected = True

    def _stop(self) -> None:
        self._sampling.clear()
        self._stop_event.set()
        with self._lock:
            connections = tuple(self._connections)
        for websocket in connections:
            try:
                websocket.close()
            except Exception:
                pass
        for thread in self._threads:
            thread.join(timeout=2)
        if self.active_threads:
            raise LiveAcceptanceError("本地 WebSocket 采样线程未能清理")


@dataclass(frozen=True, slots=True)
class LiveAcceptanceReport:
    seconds: float
    target: dict[str, object]
    native_frame_arrivals: int
    hub_frame_arrivals: int
    content_changes: int
    capture_arrival_hz: float
    nonblack_frames: int
    black_frames: int
    frame_dimensions: tuple[tuple[int, int], ...]
    mean_bgr_min: float
    mean_bgr_max: float
    runtime_phases: tuple[str, ...]
    layout_seen: bool
    overlay_seen: bool
    final_overlay: tuple[float, float, float, float] | None
    clear_messages: tuple[str, ...]
    http_ok: bool
    frames_websocket_ok: bool
    state_websocket_ok: bool
    rss_start_mb: float | None
    rss_end_mb: float | None
    rss_delta_mb: float | None
    process_cpu_percent: float
    capture_phases: tuple[str, ...]
    capture_final_phase: str
    capture_running: bool
    capture_hwnd: int | None
    capture_status_hwnd: int | None
    final_frame_age_ms: float | None
    freshness_threshold_ms: float | None
    capture_status_event_gap: bool
    target_present_at_end: bool
    target_enumeration_error: str | None
    hub_arrival_hz: float = 0.0
    preview_ws_packet_count: int = 0
    preview_ws_unique_frames: int = 0
    preview_ws_frame_ids: tuple[int, ...] = ()
    preview_ws_first_received_s: float | None = None
    preview_ws_last_received_s: float | None = None
    preview_ws_hz: float | None = None
    preview_ws_duplicate_frames: int = 0
    preview_ws_out_of_order_frames: int = 0
    preview_ws_invalid_packets: int = 0
    preview_ws_dropped_observations: int = 0
    preview_ws_disconnected: bool = False
    state_ws_packets: int = 0
    state_ws_disconnected: bool = False
    sampling_elapsed_ns: int = 0

    @property
    def negative_sample_pass(self) -> bool:
        return (
            self.nonblack_frames > 0
            and self.black_frames == 0
            and self.native_frame_arrivals >= 2
            and self.hub_frame_arrivals >= 1
            and self.capture_phases == (CapturePhase.CAPTURING.value,)
            and not self.capture_status_event_gap
            and self.capture_final_phase == CapturePhase.CAPTURING.value
            and self.capture_running
            and self.capture_hwnd == self.target.get("hwnd")
            and self.capture_status_hwnd == self.target.get("hwnd")
            and self.final_frame_age_ms is not None
            and self.freshness_threshold_ms is not None
            and self.final_frame_age_ms <= self.freshness_threshold_ms
            and self.target_present_at_end
            and self.target_enumeration_error is None
            and not self.layout_seen
            and not self.overlay_seen
            and self.final_overlay is None
            and self.http_ok
            and self.frames_websocket_ok
            and self.state_websocket_ok
        )

    @property
    def content_change_ratio(self) -> float:
        if self.native_frame_arrivals <= 0:
            return 0.0
        return round(self.content_changes / self.native_frame_arrivals, 3)

    @property
    def fps_target_met(self) -> bool | None:
        if self.sampling_elapsed_ns < 10_000_000_000:
            return None
        if self.native_frame_arrivals <= 0:
            return None
        native_hz = (
            self.native_frame_arrivals * 1_000_000_000 / self.sampling_elapsed_ns
        )
        if (
            self.content_changes * 5 < self.native_frame_arrivals * 4
            or native_hz < 25.0
        ):
            return None
        if (
            not self.frames_websocket_ok
            or not self.state_websocket_ok
            or self.preview_ws_disconnected
            or self.state_ws_disconnected
            or self.preview_ws_invalid_packets > 0
            or self.preview_ws_dropped_observations > 0
            or self.preview_ws_duplicate_frames > 0
            or self.preview_ws_out_of_order_frames > 0
        ):
            return False
        if self.preview_ws_unique_frames < 2 or self.preview_ws_hz is None:
            return False
        hub_hz = self.hub_frame_arrivals * 1_000_000_000 / self.sampling_elapsed_ns
        return (
            25.0 <= hub_hz <= 31.0
            and 25.0 <= self.preview_ws_hz <= 31.0
        )

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "mode": "live_wgc",
                "negative_sample_pass": self.negative_sample_pass,
                "positive_ocr_validated": False,
                "fps_target_met": self.fps_target_met,
                "content_change_ratio": self.content_change_ratio,
                "preview_ws_jpeg_valid": (
                    self.preview_ws_unique_frames > 0
                    and self.preview_ws_invalid_packets == 0
                ),
                "fps_boundary": (
                    "至少采样 10 秒，内容变化率 >=0.8 且原生 arrival >=25 Hz；"
                    "Hub 与预览 WebSocket 均须在 25-30 Hz，允许 30 Hz 调度"
                    "最多 +1 Hz 上界容差。静态/低变化窗口不作 FPS 结论。"
                ),
                "preview_delivery_mode": "latest_only",
                "preview_ws_latest_only_drops": max(
                    0,
                    self.hub_frame_arrivals - self.preview_ws_unique_frames,
                ),
            }
        )
        return payload


def ensure_websocket_backend(
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> None:
    if find_spec("websockets") is None:
        raise LiveAcceptanceError(
            "缺少生产 WebSocket backend：请安装项目依赖 websockets>=15,<16"
        )


class _LoopbackUvicorn:
    def __init__(self, app: Any, startup_timeout: float) -> None:
        self._startup_timeout = startup_timeout
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(128)
        self.port = int(self._listener.getsockname()[1])
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                lifespan="on",
            )
        )
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [self._listener]},
            name="xyq-live-acceptance-uvicorn",
        )

    def __enter__(self) -> _LoopbackUvicorn:
        self.thread.start()
        deadline = time.monotonic() + self._startup_timeout
        while not self.server.started:
            if not self.thread.is_alive():
                self._listener.close()
                raise LiveAcceptanceError("临时 Uvicorn 在启动完成前退出")
            if time.monotonic() >= deadline:
                self._stop()
                raise LiveAcceptanceError("临时 Uvicorn/OCR 预热启动超时")
            time.sleep(0.02)
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop()

    def _stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(timeout=2)
        self._listener.close()
        if self.thread.is_alive():
            raise LiveAcceptanceError("临时 Uvicorn 未能清理线程/端口")


def run_live_acceptance(
    *,
    config_path: Path | None,
    seconds: float,
    process_names: Sequence[str],
    class_names: Sequence[str],
    startup_timeout: float,
) -> LiveAcceptanceReport:
    if seconds <= 0:
        raise LiveAcceptanceError("--seconds 必须大于 0")
    if startup_timeout <= 0:
        raise LiveAcceptanceError("--startup-timeout 必须大于 0")
    if not process_names or not class_names:
        raise LiveAcceptanceError("必须显式提供进程名和窗口类规则")
    ensure_websocket_backend()

    windows = enumerate_windows()
    target = select_window(windows, process_names, class_names)
    if target is None:
        raise LiveAcceptanceError(
            "未找到符合规则的目标窗口："
            f"process={list(process_names)}, class={list(class_names)}"
        )

    config = AppConfig.load(config_path).model_copy(
        update={
            "window": WindowConfig(
                process_names=list(process_names),
                class_names=list(class_names),
            )
        }
    )
    services = build_services(config)
    app = create_app(services)
    with _LoopbackUvicorn(app, startup_timeout) as server:
        _wait_for_first_frame(services, startup_timeout)
        capture_target = _validate_initial_capture_target(services, target)
        http_ok = _verify_loopback_http(server.port)
        with _LoopbackWebMonitor(
            server.port,
            startup_timeout=startup_timeout,
        ) as web_monitor:
            report = _collect_report(
                services,
                capture_target,
                seconds,
                http_ok=http_ok,
                frames_websocket_ok=True,
                state_websocket_ok=True,
                web_monitor=web_monitor,
            )
    return report


def _wait_for_first_frame(services: Any, timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = services.capture.status()
        if status.phase is CapturePhase.ERROR:
            raise LiveAcceptanceError(f"WGC 启动失败：{status.message or '未知错误'}")
        frame = services.hub.snapshot()
        if frame is not None:
            return frame
        time.sleep(0.02)
    raise LiveAcceptanceError("等待 WGC 首帧超时")


def _validate_initial_capture_target(
    services: Any,
    selected_target: WindowTarget,
) -> WindowTarget:
    status = services.capture.status()
    stats = services.capture.capture_stats()
    if status.phase is not CapturePhase.CAPTURING or status.target is None:
        raise LiveAcceptanceError("WGC 首帧到达后捕获状态不是 CAPTURING")
    if status.target.hwnd != selected_target.hwnd or stats.hwnd != selected_target.hwnd:
        raise LiveAcceptanceError(
            "WGC 实际捕获窗口与最初选定窗口不一致："
            f"selected={selected_target.hwnd}, status={status.target.hwnd}, stats={stats.hwnd}"
        )
    if not stats.running:
        raise LiveAcceptanceError("WGC 首帧到达后原生会话已停止")
    return status.target


def _verify_loopback_http(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3) as response:
            http_ok = response.status == 200
    except (OSError, TimeoutError, ValueError) as error:
        raise LiveAcceptanceError(f"本地 HTTP 验证失败：{error}") from error
    if not http_ok:
        raise LiveAcceptanceError("本地 HTTP 响应格式不符合生产协议")
    return True


def _collect_report(
    services: Any,
    target: WindowTarget,
    seconds: float,
    *,
    http_ok: bool,
    frames_websocket_ok: bool,
    state_websocket_ok: bool,
    web_monitor: Any | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleeper: Callable[[float], None] = time.sleep,
    window_enumerator: Callable[[], list[WindowTarget]] = enumerate_windows,
) -> LiveAcceptanceReport:
    sample_interval = 0.01
    start_rss = _process_rss_mb()
    start_cpu = time.process_time()
    started_ns = clock_ns()
    sampling_start = services.capture.status_sampling_start()
    start_capture_status = sampling_start.current_status
    start_stats = services.capture.capture_stats()
    start_frame = services.hub.snapshot()
    deadline_ns = started_ns + round(seconds * 1_000_000_000)
    last_frame_id = -1 if start_frame is None else start_frame.frame_id
    if web_monitor is not None:
        web_monitor.begin_sample(
            sample_started_at=started_ns / 1_000_000_000,
            minimum_frame_id=last_frame_id,
        )
    nonblack = 0
    black = 0
    means: list[float] = []
    dimensions: set[tuple[int, int]] = set()
    phases: list[str] = []
    clear_messages: list[str] = []
    layout_seen = False
    overlay_seen = False
    capture_phases: list[str] = []

    def observe_frame(frame: Any) -> None:
        nonlocal last_frame_id, nonblack, black
        if (
            frame is None
            or frame.frame_id <= last_frame_id
            or frame.captured_at_ns < started_ns
        ):
            return
        last_frame_id = frame.frame_id
        width, height = frame.bgr.shape[1], frame.bgr.shape[0]
        dimensions.add((width, height))
        mean = float(np.mean(frame.bgr))
        means.append(mean)
        if _is_black_frame(frame):
            black += 1
        else:
            nonblack += 1

    while clock_ns() < deadline_ns:
        capture_status = services.capture.status()
        capture_phase = capture_status.phase.value
        if not capture_phases or capture_phases[-1] != capture_phase:
            capture_phases.append(capture_phase)
        observe_frame(services.hub.snapshot())

        runtime = services.runtime.snapshot()
        phase = runtime.phase.value
        if not phases or phases[-1] != phase:
            phases.append(phase)
        if runtime.message and runtime.message not in clear_messages:
            clear_messages.append(runtime.message)
        if runtime.phase in {
            RuntimePhase.RECOGNIZING,
            RuntimePhase.ANSWERED,
            RuntimePhase.UNCERTAIN,
        }:
            layout_seen = True
        overlay_seen = overlay_seen or runtime.overlay is not None
        sleeper(sample_interval)

    observe_frame(services.hub.snapshot())
    end_stats = services.capture.capture_stats()
    if web_monitor is not None:
        web_monitor.freeze_sample()
    ended_ns = clock_ns()
    elapsed_ns = max(1, ended_ns - started_ns)
    elapsed = elapsed_ns / 1_000_000_000
    web_sample = (
        None
        if web_monitor is None
        else web_monitor.finish_sample(
            sample_ended_at=ended_ns / 1_000_000_000,
        )
    )
    process_cpu = max(0.0, (time.process_time() - start_cpu) / elapsed * 100)
    end_rss = _process_rss_mb()
    end_status = services.capture.status()
    capture_status_event_gap = False
    window = services.capture.status_events_after(sampling_start.cursor)
    capture_status_event_gap = window.gap
    capture_phases = [start_capture_status.phase.value]
    for event in window.events:
        phase = event.status.phase.value
        if capture_phases[-1] != phase:
            capture_phases.append(phase)
    final_runtime = services.runtime.snapshot()
    native_arrivals = max(0, end_stats.frame_count - start_stats.frame_count)
    hub_arrivals = nonblack + black
    content_changes = max(
        0,
        end_stats.content_change_count - start_stats.content_change_count,
    )
    rss_delta = (
        None
        if start_rss is None or end_rss is None
        else round(end_rss - start_rss, 3)
    )
    freshness_threshold_ms = _observed_freshness_threshold_ms(
        elapsed,
        native_arrivals,
    )
    target_present_at_end, target_enumeration_error = _target_presence_at_end(
        target,
        window_enumerator,
    )
    preview = None if web_sample is None else web_sample.preview
    effective_frames_websocket_ok = frames_websocket_ok and (
        web_sample is None
        or (
            web_sample.frames_connected
            and not web_sample.frames_disconnected
            and preview.invalid_packets == 0
        )
    )
    effective_state_websocket_ok = state_websocket_ok and (
        web_sample is None
        or (web_sample.state_connected and not web_sample.state_disconnected)
    )
    return LiveAcceptanceReport(
        seconds=round(elapsed, 3),
        target=_target_payload(target),
        native_frame_arrivals=native_arrivals,
        hub_frame_arrivals=hub_arrivals,
        content_changes=content_changes,
        capture_arrival_hz=round(native_arrivals / elapsed, 3),
        nonblack_frames=nonblack,
        black_frames=black,
        frame_dimensions=tuple(sorted(dimensions)),
        mean_bgr_min=round(min(means), 3) if means else 0.0,
        mean_bgr_max=round(max(means), 3) if means else 0.0,
        runtime_phases=tuple(phases),
        layout_seen=layout_seen,
        overlay_seen=overlay_seen,
        final_overlay=final_runtime.overlay,
        clear_messages=tuple(clear_messages),
        http_ok=http_ok,
        frames_websocket_ok=effective_frames_websocket_ok,
        state_websocket_ok=effective_state_websocket_ok,
        rss_start_mb=start_rss,
        rss_end_mb=end_rss,
        rss_delta_mb=rss_delta,
        process_cpu_percent=round(process_cpu, 3),
        capture_phases=tuple(capture_phases),
        capture_final_phase=end_status.phase.value,
        capture_running=end_stats.running,
        capture_hwnd=end_stats.hwnd,
        capture_status_hwnd=(
            None if end_status.target is None else end_status.target.hwnd
        ),
        final_frame_age_ms=(
            None
            if end_stats.frame_age_ms is None
            else round(end_stats.frame_age_ms, 3)
        ),
        freshness_threshold_ms=freshness_threshold_ms,
        capture_status_event_gap=capture_status_event_gap,
        target_present_at_end=target_present_at_end,
        target_enumeration_error=target_enumeration_error,
        hub_arrival_hz=round(hub_arrivals / elapsed, 3),
        preview_ws_packet_count=0 if preview is None else preview.packet_count,
        preview_ws_unique_frames=0 if preview is None else preview.unique_frames,
        preview_ws_frame_ids=() if preview is None else preview.frame_ids,
        preview_ws_first_received_s=(
            None if preview is None else preview.first_received_s
        ),
        preview_ws_last_received_s=(
            None if preview is None else preview.last_received_s
        ),
        preview_ws_hz=None if preview is None else preview.preview_hz,
        preview_ws_duplicate_frames=(
            0 if preview is None else preview.duplicate_frames
        ),
        preview_ws_out_of_order_frames=(
            0 if preview is None else preview.out_of_order_frames
        ),
        preview_ws_invalid_packets=0 if preview is None else preview.invalid_packets,
        preview_ws_dropped_observations=(
            0 if preview is None else preview.dropped_observations
        ),
        preview_ws_disconnected=(
            False if web_sample is None else web_sample.frames_disconnected
        ),
        state_ws_packets=0 if web_sample is None else web_sample.state_packets,
        state_ws_disconnected=(
            False if web_sample is None else web_sample.state_disconnected
        ),
        sampling_elapsed_ns=elapsed_ns,
    )


def _observed_freshness_threshold_ms(
    elapsed_seconds: float,
    native_frame_arrivals: int,
) -> float | None:
    if native_frame_arrivals <= 0:
        return None
    observed_period_ms = elapsed_seconds * 1000 / native_frame_arrivals
    return round(min(2000.0, max(500.0, 3 * observed_period_ms)), 3)


def _target_presence_at_end(
    target: WindowTarget,
    window_enumerator: Callable[[], list[WindowTarget]],
) -> tuple[bool, str | None]:
    try:
        present = any(
            candidate.hwnd == target.hwnd
            for candidate in window_enumerator()
        )
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return present, None


def _target_payload(target: WindowTarget) -> dict[str, object]:
    return {
        "hwnd": target.hwnd,
        "title": target.title,
        "process_id": target.process_id,
        "process_name": target.process_name,
        "class_name": target.class_name,
        "rect": asdict(target.rect),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读 WGC/网页实时验收")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--process-name", action="append", required=True)
    parser.add_argument("--class-name", action="append", required=True)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., LiveAcceptanceReport] = run_live_acceptance,
) -> int:
    args = _parser().parse_args(argv)
    try:
        report = runner(
            config_path=args.config,
            seconds=args.seconds,
            process_names=tuple(args.process_name),
            class_names=tuple(args.class_name),
            startup_timeout=args.startup_timeout,
        )
    except (LiveAcceptanceError, OSError, RuntimeError, ValueError) as error:
        print(f"实时 WGC 验收失败：{error}", file=sys.stderr)
        return 2

    payload = report.to_json()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if report.nonblack_frames == 0:
        print("实时 WGC 验收失败：捕获为黑屏或没有可分析帧。", file=sys.stderr)
        return 2
    if not report.negative_sample_pass:
        print("实时 WGC 验收未通过负样本边界；请检查状态与 overlay。", file=sys.stderr)
        return 2
    if report.fps_target_met is False:
        print(
            "当前无题屏负样本边界通过，但动态预览链路 FPS 门禁未通过；"
            "正题/OCR 未验收。",
            file=sys.stderr,
        )
        return 2
    if report.fps_target_met is True:
        print(
            "当前无题屏负样本通过，动态预览链路 FPS 门禁通过；"
            "正题/OCR 未验收，活动现场仍未验收。",
            file=sys.stderr,
        )
    else:
        print(
            "当前无题屏负样本通过：非黑、无布局、无 overlay；正题/OCR 未验收，"
            "25-30 FPS 未作达标结论。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
