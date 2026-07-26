from __future__ import annotations

from collections.abc import Callable
import ctypes
import json
import logging
import mmap
import msvcrt
from pathlib import Path
import secrets
import struct
import subprocess
import threading
import time
from typing import BinaryIO
from uuid import uuid4

import numpy as np

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturedFrame, CapturePhase, CaptureStatus, WindowTarget
from xyq_quiz.capture.service import CaptureService
from xyq_quiz.capture.video import LatestVideoHub
from xyq_quiz.capture.wgc import WGCCaptureStats
from xyq_quiz.capture.windowing import enumerate_windows, select_window
from xyq_quiz.config import AppConfig


_LOGGER = logging.getLogger(__name__)

_PIPE_MAGIC = 0x51595848
_PIPE_VERSION = 1
_PIPE_HEADER = struct.Struct("<IHHI")
_VIDEO_HEADER = struct.Struct("<QqIIB7x")
_SHARED_HEADER = struct.Struct("<IIIIIi40x")
_SHARED_SLOT = struct.Struct("<QqIIII32x")
_PREVIEW_LAYOUT = struct.Struct("<QiiiifI")
_PREVIEW_OVERLAY = struct.Struct("<fffffI")
_MAPPING_MAGIC = 0x5146524D
_MAPPING_VERSION = 1

_HELLO = 1
_READY = 2
_VIDEO = 3
_ERROR = 4
_FORCE_KEY_FRAME = 5
_STOP = 6
_DEBUG = 7
_PREVIEW_LAYOUT_MESSAGE = 8
_PREVIEW_OVERLAY_MESSAGE = 9

# A synchronous ReadFile issued by CPython can remain uncancellable in a
# vendor/driver startup hang.  Keep the rare abandoned file object alive until
# the daemon reader observes process teardown; dropping its last reference on
# the stopping thread would call close() and recreate the same hang.
_ABANDONED_PIPE_READERS: list[tuple[BinaryIO, threading.Thread]] = []


class NativePreviewError(RuntimeError):
    pass


class _SharedFrameReader:
    def __init__(self, name: str, capacity: int) -> None:
        self._capacity = capacity
        total = _SHARED_HEADER.size + 2 * (_SHARED_SLOT.size + capacity)
        self._mapping = mmap.mmap(-1, total, tagname=name, access=mmap.ACCESS_READ)
        header = _SHARED_HEADER.unpack_from(self._mapping, 0)
        if header[:5] != (
            _MAPPING_MAGIC,
            _MAPPING_VERSION,
            _SHARED_HEADER.size,
            _SHARED_SLOT.size,
            capacity,
        ):
            self._mapping.close()
            raise NativePreviewError("硬件预览共享内存协议不匹配")

    def close(self) -> None:
        self._mapping.close()

    def latest(self, last_frame_id: int) -> CapturedFrame | None:
        for _attempt in range(3):
            active = _SHARED_HEADER.unpack_from(self._mapping, 0)[5]
            if active not in (0, 1):
                return None
            slot_offset = _SHARED_HEADER.size + active * (
                _SHARED_SLOT.size + self._capacity
            )
            frame_id, _captured_qpc, width, height, stride, payload_size = (
                _SHARED_SLOT.unpack_from(self._mapping, slot_offset)
            )
            if frame_id <= last_frame_id:
                return None
            if (
                width <= 0
                or height <= 0
                or stride < width * 4
                or payload_size != stride * height
                or payload_size > self._capacity
            ):
                raise NativePreviewError("硬件预览共享帧尺寸无效")
            start = slot_offset + _SHARED_SLOT.size
            pixels = bytes(self._mapping[start : start + payload_size])
            if _SHARED_HEADER.unpack_from(self._mapping, 0)[5] != active:
                continue
            bgra = np.frombuffer(pixels, dtype=np.uint8).reshape(height, stride // 4, 4)
            bgr = np.ascontiguousarray(bgra[:, :width, :3])
            return CapturedFrame.create(frame_id, time.monotonic_ns(), bgr)
        return None


class NativePreviewSession:
    def __init__(
        self,
        *,
        helper_path: Path,
        target: WindowTarget,
        adapter_id: int,
        video_hub: LatestVideoHub,
        preview_width: int,
        preview_fps: int,
        recognition_fps: int,
        mapping_capacity: int,
        native_window: bool = False,
        on_preview_fps: Callable[[float], None] | None = None,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self._helper_path = Path(helper_path)
        self._target = target
        self._adapter_id = adapter_id
        self._video_hub = video_hub
        self._preview_width = preview_width
        self._preview_fps = preview_fps
        self._recognition_fps = recognition_fps
        self._mapping_capacity = mapping_capacity
        self._native_window = native_window
        self._on_preview_fps = on_preview_fps or (lambda _fps: None)
        self._popen = popen
        nonce = uuid4().hex
        self._pipe_name = rf"\\.\pipe\XYQQuizPreview-{nonce}"
        self._mapping_name = f"Local\\XYQQuizPreview-{nonce}"
        self._token = secrets.token_urlsafe(32)
        self._process: subprocess.Popen[bytes] | None = None
        self._pipe: BinaryIO | None = None
        self._shared: _SharedFrameReader | None = None
        self._reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._first_present = threading.Event()
        self._error: BaseException | None = None
        self._first_fps_logged = False
        self.debug_messages: list[str] = []

    def start(self, timeout: float = 15.0) -> None:
        self._stop.clear()
        self._first_present.clear()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [
            str(self._helper_path),
            "--serve",
            "--adapter",
            "auto" if self._adapter_id < 0 else str(self._adapter_id),
            "--hwnd",
            str(self._target.hwnd),
            "--pipe",
            self._pipe_name,
            "--token",
            self._token,
            "--mapping",
            self._mapping_name,
            "--preview-width",
            str(self._preview_width),
            "--preview-fps",
            str(self._preview_fps),
            "--recognition-fps",
            str(self._recognition_fps),
            "--mapping-capacity",
            str(self._mapping_capacity),
        ]
        if self._native_window:
            command.append("--native-window")
        self._process = self._popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._pipe = open(self._pipe_name, "r+b", buffering=0)
                break
            except OSError as error:
                if self._process.poll() is not None:
                    raise NativePreviewError(self._process_failure()) from error
                if time.monotonic() >= deadline:
                    raise NativePreviewError("连接硬件预览辅助进程超时") from error
                time.sleep(0.02)
        self._send(_HELLO, self._token.encode("utf-8"))
        while True:
            message_type, payload = self._read_message()
            if message_type == _READY:
                self._validate_ready(payload)
                break
            if message_type == _VIDEO:
                self._publish_video(payload)
            elif message_type == _ERROR:
                raise NativePreviewError(payload.decode("utf-8", errors="replace"))
            else:
                raise NativePreviewError("硬件预览辅助进程握手顺序无效")
            if time.monotonic() >= deadline:
                raise NativePreviewError("等待硬件预览辅助进程就绪超时")
        self._shared = _SharedFrameReader(self._mapping_name, self._mapping_capacity)
        self._video_hub.set_key_frame_requester(self.request_key_frame)
        self._reader = threading.Thread(
            target=self._read_loop,
            name="xyq-native-preview-pipe",
            daemon=True,
        )
        self._reader.start()

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            self._process = None
        pipe = self._pipe
        reader = self._reader
        if pipe is not None:
            try:
                handle = msvcrt.get_osfhandle(pipe.fileno())
                ctypes.windll.kernel32.CancelIoEx(handle, None)
            except (OSError, ValueError):
                pass
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2)
        if pipe is not None:
            if reader is not None and reader.is_alive():
                _ABANDONED_PIPE_READERS.append((pipe, reader))
            else:
                try:
                    pipe.close()
                except OSError:
                    pass
            self._pipe = None
        shared = self._shared
        if shared is not None:
            shared.close()
            self._shared = None
        self._video_hub.set_key_frame_requester(None)

    def latest(self, last_frame_id: int) -> CapturedFrame | None:
        self.raise_if_failed()
        if self._shared is None:
            return None
        return self._shared.latest(last_frame_id)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            if self._process is not None and self._process.poll() is not None:
                raise NativePreviewError(
                    f"{self._error}；{self._process_failure()}"
                ) from self._error
            raise NativePreviewError(str(self._error)) from self._error
        if self._process is not None and self._process.poll() is not None and not self._stop.is_set():
            raise NativePreviewError(self._process_failure())

    def wait_first_present(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self._stop.is_set():
            self.raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._first_present.wait(min(0.1, remaining)):
                return True
        return False

    def request_key_frame(self) -> None:
        if not self._stop.is_set():
            try:
                self._send(_FORCE_KEY_FRAME, b"")
            except OSError as error:
                if not self._stop.is_set():
                    self._error = error

    def update_preview_layout(
        self,
        *,
        owner_hwnd: int,
        x: int,
        y: int,
        width: int,
        height: int,
        scale: float,
        visible: bool,
    ) -> None:
        if not self._native_window or self._stop.is_set() or self._pipe is None:
            return
        self._send(
            _PREVIEW_LAYOUT_MESSAGE,
            _PREVIEW_LAYOUT.pack(
                max(0, int(owner_hwnd)),
                int(x),
                int(y),
                max(0, int(width)),
                max(0, int(height)),
                max(0.01, float(scale)),
                int(bool(visible)),
            ),
        )

    def update_preview_overlay(
        self,
        rect: tuple[float, float, float, float] | None,
        *,
        score: float,
        level: int,
    ) -> None:
        if not self._native_window or self._stop.is_set() or self._pipe is None:
            return
        x, y, width, height = rect or (0.0, 0.0, 0.0, 0.0)
        self._send(
            _PREVIEW_OVERLAY_MESSAGE,
            _PREVIEW_OVERLAY.pack(
                float(x),
                float(y),
                float(width),
                float(height),
                min(100.0, max(0.0, float(score))),
                max(0, int(level)),
            ),
        )

    def _read_loop(self) -> None:
        try:
            while not self._stop.is_set():
                # A synchronous ReadFile on a duplex named-pipe handle also
                # serializes writes on that handle.  Poll before reading so
                # layout/overlay controls can be sent at any time instead of
                # waiting behind an indefinitely pending read.
                if not self._message_available():
                    self._stop.wait(0.005)
                    continue
                message_type, payload = self._read_message()
                if message_type == _VIDEO:
                    self._publish_video(payload)
                elif message_type == _ERROR:
                    raise NativePreviewError(payload.decode("utf-8", errors="replace"))
                elif message_type == _DEBUG:
                    message = payload.decode("utf-8", errors="replace")
                    self.debug_messages.append(message)
                    if message == "native_preview_first_present":
                        self._first_present.set()
                    elif message.startswith("native_preview_fps="):
                        try:
                            fps = float(message.partition("=")[2])
                            self._on_preview_fps(fps)
                            if not self._first_fps_logged:
                                self._first_fps_logged = True
                                _LOGGER.info("原生预览首次帧率回报：%.1f FPS", fps)
                        except ValueError:
                            pass
        except BaseException as error:
            if not self._stop.is_set():
                self._error = error

    def _message_available(self) -> bool:
        pipe = self._pipe
        if pipe is None:
            return False
        try:
            handle = msvcrt.get_osfhandle(pipe.fileno())
        except (AttributeError, OSError, ValueError):
            return True
        available = ctypes.c_ulong()
        succeeded = ctypes.windll.kernel32.PeekNamedPipe(
            ctypes.c_void_p(handle),
            None,
            0,
            None,
            ctypes.byref(available),
            None,
        )
        if not succeeded:
            raise ctypes.WinError()
        return available.value >= _PIPE_HEADER.size

    def _publish_video(self, payload: bytes) -> None:
        if len(payload) <= _VIDEO_HEADER.size:
            raise NativePreviewError("硬件预览返回了空 H.264 帧")
        frame_id, timestamp_100ns, width, height, key_frame = _VIDEO_HEADER.unpack_from(payload)
        self._video_hub.publish(
            frame_id=frame_id,
            timestamp_us=timestamp_100ns // 10,
            width=width,
            height=height,
            key_frame=bool(key_frame),
            payload=payload[_VIDEO_HEADER.size :],
        )

    def _validate_ready(self, payload: bytes) -> None:
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise NativePreviewError("硬件预览就绪消息无效") from error
        if not isinstance(value, dict) or value.get("ok") is not True:
            raise NativePreviewError("硬件预览未能就绪")

    def _read_message(self) -> tuple[int, bytes]:
        if self._pipe is None:
            raise NativePreviewError("硬件预览管道未连接")
        header = _read_exact(self._pipe, _PIPE_HEADER.size)
        magic, version, message_type, payload_size = _PIPE_HEADER.unpack(header)
        if magic != _PIPE_MAGIC or version != _PIPE_VERSION:
            raise NativePreviewError("硬件预览管道协议不匹配")
        if payload_size > 16 * 1024 * 1024:
            raise NativePreviewError("硬件预览消息过大")
        return message_type, _read_exact(self._pipe, payload_size)

    def _send(self, message_type: int, payload: bytes) -> None:
        pipe = self._pipe
        if pipe is None:
            raise NativePreviewError("硬件预览管道未连接")
        packet = _PIPE_HEADER.pack(
            _PIPE_MAGIC,
            _PIPE_VERSION,
            message_type,
            len(payload),
        ) + payload
        with self._write_lock:
            try:
                handle = msvcrt.get_osfhandle(pipe.fileno())
            except (AttributeError, OSError, ValueError):
                # In-memory streams are useful in protocol unit tests.  The
                # production Windows path below deliberately bypasses the
                # CPython FileIO lock held by the blocking reader thread.
                pipe.write(packet)
                pipe.flush()
                return
            written = ctypes.c_ulong()
            buffer = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
            succeeded = ctypes.windll.kernel32.WriteFile(
                ctypes.c_void_p(handle),
                buffer,
                len(packet),
                ctypes.byref(written),
                None,
            )
            if not succeeded:
                raise ctypes.WinError()
            if written.value != len(packet):
                raise NativePreviewError("硬件预览管道写入不完整")

    def _process_failure(self) -> str:
        process = self._process
        if process is None:
            return "硬件预览辅助进程未启动"
        details: list[str] = []
        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                output = stream.read()
            except OSError:
                continue
            detail = output.decode("utf-8", errors="replace").strip()
            if detail:
                details.append(detail)
        detail = "；".join(details)
        return f"硬件预览辅助进程已退出（{process.returncode}）" + (
            f"：{detail}" if detail else ""
        )


class ResilientNativeCaptureService:
    """Run native display and low-rate CPU OCR capture as independent paths."""

    def __init__(
        self,
        config: AppConfig,
        hub: LatestFrameHub,
        video_hub: LatestVideoHub,
        *,
        adapter_id: int,
        helper_path: Path | None = None,
        on_success: Callable[[], None] | None = None,
        on_failure: Callable[[str], None] | None = None,
        on_preview_fps: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config
        self._hub = hub
        self._video_hub = video_hub
        self._adapter_id = adapter_id
        if helper_path is None:
            raise ValueError("native preview helper path is required")
        self._helper_path = Path(helper_path)
        self._on_success = on_success or (lambda: None)
        self._on_failure = on_failure or (lambda _reason: None)
        self._on_preview_fps = on_preview_fps or (lambda _fps: None)
        # GPU preview never reads its D3D11 texture back through Python.  A
        # separate low-rate WGC session supplies full-resolution OCR frames;
        # it also remains available as the CPU preview fallback if the native
        # renderer fails during this run.
        self._ocr_capture = CaptureService(
            config,
            hub,
            capture_fps=min(5, config.recognition.scan_fps),
        )
        self._cpu_fallback: CaptureService | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._session: NativePreviewSession | None = None
        self._fallback_active = False
        self._status = CaptureStatus(CapturePhase.WAITING_FOR_WINDOW)
        self._owner_hwnd = 0
        self._preview_layout = (0, 0, 0, 0, 1.0, False)
        self._preview_overlay: tuple[tuple[float, float, float, float] | None, float, int] = (
            None,
            0.0,
            0,
        )

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._fallback_active = False
            self._status = CaptureStatus(CapturePhase.WAITING_FOR_WINDOW)
            self._worker = threading.Thread(
                target=self._run,
                name="xyq-native-capture-service",
                daemon=True,
            )
            self._ocr_capture.start()
            self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            session = self._session
            cpu_fallback = self._cpu_fallback
            worker = self._worker
        if session is not None:
            session.stop()
        self._ocr_capture.stop()
        if cpu_fallback is not None:
            cpu_fallback.stop()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3)
            if worker.is_alive():
                raise RuntimeError("native capture worker did not stop within three seconds")

    def status(self) -> CaptureStatus:
        with self._lock:
            cpu_fallback = self._cpu_fallback
        if cpu_fallback is not None:
            return cpu_fallback.status()
        return self._ocr_capture.status()

    def capture_stats(self) -> WGCCaptureStats:
        with self._lock:
            cpu_fallback = self._cpu_fallback
        if cpu_fallback is not None:
            return cpu_fallback.capture_stats()
        return self._ocr_capture.capture_stats()

    @property
    def native_preview(self) -> bool:
        with self._lock:
            return not self._fallback_active

    def set_preview_owner(self, hwnd: int) -> None:
        with self._lock:
            self._owner_hwnd = max(0, int(hwnd))
            session = self._session
            layout = self._preview_layout
        if session is not None:
            self._send_layout(session, layout)

    def set_preview_layout(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        scale: float,
        visible: bool,
    ) -> None:
        layout = (
            int(x),
            int(y),
            max(0, int(width)),
            max(0, int(height)),
            max(0.01, float(scale)),
            bool(visible),
        )
        with self._lock:
            self._preview_layout = layout
            session = self._session
        if session is not None:
            self._send_layout(session, layout)

    def set_preview_overlay(
        self,
        rect: tuple[float, float, float, float] | None,
        score: float,
        level: int,
    ) -> None:
        overlay = (rect, float(score), int(level))
        with self._lock:
            self._preview_overlay = overlay
            session = self._session
        if session is not None:
            session.update_preview_overlay(rect, score=score, level=level)

    def _send_layout(
        self,
        session: NativePreviewSession,
        layout: tuple[int, int, int, int, float, bool],
    ) -> None:
        with self._lock:
            owner_hwnd = self._owner_hwnd
        x, y, width, height, scale, visible = layout
        session.update_preview_layout(
            owner_hwnd=owner_hwnd,
            x=x,
            y=y,
            width=width,
            height=height,
            scale=scale,
            visible=visible,
        )

    def _run(self) -> None:
        try:
            target: WindowTarget | None = None
            while not self._stop.is_set() and target is None:
                target = select_window(
                    enumerate_windows(),
                    self._config.window.process_names,
                    self._config.window.class_names,
                )
                if target is None:
                    self._stop.wait(0.5)
            if target is None:
                return
            session = NativePreviewSession(
                helper_path=self._helper_path,
                target=target,
                adapter_id=self._adapter_id,
                video_hub=self._video_hub,
                preview_width=1024,
                preview_fps=self._config.capture.preview_fps,
                recognition_fps=0,
                mapping_capacity=3840 * 2160 * 4,
                native_window=True,
                on_preview_fps=self._on_preview_fps,
            )
            with self._lock:
                self._session = session
            session.start()
            with self._lock:
                layout = self._preview_layout
                overlay = self._preview_overlay
            self._send_layout(session, layout)
            session.update_preview_overlay(
                overlay[0],
                score=overlay[1],
                level=overlay[2],
            )
            if not session.wait_first_present(20.0):
                if self._stop.is_set():
                    return
                raise NativePreviewError("原生预览在二十秒内没有完成首次显示")
            if self._stop.is_set():
                return
            with self._lock:
                self._status = CaptureStatus(CapturePhase.CAPTURING, target)
            self._on_success()
            while not self._stop.wait(0.1):
                session.raise_if_failed()
        except Exception as error:
            if self._stop.is_set():
                return
            reason = f"Windows 硬件预览运行失败：{error}"
            _LOGGER.exception("%s；本次运行仅将预览回退 CPU/I420", reason)
            self._on_failure(reason)
            self._video_hub.clear()
            self._ocr_capture.stop()
            if self._stop.is_set():
                return
            cpu_fallback = CaptureService(self._config, self._hub)
            with self._lock:
                self._fallback_active = True
                self._cpu_fallback = cpu_fallback
            cpu_fallback.start()
        finally:
            with self._lock:
                session = self._session
                self._session = None
            if session is not None:
                session.stop()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise NativePreviewError("硬件预览管道已断开")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "NativePreviewError",
    "NativePreviewSession",
    "ResilientNativeCaptureService",
]
