from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import threading
import time

import cv2
import numpy as np
import pytest
import uvicorn

import xyq_quiz.acceptance.live as live_module
from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturedFrame, CapturePhase, CaptureStatus
from xyq_quiz.config import MatchConfig
from xyq_quiz.runtime.state import RuntimePhase, RuntimeStore
from xyq_quiz.web.app import Services, create_app


class _Lifecycle:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def invalidate_cache(self) -> None:
        pass


class _Capture(_Lifecycle):
    def status(self) -> CaptureStatus:
        return CaptureStatus(CapturePhase.CAPTURING)


class _Pipeline:
    def warm_up(self) -> None:
        pass

    def close(self) -> None:
        pass

    def replace_matcher(self, _matcher: object) -> None:
        pass


class _Updater:
    data_dir = Path(".")


def _services(*, runtime: object | None = None) -> Services:
    hub = LatestFrameHub()
    hub.publish(CapturedFrame.create(7, 11, np.full((12, 20, 3), 80, np.uint8)))
    live_runtime = RuntimeStore()
    live_runtime.set_phase(RuntimePhase.MONITORING)
    return Services(
        hub=hub,
        runtime=live_runtime if runtime is None else runtime,  # type: ignore[arg-type]
        capture=_Capture(),
        coordinator=_Lifecycle(),
        pipeline=_Pipeline(),
        updater=_Updater(),  # type: ignore[arg-type]
        match_config=MatchConfig(),
        preview_width=10,
    )


class _ActualUvicorn:
    def __init__(self, services: Services, *, startup_timeout: float = 5.0) -> None:
        self._startup_timeout = startup_timeout
        self._thread_error: BaseException | None = None
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(128)
        self.port = int(self._listener.getsockname()[1])
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(services),
                host="127.0.0.1",
                port=self.port,
                log_level="error",
                lifespan="on",
            )
        )
        self.thread = threading.Thread(
            target=self._run,
            name="actual-uvicorn-test",
        )

    def __enter__(self) -> _ActualUvicorn:
        self.thread.start()
        deadline = time.monotonic() + self._startup_timeout
        try:
            while not self.server.started:
                if not self.thread.is_alive():
                    raise RuntimeError("actual Uvicorn exited during startup") from self._thread_error
                if time.monotonic() >= deadline:
                    raise TimeoutError("actual Uvicorn did not start")
                time.sleep(0.01)
            return self
        except BaseException:
            self._stop()
            raise

    def __exit__(self, *_args: object) -> None:
        self._stop()

    def _run(self) -> None:
        try:
            self.server.run(sockets=[self._listener])
        except BaseException as error:
            self._thread_error = error

    def _stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self._listener.close()
        assert not self.thread.is_alive()


def _connect_websocket(port: int, path: str) -> socket.socket:
    client = socket.create_connection(("127.0.0.1", port), timeout=3)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    client.sendall(request.encode("ascii"))
    response = _recv_until(client, b"\r\n\r\n")
    status_line = response.split(b"\r\n", 1)[0]
    assert status_line == b"HTTP/1.1 101 Switching Protocols", response.decode(
        "latin-1", errors="replace"
    )
    return client


def _recv_until(client: socket.socket, marker: bytes) -> bytes:
    data = bytearray()
    while marker not in data:
        # Do not consume the first WebSocket frame when Uvicorn coalesces it
        # with the HTTP 101 response in the same TCP packet.
        chunk = client.recv(1)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def _receive_server_frame(client: socket.socket) -> tuple[int, bytes]:
    first, second = _receive_exact(client, 2)
    opcode = first & 0x0F
    assert second & 0x80 == 0
    length = second & 0x7F
    if length == 126:
        length = int.from_bytes(_receive_exact(client, 2), "big")
    elif length == 127:
        length = int.from_bytes(_receive_exact(client, 8), "big")
    return opcode, _receive_exact(client, length)


def _receive_exact(client: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = client.recv(size - len(data))
        if not chunk:
            raise EOFError("WebSocket closed before a complete frame arrived")
        data.extend(chunk)
    return bytes(data)


def test_actual_uvicorn_serves_binary_frames_and_json_state_over_websockets() -> None:
    services = _services()
    with _ActualUvicorn(services) as server:
        with _connect_websocket(server.port, "/ws/frames") as frames:
            opcode, packet = _receive_server_frame(frames)
        services.hub.publish(
            CapturedFrame.create(8, 12, np.full((12, 20, 3), 90, np.uint8))
        )
        with _connect_websocket(server.port, "/ws/state") as state:
            state_opcode, state_bytes = _receive_server_frame(state)
        services.runtime.clear_question("client_closed")
        time.sleep(0.05)

    assert opcode == 2
    assert int.from_bytes(packet[:8], "big") == 7
    image = cv2.imdecode(np.frombuffer(packet[8:], np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (6, 10)
    assert state_opcode == 1
    payload = json.loads(state_bytes)
    assert payload["phase"] == "MONITORING"
    assert payload["capture"]["phase"] == "CAPTURING"
    assert payload["overlay"] is None


def test_actual_uvicorn_logs_internal_state_error_instead_of_clean_disconnect(
    capfd,
) -> None:
    base = RuntimeStore()
    base.set_phase(RuntimePhase.MONITORING)

    class RaisingRuntime:
        def snapshot(self):
            return base.snapshot()

        def wait_after(self, _version: int, _timeout: float):
            raise RuntimeError("state store failed")

    services = _services(runtime=RaisingRuntime())
    with _ActualUvicorn(services) as server:
        with _connect_websocket(server.port, "/ws/state") as state:
            opcode, _payload = _receive_server_frame(state)
            try:
                _receive_server_frame(state)
            except EOFError:
                pass
            else:
                raise AssertionError("internal error must not become a clean close frame")

    assert opcode == 1
    assert "RuntimeError: state store failed" in capfd.readouterr().err


def test_actual_uvicorn_enter_cleans_listener_when_lifespan_startup_fails() -> None:
    services = _services()

    def fail_warm_up() -> None:
        raise RuntimeError("warm-up failed")

    services.pipeline.warm_up = fail_warm_up  # type: ignore[method-assign]
    server = _ActualUvicorn(services)

    with pytest.raises((AssertionError, RuntimeError)):
        with server:
            raise AssertionError("server must not start")

    assert not server.thread.is_alive()
    assert server._listener.fileno() == -1
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", server.port), timeout=0.2)


def test_actual_uvicorn_enter_cleans_thread_and_listener_on_startup_timeout() -> None:
    services = _services()
    holder: dict[str, _ActualUvicorn] = {}

    def block_until_stopped() -> None:
        while not holder["server"].server.should_exit:
            time.sleep(0.005)

    services.pipeline.warm_up = block_until_stopped  # type: ignore[method-assign]
    server = _ActualUvicorn(services, startup_timeout=0.05)
    holder["server"] = server

    with pytest.raises(TimeoutError, match="did not start"):
        with server:
            raise AssertionError("server must not start")

    assert not server.thread.is_alive()
    assert server._listener.fileno() == -1


def test_live_monitor_receives_frames_for_whole_sample_and_cleans_threads() -> None:
    services = _services()
    monitor = None

    with _ActualUvicorn(services) as server:
        monitor = live_module._LoopbackWebMonitor(server.port, startup_timeout=3.0)
        with monitor:
            started = time.perf_counter()
            monitor.begin_sample(sample_started_at=started, minimum_frame_id=7)
            for frame_id in range(8, 20):
                services.hub.publish(
                    CapturedFrame.create(
                        frame_id,
                        frame_id,
                        np.full((12, 20, 3), 90 + frame_id, np.uint8),
                    )
                )
                time.sleep(0.015)
            services.runtime.clear_question("still_healthy")
            time.sleep(0.05)
            monitor.freeze_sample()
            services.hub.publish(
                CapturedFrame.create(20, 20, np.full((12, 20, 3), 120, np.uint8))
            )
            time.sleep(0.03)
            sample = monitor.finish_sample(sample_ended_at=time.perf_counter())

    assert monitor is not None
    assert sample.preview.packet_count >= 10
    assert sample.preview.unique_frames >= 10
    assert sample.preview.frame_ids == tuple(sorted(set(sample.preview.frame_ids)))
    assert sample.preview.frame_ids[0] >= 8
    assert sample.preview.frame_ids[-1] <= 19
    assert sample.preview.invalid_packets == 0
    assert sample.preview.duplicate_frames == 0
    assert sample.preview.out_of_order_frames == 0
    assert sample.preview.preview_hz is not None
    assert sample.frames_connected is True
    assert sample.state_connected is True
    assert sample.frames_disconnected is False
    assert sample.state_disconnected is False
    assert sample.state_packets >= 2
    assert monitor.active_threads == 0
