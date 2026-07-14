from __future__ import annotations

import hashlib
import json
import asyncio
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturedFrame, CapturePhase, CaptureStatus
from xyq_quiz.config import MatchConfig
from xyq_quiz.diagnostics import DiagnosticSnapshot, DiagnosticWriter
from xyq_quiz.knowledge.models import QuestionRecord
from xyq_quiz.knowledge.store import QuestionBank
from xyq_quiz.knowledge.updater import UpdateResult
from xyq_quiz.runtime.state import RuntimePhase, RuntimeStore
from xyq_quiz.web.app import Services, _stream_frames, _stream_state, create_app
from xyq_quiz.web.security import LocalWebSecurity, SESSION_COOKIE, TOKEN_HEADER


class LifecycleFake:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def start(self) -> None:
        self.events.append(f"{self.name}.start")

    def stop(self) -> None:
        self.events.append(f"{self.name}.stop")

    def invalidate_cache(self) -> None:
        self.events.append(f"{self.name}.invalidate_cache")


class FakeCapture(LifecycleFake):
    def status(self) -> CaptureStatus:
        return CaptureStatus(CapturePhase.CAPTURING)


class FakePipeline:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.matchers: list[object] = []

    def replace_matcher(self, matcher: object) -> None:
        self.matchers.append(matcher)

    def warm_up(self) -> None:
        self.events.append("pipeline.warm_up")

    def close(self) -> None:
        self.events.append("pipeline.close")


class FakeUpdater:
    def __init__(self, data_dir: Path, *, error: Exception | None = None) -> None:
        self.data_dir = data_dir
        self.error = error

    def update(self) -> UpdateResult:
        if self.error is not None:
            raise self.error
        current = json.loads((self.data_dir / "current.json").read_text("utf-8"))
        generation_id = current["generation_id"]
        return UpdateResult(
            generation_id=generation_id,
            source_url="fixture://source",
            chunk_url="fixture://chunk",
            module_id=7,
            record_count=1,
            raw_record_count=1,
            published_record_count=1,
            filtered_ids=(),
            normalized_duplicate_rate=0.0,
            sha256="digest",
        )


class FakeDiagnosticWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.snapshots: list[DiagnosticSnapshot] = []

    def write(self, snapshot: DiagnosticSnapshot) -> Path:
        self.snapshots.append(snapshot)
        return self.path


class RecordingWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.messages.append(payload)


class BlockingRecordingWebSocket(RecordingWebSocket):
    async def receive(self) -> dict[str, str]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send_bytes(self, _payload: bytes) -> None:
        raise AssertionError("hub failure must happen before encoding")


class ScriptedRuntime:
    def __init__(self) -> None:
        self.current = RuntimeStore()
        self.current.set_phase(RuntimePhase.MONITORING)
        self.updated = RuntimeStore()
        self.updated.set_phase(RuntimePhase.MONITORING)
        self.updated.clear_question("changed")
        self.wait_results = [None, None, self.updated.snapshot()]

    def snapshot(self):
        return self.current.snapshot()

    def wait_after(self, version: int, timeout: float):
        del version, timeout
        if self.wait_results:
            return self.wait_results.pop(0)
        raise WebSocketDisconnect()


@dataclass
class ServiceFixture:
    services: Services
    events: list[str]
    pipeline: FakePipeline


def _write_generation(data_dir: Path, generation_id: str, question: str) -> None:
    generation_dir = data_dir / "generations" / generation_id
    generation_dir.mkdir(parents=True)
    rows = [
        {
            "source_id": "1",
            "question": question,
            "answer": "新答案",
            "normalized_question": question,
        }
    ]
    question_bytes = (json.dumps(rows, ensure_ascii=False) + "\n").encode()
    (generation_dir / "keju_questions.json").write_bytes(question_bytes)
    metadata = {
        "generation_id": generation_id,
        "published_record_count": 1,
        "sha256": hashlib.sha256(question_bytes).hexdigest(),
    }
    (generation_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    (data_dir / "current.json").write_text(
        json.dumps({"generation_id": generation_id}), encoding="utf-8"
    )


def _services(tmp_path: Path, *, updater_error: Exception | None = None) -> ServiceFixture:
    _write_generation(tmp_path, "new-generation", "新问题")
    events: list[str] = []
    pipeline = FakePipeline(events)
    runtime = RuntimeStore()
    runtime.set_phase(RuntimePhase.MONITORING)
    services = Services(
        hub=LatestFrameHub(),
        runtime=runtime,
        capture=FakeCapture("capture", events),
        coordinator=LifecycleFake("coordinator", events),
        pipeline=pipeline,
        updater=FakeUpdater(tmp_path, error=updater_error),
        match_config=MatchConfig(question_score=92, question_gap=5, option_score=90),
        preview_width=4,
    )
    return ServiceFixture(services, events, pipeline)


def test_lifespan_starts_capture_then_coordinator_and_stops_in_required_order(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)

    with TestClient(create_app(fixture.services)):
        assert fixture.events == [
            "pipeline.warm_up",
            "capture.start",
            "coordinator.start",
        ]

    assert fixture.events == [
        "pipeline.warm_up",
        "capture.start",
        "coordinator.start",
        "coordinator.stop",
        "capture.stop",
        "pipeline.close",
    ]


def test_lifespan_propagates_warm_up_failure_before_capture_starts(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)

    def fail_warm_up() -> None:
        fixture.events.append("pipeline.warm_up")
        raise RuntimeError("OCR warm-up failed")

    fixture.pipeline.warm_up = fail_warm_up  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="OCR warm-up failed"):
        with TestClient(create_app(fixture.services)):
            raise AssertionError("lifespan should not start")

    assert fixture.events == ["pipeline.warm_up", "pipeline.close"]


def test_services_are_single_use_across_app_lifespans(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    app = create_app(fixture.services)
    with TestClient(app):
        pass

    try:
        with TestClient(app):
            pass
    except RuntimeError as error:
        assert "single-use" in str(error)
    else:
        raise AssertionError("stopped services were reused across lifespans")


def test_status_reports_runtime_and_capture_state(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    with TestClient(create_app(fixture.services)) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["phase"] == "MONITORING"
    assert response.json()["capture"]["phase"] == "CAPTURING"


def test_frame_websocket_immediately_sends_latest_downscaled_jpeg(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    fixture.services.hub.publish(
        CapturedFrame.create(1, 1, np.full((10, 20, 3), 40, np.uint8))
    )
    fixture.services.hub.publish(
        CapturedFrame.create(2, 2, np.full((10, 20, 3), 80, np.uint8))
    )

    with TestClient(create_app(fixture.services)) as client:
        with client.websocket_connect("/ws/frames") as socket:
            packet = socket.receive_bytes()

    assert int.from_bytes(packet[:8], "big") == 2
    decoded = cv2.imdecode(np.frombuffer(packet[8:], np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 4
    assert decoded.shape[0] == 2


def test_state_websocket_sends_current_then_clear_overlay_without_new_frame(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)

    with TestClient(create_app(fixture.services)) as client:
        with client.websocket_connect("/ws/state") as socket:
            current = socket.receive_json()
            fixture.services.runtime.clear_question("dialog_missing")
            cleared = socket.receive_json()

    assert current["phase"] == "MONITORING"
    assert current["capture"]["phase"] == "CAPTURING"
    assert cleared["overlay"] is None
    assert cleared["message"] == "dialog_missing"


def test_state_stream_does_not_resend_when_version_wait_times_out(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    runtime = ScriptedRuntime()
    fixture.services.runtime = runtime  # type: ignore[assignment]
    socket = RecordingWebSocket()

    async def exercise() -> None:
        try:
            await _stream_state(socket, fixture.services)  # type: ignore[arg-type]
        except WebSocketDisconnect:
            pass

    asyncio.run(exercise())

    assert [message["version"] for message in socket.messages] == [1, 2]
    assert socket.messages[-1]["message"] == "changed"


def test_frame_stream_propagates_internal_hub_runtime_error(tmp_path: Path) -> None:
    fixture = _services(tmp_path)

    class RaisingHub:
        def wait_after(self, _frame_id: int, _timeout: float) -> None:
            raise RuntimeError("hub failed")

    fixture.services.hub = RaisingHub()  # type: ignore[assignment]

    async def exercise() -> None:
        await _stream_frames(BlockingRecordingWebSocket(), fixture.services)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="hub failed"):
        asyncio.run(exercise())


def test_successful_update_replaces_matcher_from_new_generation(tmp_path: Path) -> None:
    fixture = _services(tmp_path)

    with TestClient(create_app(fixture.services)) as client:
        response = client.post("/api/question-bank/update")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["generation_id"] == "new-generation"
    assert len(fixture.pipeline.matchers) == 1
    matcher = fixture.pipeline.matchers[0]
    match = matcher.match_question("新问题")
    assert match is not None
    assert match.record.answer == "新答案"
    assert fixture.services.diagnostic_metadata["generation_id"] == "new-generation"
    assert fixture.events.count("coordinator.invalidate_cache") == 2


def test_failed_update_keeps_old_matcher_and_returns_structured_error(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path, updater_error=RuntimeError("网络不可用"))
    fixture.services.diagnostic_metadata = {"generation_id": "old-generation"}

    with TestClient(create_app(fixture.services)) as client:
        response = client.post("/api/question-bank/update")

    assert response.status_code == 500
    assert response.json() == {"ok": False, "error": "题库更新失败：网络不可用"}
    assert fixture.pipeline.matchers == []
    assert fixture.services.snapshot_diagnostic_metadata() == {
        "generation_id": "old-generation"
    }


def test_update_and_diagnostic_metadata_are_one_atomic_knowledge_version(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    writer = FakeDiagnosticWriter(tmp_path / "diagnostics.zip")
    fixture.services.diagnostic_writer = writer
    fixture.services.diagnostic_metadata = {"generation_id": "old-generation"}
    fixture.services.hub.publish(
        CapturedFrame.create(21, 22, np.full((5, 7, 3), 90, np.uint8))
    )
    replace_entered = threading.Event()
    replace_release = threading.Event()
    original_replace = fixture.pipeline.replace_matcher

    def gated_replace(matcher: object) -> None:
        original_replace(matcher)
        replace_entered.set()
        assert replace_release.wait(timeout=2)

    fixture.pipeline.replace_matcher = gated_replace  # type: ignore[method-assign]
    responses: dict[str, object] = {}

    with TestClient(create_app(fixture.services)) as client:
        update_thread = threading.Thread(
            target=lambda: responses.__setitem__(
                "update", client.post("/api/question-bank/update")
            )
        )
        update_thread.start()
        assert replace_entered.wait(timeout=2)

        diagnostic_thread = threading.Thread(
            target=lambda: responses.__setitem__(
                "diagnostic", client.post("/api/diagnostics")
            )
        )
        diagnostic_thread.start()
        time.sleep(0.05)
        assert diagnostic_thread.is_alive()
        assert writer.snapshots == []

        replace_release.set()
        update_thread.join(timeout=2)
        diagnostic_thread.join(timeout=2)

    assert not update_thread.is_alive()
    assert not diagnostic_thread.is_alive()
    assert responses["update"].status_code == 200  # type: ignore[union-attr]
    assert responses["diagnostic"].status_code == 200  # type: ignore[union-attr]
    assert writer.snapshots[0].metadata["generation_id"] == "new-generation"


def test_static_b_layout_contract(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    with TestClient(create_app(fixture.services)) as client:
        html = client.get("/")
        css = client.get("/app.css")
        javascript = client.get("/app.js")

    assert html.status_code == css.status_code == javascript.status_code == 200
    assert 'id="frameCanvas"' in html.text
    assert 'id="overlayCanvas"' in html.text
    assert "overlayCtx.clearRect" in javascript.text
    assert "state.overlay" in javascript.text
    assert 'style.aspectRatio = `${bitmap.width} / ${bitmap.height}`' in javascript.text
    assert "let activeFrameDecode = false" in javascript.text
    assert "let pendingFrameBuffer = null" in javascript.text
    assert "function createLatestFrameDecoder(decodeFrame, renderFrame)" in javascript.text
    assert "const frameDecoder = createLatestFrameDecoder(decodeFrame, renderFrame)" in javascript.text
    assert "frameDecoder.enqueue(data)" in javascript.text
    assert "frameSocket.onmessage = ({data}) =>" in javascript.text
    assert "frameSocket.onmessage = async" not in javascript.text
    assert "window.confirm" in javascript.text
    assert "完整游戏画面" in javascript.text
    assert "saveRecognitionDiagnostics(currentTarget)" in javascript.text
    assert "pendingFrameBuffer = data" in javascript.text
    assert "while (pendingFrameBuffer !== null)" in javascript.text
    assert "if (pendingFrameBuffer !== null)" in javascript.text
    assert "bitmap.close();\n          continue;" in javascript.text
    assert javascript.text.count("createImageBitmap") == 1
    bitmap_ready = javascript.text.index("const bitmap = await createImageBitmap")
    stale_check = javascript.text.index("if (pendingFrameBuffer !== null)", bitmap_ready)
    frame_advance = javascript.text.index("currentFrameId = frameId", stale_check)
    draw = javascript.text.index("frameCtx.drawImage", frame_advance)
    assert bitmap_ready < stale_check < frame_advance < draw
    assert "250" in javascript.text and "5000" in javascript.text
    assert "#overlayCanvas" in css.text and "position: absolute" in css.text
    assert "@media (max-width: 960px)" in css.text


def test_diagnostics_endpoint_writes_only_on_post_from_current_services(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    writer = FakeDiagnosticWriter(tmp_path / "diagnostics" / "bundle.zip")
    fixture.services.diagnostic_writer = writer
    fixture.services.diagnostic_config = {"token": "secret"}
    fixture.services.diagnostic_metadata = {"generation_id": "g1"}
    fixture.services.hub.publish(
        CapturedFrame.create(11, 22, np.full((5, 7, 3), 90, np.uint8))
    )

    with TestClient(create_app(fixture.services)) as client:
        assert writer.snapshots == []
        response = client.post("/api/diagnostics")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "path": str(writer.path),
    }
    assert len(writer.snapshots) == 1
    snapshot = writer.snapshots[0]
    assert snapshot.frame is not None and snapshot.frame.frame_id == 11
    assert snapshot.runtime == fixture.services.runtime.snapshot()
    assert snapshot.config == {"token": "secret"}
    assert snapshot.metadata == {"generation_id": "g1"}


def test_diagnostics_endpoint_without_writer_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)

    with TestClient(create_app(fixture.services)) as client:
        response = client.post("/api/diagnostics")

    assert response.status_code == 503
    assert response.json() == {"ok": False, "error": "诊断导出服务未配置"}


def test_diagnostics_endpoint_with_real_writer_creates_one_zip_on_post(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    output_dir = tmp_path / "diagnostics"
    fixture.services.diagnostic_writer = DiagnosticWriter(output_dir)
    fixture.services.hub.publish(
        CapturedFrame.create(12, 23, np.full((5, 7, 3), 70, np.uint8))
    )
    assert not output_dir.exists()

    with TestClient(create_app(fixture.services)) as client:
        response = client.post("/api/diagnostics")

    assert response.status_code == 200
    bundles = list(output_dir.glob("*.zip"))
    assert bundles == [Path(response.json()["path"])]


def test_diagnostics_endpoint_rejects_export_before_first_frame(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    output_dir = tmp_path / "diagnostics"
    fixture.services.diagnostic_writer = DiagnosticWriter(output_dir)

    with TestClient(create_app(fixture.services)) as client:
        response = client.post("/api/diagnostics")

    assert response.status_code == 409
    assert "没有可用画面" in response.json()["error"]
    assert list(output_dir.glob("*.zip")) == []


def test_app_does_not_install_wide_cors_middleware(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    app = create_app(fixture.services)

    assert all(
        middleware.cls.__name__ != "CORSMiddleware"
        for middleware in app.user_middleware
    )


def _bootstrap_secure_client(
    client: TestClient,
    security: LocalWebSecurity,
) -> str:
    browser_url = security.issue_browser_url(security.expected_origin)
    bootstrap_token = browser_url.split("#token=", 1)[1]
    response = client.post(
        "/api/session/bootstrap",
        headers={"Origin": security.expected_origin},
        json={"token": bootstrap_token},
    )
    assert response.status_code == 200
    return response.json()["token"]


def test_secure_app_rejects_wrong_host_origin_and_missing_token(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    security = LocalWebSecurity("127.0.0.1", 8765, process_token="process-secret")

    with TestClient(
        create_app(fixture.services, security),
        base_url=security.expected_origin,
    ) as client:
        bad_host = client.get("/api/health", headers={"Host": "localhost:8765"})
        bad_origin = client.post(
            "/api/session/bootstrap",
            headers={"Origin": "https://attacker.example"},
            json={"token": "unknown"},
        )
        missing_token = client.post(
            "/api/question-bank/update",
            headers={"Origin": security.expected_origin},
            json={},
        )

    assert bad_host.status_code == 400
    assert bad_origin.status_code == 403
    assert missing_token.status_code == 403
    assert fixture.pipeline.matchers == []


def test_bootstrap_is_single_use_and_authorizes_protected_post(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    security = LocalWebSecurity("127.0.0.1", 8765, process_token="process-secret")
    browser_url = security.issue_browser_url(security.expected_origin)
    bootstrap_token = browser_url.split("#token=", 1)[1]

    with TestClient(
        create_app(fixture.services, security),
        base_url=security.expected_origin,
    ) as client:
        first = client.post(
            "/api/session/bootstrap",
            headers={"Origin": security.expected_origin},
            json={"token": bootstrap_token},
        )
        replay = client.post(
            "/api/session/bootstrap",
            headers={"Origin": security.expected_origin},
            json={"token": bootstrap_token},
        )
        update = client.post(
            "/api/question-bank/update",
            headers={
                "Origin": security.expected_origin,
                TOKEN_HEADER: first.json()["token"],
            },
            json={},
        )

    assert first.status_code == 200
    assert replay.status_code == 403
    assert update.status_code == 200
    assert len(fixture.pipeline.matchers) == 1


def test_clean_page_restores_same_process_without_persisting_process_token(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    security = LocalWebSecurity("127.0.0.1", 8765, process_token="process-secret")
    browser_url = security.issue_browser_url(security.expected_origin)
    bootstrap_token = browser_url.split("#token=", 1)[1]

    with TestClient(
        create_app(fixture.services, security),
        base_url=security.expected_origin,
    ) as client:
        bootstrap = client.post(
            "/api/session/bootstrap",
            headers={"Origin": security.expected_origin},
            json={"token": bootstrap_token},
        )
        cookie_header = bootstrap.headers["set-cookie"].lower()
        browser_session = client.cookies.get(SESSION_COOKIE)
        restore = client.post(
            "/api/session/restore",
            headers={"Origin": security.expected_origin},
            json={},
        )
        cross_site = client.post(
            "/api/session/restore",
            headers={"Origin": "https://attacker.example"},
            json={},
        )

    assert bootstrap.status_code == 200
    assert browser_session and browser_session != "process-secret"
    assert "httponly" in cookie_header
    assert "samesite=strict" in cookie_header
    assert "max-age" not in cookie_header
    assert "expires=" not in cookie_header
    assert restore.status_code == 200
    assert restore.json() == {"ok": True, "token": "process-secret"}
    assert cross_site.status_code == 403

    restarted_fixture = _services(tmp_path / "restarted")
    restarted_security = LocalWebSecurity(
        "127.0.0.1",
        8765,
        process_token="new-process-secret",
    )
    with TestClient(
        create_app(restarted_fixture.services, restarted_security),
        base_url=restarted_security.expected_origin,
    ) as restarted_client:
        stale = restarted_client.post(
            "/api/session/restore",
            headers={
                "Origin": restarted_security.expected_origin,
                "Cookie": f"{SESSION_COOKIE}={browser_session}",
            },
            json={},
        )

    assert stale.status_code == 403


def test_static_client_restores_without_browser_storage(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    with TestClient(create_app(fixture.services)) as client:
        javascript = client.get("/app.js").text

    assert '"/api/session/restore"' in javascript
    assert "fetch(endpoint" in javascript
    assert "localStorage" not in javascript
    assert "sessionStorage" not in javascript


def test_websocket_rejects_cross_site_and_bad_token_before_state_leak(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    security = LocalWebSecurity("127.0.0.1", 8765, process_token="process-secret")

    with TestClient(
        create_app(fixture.services, security),
        base_url=security.expected_origin,
    ) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws/state",
                headers={
                    "Host": security.expected_host,
                    "Origin": "https://attacker.example",
                },
            ):
                pass

        with client.websocket_connect(
            "/ws/state",
            headers={
                "Host": security.expected_host,
                "Origin": security.expected_origin,
            },
        ) as socket:
            socket.send_json({"type": "authenticate", "token": "wrong"})
            with pytest.raises(WebSocketDisconnect):
                socket.receive_json()


def test_websocket_sends_state_only_after_valid_authentication(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    security = LocalWebSecurity("127.0.0.1", 8765, process_token="process-secret")

    with TestClient(
        create_app(fixture.services, security),
        base_url=security.expected_origin,
    ) as client:
        token = _bootstrap_secure_client(client, security)
        with client.websocket_connect(
            "/ws/state",
            headers={
                "Host": security.expected_host,
                "Origin": security.expected_origin,
            },
        ) as socket:
            socket.send_json({"type": "authenticate", "token": token})
            state = socket.receive_json()

    assert state["phase"] == RuntimePhase.MONITORING.value
