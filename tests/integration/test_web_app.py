from __future__ import annotations

import hashlib
import json
import asyncio
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturedFrame, CapturePhase, CaptureStatus, Rect
from xyq_quiz.capture.video import LatestVideoHub
from xyq_quiz.config import MatchConfig
from xyq_quiz.diagnostics import DiagnosticSnapshot, DiagnosticWriter
from xyq_quiz.knowledge.local import LocalQuestionStore
from xyq_quiz.knowledge.models import QuestionRecord
from xyq_quiz.knowledge.store import QuestionBank
from xyq_quiz.knowledge.updater import UpdateResult, load_current_generation
from xyq_quiz.recognition.models import (
    ConfidenceLevel,
    RecognitionResult,
    RecognitionTimings,
)
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


class FakeNativeCapture(FakeCapture):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__(name, events)
        self.layouts: list[tuple[int, int, int, int, float, bool]] = []
        self.overlays: list[
            tuple[tuple[float, float, float, float] | None, float, int]
        ] = []

    def set_preview_layout(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        scale: float,
        visible: bool,
    ) -> None:
        self.layouts.append((x, y, width, height, scale, visible))

    def set_preview_overlay(
        self,
        rect: tuple[float, float, float, float] | None,
        score: float,
        level: int,
    ) -> None:
        self.overlays.append((rect, score, level))


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


class FakePerformance:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.saved: list[tuple[str, str]] = []
        self.fps: list[float] = []

    def start(self) -> None:
        self.events.append("performance.start")

    def stop(self) -> None:
        self.events.append("performance.stop")

    def payload(self) -> dict[str, object]:
        return {
            "ocr": {
                "requested": "auto",
                "effective": "cpu",
                "label": "OCR CPU",
                "fallback_reason": None,
            },
            "preview": {
                "requested": "auto",
                "effective": "cpu",
                "label": "预览 CPU",
                "fallback_reason": None,
            },
            "pending_ocr": "auto",
            "pending_preview": "auto",
            "ocr_options": [],
            "preview_options": [],
            "probing": False,
            "benchmark_status": "idle",
            "canvas_fps": None,
        }

    def save(self, *, ocr_backend: str, preview_backend: str):
        self.saved.append((ocr_backend, preview_backend))
        return SimpleNamespace(
            ocr_backend=ocr_backend,
            preview_backend=preview_backend,
        )

    def record_canvas_fps(self, fps: float) -> None:
        self.fps.append(fps)

    def snapshot(self):
        return SimpleNamespace(preview=SimpleNamespace(effective="cpu"))


class HardwarePreviewPerformance(FakePerformance):
    def snapshot(self):
        return SimpleNamespace(
            preview=SimpleNamespace(effective="windows_hardware:auto")
        )


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
    generation = load_current_generation(tmp_path)
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
        local_question_store=LocalQuestionStore(
            tmp_path / "user-data" / "questions.json"
        ),
        official_bank=generation.question_bank,
        official_metadata=generation.metadata,
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


def test_native_preview_layout_and_overlay_are_forwarded_to_capture(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    capture = FakeNativeCapture("capture", fixture.events)
    fixture.services.capture = capture

    with TestClient(create_app(fixture.services)) as client:
        layout = client.post(
            "/api/preview/layout",
            json={
                "x": 32,
                "y": 45,
                "width": 960,
                "height": 720,
                "scale": 1.5,
                "visible": True,
            },
        )
        overlay = client.post(
            "/api/preview/overlay",
            json={
                "rect": [0.1, 0.2, 0.3, 0.4],
                "score": 87.5,
                "level": 2,
            },
        )

    assert layout.json() == {"ok": True, "native": True}
    assert overlay.json() == {"ok": True, "native": True}
    assert capture.layouts == [(32, 45, 960, 720, 1.5, True)]
    assert capture.overlays == [((0.1, 0.2, 0.3, 0.4), 87.5, 2)]


def test_frame_websocket_immediately_sends_latest_downscaled_i420(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    fixture.services.hub.publish(
        CapturedFrame.create(1, 1, np.full((10, 20, 3), 40, np.uint8))
    )
    fixture.services.hub.publish(
        CapturedFrame.create(2, 2, np.full((10, 20, 3), 80, np.uint8))
    )

    with TestClient(create_app(fixture.services)) as client:
        with client.websocket_connect("/ws/frames") as socket:
            config = socket.receive_json()
            packet = socket.receive_bytes()

    assert config == {"type": "preview-config", "mode": "i420"}
    assert int.from_bytes(packet[:8], "big") == 2
    assert int.from_bytes(packet[16:20], "big") == 4
    assert int.from_bytes(packet[20:24], "big") == 2
    assert len(packet[24:]) == 4 * 2 * 3 // 2


def test_frame_websocket_sends_wgc_bgra_hardware_preview(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    video_hub = LatestVideoHub(pixel_format="bgra")
    fixture.services.video_hub = video_hub
    fixture.services.performance = HardwarePreviewPerformance(fixture.events)  # type: ignore[assignment]
    video_hub.publish(
        frame_id=7,
        timestamp_us=123,
        width=4,
        height=2,
        key_frame=True,
        payload=bytes(range(32)),
    )

    with TestClient(create_app(fixture.services)) as client:
        with client.websocket_connect("/ws/frames") as socket:
            config = socket.receive_json()
            packet = socket.receive_bytes()

    assert config == {"type": "preview-config", "mode": "bgra"}
    assert packet[0] == 1
    assert int.from_bytes(packet[1:9], "big") == 7
    assert int.from_bytes(packet[17:21], "big") == 4
    assert int.from_bytes(packet[21:25], "big") == 2
    assert packet[25:] == bytes(range(32))


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
    assert current["confidence_level"] == "NONE"
    assert current["confidence_score"] == 0.0
    assert cleared["overlay"] is None
    assert cleared["message"] == "dialog_missing"


def test_state_websocket_serializes_candidate_overlay_confidence(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    generation = fixture.services.runtime.begin_question(
        "candidate",
        frame_id=10,
        frame_size=(100, 50),
    )
    fixture.services.runtime.complete(
        generation,
        RecognitionResult(
            generation_id=generation,
            frame_id=10,
            question_text="题目",
            option_texts=("甲", "乙", "丙", "丁"),
            official_answer="乙",
            question_score=88.0,
            question_runner_up_score=70.0,
            option_score=72.0,
            option_runner_up_score=60.0,
            high_confidence=False,
            option_index=1,
            overlay_rect=Rect(20, 10, 20, 10),
            timings=RecognitionTimings(1.0, 2.0, 3.0, 6.0),
            confidence_level=ConfidenceLevel.CANDIDATE,
            confidence_score=68.5,
            confidence_reason="唯一候选，选项匹配较弱",
        ),
    )

    with TestClient(create_app(fixture.services)) as client:
        with client.websocket_connect("/ws/state") as socket:
            current = socket.receive_json()

    assert current["phase"] == "CANDIDATE"
    assert current["overlay"] == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert current["high_confidence"] is False
    assert current["confidence_level"] == "CANDIDATE"
    assert current["confidence_score"] == 68.5
    assert current["confidence_reason"] == "唯一候选，选项匹配较弱"


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


def test_local_question_crud_rebuilds_combined_matcher_immediately(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)

    with TestClient(create_app(fixture.services)) as client:
        initial = client.get("/api/local-questions")
        assert initial.status_code == 200
        assert initial.json()["sha256"] is None

        supplement = client.post(
            "/api/local-questions",
            json={
                "sha256": None,
                "id": "local-extra",
                "mode": "supplement",
                "question": "本地新增问题",
                "answer": "本地答案",
                "answer_aliases": ["答案别名"],
                "enabled": True,
                "target_source_id": None,
            },
        )
        assert supplement.status_code == 201
        supplement_payload = supplement.json()
        assert supplement_payload["records"][0]["answer_aliases"] == ["答案别名"]
        assert fixture.pipeline.matchers[-1].match_question(
            "本地新增问题"
        ).record.answer == "本地答案"

        override = client.post(
            "/api/local-questions",
            json={
                "sha256": supplement_payload["sha256"],
                "id": "local-override",
                "mode": "override",
                "question": "新问题",
                "answer": "本地覆盖答案",
                "answer_aliases": ["覆盖别名"],
                "enabled": True,
                "target_source_id": None,
            },
        )
        assert override.status_code == 201
        override_payload = override.json()
        assert override_payload["records"][1]["target_source_id"] == "1"
        assert fixture.pipeline.matchers[-1].match_question(
            "新问题"
        ).record.answer == "本地覆盖答案"

        disabled = client.put(
            "/api/local-questions/local-extra",
            json={
                "sha256": override_payload["sha256"],
                "mode": "supplement",
                "question": "本地新增问题",
                "answer": "修改后的答案",
                "answer_aliases": [],
                "enabled": False,
                "target_source_id": None,
            },
        )
        assert disabled.status_code == 200
        disabled_payload = disabled.json()
        assert all(
            record.source_id != "local:local-extra"
            for record in fixture.pipeline.matchers[-1]._bank.records
        )

        deleted_override = client.request(
            "DELETE",
            "/api/local-questions/local-override",
            json={"sha256": disabled_payload["sha256"]},
        )
        assert deleted_override.status_code == 200
        assert fixture.pipeline.matchers[-1].match_question(
            "新问题"
        ).record.answer == "新答案"

        deleted_supplement = client.request(
            "DELETE",
            "/api/local-questions/local-extra",
            json={"sha256": deleted_override.json()["sha256"]},
        )
        assert deleted_supplement.status_code == 200
        assert deleted_supplement.json()["records"] == []

    stored = json.loads(
        (tmp_path / "user-data" / "questions.json").read_text("utf-8")
    )
    assert stored == {"schema_version": 1, "records": []}


def test_local_question_sha_conflict_never_overwrites_newer_file(tmp_path: Path) -> None:
    fixture = _services(tmp_path)

    with TestClient(create_app(fixture.services)) as client:
        created = client.post(
            "/api/local-questions",
            json={
                "sha256": None,
                "question": "先写入的题目",
                "answer": "先写入的答案",
            },
        )
        assert created.status_code == 201
        path = tmp_path / "user-data" / "questions.json"
        before = path.read_bytes()

        stale = client.post(
            "/api/local-questions",
            json={
                "sha256": None,
                "question": "不应写入的题目",
                "answer": "不应写入的答案",
            },
        )

    assert stale.status_code == 409
    assert stale.json()["current_sha256"] == created.json()["sha256"]
    assert path.read_bytes() == before


def test_local_question_post_and_put_keep_create_update_semantics(tmp_path: Path) -> None:
    fixture = _services(tmp_path)

    with TestClient(create_app(fixture.services)) as client:
        missing = client.put(
            "/api/local-questions/not-there",
            json={
                "sha256": None,
                "question": "不存在",
                "answer": "不存在",
            },
        )
        created = client.post(
            "/api/local-questions",
            json={
                "sha256": None,
                "id": "fixed-id",
                "question": "已存在",
                "answer": "原答案",
            },
        )
        duplicate = client.post(
            "/api/local-questions",
            json={
                "sha256": created.json()["sha256"],
                "id": "fixed-id",
                "question": "不应覆盖",
                "answer": "不应覆盖",
            },
        )

    assert missing.status_code == 404
    assert created.status_code == 201
    assert duplicate.status_code == 409
    document = json.loads(
        (tmp_path / "user-data" / "questions.json").read_text("utf-8")
    )
    assert document["records"][0]["answer"] == "原答案"


def test_official_update_and_local_write_share_one_mutation_lock(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    update_entered = threading.Event()
    update_release = threading.Event()
    original_update = fixture.services.updater.update

    def gated_update():
        update_entered.set()
        assert update_release.wait(timeout=2)
        return original_update()

    fixture.services.updater.update = gated_update  # type: ignore[method-assign]
    responses: dict[str, object] = {}

    with TestClient(create_app(fixture.services)) as client:
        update_thread = threading.Thread(
            target=lambda: responses.__setitem__(
                "update",
                client.post("/api/question-bank/update"),
            )
        )
        update_thread.start()
        assert update_entered.wait(timeout=2)

        local_thread = threading.Thread(
            target=lambda: responses.__setitem__(
                "local",
                client.post(
                    "/api/local-questions",
                    json={
                        "sha256": None,
                        "question": "锁内补题",
                        "answer": "锁内答案",
                    },
                ),
            )
        )
        local_thread.start()
        time.sleep(0.05)
        assert local_thread.is_alive()
        assert not (tmp_path / "user-data" / "questions.json").exists()

        update_release.set()
        update_thread.join(timeout=2)
        local_thread.join(timeout=2)

    assert responses["update"].status_code == 200  # type: ignore[union-attr]
    assert responses["local"].status_code == 201  # type: ignore[union-attr]
    assert fixture.pipeline.matchers[-1].match_question("锁内补题") is not None


def test_corrupt_local_file_keeps_official_bank_and_is_never_overwritten(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    path = tmp_path / "user-data" / "questions.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"schema_version": 1, "records": [broken')
    before = path.read_bytes()

    with TestClient(create_app(fixture.services)) as client:
        update = client.post("/api/question-bank/update")
        listing = client.get("/api/local-questions")
        write = client.post(
            "/api/local-questions",
            json={
                "sha256": None,
                "question": "不能覆盖损坏文件",
                "answer": "不能写入",
            },
        )

    assert update.status_code == 200
    assert fixture.pipeline.matchers[-1].match_question("新问题") is not None
    assert listing.status_code == 409
    assert listing.json()["writable"] is False
    assert "继续使用官方题库" in listing.json()["error"]
    assert write.status_code == 409
    assert write.json()["writable"] is False
    assert "禁止覆盖" in write.json()["error"]
    assert path.read_bytes() == before


def test_diagnostic_metadata_contains_only_local_summary(tmp_path: Path) -> None:
    fixture = _services(tmp_path)

    with TestClient(create_app(fixture.services)) as client:
        created = client.post(
            "/api/local-questions",
            json={
                "sha256": None,
                "question": "诊断中不能出现的私有题目",
                "answer": "诊断中不能出现的私有答案",
            },
        )

    assert created.status_code == 201
    metadata = fixture.services.snapshot_diagnostic_metadata()
    serialized = json.dumps(metadata, ensure_ascii=False)
    assert metadata["local_questions"]["record_count"] == 1
    assert metadata["local_questions"]["sha256"] == created.json()["sha256"]
    assert "私有题目" not in serialized
    assert "私有答案" not in serialized


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


def test_performance_api_reports_saves_and_records_canvas_fps(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    performance = FakePerformance(fixture.events)
    fixture.services.performance = performance  # type: ignore[assignment]

    with TestClient(create_app(fixture.services)) as client:
        status = client.get("/api/performance")
        saved = client.post(
            "/api/performance/settings",
            json={
                "action": "save",
                "ocr_backend": "directml:0",
                "preview_backend": "cpu",
            },
        )
        fps = client.post("/api/performance/canvas-fps", json={"fps": 29.8})

    assert status.status_code == 200
    assert status.json()["ocr"]["effective"] == "cpu"
    assert saved.json() == {
        "ok": True,
        "action": "save",
        "pending_ocr": "directml:0",
        "pending_preview": "cpu",
    }
    assert performance.saved == [("directml:0", "cpu")]
    assert fps.json() == {"ok": True}
    assert performance.fps == [29.8]
    assert "performance.start" in fixture.events
    assert "performance.stop" in fixture.events


def test_performance_apply_schedules_restart_after_response(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    performance = FakePerformance(fixture.events)
    restarted = threading.Event()
    fixture.services.performance = performance  # type: ignore[assignment]
    fixture.services.restart = restarted.set

    with TestClient(create_app(fixture.services)) as client:
        response = client.post(
            "/api/performance/settings",
            json={
                "action": "apply",
                "ocr_backend": "cpu",
                "preview_backend": "auto",
            },
        )
        assert response.status_code == 200
        assert restarted.wait(1.0)


def test_performance_api_rejects_invalid_payload(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    fixture.services.performance = FakePerformance(fixture.events)  # type: ignore[assignment]

    with TestClient(create_app(fixture.services)) as client:
        response = client.post(
            "/api/performance/settings",
            json={
                "action": "unknown",
                "ocr_backend": "cpu",
                "preview_backend": "cpu",
            },
        )
        bad_fps = client.post(
            "/api/performance/canvas-fps",
            json={"fps": "fast"},
        )

    assert response.status_code == 400
    assert bad_fps.status_code == 400


def test_static_b_layout_contract(tmp_path: Path) -> None:
    fixture = _services(tmp_path)
    with TestClient(create_app(fixture.services)) as client:
        html = client.get("/")
        css = client.get("/app.css")
        javascript = client.get("/app.js")

    assert html.status_code == css.status_code == javascript.status_code == 200
    assert 'id="frameCanvas"' in html.text
    assert 'id="overlayCanvas"' in html.text
    assert 'id="confidenceLevel"' in html.text
    assert 'id="confidenceScore"' in html.text
    assert 'id="confidenceReason"' in html.text
    assert 'class="local-bank-panel"' in html.text
    assert 'id="localQuestionForm"' in html.text
    assert 'id="localQuestionList"' in html.text
    assert 'id="backendStatus"' in html.text
    assert 'id="backendSettingsDialog"' in html.text
    assert 'id="ocrBackendSelect"' in html.text
    assert 'id="previewBackendSelect"' in html.text
    assert "不会随诊断文件或发布包导出" in html.text
    assert "body { margin: 0; min-height: 100vh; overflow: hidden;" in css.text
    assert "height: 100dvh; min-height: 0;" in css.text
    assert "place-items: start center" in css.text
    assert ".canvas-stack { position: relative; width: 100%; height: 100%;" in css.text
    assert "object-position: center top" in css.text
    assert ".sidebar { min-height: 0; overflow-y: auto;" in css.text
    assert "body { overflow: auto; }" in css.text
    assert ".canvas-stack { height: auto; }" in css.text
    assert ".sidebar { order: 2; overflow: visible; }" in css.text
    assert "overlayCtx.clearRect" in javascript.text
    assert "state.overlay" in javascript.text
    assert "confidencePresentation" in javascript.text
    assert 'frameCanvas.getContext("2d", {alpha: false})' in javascript.text
    assert "const canvasSizeChanged = (" in javascript.text
    assert "if (canvasSizeChanged) drawOverlay();" in javascript.text
    assert "function setText(element, value)" in javascript.text
    assert "const overlayChanged = updateOverlayState(state);" in javascript.text
    assert "if (overlayChanged) drawOverlay();" in javascript.text
    assert "const jpeg = new Uint8Array(data, 8);" in javascript.text
    assert "data.slice(8)" not in javascript.text
    assert "hsl(${hue.toFixed(1)}, 85%, 52%)" in javascript.text
    assert "overlayCtx.setLineDash" in javascript.text
    assert "评分 ${Math.round(presentation.score)}/100" in javascript.text
    assert 'alpha: level === "HIGH" ? 1 : 0.68' in javascript.text
    assert 'style.aspectRatio = `${bitmapWidth} / ${bitmapHeight}`' in javascript.text
    assert 'new VideoDecoder({' in javascript.text
    assert 'new EncodedVideoChunk({' in javascript.text
    assert 'message.mode === "h264"' in javascript.text
    assert 'message.mode === "i420"' in javascript.text
    assert 'message.mode === "nv12"' in javascript.text
    assert 'message.mode === "bgra"' in javascript.text
    assert 'format: "I420"' in javascript.text
    assert 'format: "NV12"' in javascript.text
    assert 'format: "BGRA"' in javascript.text
    assert "let activeFrameDecode = false" in javascript.text
    assert "let pendingFrameBuffer = null" in javascript.text
    assert "function createLatestFrameDecoder(decodeFrame, renderFrame)" in javascript.text
    assert "const frameDecoder = createLatestFrameDecoder(decodeFrame, renderFrame)" in javascript.text
    assert "const i420FrameDecoder = createLatestFrameDecoder(decodeI420Frame, renderFrame)" in javascript.text
    assert "const nv12FrameDecoder = createLatestFrameDecoder(decodeNV12Frame, renderFrame)" in javascript.text
    assert "const bgraFrameDecoder = createLatestFrameDecoder(decodeBGRAFrame, renderFrame)" in javascript.text
    assert "frameDecoder.enqueue(data)" in javascript.text
    assert 'apiFetch("/api/local-questions", {method: "GET"})' in javascript.text
    assert 'method: recordId ? "PUT" : "POST"' in javascript.text
    assert 'method: "DELETE", body: {sha256: localQuestionSha256}' in javascript.text
    assert "list.replaceChildren()" in javascript.text
    assert "frameSocket.onmessage = ({data}) =>" in javascript.text
    assert "frameSocket.onmessage = async" not in javascript.text
    assert "window.confirm" in javascript.text
    assert "请关闭当前页面并重新打开 XYQQuiz" in javascript.text
    assert "重新双击 XYQQuiz.exe" not in javascript.text
    assert "完整游戏画面" in javascript.text
    assert "saveRecognitionDiagnostics(currentTarget)" in javascript.text
    assert 'apiFetch("/api/performance", {method: "GET"})' in javascript.text
    assert 'apiFetch("/api/performance/settings"' in javascript.text
    assert 'apiFetch("/api/performance/canvas-fps"' in javascript.text
    assert "preserveDialogSelection: true" in javascript.text
    assert "renderPerformanceDialog({preserveSelection:" in javascript.text
    assert 'performanceSnapshot.pending_ocr' in javascript.text
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
    assert '#confidenceLevel[data-level="CANDIDATE"]' in css.text
    assert '#confidenceLevel[data-level="HIGH"]' in css.text
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
        missing_local_token = client.get(
            "/api/local-questions",
            headers={"Origin": security.expected_origin},
        )

    assert bad_host.status_code == 400
    assert bad_origin.status_code == 403
    assert missing_token.status_code == 403
    assert missing_local_token.status_code == 403
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


def test_protected_get_allows_browser_omitted_origin_with_valid_token(
    tmp_path: Path,
) -> None:
    fixture = _services(tmp_path)
    fixture.services.performance = FakePerformance(fixture.events)  # type: ignore[assignment]
    security = LocalWebSecurity("127.0.0.1", 8765, process_token="process-secret")

    with TestClient(
        create_app(fixture.services, security),
        base_url=security.expected_origin,
    ) as client:
        token = _bootstrap_secure_client(client, security)
        accepted = client.get(
            "/api/performance",
            headers={TOKEN_HEADER: token},
        )
        wrong_origin = client.get(
            "/api/performance",
            headers={
                "Origin": "https://attacker.example",
                TOKEN_HEADER: token,
            },
        )
        missing_token = client.get("/api/performance")

    assert accepted.status_code == 200
    assert accepted.json()["ok"] is True
    assert wrong_origin.status_code == 403
    assert missing_token.status_code == 403


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
