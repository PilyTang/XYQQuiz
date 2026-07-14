from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
import threading
from typing import Any, Callable, Protocol

import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CaptureStatus, CapturedFrame
from xyq_quiz.config import MatchConfig
from xyq_quiz.diagnostics import (
    DiagnosticSnapshot,
    DiagnosticUnavailable,
    DiagnosticWriter,
    EnvironmentDiagnosticWriter,
)
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.updater import QuestionBankUpdater, load_current_generation
from xyq_quiz.runtime.state import RuntimeSnapshot, RuntimeStore
from xyq_quiz.web.protocol import encode_frame_packet
from xyq_quiz.web.security import (
    APP_ID,
    SESSION_COOKIE,
    LocalWebSecurity,
    TOKEN_HEADER,
)


class LifecycleService(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


class CoordinatorLifecycle(LifecycleService, Protocol):
    def invalidate_cache(self) -> None: ...


class CaptureLifecycle(LifecycleService, Protocol):
    def status(self) -> CaptureStatus: ...


class MatcherPipeline(Protocol):
    def warm_up(self) -> None: ...
    def replace_matcher(self, matcher: QuestionMatcher) -> None: ...
    def close(self) -> None: ...

    def latest_crops(self) -> tuple[Any, ...]: ...


@dataclass(slots=True)
class Services:
    """One single-use service graph owned by one FastAPI lifespan."""

    hub: LatestFrameHub
    runtime: RuntimeStore
    capture: CaptureLifecycle
    coordinator: CoordinatorLifecycle
    pipeline: MatcherPipeline
    updater: QuestionBankUpdater
    match_config: MatchConfig
    preview_width: int = 1280
    owns_lifecycle: bool = True
    diagnostic_writer: DiagnosticWriter | None = None
    environment_diagnostic_writer: EnvironmentDiagnosticWriter | None = None
    diagnostic_config: Any = field(default_factory=dict)
    diagnostic_metadata: Any = field(default_factory=dict)
    shutdown: Callable[[], None] | None = None
    _lifespan_claimed: bool = field(default=False, init=False, repr=False)
    _claim_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _knowledge_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def claim_lifespan(self) -> None:
        if not self.owns_lifecycle:
            return
        with self._claim_lock:
            if self._lifespan_claimed:
                raise RuntimeError("services are single-use across app lifespans")
            self._lifespan_claimed = True

    def replace_knowledge(
        self,
        matcher: QuestionMatcher,
        metadata: Any,
    ) -> None:
        with self._knowledge_lock:
            self.coordinator.invalidate_cache()
            self.pipeline.replace_matcher(matcher)
            self.coordinator.invalidate_cache()
            self.diagnostic_metadata = copy.deepcopy(metadata)

    def snapshot_diagnostic_metadata(self) -> Any:
        with self._knowledge_lock:
            return copy.deepcopy(self.diagnostic_metadata)


def create_app(
    services: Services,
    security: LocalWebSecurity | None = None,
) -> FastAPI:
    static_dir = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        services.claim_lifespan()
        capture_started = False
        coordinator_started = False
        try:
            await asyncio.to_thread(services.pipeline.warm_up)
            services.capture.start()
            capture_started = True
            services.coordinator.start()
            coordinator_started = True
            yield
        finally:
            try:
                if coordinator_started:
                    await asyncio.to_thread(services.coordinator.stop)
            finally:
                try:
                    if capture_started:
                        await asyncio.to_thread(services.capture.stop)
                finally:
                    await asyncio.to_thread(services.pipeline.close)

    app = FastAPI(title="XYQ Quiz", lifespan=lifespan)

    if security is not None:
        @app.middleware("http")
        async def enforce_local_boundary(request: Request, call_next):
            path = request.url.path
            if path in {"/api/session/bootstrap", "/api/session/restore"}:
                decision = security.authorize_http(
                    host=request.headers.get("host"),
                    origin=request.headers.get("origin"),
                    token=None,
                    bootstrap=True,
                )
            elif path.startswith("/api/") and path != "/api/health":
                decision = security.authorize_http(
                    host=request.headers.get("host"),
                    origin=request.headers.get("origin"),
                    token=request.headers.get(TOKEN_HEADER),
                )
            else:
                decision = security.validate_host(request.headers.get("host"))
            if not decision.allowed:
                return JSONResponse(
                    status_code=decision.status_code,
                    content={"ok": False, "error": decision.message},
                )
            return await call_next(request)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "app_id": APP_ID, "ready": True}

    @app.post("/api/session/bootstrap")
    async def bootstrap_session(request: Request) -> JSONResponse:
        if security is None:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "本机会话认证未启用"},
            )
        try:
            payload = await request.json()
        except Exception:
            payload = None
        token = payload.get("token") if isinstance(payload, dict) else None
        process_token = security.consume_bootstrap(token)
        if process_token is None:
            return JSONResponse(
                status_code=403,
                content={"ok": False, "error": "浏览器引导令牌无效或已过期"},
            )
        response = JSONResponse(content={"ok": True, "token": process_token})
        response.set_cookie(
            key=SESSION_COOKIE,
            value=security.issue_browser_session(),
            path="/",
            httponly=True,
            samesite="strict",
        )
        return response

    @app.post("/api/session/restore")
    async def restore_session(request: Request) -> JSONResponse:
        if security is None:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "本机会话认证未启用"},
            )
        process_token = security.restore_browser_session(
            request.cookies.get(SESSION_COOKIE)
        )
        if process_token is None:
            return JSONResponse(
                status_code=403,
                content={"ok": False, "error": "浏览器会话无效或后台已重启"},
            )
        return JSONResponse(content={"ok": True, "token": process_token})

    @app.api_route("/api/status", methods=["GET", "POST"])
    async def status() -> dict[str, object]:
        payload = _runtime_payload(services.runtime.snapshot())
        payload["capture"] = jsonable_encoder(asdict(services.capture.status()))
        return payload

    @app.post("/api/question-bank/update")
    async def update_question_bank() -> JSONResponse:
        try:
            result = await asyncio.to_thread(services.updater.update)
            generation = await asyncio.to_thread(
                load_current_generation,
                services.updater.data_dir,
            )
            config = services.match_config
            matcher = QuestionMatcher(
                generation.question_bank,
                config.question_score,
                config.question_gap,
                config.option_score,
            )
            await asyncio.to_thread(
                services.replace_knowledge,
                matcher,
                generation.metadata,
            )
        except Exception as error:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"题库更新失败：{error}"},
            )
        return JSONResponse(
            content={
                "ok": True,
                "generation_id": generation.generation_id,
                "record_count": result.record_count,
            }
        )

    @app.post("/api/diagnostics")
    async def diagnostics() -> JSONResponse:
        writer = services.diagnostic_writer
        if writer is None:
            return JSONResponse(
                status_code=503,
                content={"ok": False, "error": "诊断导出服务未配置"},
            )
        latest_crops = getattr(services.pipeline, "latest_crops", None)
        crops = latest_crops() if latest_crops is not None else ()
        metadata = await asyncio.to_thread(services.snapshot_diagnostic_metadata)
        snapshot = DiagnosticSnapshot(
            frame=services.hub.snapshot(),
            runtime=services.runtime.snapshot(),
            crops=crops,
            config=services.diagnostic_config,
            metadata=metadata,
        )
        try:
            path = await asyncio.to_thread(writer.write, snapshot)
        except DiagnosticUnavailable as error:
            return JSONResponse(
                status_code=409,
                content={"ok": False, "error": str(error)},
            )
        except Exception as error:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"诊断导出失败：{error}"},
            )
        return JSONResponse(content={"ok": True, "path": str(path)})

    @app.post("/api/environment-diagnostics")
    async def environment_diagnostics() -> JSONResponse:
        writer = services.environment_diagnostic_writer
        if writer is None:
            return JSONResponse(
                status_code=503,
                content={"ok": False, "error": "环境诊断导出服务未配置"},
            )
        try:
            path = await asyncio.to_thread(writer.write, services.diagnostic_config)
        except Exception as error:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"环境诊断导出失败：{error}"},
            )
        return JSONResponse(content={"ok": True, "path": str(path)})

    @app.post("/api/shutdown")
    async def shutdown() -> JSONResponse:
        callback = services.shutdown
        if callback is None:
            return JSONResponse(
                status_code=503,
                content={"ok": False, "error": "退出服务未配置"},
            )
        await asyncio.to_thread(callback)
        return JSONResponse(content={"ok": True})

    @app.websocket("/ws/frames")
    async def frames(websocket: WebSocket) -> None:
        if not await _authorize_websocket(websocket, security):
            return
        try:
            await _stream_frames(websocket, services)
        except (WebSocketDisconnect, asyncio.CancelledError):
            return

    @app.websocket("/ws/state")
    async def state(websocket: WebSocket) -> None:
        if not await _authorize_websocket(websocket, security):
            return
        try:
            await _stream_state(websocket, services)
        except (WebSocketDisconnect, asyncio.CancelledError):
            return

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


async def _authorize_websocket(
    websocket: WebSocket,
    security: LocalWebSecurity | None,
) -> bool:
    if security is None:
        await websocket.accept()
        return True
    decision = security.authorize_websocket(
        host=websocket.headers.get("host"),
        origin=websocket.headers.get("origin"),
    )
    if not decision.allowed:
        await websocket.close(code=1008, reason=decision.message)
        return False
    await websocket.accept()
    try:
        message = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
    except (asyncio.TimeoutError, WebSocketDisconnect, ValueError, RuntimeError):
        await websocket.close(code=1008, reason="缺少本机会话认证")
        return False
    token = message.get("token") if isinstance(message, dict) else None
    message_type = message.get("type") if isinstance(message, dict) else None
    if message_type != "authenticate" or not security.validate_process_token(token):
        await websocket.close(code=1008, reason="本机会话令牌无效")
        return False
    return True


def _runtime_payload(snapshot: RuntimeSnapshot) -> dict[str, object]:
    return jsonable_encoder(asdict(snapshot))


async def _stream_state(websocket: WebSocket, services: Services) -> None:
    snapshot = services.runtime.snapshot()
    version = snapshot.version
    await websocket.send_json(_state_payload(snapshot, services))
    disconnect = _start_disconnect_watcher(websocket)
    try:
        while True:
            if _disconnect_finished(disconnect):
                return
            next_snapshot = await asyncio.to_thread(
                services.runtime.wait_after,
                version,
                0.25,
            )
            if next_snapshot is None or next_snapshot.version <= version:
                continue
            snapshot = next_snapshot
            version = snapshot.version
            await websocket.send_json(_state_payload(snapshot, services))
    finally:
        await _cancel_disconnect_watcher(disconnect)


async def _stream_frames(websocket: WebSocket, services: Services) -> None:
    last_frame_id = -1
    disconnect = _start_disconnect_watcher(websocket)
    try:
        while True:
            if _disconnect_finished(disconnect):
                return
            frame = await asyncio.to_thread(
                services.hub.wait_after,
                last_frame_id,
                0.25,
            )
            if frame is None:
                continue
            packet = await asyncio.to_thread(
                _encode_preview,
                frame,
                services.preview_width,
            )
            try:
                await websocket.send_bytes(packet)
            except RuntimeError:
                if _disconnect_finished(disconnect):
                    return
                raise
            last_frame_id = frame.frame_id
    finally:
        await _cancel_disconnect_watcher(disconnect)


def _start_disconnect_watcher(websocket: WebSocket) -> asyncio.Task[None] | None:
    receive = getattr(websocket, "receive", None)
    if not callable(receive):
        return None
    return asyncio.create_task(_watch_for_disconnect(receive))


async def _watch_for_disconnect(receive: Any) -> None:
    while True:
        message = await receive()
        if message.get("type") == "websocket.disconnect":
            return


def _disconnect_finished(disconnect: asyncio.Task[None] | None) -> bool:
    if disconnect is None or not disconnect.done():
        return False
    disconnect.result()
    return True


async def _cancel_disconnect_watcher(
    disconnect: asyncio.Task[None] | None,
) -> None:
    if disconnect is None:
        return
    disconnect.cancel()
    with suppress(asyncio.CancelledError):
        await disconnect


def _state_payload(
    snapshot: RuntimeSnapshot,
    services: Services,
) -> dict[str, object]:
    payload = _runtime_payload(snapshot)
    payload["capture"] = jsonable_encoder(asdict(services.capture.status()))
    return payload


def _encode_preview(frame: CapturedFrame, preview_width: int) -> bytes:
    if preview_width <= 0:
        raise ValueError("preview_width must be positive")
    image = frame.bgr
    height, width = image.shape[:2]
    if width > preview_width:
        preview_height = max(1, round(height * preview_width / width))
        image = cv2.resize(
            image,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )
    encoded, jpeg = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    if not encoded:
        raise RuntimeError("JPEG preview encoding failed")
    return encode_frame_packet(frame.frame_id, jpeg.tobytes())


__all__ = ["Services", "create_app"]
