from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
import threading
from typing import Any, Callable, Protocol
from uuid import uuid4

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
from xyq_quiz.knowledge.knowledge import (
    KnowledgeSnapshot,
    load_knowledge_snapshot,
)
from xyq_quiz.knowledge.local import (
    LocalQuestionConflictError,
    LocalQuestionDocument,
    LocalQuestionError,
    LocalQuestionMode,
    LocalQuestionRecord,
    LocalQuestionStore,
    LocalQuestionWriteConflictError,
)
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.store import QuestionBank
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


class _LocalQuestionApiError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        current_sha256: str | None = None,
        writable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.current_sha256 = current_sha256
        self.writable = writable

    def response(self) -> JSONResponse:
        content: dict[str, Any] = {"ok": False, "error": self.message}
        if self.current_sha256 is not None:
            content["current_sha256"] = self.current_sha256
        if self.writable is not None:
            content["writable"] = self.writable
        return JSONResponse(status_code=self.status_code, content=content)


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
    local_question_store: LocalQuestionStore | None = None
    official_bank: QuestionBank | None = None
    official_metadata: Any = field(default_factory=dict)
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
    _mutation_lock: threading.Lock = field(
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

    def install_knowledge(
        self,
        snapshot: KnowledgeSnapshot,
        official_metadata: Any,
    ) -> None:
        config = self.match_config
        matcher = QuestionMatcher(
            snapshot.bank,
            config.question_score,
            config.question_gap,
            config.option_score,
        )
        metadata = self.diagnostic_metadata_for(official_metadata, snapshot)
        with self._knowledge_lock:
            self.coordinator.invalidate_cache()
            self.pipeline.replace_matcher(matcher)
            self.coordinator.invalidate_cache()
            self.official_bank = snapshot.official_bank
            self.official_metadata = copy.deepcopy(official_metadata)
            self.diagnostic_metadata = metadata

    def run_knowledge_mutation(self, callback: Callable[[], Any]) -> Any:
        with self._mutation_lock:
            return callback()

    def knowledge_context(self) -> tuple[QuestionBank | None, Any]:
        with self._knowledge_lock:
            return self.official_bank, copy.deepcopy(self.official_metadata)

    def snapshot_diagnostic_metadata(self) -> Any:
        with self._knowledge_lock:
            return copy.deepcopy(self.diagnostic_metadata)

    @staticmethod
    def diagnostic_metadata_for(
        official_metadata: Any,
        snapshot: KnowledgeSnapshot,
    ) -> dict[str, Any]:
        metadata = copy.deepcopy(official_metadata)
        if not isinstance(metadata, dict):
            metadata = {"official_metadata": metadata}
        metadata["local_questions"] = {
            "record_count": len(snapshot.local_document.records),
            "active_record_count": snapshot.active_local_count,
            "sha256": snapshot.local_document.sha256,
            "load_error": snapshot.local_error is not None,
            "conflict_count": len(snapshot.local_conflicts),
            "conflict_codes": sorted(
                {conflict.code for conflict in snapshot.local_conflicts}
            ),
            "issue_count": len(snapshot.issues),
            "issue_codes": sorted({issue.code for issue in snapshot.issues}),
        }
        return metadata


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
        def update() -> tuple[Any, Any]:
            result = services.updater.update()
            generation = load_current_generation(services.updater.data_dir)
            if services.local_question_store is None:
                config = services.match_config
                matcher = QuestionMatcher(
                    generation.question_bank,
                    config.question_score,
                    config.question_gap,
                    config.option_score,
                )
                services.replace_knowledge(matcher, generation.metadata)
            else:
                snapshot = load_knowledge_snapshot(
                    generation.question_bank,
                    services.local_question_store,
                )
                services.install_knowledge(snapshot, generation.metadata)
            return result, generation

        try:
            result, generation = await asyncio.to_thread(
                services.run_knowledge_mutation,
                update,
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

    @app.get("/api/local-questions")
    async def list_local_questions() -> JSONResponse:
        try:
            snapshot = await asyncio.to_thread(
                services.run_knowledge_mutation,
                lambda: _load_local_snapshot(services),
            )
        except _LocalQuestionApiError as error:
            return error.response()
        except Exception as error:
            return _unexpected_local_question_error(error)
        return JSONResponse(content=_local_question_payload(snapshot))

    @app.post("/api/local-questions")
    async def create_local_question(request: Request) -> JSONResponse:
        try:
            payload = await _request_object(request)
            record = _local_record_from_payload(
                payload,
                record_id=_new_local_question_id(),
                allow_id=True,
            )
            snapshot = await asyncio.to_thread(
                services.run_knowledge_mutation,
                lambda: _mutate_local_questions(
                    services,
                    payload,
                    lambda store, expected, official: _create_local_record(
                        store,
                        expected,
                        _resolve_override_target(record, official),
                    ),
                ),
            )
        except _LocalQuestionApiError as error:
            return error.response()
        except Exception as error:
            return _unexpected_local_question_error(error)
        return JSONResponse(status_code=201, content=_local_question_payload(snapshot))

    @app.put("/api/local-questions/{record_id}")
    async def update_local_question(
        record_id: str,
        request: Request,
    ) -> JSONResponse:
        try:
            payload = await _request_object(request)
            record = _local_record_from_payload(
                payload,
                record_id=record_id,
                allow_id=False,
            )
            snapshot = await asyncio.to_thread(
                services.run_knowledge_mutation,
                lambda: _mutate_local_questions(
                    services,
                    payload,
                    lambda store, expected, official: _update_local_record(
                        store,
                        expected,
                        _resolve_override_target(record, official),
                    ),
                ),
            )
        except _LocalQuestionApiError as error:
            return error.response()
        except Exception as error:
            return _unexpected_local_question_error(error)
        return JSONResponse(content=_local_question_payload(snapshot))

    @app.delete("/api/local-questions/{record_id}")
    async def delete_local_question(record_id: str, request: Request) -> JSONResponse:
        try:
            payload = await _request_object(request)
            snapshot = await asyncio.to_thread(
                services.run_knowledge_mutation,
                lambda: _mutate_local_questions(
                    services,
                    payload,
                    lambda store, expected, _official: store.delete(
                        record_id,
                        expected_sha256=expected,
                    ),
                ),
            )
        except _LocalQuestionApiError as error:
            return error.response()
        except Exception as error:
            return _unexpected_local_question_error(error)
        return JSONResponse(content=_local_question_payload(snapshot))

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


async def _request_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as error:
        raise _LocalQuestionApiError(400, "请求内容必须是 JSON 对象") from error
    if not isinstance(payload, dict):
        raise _LocalQuestionApiError(400, "请求内容必须是 JSON 对象")
    return payload


def _new_local_question_id() -> str:
    return uuid4().hex


def _unexpected_local_question_error(error: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": f"本地补题操作失败：{error}"},
    )


def _local_record_from_payload(
    payload: dict[str, Any],
    *,
    record_id: str,
    allow_id: bool,
) -> LocalQuestionRecord:
    allowed = {
        "sha256",
        "question",
        "answer",
        "mode",
        "enabled",
        "target_source_id",
        "answer_aliases",
    }
    if allow_id:
        allowed.add("id")
    unexpected = set(payload) - allowed
    if unexpected:
        raise _LocalQuestionApiError(
            400,
            f"存在不支持的字段：{sorted(unexpected)[0]}",
        )

    if allow_id and "id" in payload:
        supplied_id = payload["id"]
        if not isinstance(supplied_id, str) or not supplied_id.strip():
            raise _LocalQuestionApiError(400, "id 必须是非空字符串")
        record_id = supplied_id.strip()

    question = payload.get("question")
    answer = payload.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise _LocalQuestionApiError(400, "题目不能为空")
    if not isinstance(answer, str) or not answer.strip():
        raise _LocalQuestionApiError(400, "答案不能为空")

    raw_mode = payload.get("mode", LocalQuestionMode.SUPPLEMENT.value)
    try:
        mode = LocalQuestionMode(raw_mode)
    except (TypeError, ValueError) as error:
        raise _LocalQuestionApiError(
            400,
            "模式必须是 supplement 或 override",
        ) from error

    enabled = payload.get("enabled", True)
    if not isinstance(enabled, bool):
        raise _LocalQuestionApiError(400, "enabled 必须是布尔值")

    target_source_id = payload.get("target_source_id")
    if target_source_id is not None:
        if not isinstance(target_source_id, str) or not target_source_id.strip():
            raise _LocalQuestionApiError(400, "覆盖目标 source_id 不能为空")
        target_source_id = target_source_id.strip()

    answer_aliases = payload.get("answer_aliases", [])
    if not isinstance(answer_aliases, list) or not all(
        isinstance(alias, str) for alias in answer_aliases
    ):
        raise _LocalQuestionApiError(400, "answer_aliases 必须是字符串数组")

    return LocalQuestionRecord(
        id=record_id,
        question=question.strip(),
        answer=answer.strip(),
        mode=mode,
        enabled=enabled,
        target_source_id=target_source_id,
        answer_aliases=tuple(alias.strip() for alias in answer_aliases),
    )


def _load_local_snapshot(services: Services) -> KnowledgeSnapshot:
    store = services.local_question_store
    official_bank, _official_metadata = services.knowledge_context()
    if store is None or official_bank is None:
        raise _LocalQuestionApiError(503, "本地补题服务未配置")
    snapshot = load_knowledge_snapshot(official_bank, store)
    if snapshot.local_error is not None:
        raise _LocalQuestionApiError(
            409,
            f"本地补题文件无法读取，已保留原文件且继续使用官方题库：{snapshot.local_error}",
            writable=False,
        )
    return snapshot


def _mutate_local_questions(
    services: Services,
    payload: dict[str, Any],
    operation: Callable[
        [LocalQuestionStore, str | None, QuestionBank],
        LocalQuestionDocument,
    ],
) -> KnowledgeSnapshot:
    store = services.local_question_store
    official_bank, official_metadata = services.knowledge_context()
    if store is None or official_bank is None:
        raise _LocalQuestionApiError(503, "本地补题服务未配置")

    expected_sha256 = _expected_local_sha256(payload)
    loaded = store.load_safe()
    if loaded.error is not None:
        raise _LocalQuestionApiError(
            409,
            f"本地补题文件无法读取，已保留原文件且禁止覆盖：{loaded.error}",
            writable=False,
        )
    if loaded.document.sha256 != expected_sha256:
        raise _LocalQuestionApiError(
            409,
            "本地补题已被其他操作修改，请刷新后重试",
            current_sha256=loaded.document.sha256,
        )

    try:
        operation(store, expected_sha256, official_bank)
    except LocalQuestionWriteConflictError as error:
        current = store.load_safe().document.sha256
        raise _LocalQuestionApiError(
            409,
            "本地补题已被其他操作修改，请刷新后重试",
            current_sha256=current,
        ) from error
    except LocalQuestionConflictError as error:
        raise _LocalQuestionApiError(409, f"本地补题存在冲突：{error}") from error
    except KeyError as error:
        raise _LocalQuestionApiError(404, "要修改的本地题目不存在") from error
    except LocalQuestionError as error:
        raise _LocalQuestionApiError(400, f"本地补题内容无效：{error}") from error
    except (OSError, UnicodeError) as error:
        raise _LocalQuestionApiError(500, f"本地补题保存失败：{error}") from error

    snapshot = load_knowledge_snapshot(official_bank, store)
    if snapshot.local_error is not None:
        raise _LocalQuestionApiError(
            500,
            f"本地补题保存后无法读取：{snapshot.local_error}",
            writable=False,
        )
    services.install_knowledge(snapshot, official_metadata)
    return snapshot


def _create_local_record(
    store: LocalQuestionStore,
    expected_sha256: str | None,
    record: LocalQuestionRecord,
) -> LocalQuestionDocument:
    if any(existing.id == record.id for existing in store.load().records):
        raise _LocalQuestionApiError(409, "本地题目 id 已存在")
    return store.upsert(record, expected_sha256=expected_sha256)


def _update_local_record(
    store: LocalQuestionStore,
    expected_sha256: str | None,
    record: LocalQuestionRecord,
) -> LocalQuestionDocument:
    if all(existing.id != record.id for existing in store.load().records):
        raise KeyError(record.id)
    return store.upsert(record, expected_sha256=expected_sha256)


def _resolve_override_target(
    record: LocalQuestionRecord,
    official_bank: QuestionBank,
) -> LocalQuestionRecord:
    if (
        record.mode is not LocalQuestionMode.OVERRIDE
        or record.target_source_id is not None
    ):
        return record
    candidates = official_bank.records_for(record.normalized_question)
    if len(candidates) == 1:
        return LocalQuestionRecord(
            id=record.id,
            question=record.question,
            answer=record.answer,
            mode=record.mode,
            enabled=record.enabled,
            target_source_id=candidates[0].source_id,
            answer_aliases=record.answer_aliases,
        )
    if not candidates:
        raise _LocalQuestionApiError(
            400,
            "未找到题目完全一致的官方记录，请填写官方 source_id",
        )
    raise _LocalQuestionApiError(
        409,
        "找到多条同题官方记录，请填写要覆盖的官方 source_id",
    )


def _expected_local_sha256(payload: dict[str, Any]) -> str | None:
    if "sha256" not in payload:
        raise _LocalQuestionApiError(400, "缺少 sha256，请先刷新本地补题列表")
    value = payload["sha256"]
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise _LocalQuestionApiError(400, "sha256 格式无效")
    return value


def _local_question_payload(snapshot: KnowledgeSnapshot) -> dict[str, Any]:
    return {
        "ok": True,
        "writable": snapshot.local_error is None,
        "sha256": snapshot.local_document.sha256,
        "active_count": snapshot.active_local_count,
        "records": [
            {
                "id": record.id,
                "question": record.question,
                "answer": record.answer,
                "mode": record.mode.value,
                "enabled": record.enabled,
                "target_source_id": record.target_source_id,
                "answer_aliases": list(record.answer_aliases),
            }
            for record in snapshot.local_document.records
        ],
        "conflicts": [
            {
                "code": conflict.code,
                "record_ids": list(conflict.record_ids),
                "message": conflict.message,
                "target_source_id": conflict.target_source_id,
            }
            for conflict in snapshot.local_conflicts
        ],
        "issues": [
            {
                "code": issue.code,
                "local_ids": list(issue.local_ids),
                "message": issue.message,
            }
            for issue in snapshot.issues
        ],
    }


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
