from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import threading
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturedFrame, CapturePhase, CaptureStatus, Rect
from xyq_quiz.recognition.models import DetectedLayout, RecognitionResult
from xyq_quiz.runtime.state import RuntimePhase, RuntimeStore


class CaptureStatusSource(Protocol):
    def status(self) -> CaptureStatus: ...


class LayoutDetector(Protocol):
    def detect(self, frame: NDArray[np.uint8]) -> DetectedLayout | None: ...


class Pipeline(Protocol):
    def recognize(
        self,
        frame: CapturedFrame,
        generation_id: int,
    ) -> RecognitionResult: ...


class RecognitionCoordinator:
    """Coordinate recognition for one application-service lifetime.

    ``start`` is idempotent while this instance is running. Once ``stop`` has
    begun, this coordinator cannot be restarted; a new application lifespan
    must construct a new coordinator and a new set of services. The injected
    recognition pipeline remains caller-owned and is never closed here.
    """

    def __init__(
        self,
        capture_service: CaptureStatusSource,
        frame_hub: LatestFrameHub,
        layout_detector: LayoutDetector,
        pipeline: Pipeline,
        store: RuntimeStore,
        executor_factory: Callable[[], Executor] | None = None,
    ) -> None:
        self._capture_service = capture_service
        self._frame_hub = frame_hub
        self._layout_detector = layout_detector
        self._pipeline = pipeline
        self._store = store
        self._executor_factory = executor_factory or (
            lambda: ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="xyq-quiz-recognition",
            )
        )
        self._cache: OrderedDict[str, _CachedRecognition] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_epoch = 0
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._executor: Executor | None = None
        self._restart_forbidden = False

    def invalidate_cache(self) -> None:
        """Invalidate cached answers and any result still running on an old bank."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_epoch += 1
        self._store.clear_question("knowledge_changed")

    def start(self) -> None:
        """Start once, or do nothing when the first run is already active."""
        with self._lifecycle_lock:
            if self._restart_forbidden:
                raise RuntimeError("coordinator cannot be restarted after stop")
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop_event.clear()
            self._executor = self._executor_factory()
            self._worker = threading.Thread(
                target=self._run,
                name="xyq-quiz-coordinator",
                daemon=True,
            )
            self._worker.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop permanently without waiting for a running recognition call.

        A worker that does not exit within ``timeout`` is reported only after
        the coordinator executor has received a non-waiting shutdown request.
        """
        if timeout < 0:
            raise ValueError("stop timeout must not be negative")
        with self._lifecycle_lock:
            self._restart_forbidden = True
            self._stop_event.set()
            worker = self._worker
            executor = self._executor
            self._executor = None
        if worker is not None:
            self._store.clear_question(
                "coordinator_stopped",
                phase=RuntimePhase.WAITING_FOR_WINDOW,
            )

        stop_error: RuntimeError | None = None
        try:
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=timeout)
                if worker.is_alive():
                    stop_error = RuntimeError(
                        "recognition coordinator did not stop within "
                        f"{timeout:g} seconds"
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            with self._lifecycle_lock:
                if self._worker is worker and (
                    worker is None or not worker.is_alive()
                ):
                    self._worker = None

        if stop_error is not None:
            raise stop_error

    def _run(self) -> None:
        last_frame_id = 0
        observed_hash: str | None = None
        observed_identity: _QuizCacheIdentity | None = None
        observed_layout: tuple[tuple[int, int, int, int], ...] | None = None
        layout_missing: bool | None = None
        candidate_count = 0
        active_hash: str | None = None
        active_identity: _QuizCacheIdentity | None = None
        pending: tuple[str, _QuizCacheIdentity, CapturedFrame, int, int] | None = None
        running: tuple[
            str,
            _QuizCacheIdentity,
            int,
            int,
            Future[RecognitionResult],
        ] | None = None
        observed_cache_epoch = self._current_cache_epoch()

        while not self._stop_event.is_set():
            cache_epoch = self._current_cache_epoch()
            if cache_epoch != observed_cache_epoch:
                observed_cache_epoch = cache_epoch
                observed_hash = None
                observed_identity = None
                observed_layout = None
                candidate_count = 0
                active_hash = None
                active_identity = None
                pending = None
            capture_status = self._capture_service.status()
            if capture_status.phase is not CapturePhase.CAPTURING:
                self._publish_capture_phase(capture_status)
                observed_hash = None
                observed_identity = None
                observed_layout = None
                layout_missing = None
                candidate_count = 0
                active_hash = None
                active_identity = None
                pending = None
                self._stop_event.wait(0.02)
                continue

            if self._store.snapshot().phase in {
                RuntimePhase.WAITING_FOR_WINDOW,
                RuntimePhase.CAPTURE_EMPTY,
                RuntimePhase.ERROR,
            }:
                self._store.set_phase(RuntimePhase.MONITORING)

            frame = self._frame_hub.wait_after(last_frame_id, timeout=0.02)
            if self._stop_event.is_set():
                break
            if frame is not None:
                last_frame_id = frame.frame_id
                capture_status = self._capture_service.status()
                if capture_status.phase is not CapturePhase.CAPTURING:
                    self._publish_capture_phase(capture_status)
                    observed_hash = None
                    observed_identity = None
                    observed_layout = None
                    layout_missing = None
                    candidate_count = 0
                    active_hash = None
                    active_identity = None
                    pending = None
                else:
                    layout = self._layout_detector.detect(frame.bgr)
                    if self._stop_event.is_set():
                        break
                    capture_status = self._capture_service.status()
                    if capture_status.phase is not CapturePhase.CAPTURING:
                        self._publish_capture_phase(capture_status)
                        observed_hash = None
                        observed_identity = None
                        observed_layout = None
                        layout_missing = None
                        candidate_count = 0
                        active_hash = None
                        active_identity = None
                        pending = None
                    elif layout is None:
                        if layout_missing is not True:
                            self._store.clear_question("dialog_missing")
                        observed_hash = None
                        observed_identity = None
                        observed_layout = None
                        layout_missing = True
                        candidate_count = 0
                        active_hash = None
                        active_identity = None
                        pending = None
                    else:
                        layout_signature = _layout_signature(layout)
                        layout_changed = (
                            observed_layout is not None
                            and layout_signature != observed_layout
                        )
                        if layout_changed:
                            self._store.clear_question("layout_changed")
                            observed_hash = None
                            observed_identity = None
                            candidate_count = 0
                            active_hash = None
                            active_identity = None
                            pending = None
                        observed_layout = layout_signature
                        layout_missing = False
                        question_hash = _quiz_stability_signature(frame.bgr, layout)
                        identity = _quiz_cache_identity(frame.bgr, layout)
                        identity_changed = (
                            observed_identity is None
                            or not _same_quiz_identity(identity, observed_identity)
                        )
                        if question_hash != observed_hash or identity_changed:
                            if observed_hash is not None or active_hash is not None:
                                self._store.clear_question("question_changed")
                            observed_hash = question_hash
                            observed_identity = identity
                            candidate_count = 1
                            active_hash = None
                            active_identity = None
                            pending = None
                        elif (
                            active_hash != question_hash
                            or active_identity is None
                            or not _same_quiz_identity(identity, active_identity)
                        ):
                            candidate_count += 1
                            if candidate_count >= 2:
                                active_hash = question_hash
                                active_identity = identity
                                generation = self._store.begin_question(
                                    question_hash,
                                    frame.frame_id,
                                    frame_size=(frame.bgr.shape[1], frame.bgr.shape[0]),
                                )
                                with self._cache_lock:
                                    cache_epoch = self._cache_epoch
                                    cached_entry = self._cache.get(question_hash)
                                    if cached_entry is not None:
                                        self._cache.move_to_end(question_hash)
                                    cached = (
                                        cached_entry.result
                                        if cached_entry is not None
                                        and _same_quiz_identity(
                                            identity,
                                            cached_entry.identity,
                                        )
                                        else None
                                    )
                                if cached is not None and cached.high_confidence:
                                    overlay_rect = (
                                        layout.option_rects[cached.option_index]
                                        if cached.option_index is not None
                                        else None
                                    )
                                    self._store.complete(
                                        generation,
                                        replace(
                                            cached,
                                            generation_id=generation,
                                            frame_id=frame.frame_id,
                                            overlay_rect=overlay_rect,
                                        ),
                                    )
                                else:
                                    pending = (
                                        question_hash,
                                        identity,
                                        frame,
                                        generation,
                                        cache_epoch,
                                    )

            # New frame transitions above always invalidate stale generations first.
            if self._stop_event.is_set():
                break
            if running is not None and running[4].done():
                (
                    result_hash,
                    result_identity,
                    generation,
                    result_epoch,
                    future,
                ) = running
                running = None
                try:
                    result = future.result()
                except Exception as exc:
                    self._store.fail(generation, str(exc))
                else:
                    if result.high_confidence:
                        with self._cache_lock:
                            if result_epoch == self._cache_epoch:
                                self._cache[result_hash] = _CachedRecognition(
                                    result_identity,
                                    result,
                                )
                                self._cache.move_to_end(result_hash)
                                while len(self._cache) > 128:
                                    self._cache.popitem(last=False)
                    self._store.complete(generation, result)

            if running is None and pending is not None:
                (
                    question_hash,
                    recognition_identity,
                    recognition_frame,
                    generation,
                    cache_epoch,
                ) = pending
                pending = None
                with self._lifecycle_lock:
                    executor = self._executor
                    if self._stop_event.is_set() or executor is None:
                        return
                    future = executor.submit(
                        self._pipeline.recognize,
                        recognition_frame,
                        generation,
                    )
                running = (
                    question_hash,
                    recognition_identity,
                    generation,
                    cache_epoch,
                    future,
                )

    def _current_cache_epoch(self) -> int:
        with self._cache_lock:
            return self._cache_epoch

    def _publish_capture_phase(self, status: CaptureStatus) -> None:
        if status.phase is CapturePhase.WAITING_FOR_WINDOW:
            phase = RuntimePhase.WAITING_FOR_WINDOW
        elif status.phase is CapturePhase.CAPTURE_EMPTY:
            phase = RuntimePhase.CAPTURE_EMPTY
        else:
            phase = RuntimePhase.ERROR
        self._store.set_phase(phase, status.message, clear=True)


def _quiz_stability_signature(
    frame: NDArray[np.uint8],
    layout: DetectedLayout,
) -> str:
    """Build a coarse stability bucket; cache reuse also requires full identity."""
    digest = hashlib.blake2b(digest_size=20)
    digest.update((layout.profile_name or "").encode("utf-8"))
    for rect in (layout.question_rect, *layout.option_rects):
        digest.update(f"{rect.x},{rect.y},{rect.width},{rect.height};".encode("ascii"))
        crop = frame[
            rect.y : rect.y + rect.height,
            rect.x : rect.x + rect.width,
        ]
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        normalized = cv2.resize(gray, (33, 16), interpolation=cv2.INTER_AREA)
        gradients = normalized[:, 1:] > normalized[:, :-1]
        digest.update(np.packbits(gradients).tobytes())
    return digest.hexdigest()


def _quiz_cache_identity(
    frame: NDArray[np.uint8],
    layout: DetectedLayout,
) -> _QuizCacheIdentity:
    """Hash exact full-resolution ROI content before authorizing cache reuse."""
    digest = hashlib.blake2b(digest_size=32)
    digest.update((layout.profile_name or "").encode("utf-8"))
    for rect in (layout.question_rect, *layout.option_rects):
        digest.update(f"{rect.x},{rect.y},{rect.width},{rect.height};".encode("ascii"))
        crop = frame[
            rect.y : rect.y + rect.height,
            rect.x : rect.x + rect.width,
        ]
        contiguous = np.ascontiguousarray(crop)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return _QuizCacheIdentity(
        profile_name=layout.profile_name or "",
        layout_signature=_layout_signature(layout),
        digest=digest.digest(),
    )


@dataclass(frozen=True, slots=True)
class _QuizCacheIdentity:
    profile_name: str
    layout_signature: tuple[tuple[int, int, int, int], ...]
    digest: bytes


@dataclass(frozen=True, slots=True)
class _CachedRecognition:
    identity: _QuizCacheIdentity
    result: RecognitionResult


def _same_quiz_identity(
    left: _QuizCacheIdentity,
    right: _QuizCacheIdentity,
) -> bool:
    return left == right


def _layout_signature(
    layout: DetectedLayout,
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (rect.x, rect.y, rect.width, rect.height)
        for rect in (layout.question_rect, *layout.option_rects)
    )


__all__ = ["RecognitionCoordinator"]
