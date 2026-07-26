from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import threading
import time
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturedFrame, CapturePhase, CaptureStatus, Rect
from xyq_quiz.recognition.models import (
    ConfidenceLevel,
    DetectedLayout,
    RecognitionResult,
)
from xyq_quiz.runtime.state import RuntimePhase, RuntimeStore


_LAYOUT_MISSING_GRACE_SECONDS = 0.15
_IDLE_LAYOUT_SCAN_SECONDS = 0.20
_RECOGNITION_RETRY_SECONDS = 0.20
_RECOGNITION_RETRY_MAX_SECONDS = 2.0


def _next_recognition_retry_delay(current: float) -> float:
    return min(_RECOGNITION_RETRY_MAX_SECONDS, current * 2)


def _remaining_layout_scan_delay(
    *,
    layout_present: bool,
    layout_missing_cleared: bool,
    scan_fps: int,
    last_scan_at: float | None,
    now: float,
) -> float:
    """Return how long recognition should wait before scanning the latest frame.

    Preview capture keeps publishing at its own rate.  The coordinator consumes
    only the newest available frame at the recognition cadence: ``scan_fps``
    while a quiz layout is present (including its disappearance grace period),
    and 5 Hz after absence has been confirmed and state has been cleared.
    """
    if last_scan_at is None:
        return 0.0
    active_scan = layout_present or not layout_missing_cleared
    interval = (
        1.0 / scan_fps
        if active_scan
        else _IDLE_LAYOUT_SCAN_SECONDS
    )
    return max(0.0, interval - (now - last_scan_at))


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
        *,
        scan_fps: int = 15,
    ) -> None:
        self._capture_service = capture_service
        self._frame_hub = frame_hub
        self._layout_detector = layout_detector
        self._pipeline = pipeline
        self._store = store
        if (
            not isinstance(scan_fps, int)
            or isinstance(scan_fps, bool)
            or scan_fps <= 0
        ):
            raise ValueError("scan_fps must be a positive integer")
        self._scan_fps = scan_fps
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
        observed_layout: _FrameLayoutIdentity | None = None
        last_layout_scan_at: float | None = None
        layout_missing_since: float | None = None
        layout_missing_cleared = False
        candidate_count = 0
        active_hash: str | None = None
        active_identity: _QuizCacheIdentity | None = None
        active_generation: int | None = None
        active_result_level: ConfidenceLevel | None = None
        last_attempt_at: float | None = None
        retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
        pending: tuple[
            str,
            _QuizCacheIdentity,
            CapturedFrame,
            DetectedLayout,
            float,
            int,
            int,
        ] | None = None
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
                last_layout_scan_at = None
                candidate_count = 0
                active_hash = None
                active_identity = None
                active_generation = None
                active_result_level = None
                last_attempt_at = None
                retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
                pending = None

            if (
                layout_missing_since is not None
                and not layout_missing_cleared
                and time.monotonic() - layout_missing_since
                >= _LAYOUT_MISSING_GRACE_SECONDS
            ):
                self._store.clear_question("dialog_missing")
                observed_hash = None
                observed_identity = None
                observed_layout = None
                candidate_count = 0
                active_hash = None
                active_identity = None
                active_generation = None
                active_result_level = None
                last_attempt_at = None
                retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
                pending = None
                layout_missing_cleared = True

            capture_status = self._capture_service.status()
            if capture_status.phase is not CapturePhase.CAPTURING:
                self._publish_capture_phase(capture_status)
                observed_hash = None
                observed_identity = None
                observed_layout = None
                last_layout_scan_at = None
                layout_missing_since = None
                layout_missing_cleared = False
                candidate_count = 0
                active_hash = None
                active_identity = None
                active_generation = None
                active_result_level = None
                last_attempt_at = None
                retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
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
                scan_delay = _remaining_layout_scan_delay(
                    layout_present=observed_layout is not None,
                    layout_missing_cleared=layout_missing_cleared,
                    scan_fps=self._scan_fps,
                    last_scan_at=last_layout_scan_at,
                    now=time.monotonic(),
                )
                if scan_delay > 0:
                    # Keep ``last_frame_id`` unchanged.  Once the independent
                    # recognition interval expires, ``wait_after`` returns the
                    # hub's newest frame even if capture published nothing else.
                    self._stop_event.wait(scan_delay)
                    continue
            if frame is not None:
                last_frame_id = frame.frame_id
                capture_status = self._capture_service.status()
                if capture_status.phase is not CapturePhase.CAPTURING:
                    self._publish_capture_phase(capture_status)
                    observed_hash = None
                    observed_identity = None
                    observed_layout = None
                    last_layout_scan_at = None
                    layout_missing_since = None
                    layout_missing_cleared = False
                    candidate_count = 0
                    active_hash = None
                    active_identity = None
                    active_generation = None
                    active_result_level = None
                    last_attempt_at = None
                    retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
                    pending = None
                else:
                    layout_started = time.perf_counter()
                    last_layout_scan_at = time.monotonic()
                    layout = self._layout_detector.detect(frame.bgr)
                    layout_ms = (time.perf_counter() - layout_started) * 1000.0
                    if self._stop_event.is_set():
                        break
                    capture_status = self._capture_service.status()
                    if capture_status.phase is not CapturePhase.CAPTURING:
                        self._publish_capture_phase(capture_status)
                        observed_hash = None
                        observed_identity = None
                        observed_layout = None
                        last_layout_scan_at = None
                        layout_missing_since = None
                        layout_missing_cleared = False
                        candidate_count = 0
                        active_hash = None
                        active_identity = None
                        active_generation = None
                        active_result_level = None
                        last_attempt_at = None
                        retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
                        pending = None
                    elif layout is None:
                        if layout_missing_since is None:
                            layout_missing_since = time.monotonic()
                            layout_missing_cleared = False
                    else:
                        layout_missing_since = None
                        layout_missing_cleared = False
                        layout_signature = _frame_layout_identity(frame.bgr, layout)
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
                            active_generation = None
                            active_result_level = None
                            last_attempt_at = None
                            retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
                            pending = None
                        observed_layout = layout_signature
                        question_hash = _quiz_stability_signature(frame.bgr, layout)
                        identity = _quiz_cache_identity(frame.bgr, layout)
                        question_changed = (
                            observed_hash is not None
                            and not _same_question_signature(
                                question_hash,
                                observed_hash,
                            )
                        )
                        if observed_hash is None:
                            observed_hash = question_hash
                            observed_identity = identity
                            candidate_count = 1
                        elif question_changed:
                            if observed_hash is not None or active_hash is not None:
                                self._store.clear_question("question_changed")
                            observed_hash = question_hash
                            observed_identity = identity
                            candidate_count = 1
                            active_hash = None
                            active_identity = None
                            active_generation = None
                            active_result_level = None
                            last_attempt_at = None
                            retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
                            pending = None
                        else:
                            if (
                                observed_identity is not None
                                and _same_quiz_identity(identity, observed_identity)
                            ):
                                candidate_count += 1
                            else:
                                observed_identity = identity
                                candidate_count = 1

                        if not question_changed and candidate_count >= 2:
                            identity_needs_recognition = (
                                active_identity is None
                                or not _same_quiz_identity(identity, active_identity)
                            )
                            if active_hash is None:
                                active_hash = observed_hash
                                active_identity = identity
                                generation = self._store.begin_question(
                                    active_hash,
                                    frame.frame_id,
                                    frame_size=(frame.bgr.shape[1], frame.bgr.shape[0]),
                                )
                                active_generation = generation
                                active_result_level = None
                                retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
                                with self._cache_lock:
                                    cache_epoch = self._cache_epoch
                                    cached_entry = self._cache.get(active_hash)
                                    if cached_entry is not None:
                                        self._cache.move_to_end(active_hash)
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
                                    active_result_level = ConfidenceLevel.HIGH
                                else:
                                    pending = (
                                        active_hash,
                                        identity,
                                        frame,
                                        layout,
                                        layout_ms,
                                        generation,
                                        cache_epoch,
                                    )
                            elif identity_needs_recognition:
                                # The question is unchanged, but its options are
                                # not byte-identical.  Keep the existing overlay
                                # while two matching frames settle, then remap.
                                active_identity = identity
                                retry_delay_seconds = _RECOGNITION_RETRY_SECONDS
                                generation = active_generation
                                if generation is not None:
                                    with self._cache_lock:
                                        cache_epoch = self._cache_epoch
                                        cached_entry = self._cache.get(active_hash)
                                        if cached_entry is not None:
                                            self._cache.move_to_end(active_hash)
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
                                        if self._store.complete(
                                            generation,
                                            replace(
                                                cached,
                                                generation_id=generation,
                                                frame_id=frame.frame_id,
                                                overlay_rect=overlay_rect,
                                            ),
                                        ):
                                            active_result_level = ConfidenceLevel.HIGH
                                    else:
                                        pending = (
                                            active_hash,
                                            identity,
                                            frame,
                                            layout,
                                            layout_ms,
                                            generation,
                                            cache_epoch,
                                        )
                            elif (
                                active_generation is not None
                                and active_result_level
                                in {ConfidenceLevel.NONE, ConfidenceLevel.CANDIDATE}
                                and running is None
                                and pending is None
                                and (
                                    last_attempt_at is None
                                    or time.monotonic() - last_attempt_at
                                    >= retry_delay_seconds
                                )
                            ):
                                # A retry deliberately reuses the generation so
                                # a candidate overlay remains visible in-flight.
                                pending = (
                                    active_hash,
                                    identity,
                                    frame,
                                    layout,
                                    layout_ms,
                                    active_generation,
                                    cache_epoch,
                                )
                                retry_delay_seconds = (
                                    _next_recognition_retry_delay(
                                        retry_delay_seconds
                                    )
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
                    if (
                        active_generation == generation
                        and active_hash == result_hash
                        and active_identity is not None
                        and _same_quiz_identity(result_identity, active_identity)
                        and self._store.fail(generation, str(exc))
                    ):
                        active_result_level = ConfidenceLevel.NONE
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
                    if (
                        active_generation == generation
                        and active_hash == result_hash
                        and active_identity is not None
                        and _same_quiz_identity(result_identity, active_identity)
                        and self._store.complete(generation, result)
                    ):
                        active_result_level = result.confidence_level

            if running is None and pending is not None:
                (
                    question_hash,
                    recognition_identity,
                    recognition_frame,
                    recognition_layout,
                    recognition_layout_ms,
                    generation,
                    cache_epoch,
                ) = pending
                pending = None
                with self._lifecycle_lock:
                    executor = self._executor
                    if self._stop_event.is_set() or executor is None:
                        return
                    recognize_with_layout = getattr(
                        self._pipeline,
                        "recognize_with_layout",
                        None,
                    )
                    if callable(recognize_with_layout):
                        future = executor.submit(
                            recognize_with_layout,
                            recognition_frame,
                            generation,
                            recognition_layout,
                            recognition_layout_ms,
                        )
                    else:
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
                last_attempt_at = time.monotonic()

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
    """Fingerprint only the question ROI for tolerant continuity decisions.

    Options are intentionally excluded: animation, hover state, and delayed
    rendering in those boxes must not look like a new question.  Cache reuse
    remains guarded independently by :func:`_quiz_cache_identity`.
    """
    rect = layout.question_rect
    metadata = hashlib.blake2b(digest_size=8)
    metadata.update((layout.profile_name or "").encode("utf-8"))
    metadata.update(f"{rect.x},{rect.y},{rect.width},{rect.height}".encode("ascii"))
    crop = frame[
        rect.y : rect.y + rect.height,
        rect.x : rect.x + rect.width,
    ]
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (33, 16), interpolation=cv2.INTER_AREA)
    gradients = normalized[:, 1:] > normalized[:, :-1]
    packed = np.packbits(gradients).tobytes()
    return f"{metadata.hexdigest()}:{packed.hex()}"


def _same_question_signature(left: str, right: str) -> bool:
    """Return whether two question fingerprints are perceptually equivalent."""
    if left == right:
        return True
    try:
        left_metadata, left_payload = left.split(":", 1)
        right_metadata, right_payload = right.split(":", 1)
        left_bits = bytes.fromhex(left_payload)
        right_bits = bytes.fromhex(right_payload)
    except (ValueError, TypeError):
        return False
    if left_metadata != right_metadata or len(left_bits) != len(right_bits):
        return False
    changed_bits = sum(
        (left_byte ^ right_byte).bit_count()
        for left_byte, right_byte in zip(left_bits, right_bits, strict=True)
    )
    total_bits = len(left_bits) * 8
    return changed_bits <= max(4, round(total_bits * 0.06))


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
class _FrameLayoutIdentity:
    profile_name: str
    frame_size: tuple[int, int]
    layout_signature: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class _CachedRecognition:
    identity: _QuizCacheIdentity
    result: RecognitionResult


def _same_quiz_identity(
    left: _QuizCacheIdentity,
    right: _QuizCacheIdentity,
) -> bool:
    return left == right


def _frame_layout_identity(
    frame: NDArray[np.uint8],
    layout: DetectedLayout,
) -> _FrameLayoutIdentity:
    height, width = frame.shape[:2]
    return _FrameLayoutIdentity(
        profile_name=layout.profile_name or "",
        frame_size=(width, height),
        layout_signature=_layout_signature(layout),
    )


def _layout_signature(
    layout: DetectedLayout,
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (rect.x, rect.y, rect.width, rect.height)
        for rect in (layout.question_rect, *layout.option_rects)
    )


__all__ = ["RecognitionCoordinator"]
