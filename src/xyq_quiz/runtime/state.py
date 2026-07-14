from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import threading
import time

from xyq_quiz.recognition.models import RecognitionResult, RecognitionTimings


class RuntimePhase(StrEnum):
    WAITING_FOR_WINDOW = "WAITING_FOR_WINDOW"
    MONITORING = "MONITORING"
    RECOGNIZING = "RECOGNIZING"
    ANSWERED = "ANSWERED"
    UNCERTAIN = "UNCERTAIN"
    CAPTURE_EMPTY = "CAPTURE_EMPTY"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    version: int = 0
    generation_id: int = 0
    phase: RuntimePhase = RuntimePhase.WAITING_FOR_WINDOW
    frame_id: int | None = None
    question_hash: str | None = None
    question_text: str = ""
    option_texts: tuple[str, ...] = ("", "", "", "")
    official_answer: str | None = None
    question_score: float = 0.0
    question_runner_up_score: float = 0.0
    option_score: float = 0.0
    option_runner_up_score: float = 0.0
    high_confidence: bool = False
    option_index: int | None = None
    overlay: tuple[float, float, float, float] | None = None
    timings: RecognitionTimings | None = None
    message: str | None = None
    clear_monotonic_ns: int | None = None


class RuntimeStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._snapshot = RuntimeSnapshot()
        self._frame_size: tuple[int, int] | None = None

    def snapshot(self) -> RuntimeSnapshot:
        with self._condition:
            return self._snapshot

    def wait_after(
        self,
        version: int,
        timeout: float,
    ) -> RuntimeSnapshot | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._snapshot.version > version,
                timeout=timeout,
            )
            if self._snapshot.version <= version:
                return None
            return self._snapshot

    def begin_question(
        self,
        question_hash: str,
        frame_id: int,
        frame_size: tuple[int, int],
    ) -> int:
        with self._condition:
            generation_id = self._snapshot.generation_id + 1
            self._frame_size = frame_size
            self._publish_locked(
                replace(
                    _empty_snapshot(
                        self._snapshot,
                        generation_id=generation_id,
                        phase=RuntimePhase.RECOGNIZING,
                    ),
                    frame_id=frame_id,
                    question_hash=question_hash,
                    clear_monotonic_ns=self._snapshot.clear_monotonic_ns,
                )
            )
            return generation_id

    def complete(
        self,
        generation_id: int,
        result: RecognitionResult,
    ) -> bool:
        with self._condition:
            if generation_id != self._snapshot.generation_id:
                return False
            overlay = None
            answered = (
                result.high_confidence
                and result.option_index is not None
                and result.overlay_rect is not None
                and self._frame_size is not None
            )
            if answered:
                frame_width, frame_height = self._frame_size
                overlay = result.overlay_rect.normalized(frame_width, frame_height)
            self._publish_locked(
                replace(
                    self._snapshot,
                    frame_id=result.frame_id,
                    phase=(
                        RuntimePhase.ANSWERED
                        if answered
                        else RuntimePhase.UNCERTAIN
                    ),
                    question_text=result.question_text,
                    option_texts=result.option_texts,
                    official_answer=result.official_answer,
                    question_score=result.question_score,
                    question_runner_up_score=result.question_runner_up_score,
                    option_score=result.option_score,
                    option_runner_up_score=result.option_runner_up_score,
                    high_confidence=answered,
                    option_index=result.option_index if answered else None,
                    overlay=overlay,
                    timings=result.timings,
                    message=None,
                )
            )
            return True

    def clear_question(
        self,
        message: str,
        *,
        phase: RuntimePhase = RuntimePhase.MONITORING,
    ) -> int:
        with self._condition:
            generation_id = self._snapshot.generation_id + 1
            self._frame_size = None
            self._publish_locked(
                replace(
                    _empty_snapshot(
                        self._snapshot,
                        generation_id=generation_id,
                        phase=phase,
                    ),
                    message=message,
                    clear_monotonic_ns=time.monotonic_ns(),
                )
            )
            return generation_id

    def set_phase(
        self,
        phase: RuntimePhase,
        message: str | None = None,
        *,
        clear: bool = False,
    ) -> None:
        with self._condition:
            if clear:
                if (
                    self._snapshot.phase is phase
                    and self._snapshot.message == message
                    and self._snapshot.overlay is None
                    and self._snapshot.question_hash is None
                ):
                    return
                generation_id = self._snapshot.generation_id + 1
                self._frame_size = None
                self._publish_locked(
                    replace(
                        _empty_snapshot(
                            self._snapshot,
                            generation_id=generation_id,
                            phase=phase,
                        ),
                        message=message,
                        clear_monotonic_ns=time.monotonic_ns(),
                    )
                )
                return
            if self._snapshot.phase is phase and self._snapshot.message == message:
                return
            self._publish_locked(
                replace(self._snapshot, phase=phase, message=message)
            )

    def fail(self, generation_id: int, message: str) -> bool:
        with self._condition:
            if generation_id != self._snapshot.generation_id:
                return False
            self._publish_locked(
                replace(
                    self._snapshot,
                    phase=RuntimePhase.ERROR,
                    overlay=None,
                    high_confidence=False,
                    option_index=None,
                    message=message,
                )
            )
            return True

    def _publish_locked(self, snapshot: RuntimeSnapshot) -> None:
        self._snapshot = replace(snapshot, version=self._snapshot.version + 1)
        self._condition.notify_all()


def _empty_snapshot(
    previous: RuntimeSnapshot,
    *,
    generation_id: int,
    phase: RuntimePhase,
) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        version=previous.version,
        generation_id=generation_id,
        phase=phase,
    )


__all__ = ["RuntimePhase", "RuntimeSnapshot", "RuntimeStore"]
