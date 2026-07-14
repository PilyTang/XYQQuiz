from __future__ import annotations

import time
import re
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from typing import Protocol

from numpy.typing import NDArray
import numpy as np
import cv2

from xyq_quiz.capture.models import CapturedFrame, Rect
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.models import OptionMatch
from xyq_quiz.recognition.models import (
    DetectedLayout,
    OCRText,
    RecognitionResult,
    RecognitionTimings,
)
from xyq_quiz.recognition.ocr import OCREngine, OCRRole


class LayoutDetector(Protocol):
    def detect(self, frame: NDArray[np.uint8]) -> DetectedLayout | None: ...


class RecognitionPipeline:
    def __init__(
        self,
        layout_detector: LayoutDetector,
        ocr_engine: OCREngine,
        matcher: QuestionMatcher,
    ) -> None:
        self._layout_detector = layout_detector
        self._ocr_engine = ocr_engine
        self._matcher = matcher
        self._matcher_lock = threading.Lock()
        self._crops_lock = threading.Lock()
        self._latest_crops: tuple[NDArray[np.uint8], ...] = ()
        self._lifecycle_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="xyq-quiz-ocr",
        )
        self._warm_up_future: Future[OCRText] | None = None
        self._closed = False

    def warm_up(self) -> None:
        """Initialize OCR on its long-lived executor before capture starts."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("recognition pipeline is closed")
            if self._warm_up_future is None:
                image = np.full((96, 384, 3), 255, dtype=np.uint8)
                cv2.putText(
                    image,
                    "warmup",
                    (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                self._warm_up_future = self._executor.submit(
                    self._recognize_crop,
                    image,
                    OCRRole.QUESTION,
                    image,
                )
            future = self._warm_up_future
        future.result()

    def recognize(
        self,
        frame: CapturedFrame,
        generation_id: int,
    ) -> RecognitionResult:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("recognition pipeline is closed")
        with self._matcher_lock:
            matcher = self._matcher
        started = time.perf_counter()
        layout_started = time.perf_counter()
        layout = self._layout_detector.detect(frame.bgr)
        layout_ms = _milliseconds_since(layout_started)
        if layout is None:
            return self._empty_result(
                frame,
                generation_id,
                layout_ms,
                _milliseconds_since(started),
            )

        raw_crops = (
            _crop(frame.bgr, layout.question_rect),
            *(_crop(frame.bgr, rect) for rect in layout.option_rects),
        )
        crops = (
            raw_crops[0],
            *(
                cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                for crop in raw_crops[1:]
            ),
        )
        stored_crops = tuple(np.ascontiguousarray(crop).copy() for crop in crops)
        for crop in stored_crops:
            crop.setflags(write=False)
        with self._crops_lock:
            self._latest_crops = stored_crops
        ocr_started = time.perf_counter()
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("recognition pipeline is closed")
            roles = (OCRRole.QUESTION,) + (OCRRole.OPTION,) * len(
                layout.option_rects
            )
            futures = [
                self._executor.submit(
                    self._recognize_crop,
                    raw_crop,
                    role,
                    fallback_crop,
                )
                for raw_crop, role, fallback_crop in zip(
                    raw_crops,
                    roles,
                    crops,
                    strict=True,
                )
            ]
        ocr_results = tuple(future.result() for future in futures)
        ocr_ms = _milliseconds_since(ocr_started)

        question_ocr = ocr_results[0]
        option_ocrs = ocr_results[1:]
        option_texts = tuple(result.text for result in option_ocrs)
        match_started = time.perf_counter()
        question_match = matcher.match_question(_extract_question_body(question_ocr.text))
        official_answer = (
            question_match.record.answer if question_match is not None else None
        )
        option_match = self._map_official_answer(
            matcher,
            official_answer,
            option_texts,
        )
        high_confidence = matcher.is_high_confidence(
            question_match,
            option_match,
        )
        match_ms = _milliseconds_since(match_started)

        option_index = option_match.option_index if high_confidence else None
        overlay_rect = (
            layout.option_rects[option_index] if option_index is not None else None
        )
        return RecognitionResult(
            generation_id=generation_id,
            frame_id=frame.frame_id,
            question_text=question_ocr.text,
            option_texts=option_texts,  # type: ignore[arg-type]
            official_answer=official_answer,
            question_score=question_match.score if question_match else 0.0,
            question_runner_up_score=(
                question_match.runner_up_score if question_match else 0.0
            ),
            option_score=option_match.score if option_match else 0.0,
            option_runner_up_score=(
                option_match.runner_up_score if option_match else 0.0
            ),
            high_confidence=high_confidence,
            option_index=option_index,
            overlay_rect=overlay_rect,
            timings=RecognitionTimings(
                layout_ms=layout_ms,
                ocr_ms=ocr_ms,
                match_ms=match_ms,
                total_ms=_milliseconds_since(started),
            ),
            source_id=(question_match.record.source_id if question_match else None),
        )

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def replace_matcher(self, matcher: QuestionMatcher) -> None:
        """Atomically replace the matcher used by future recognitions.

        A recognition already in its matching phase keeps the one snapshot it
        acquired, so question and option matching cannot mix generations.
        """
        with self._matcher_lock:
            self._matcher = matcher

    def latest_crops(self) -> tuple[NDArray[np.uint8], ...]:
        """Return independent read-only copies of the last OCR input crops."""
        with self._crops_lock:
            snapshot = tuple(crop.copy() for crop in self._latest_crops)
        for crop in snapshot:
            crop.setflags(write=False)
        return snapshot

    def _recognize_crop(
        self,
        image: NDArray[np.uint8],
        role: OCRRole,
        fallback_image: NDArray[np.uint8],
    ) -> OCRText:
        recognize_region = getattr(self._ocr_engine, "recognize_region", None)
        if callable(recognize_region):
            return recognize_region(
                image,
                role,
                fallback_image=fallback_image,
            )
        return self._ocr_engine.recognize(fallback_image)

    def _map_official_answer(
        self,
        matcher: QuestionMatcher,
        official_answer: str | None,
        option_texts: tuple[str, ...],
    ) -> OptionMatch | None:
        if official_answer is None:
            return None
        aliases = [
            alias.strip()
            for alias in official_answer.split(",")
            if alias.strip()
        ]
        successful_matches = [
            match
            for alias in aliases
            if (match := matcher.map_answer(alias, option_texts)) is not None
        ]
        if not successful_matches:
            return None
        option_indexes = {match.option_index for match in successful_matches}
        if len(option_indexes) != 1:
            return None
        return max(successful_matches, key=lambda match: match.score)

    @staticmethod
    def _empty_result(
        frame: CapturedFrame,
        generation_id: int,
        layout_ms: float,
        total_ms: float,
    ) -> RecognitionResult:
        return RecognitionResult(
            generation_id=generation_id,
            frame_id=frame.frame_id,
            question_text="",
            option_texts=("", "", "", ""),
            official_answer=None,
            question_score=0.0,
            question_runner_up_score=0.0,
            option_score=0.0,
            option_runner_up_score=0.0,
            high_confidence=False,
            option_index=None,
            overlay_rect=None,
            timings=RecognitionTimings(layout_ms, 0.0, 0.0, total_ms),
        )


def _crop(frame: NDArray[np.uint8], rect: Rect) -> NDArray[np.uint8]:
    return frame[
        rect.y : rect.y + rect.height,
        rect.x : rect.x + rect.width,
    ]


def _milliseconds_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _extract_question_body(text: str) -> str:
    prompt = re.search(
        r"第\s*\d+\s*关\s*[:：].{0,60}?题目\s*[:：]",
        text[:120],
    )
    return text[prompt.end() :] if prompt else text


__all__ = ["RecognitionPipeline"]
