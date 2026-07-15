from __future__ import annotations

import time
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Protocol, Sequence

from numpy.typing import NDArray
import numpy as np
import cv2

from xyq_quiz.capture.models import CapturedFrame, Rect
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.models import OptionMatch, QuestionMatch, normalize_text
from xyq_quiz.recognition.models import (
    ConfidenceLevel,
    DetectedLayout,
    OCRText,
    RecognitionResult,
    RecognitionTimings,
)
from xyq_quiz.recognition.ocr import OCREngine, OCRRole


_QUESTION_CANDIDATE_SCORE = 62.0
_QUESTION_CANDIDATE_GAP = 2.0
_OPTION_CANDIDATE_SCORE = 55.0
_OPTION_CANDIDATE_GAP = 1.0


@dataclass(frozen=True, slots=True)
class _QuestionAttempt:
    ocr: OCRText
    match: QuestionMatch | None
    crop: NDArray[np.uint8]
    level: int
    match_ms: float

    @property
    def credible(self) -> bool:
        return self.match is not None and (
            self.match.score >= _QUESTION_CANDIDATE_SCORE
            and self.match.score - self.match.runner_up_score
            >= _QUESTION_CANDIDATE_GAP
        )


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
        return self._recognize(frame, generation_id)

    def recognize_with_layout(
        self,
        frame: CapturedFrame,
        generation_id: int,
        layout: DetectedLayout,
        layout_ms: float = 0.0,
    ) -> RecognitionResult:
        """Recognize a frame using the layout already measured by the coordinator."""

        if layout_ms < 0:
            raise ValueError("layout_ms must not be negative")
        return self._recognize(
            frame,
            generation_id,
            detected_layout=layout,
            detected_layout_ms=layout_ms,
        )

    def _recognize(
        self,
        frame: CapturedFrame,
        generation_id: int,
        *,
        detected_layout: DetectedLayout | None = None,
        detected_layout_ms: float = 0.0,
    ) -> RecognitionResult:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("recognition pipeline is closed")
        with self._matcher_lock:
            matcher = self._matcher
        started = time.perf_counter()
        layout = detected_layout
        layout_ms = detected_layout_ms
        if layout is None:
            layout_started = time.perf_counter()
            layout = self._layout_detector.detect(frame.bgr)
            layout_ms = _milliseconds_since(layout_started)
        if layout is None:
            self._store_crops(())
            return self._empty_result(
                frame,
                generation_id,
                layout_ms,
                _milliseconds_since(started),
            )

        ocr_started = time.perf_counter()
        question_attempt = self._recognize_question(frame.bgr, layout, matcher)

        option_raw_crops = tuple(
            _crop(frame.bgr, rect) for rect in layout.option_rects
        )
        option_crops = tuple(
            cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            for crop in option_raw_crops
        )
        self._store_crops((question_attempt.crop, *option_crops))

        # Question recognition and option localization are intentionally
        # separate.  A weak/ambiguous question cannot trigger four unnecessary
        # OCR calls or expose a plausible-looking answer from the wrong row.
        if not question_attempt.credible:
            ocr_ms = max(
                0.0,
                _milliseconds_since(ocr_started) - question_attempt.match_ms,
            )
            question_match = question_attempt.match
            return RecognitionResult(
                generation_id=generation_id,
                frame_id=frame.frame_id,
                question_text=question_attempt.ocr.text,
                option_texts=tuple("" for _ in layout.option_rects),
                official_answer=None,
                question_score=(question_match.score if question_match else 0.0),
                question_runner_up_score=(
                    question_match.runner_up_score if question_match else 0.0
                ),
                option_score=0.0,
                option_runner_up_score=0.0,
                high_confidence=False,
                option_index=None,
                overlay_rect=None,
                timings=RecognitionTimings(
                    layout_ms=layout_ms,
                    ocr_ms=ocr_ms,
                    match_ms=question_attempt.match_ms,
                    total_ms=_milliseconds_since(started),
                ),
                confidence_level=ConfidenceLevel.NONE,
                confidence_score=0.0,
                confidence_reason="未找到唯一且可信的题目候选",
            )

        question_match = question_attempt.match
        assert question_match is not None
        option_ocrs = self._recognize_options(option_raw_crops, option_crops)
        ocr_ms = max(
            0.0,
            _milliseconds_since(ocr_started) - question_attempt.match_ms,
        )
        option_texts = tuple(result.text for result in option_ocrs)

        match_started = time.perf_counter()
        option_match = self._locate_option(matcher, question_match, option_texts)
        high_confidence = matcher.is_high_confidence(question_match, option_match)
        match_ms = question_attempt.match_ms + _milliseconds_since(match_started)

        official_answer, source_id = _matched_record(question_match, option_match)
        option_index = option_match.option_index if option_match is not None else None
        overlay_rect = (
            layout.option_rects[option_index]
            if option_index is not None and option_index < len(layout.option_rects)
            else None
        )
        if overlay_rect is None:
            confidence_level = ConfidenceLevel.NONE
            confidence_score = 0.0
            confidence_reason = "题目已识别，但选项无法唯一定位"
            option_index = None
        else:
            confidence_level = (
                ConfidenceLevel.HIGH
                if high_confidence
                else ConfidenceLevel.CANDIDATE
            )
            confidence_score = _combined_confidence_score(
                question_match,
                option_match,
                question_attempt.ocr,
                option_ocrs[option_index],
            )
            confidence_reason = (
                "题目与选项匹配稳定"
                if high_confidence
                else "候选唯一，但综合评分未达到高可信阈值"
            )
        return RecognitionResult(
            generation_id=generation_id,
            frame_id=frame.frame_id,
            question_text=question_attempt.ocr.text,
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
            source_id=source_id,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
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

    def _recognize_question(
        self,
        image: NDArray[np.uint8],
        layout: DetectedLayout,
        matcher: QuestionMatcher,
    ) -> _QuestionAttempt:
        attempts: list[_QuestionAttempt] = []
        match_ms = 0.0
        for level, rect in enumerate(_question_search_rects(image, layout), start=1):
            crop = _crop(image, rect)
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("recognition pipeline is closed")
                future = self._executor.submit(
                    self._recognize_crop,
                    crop,
                    OCRRole.QUESTION,
                    crop,
                )
            ocr = future.result()
            match_started = time.perf_counter()
            match = matcher.match_question(_extract_question_body(ocr.text))
            match_ms += _milliseconds_since(match_started)
            attempt = _QuestionAttempt(ocr, match, crop, level, match_ms)
            attempts.append(attempt)
            if attempt.credible:
                return attempt

        # Preserve the most useful OCR text and crop for the status panel and
        # saved diagnostics, without exposing its answer as a valid candidate.
        best = max(
            attempts,
            key=lambda attempt: (
                attempt.match.score if attempt.match is not None else 0.0,
                attempt.ocr.confidence,
                len(normalize_text(attempt.ocr.text)),
                attempt.level,
            ),
        )
        return _QuestionAttempt(best.ocr, best.match, best.crop, best.level, match_ms)

    def _recognize_options(
        self,
        raw_crops: Sequence[NDArray[np.uint8]],
        fallback_crops: Sequence[NDArray[np.uint8]],
    ) -> tuple[OCRText, ...]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("recognition pipeline is closed")
            futures = tuple(
                self._executor.submit(
                    self._recognize_crop,
                    raw_crop,
                    OCRRole.OPTION,
                    fallback_crop,
                )
                for raw_crop, fallback_crop in zip(
                    raw_crops,
                    fallback_crops,
                    strict=True,
                )
            )
        return tuple(future.result() for future in futures)

    def _store_crops(self, crops: Sequence[NDArray[np.uint8]]) -> None:
        stored_crops = tuple(np.ascontiguousarray(crop).copy() for crop in crops)
        for crop in stored_crops:
            crop.setflags(write=False)
        with self._crops_lock:
            self._latest_crops = stored_crops

    def _locate_option(
        self,
        matcher: QuestionMatcher,
        question: QuestionMatch,
        option_texts: tuple[str, ...],
    ) -> OptionMatch | None:
        unique_option_candidate = getattr(matcher, "unique_option_candidate", None)
        if callable(unique_option_candidate):
            return unique_option_candidate(
                question,
                option_texts,
                score_cutoff=_OPTION_CANDIDATE_SCORE,
                minimum_gap=_OPTION_CANDIDATE_GAP,
            )
        return self._map_official_answer(
            matcher,
            question.record.answer,
            option_texts,
        )

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

        # A comma can be part of the literal answer (for example "3,36" or
        # "新北京,新奥运").  Try the complete answer before treating commas as
        # the legacy alias encoding used by a few old rows.
        full_match = matcher.map_answer(official_answer, option_texts)
        if full_match is not None:
            return full_match
        aliases = [
            alias.strip()
            for alias in official_answer.split(",")
            if alias.strip()
        ]
        if len(aliases) < 2:
            return None
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


def _question_search_rects(
    frame: NDArray[np.uint8],
    layout: DetectedLayout,
) -> tuple[Rect, ...]:
    """Build exact, expanded and panel-wide question OCR regions.

    Layout detection still supplies the fast path.  The two wider regions are
    derived from the detected panel and option rows rather than a fixed game
    resolution, so they remain valid after the game window is resized.
    """

    frame_height, frame_width = frame.shape[:2]
    exact = _clamp_rect(layout.question_rect, frame_width, frame_height)
    horizontal_margin = max(2, round(exact.width * 0.08))
    vertical_margin = max(2, round(exact.height * 0.18))
    option_tops = tuple(
        rect.y
        for rect in layout.option_rects
        if rect.y > exact.y + max(1, exact.height // 3)
    )
    option_top = min(option_tops) if option_tops else frame_height

    expanded_bottom = min(
        frame_height,
        exact.y + exact.height + vertical_margin,
        option_top - 1 if option_top > exact.y + exact.height else frame_height,
    )
    expanded = _rect_from_edges(
        exact.x - horizontal_margin,
        exact.y - vertical_margin,
        exact.x + exact.width + horizontal_margin,
        max(exact.y + exact.height, expanded_bottom),
        frame_width,
        frame_height,
    )

    all_rects = (exact, *layout.option_rects)
    panel_margin_x = max(4, round(frame_width * 0.015))
    panel_margin_y = max(3, round(exact.height * 0.35))
    panel_left = min(rect.x for rect in all_rects) - panel_margin_x
    panel_right = max(rect.x + rect.width for rect in all_rects) + panel_margin_x
    panel_bottom = (
        option_top - 1
        if option_top > exact.y + max(1, exact.height // 3)
        else exact.y + exact.height + vertical_margin
    )
    panel = _rect_from_edges(
        panel_left,
        exact.y - panel_margin_y,
        panel_right,
        max(exact.y + exact.height, panel_bottom),
        frame_width,
        frame_height,
    )

    unique: list[Rect] = []
    for rect in (exact, expanded, panel):
        if rect not in unique:
            unique.append(rect)
    return tuple(unique)


def _clamp_rect(rect: Rect, frame_width: int, frame_height: int) -> Rect:
    return _rect_from_edges(
        rect.x,
        rect.y,
        rect.x + rect.width,
        rect.y + rect.height,
        frame_width,
        frame_height,
    )


def _rect_from_edges(
    left: int,
    top: int,
    right: int,
    bottom: int,
    frame_width: int,
    frame_height: int,
) -> Rect:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    clamped_left = min(max(0, int(left)), frame_width - 1)
    clamped_top = min(max(0, int(top)), frame_height - 1)
    clamped_right = min(max(clamped_left + 1, int(right)), frame_width)
    clamped_bottom = min(max(clamped_top + 1, int(bottom)), frame_height)
    return Rect(
        clamped_left,
        clamped_top,
        clamped_right - clamped_left,
        clamped_bottom - clamped_top,
    )


def _matched_record(
    question: QuestionMatch,
    option: OptionMatch | None,
) -> tuple[str | None, str | None]:
    if option is not None and option.source_id is not None:
        selected = next(
            (
                record
                for record in question.candidate_records
                if record.source_id == option.source_id
            ),
            None,
        )
        if selected is not None:
            return selected.answer, selected.source_id

    normalized_answers = {
        normalize_text(record.answer) for record in question.candidate_records
    }
    if len(normalized_answers) == 1:
        return question.record.answer, question.record.source_id
    return None, None


def _combined_confidence_score(
    question: QuestionMatch,
    option: OptionMatch,
    question_ocr: OCRText,
    option_ocr: OCRText,
) -> float:
    question_gap = max(0.0, question.score - question.runner_up_score)
    option_gap = max(0.0, option.score - option.runner_up_score)
    match_score = 0.55 * question.score + 0.45 * option.score
    gap_factor = 0.5 * min(1.0, question_gap / 10.0) + 0.5 * min(
        1.0,
        option_gap / 10.0,
    )
    ocr_factor = 0.85 + 0.15 * min(
        1.0,
        max(0.0, question_ocr.confidence),
        max(0.0, option_ocr.confidence),
    )
    score = match_score * (0.72 + 0.28 * gap_factor) * ocr_factor
    return round(min(100.0, max(0.0, score)), 2)


def _milliseconds_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _extract_question_body(text: str) -> str:
    prompt = re.search(
        r"第\s*\d+\s*关\s*[:：].{0,60}?题目\s*[:：]",
        text[:120],
    )
    return text[prompt.end() :] if prompt else text


__all__ = ["RecognitionPipeline"]
