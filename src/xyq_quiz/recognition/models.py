from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path

from xyq_quiz.capture.models import Rect


class ConfidenceLevel(StrEnum):
    """Presentation-safe recognition confidence tiers.

    ``confidence_score`` is a relative 0-100 score used for display and
    ranking.  It is deliberately not described as a probability.
    """

    NONE = "NONE"
    CANDIDATE = "CANDIDATE"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class NormalizedRect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError("normalized rectangle values must be numbers")
        if self.x < 0 or self.y < 0 or self.x >= 1 or self.y >= 1:
            raise ValueError("normalized rectangle origin must be inside the frame")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("normalized rectangle dimensions must be positive")


@dataclass(frozen=True, slots=True)
class AnchorProfile:
    search_rect: NormalizedRect
    template_path: Path
    threshold: float
    scale_range: tuple[float, float] = (1.0, 1.0)
    reference_rect: NormalizedRect | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("anchor threshold must be between zero and one")
        low, high = self.scale_range
        if low <= 0 or high < low:
            raise ValueError("anchor scale_range must be positive and ordered")


@dataclass(frozen=True, slots=True)
class DetectedLayout:
    question_rect: Rect
    option_rects: tuple[Rect, ...]
    anchor_scores: tuple[float, ...]
    profile_name: str | None = None


@dataclass(frozen=True, slots=True)
class OCRText:
    text: str
    confidence: float
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class RecognitionTimings:
    layout_ms: float
    ocr_ms: float
    match_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    generation_id: int
    frame_id: int
    question_text: str
    option_texts: tuple[str, ...]
    official_answer: str | None
    question_score: float
    question_runner_up_score: float
    option_score: float
    option_runner_up_score: float
    high_confidence: bool
    option_index: int | None
    overlay_rect: Rect | None
    timings: RecognitionTimings
    source_id: str | None = None
    confidence_level: ConfidenceLevel = ConfidenceLevel.NONE
    confidence_score: float | None = None
    confidence_reason: str | None = None

    def __post_init__(self) -> None:
        # Preserve compatibility with callers that only know the v0.1
        # ``high_confidence`` flag.  New callers can express a drawable, lower
        # confidence candidate without setting that legacy flag.
        level = self.confidence_level
        if not isinstance(level, ConfidenceLevel):
            level = ConfidenceLevel(level)
        if self.high_confidence:
            level = ConfidenceLevel.HIGH
        elif level is ConfidenceLevel.HIGH:
            object.__setattr__(self, "high_confidence", True)
        object.__setattr__(self, "confidence_level", level)

        score = self.confidence_score
        if score is None:
            score = (
                min(self.question_score, self.option_score)
                if level is not ConfidenceLevel.NONE
                else 0.0
            )
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            raise ValueError("confidence_score must be finite and between 0 and 100")
        object.__setattr__(self, "confidence_score", score)
