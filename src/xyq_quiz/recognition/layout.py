from __future__ import annotations

import json
from itertools import combinations
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol, Self

import cv2
import numpy as np
from numpy.typing import NDArray

from xyq_quiz.capture.models import Rect
from xyq_quiz.recognition.models import (
    AnchorProfile,
    DetectedLayout,
    NormalizedRect,
)


class LayoutTransformError(ValueError):
    """The matched anchors cannot define a safe ROI transform."""


@dataclass(frozen=True, slots=True)
class LayoutProfile:
    reference_width: int
    reference_height: int
    question_rect: NormalizedRect
    option_rects: tuple[
        NormalizedRect,
        NormalizedRect,
        NormalizedRect,
        NormalizedRect,
    ]
    anchors: tuple[AnchorProfile, ...]
    name: str = ""

    @classmethod
    def load(cls, path: Path) -> Self:
        profile_path = Path(path)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("layout profile must be an object")

        reference_size = payload.get("reference_size")
        if (
            not isinstance(reference_size, list)
            or len(reference_size) != 2
            or not all(isinstance(value, int) for value in reference_size)
            or any(value <= 0 for value in reference_size)
        ):
            raise ValueError("reference_size must contain two positive integers")

        raw_options = payload.get("option_rects")
        if not isinstance(raw_options, list) or len(raw_options) != 4:
            raise ValueError("layout profile must contain exactly four options")

        raw_anchors = payload.get("anchors")
        if not isinstance(raw_anchors, list) or len(raw_anchors) < 2:
            raise ValueError("layout profile must contain at least two anchors")

        anchors = tuple(
            cls._parse_anchor(raw_anchor, profile_path.parent, index)
            for index, raw_anchor in enumerate(raw_anchors)
        )
        reference_rects = tuple(
            anchor.reference_rect
            for anchor in anchors
            if anchor.reference_rect is not None
        )
        if len(reference_rects) == len(anchors):
            validate_anchor_reference_geometry(
                reference_rects,
                reference_size[0],
                reference_size[1],
            )
        option_rects = tuple(cls._parse_rect(value) for value in raw_options)
        return cls(
            reference_width=reference_size[0],
            reference_height=reference_size[1],
            question_rect=cls._parse_rect(payload.get("question_rect")),
            option_rects=option_rects,  # type: ignore[arg-type]
            anchors=anchors,
            name=(
                payload["name"]
                if isinstance(payload.get("name"), str) and payload["name"].strip()
                else profile_path.stem
            ),
        )

    @staticmethod
    def _parse_rect(value: Any) -> NormalizedRect:
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("normalized rectangle must contain four numbers")
        if not all(isinstance(item, (int, float)) for item in value):
            raise ValueError("normalized rectangle must contain four numbers")
        return NormalizedRect(*(float(item) for item in value))

    @classmethod
    def _parse_anchor(
        cls,
        value: Any,
        profile_directory: Path,
        index: int,
    ) -> AnchorProfile:
        if not isinstance(value, dict):
            raise ValueError(f"anchor {index} must be an object")
        template_path = value.get("template_path")
        if not isinstance(template_path, str) or not template_path.strip():
            raise ValueError(f"anchor {index} has invalid template_path")
        threshold = value.get("threshold")
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"anchor {index} has invalid threshold")
        raw_scale_range = value.get("scale_range", [1.0, 1.0])
        if (
            not isinstance(raw_scale_range, list)
            or len(raw_scale_range) != 2
            or not all(isinstance(item, (int, float)) for item in raw_scale_range)
        ):
            raise ValueError(f"anchor {index} has invalid scale_range")
        raw_template_path = Path(template_path)
        resolved_template_path = (
            raw_template_path
            if raw_template_path.is_absolute()
            else profile_directory / raw_template_path
        )
        return AnchorProfile(
            search_rect=cls._parse_rect(value.get("search_rect")),
            template_path=resolved_template_path.resolve(),
            threshold=float(threshold),
            scale_range=(float(raw_scale_range[0]), float(raw_scale_range[1])),
            reference_rect=(
                cls._parse_rect(value["reference_rect"])
                if "reference_rect" in value
                else None
            ),
        )


class TemplateLayoutDetector:
    def __init__(self, profile: LayoutProfile) -> None:
        self._profile = profile
        self._templates = validate_anchor_templates(profile)

    def detect(self, frame: NDArray[np.uint8]) -> DetectedLayout | None:
        if frame.ndim not in (2, 3) or frame.size == 0:
            return None
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return None
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        matches: list[_AnchorMatch] = []
        for anchor, template in zip(
            self._profile.anchors,
            self._templates,
            strict=True,
        ):
            search_rect = _map_rect(anchor.search_rect, frame_width, frame_height)
            search_image = gray[
                search_rect.y : search_rect.y + search_rect.height,
                search_rect.x : search_rect.x + search_rect.width,
            ]
            match = self._match_anchor(
                search_image,
                template,
                anchor.scale_range,
                frame_width / self._profile.reference_width,
                frame_height / self._profile.reference_height,
            )
            if match.score < anchor.threshold:
                return None
            matches.append(
                _AnchorMatch(
                    match.score,
                    match.x + search_rect.x,
                    match.y + search_rect.y,
                    match.width,
                    match.height,
                )
            )

        try:
            mapper = self._rect_mapper(matches, frame_width, frame_height)
        except LayoutTransformError:
            return None
        option_rects = tuple(mapper(rect) for rect in self._profile.option_rects)
        return DetectedLayout(
            question_rect=mapper(self._profile.question_rect),
            option_rects=option_rects,  # type: ignore[arg-type]
            anchor_scores=tuple(match.score for match in matches),
            profile_name=self._profile.name,
        )

    def _rect_mapper(
        self,
        matches: Sequence[_AnchorMatch],
        frame_width: int,
        frame_height: int,
    ):
        reference_rects = tuple(anchor.reference_rect for anchor in self._profile.anchors)
        if any(rect is None for rect in reference_rects):
            return lambda rect: _map_rect(rect, frame_width, frame_height)
        complete_references = tuple(
            reference for reference in reference_rects if reference is not None
        )
        scale_x, scale_y, offset_x, offset_y = _fit_anchor_transform(
            complete_references,
            matches,
            self._profile.reference_width,
            self._profile.reference_height,
        )
        _validate_transform_scale(
            scale_x,
            scale_y,
            complete_references,
            self._profile.anchors,
            self._profile.reference_width,
            self._profile.reference_height,
            frame_width,
            frame_height,
        )
        return lambda rect: _map_transformed_rect(
            rect,
            self._profile.reference_width,
            self._profile.reference_height,
            frame_width,
            frame_height,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
        )

    @staticmethod
    def _match_anchor(
        search_image: NDArray[np.uint8],
        template: NDArray[np.uint8],
        scale_range: tuple[float, float],
        frame_scale_x: float,
        frame_scale_y: float,
    ) -> _AnchorMatch:
        best_score = -1.0
        best_match = _AnchorMatch(-1.0, 0, 0, 0, 0)
        multipliers = _scale_candidates(scale_range)
        for multiplier_x in multipliers:
            for multiplier_y in multipliers:
                width = max(
                    1,
                    round(template.shape[1] * frame_scale_x * multiplier_x),
                )
                height = max(
                    1,
                    round(template.shape[0] * frame_scale_y * multiplier_y),
                )
                if width > search_image.shape[1] or height > search_image.shape[0]:
                    continue
                scaled = cv2.resize(
                    template,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                result = cv2.matchTemplate(
                    search_image,
                    scaled,
                    cv2.TM_CCOEFF_NORMED,
                )
                if result.size == 0:
                    continue
                (
                    _minimum,
                    maximum,
                    _minimum_location,
                    _maximum_location,
                ) = cv2.minMaxLoc(result)
                if math.isfinite(maximum):
                    best_score = float(maximum)
                    if best_score > best_match.score:
                        best_match = _AnchorMatch(
                            best_score,
                            int(_maximum_location[0]),
                            int(_maximum_location[1]),
                            width,
                            height,
                        )
        return best_match


@dataclass(frozen=True, slots=True)
class _AnchorMatch:
    score: float
    x: int
    y: int
    width: int
    height: int


def _fit_anchor_transform(
    references: Sequence[NormalizedRect],
    matches: Sequence[_AnchorMatch],
    reference_width: int,
    reference_height: int,
) -> tuple[float, float, float, float]:
    reference_centers_x = np.asarray(
        [(rect.x + rect.width / 2.0) * reference_width for rect in references],
        dtype=np.float64,
    )
    reference_centers_y = np.asarray(
        [(rect.y + rect.height / 2.0) * reference_height for rect in references],
        dtype=np.float64,
    )
    match_centers_x = np.asarray(
        [match.x + match.width / 2.0 for match in matches],
        dtype=np.float64,
    )
    match_centers_y = np.asarray(
        [match.y + match.height / 2.0 for match in matches],
        dtype=np.float64,
    )
    if not all(
        np.isfinite(values).all()
        for values in (
            reference_centers_x,
            reference_centers_y,
            match_centers_x,
            match_centers_y,
        )
    ):
        raise LayoutTransformError("anchor centers must be finite")
    scale_x, offset_x = _fit_axis(reference_centers_x, match_centers_x)
    scale_y, offset_y = _fit_axis(reference_centers_y, match_centers_y)
    return scale_x, scale_y, offset_x, offset_y


def _fit_axis(
    reference_centers: NDArray[np.float64],
    match_centers: NDArray[np.float64],
) -> tuple[float, float]:
    centered = reference_centers - reference_centers.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        raise LayoutTransformError("anchor reference centers must span both axes")
    scale = float(np.dot(centered, match_centers - match_centers.mean()) / denominator)
    offset = float(match_centers.mean() - scale * reference_centers.mean())
    return scale, offset


def validate_anchor_reference_geometry(
    references: Sequence[NormalizedRect],
    reference_width: int,
    reference_height: int,
) -> None:
    centers_x = tuple(
        (rect.x + rect.width / 2.0) * reference_width for rect in references
    )
    centers_y = tuple(
        (rect.y + rect.height / 2.0) * reference_height for rect in references
    )
    span_x = max(centers_x) - min(centers_x)
    span_y = max(centers_y) - min(centers_y)
    minimum_x = max(rect.width * reference_width for rect in references)
    minimum_y = max(rect.height * reference_height for rect in references)
    if span_x + 1e-12 < minimum_x:
        raise ValueError(
            "anchor reference x center span must be at least the widest anchor"
        )
    if span_y + 1e-12 < minimum_y:
        raise ValueError(
            "anchor reference y center span must be at least the tallest anchor"
        )


def _validate_transform_scale(
    scale_x: float,
    scale_y: float,
    references: Sequence[NormalizedRect],
    anchors: Sequence[AnchorProfile],
    reference_width: int,
    reference_height: int,
    frame_width: int,
    frame_height: int,
) -> None:
    values = (scale_x, scale_y)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise LayoutTransformError("anchor transform scale must be finite and positive")
    centers_x = tuple(
        (rect.x + rect.width / 2.0) * reference_width for rect in references
    )
    centers_y = tuple(
        (rect.y + rect.height / 2.0) * reference_height for rect in references
    )
    span_x = max(centers_x) - min(centers_x)
    span_y = max(centers_y) - min(centers_y)
    tolerance_x = 1.0 / span_x
    tolerance_y = 1.0 / span_y
    frame_scale_x = frame_width / reference_width
    frame_scale_y = frame_height / reference_height
    lower_x = max(frame_scale_x * anchor.scale_range[0] for anchor in anchors)
    upper_x = min(frame_scale_x * anchor.scale_range[1] for anchor in anchors)
    lower_y = max(frame_scale_y * anchor.scale_range[0] for anchor in anchors)
    upper_y = min(frame_scale_y * anchor.scale_range[1] for anchor in anchors)
    if (
        scale_x < lower_x - tolerance_x
        or scale_x > upper_x + tolerance_x
        or scale_y < lower_y - tolerance_y
        or scale_y > upper_y + tolerance_y
    ):
        raise LayoutTransformError(
            "anchor transform scale is outside the profile scale_range envelope"
        )


class _Detector(Protocol):
    def detect(self, frame: NDArray[np.uint8]) -> DetectedLayout | None: ...


class MultiProfileLayoutDetector:
    """Select the strongest complete anchor match from named layout profiles."""

    def __init__(
        self,
        detectors: Sequence[tuple[str, _Detector]],
        *,
        ambiguity_margin: float = 0.02,
    ) -> None:
        if not detectors:
            raise ValueError("at least one layout detector is required")
        if not 0 <= ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin must be between zero and one")
        names = [name for name, _detector in detectors]
        if any(not name.strip() for name in names) or len(names) != len(set(names)):
            raise ValueError("layout detector names must be non-empty and unique")
        self._detectors = tuple(detectors)
        self._ambiguity_margin = float(ambiguity_margin)

    def detect(self, frame: NDArray[np.uint8]) -> DetectedLayout | None:
        matches: list[tuple[int, str, DetectedLayout]] = []
        for index, (name, detector) in enumerate(self._detectors):
            layout = detector.detect(frame)
            if layout is not None:
                matches.append((index, name, layout))
        if not matches:
            return None
        matches.sort(key=lambda item: _layout_quality(item[2]), reverse=True)
        _index, name, layout = matches[0]
        if len(matches) > 1:
            gap = _layout_quality(layout) - _layout_quality(matches[1][2])
            if gap + 1e-12 < self._ambiguity_margin:
                return None
        return replace(layout, profile_name=name)


class PanelGeometryLayoutDetector:
    """Locate the fixed 2x2 pale option grid without image templates."""

    def detect(self, frame: NDArray[np.uint8]) -> DetectedLayout | None:
        if frame.ndim not in (2, 3) or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        if width < 100 or height < 100:
            return None
        bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (105, 5, 130), (130, 90, 255))
        # Long picture questions can extend well above the option grid. Keep the
        # upper panel visible here; the 2x2 grid geometry remains the gate that
        # prevents unrelated pale game panels from becoming a layout.
        mask[: round(height * 0.1)] = 0
        mask[round(height * 0.8) :] = 0
        # Tightly cropped DianShi windows can place the right option column at
        # 95% of the frame width. The grid geometry is a stronger false-positive
        # guard than hard clipping the outer game area here.
        mask[:, : round(width * 0.1)] = 0
        mask[:, round(width * 0.98) :] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
        )
        contours = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        panels: list[Rect] = []
        candidates: list[Rect] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            normalized_width = box_width / width
            normalized_height = box_height / height
            area = float(cv2.contourArea(contour))
            if area / (box_width * box_height) < 0.55:
                continue
            panel = Rect(x, y, box_width, box_height)
            if 0.08 < normalized_width < 0.75 and 0.025 < normalized_height < 0.45:
                panels.append(panel)
            if not (
                0.08 < normalized_width < 0.35
                and 0.025 < normalized_height < 0.13
            ):
                continue
            candidates.append(panel)

        grid = _select_option_grid(candidates, width, height)
        if grid is None:
            return None
        top_left, top_right = grid[:2]
        option_width_ratio = sum(item.width for item in grid) / (len(grid) * width)
        picture_layout = option_width_ratio < 0.17
        question_panel = _select_question_panel(panels, grid, width, height)
        if question_panel is not None:
            left_inset = round(width * 0.01)
            right_inset = round(width * 0.005)
            vertical_inset = round(height * 0.01)
            question = _clamp_rect(
                question_panel.x + left_inset,
                question_panel.y + vertical_inset,
                question_panel.width - left_inset - right_inset,
                question_panel.height - vertical_inset * 2,
                width,
                height,
            )
        else:
            question_x = top_left.x + round(
                width * (0.015 if picture_layout else 0.02)
            )
            question_right = top_right.x + top_right.width - round(width * 0.005)
            question_y = top_left.y - round(
                height * (0.19 if picture_layout else 0.15)
            )
            question_height = round(
                height * (0.16 if picture_layout else 0.11)
            )
            question = _clamp_rect(
                question_x,
                question_y,
                question_right - question_x,
                question_height,
                width,
                height,
            )
        horizontal_padding = round(width * 0.005)
        vertical_padding = round(height * 0.004)
        options = tuple(
            _clamp_rect(
                item.x - horizontal_padding,
                item.y - vertical_padding,
                item.width + horizontal_padding * 2,
                item.height + vertical_padding * 2,
                width,
                height,
            )
            for item in grid
        )
        return DetectedLayout(
            question_rect=question,
            option_rects=options,  # type: ignore[arg-type]
            anchor_scores=(1.0,),
            profile_name="keju-panel-fallback",
        )


class FallbackLayoutDetector:
    def __init__(self, primary: _Detector, fallback: _Detector) -> None:
        self._primary = primary
        self._fallback = fallback

    def detect(self, frame: NDArray[np.uint8]) -> DetectedLayout | None:
        return self._primary.detect(frame) or self._fallback.detect(frame)


def _select_option_grid(
    candidates: Sequence[Rect],
    frame_width: int,
    frame_height: int,
) -> tuple[Rect, ...] | None:
    four_options = _select_four_option_grid(candidates, frame_width, frame_height)
    if four_options is not None:
        return four_options
    return _select_three_option_grid(candidates, frame_width, frame_height)


def _select_four_option_grid(
    candidates: Sequence[Rect],
    frame_width: int,
    frame_height: int,
) -> tuple[Rect, Rect, Rect, Rect] | None:
    best: tuple[float, tuple[Rect, Rect, Rect, Rect]] | None = None
    for group in combinations(candidates, 4):
        by_y = sorted(group, key=lambda item: (item.y + item.height / 2, item.x))
        top = sorted(by_y[:2], key=lambda item: item.x)
        bottom = sorted(by_y[2:], key=lambda item: item.x)
        widths = [item.width for item in group]
        heights = [item.height for item in group]
        if max(widths) / min(widths) > 1.25 or max(heights) / min(heights) > 1.4:
            continue
        row_tolerance = max(heights) * 0.5
        top_centers = [item.y + item.height / 2 for item in top]
        bottom_centers = [item.y + item.height / 2 for item in bottom]
        if (
            abs(top_centers[0] - top_centers[1]) > row_tolerance
            or abs(bottom_centers[0] - bottom_centers[1]) > row_tolerance
        ):
            continue
        row_gap = sum(bottom_centers) / 2 - sum(top_centers) / 2
        if not frame_height * 0.05 < row_gap < frame_height * 0.2:
            continue
        top_x = [item.x + item.width / 2 for item in top]
        bottom_x = [item.x + item.width / 2 for item in bottom]
        if any(
            abs(first - second) > frame_width * 0.035
            for first, second in zip(top_x, bottom_x, strict=True)
        ):
            continue
        horizontal_gaps = [
            top[1].x - (top[0].x + top[0].width),
            bottom[1].x - (bottom[0].x + bottom[0].width),
        ]
        if any(gap < 0 or gap > frame_width * 0.15 for gap in horizontal_gaps):
            continue
        score = (
            abs(top_centers[0] - top_centers[1])
            + abs(bottom_centers[0] - bottom_centers[1])
            + abs(top_x[0] - bottom_x[0])
            + abs(top_x[1] - bottom_x[1])
            + abs(widths[0] - widths[1])
            + abs(widths[2] - widths[3])
        )
        ordered = (top[0], top[1], bottom[0], bottom[1])
        if best is None or score < best[0]:
            best = (score, ordered)
    return None if best is None else best[1]


def _select_three_option_grid(
    candidates: Sequence[Rect],
    frame_width: int,
    frame_height: int,
) -> tuple[Rect, Rect, Rect] | None:
    best: tuple[float, tuple[Rect, Rect, Rect]] | None = None
    for group in combinations(candidates, 3):
        by_y = sorted(group, key=lambda item: (item.y + item.height / 2, item.x))
        top = sorted(by_y[:2], key=lambda item: item.x)
        bottom_left = by_y[2]
        widths = [item.width for item in group]
        heights = [item.height for item in group]
        if max(widths) / min(widths) > 1.25 or max(heights) / min(heights) > 1.4:
            continue
        row_tolerance = max(heights) * 0.5
        top_centers_y = [item.y + item.height / 2 for item in top]
        if abs(top_centers_y[0] - top_centers_y[1]) > row_tolerance:
            continue
        top_center_y = sum(top_centers_y) / 2
        bottom_center_y = bottom_left.y + bottom_left.height / 2
        row_gap = bottom_center_y - top_center_y
        if not frame_height * 0.05 < row_gap < frame_height * 0.2:
            continue
        top_centers_x = [item.x + item.width / 2 for item in top]
        bottom_center_x = bottom_left.x + bottom_left.width / 2
        if abs(top_centers_x[0] - bottom_center_x) > frame_width * 0.035:
            continue
        horizontal_gap = top[1].x - (top[0].x + top[0].width)
        if horizontal_gap < 0 or horizontal_gap > frame_width * 0.15:
            continue
        score = (
            abs(top_centers_y[0] - top_centers_y[1])
            + abs(top_centers_x[0] - bottom_center_x)
            + abs(widths[0] - widths[1])
            + abs(widths[0] - widths[2])
        )
        ordered = (top[0], top[1], bottom_left)
        if best is None or score < best[0]:
            best = (score, ordered)
    return None if best is None else best[1]


def _select_question_panel(
    panels: Sequence[Rect],
    grid: Sequence[Rect],
    frame_width: int,
    frame_height: int,
) -> Rect | None:
    grid_left = min(item.x for item in grid)
    grid_right = max(item.x + item.width for item in grid)
    grid_width = grid_right - grid_left
    option_top = min(item.y for item in grid)
    option_height = sum(item.height for item in grid) / len(grid)
    best: tuple[float, Rect] | None = None
    for panel in panels:
        if panel in grid or panel.y >= option_top:
            continue
        panel_right = panel.x + panel.width
        bottom_gap = option_top - (panel.y + panel.height)
        if not (
            panel.width >= grid_width * 0.75
            and panel.height >= option_height * 1.1
            and -frame_height * 0.02 <= bottom_gap <= frame_height * 0.1
            and abs(panel.x - grid_left) <= frame_width * 0.08
            and abs(panel_right - grid_right) <= frame_width * 0.08
        ):
            continue
        score = (
            abs(panel.x - grid_left)
            + abs(panel_right - grid_right)
            + abs(bottom_gap)
        )
        if best is None or score < best[0]:
            best = (score, panel)
    return None if best is None else best[1]


def _clamp_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
) -> Rect:
    left = min(frame_width - 1, max(0, x))
    top = min(frame_height - 1, max(0, y))
    right = min(frame_width, max(left + 1, x + width))
    bottom = min(frame_height, max(top + 1, y + height))
    return Rect(left, top, right - left, bottom - top)


def _layout_quality(layout: DetectedLayout) -> float:
    average = sum(layout.anchor_scores) / len(layout.anchor_scores)
    weakest = min(layout.anchor_scores)
    return (average + weakest) / 2.0


def build_layout_detector(
    profiles: Sequence[LayoutProfile],
) -> TemplateLayoutDetector | FallbackLayoutDetector:
    if not profiles:
        raise ValueError("at least one layout profile is required")
    if len(profiles) == 1:
        return TemplateLayoutDetector(profiles[0])
    return FallbackLayoutDetector(
        PanelGeometryLayoutDetector(),
        MultiProfileLayoutDetector(
            tuple((profile.name, TemplateLayoutDetector(profile)) for profile in profiles)
        ),
    )


def _scale_candidates(scale_range: tuple[float, float]) -> tuple[float, ...]:
    low, high = scale_range
    if low == high:
        return (low,)
    values = {low, high, min(max(1.0, low), high)}
    step = 0.05
    current = math.ceil(low / step) * step
    while current < high:
        values.add(round(current, 6))
        current += step
    return tuple(sorted(values))


def _map_rect(
    normalized: NormalizedRect,
    frame_width: int,
    frame_height: int,
) -> Rect:
    left = min(frame_width - 1, max(0, round(normalized.x * frame_width)))
    top = min(frame_height - 1, max(0, round(normalized.y * frame_height)))
    right = min(
        frame_width,
        max(left + 1, round((normalized.x + normalized.width) * frame_width)),
    )
    bottom = min(
        frame_height,
        max(top + 1, round((normalized.y + normalized.height) * frame_height)),
    )
    return Rect(left, top, right - left, bottom - top)


def _map_transformed_rect(
    normalized: NormalizedRect,
    reference_width: int,
    reference_height: int,
    frame_width: int,
    frame_height: int,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> Rect:
    left = round(normalized.x * reference_width * scale_x + offset_x)
    top = round(normalized.y * reference_height * scale_y + offset_y)
    right = round(
        (normalized.x + normalized.width) * reference_width * scale_x + offset_x
    )
    bottom = round(
        (normalized.y + normalized.height) * reference_height * scale_y + offset_y
    )
    left = min(frame_width - 1, max(0, left))
    top = min(frame_height - 1, max(0, top))
    right = min(frame_width, max(left + 1, right))
    bottom = min(frame_height, max(top + 1, bottom))
    return Rect(left, top, right - left, bottom - top)


def validate_anchor_templates(profile: LayoutProfile) -> tuple[NDArray[np.uint8], ...]:
    templates: list[NDArray[np.uint8]] = []
    for anchor in profile.anchors:
        template = cv2.imread(str(anchor.template_path), cv2.IMREAD_GRAYSCALE)
        if template is None or template.size == 0:
            raise ValueError(f"anchor 不可读：{anchor.template_path}")
        templates.append(template)
    return tuple(templates)


__all__ = [
    "FallbackLayoutDetector",
    "LayoutTransformError",
    "LayoutProfile",
    "MultiProfileLayoutDetector",
    "PanelGeometryLayoutDetector",
    "TemplateLayoutDetector",
    "build_layout_detector",
    "validate_anchor_templates",
    "validate_anchor_reference_geometry",
]
