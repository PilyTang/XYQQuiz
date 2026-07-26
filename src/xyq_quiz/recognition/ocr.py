from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from xyq_quiz.recognition.models import OCRText


class OCRUnavailable(RuntimeError):
    """Raised when the optional local OCR runtime cannot be initialized."""


class LineSegmentationError(ValueError):
    """Raised when a crop cannot be split into a safe number of text lines."""


class OCRRole(str, Enum):
    QUESTION = "question"
    OPTION = "option"


@dataclass(frozen=True, slots=True)
class OCRDiagnostics:
    rec_only_success_count: int
    fallback_count: int
    line_count_distribution: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class _LineProfile:
    x_start: float
    x_end: float
    y_start: float
    y_end: float
    max_lines: int


_LINE_PROFILES = {
    # Picture questions can contain nine or more lines. The left inset also
    # excludes the decorative question badge that sits outside the pale panel.
    OCRRole.QUESTION: _LineProfile(0.05, 0.995, 0.05, 0.97, 12),
    OCRRole.OPTION: _LineProfile(0.14, 0.97, 0.22, 0.88, 3),
}
_DARK_THRESHOLD = 150
_MIN_REC_ONLY_CONFIDENCE = 0.5
_PADDING_X = 10
_PADDING_Y = 8
_RAPIDOCR_PROVIDER_PATCH_LOCK = threading.Lock()


class OCREngine(Protocol):
    def recognize(self, image: NDArray[np.uint8]) -> OCRText: ...


class RoleAwareOCREngine(OCREngine, Protocol):
    def recognize_region(
        self,
        image: NDArray[np.uint8],
        role: OCRRole,
        *,
        fallback_image: NDArray[np.uint8],
    ) -> OCRText: ...


class RapidOCREngine:
    def __init__(
        self,
        engine_factory: Callable[[], Any] | None = None,
        *,
        backend: str = "cpu",
        device_id: int | None = None,
    ) -> None:
        if backend not in {"cpu", "directml"}:
            raise ValueError("RapidOCR backend must be cpu or directml")
        if backend == "directml" and (
            device_id is None or device_id < 0
        ):
            raise ValueError("DirectML requires a non-negative device_id")
        if backend == "cpu" and device_id is not None:
            raise ValueError("CPU OCR does not accept a device_id")
        self.backend = backend
        self.device_id = device_id
        self._engine_factory = engine_factory or (
            lambda: _default_engine_factory(
                backend=backend,
                device_id=device_id,
            )
        )
        self._thread_local = threading.local()
        self._initialization_lock = threading.Lock()
        self._initialization_error: OCRUnavailable | None = None
        self._diagnostics_lock = threading.Lock()
        self._rec_only_success_count = 0
        self._fallback_count = 0
        self._line_count_distribution: dict[int, int] = {}

    def recognize(self, image: NDArray[np.uint8]) -> OCRText:
        output = self._get_engine()(image)
        return _parse_detection_output(output)

    def recognize_region(
        self,
        image: NDArray[np.uint8],
        role: OCRRole,
        *,
        fallback_image: NDArray[np.uint8],
    ) -> OCRText:
        """Recognize a question/option crop by public rec-only calls per line.

        Projection is used only to locate lines. The recognizer always receives
        original-color pixels, and any unsafe result falls back to the existing
        complete detection input supplied by the pipeline.
        """
        try:
            lines = segment_text_lines(image, role)
        except LineSegmentationError:
            self._record_fallback()
            return self.recognize(fallback_image)

        self._record_line_count(len(lines))
        try:
            results = tuple(self._recognize_line(line) for line in lines)
            if role is OCRRole.QUESTION and len(results) > 1:
                # Wide dialog borders can form a projection line, but rec-only
                # reliably identifies them as completely empty. Ignore only
                # zero-confidence empty question lines; all non-empty or
                # uncertain lines still participate in the safety check.
                results = tuple(
                    result
                    for result in results
                    if result.text.strip() or result.confidence != 0.0
                )
            if any(
                not result.text.strip()
                or result.confidence < _MIN_REC_ONLY_CONFIDENCE
                for result in results
            ) or not results:
                raise ValueError("unsafe rec-only result")
        except Exception as exc:
            if isinstance(exc, OCRUnavailable):
                raise
            self._record_fallback()
            return self.recognize(fallback_image)

        with self._diagnostics_lock:
            self._rec_only_success_count += 1
        return OCRText(
            "".join(result.text for result in results),
            min(result.confidence for result in results),
            sum(result.elapsed_ms for result in results),
        )

    def diagnostics_snapshot(self) -> OCRDiagnostics:
        with self._diagnostics_lock:
            return OCRDiagnostics(
                self._rec_only_success_count,
                self._fallback_count,
                MappingProxyType(dict(self._line_count_distribution)),
            )

    def execution_providers(self) -> tuple[tuple[str, ...], ...]:
        """Return providers used by RapidOCR's det/cls/rec sessions.

        This is intentionally a fail-closed self-test hook.  Merely having the
        DML provider installed is insufficient because ORT can rebuild a
        rejected provider configuration on CPU while OCR still succeeds.
        """
        engine = self._get_engine()
        providers: list[tuple[str, ...]] = []
        for attribute in ("text_det", "text_cls", "text_rec"):
            component = getattr(engine, attribute, None)
            wrapper = getattr(component, "session", None)
            session = getattr(wrapper, "session", None)
            get_providers = getattr(session, "get_providers", None)
            if not callable(get_providers):
                raise OCRUnavailable(
                    f"RapidOCR {attribute} 无法报告实际 ONNX Runtime provider"
                )
            current = tuple(str(item) for item in get_providers())
            if not current:
                raise OCRUnavailable(f"RapidOCR {attribute} provider 列表为空")
            providers.append(current)
        return tuple(providers)

    def _recognize_line(self, image: NDArray[np.uint8]) -> OCRText:
        output = self._get_engine()(
            image,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
        return _parse_rec_only_output(output)

    def _record_line_count(self, line_count: int) -> None:
        with self._diagnostics_lock:
            self._line_count_distribution[line_count] = (
                self._line_count_distribution.get(line_count, 0) + 1
            )

    def _record_fallback(self) -> None:
        with self._diagnostics_lock:
            self._fallback_count += 1

    def _get_engine(self) -> Any:
        engine = getattr(self._thread_local, "engine", None)
        if engine is not None:
            return engine
        if self._initialization_error is not None:
            raise self._initialization_error
        with self._initialization_lock:
            engine = getattr(self._thread_local, "engine", None)
            if engine is not None:
                return engine
            if self._initialization_error is not None:
                raise self._initialization_error
            try:
                engine = self._engine_factory()
            except Exception as exc:
                self._initialization_error = OCRUnavailable(
                    "RapidOCR is unavailable; install it with "
                    "`.venv\\Scripts\\python.exe -m pip install rapidocr onnxruntime`"
                )
                raise self._initialization_error from exc
            self._thread_local.engine = engine
            return engine


def _parse_detection_output(output: Any) -> OCRText:
    texts = getattr(output, "txts", None)
    if not texts:
        return OCRText("", 0.0, _elapsed_ms(output))

    boxes = getattr(output, "boxes", None)
    items = list(enumerate(str(text) for text in texts))
    if boxes is not None and len(boxes) == len(items):
        items = _reading_order(items, boxes)
    joined_text = "".join(text for _index, text in items)

    scores = getattr(output, "scores", None)
    available_scores = (
        [float(score) for score in scores if score is not None]
        if scores is not None
        else []
    )
    confidence = (
        sum(available_scores) / len(available_scores)
        if available_scores
        else 0.0
    )
    return OCRText(joined_text, confidence, _elapsed_ms(output))


def _parse_rec_only_output(output: Any) -> OCRText:
    texts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    text_items = _output_items(texts)
    score_items = _output_items(scores)
    if (
        text_items is None
        or score_items is None
        or not text_items
        or len(text_items) != len(score_items)
        or not all(isinstance(text, str) for text in text_items)
    ):
        return OCRText("", 0.0, _elapsed_ms(output))
    available_scores: list[float] = []
    for score in score_items:
        if not isinstance(score, Real) or isinstance(score, (bool, np.bool_)):
            return OCRText("", 0.0, _elapsed_ms(output))
        try:
            normalized_score = float(score)
        except (TypeError, ValueError, OverflowError):
            return OCRText("", 0.0, _elapsed_ms(output))
        if (
            not math.isfinite(normalized_score)
            or normalized_score < 0.0
            or normalized_score > 1.0
        ):
            return OCRText("", 0.0, _elapsed_ms(output))
        available_scores.append(normalized_score)
    if not available_scores:
        return OCRText("", 0.0, _elapsed_ms(output))
    return OCRText(
        "".join(text_items),
        min(available_scores),
        _elapsed_ms(output),
    )


def _output_items(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, (str, bytes)) or value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        return tuple(value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return None


def segment_text_lines(
    image: NDArray[np.uint8],
    role: OCRRole,
) -> tuple[NDArray[np.uint8], ...]:
    """Locate text lines with threshold projections and return color crops."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim not in (2, 3)
        or image.size == 0
        or image.shape[0] == 0
        or image.shape[1] == 0
        or (image.ndim == 3 and image.shape[2] != 3)
    ):
        raise LineSegmentationError("invalid or empty OCR crop")
    try:
        profile = _LINE_PROFILES[OCRRole(role)]
    except (KeyError, ValueError) as exc:
        raise LineSegmentationError("unsupported OCR role") from exc

    height, width = image.shape[:2]
    left = int(np.floor(width * profile.x_start))
    right = int(np.ceil(width * profile.x_end))
    top = int(np.floor(height * profile.y_start))
    bottom = int(np.ceil(height * profile.y_end))
    if right <= left or bottom <= top:
        raise LineSegmentationError("OCR crop is too small")

    color_band = image[top:bottom, left:right]
    gray = (
        color_band
        if color_band.ndim == 2
        else _to_gray_without_mutating(color_band)
    )
    dark = gray < _DARK_THRESHOLD
    # A tall question panel makes decorative borders occupy a smaller fraction
    # of the vertical band, so accept half-height edge strokes as persistent.
    persistent_candidates = dark.mean(axis=0) >= 0.50
    edge_width = max(1, int(np.ceil(dark.shape[1] * 0.12)))
    edge_columns = np.zeros(dark.shape[1], dtype=bool)
    edge_columns[:edge_width] = True
    edge_columns[-edge_width:] = True
    persistent_columns = persistent_candidates & edge_columns
    projection_dark = dark.copy()
    projection_dark[:, persistent_columns] = False
    row_counts = projection_dark.sum(axis=1)
    persistent_floor = max(
        int(np.quantile(dark.sum(axis=1), 0.25)),
        int(np.quantile(row_counts, 0.25)),
    )
    min_row_pixels = max(
        2,
        int(np.ceil(dark.shape[1] * 0.002)),
        persistent_floor + 2,
    )
    active_rows = np.flatnonzero(row_counts >= min_row_pixels)
    if active_rows.size == 0:
        raise LineSegmentationError("no text lines found")

    max_gap = max(2, int(round(height * 0.015)))
    groups = _group_projection_indexes(active_rows, max_gap)
    min_line_height = max(2, int(np.ceil(height * 0.04)))
    groups = tuple(
        (start, end) for start, end in groups if end - start >= min_line_height
    )
    if not groups:
        raise LineSegmentationError("no text lines found")
    if len(groups) > profile.max_lines:
        raise LineSegmentationError("too many text lines")
    group_heights = tuple(end - start for start, end in groups)
    if (
        len(group_heights) > 1
        and max(group_heights) / min(group_heights) >= 4.0
    ):
        raise LineSegmentationError(
            f"unreliable text line height outlier: {group_heights}"
        )
    for start, end in groups:
        columns = np.flatnonzero(projection_dark[start:end].any(axis=0))
        ink = projection_dark[start:end, int(columns[0]) : int(columns[-1]) + 1]
        if float(ink.mean()) >= 0.85:
            raise LineSegmentationError("unreliable dense text line")
    max_line_height = int(np.ceil(projection_dark.shape[0] * 0.55))
    for start, end in groups:
        line_height = end - start
        if line_height <= max_line_height:
            continue
        columns = np.flatnonzero(projection_dark[start:end].any(axis=0))
        horizontal_span = int(columns[-1] - columns[0] + 1)
        if horizontal_span / line_height < 2.0:
            raise LineSegmentationError("unreliable text line geometry")

    lines: list[NDArray[np.uint8]] = []
    for row_start, row_end in groups:
        line_mask = projection_dark[row_start:row_end]
        active_columns = np.flatnonzero(line_mask.any(axis=0))
        if active_columns.size == 0:
            raise LineSegmentationError("text line has no content")
        crop_left = max(0, left + int(active_columns[0]) - _PADDING_X)
        crop_right = min(
            width,
            left + int(active_columns[-1]) + 1 + _PADDING_X,
        )
        crop_top = max(0, top + row_start - _PADDING_Y)
        crop_bottom = min(height, top + row_end + _PADDING_Y)
        line = image[crop_top:crop_bottom, crop_left:crop_right]
        if line.size == 0:
            raise LineSegmentationError("empty text line crop")
        lines.append(np.ascontiguousarray(line))
    return tuple(lines)


def _to_gray_without_mutating(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # Integer BT.601 approximation; thresholding is only used for projection.
    blue = image[:, :, 0].astype(np.uint16)
    green = image[:, :, 1].astype(np.uint16)
    red = image[:, :, 2].astype(np.uint16)
    return ((29 * blue + 150 * green + 77 * red) >> 8).astype(np.uint8)


def _group_projection_indexes(
    indexes: NDArray[np.intp],
    max_gap: int,
) -> tuple[tuple[int, int], ...]:
    groups: list[tuple[int, int]] = []
    start = previous = int(indexes[0])
    for raw_index in indexes[1:]:
        index = int(raw_index)
        if index - previous > max_gap + 1:
            groups.append((start, previous + 1))
            start = index
        previous = index
    groups.append((start, previous + 1))
    return tuple(groups)


def _default_engine_factory(
    *,
    backend: str = "cpu",
    device_id: int | None = None,
) -> Any:
    from rapidocr import RapidOCR

    # ONNX Runtime otherwise expands one small OCR request across every logical
    # processor.  Two intra-op workers measured faster than the unrestricted
    # default on the production-sized question crop while using far less total
    # CPU; OCR calls are already serialized by RecognitionPipeline.
    params: dict[str, object] = {
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    }
    if backend == "directml":
        if device_id is None or device_id < 0:
            raise ValueError("DirectML requires a non-negative device_id")
        params.update(
            {
                "EngineConfig.onnxruntime.use_dml": True,
                "EngineConfig.onnxruntime.dml_ep_cfg": {
                    # ORT 1.24.x requires provider option values to be strings;
                    # an int makes it reject the whole provider list and
                    # silently rebuild the session on CPU.
                    "device_id": str(device_id),
                },
            }
        )
    elif backend != "cpu":
        raise ValueError("RapidOCR backend must be cpu or directml")
    if backend != "directml":
        return RapidOCR(params=params)

    # RapidOCR 3.9.1 returns its DirectML provider options as an OmegaConf
    # DictConfig. ORT 1.24.4 accepts only a real dict in provider tuples and
    # otherwise silently retries on CPU. Patch the construction window only;
    # the upstream class is restored before this function returns.
    try:
        from rapidocr.inference_engine.onnxruntime.provider_config import (
            ProviderConfig,
        )
    except ImportError:
        return RapidOCR(params=params)

    original = ProviderConfig.dml_ep_cfg

    def dml_ep_cfg(config: Any) -> dict[str, Any]:
        value = config.cfg.dml_ep_cfg
        if value is not None:
            return dict(value)
        if config.is_cuda_available():
            return config.cuda_ep_cfg()
        return config.cpu_ep_cfg()

    with _RAPIDOCR_PROVIDER_PATCH_LOCK:
        ProviderConfig.dml_ep_cfg = dml_ep_cfg
        try:
            return RapidOCR(params=params)
        finally:
            ProviderConfig.dml_ep_cfg = original


def _box_geometry(box: Any) -> tuple[float, float, float]:
    points = np.asarray(box, dtype=np.float64)
    top = float(points[:, 1].min())
    bottom = float(points[:, 1].max())
    return (top + bottom) / 2.0, float(points[:, 0].min()), bottom - top


def _reading_order(
    items: list[tuple[int, str]],
    boxes: Any,
) -> list[tuple[int, str]]:
    positioned = [
        (*_box_geometry(boxes[index]), index, text)
        for index, text in items
    ]
    positioned.sort(key=lambda item: item[0])
    lines: list[list[tuple[float, float, float, int, str]]] = []
    for item in positioned:
        center_y, _left, height, _index, _text = item
        for line in lines:
            line_center = sum(entry[0] for entry in line) / len(line)
            line_height = max(entry[2] for entry in line)
            if abs(center_y - line_center) <= max(height, line_height) * 0.5:
                line.append(item)
                break
        else:
            lines.append([item])

    ordered: list[tuple[int, str]] = []
    for line in lines:
        line.sort(key=lambda item: item[1])
        ordered.extend((index, text) for _center, _left, _height, index, text in line)
    return ordered


def _elapsed_ms(output: Any) -> float:
    elapsed_seconds = getattr(output, "elapse", 0.0)
    if not isinstance(elapsed_seconds, (int, float)):
        return 0.0
    return max(0.0, float(elapsed_seconds) * 1000.0)


__all__ = [
    "LineSegmentationError",
    "OCREngine",
    "OCRDiagnostics",
    "OCRRole",
    "OCRUnavailable",
    "RapidOCREngine",
    "RoleAwareOCREngine",
    "segment_text_lines",
]
