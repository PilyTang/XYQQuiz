from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2
from PIL import Image, ImageDraw, ImageFont

from xyq_quiz.recognition.ocr import (
    LineSegmentationError,
    OCRRole,
    RapidOCREngine,
    segment_text_lines,
)


def _canvas(height: int = 100, width: int = 200) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.45,
    color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _banner_counterexample() -> np.ndarray:
    image = _canvas(height=160, width=480)
    cv2.rectangle(image, (20, 30), (460, 72), (0, 0, 0), -1)
    cv2.putText(
        image,
        "KEJU EVENT",
        (70, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "small question",
        (70, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return image


def test_question_line_segmentation_uses_projection_padding_and_original_color() -> None:
    image = _canvas()
    _draw_text(image, "Alpha", (20, 32), color=(10, 20, 30))
    _draw_text(image, "Beta", (40, 62), color=(30, 40, 50))

    lines = segment_text_lines(image, OCRRole.QUESTION)

    assert len(lines) == 2
    assert np.any(lines[0][:, :, 0] < lines[0][:, :, 1])
    assert np.any(lines[1][:, :, 0] < lines[1][:, :, 1])


def test_option_line_segmentation_ignores_border_and_is_not_layout_specific() -> None:
    image = _canvas()
    image[5:12, 40:80] = 0  # outside the option role's 22%..88% vertical band
    _draw_text(image, "option", (40, 45))

    lines = segment_text_lines(image, OCRRole.OPTION)

    assert len(lines) == 1
    assert np.any(lines[0] < 150)


def test_line_segmentation_ignores_isolated_noise_but_keeps_small_punctuation() -> None:
    image = _canvas()
    image[25, 30] = 0
    _draw_text(image, "i,.", (50, 50), scale=0.35)

    lines = segment_text_lines(image, OCRRole.QUESTION)

    assert len(lines) == 1
    assert np.count_nonzero(np.any(lines[0] < 150, axis=2)) >= 8


def test_line_segmentation_subtracts_persistent_vertical_border_projection() -> None:
    image = _canvas()
    image[18:67, 10:13] = 0
    _draw_text(image, "line one", (40, 40), scale=0.35)
    _draw_text(image, "line two", (50, 65), scale=0.35)

    lines = segment_text_lines(image, OCRRole.QUESTION)

    assert len(lines) == 2


def test_line_segmentation_handles_discontinuous_vertical_decoration_and_small_lines() -> None:
    image = _canvas()
    for top, bottom in ((18, 35), (38, 55), (58, 76)):
        image[top:bottom, 10:13] = 0
    _draw_text(image, "tiny", (40, 34), scale=0.25)
    _draw_text(image, "small", (50, 59), scale=0.25)

    lines = segment_text_lines(image, OCRRole.QUESTION)

    assert len(lines) == 2


def test_line_segmentation_rejects_one_implausibly_tall_vertical_component() -> None:
    image = _canvas()
    image[25:65, 30:33] = 0

    with pytest.raises(LineSegmentationError, match="unreliable"):
        segment_text_lines(image, OCRRole.QUESTION)


def test_line_segmentation_accepts_tall_group_with_textlike_horizontal_span() -> None:
    image = _canvas(width=320)
    cv2.putText(
        image,
        "WIDE TEXT",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    lines = segment_text_lines(image, OCRRole.QUESTION)

    assert len(lines) == 1


def test_line_segmentation_rejects_dense_banner_before_losing_small_question() -> None:
    with pytest.raises(LineSegmentationError, match="unreliable"):
        segment_text_lines(_banner_counterexample(), OCRRole.QUESTION)


def test_line_segmentation_keeps_small_chinese_text_and_punctuation() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    canvas = Image.new("RGB", (240, 100), "white")
    ImageDraw.Draw(canvas).text(
        (35, 35),
        "《小题》，。",
        font=ImageFont.truetype(str(font_path), 14),
        fill=(20, 30, 40),
    )
    bgr = np.asarray(canvas)[:, :, ::-1].copy()

    lines = segment_text_lines(bgr, OCRRole.QUESTION)

    assert len(lines) == 1
    assert lines[0].ndim == 3
    assert np.any(lines[0] != 255)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 10, 3), dtype=np.uint8),
        np.empty((10, 0, 3), dtype=np.uint8),
        np.zeros((10,), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
    ],
)
def test_line_segmentation_rejects_empty_or_invalid_images(image: np.ndarray) -> None:
    with pytest.raises(LineSegmentationError):
        segment_text_lines(image, OCRRole.QUESTION)


def test_line_segmentation_rejects_blank_and_excessive_lines() -> None:
    with pytest.raises(LineSegmentationError, match="no text lines"):
        segment_text_lines(_canvas(), OCRRole.QUESTION)

    image = _canvas(height=420)
    for top in range(25, 386, 30):
        cv2.rectangle(image, (40, top), (160, top + 17), (0, 0, 0), -1)
    with pytest.raises(LineSegmentationError, match="too many"):
        segment_text_lines(image, OCRRole.QUESTION)


def test_question_line_segmentation_accepts_long_picture_question() -> None:
    image = _canvas(height=300, width=420)
    for index, baseline in enumerate(range(35, 252, 27)):
        _draw_text(image, f"picture question line {index}", (30, baseline), scale=0.5)

    lines = segment_text_lines(image, OCRRole.QUESTION)

    assert len(lines) == 9


def test_segmented_rec_only_uses_public_rapidocr_flags_and_joins_in_order() -> None:
    calls: list[dict[str, object]] = []

    class FakeRapidOCR:
        def __call__(self, image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            calls.append({"shape": image.shape, **kwargs})
            index = len(calls) - 1
            return SimpleNamespace(
                boxes=None,
                txts=(("第一行", "，第二行")[index],),
                scores=((0.97, 0.83)[index],),
                elapse=(0.004, 0.006)[index],
            )

    image = _canvas()
    _draw_text(image, "first", (20, 32))
    _draw_text(image, "second", (30, 62))
    expected_shapes = [
        line.shape for line in segment_text_lines(image, OCRRole.QUESTION)
    ]
    engine = RapidOCREngine(engine_factory=FakeRapidOCR)

    result = engine.recognize_region(image, OCRRole.QUESTION, fallback_image=image)

    assert result.text == "第一行，第二行"
    assert result.confidence == pytest.approx(0.83)
    assert result.elapsed_ms == pytest.approx(10.0)
    assert calls == [
        {
            "shape": shape,
            "use_det": False,
            "use_cls": False,
            "use_rec": True,
        }
        for shape in expected_shapes
    ]


def test_question_rec_only_ignores_zero_confidence_empty_decorative_line() -> None:
    calls = 0

    class FakeRapidOCR:
        def __call__(self, _image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            nonlocal calls
            assert kwargs
            outputs = (
                ("", 0.0),
                ("科举大赛殿试部分第2关：", 0.99),
                ("哪个时期国家设立五经博士", 0.98),
            )
            text, score = outputs[calls]
            calls += 1
            return SimpleNamespace(txts=(text,), scores=(score,), elapse=0.001)

    image = _canvas(height=160, width=480)
    _draw_text(image, "decoration", (20, 35))
    _draw_text(image, "question line one", (40, 80))
    _draw_text(image, "question line two", (40, 125))
    engine = RapidOCREngine(engine_factory=FakeRapidOCR)

    result = engine.recognize_region(
        image,
        OCRRole.QUESTION,
        fallback_image=image,
    )

    assert result.text == "科举大赛殿试部分第2关：哪个时期国家设立五经博士"
    assert result.confidence == pytest.approx(0.98)
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.rec_only_success_count == 1
    assert diagnostics.fallback_count == 0


@pytest.mark.parametrize("failure", ["empty", "low", "unsupported", "error"])
def test_segmented_rec_only_safely_falls_back_to_complete_detection(failure: str) -> None:
    calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    class FakeRapidOCR:
        def __call__(self, image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            calls.append((image.shape, kwargs))
            if kwargs:
                if failure == "unsupported":
                    raise TypeError("unexpected keyword argument use_det")
                if failure == "error":
                    raise RuntimeError("recognizer failed")
                if failure == "empty":
                    return SimpleNamespace(txts=("",), scores=(0.99,), elapse=0.001)
                return SimpleNamespace(txts=("低置信",), scores=(0.2,), elapse=0.001)
            return SimpleNamespace(
                boxes=None,
                txts=("完整检测",),
                scores=(0.95,),
                elapse=0.020,
            )

    raw = _canvas()
    _draw_text(raw, "option", (40, 48))
    fallback = np.repeat(np.repeat(raw, 3, axis=0), 3, axis=1)
    engine = RapidOCREngine(engine_factory=FakeRapidOCR)

    result = engine.recognize_region(raw, OCRRole.OPTION, fallback_image=fallback)

    assert result.text == "完整检测"
    assert result.confidence == pytest.approx(0.95)
    assert calls[-1] == (fallback.shape, {})
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.fallback_count == 1
    assert diagnostics.line_count_distribution == {1: 1}


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01])
def test_rec_only_invalid_score_always_falls_back(score: float) -> None:
    class FakeRapidOCR:
        def __call__(self, _image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            if kwargs:
                return SimpleNamespace(txts=("快路",), scores=(score,), elapse=0.001)
            return SimpleNamespace(txts=("完整检测",), scores=(0.9,), elapse=0.010)

    image = _canvas()
    _draw_text(image, "option", (40, 48))
    engine = RapidOCREngine(engine_factory=FakeRapidOCR)

    assert engine.recognize_region(
        image, OCRRole.OPTION, fallback_image=image
    ).text == "完整检测"
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.rec_only_success_count == 0
    assert diagnostics.fallback_count == 1


@pytest.mark.parametrize(
    ("texts", "scores"),
    [
        (("两项", "文本"), (0.9,)),
        ((123,), (0.9,)),
        (("文本",), (True,)),
        ("文本", (0.9,)),
        (("文本",), "0.9"),
        (("文本",), (object(),)),
    ],
)
def test_rec_only_malformed_output_always_falls_back(
    texts: object,
    scores: object,
) -> None:
    class FakeRapidOCR:
        def __call__(self, _image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            if kwargs:
                return SimpleNamespace(txts=texts, scores=scores, elapse=0.001)
            return SimpleNamespace(txts=("完整检测",), scores=(0.9,), elapse=0.010)

    image = _canvas()
    _draw_text(image, "option", (40, 48))
    engine = RapidOCREngine(engine_factory=FakeRapidOCR)

    result = engine.recognize_region(image, OCRRole.OPTION, fallback_image=image)

    assert result.text == "完整检测"
    assert engine.diagnostics_snapshot().fallback_count == 1


@pytest.mark.parametrize("score", [np.bool_(True), np.bool_(False)])
def test_numpy_boolean_score_always_falls_back(score: np.bool_) -> None:
    class FakeRapidOCR:
        def __call__(self, _image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            if kwargs:
                return SimpleNamespace(txts=("快路",), scores=(score,), elapse=0.001)
            return SimpleNamespace(txts=("完整检测",), scores=(0.9,), elapse=0.010)

    image = _canvas()
    cv2.putText(
        image,
        "text",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    engine = RapidOCREngine(engine_factory=FakeRapidOCR)

    result = engine.recognize_region(image, OCRRole.OPTION, fallback_image=image)

    assert result.text == "完整检测"
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.rec_only_success_count == 0
    assert diagnostics.fallback_count == 1


def test_dense_banner_forces_complete_detection_before_fake_rec_only() -> None:
    calls: list[dict[str, object]] = []

    class FakeRapidOCR:
        def __call__(self, _image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(txts=("完整题面",), scores=(0.9,), elapse=0.010)

    engine = RapidOCREngine(engine_factory=FakeRapidOCR)

    result = engine.recognize_region(
        _banner_counterexample(),
        OCRRole.QUESTION,
        fallback_image=_banner_counterexample(),
    )

    assert result.text == "完整题面"
    assert calls == [{}]
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.rec_only_success_count == 0
    assert diagnostics.fallback_count == 1


def test_unreliable_segmentation_falls_back_before_any_rec_only_call() -> None:
    calls: list[dict[str, object]] = []

    class FakeRapidOCR:
        def __call__(self, _image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(txts=("完整检测",), scores=(0.9,), elapse=0.010)

    image = _canvas()
    image[25:65, 30:33] = 0
    engine = RapidOCREngine(engine_factory=FakeRapidOCR)

    result = engine.recognize_region(image, OCRRole.QUESTION, fallback_image=image)

    assert result.text == "完整检测"
    assert calls == [{}]
    assert engine.diagnostics_snapshot().fallback_count == 1


def test_diagnostics_snapshot_does_not_expose_mutable_internal_state() -> None:
    engine = RapidOCREngine(engine_factory=lambda: None)
    snapshot = engine.diagnostics_snapshot()

    with pytest.raises(TypeError):
        snapshot.line_count_distribution[1] = 99  # type: ignore[index]

    assert engine.diagnostics_snapshot().line_count_distribution == {}


def test_segmentation_failure_falls_back_without_persisting_line_images() -> None:
    class FakeRapidOCR:
        def __call__(self, image: np.ndarray, **kwargs: object) -> SimpleNamespace:
            assert kwargs == {}
            return SimpleNamespace(txts=("fallback",), scores=(0.9,), elapse=0.001)

    engine = RapidOCREngine(engine_factory=FakeRapidOCR)
    result = engine.recognize_region(
        _canvas(), OCRRole.QUESTION, fallback_image=_canvas()
    )

    assert result.text == "fallback"
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.fallback_count == 1
    assert diagnostics.line_count_distribution == {}
