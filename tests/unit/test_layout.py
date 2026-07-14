from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

import xyq_quiz.tools.calibrate as calibrate_module
from xyq_quiz.recognition.layout import (
    _AnchorMatch,
    _fit_anchor_transform,
    FallbackLayoutDetector,
    LayoutProfile,
    MultiProfileLayoutDetector,
    PanelGeometryLayoutDetector,
    TemplateLayoutDetector,
    build_layout_detector,
)
from xyq_quiz.recognition.models import DetectedLayout, NormalizedRect
from xyq_quiz.capture.models import Rect
from xyq_quiz.tools.calibrate import calibrate


_CALIBRATION_SELECTIONS = (
    (20, 20, 100, 15),
    (20, 45, 50, 10),
    (90, 45, 50, 10),
    (20, 65, 50, 10),
    (90, 65, 50, 10),
    (5, 5, 8, 8),
    (140, 85, 8, 8),
)


@pytest.mark.parametrize(
    ("boxes", "expected_question"),
    [
        (
            ((390, 420, 190, 55), (590, 420, 190, 55), (390, 510, 190, 55), (590, 510, 190, 55)),
            Rect(410, 300, 365, 88),
        ),
        (
            ((490, 450, 140, 44), (640, 450, 140, 44), (490, 520, 140, 44), (640, 520, 140, 44)),
            Rect(505, 298, 270, 128),
        ),
    ],
)
def test_panel_geometry_fallback_detects_text_and_picture_option_grids(
    boxes: tuple[tuple[int, int, int, int], ...],
    expected_question: Rect,
) -> None:
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    for x, y, width, height in boxes:
        frame[y : y + height, x : x + width] = (236, 215, 213)

    layout = PanelGeometryLayoutDetector().detect(frame)

    assert layout is not None
    assert layout.profile_name == "keju-panel-fallback"
    assert layout.question_rect == expected_question
    assert layout.option_rects[0] == Rect(boxes[0][0] - 5, boxes[0][1] - 3, boxes[0][2] + 10, boxes[0][3] + 6)


def test_panel_geometry_fallback_detects_three_option_xiangshi_grid() -> None:
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    for x, y in ((390, 420), (590, 420), (390, 510)):
        frame[y : y + 55, x : x + 190] = (236, 215, 213)

    layout = PanelGeometryLayoutDetector().detect(frame)

    assert layout is not None
    assert len(layout.option_rects) == 3


def test_panel_geometry_fallback_rejects_incomplete_two_option_grid() -> None:
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    for x, y in ((390, 420), (590, 420)):
        frame[y : y + 55, x : x + 190] = (236, 215, 213)

    assert PanelGeometryLayoutDetector().detect(frame) is None


def test_panel_geometry_fallback_uses_full_variable_height_question_panel() -> None:
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    question_panel = (490, 180, 290, 250)
    options = (
        (490, 450, 140, 44),
        (640, 450, 140, 44),
        (490, 520, 140, 44),
        (640, 520, 140, 44),
    )
    for x, y, width, height in (question_panel, *options):
        frame[y : y + height, x : x + width] = (236, 215, 213)

    layout = PanelGeometryLayoutDetector().detect(frame)

    assert layout is not None
    assert layout.question_rect == Rect(500, 188, 275, 234)


def test_panel_geometry_fallback_accepts_wide_dianshi_dialog_near_right_edge() -> None:
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    question_panel = (330, 210, 620, 190)
    options = (
        (330, 450, 285, 70),
        (655, 450, 285, 70),
        (330, 560, 285, 70),
        (655, 560, 285, 70),
    )
    for x, y, width, height in (question_panel, *options):
        frame[y : y + height, x : x + width] = (236, 215, 213)

    layout = PanelGeometryLayoutDetector().detect(frame)

    assert layout is not None
    assert len(layout.option_rects) == 4
    assert layout.question_rect == Rect(340, 218, 605, 174)


def test_fallback_detector_prefers_primary_result() -> None:
    expected = DetectedLayout(
        Rect(1, 2, 3, 4),
        (Rect(1, 6, 3, 4),) * 4,
        (0.99, 0.98),
    )

    class Primary:
        def detect(self, _frame: np.ndarray) -> DetectedLayout:
            return expected

    class RejectFallback:
        def detect(self, _frame: np.ndarray) -> None:
            raise AssertionError("fallback must not run when primary matched")

    detector = FallbackLayoutDetector(Primary(), RejectFallback())

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) is expected


def _write_profile(tmp_path: Path) -> tuple[Path, np.ndarray]:
    anchor_one = np.array(
        [
            [0, 0, 255, 0, 0, 255, 0, 0],
            [0, 255, 255, 255, 255, 255, 255, 0],
            [255, 255, 0, 0, 0, 0, 255, 255],
            [0, 255, 0, 255, 255, 0, 255, 0],
            [0, 255, 0, 255, 255, 0, 255, 0],
            [255, 255, 0, 0, 0, 0, 255, 255],
            [0, 255, 255, 255, 255, 255, 255, 0],
            [0, 0, 255, 0, 0, 255, 0, 0],
        ],
        dtype=np.uint8,
    )
    anchor_two = np.rot90(anchor_one).copy()
    assert cv2.imwrite(str(tmp_path / "anchor-one.png"), anchor_one)
    assert cv2.imwrite(str(tmp_path / "anchor-two.png"), anchor_two)

    profile_path = tmp_path / "layout.json"
    profile_path.write_text(
        json.dumps(
            {
                "reference_size": [100, 80],
                "question_rect": [0.1, 0.1, 0.8, 0.2],
                "option_rects": [
                    [0.1, 0.4, 0.35, 0.15],
                    [0.55, 0.4, 0.35, 0.15],
                    [0.1, 0.65, 0.35, 0.15],
                    [0.55, 0.65, 0.35, 0.15],
                ],
                "anchors": [
                    {
                        "search_rect": [0.0, 0.0, 0.5, 0.5],
                        "template_path": "anchor-one.png",
                        "threshold": 0.99,
                        "scale_range": [1.0, 1.0],
                    },
                    {
                        "search_rect": [0.5, 0.5, 0.5, 0.5],
                        "template_path": "anchor-two.png",
                        "threshold": 0.99,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    scaled_one = cv2.resize(anchor_one, (16, 16), interpolation=cv2.INTER_LINEAR)
    scaled_two = cv2.resize(anchor_two, (16, 16), interpolation=cv2.INTER_LINEAR)
    frame[20:36, 30:46] = scaled_one[:, :, None]
    frame[110:126, 140:156] = scaled_two[:, :, None]
    return profile_path, frame


def test_layout_maps_normalized_rois_to_current_frame(tmp_path: Path) -> None:
    profile_path, frame = _write_profile(tmp_path)

    layout = TemplateLayoutDetector(LayoutProfile.load(profile_path)).detect(frame)

    assert layout is not None
    assert layout.question_rect.x == 20
    assert layout.question_rect.y == 16
    assert layout.question_rect.width == 160
    assert layout.question_rect.height == 32
    assert len(layout.option_rects) == 4
    assert all(rect.width > 0 and rect.height > 0 for rect in layout.option_rects)
    assert all(score >= 0.99 for score in layout.anchor_scores)


def test_layout_rejects_frame_when_any_required_anchor_fails(tmp_path: Path) -> None:
    profile_path, frame = _write_profile(tmp_path)
    frame[110:126, 140:156] = 0

    layout = TemplateLayoutDetector(LayoutProfile.load(profile_path)).detect(frame)

    assert layout is None


class _ScoredDetector:
    def __init__(self, scores: tuple[float, ...] | None, *, x: int = 1) -> None:
        self.scores = scores
        self.x = x

    def detect(self, _frame: np.ndarray) -> DetectedLayout | None:
        if self.scores is None:
            return None
        return DetectedLayout(
            Rect(self.x, 1, 2, 2),
            (Rect(1, 4, 2, 2),) * 4,
            self.scores,
        )


def test_multi_profile_rejects_failed_profile_and_selects_best_total_score() -> None:
    detector = MultiProfileLayoutDetector(
        (
            ("failed", _ScoredDetector(None)),
            ("lower", _ScoredDetector((0.96, 0.95))),
            ("winner", _ScoredDetector((0.99, 0.98))),
        )
    )

    layout = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert layout is not None
    assert layout.profile_name == "winner"
    assert layout.anchor_scores == (0.99, 0.98)


def test_multi_profile_uses_minimum_anchor_score_to_break_total_score_tie() -> None:
    detector = MultiProfileLayoutDetector(
        (
            ("weak-anchor", _ScoredDetector((1.0, 0.8))),
            ("balanced", _ScoredDetector((0.9, 0.9))),
        )
    )

    layout = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert layout is not None
    assert layout.profile_name == "balanced"


def test_multi_profile_rejects_exactly_tied_different_rois() -> None:
    detector = MultiProfileLayoutDetector(
        (
            ("first", _ScoredDetector((0.98, 0.98), x=1)),
            ("second", _ScoredDetector((0.98, 0.98), x=6)),
        )
    )

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_multi_profile_rejects_quality_gap_below_default_margin() -> None:
    detector = MultiProfileLayoutDetector(
        (
            ("best", _ScoredDetector((0.99, 0.99))),
            ("runner-up", _ScoredDetector((0.971, 0.971))),
        )
    )

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_multi_profile_accepts_quality_gap_at_configured_margin() -> None:
    detector = MultiProfileLayoutDetector(
        (
            ("best", _ScoredDetector((0.99, 0.99))),
            ("runner-up", _ScoredDetector((0.97, 0.97))),
        ),
        ambiguity_margin=0.02,
    )

    layout = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert layout is not None
    assert layout.profile_name == "best"


def test_single_profile_builder_keeps_template_detector_compatibility(
    tmp_path: Path,
) -> None:
    profile_path, frame = _write_profile(tmp_path)

    detector = build_layout_detector((LayoutProfile.load(profile_path),))

    assert isinstance(detector, TemplateLayoutDetector)
    assert detector.detect(frame) is not None


def test_anchor_reference_rects_translate_rois_across_canvas_aspect_ratios(
    tmp_path: Path,
) -> None:
    pattern = np.array(
        [
            [0, 0, 255, 0, 255, 0, 0, 255],
            [0, 255, 0, 255, 0, 255, 255, 0],
            [255, 255, 0, 0, 255, 0, 255, 0],
            [0, 255, 255, 0, 0, 255, 0, 255],
            [255, 0, 255, 255, 0, 0, 255, 0],
            [0, 255, 0, 255, 255, 0, 0, 255],
            [255, 0, 255, 0, 255, 255, 255, 0],
            [0, 255, 0, 255, 0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )
    second = np.rot90(pattern).copy()
    assert cv2.imwrite(str(tmp_path / "one.png"), pattern)
    assert cv2.imwrite(str(tmp_path / "two.png"), second)
    profile_path = tmp_path / "translated.json"
    profile_path.write_text(
        json.dumps({
            "reference_size": [100, 80],
            "question_rect": [0.1, 0.1, 0.8, 0.2],
            "option_rects": [[0.1, 0.4, 0.35, 0.15]] * 4,
            "anchors": [
                {
                    "reference_rect": [0.05, 0.0625, 0.08, 0.1],
                    "search_rect": [0, 0, 0.5, 0.5],
                    "template_path": "one.png",
                    "threshold": 0.99,
                    "scale_range": [0.7, 1.0],
                },
                {
                    "reference_rect": [0.8, 0.8125, 0.08, 0.1],
                    "search_rect": [0.5, 0.5, 0.5, 0.5],
                    "template_path": "two.png",
                    "threshold": 0.99,
                    "scale_range": [0.7, 1.0],
                },
            ],
        }),
        encoding="utf-8",
    )
    frame = np.zeros((220, 200, 3), dtype=np.uint8)
    frame[40:56, 10:26] = cv2.resize(pattern, (16, 16))[:, :, None]
    frame[160:176, 160:176] = cv2.resize(second, (16, 16))[:, :, None]

    layout = TemplateLayoutDetector(LayoutProfile.load(profile_path)).detect(frame)

    assert layout is not None
    assert layout.question_rect == Rect(20, 46, 160, 32)


def test_anchor_templates_and_rois_support_true_nonuniform_scaling(
    tmp_path: Path,
) -> None:
    profile_path, _frame = _write_profile(tmp_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["anchors"][0]["reference_rect"] = [0.15, 0.125, 0.08, 0.1]
    payload["anchors"][1]["reference_rect"] = [0.7, 0.6875, 0.08, 0.1]
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    one = cv2.imread(str(tmp_path / "anchor-one.png"), cv2.IMREAD_GRAYSCALE)
    two = cv2.imread(str(tmp_path / "anchor-two.png"), cv2.IMREAD_GRAYSCALE)
    frame = np.zeros((240, 200, 3), dtype=np.uint8)
    frame[30:54, 30:46] = cv2.resize(one, (16, 24))[:, :, None]
    frame[165:189, 140:156] = cv2.resize(two, (16, 24))[:, :, None]

    layout = TemplateLayoutDetector(LayoutProfile.load(profile_path)).detect(frame)

    assert layout is not None
    assert layout.question_rect == Rect(20, 24, 160, 48)


def test_anchor_transform_uses_centers_not_quantized_template_sizes() -> None:
    references = (
        NormalizedRect(0.1, 0.1, 0.08, 0.1),
        NormalizedRect(0.8, 0.7, 0.08, 0.1),
    )
    matches = (
        _AnchorMatch(0.99, 31, 20, 14, 18),
        _AnchorMatch(0.99, 169, 118, 18, 14),
    )

    transform = _fit_anchor_transform(references, matches, 100, 80)

    assert transform == pytest.approx((2.0, 2.0, 10.0, 5.0))


@pytest.mark.parametrize(
    ("second_reference", "axis"),
    [
        ([0.15, 0.7, 0.08, 0.1], "x"),
        ([0.7, 0.125, 0.08, 0.1], "y"),
        ([0.2, 0.7, 0.08, 0.1], "x"),
        ([0.7, 0.19, 0.08, 0.1], "y"),
    ],
)
def test_profile_rejects_anchor_centers_with_insufficient_axis_span(
    tmp_path: Path,
    second_reference: list[float],
    axis: str,
) -> None:
    profile_path, _frame = _write_profile(tmp_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["anchors"][0]["reference_rect"] = [0.15, 0.125, 0.08, 0.1]
    payload["anchors"][1]["reference_rect"] = second_reference
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"anchor.*{axis}.*span"):
        LayoutProfile.load(profile_path)


@pytest.mark.parametrize(
    "second_anchor",
    [
        (5, 85, 8, 8),
        (140, 5, 8, 8),
        (10, 85, 8, 8),
        (140, 10, 8, 8),
    ],
)
def test_calibration_rejects_degenerate_anchor_geometry_before_writing(
    tmp_path: Path,
    second_anchor: tuple[int, int, int, int],
) -> None:
    image_path, _image = _write_calibration_image(tmp_path, "bad", False)
    output_path = tmp_path / "layouts" / "bad.json"
    selections = iter((*_CALIBRATION_SELECTIONS[:6], second_anchor))

    with pytest.raises(ValueError, match=r"anchor.*(x|y).*span"):
        calibrate(
            image_path,
            output_path,
            selector=lambda _image, _label: next(selections),
        )

    assert not output_path.exists()
    assert not output_path.parent.exists()


@pytest.mark.parametrize(
    "matches",
    [
        (
            _AnchorMatch(0.99, 150, 20, 16, 16),
            _AnchorMatch(0.99, 20, 112, 16, 16),
        ),
        (
            _AnchorMatch(0.99, 20, 120, 16, 16),
            _AnchorMatch(0.99, 160, 16, 16, 16),
        ),
        (
            _AnchorMatch(0.99, float("inf"), 16, 16, 16),
            _AnchorMatch(0.99, 160, 112, 16, 16),
        ),
        (
            _AnchorMatch(0.99, 20, 16, 16, 16),
            _AnchorMatch(0.99, 5355, 112, 16, 16),
        ),
    ],
    ids=("reverse-x", "reverse-y", "non-finite", "extreme-scale-97"),
)
def test_runtime_invalid_anchor_transform_returns_no_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    matches: tuple[_AnchorMatch, _AnchorMatch],
) -> None:
    profile_path, _frame = _write_profile(tmp_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["anchors"][0]["reference_rect"] = [0.15, 0.125, 0.08, 0.1]
    payload["anchors"][1]["reference_rect"] = [0.7, 0.6875, 0.08, 0.1]
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    detector = TemplateLayoutDetector(LayoutProfile.load(profile_path))
    queued = iter(matches)
    monkeypatch.setattr(
        detector,
        "_match_anchor",
        lambda *_args, **_kwargs: next(queued),
    )

    assert detector.detect(np.zeros((160, 200, 3), dtype=np.uint8)) is None


def test_unexpected_layout_exception_is_not_silently_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path, _frame = _write_profile(tmp_path)
    detector = TemplateLayoutDetector(LayoutProfile.load(profile_path))
    monkeypatch.setattr(
        detector,
        "_match_anchor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bug")),
    )

    with pytest.raises(RuntimeError, match="bug"):
        detector.detect(np.zeros((160, 200, 3), dtype=np.uint8))


def test_layout_profile_requires_exactly_four_options_and_two_anchors(
    tmp_path: Path,
) -> None:
    profile_path, _ = _write_profile(tmp_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["anchors"] = payload["anchors"][:1]
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="at least two anchors"):
        LayoutProfile.load(profile_path)


def test_calibration_cancel_does_not_write_partial_output(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    output_path = tmp_path / "layouts" / "keju-default.json"
    assert cv2.imwrite(str(image_path), np.full((100, 160, 3), 255, np.uint8))
    selections: Iterator[tuple[int, int, int, int] | None] = iter(
        [
            (10, 10, 100, 20),
            (10, 40, 50, 10),
            (80, 40, 50, 10),
            None,
        ]
    )

    completed = calibrate(
        image_path,
        output_path,
        selector=lambda _image, _label: next(selections),
    )

    assert completed is False
    assert not output_path.exists()
    assert not (output_path.parent / "anchors").exists()


@pytest.mark.parametrize("failure_stage", ["second_anchor", "profile"])
def test_failed_calibration_preserves_existing_profile_and_anchors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    output_path = tmp_path / "layouts" / "keju-default.json"
    old_image_path, old_image = _write_calibration_image(tmp_path, "old", False)
    _run_calibration(old_image_path, output_path)
    old_profile_bytes = output_path.read_bytes()
    old_anchor_paths = _profile_anchor_paths(output_path)
    old_anchor_bytes = {path: path.read_bytes() for path in old_anchor_paths}
    old_anchor_names = {path.name for path in old_anchor_paths}
    assert TemplateLayoutDetector(LayoutProfile.load(output_path)).detect(old_image)

    new_image_path, _new_image = _write_calibration_image(tmp_path, "new", True)
    real_replace = calibrate_module.os.replace
    published_anchor_count = 0

    def injected_replace(
        source: os.PathLike[str],
        destination: os.PathLike[str],
    ) -> None:
        nonlocal published_anchor_count
        destination_path = Path(destination)
        if destination_path.parent == output_path.parent / "anchors":
            published_anchor_count += 1
            if failure_stage == "second_anchor" and published_anchor_count == 2:
                raise OSError("injected second anchor publish failure")
        if failure_stage == "profile" and destination_path == output_path:
            raise OSError("injected profile publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(calibrate_module.os, "replace", injected_replace)

    with pytest.raises(OSError, match=f"injected {failure_stage.replace('_', ' ')}"):
        _run_calibration(new_image_path, output_path)

    assert output_path.read_bytes() == old_profile_bytes
    assert {path.name for path in (output_path.parent / "anchors").iterdir()} == (
        old_anchor_names
    )
    assert all(path.read_bytes() == old_anchor_bytes[path] for path in old_anchor_paths)
    assert not any(
        path.name.startswith(f".{output_path.stem}-")
        for path in output_path.parent.iterdir()
    )
    assert TemplateLayoutDetector(LayoutProfile.load(output_path)).detect(old_image)


def test_successful_recalibration_uses_immutable_unique_anchor_names(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "layouts" / "keju-default.json"
    old_image_path, _old_image = _write_calibration_image(tmp_path, "old", False)
    _run_calibration(old_image_path, output_path)
    old_anchor_paths = _profile_anchor_paths(output_path)
    old_anchor_bytes = {path: path.read_bytes() for path in old_anchor_paths}

    new_image_path, new_image = _write_calibration_image(tmp_path, "new", True)
    _run_calibration(new_image_path, output_path)
    new_anchor_paths = _profile_anchor_paths(output_path)

    assert set(old_anchor_paths).isdisjoint(new_anchor_paths)
    assert all(path.read_bytes() == old_anchor_bytes[path] for path in old_anchor_paths)
    assert all(path.exists() for path in new_anchor_paths)
    assert TemplateLayoutDetector(LayoutProfile.load(output_path)).detect(new_image)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert all("reference_rect" in anchor for anchor in payload["anchors"])


def test_successful_calibration_fsyncs_anchors_and_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path, _image = _write_calibration_image(tmp_path, "new", True)
    output_path = tmp_path / "layouts" / "keju-default.json"
    real_fsync = calibrate_module.os.fsync
    fsync_calls = 0

    def recording_fsync(file_descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        real_fsync(file_descriptor)

    monkeypatch.setattr(calibrate_module.os, "fsync", recording_fsync)

    _run_calibration(image_path, output_path)

    assert fsync_calls == 3


def test_calibration_anchor_names_are_unique_safe_ascii(tmp_path: Path) -> None:
    image_path, _image = _write_calibration_image(tmp_path, "new", True)
    output_path = tmp_path / "layouts" / "科举 布局.json"

    _run_calibration(image_path, output_path)

    assert all(
        re.fullmatch(r"anchor-[0-9a-f]{32}-[12]\.png", path.name)
        for path in _profile_anchor_paths(output_path)
    )


def _write_calibration_image(
    tmp_path: Path,
    name: str,
    inverted: bool,
) -> tuple[Path, np.ndarray]:
    pattern = np.array(
        [
            [0, 0, 255, 0, 0, 255, 0, 0],
            [0, 255, 255, 255, 255, 255, 255, 0],
            [255, 255, 0, 0, 0, 0, 255, 255],
            [0, 255, 0, 255, 255, 0, 255, 0],
            [0, 255, 0, 255, 255, 0, 255, 0],
            [255, 255, 0, 0, 0, 0, 255, 255],
            [0, 255, 255, 255, 255, 255, 255, 0],
            [0, 0, 255, 0, 0, 255, 0, 0],
        ],
        dtype=np.uint8,
    )
    if inverted:
        pattern = 255 - pattern
    image = np.full((100, 160, 3), 64 if inverted else 192, np.uint8)
    image[5:13, 5:13] = pattern[:, :, None]
    image[85:93, 140:148] = np.rot90(pattern).copy()[:, :, None]
    image_path = tmp_path / f"{name}.png"
    assert cv2.imwrite(str(image_path), image)
    return image_path, image


def _run_calibration(image_path: Path, output_path: Path) -> None:
    selections = iter(_CALIBRATION_SELECTIONS)
    assert calibrate(
        image_path,
        output_path,
        selector=lambda _image, _label: next(selections),
    )


def _profile_anchor_paths(output_path: Path) -> tuple[Path, ...]:
    profile = json.loads(output_path.read_text(encoding="utf-8"))
    return tuple(
        output_path.parent / anchor["template_path"]
        for anchor in profile["anchors"]
    )
