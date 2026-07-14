from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import pytest

from xyq_quiz.acceptance.fixtures import FixtureKind, RecognitionFixture
from xyq_quiz.capture.models import CapturedFrame
from xyq_quiz.config import AppConfig
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.updater import load_current_generation
from xyq_quiz.recognition.layout import LayoutProfile, build_layout_detector
from xyq_quiz.recognition.ocr import OCRDiagnostics, RapidOCREngine
from xyq_quiz.recognition.pipeline import RecognitionPipeline


@dataclass(frozen=True)
class RealRecognizer:
    pipeline: RecognitionPipeline
    ocr: RapidOCREngine
    baseline: OCRDiagnostics


@pytest.fixture(scope="module")
def real_recognizer(request: pytest.FixtureRequest):
    layout_paths = request.config.getoption("--recognition-layout")
    if not layout_paths:
        if request.config.getoption("--recognition-manifest") is None:
            pytest.skip("等待真实截图：未显式传入 --recognition-manifest")
        pytest.fail("等待真实校准：显式真实 fixture 验收还必须传入 --recognition-layout")
    config = AppConfig()
    current = load_current_generation(config.data_dir)
    matcher = QuestionMatcher(
        current.question_bank,
        config.match.question_score,
        config.match.question_gap,
        config.match.option_score,
    )
    ocr = RapidOCREngine()
    pipeline = RecognitionPipeline(
        build_layout_detector(tuple(LayoutProfile.load(Path(path)) for path in layout_paths)),
        ocr,
        matcher,
    )
    pipeline.warm_up()
    yield RealRecognizer(pipeline, ocr, ocr.diagnostics_snapshot())
    pipeline.close()


def test_real_recognition_fixture(
    recognition_case: RecognitionFixture | None,
    real_recognizer: RealRecognizer,
    request: pytest.FixtureRequest,
) -> None:
    assert recognition_case is not None
    assert recognition_case.human_verified, "真实 fixture 必须先人工核验 expected 字段"
    manifest_path = request.config.getoption("--recognition-manifest")
    image = cv2.imread(str(Path(manifest_path).parent / recognition_case.file), cv2.IMREAD_COLOR)
    assert image is not None and image.size > 0
    for round_index in range(10):
        result = real_recognizer.pipeline.recognize(
            CapturedFrame.create(round_index + 1, time.monotonic_ns(), image),
            generation_id=1,
        )
        if recognition_case.kind is FixtureKind.NEGATIVE:
            assert result.overlay_rect is None
            assert result.high_confidence is False
            continue
        assert result.high_confidence is True
        assert result.source_id == recognition_case.expected_source_id
        assert result.option_index == recognition_case.expected_option_index


def test_real_recognition_diagnostics_prove_all_positive_rois_used_rec_only(
    real_recognizer: RealRecognizer,
) -> None:
    current = real_recognizer.ocr.diagnostics_snapshot()
    baseline = real_recognizer.baseline
    line_distribution = {
        line_count: count - baseline.line_count_distribution.get(line_count, 0)
        for line_count, count in current.line_count_distribution.items()
        if count - baseline.line_count_distribution.get(line_count, 0) > 0
    }

    assert current.rec_only_success_count - baseline.rec_only_success_count == 250
    assert current.fallback_count - baseline.fallback_count == 0
    assert sum(line_distribution.values()) == 250
    assert all(1 <= line_count <= 12 for line_count in line_distribution)
    assert any(line_count > 1 for line_count in line_distribution)
