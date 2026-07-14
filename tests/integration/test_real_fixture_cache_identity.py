from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from xyq_quiz.acceptance.fixtures import FixtureKind, decode_png_or_jpeg, load_manifest
from xyq_quiz.recognition.layout import LayoutProfile, build_layout_detector
from xyq_quiz.runtime.coordinator import _quiz_cache_identity, _same_quiz_identity


def test_five_real_positive_q98_images_never_authorize_cached_answer_reuse() -> None:
    root = Path(__file__).parents[2]
    fixtures = root / "tests" / "fixtures" / "recognition"
    manifest_path = fixtures / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("private real-image corpus is not included in the public repository")
    manifest = load_manifest(manifest_path, require_assets=True)
    detector = build_layout_detector(
        (
            LayoutProfile.load(root / "data" / "layouts" / "keju-default.json"),
            LayoutProfile.load(root / "data" / "layouts" / "keju-picture.json"),
        )
    )
    checked = 0
    for case in manifest.cases:
        if case.kind is not FixtureKind.POSITIVE:
            continue
        original = decode_png_or_jpeg(fixtures / case.file)
        encoded, payload = cv2.imencode(
            ".jpg",
            original,
            [cv2.IMWRITE_JPEG_QUALITY, 98],
        )
        assert encoded
        compressed = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        original_layout = detector.detect(original)
        compressed_layout = detector.detect(compressed)
        assert original_layout is not None
        if compressed_layout is not None:
            assert not _same_quiz_identity(
                _quiz_cache_identity(original, original_layout),
                _quiz_cache_identity(compressed, compressed_layout),
            )
        checked += 1

    assert checked == 5
