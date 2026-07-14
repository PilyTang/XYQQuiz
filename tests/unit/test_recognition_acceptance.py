from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from xyq_quiz.acceptance.fixtures import (
    FixtureKind,
    ManifestError,
    load_manifest,
)
from xyq_quiz.tools.import_fixture import import_fixture


def _write_image(path: Path, width: int = 32, height: int = 24) -> None:
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "file": "keju-example.png",
        "kind": "positive",
        "expected_source_id": "1001",
        "expected_option_index": 2,
        "window_size": [1292, 1023],
        "dpi": [96, 96],
        "provenance": "web",
        "human_verified": True,
        "sha256": "a" * 64,
    }
    case.update(overrides)
    return case


def _write_manifest(path: Path, cases: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "cases": cases}),
        encoding="utf-8",
    )


def test_manifest_loads_strict_verified_positive_and_negative(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            _case(),
            _case(
                file="non-keju-chat.jpg",
                kind="negative",
                expected_source_id=None,
                expected_option_index=None,
            ),
        ],
    )

    loaded = load_manifest(manifest, require_assets=False)

    assert loaded.cases[0].kind is FixtureKind.POSITIVE
    assert loaded.cases[0].window_size == (1292, 1023)
    assert loaded.cases[1].expected_option_index is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unexpected": 1}, "unknown fields"),
        ({"file": "../keju.png"}, "plain file name"),
        ({"expected_option_index": 4}, "0 through 3"),
        ({"human_verified": True, "expected_source_id": None}, "source id"),
        ({"kind": "negative", "file": "keju-no.png"}, "non-keju-"),
    ],
)
def test_manifest_rejects_invalid_or_ambiguous_cases(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [_case(**change)])

    with pytest.raises(ManifestError, match=message):
        load_manifest(manifest, require_assets=False)


def test_manifest_requires_readable_asset_with_matching_size_and_sha(tmp_path: Path) -> None:
    image = tmp_path / "keju-example.png"
    _write_image(image)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [_case(window_size=[99, 99])])

    with pytest.raises(ManifestError, match="window_size"):
        load_manifest(manifest, require_assets=True)


def test_import_fixture_copies_valid_image_and_leaves_answers_for_human(
    tmp_path: Path,
) -> None:
    source = tmp_path / "downloaded.jpg"
    _write_image(source, 40, 30)
    destination = tmp_path / "real-fixtures"

    case = import_fixture(
        source,
        destination,
        kind=FixtureKind.POSITIVE,
        provenance="web",
        dpi=(120, 120),
        filename="keju-web-001.jpg",
    )

    assert (destination / "keju-web-001.jpg").is_file()
    assert case.window_size == (40, 30)
    assert case.sha256 and len(case.sha256) == 64
    assert case.expected_source_id is None
    assert case.expected_option_index is None
    assert case.human_verified is False
    draft = load_manifest(destination / "manifest.json", require_assets=True)
    assert draft.cases == (case,)


def test_import_fixture_rejects_unreadable_input_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "broken.png"
    source.write_bytes(b"not an image")
    destination = tmp_path / "real-fixtures"

    with pytest.raises(ValueError, match="readable PNG or JPEG"):
        import_fixture(
            source,
            destination,
            kind=FixtureKind.NEGATIVE,
            provenance="local_wgc",
            dpi=(96, 96),
            filename="non-keju-broken.png",
        )

    assert not destination.exists()


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_import_fixture_rejects_bmp_renamed_as_png_or_jpeg(
    tmp_path: Path,
    suffix: str,
) -> None:
    source = tmp_path / f"disguised{suffix}"
    ok, encoded = cv2.imencode(".bmp", np.zeros((8, 9, 3), dtype=np.uint8))
    assert ok
    source.write_bytes(encoded.tobytes())
    destination = tmp_path / "real-fixtures"

    with pytest.raises(ValueError, match="actual PNG or JPEG"):
        import_fixture(
            source,
            destination,
            kind=FixtureKind.POSITIVE,
            provenance="web",
            dpi=(96, 96),
            filename=f"keju-disguised{suffix}",
        )

    assert not destination.exists()


def test_manifest_rejects_bmp_content_even_when_extension_and_sha_match(
    tmp_path: Path,
) -> None:
    image = tmp_path / "keju-example.png"
    ok, encoded = cv2.imencode(".bmp", np.zeros((8, 9, 3), dtype=np.uint8))
    assert ok
    image.write_bytes(encoded.tobytes())
    import hashlib
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [_case(window_size=[9, 8], sha256=hashlib.sha256(image.read_bytes()).hexdigest())],
    )

    with pytest.raises(ManifestError, match="actual PNG or JPEG"):
        load_manifest(manifest, require_assets=True)


def test_import_prevalidates_merged_manifest_without_leaving_orphan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _write_image(source)
    destination = tmp_path / "real-fixtures"
    destination.mkdir()
    _write_manifest(destination / "manifest.json", [_case(human_verified=False, expected_source_id=None, expected_option_index=None)])

    with pytest.raises(ManifestError, match="unique"):
        import_fixture(
            source,
            destination,
            kind=FixtureKind.POSITIVE,
            provenance="web",
            dpi=(96, 96),
            filename="keju-example.png",
        )

    assert not (destination / "keju-example.png").exists()
