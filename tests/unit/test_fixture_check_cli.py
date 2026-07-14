from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from xyq_quiz.tools.check_fixtures import main


def test_fixture_check_requires_explicit_manifest(capsys) -> None:
    assert main([]) == 2
    assert "必须显式传入 --manifest" in capsys.readouterr().err


def test_fixture_check_empty_manifest_waits_for_real_screenshots(
    tmp_path: Path,
    capsys,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "cases": []}),
        encoding="utf-8",
    )

    assert main(["--manifest", str(manifest)]) == 2
    output = capsys.readouterr()
    assert "等待真实截图" in output.err
    assert "通过" not in output.out


def test_fixture_check_rejects_unreadable_anchor_images(tmp_path: Path, capsys) -> None:
    fixture = tmp_path / "non-keju-chat.png"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert cv2.imwrite(str(fixture), image)
    import hashlib
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "file": fixture.name,
                        "kind": "negative",
                        "expected_source_id": None,
                        "expected_option_index": None,
                        "window_size": [10, 10],
                        "dpi": [96, 96],
                        "provenance": "web",
                        "human_verified": True,
                        "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    (anchors / "one.png").write_bytes(b"broken")
    (anchors / "two.png").write_bytes(b"broken")
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "reference_size": [10, 10],
                "question_rect": [0, 0, 0.2, 0.2],
                "option_rects": [[0, 0.2, 0.2, 0.2]] * 4,
                "anchors": [
                    {"search_rect": [0, 0, 0.5, 0.5], "template_path": "anchors/one.png", "threshold": 0.8},
                    {"search_rect": [0.5, 0, 0.5, 0.5], "template_path": "anchors/two.png", "threshold": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--manifest", str(manifest), "--layout", str(layout)]) == 2
    assert "anchor 不可读" in capsys.readouterr().err


def test_fixture_check_accepts_repeated_layout_profiles(
    tmp_path: Path,
    capsys,
) -> None:
    fixture = tmp_path / "non-keju-chat.png"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert cv2.imwrite(str(fixture), image)
    import hashlib
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{
                    "file": fixture.name,
                    "kind": "negative",
                    "expected_source_id": None,
                    "expected_option_index": None,
                    "window_size": [10, 10],
                    "dpi": [96, 96],
                    "provenance": "web",
                    "human_verified": True,
                    "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                }],
            }
        ),
        encoding="utf-8",
    )
    layouts: list[Path] = []
    for name in ("normal", "alternate"):
        anchors = tmp_path / f"{name}-anchors"
        anchors.mkdir()
        for index in (1, 2):
            assert cv2.imwrite(
                str(anchors / f"{index}.png"),
                np.arange(64, dtype=np.uint8).reshape(8, 8),
            )
        layout = tmp_path / f"{name}.json"
        layout.write_text(
            json.dumps({
                "reference_size": [10, 10],
                "question_rect": [0, 0, 0.2, 0.2],
                "option_rects": [[0, 0.2, 0.2, 0.2]] * 4,
                "anchors": [
                    {"search_rect": [0, 0, 0.5, 0.5], "template_path": f"{name}-anchors/1.png", "threshold": 0.8},
                    {"search_rect": [0.5, 0, 0.5, 0.5], "template_path": f"{name}-anchors/2.png", "threshold": 0.8},
                ],
            }),
            encoding="utf-8",
        )
        layouts.append(layout)

    assert main([
        "--manifest", str(manifest),
        "--layout", str(layouts[0]),
        "--layout", str(layouts[1]),
    ]) == 0
    assert "2 套布局" in capsys.readouterr().out
