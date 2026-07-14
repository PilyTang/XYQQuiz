from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from xyq_quiz.capture.models import CapturedFrame
from xyq_quiz.diagnostics import (
    DiagnosticSnapshot,
    DiagnosticWriter,
    EnvironmentDiagnosticWriter,
)
from xyq_quiz.runtime.state import RuntimePhase, RuntimeSnapshot


def test_diagnostic_zip_contains_only_explicit_current_snapshot(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    log_path.write_text("old line\ncurrent line\n", encoding="utf-8")
    frame = CapturedFrame.create(
        7,
        9,
        np.full((12, 16, 3), 80, dtype=np.uint8),
    )
    crops = tuple(
        np.full((4, 5, 3), index * 20, dtype=np.uint8)
        for index in range(1, 6)
    )
    snapshot = DiagnosticSnapshot(
        frame=frame,
        runtime=RuntimeSnapshot(
            version=3,
            phase=RuntimePhase.ANSWERED,
            frame_id=7,
            question_text="问题",
        ),
        crops=crops,  # type: ignore[arg-type]
        config={"web": {"host": "127.0.0.1"}},
        metadata={"generation_id": "g1"},
    )

    path = DiagnosticWriter(tmp_path / "diagnostics", log_path=log_path).write(
        snapshot
    )

    with ZipFile(path) as archive:
        names = set(archive.namelist())
        assert names >= {
            "frame.jpg",
            "question.png",
            "option-a.png",
            "option-b.png",
            "option-c.png",
            "option-d.png",
            "state.json",
            "config.json",
            "metadata.json",
            "app.log",
        }
        assert not any(name.endswith(".mp4") for name in names)
        assert json.loads(archive.read("state.json"))["phase"] == "ANSWERED"
        assert json.loads(archive.read("metadata.json"))["generation_id"] == "g1"
        assert archive.read("app.log").decode("utf-8").splitlines()[-1] == "current line"


def test_config_redaction_is_recursive_for_credential_like_names(tmp_path: Path) -> None:
    snapshot = DiagnosticSnapshot(
        frame=CapturedFrame.create(
            1,
            1,
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        runtime=RuntimeSnapshot(),
        config={
            "token": "one",
            "nested": {
                "apiKey": "two",
                "client_secret": "three",
                "PASSWORD": "four",
                "authorization": "five",
                "cookieJar": "six",
                "safe": "visible",
            },
            "items": [{"refresh-token": "seven"}],
        },
        metadata={},
    )

    path = DiagnosticWriter(tmp_path).write(snapshot)

    with ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    assert config["token"] == "[REDACTED]"
    assert config["nested"]["apiKey"] == "[REDACTED]"
    assert config["nested"]["client_secret"] == "[REDACTED]"
    assert config["nested"]["PASSWORD"] == "[REDACTED]"
    assert config["nested"]["authorization"] == "[REDACTED]"
    assert config["nested"]["cookieJar"] == "[REDACTED]"
    assert config["nested"]["safe"] == "visible"
    assert config["items"][0]["refresh-token"] == "[REDACTED]"


def test_diagnostic_zip_accepts_three_option_crops(tmp_path: Path) -> None:
    snapshot = DiagnosticSnapshot(
        frame=CapturedFrame.create(
            1,
            1,
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        runtime=RuntimeSnapshot(),
        crops=tuple(np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(4)),
    )

    path = DiagnosticWriter(tmp_path).write(snapshot)

    with ZipFile(path) as archive:
        names = set(archive.namelist())
    assert {"question.png", "option-a.png", "option-b.png", "option-c.png"} <= names
    assert "option-d.png" not in names


def test_log_entry_is_bounded_to_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    log_path.write_bytes(b"discard-me\n" + b"x" * 32)
    snapshot = DiagnosticSnapshot(
        frame=CapturedFrame.create(
            1,
            1,
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        runtime=RuntimeSnapshot(),
        config={},
        metadata={},
    )

    path = DiagnosticWriter(tmp_path / "out", log_path=log_path, log_tail_bytes=32).write(
        snapshot
    )

    with ZipFile(path) as archive:
        assert archive.read("app.log") == b"x" * 32


def test_recognition_diagnostic_redacts_paths_and_log_secrets(tmp_path: Path) -> None:
    app_root = tmp_path / "portable-user-name"
    log_path = app_root / "logs" / "app.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        f"root={app_root}\ntoken=not-exported\nCookie: private-cookie\n",
        encoding="utf-8",
    )
    snapshot = DiagnosticSnapshot(
        frame=CapturedFrame.create(1, 1, np.zeros((2, 2, 3), dtype=np.uint8)),
        runtime=RuntimeSnapshot(),
    )

    path = DiagnosticWriter(
        app_root / "diagnostics",
        log_path=log_path,
        app_root=app_root,
    ).write(snapshot)

    with ZipFile(path) as archive:
        log = archive.read("app.log").decode("utf-8")
    assert str(app_root) not in log
    assert "not-exported" not in log
    assert "private-cookie" not in log
    assert "<APP_ROOT>" in log
    assert log.count("[REDACTED]") == 2


def test_writer_never_constructs_capture_session(tmp_path: Path, monkeypatch) -> None:
    import xyq_quiz.capture.wgc as wgc_module

    monkeypatch.setattr(
        wgc_module,
        "WGCCapture",
        lambda: (_ for _ in ()).throw(AssertionError("capture must not start")),
    )
    snapshot = DiagnosticSnapshot(
        frame=CapturedFrame.create(
            1,
            1,
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        runtime=RuntimeSnapshot(),
        config={},
        metadata={},
    )

    assert DiagnosticWriter(tmp_path).write(snapshot).is_file()


@pytest.mark.parametrize("crop_count", [1, 2, 3, 6])
def test_writer_rejects_partial_or_extra_crops_without_artifacts(
    tmp_path: Path,
    crop_count: int,
) -> None:
    output = tmp_path / "diagnostics"
    snapshot = DiagnosticSnapshot(
        frame=CapturedFrame.create(
            1,
            1,
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        runtime=RuntimeSnapshot(),
        crops=tuple(
            np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(crop_count)
        ),
        config={},
        metadata={},
    )

    with pytest.raises(ValueError, match="0, 4, or 5"):
        DiagnosticWriter(output).write(snapshot)

    assert not output.exists()


def test_environment_diagnostic_has_no_pixels_or_question_bank_and_redacts_paths(
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "portable-user-name"
    resource_root = app_root / "_internal"
    resource_root.mkdir(parents=True)
    log = app_root / "logs" / "app.log"
    log.parent.mkdir()
    log.write_text(f"root={app_root}\ntoken=not-exported\n", encoding="utf-8")
    writer = EnvironmentDiagnosticWriter(
        app_root / "diagnostics",
        app_root=app_root,
        resource_root=resource_root,
        log_path=log,
    )

    path = writer.write(
        {
            "data_dir": app_root / "data",
            "api_token": "secret",
        }
    )

    with ZipFile(path) as archive:
        names = set(archive.namelist())
        assert names == {
            "system.json",
            "config.json",
            "build-manifest-summary.json",
            "app.log",
        }
        assert not any(name.endswith((".jpg", ".png")) for name in names)
        config = json.loads(archive.read("config.json"))
        assert config["api_token"] == "[REDACTED]"
        assert config["data_dir"] == "<APP_ROOT>\\data"
        exported_log = archive.read("app.log").decode("utf-8")
        assert str(app_root) not in exported_log
        assert "not-exported" not in exported_log
        assert "token=[REDACTED]" in exported_log
