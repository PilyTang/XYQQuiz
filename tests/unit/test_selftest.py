from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xyq_quiz.runtime.paths import RuntimePaths
from xyq_quiz.selftest import run_self_test, verify_build_manifest, write_version_report


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths.discover(
        executable=tmp_path / "XYQQuiz.exe",
        bundle_root=tmp_path / "_internal",
        frozen=True,
    )


def test_self_test_writes_machine_and_human_reports_with_exit_boundary(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    checks = (
        ("pass", lambda: "ok"),
        ("fail", lambda: (_ for _ in ()).throw(RuntimeError("broken"))),
    )

    report, json_path, html_path = run_self_test(
        paths,
        config_path=config,
        report_dir=tmp_path / "reports",
        headless=True,
        checks=checks,
    )

    assert report.ok is False
    assert [check.status for check in report.checks] == ["PASS", "FAIL"]
    assert json.loads(json_path.read_text("utf-8"))["ok"] is False
    assert "XYQQuiz 自检：失败" in html_path.read_text("utf-8")


def test_manifest_verifier_checks_hash_and_rejects_extra_files(tmp_path: Path) -> None:
    internal = tmp_path / "_internal"
    internal.mkdir()
    payload = b"runtime"
    (internal / "python.dll").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "app_version": "0.1.0",
        "git_commit": "0" * 40,
        "built_at": "2026-07-14T00:00:00+00:00",
        "build_system": "test",
        "target": "Windows x64",
        "verified_platforms": ["Windows 11 x64"],
        "signed": False,
        "dependencies": {},
        "question_bank": {
            "generation_id": "g1",
            "source_url": "https://example.invalid/questions",
            "retrieved_at": "2026-07-14T00:00:00+00:00",
            "record_count": 1,
            "sha256": "0" * 64,
        },
        "files": [
            {
                "path": "python.dll",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    (internal / "build-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert verify_build_manifest(internal) == "已校验 1 个不可变文件"
    (internal / "stale.dll").write_bytes(b"stale")
    try:
        verify_build_manifest(internal)
    except ValueError as error:
        assert "未声明文件" in str(error)
    else:
        raise AssertionError("unlisted file must fail verification")


def test_version_report_is_json_and_does_not_need_console(tmp_path: Path) -> None:
    path = write_version_report(_paths(tmp_path), tmp_path / "reports")
    payload = json.loads(path.read_text("utf-8"))

    assert path.name == "version.json"
    assert payload["app_id"] == "xyq-quiz"
    assert payload["target"].startswith("Windows 10 1903+")
