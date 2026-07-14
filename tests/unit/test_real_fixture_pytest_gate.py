from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_explicit_empty_manifest_is_a_nonzero_pytest_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "cases": []}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/test_recognition_fixtures.py",
            "-q",
            "--recognition-manifest",
            str(manifest),
        ],
        cwd=Path(__file__).parents[2],
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode != 0
    assert "显式 manifest 中没有真实样本" in completed.stderr + completed.stdout
