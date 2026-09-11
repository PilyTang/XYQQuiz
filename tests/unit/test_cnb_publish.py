import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("publish_cnb", Path(__file__).parents[2] / "scripts/publish_cnb.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def test_prepare_validates_archive_version_and_hash(tmp_path):
    package = tmp_path / "XYQQuiz-v0.5.6-win10-win11-x64.zip"
    notes = tmp_path / "notes.md"
    notes.write_text("新版本", encoding="utf-8")
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("XYQQuiz/_internal/build-manifest.json", json.dumps({"app_version": "0.5.6"}))
    sha = Path(str(package) + ".sha256")
    sha.write_text(hashlib.sha256(package.read_bytes()).hexdigest() + "  " + package.name)
    assert publisher.prepare(package, notes)["version"] == "0.5.6"
    sha.write_text("0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        publisher.prepare(package, notes)


def test_token_parser_accepts_private_note_and_rejects_ambiguity(tmp_path):
    path = tmp_path / "private.txt"
    path.write_text("description:release\ngit username:cnb\ntoken:" + "x" * 27)
    assert publisher.read_token(path) == "x" * 27
    path.write_text("x" * 27 + "\n" + "y" * 27)
    with pytest.raises(ValueError):
        publisher.read_token(path)
