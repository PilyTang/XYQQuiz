from pathlib import Path, PurePosixPath
import runpy


ROOT = Path(__file__).parents[2]
AUDIT = runpy.run_path(str(ROOT / "scripts" / "check_public_tree.py"))
audit_paths = AUDIT["audit_paths"]
audit_project_license = AUDIT["audit_project_license"]


def test_public_tree_audit_rejects_private_fixture_and_personal_path(
    tmp_path: Path,
) -> None:
    private = PurePosixPath("tests/fixtures/recognition/local.png")
    source = PurePosixPath("src/example.py")
    (tmp_path / private.parent).mkdir(parents=True)
    (tmp_path / source.parent).mkdir(parents=True)
    (tmp_path / private).write_bytes(b"image")
    (tmp_path / source).write_text(
        'path = "' + r"C:\Users" + r'\someone\project"',
        encoding="utf-8",
    )

    findings = audit_paths(tmp_path, (private, source))

    assert any("private recognition fixture" in item for item in findings)
    assert any("personal Windows user path" in item for item in findings)


def test_public_tree_audit_accepts_placeholders_and_source_text(tmp_path: Path) -> None:
    source = PurePosixPath("README.md")
    (tmp_path / source).write_text(
        r"Use C:\Users\<name> and <USERPROFILE> placeholders.",
        encoding="utf-8",
    )

    assert audit_paths(tmp_path, (source,)) == []


def test_public_tree_audit_requires_polyform_noncommercial_license(
    tmp_path: Path,
) -> None:
    (tmp_path / "LICENSE").write_text(
        "# PolyForm Noncommercial License 1.0.0\n\n"
        "<https://polyformproject.org/licenses/noncommercial/1.0.0>\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nlicense = "PolyForm-Noncommercial-1.0.0"\n',
        encoding="utf-8",
    )

    assert audit_project_license(tmp_path) == []


def test_public_tree_audit_rejects_missing_or_wrong_project_license(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nlicense = "LicenseRef-Wrong"\n',
        encoding="utf-8",
    )

    findings = audit_project_license(tmp_path)

    assert "project LICENSE is missing" in findings
    assert any("PolyForm-Noncommercial-1.0.0" in item for item in findings)
