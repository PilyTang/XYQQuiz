from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_pyinstaller_contract_is_onedir_windowed_and_as_invoker() -> None:
    spec = (ROOT / "packaging" / "XYQQuiz.spec").read_text(encoding="utf-8")
    manifest = (ROOT / "packaging" / "XYQQuiz.manifest").read_text(
        encoding="utf-8"
    )

    assert 'console=False' in spec
    assert 'contents_directory="_internal"' in spec
    assert 'name="XYQQuiz"' in spec
    assert 'level="asInvoker"' in manifest
    assert "requireAdministrator" not in manifest


def test_release_script_generates_manifest_zip_and_sha256() -> None:
    script = (ROOT / "scripts" / "build-release.ps1").read_text(encoding="utf-8")

    assert "generate_build_manifest.py" in script
    assert "Compress-Archive" in script
    assert "Get-FileHash" in script
    assert "Refusing to reset path outside project root" in script
    assert "config.example.json" in script
    assert 'Test-Path -LiteralPath $LicensePath' in script
    assert "PolyForm Noncommercial License 1.0.0" in script
    assert 'Join-Path $PackageRoot "LICENSE.txt"' in script
    assert "full 40-character Git commit SHA" in script
    assert "clean Git worktree" in script
    assert "Get-Command $Python -CommandType Application" in script


def test_pyinstaller_excludes_upstream_sbom_with_build_machine_paths() -> None:
    script = (ROOT / "scripts" / "build-release.ps1").read_text(encoding="utf-8")

    assert '"windows_capture-*.dist-info"' in script
    assert 'Join-Path $MetadataDirectory.FullName "sboms"' in script
    assert "Refusing to remove upstream SBOM outside package root" in script


def test_one_click_self_test_waits_for_gui_exe_and_preserves_exit_code() -> None:
    script = (ROOT / "packaging" / "一键自检.cmd").read_text(encoding="utf-8")

    assert 'start "" /wait' in script
    assert "--self-test --report-dir" in script
    assert "exit /b %RESULT%" in script
