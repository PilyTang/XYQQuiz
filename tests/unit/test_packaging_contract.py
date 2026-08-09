from pathlib import Path
import re
import tomllib


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


def test_desktop_shell_dependencies_are_pinned_to_pywebview_6() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    release_input = (ROOT / "requirements-release.in").read_text(encoding="utf-8")

    assert "pywebview>=6.2.1,<7" in project["project"]["dependencies"]
    assert "pywebview==6.2.1" in release_input.splitlines()


def test_release_uses_one_directml_runtime_for_cpu_and_gpu_ocr() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    release_input = (ROOT / "requirements-release.in").read_text(encoding="utf-8")

    assert "onnxruntime-directml>=1.20,<2" in dependencies
    assert "onnxruntime>=1.20,<2" not in dependencies
    assert "onnxruntime-directml==1.24.4" in release_input.splitlines()
    assert not any(line.startswith("onnxruntime==") for line in release_input.splitlines())


def test_pyinstaller_bundles_only_the_windows_webview2_backend() -> None:
    spec = (ROOT / "packaging" / "XYQQuiz.spec").read_text(encoding="utf-8")

    assert '"webview.platforms.winforms"' in spec
    assert '"webview.platforms.edgechromium"' in spec
    assert '"webview.platforms.mshtml"' in spec
    assert '"webview.platforms.cef"' in spec
    assert '"webview.platforms.gtk"' in spec
    assert '"webview.platforms.qt"' in spec
    assert '"webview.platforms.cocoa"' in spec
    assert '"webview.platforms.android"' in spec
    assert '"PyQt5"' in spec
    assert '"PyQt6"' in spec
    assert '"PySide2"' in spec
    assert '"PySide6"' in spec
    assert '"cefpython3"' in spec
    assert '"gi"' in spec
    assert '"jnius"' in spec
    assert not re.search(r"collect_all\(\s*['\"]webview['\"]", spec)
    assert '"webview/lib/runtimes/win-arm64/"' not in spec
    assert '"webview/lib/runtimes/win-x86/"' not in spec
    assert "Keep all" in spec
    assert "WebView2 loader directories" in spec
    assert '"webview/lib/pywebview-android.jar"' in spec
    assert '"webview/lib/webbrowserinterop.x64.dll"' in spec
    assert "keep_target_webview_payload" in spec


def test_release_bundles_the_native_preview_helper() -> None:
    spec = (ROOT / "packaging" / "XYQQuiz.spec").read_text(encoding="utf-8")
    build_script = (ROOT / "scripts" / "build-release.ps1").read_text(
        encoding="utf-8"
    )

    assert '"native" / "XYQPreviewHelper.exe"' in spec
    assert "build-native-helper.ps1" in build_script


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
    assert "Select-Object -First 1" in script
    assert "check_public_tree.py" in script
    assert "Assert-NoPrivateRuntimePayload" in script
    assert "Assert-NoPrivateZipEntries" in script
    assert '"user-data"' in script
    assert '"diagnostics"' in script
    assert '"questions.json"' in script


def test_release_version_metadata_is_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    package_init = (ROOT / "src" / "xyq_quiz" / "__init__.py").read_text(
        encoding="utf-8"
    )
    release_script = (ROOT / "scripts" / "build-release.ps1").read_text(
        encoding="utf-8"
    )
    version_info = (ROOT / "packaging" / "version_info.txt").read_text(
        encoding="utf-8"
    )

    assert version == "0.3.0"
    assert re.search(rf'^__version__ = "{re.escape(version)}"$', package_init, re.M)
    assert f'[string]$Version = "{version}"' in release_script
    assert "filevers=(0, 3, 0, 0)" in version_info
    assert "prodvers=(0, 3, 0, 0)" in version_info
    assert f"StringStruct('FileVersion', '{version}')" in version_info
    assert f"StringStruct('ProductVersion', '{version}')" in version_info


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
