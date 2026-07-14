from __future__ import annotations

from pathlib import Path

import pytest

from xyq_quiz.acceptance.fixtures import load_manifest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("real recognition fixtures")
    group.addoption(
        "--recognition-manifest",
        type=Path,
        help="显式指定真实科举截图 manifest；未指定时真实验收不会运行",
    )
    group.addoption(
        "--recognition-layout",
        type=Path,
        action="append",
        help="显式指定由真实截图校准的 layout profile",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "recognition_case" not in metafunc.fixturenames:
        return
    manifest_path = metafunc.config.getoption("--recognition-manifest")
    if manifest_path is None:
        metafunc.parametrize(
            "recognition_case",
            [pytest.param(None, marks=pytest.mark.skip(reason="等待真实截图：未显式传入 --recognition-manifest"), id="waiting-for-real-screenshots")],
        )
        return
    manifest = load_manifest(manifest_path, require_assets=True)
    if not manifest.cases:
        message = "显式 manifest 中没有真实样本；Task 10B 验收未执行"
        terminal = metafunc.config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write_line(f"ERROR: {message}", red=True)
        pytest.exit(
            message,
            returncode=4,
        )
    metafunc.parametrize("recognition_case", manifest.cases, ids=lambda case: case.file)
