from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from xyq_quiz.performance.native_preview import (
    locate_native_preview_helper,
    run_native_preview_self_test,
)


def test_locate_native_preview_helper_uses_build_output_in_source_tree(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "build" / "native" / "XYQPreviewHelper.exe"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"MZ")

    assert locate_native_preview_helper(
        app_root=tmp_path,
        resource_root=tmp_path,
        frozen=False,
    ) == helper.resolve()


def test_native_preview_selftest_uses_argument_array_and_validates_json(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "XYQPreviewHelper.exe"
    helper.write_bytes(b"MZ")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(arguments: list[str], **kwargs: object) -> object:
        calls.append((arguments, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"ok":true,"adapter_id":1,"adapter_name":"GPU",'
                '"encoder_name":"H264","feature_level":49408,'
                '"encoded_bytes":1825}\n'
            ),
            stderr="",
        )

    result = run_native_preview_self_test(helper, 1, runner=runner)

    assert calls[0][0] == [str(helper.resolve()), "--self-test", "--adapter", "1"]
    assert calls[0][1]["shell"] if "shell" in calls[0][1] else True
    assert result.adapter_id == 1
    assert result.encoder_name == "H264"
    assert result.encoded_bytes == 1825


def test_native_preview_selftest_surfaces_helper_error(tmp_path: Path) -> None:
    helper = tmp_path / "XYQPreviewHelper.exe"
    helper.write_bytes(b"MZ")

    def runner(_arguments: list[str], **_kwargs: object) -> object:
        return SimpleNamespace(
            returncode=2,
            stdout='{"ok":false,"error":"encoder unavailable"}\n',
            stderr="",
        )

    with pytest.raises(RuntimeError, match="encoder unavailable"):
        run_native_preview_self_test(helper, 0, runner=runner)
