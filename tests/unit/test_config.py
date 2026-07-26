from pathlib import Path

import pytest
from pydantic import ValidationError

from xyq_quiz.config import AppConfig


def test_defaults_are_local_and_target_mhtab() -> None:
    cfg = AppConfig.load(None)
    assert cfg.web.host == "127.0.0.1"
    assert cfg.window.process_names == ["mhtab.exe"]
    assert cfg.window.class_names == ["MHXYMainFrame"]
    assert cfg.capture.preview_fps == 30
    assert cfg.capture.black_frame_count == 10
    assert cfg.match.question_score == 92
    assert cfg.match.question_gap == 5
    assert cfg.match.option_score == 90
    assert cfg.recognition.scan_fps == 15
    assert cfg.recognition.ocr_workers == 1


def test_load_overrides_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"web":{"port":8877},"capture":{"preview_fps":24},'
        '"recognition":{"scan_fps":12,"ocr_workers":1}}',
        encoding="utf-8",
    )
    cfg = AppConfig.load(path)
    assert cfg.web.port == 8877
    assert cfg.capture.preview_fps == 24
    assert cfg.recognition.scan_fps == 12
    assert cfg.recognition.ocr_workers == 1
    assert cfg.log_path == (tmp_path / "logs" / "app.log").resolve()


def test_legacy_config_without_scan_fps_keeps_compatible_default(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"capture":{"preview_fps":15},"recognition":{"ocr_workers":1}}',
        encoding="utf-8",
    )

    cfg = AppConfig.load(path)

    assert cfg.capture.preview_fps == 15
    assert cfg.recognition.scan_fps == 15


@pytest.mark.parametrize("scan_fps", [0, 61])
def test_scan_fps_is_bounded(scan_fps: int) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"recognition": {"scan_fps": scan_fps}})


def test_load_resolves_multiple_layout_profiles_and_keeps_legacy_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"layout_paths":["layouts/normal.json","layouts/picture.json"]}',
        encoding="utf-8",
    )

    cfg = AppConfig.load(path)

    assert cfg.effective_layout_paths == (
        (tmp_path / "layouts" / "normal.json").resolve(),
        (tmp_path / "layouts" / "picture.json").resolve(),
    )
    assert AppConfig().effective_layout_paths == (Path("data/layouts/keju-default.json"),)


def test_legacy_multi_worker_config_has_clear_migration_error() -> None:
    with pytest.raises(
        ValidationError,
        match="ocr_workers 现在只支持 1.*请改为 1 或删除该配置项",
    ):
        AppConfig.model_validate({"recognition": {"ocr_workers": 5}})


def test_readme_documents_fixed_single_worker() -> None:
    readme = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")

    assert "ocr_workers` 现在只允许 `1`" in readme
