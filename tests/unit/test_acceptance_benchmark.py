from __future__ import annotations

import json
from pathlib import Path
import sys

from xyq_quiz.acceptance.benchmark import (
    FRAMEWORK_ONLY_NOTICE,
    STATIC_FIXTURE_NOTICE,
    StaticBenchmarkReport,
    main,
    run_synthetic_replay,
    _process_rss_mb,
)


def test_synthetic_replay_is_latest_only_and_reports_all_percentiles() -> None:
    report = run_synthetic_replay(iterations=200)

    assert report.iterations == 200
    assert report.published_frames > report.processed_frames == 200
    assert report.delivered_frames < report.published_frames
    assert report.skipped_frames == report.published_frames - report.delivered_frames
    assert report.queued_frame_count == 0
    assert report.clear_ms < 100
    assert report.clear_ms > 0
    assert report.measurement_source == "perf_counter_ns at production-path event boundaries"
    assert set(report.p50_ms) == {
        "capture_to_layout_ms",
        "ocr_ms",
        "match_ms",
        "state_publish_ms",
        "total_ms",
    }
    assert report.notice == FRAMEWORK_ONLY_NOTICE
    assert "不代表真实 OCR/WGC 性能" in report.notice


def test_benchmark_cli_writes_framework_report_without_claiming_live_target(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "benchmark.json"

    assert main(["--iterations", "200", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "synthetic"
    assert payload["targets_met"] is None
    assert "不代表真实 OCR/WGC 性能" in capsys.readouterr().out


def test_live_benchmark_without_real_manifest_fails_and_prints_no_pass(capsys) -> None:
    exit_code = main(["--live", "--iterations", "200"])

    output = capsys.readouterr()
    assert exit_code == 2
    assert "等待真实截图" in output.err
    assert "达标" not in output.out
    assert "P50" not in output.out


def test_static_benchmark_cli_reports_fixture_mode_without_claiming_wgc(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    report = StaticBenchmarkReport(
        mode="static_fixture",
        fixture_count=5,
        rounds=2,
        measured_runs=10,
        cold_start_ms=123.0,
        end_to_end_p50_ms=700.0,
        end_to_end_p95_ms=900.0,
        pipeline_component_p50_ms={"total_ms": 100.0},
        pipeline_component_p95_ms={"total_ms": 120.0},
        rss_delta_mb=80.0,
        rec_only_success_count=10,
        fallback_count=0,
        line_count_distribution={1: 40, 2: 10},
        targets_met=False,
        measurement_source=(
            "LatestFrameHub -> RecognitionCoordinator -> RecognitionPipeline "
            "-> RuntimeStore"
        ),
        notice=STATIC_FIXTURE_NOTICE,
    )
    calls: list[tuple[Path, tuple[Path, ...], int]] = []
    monkeypatch.setattr(
        "xyq_quiz.acceptance.benchmark.run_static_fixture_benchmark",
        lambda manifest, layouts, rounds: calls.append((manifest, layouts, rounds)) or report,
    )
    output = tmp_path / "static.json"

    assert main([
        "--static",
        "--manifest", str(tmp_path / "manifest.json"),
        "--layout", str(tmp_path / "layout.json"),
        "--rounds", "2",
        "--output", str(output),
    ]) == 0

    assert calls == [
        (tmp_path / "manifest.json", (tmp_path / "layout.json",), 2)
    ]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["targets_met"] is False
    assert payload["rec_only_success_count"] == 10
    assert payload["fallback_count"] == 0
    assert payload["line_count_distribution"] == {"1": 40, "2": 10}
    assert payload["end_to_end_p95_ms"] == 900.0
    assert payload["pipeline_component_p95_ms"]["total_ms"] == 120.0
    assert "不是 WGC/FPS" in capsys.readouterr().out


def test_process_rss_is_reported_on_windows() -> None:
    rss = _process_rss_mb()
    if sys.platform == "win32":
        assert rss is not None and rss > 0
