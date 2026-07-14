from __future__ import annotations

from xyq_quiz.acceptance.benchmark import main, run_synthetic_replay


def test_synthetic_latest_only_replay_and_clear_gate() -> None:
    report = run_synthetic_replay(iterations=200)

    assert report.iterations == 200
    assert report.processed_frames == 200
    assert report.published_frames > report.delivered_frames
    assert report.skipped_frames >= 2
    assert report.queued_frame_count == 0
    assert report.clear_ms < 100
    assert all(value > 0 for value in report.p50_ms.values())
    assert report.targets_met is None
    assert "不代表真实 OCR/WGC 性能" in report.notice


if __name__ == "__main__":
    raise SystemExit(main())
