from __future__ import annotations

import argparse
from pathlib import Path
import time

from xyq_quiz.capture.native import NativePreviewSession
from xyq_quiz.capture.video import LatestVideoHub
from xyq_quiz.capture.windowing import enumerate_windows, select_window
from xyq_quiz.config import AppConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-window native preview smoke test")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--x", type=int, default=60)
    parser.add_argument("--y", type=int, default=60)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--recognition-fps", type=int, default=0)
    args = parser.parse_args()

    config = AppConfig.load(Path("config.json"))
    target = select_window(
        enumerate_windows(),
        config.window.process_names,
        config.window.class_names,
    )
    if target is None:
        raise RuntimeError("configured game window was not found")

    fps_samples: list[float] = []
    session = NativePreviewSession(
        helper_path=Path("build/native/XYQPreviewHelper.exe"),
        target=target,
        adapter_id=-1,
        video_hub=LatestVideoHub(),
        preview_width=1024,
        preview_fps=30,
        recognition_fps=args.recognition_fps,
        mapping_capacity=3840 * 2160 * 4,
        native_window=True,
        on_preview_fps=fps_samples.append,
    )
    recognition_frames = 0
    frame_id = 0
    started = time.monotonic()
    try:
        session.start()
        session.update_preview_layout(
            owner_hwnd=0,
            x=args.x,
            y=args.y,
            width=args.width,
            height=args.height,
            scale=1.0,
            visible=True,
        )
        session.update_preview_overlay(
            (0.55, 0.55, 0.30, 0.16),
            score=87,
            level=2,
        )
        deadline = started + args.seconds
        while time.monotonic() < deadline:
            session.raise_if_failed()
            frame = session.latest(frame_id)
            if frame is not None:
                frame_id = frame.frame_id
                recognition_frames += 1
            time.sleep(0.01)
    finally:
        session.stop()

    elapsed = time.monotonic() - started
    average_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
    print(
        "SMOKE_RESULT "
        f"elapsed={elapsed:.2f} recognition_frames={recognition_frames} "
        f"fps_samples={len(fps_samples)} average_preview_fps={average_fps:.1f} "
        f"debug={','.join(session.debug_messages)}",
        flush=True,
    )
    recognition_ok = args.recognition_fps == 0 or recognition_frames > 0
    return 0 if recognition_ok and fps_samples else 2


if __name__ == "__main__":
    raise SystemExit(main())
