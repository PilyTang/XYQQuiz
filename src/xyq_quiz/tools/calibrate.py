from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from xyq_quiz.recognition.layout import validate_anchor_reference_geometry
from xyq_quiz.recognition.models import NormalizedRect


Selection = tuple[int, int, int, int]
Selector = Callable[[NDArray[np.uint8], str], Selection | None]
_LABELS = (
    "Question ROI",
    "Option A",
    "Option B",
    "Option C",
    "Option D",
    "Anchor 1",
    "Anchor 2",
)


def calibrate(
    image_path: Path,
    output_path: Path,
    *,
    selector: Selector | None = None,
) -> bool:
    image_file = Path(image_path)
    output_file = Path(output_path)
    image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"cannot load calibration image: {image_file}")

    choose = selector or select_rectangle
    selections: list[Selection] = []
    for label in _LABELS:
        selection = choose(image, label)
        if selection is None:
            return False
        selections.append(_validate_selection(selection, image.shape[1], image.shape[0]))

    validate_anchor_reference_geometry(
        tuple(
            NormalizedRect(
                *_normalize(
                    selection,
                    image.shape[1],
                    image.shape[0],
                )
            )
            for selection in selections[5:]
        ),
        image.shape[1],
        image.shape[0],
    )

    _write_calibration(image, selections, output_file)
    return True


def select_rectangle(
    image: NDArray[np.uint8],
    label: str,
) -> Selection | None:
    window_name = f"XYQ Quiz Calibration - {label}"
    drag_start: tuple[int, int] | None = None
    drag_end: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        nonlocal drag_start, drag_end
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            drag_end = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            display = image.copy()
            if drag_start is not None and drag_end is not None:
                cv2.rectangle(display, drag_start, drag_end, (0, 255, 0), 2)
            cv2.putText(
                display,
                f"{label}: drag, Enter confirms, Esc cancels",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                return None
            if key in (10, 13) and drag_start is not None and drag_end is not None:
                selection = _selection_from_points(drag_start, drag_end)
                if selection[2] > 0 and selection[3] > 0:
                    return selection
    finally:
        cv2.destroyWindow(window_name)


def _write_calibration(
    image: NDArray[np.uint8],
    selections: Sequence[Selection],
    output_path: Path,
) -> None:
    frame_height, frame_width = image.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{output_path.stem}-", dir=output_path.parent)
    )
    published_anchors: list[Path] = []
    profile_committed = False
    try:
        calibration_id = uuid.uuid4().hex
        anchor_names = (
            f"anchor-{calibration_id}-1.png",
            f"anchor-{calibration_id}-2.png",
        )
        for selection, anchor_name in zip(selections[5:], anchor_names, strict=True):
            x, y, width, height = selection
            encoded, png_bytes = cv2.imencode(
                ".png",
                image[y : y + height, x : x + width],
            )
            if not encoded:
                raise OSError(f"failed to write anchor template: {anchor_name}")
            _write_durable(temporary_directory / anchor_name, png_bytes.tobytes())

        payload = {
            "reference_size": [frame_width, frame_height],
            "question_rect": _normalize(selections[0], frame_width, frame_height),
            "option_rects": [
                _normalize(selection, frame_width, frame_height)
                for selection in selections[1:5]
            ],
            "anchors": [
                {
                    "reference_rect": _normalize(
                        selection,
                        frame_width,
                        frame_height,
                    ),
                    "search_rect": _normalize(
                        _expand(selection, frame_width, frame_height),
                        frame_width,
                        frame_height,
                    ),
                    "template_path": f"anchors/{anchor_name}",
                    "threshold": 0.85,
                    "scale_range": [0.9, 1.1],
                }
                for selection, anchor_name in zip(
                    selections[5:],
                    anchor_names,
                    strict=True,
                )
            ],
        }
        temporary_profile = temporary_directory / output_path.name
        _write_durable(
            temporary_profile,
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            ),
        )

        anchors_directory = output_path.parent / "anchors"
        anchors_directory.mkdir(exist_ok=True)
        for anchor_name in anchor_names:
            final_anchor = anchors_directory / anchor_name
            if final_anchor.exists():
                raise FileExistsError(f"anchor already exists: {final_anchor}")
            os.replace(
                temporary_directory / anchor_name,
                final_anchor,
            )
            published_anchors.append(final_anchor)
        os.replace(temporary_profile, output_path)
        profile_committed = True
    finally:
        if not profile_committed:
            for published_anchor in published_anchors:
                published_anchor.unlink(missing_ok=True)
        shutil.rmtree(temporary_directory, ignore_errors=True)


def _write_durable(path: Path, contents: bytes) -> None:
    with path.open("wb") as file:
        file.write(contents)
        file.flush()
        os.fsync(file.fileno())


def _validate_selection(
    selection: Selection,
    frame_width: int,
    frame_height: int,
) -> Selection:
    x, y, width, height = selection
    left = min(frame_width - 1, max(0, int(x)))
    top = min(frame_height - 1, max(0, int(y)))
    right = min(frame_width, max(left + 1, int(x + width)))
    bottom = min(frame_height, max(top + 1, int(y + height)))
    return left, top, right - left, bottom - top


def _selection_from_points(
    start: tuple[int, int],
    end: tuple[int, int],
) -> Selection:
    left, right = sorted((start[0], end[0]))
    top, bottom = sorted((start[1], end[1]))
    return left, top, right - left, bottom - top


def _normalize(
    selection: Selection,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    x, y, width, height = selection
    return [
        x / frame_width,
        y / frame_height,
        width / frame_width,
        height / frame_height,
    ]


def _expand(
    selection: Selection,
    frame_width: int,
    frame_height: int,
) -> Selection:
    x, y, width, height = selection
    left = max(0, x - width)
    top = max(0, y - height)
    right = min(frame_width, x + width * 2)
    bottom = min(frame_height, y + height * 2)
    return left, top, right - left, bottom - top


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate fixed XYQ quiz layout")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    completed = calibrate(arguments.image, arguments.output)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
