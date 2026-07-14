from __future__ import annotations

from xyq_quiz.capture.models import Rect, WindowTarget
from xyq_quiz.capture.windowing import select_window


def target(
    hwnd: int,
    process_name: str,
    class_name: str,
    width: int,
    height: int,
) -> WindowTarget:
    return WindowTarget(
        hwnd=hwnd,
        title=f"Window {hwnd}",
        process_id=hwnd + 1000,
        process_name=process_name,
        class_name=class_name,
        rect=Rect(x=10, y=20, width=width, height=height),
    )


def test_select_window_prefers_handle_then_process_class_and_area() -> None:
    small = target(1, "mhtab.exe", "MHXYMainFrame", 640, 480)
    large = target(2, "mhtab.exe", "MHXYMainFrame", 1292, 1023)

    assert select_window([small, large], ["mhtab.exe"], ["MHXYMainFrame"], 1) == small
    assert select_window([small, large], ["mhtab.exe"], ["MHXYMainFrame"], None) == large


def test_select_window_rejects_wrong_class() -> None:
    chat = target(3, "mhtab.exe", "Chrome_WidgetWin_1", 1600, 1000)

    assert select_window([chat], ["mhtab.exe"], ["MHXYMainFrame"], None) is None


def test_select_window_normalizes_exe_suffix_and_process_case() -> None:
    game = target(4, "MHTAB.EXE", "MHXYMainFrame", 1292, 1023)

    assert select_window([game], ["mhtab"], [], None) == game
