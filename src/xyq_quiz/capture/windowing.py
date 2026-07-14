from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import threading
from collections.abc import Iterable, Sequence

from xyq_quiz.capture.models import Rect, WindowTarget


DESKTOP_ENUMERATE = 0x0040
DESKTOP_QUERY = 0x0002
DESKTOP_READOBJECTS = 0x0001
DESKTOP_SWITCHDESKTOP = 0x0100
DESKTOP_WRITEOBJECTS = 0x0080
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_thread_desktop = threading.local()


def attach_thread_to_input_desktop() -> bool:
    if getattr(_thread_desktop, "attached_to_input", False):
        return True

    user32 = ctypes.windll.user32
    user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    user32.OpenInputDesktop.restype = wintypes.HANDLE
    access = (
        DESKTOP_ENUMERATE
        | DESKTOP_QUERY
        | DESKTOP_READOBJECTS
        | DESKTOP_WRITEOBJECTS
        | DESKTOP_SWITCHDESKTOP
    )
    input_desktop = user32.OpenInputDesktop(0, False, access)
    if not input_desktop:
        return False

    user32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
    user32.SetThreadDesktop.restype = wintypes.BOOL
    if not user32.SetThreadDesktop(input_desktop):
        user32.CloseDesktop(input_desktop)
        return False

    _thread_desktop.handle = input_desktop
    _thread_desktop.attached_to_input = True
    return True


def enumerate_windows() -> list[WindowTarget]:
    attach_thread_to_input_desktop()
    user32 = ctypes.windll.user32
    windows: list[WindowTarget] = []
    enum_proc_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True

        title_length = user32.GetWindowTextLengthW(hwnd)
        if title_length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, title_length + 1)

        native_rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(native_rect)):
            return True
        width = native_rect.right - native_rect.left
        height = native_rect.bottom - native_rect.top
        if width <= 0 or height <= 0:
            return True

        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))

        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, len(class_buffer))

        windows.append(
            WindowTarget(
                hwnd=int(hwnd),
                title=title_buffer.value,
                process_id=int(process_id.value),
                process_name=_query_process_name(int(process_id.value)),
                class_name=class_buffer.value,
                rect=Rect(
                    x=native_rect.left,
                    y=native_rect.top,
                    width=width,
                    height=height,
                ),
            )
        )
        return True

    callback_proc = enum_proc_type(callback)
    user32.EnumWindows.argtypes = [enum_proc_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.EnumWindows(callback_proc, 0)
    return windows


def select_window(
    windows: Iterable[WindowTarget],
    process_names: Sequence[str],
    class_names: Sequence[str],
    preferred_hwnd: int | None = None,
) -> WindowTarget | None:
    window_list = list(windows)
    if preferred_hwnd:
        exact = next((window for window in window_list if window.hwnd == preferred_hwnd), None)
        if exact is not None:
            return exact

    process_set = {normalize_process_name(value) for value in process_names}
    class_set = {value.casefold() for value in class_names}
    candidates = [
        window
        for window in window_list
        if normalize_process_name(window.process_name) in process_set
        and (not class_set or window.class_name.casefold() in class_set)
        and window.rect.width > 0
        and window.rect.height > 0
    ]
    return max(
        candidates,
        key=lambda window: window.rect.width * window.rect.height,
        default=None,
    )


def normalize_process_name(process_name: str) -> str:
    normalized = process_name.strip().casefold()
    if normalized.endswith(".exe"):
        normalized = normalized[:-4]
    return normalized


def _query_process_name(process_id: int) -> str:
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    process_handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        process_id,
    )
    if not process_handle:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(
            process_handle,
            0,
            buffer,
            ctypes.byref(size),
        ):
            return ""
        return Path(buffer.value).name
    finally:
        kernel32.CloseHandle(process_handle)
