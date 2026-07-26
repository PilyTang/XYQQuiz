from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
import importlib
import json
import logging
import socket
import threading
import time
from typing import Any
from urllib.request import urlopen

from xyq_quiz.web.security import APP_ID


_LOGGER = logging.getLogger(__name__)


class DesktopUnavailable(RuntimeError):
    """The native pywebview/WebView2 UI could not be used.

    ``server_started`` identifies the fallback boundary.  When it is false the
    caller may start its normal browser runner with the untouched Uvicorn
    server.  When it is true this controller has already stopped and joined
    that server; callers must not invoke ``server.run`` on the same instance a
    second time.  Supplying ``fallback_opener`` to :meth:`run` avoids that
    boundary by continuing with the already-running server.
    """

    def __init__(self, message: str, *, server_started: bool) -> None:
        super().__init__(message)
        self.server_started = server_started


class DesktopServerError(RuntimeError):
    """The background Uvicorn server failed independently of pywebview."""


class DesktopRunMode(str, Enum):
    DESKTOP = "desktop"
    BROWSER_FALLBACK = "browser_fallback"


class WebViewDesktopController:
    """Own one pywebview window and one background Uvicorn server.

    The controller is intentionally single-use.  Construct it before the
    single-instance activation server so :meth:`request_focus` can be used as
    its thread-safe callback, then invoke :meth:`run` from the process main
    thread.
    """

    def __init__(
        self,
        *,
        title: str = "XYQQuiz 科举答题助手",
        width: int = 1440,
        height: int = 900,
        min_size: tuple[int, int] = (1000, 650),
        maximized: bool = False,
        background_color: str = "#10151f",
        startup_timeout: float = 30.0,
        load_timeout: float = 10.0,
        join_timeout: float = 5.0,
        webview_loader: Callable[[], Any] | None = None,
        thread_factory: Callable[..., Any] = threading.Thread,
        health_check: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] = time.sleep,
        native_window_callback: Callable[[int], None] | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("desktop window size must be positive")
        if min_size[0] <= 0 or min_size[1] <= 0:
            raise ValueError("desktop minimum window size must be positive")
        if startup_timeout <= 0 or load_timeout <= 0 or join_timeout <= 0:
            raise ValueError("desktop timeouts must be positive")

        self.title = title
        self.width = width
        self.height = height
        self.min_size = min_size
        self.maximized = maximized
        self.background_color = background_color
        self.startup_timeout = startup_timeout
        self.load_timeout = load_timeout
        self.join_timeout = join_timeout
        self._webview_loader = webview_loader or _load_webview
        self._thread_factory = thread_factory
        self._health_check = health_check or _http_is_ready
        self._clock = clock
        self._wait = wait
        self._native_window_callback = native_window_callback

        self._state_lock = threading.RLock()
        self._focus_lock = threading.Lock()
        self._server: Any = None
        self._server_thread: Any = None
        self._server_error: BaseException | None = None
        self._server_started_once = False
        self._server_exited = False
        self._server_exit_unexpected = False
        self._server_exit_before_window_shown = False
        self._window: Any = None
        self._window_shown = False
        self._window_ever_shown = False
        self._window_loaded = False
        self._window_closed = False
        self._window_minimized = False
        self._window_maximized = maximized
        self._pending_focus = False
        self._close_requested = False
        self._close_window_when_shown = False
        self._gui_started = False
        self._run_started = False
        self._run_finished = False
        self._renderer_error: str | None = None
        self._load_error: str | None = None
        self._preserve_server_on_window_close = False
        self._load_watchdog_cancel: threading.Event | None = None
        self._load_watchdog_thread: Any = None
        self._fallback_opener: Callable[[str], Any] | None = None
        self._browser_url_factory: Callable[[], str] | None = None
        self._fallback_active = False

    def request_focus(self) -> bool | None:
        """Show, restore and activate the native window from any thread.

        ``None`` means the window is still starting and the request was queued;
        ``False`` means the window has closed or activation failed; ``True``
        means activation was accepted.  In browser fallback mode a fresh
        one-time URL is opened instead.
        """

        with self._state_lock:
            if self._close_requested:
                return False
            if self._fallback_active:
                opener = self._fallback_opener
                url_factory = self._browser_url_factory
                window = None
                was_minimized = False
                was_maximized = False
            elif self._window_closed or self._run_finished:
                return False
            elif self._window is None or not self._window_shown:
                self._pending_focus = True
                return None
            else:
                opener = None
                url_factory = None
                window = self._window
                was_minimized = self._window_minimized
                was_maximized = self._window_maximized
                self._pending_focus = False

        try:
            with self._focus_lock:
                if opener is not None and url_factory is not None:
                    result = opener(_issue_browser_url(url_factory))
                    return result is not False
                if window is None:
                    return False
                if was_minimized:
                    window.restore()
                    if was_maximized:
                        window.maximize()
                # On the WinForms backend pywebview's show() calls both Show()
                # and Activate(), so this also gives the window keyboard focus.
                window.show()
            return True
        except Exception:
            _LOGGER.exception("XYQQuiz desktop window activation failed")
            return False

    def request_close(self) -> None:
        """Request both the native window and Uvicorn server to stop."""

        with self._state_lock:
            self._close_requested = True
            server = self._server
            window = self._window if self._window_shown and not self._window_closed else None
            if self._window is not None and not self._window_shown and not self._window_closed:
                self._close_window_when_shown = True
        if server is not None:
            server.should_exit = True
        if window is not None:
            try:
                window.destroy()
            except Exception:
                _LOGGER.exception("XYQQuiz desktop window close request failed")

    def enable_browser_fallback(
        self,
        opener: Callable[[str], Any],
        browser_url_factory: Callable[[], str],
    ) -> None:
        """Route activation requests to an externally hosted browser window.

        This is primarily for the pre-server fallback boundary: when delayed
        pywebview import fails, the launcher can enable this routing before it
        invokes its existing browser runner.  Each activation gets a fresh
        one-time URL.
        """

        with self._state_lock:
            if self._close_requested:
                raise RuntimeError("desktop controller is closing")
            self._fallback_opener = opener
            self._browser_url_factory = browser_url_factory
            self._fallback_active = True

    def disable_browser_fallback(self) -> None:
        with self._state_lock:
            self._fallback_active = False
            self._fallback_opener = None

    def run(
        self,
        server: Any,
        health_url: str,
        browser_url_factory: Callable[[], str],
        *,
        sockets: Sequence[socket.socket] | None = None,
        fallback_opener: Callable[[str], Any] | None = None,
    ) -> DesktopRunMode:
        """Run Uvicorn in the background and pywebview on the main thread.

        ``fallback_opener`` is the safe post-start fallback: if WebView2 fails
        after Uvicorn is already healthy, the same server remains alive, the
        one-time URL is opened externally, and this call waits for that server
        to finish.  Without it, a post-start failure stops and joins Uvicorn
        before raising :class:`DesktopUnavailable`.
        """

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("pywebview GUI loop must run on the main thread")
        self._claim_run(server, browser_url_factory)
        try:
            webview = self._load_webview_module()
            if self._close_requested:
                return DesktopRunMode.DESKTOP
            self._start_server_thread(server, sockets)
            self._wait_until_healthy(server, health_url)
            if self._close_requested:
                return DesktopRunMode.DESKTOP

            browser_url = _issue_browser_url(browser_url_factory)
            try:
                window = webview.create_window(
                    self.title,
                    browser_url,
                    width=self.width,
                    height=self.height,
                    min_size=self.min_size,
                    resizable=True,
                    maximized=self.maximized,
                    background_color=self.background_color,
                )
                self._bind_window(window)
                with self._state_lock:
                    self._gui_started = True
                self._start_load_watchdog(window)
                try:
                    webview.start(gui="edgechromium")
                finally:
                    self._stop_load_watchdog()
            except Exception as error:
                # A server failure can race with WebView initialization.  The
                # backend failure is causal and must never be hidden behind a
                # DesktopUnavailable/browser fallback result.
                self._raise_server_failure_if_any()
                unavailable = DesktopUnavailable(
                    f"WebView2 桌面窗口初始化失败：{error}",
                    server_started=True,
                )
                return self._handle_post_start_unavailable(
                    unavailable,
                    fallback_opener,
                    cause=error,
                )
            self._raise_server_failure_if_any()
            unavailable = self._desktop_start_failure()
            if unavailable is not None:
                return self._handle_post_start_unavailable(
                    unavailable,
                    fallback_opener,
                )

            self._stop_and_join_server()
            self._raise_server_failure_if_any()
            return DesktopRunMode.DESKTOP
        finally:
            with self._state_lock:
                fallback_active = self._fallback_active
                thread = self._server_thread
            if not fallback_active and thread is not None and thread.is_alive():
                self._stop_and_join_server()
            with self._state_lock:
                self._run_finished = True

    def _claim_run(
        self,
        server: Any,
        browser_url_factory: Callable[[], str],
    ) -> None:
        with self._state_lock:
            if self._run_started:
                raise RuntimeError("desktop controller is single-use")
            self._run_started = True
            self._server = server
            self._browser_url_factory = browser_url_factory

    def _load_webview_module(self) -> Any:
        try:
            module = self._webview_loader()
            if not callable(getattr(module, "create_window", None)) or not callable(
                getattr(module, "start", None)
            ):
                raise TypeError("pywebview module is missing create_window/start")
            return module
        except Exception as error:
            raise DesktopUnavailable(
                f"pywebview 不可用：{error}",
                server_started=False,
            ) from error

    def _start_server_thread(
        self,
        server: Any,
        sockets: Sequence[socket.socket] | None,
    ) -> None:
        forwarded_sockets = None if sockets is None else list(sockets)

        def run_server() -> None:
            try:
                if forwarded_sockets is None:
                    server.run()
                else:
                    server.run(sockets=forwarded_sockets)
            except BaseException as error:
                with self._state_lock:
                    expected_shutdown = self._close_requested or (
                        self._window_closed
                        and not self._preserve_server_on_window_close
                    )
                    if not expected_shutdown:
                        self._server_error = error
                if expected_shutdown:
                    _LOGGER.info(
                        "Ignoring Uvicorn exception during requested shutdown: %s",
                        error,
                    )
            finally:
                with self._state_lock:
                    self._server_exited = True
                    self._server_exit_before_window_shown = not self._window_ever_shown
                    self._server_exit_unexpected = not (
                        self._close_requested
                        or self._window_closed
                        or bool(getattr(server, "should_exit", False))
                    )
                self._close_window_after_server_exit()

        thread = self._thread_factory(
            target=run_server,
            name="xyq-quiz-uvicorn",
            daemon=False,
        )
        with self._state_lock:
            self._server_thread = thread
            self._server_started_once = True
        thread.start()

    def _wait_until_healthy(self, server: Any, health_url: str) -> None:
        deadline = self._clock() + self.startup_timeout
        while True:
            with self._state_lock:
                thread = self._server_thread
                server_error = self._server_error
                close_requested = self._close_requested
            if server_error is not None:
                raise DesktopServerError(
                    f"Uvicorn 启动失败：{server_error}"
                ) from server_error
            if close_requested:
                return
            if bool(getattr(server, "started", False)) and self._health_check(health_url):
                return
            if thread is not None and not thread.is_alive():
                raise DesktopServerError("Uvicorn 在健康检查完成前退出")
            if self._clock() >= deadline:
                raise DesktopServerError("等待 Uvicorn 健康检查超时")
            self._wait(0.05)

    def _bind_window(self, window: Any) -> None:
        window.events.initialized += self._on_initialized
        window.events.shown += self._on_shown
        window.events.loaded += self._on_loaded
        window.events.minimized += self._on_minimized
        window.events.maximized += self._on_maximized
        window.events.restored += self._on_restored
        window.events.closed += self._on_closed
        with self._state_lock:
            self._window = window

    def _on_initialized(self, renderer: str) -> bool | None:
        if renderer == "edgechromium":
            return None
        with self._state_lock:
            self._renderer_error = (
                "pywebview 未能使用 WebView2（实际 renderer："
                f"{renderer or 'unknown'}）"
            )
        # initialized is a cancellable pywebview event.  False prevents a
        # silent fallback to the obsolete MSHTML renderer.
        return False

    def _on_shown(self) -> None:
        with self._state_lock:
            self._window_shown = True
            self._window_ever_shown = True
            pending = self._pending_focus
            close_window = self._close_window_when_shown
            if close_window:
                self._close_window_when_shown = False
                window = self._window
            else:
                window = None
            native_window = self._window
            native_callback = self._native_window_callback
        if native_callback is not None and native_window is not None:
            try:
                native = getattr(native_window, "native", None)
                handle = getattr(native, "Handle", None)
                hwnd = int(handle.ToInt64()) if handle is not None else 0
                if hwnd:
                    native_callback(hwnd)
            except Exception:
                _LOGGER.exception("Failed to publish pywebview native HWND")
        if window is not None:
            try:
                window.destroy()
            except Exception:
                _LOGGER.exception("Failed to close desktop window after early Uvicorn exit")
        elif pending:
            self.request_focus()

    def _on_loaded(self) -> None:
        with self._state_lock:
            self._window_loaded = True

    def _on_minimized(self) -> None:
        with self._state_lock:
            self._window_minimized = True

    def _on_maximized(self) -> None:
        with self._state_lock:
            self._window_minimized = False
            self._window_maximized = True

    def _on_restored(self) -> None:
        with self._state_lock:
            self._window_minimized = False
            self._window_maximized = False

    def _on_closed(self) -> None:
        with self._state_lock:
            self._window_closed = True
            self._window_shown = False
            server = self._server
            preserve_server = self._preserve_server_on_window_close
            native_callback = self._native_window_callback
        if native_callback is not None:
            try:
                native_callback(0)
            except Exception:
                _LOGGER.exception("Failed to clear pywebview native HWND")
        if server is not None and not preserve_server:
            server.should_exit = True

    def set_native_window_callback(
        self,
        callback: Callable[[int], None] | None,
    ) -> None:
        with self._state_lock:
            self._native_window_callback = callback

    def _desktop_start_failure(self) -> DesktopUnavailable | None:
        with self._state_lock:
            renderer_error = self._renderer_error
            load_error = self._load_error
            shown = self._window_shown
            closed = self._window_closed
            close_requested = self._close_requested
        if renderer_error is not None:
            return DesktopUnavailable(renderer_error, server_started=True)
        if load_error is not None:
            return DesktopUnavailable(load_error, server_started=True)
        if not shown and not closed and not close_requested:
            return DesktopUnavailable(
                "WebView2 GUI 循环已退出，但窗口从未显示",
                server_started=True,
            )
        return None

    def _handle_post_start_unavailable(
        self,
        unavailable: DesktopUnavailable,
        fallback_opener: Callable[[str], Any] | None,
        *,
        cause: BaseException | None = None,
    ) -> DesktopRunMode:
        self._raise_server_failure_if_any()
        with self._state_lock:
            server = self._server
            server_thread = self._server_thread
            browser_url_factory = self._browser_url_factory
            can_continue = (
                fallback_opener is not None
                and browser_url_factory is not None
                and server is not None
                and server_thread is not None
                and server_thread.is_alive()
                and not bool(getattr(server, "should_exit", False))
            )
            if can_continue:
                self._window = None
                self._window_shown = False
                self._window_loaded = False
                self._window_closed = False
                self._preserve_server_on_window_close = False

        if not can_continue:
            self._stop_and_join_server()
            self._raise_server_failure_if_any()
            if cause is None:
                raise unavailable
            raise unavailable from cause

        _LOGGER.warning("%s；改用系统浏览器继续运行", unavailable)
        self._raise_server_failure_if_any()
        self.enable_browser_fallback(fallback_opener, browser_url_factory)
        try:
            # The WebView may already have consumed its bootstrap token before
            # an asynchronous renderer/load failure.  A browser fallback must
            # always receive a newly issued one-time URL.
            result = fallback_opener(_issue_browser_url(browser_url_factory))
            if result is False:
                raise RuntimeError("系统浏览器拒绝打开页面")
        except Exception as error:
            self.disable_browser_fallback()
            self._stop_and_join_server()
            raise DesktopUnavailable(
                f"WebView2 不可用，系统浏览器回退也失败：{error}",
                server_started=True,
            ) from error
        try:
            # This is the external-browser application lifetime.  The Exit
            # action calls request_close(), which lets the same server unwind.
            server_thread.join()
            self._raise_server_failure_if_any()
            return DesktopRunMode.BROWSER_FALLBACK
        finally:
            self.disable_browser_fallback()

    def _close_window_after_server_exit(self) -> None:
        with self._state_lock:
            window = (
                self._window
                if self._gui_started and self._window_shown and not self._window_closed
                else None
            )
            if window is None and not self._window_closed:
                # Uvicorn may exit after health succeeds but before pywebview
                # emits shown.  Window methods cannot be called safely yet;
                # queue the destroy for _on_shown instead.
                self._close_window_when_shown = True
        if window is not None:
            try:
                window.destroy()
            except Exception:
                _LOGGER.exception("Failed to close desktop window after Uvicorn exit")

    def _start_load_watchdog(self, window: Any) -> None:
        cancel = threading.Event()

        def watch_initial_load() -> None:
            while not cancel.is_set():
                with self._state_lock:
                    shown = self._window_shown
                    loaded = self._window_loaded
                    closed = self._window_closed
                    close_requested = self._close_requested
                if loaded or closed or close_requested:
                    return
                if shown:
                    break
                self._wait(0.02)

            deadline = self._clock() + self.load_timeout
            while not cancel.is_set():
                with self._state_lock:
                    if (
                        self._window_loaded
                        or self._window_closed
                        or self._close_requested
                    ):
                        return
                    if self._clock() >= deadline:
                        self._load_error = (
                            "WebView2 窗口已显示，但本机页面在 "
                            f"{self.load_timeout:g} 秒内未加载完成"
                        )
                        # The close is a controlled hand-off, not an
                        # application exit.  Keep Uvicorn alive so the exact
                        # same server can continue in the system browser.
                        self._preserve_server_on_window_close = True
                        break
                self._wait(0.02)

            if cancel.is_set():
                return
            try:
                window.destroy()
            except Exception:
                _LOGGER.exception("Failed to close unresponsive WebView2 window")

        watchdog = self._thread_factory(
            target=watch_initial_load,
            name="xyq-quiz-webview-load-watchdog",
            daemon=True,
        )
        with self._state_lock:
            self._load_watchdog_cancel = cancel
            self._load_watchdog_thread = watchdog
        watchdog.start()

    def _stop_load_watchdog(self) -> None:
        with self._state_lock:
            cancel = self._load_watchdog_cancel
            watchdog = self._load_watchdog_thread
            self._load_watchdog_cancel = None
            self._load_watchdog_thread = None
        if cancel is not None:
            cancel.set()
        if watchdog is not None and threading.current_thread() is not watchdog:
            watchdog.join(timeout=min(1.0, self.join_timeout))
            if watchdog.is_alive():
                _LOGGER.warning("WebView2 initial-load watchdog did not stop promptly")

    def _stop_and_join_server(self) -> None:
        with self._state_lock:
            server = self._server
            thread = self._server_thread
        if server is not None:
            server.should_exit = True
        if thread is None or threading.current_thread() is thread:
            return
        thread.join(timeout=self.join_timeout)
        if thread.is_alive():
            raise DesktopServerError("Uvicorn 未能在桌面窗口关闭后及时退出")

    def _raise_server_failure_if_any(self) -> None:
        with self._state_lock:
            error = self._server_error
            exited_unexpectedly = self._server_exit_unexpected
            exited_before_shown = self._server_exit_before_window_shown
        if error is not None:
            raise DesktopServerError(f"Uvicorn 运行失败：{error}") from error
        if exited_unexpectedly:
            phase = "在桌面窗口显示前" if exited_before_shown else "运行期间"
            raise DesktopServerError(f"Uvicorn {phase}意外退出")


def _load_webview() -> Any:
    # Keep pywebview optional at import time so the launcher can retain its
    # external-browser fallback even when packaging or runtime dependencies are
    # missing.
    return importlib.import_module("webview")


def _issue_browser_url(factory: Callable[[], str]) -> str:
    url = factory()
    if not isinstance(url, str) or not url.startswith("http://127.0.0.1:"):
        raise ValueError("browser URL factory returned an invalid loopback URL")
    return url


def _http_is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=0.5) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return (
                isinstance(payload, dict)
                and payload.get("ok") is True
                and payload.get("app_id") == APP_ID
                and payload.get("ready") is True
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


__all__ = [
    "DesktopRunMode",
    "DesktopServerError",
    "DesktopUnavailable",
    "WebViewDesktopController",
]
