from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any

import pytest

from xyq_quiz.desktop.webview import (
    DesktopRunMode,
    DesktopServerError,
    DesktopUnavailable,
    WebViewDesktopController,
)


class FakeEvent:
    def __init__(self) -> None:
        self.handlers: list[Callable[..., Any]] = []

    def __iadd__(self, handler: Callable[..., Any]):
        self.handlers.append(handler)
        return self

    def emit(self, *args: object) -> list[object]:
        return [handler(*args) for handler in tuple(self.handlers)]


class FakeEvents:
    def __init__(self) -> None:
        self.initialized = FakeEvent()
        self.shown = FakeEvent()
        self.loaded = FakeEvent()
        self.minimized = FakeEvent()
        self.maximized = FakeEvent()
        self.restored = FakeEvent()
        self.closed = FakeEvent()


class FakeWindow:
    def __init__(self) -> None:
        self.events = FakeEvents()
        self.calls: list[str] = []
        self._destroyed = False

    def show(self) -> None:
        self.calls.append("show")

    def restore(self) -> None:
        self.calls.append("restore")

    def maximize(self) -> None:
        self.calls.append("maximize")

    def destroy(self) -> None:
        self.calls.append("destroy")
        if not self._destroyed:
            self._destroyed = True
            self.events.closed.emit()


class FakeWebview:
    def __init__(
        self,
        *,
        renderer: str = "edgechromium",
        before_show_action: Callable[[FakeWindow], None] | None = None,
        start_action: Callable[[FakeWindow], None] | None = None,
    ) -> None:
        self.renderer = renderer
        self.before_show_action = before_show_action
        self.start_action = start_action
        self.window = FakeWindow()
        self.create_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.start_calls: list[dict[str, object]] = []
        self.start_thread: threading.Thread | None = None

    def create_window(self, *args: object, **kwargs: object) -> FakeWindow:
        self.create_calls.append((args, kwargs))
        return self.window

    def start(self, **kwargs: object) -> None:
        self.start_calls.append(kwargs)
        self.start_thread = threading.current_thread()
        cancelled = False in self.window.events.initialized.emit(self.renderer)
        if cancelled:
            return
        if self.before_show_action is not None:
            self.before_show_action(self.window)
        self.window.events.shown.emit()
        if self.start_action is None:
            self.window.events.closed.emit()
        else:
            self.start_action(self.window)


class FakeServer:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.started = False
        self.should_exit = False
        self.failure = failure
        self.run_calls: list[list[object] | None] = []
        self.run_thread: threading.Thread | None = None

    def run(self, *, sockets: list[object] | None = None) -> None:
        self.run_calls.append(sockets)
        self.run_thread = threading.current_thread()
        if self.failure is not None:
            raise self.failure
        self.started = True
        while not self.should_exit:
            time.sleep(0.001)


class PostHealthExitServer:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.started = False
        self.should_exit = False
        self.failure = failure
        self.release = threading.Event()
        self.run_calls = 0

    def run(self, **_kwargs: object) -> None:
        self.run_calls += 1
        self.started = True
        assert self.release.wait(timeout=1)
        if self.failure is not None:
            raise self.failure


class FailureOnShutdownServer:
    def __init__(self, message: str = "connection is closing") -> None:
        self.started = False
        self.should_exit = False
        self.message = message
        self.run_calls = 0

    def run(self, **_kwargs: object) -> None:
        self.run_calls += 1
        self.started = True
        deadline = time.monotonic() + 1
        while not self.should_exit and time.monotonic() < deadline:
            time.sleep(0.001)
        assert self.should_exit
        raise RuntimeError(self.message)


class RecordingThreadFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> threading.Thread:
        self.calls.append(kwargs)
        return threading.Thread(**kwargs)


def make_controller(
    webview: FakeWebview,
    **kwargs: object,
) -> WebViewDesktopController:
    return WebViewDesktopController(
        webview_loader=lambda: webview,
        health_check=lambda _url: True,
        startup_timeout=1,
        join_timeout=1,
        **kwargs,
    )


def test_pywebview_is_loaded_lazily_before_server_is_started() -> None:
    server = FakeServer()

    def missing_webview() -> object:
        raise ModuleNotFoundError("No module named 'webview'")

    controller = WebViewDesktopController(webview_loader=missing_webview)

    with pytest.raises(DesktopUnavailable, match="pywebview 不可用") as captured:
        controller.run(
            server,
            "http://127.0.0.1:8765/api/health",
            lambda: "http://127.0.0.1:8765/#token=once",
        )

    assert captured.value.server_started is False
    assert server.run_calls == []


def test_run_uses_background_server_and_main_thread_edgechromium_gui() -> None:
    webview = FakeWebview()
    server = FakeServer()
    thread_factory = RecordingThreadFactory()
    reserved = object()
    issued = 0

    def issue_url() -> str:
        nonlocal issued
        issued += 1
        return "http://127.0.0.1:8765/#token=once"

    controller = make_controller(webview, thread_factory=thread_factory)

    assert controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        issue_url,
        sockets=[reserved],
    ) is DesktopRunMode.DESKTOP

    assert issued == 1
    assert webview.start_calls == [{"gui": "edgechromium"}]
    assert webview.start_thread is threading.main_thread()
    assert server.run_thread is not threading.main_thread()
    assert server.run_calls == [[reserved]]
    assert len(thread_factory.calls) == 2
    assert thread_factory.calls[0]["name"] == "xyq-quiz-uvicorn"
    assert thread_factory.calls[0]["daemon"] is False
    assert thread_factory.calls[1]["name"] == "xyq-quiz-webview-load-watchdog"
    assert thread_factory.calls[1]["daemon"] is True
    assert server.should_exit is True


def test_default_window_is_resizable_and_not_maximized() -> None:
    webview = FakeWebview()
    controller = make_controller(webview)

    controller.run(
        FakeServer(),
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
    )

    args, kwargs = webview.create_calls[0]
    assert args == (
        "XYQQuiz 科举答题助手",
        "http://127.0.0.1:8765/#token=once",
    )
    assert kwargs == {
        "width": 1440,
        "height": 900,
        "min_size": (1000, 650),
        "resizable": True,
        "maximized": False,
        "background_color": "#10151f",
    }


def test_focus_requested_before_shown_is_queued_and_activated_after_show() -> None:
    def start_action(window: FakeWindow) -> None:
        window.events.closed.emit()

    webview = FakeWebview(start_action=start_action)
    controller = make_controller(webview)

    assert controller.request_focus() is None
    controller.run(
        FakeServer(),
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
    )

    assert webview.window.calls == ["show"]
    assert controller.request_focus() is False


def test_focus_restores_minimized_maximized_window_then_activates_it() -> None:
    results: list[bool | None] = []

    def start_action(window: FakeWindow) -> None:
        window.events.maximized.emit()
        window.events.minimized.emit()
        results.append(controller.request_focus())
        window.events.closed.emit()

    webview = FakeWebview(start_action=start_action)
    controller = make_controller(webview)

    controller.run(
        FakeServer(),
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
    )

    assert results == [True]
    assert webview.window.calls == ["restore", "maximize", "show"]


def test_focus_restores_default_minimized_window_without_maximizing_it() -> None:
    results: list[bool | None] = []

    def start_action(window: FakeWindow) -> None:
        window.events.minimized.emit()
        results.append(controller.request_focus())
        window.events.closed.emit()

    webview = FakeWebview(start_action=start_action)
    controller = make_controller(webview)

    controller.run(
        FakeServer(),
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
    )

    assert results == [True]
    assert webview.window.calls == ["restore", "show"]


def test_request_close_from_worker_stops_server_and_destroys_window() -> None:
    close_result: list[BaseException] = []

    def start_action(_window: FakeWindow) -> None:
        def close() -> None:
            try:
                controller.request_close()
            except BaseException as error:
                close_result.append(error)

        worker = threading.Thread(target=close)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive()

    webview = FakeWebview(start_action=start_action)
    controller = make_controller(webview)
    server = FakeServer()

    controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
    )

    assert close_result == []
    assert server.should_exit is True
    assert "destroy" in webview.window.calls


def test_window_close_ignores_server_exception_from_shutdown_cleanup() -> None:
    server = FailureOnShutdownServer()

    def user_closes(window: FakeWindow) -> None:
        window.events.closed.emit()

    controller = make_controller(FakeWebview(start_action=user_closes))

    assert controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
    ) is DesktopRunMode.DESKTOP

    assert server.run_calls == 1


def test_request_close_ignores_server_exception_from_shutdown_cleanup() -> None:
    server = FailureOnShutdownServer()

    def close_from_ui(_window: FakeWindow) -> None:
        controller.request_close()

    controller = make_controller(FakeWebview(start_action=close_from_ui))

    assert controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
    ) is DesktopRunMode.DESKTOP

    assert server.run_calls == 1


def test_initial_page_load_completes_without_triggering_browser_fallback() -> None:
    def finish_load(window: FakeWindow) -> None:
        window.events.loaded.emit()
        window.events.closed.emit()

    webview = FakeWebview(start_action=finish_load)
    opened: list[str] = []
    controller = make_controller(webview, load_timeout=0.05)

    assert controller.run(
        FakeServer(),
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
        fallback_opener=opened.append,
    ) is DesktopRunMode.DESKTOP

    assert opened == []


def test_initial_page_load_timeout_destroys_blank_window_and_falls_back() -> None:
    server = FakeServer()

    def wait_for_watchdog(window: FakeWindow) -> None:
        deadline = time.monotonic() + 1
        while not window._destroyed and time.monotonic() < deadline:
            time.sleep(0.001)
        assert window._destroyed

    webview = FakeWebview(start_action=wait_for_watchdog)
    opened: list[str] = []

    def open_browser(url: str) -> bool:
        opened.append(url)
        server.should_exit = True
        return True

    controller = make_controller(webview, load_timeout=0.02)

    assert controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
        fallback_opener=open_browser,
    ) is DesktopRunMode.BROWSER_FALLBACK

    assert opened == ["http://127.0.0.1:8765/#token=once"]
    assert webview.window.calls == ["destroy"]
    assert len(server.run_calls) == 1


def test_user_close_before_initial_load_is_not_mislabeled_as_timeout() -> None:
    def user_closes(window: FakeWindow) -> None:
        window.events.closed.emit()

    webview = FakeWebview(start_action=user_closes)
    opened: list[str] = []
    controller = make_controller(webview, load_timeout=0.02)

    assert controller.run(
        FakeServer(),
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
        fallback_opener=opened.append,
    ) is DesktopRunMode.DESKTOP

    assert opened == []


def test_webview2_start_failure_continues_on_same_server_in_browser() -> None:
    def fail_start(_window: FakeWindow) -> None:
        raise RuntimeError("WebView2 runtime missing")

    webview = FakeWebview(start_action=fail_start)
    server = FakeServer()
    opened: list[str] = []
    issued = 0

    def issue_url() -> str:
        nonlocal issued
        issued += 1
        return f"http://127.0.0.1:8765/#token={issued}"

    def open_browser(url: str) -> bool:
        opened.append(url)
        server.should_exit = True
        return True

    controller = make_controller(webview)
    mode = controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        issue_url,
        fallback_opener=open_browser,
    )

    assert mode is DesktopRunMode.BROWSER_FALLBACK
    assert opened == ["http://127.0.0.1:8765/#token=2"]
    assert issued == 2
    assert len(server.run_calls) == 1


def test_activation_in_browser_fallback_issues_a_fresh_one_time_url() -> None:
    def fail_start(_window: FakeWindow) -> None:
        raise RuntimeError("WebView2 init failed")

    webview = FakeWebview(start_action=fail_start)
    server = FakeServer()
    opened: list[str] = []
    issued = 0

    def issue_url() -> str:
        nonlocal issued
        issued += 1
        return f"http://127.0.0.1:8765/#token={issued}"

    def open_browser(url: str) -> bool:
        opened.append(url)
        if len(opened) == 1:
            assert controller.request_focus() is True
        else:
            server.should_exit = True
        return True

    controller = make_controller(webview)

    assert controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        issue_url,
        fallback_opener=open_browser,
    ) is DesktopRunMode.BROWSER_FALLBACK

    assert opened == [
        "http://127.0.0.1:8765/#token=2",
        "http://127.0.0.1:8765/#token=3",
    ]


def test_browser_fallback_still_reports_unrequested_server_failure() -> None:
    server = PostHealthExitServer(RuntimeError("fallback backend crash"))

    def fail_start(_window: FakeWindow) -> None:
        raise RuntimeError("WebView2 init failed")

    def open_browser(_url: str) -> bool:
        server.release.set()
        return True

    controller = make_controller(FakeWebview(start_action=fail_start))

    with pytest.raises(DesktopServerError, match="fallback backend crash"):
        controller.run(
            server,
            "http://127.0.0.1:8765/api/health",
            lambda: "http://127.0.0.1:8765/#token=once",
            fallback_opener=open_browser,
        )


def test_browser_fallback_request_close_ignores_shutdown_cleanup_failure() -> None:
    server = FailureOnShutdownServer()

    def fail_start(_window: FakeWindow) -> None:
        raise RuntimeError("WebView2 init failed")

    def open_browser(_url: str) -> bool:
        controller.request_close()
        return True

    controller = make_controller(FakeWebview(start_action=fail_start))

    assert controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
        fallback_opener=open_browser,
    ) is DesktopRunMode.BROWSER_FALLBACK


def test_post_start_failure_without_opener_stops_server_before_raising() -> None:
    def fail_start(_window: FakeWindow) -> None:
        raise RuntimeError("WebView2 runtime missing")

    webview = FakeWebview(start_action=fail_start)
    server = FakeServer()
    controller = make_controller(webview)

    with pytest.raises(DesktopUnavailable, match="WebView2") as captured:
        controller.run(
            server,
            "http://127.0.0.1:8765/api/health",
            lambda: "http://127.0.0.1:8765/#token=once",
        )

    assert captured.value.server_started is True
    assert server.should_exit is True
    assert len(server.run_calls) == 1


def test_pre_start_browser_fallback_keeps_activation_callback_useful() -> None:
    server = FakeServer()
    opened: list[str] = []
    issued = 0

    def missing_webview() -> object:
        raise ModuleNotFoundError("webview")

    def issue_url() -> str:
        nonlocal issued
        issued += 1
        return f"http://127.0.0.1:8765/#token={issued}"

    controller = WebViewDesktopController(webview_loader=missing_webview)
    with pytest.raises(DesktopUnavailable):
        controller.run(
            server,
            "http://127.0.0.1:8765/api/health",
            issue_url,
        )

    controller.enable_browser_fallback(opened.append, issue_url)
    assert controller.request_focus() is True
    controller.disable_browser_fallback()

    assert opened == ["http://127.0.0.1:8765/#token=1"]
    assert controller.request_focus() is False


def test_mshtml_silent_fallback_is_rejected_and_browser_is_used() -> None:
    webview = FakeWebview(renderer="mshtml")
    server = FakeServer()
    opened: list[str] = []

    def open_browser(url: str) -> bool:
        opened.append(url)
        server.should_exit = True
        return True

    controller = make_controller(webview)

    assert controller.run(
        server,
        "http://127.0.0.1:8765/api/health",
        lambda: "http://127.0.0.1:8765/#token=once",
        fallback_opener=open_browser,
    ) is DesktopRunMode.BROWSER_FALLBACK

    assert opened == ["http://127.0.0.1:8765/#token=once"]


def test_server_failure_is_not_mislabeled_as_desktop_unavailable() -> None:
    server = FakeServer(failure=RuntimeError("bind failed"))
    controller = make_controller(FakeWebview())

    with pytest.raises(DesktopServerError, match="bind failed"):
        controller.run(
            server,
            "http://127.0.0.1:8765/api/health",
            lambda: "http://127.0.0.1:8765/#token=once",
        )


@pytest.mark.parametrize(
    "server_failure",
    [None, RuntimeError("post-health crash")],
    ids=["early-return", "exception"],
)
def test_post_health_server_exit_before_show_is_queued_and_wins_over_webview_failure(
    server_failure: BaseException | None,
) -> None:
    server = PostHealthExitServer(server_failure)
    opened: list[str] = []

    def exit_server_before_show(_window: FakeWindow) -> None:
        server.release.set()
        deadline = time.monotonic() + 1
        while controller._server_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert not controller._server_thread.is_alive()

    def simultaneous_webview_failure(_window: FakeWindow) -> None:
        raise RuntimeError("WebView2 also failed")

    webview = FakeWebview(
        before_show_action=exit_server_before_show,
        start_action=simultaneous_webview_failure,
    )
    controller = make_controller(webview)

    expected = "post-health crash" if server_failure is not None else "显示前意外退出"
    with pytest.raises(DesktopServerError, match=expected):
        controller.run(
            server,
            "http://127.0.0.1:8765/api/health",
            lambda: "http://127.0.0.1:8765/#token=once",
            fallback_opener=opened.append,
        )

    assert opened == []
    assert server.run_calls == 1
    assert webview.window.calls == ["destroy"]


def test_gui_loop_is_rejected_off_main_thread_before_server_start() -> None:
    controller = make_controller(FakeWebview())
    server = FakeServer()
    errors: list[BaseException] = []

    def exercise() -> None:
        try:
            controller.run(
                server,
                "http://127.0.0.1:8765/api/health",
                lambda: "http://127.0.0.1:8765/#token=once",
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=exercise)
    worker.start()
    worker.join(timeout=1)

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "main thread" in str(errors[0])
    assert server.run_calls == []


def test_health_timeout_stops_server_without_creating_window() -> None:
    webview = FakeWebview()
    server = FakeServer()
    now = iter([0.0, 2.0])
    controller = WebViewDesktopController(
        webview_loader=lambda: webview,
        health_check=lambda _url: False,
        clock=lambda: next(now),
        wait=lambda _seconds: None,
        startup_timeout=1,
        join_timeout=1,
    )

    with pytest.raises(DesktopServerError, match="健康检查超时"):
        controller.run(
            server,
            "http://127.0.0.1:8765/api/health",
            lambda: "http://127.0.0.1:8765/#token=once",
        )

    assert server.should_exit is True
    assert webview.create_calls == []
