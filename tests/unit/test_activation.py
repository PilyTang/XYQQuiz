from __future__ import annotations

from multiprocessing.connection import Client
import threading
import time
from uuid import uuid4

import pytest

import xyq_quiz.runtime.activation as activation_module
from xyq_quiz.runtime.activation import (
    ACTIVATION_PROTOCOL,
    ActivationFocusError,
    ActivationNotReadyError,
    ActivationProtocolError,
    ActivationTimeoutError,
    ActivationServer,
    InstanceNames,
    activate_existing,
    handle_activation_request,
)
from xyq_quiz.web.security import APP_ID


class FakeConnection:
    def __init__(self, response: object) -> None:
        self.response = response
        self.request: object = None
        self.poll_timeouts: list[float] = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def send(self, value: object) -> None:
        self.request = value

    def recv(self) -> object:
        if callable(self.response):
            return self.response(self.request)
        return self.response

    def poll(self, timeout: float) -> bool:
        self.poll_timeouts.append(timeout)
        return True


def test_instance_names_are_stable_per_user_and_session() -> None:
    first = InstanceNames.for_identity("S-1-5-21-test", 3)
    same = InstanceNames.for_identity("S-1-5-21-test", 3)
    other = InstanceNames.for_identity("S-1-5-21-test", 4)

    assert first == same
    assert first != other
    assert first.mutex.startswith("Local\\XYQQuizBackend-")
    assert first.pipe.startswith(r"\\.\pipe\XYQQuiz-")


def test_activation_protocol_echoes_challenge_and_current_url() -> None:
    response = handle_activation_request(
        {
            "protocol": ACTIVATION_PROTOCOL,
            "app_id": APP_ID,
            "action": "ACTIVATE",
            "challenge": "challenge",
        },
        lambda: "http://127.0.0.1:8765/#token=once",
    )

    assert response == {
        "ok": True,
        "protocol": ACTIVATION_PROTOCOL,
        "app_id": APP_ID,
        "challenge": "challenge",
        "url": "http://127.0.0.1:8765/#token=once",
    }


def test_desktop_activation_focuses_without_exposing_browser_url() -> None:
    calls: list[str] = []

    response = handle_activation_request(
        {
            "protocol": ACTIVATION_PROTOCOL,
            "app_id": APP_ID,
            "action": "ACTIVATE",
            "challenge": "challenge",
        },
        focus_callback=lambda: calls.append("focus") is None,
    )

    assert calls == ["focus"]
    assert response == {
        "ok": True,
        "protocol": ACTIVATION_PROTOCOL,
        "app_id": APP_ID,
        "challenge": "challenge",
        "activation": "desktop",
    }
    assert "url" not in response


@pytest.mark.parametrize(
    ("focus_result", "expected_error"),
    [
        (None, "not_ready"),
        (False, "focus_failed"),
    ],
)
def test_desktop_activation_distinguishes_not_ready_from_focus_failure(
    focus_result: bool | None,
    expected_error: str,
) -> None:
    response = handle_activation_request(
        {
            "protocol": ACTIVATION_PROTOCOL,
            "app_id": APP_ID,
            "action": "ACTIVATE",
            "challenge": "challenge",
        },
        focus_callback=lambda: focus_result,
    )

    assert response["ok"] is False
    assert response["error"] == expected_error
    assert response["challenge"] == "challenge"


def test_desktop_activation_contains_focus_callback_failure() -> None:
    def fail_to_focus() -> bool:
        raise RuntimeError("window was destroyed")

    response = handle_activation_request(
        {
            "protocol": ACTIVATION_PROTOCOL,
            "app_id": APP_ID,
            "action": "ACTIVATE",
            "challenge": "challenge",
        },
        focus_callback=fail_to_focus,
    )

    assert response["ok"] is False
    assert response["error"] == "focus_failed"
    assert "window was destroyed" not in str(response)


def test_activate_existing_validates_response_before_opening() -> None:
    opened: list[str] = []

    def connector(*_args, **_kwargs):
        def response(request):
            return {
                "ok": True,
                "protocol": ACTIVATION_PROTOCOL,
                "app_id": APP_ID,
                "challenge": request["challenge"],
                "url": "http://127.0.0.1:8765/#token=once",
            }

        return FakeConnection(response)

    assert activate_existing("pipe", opened.append, connector=connector) is True
    assert opened == ["http://127.0.0.1:8765/#token=once"]


def test_activate_existing_accepts_desktop_focus_without_calling_opener() -> None:
    opened: list[str] = []

    def connector(*_args, **_kwargs):
        def response(request):
            return {
                "ok": True,
                "protocol": ACTIVATION_PROTOCOL,
                "app_id": APP_ID,
                "challenge": request["challenge"],
                "activation": "desktop",
            }

        return FakeConnection(response)

    assert activate_existing("pipe", opened.append, connector=connector) is True
    assert opened == []


def test_activate_existing_missing_pipe_is_not_an_error() -> None:
    def connector(*_args, **_kwargs):
        raise FileNotFoundError

    assert activate_existing("pipe", lambda _url: None, connector=connector) is False


def test_zero_retry_probe_still_has_bounded_response_window() -> None:
    connection: FakeConnection | None = None

    def connector(*_args, **_kwargs):
        nonlocal connection

        def response(request):
            return {
                "ok": True,
                "protocol": ACTIVATION_PROTOCOL,
                "app_id": APP_ID,
                "challenge": request["challenge"],
                "activation": "desktop",
            }

        connection = FakeConnection(response)
        return connection

    assert activate_existing(
        "pipe",
        lambda _url: None,
        connector=connector,
        response_timeout=0.25,
        clock=lambda: 10.0,
    )
    assert connection is not None
    assert connection.poll_timeouts == pytest.approx([0.25])


def test_connected_instance_response_timeout_is_bounded() -> None:
    class SilentConnection(FakeConnection):
        def poll(self, timeout: float) -> bool:
            self.poll_timeouts.append(timeout)
            return False

        def recv(self) -> object:
            pytest.fail("recv must not run without a successful poll")

    connection = SilentConnection(None)

    with pytest.raises(ActivationTimeoutError, match="时限"):
        activate_existing(
            "pipe",
            lambda _url: None,
            connector=lambda *_args, **_kwargs: connection,
            response_timeout=0.4,
            clock=lambda: 20.0,
        )

    assert connection.poll_timeouts == pytest.approx([0.4])


def test_activate_existing_rejects_impostor_response() -> None:
    def connector(*_args, **_kwargs):
        return FakeConnection({"ok": True, "url": "http://attacker"})

    with pytest.raises(ActivationProtocolError, match="协议校验失败"):
        activate_existing("pipe", lambda _url: None, connector=connector)


def test_activate_existing_reports_desktop_not_ready_separately() -> None:
    def connector(*_args, **_kwargs):
        def response(request):
            return {
                "ok": False,
                "error": "not_ready",
                "protocol": ACTIVATION_PROTOCOL,
                "app_id": APP_ID,
                "challenge": request["challenge"],
            }

        return FakeConnection(response)

    with pytest.raises(ActivationNotReadyError, match="仍在启动"):
        activate_existing("pipe", lambda _url: None, connector=connector)


def test_activate_existing_reports_focus_failure_separately() -> None:
    def connector(*_args, **_kwargs):
        def response(request):
            return {
                "ok": False,
                "error": "focus_failed",
                "protocol": ACTIVATION_PROTOCOL,
                "app_id": APP_ID,
                "challenge": request["challenge"],
            }

        return FakeConnection(response)

    with pytest.raises(ActivationFocusError, match="窗口"):
        activate_existing("pipe", lambda _url: None, connector=connector)


def test_activate_existing_rejects_desktop_response_containing_url() -> None:
    def connector(*_args, **_kwargs):
        def response(request):
            return {
                "ok": True,
                "protocol": ACTIVATION_PROTOCOL,
                "app_id": APP_ID,
                "challenge": request["challenge"],
                "activation": "desktop",
                "url": "http://127.0.0.1:8765/#token=must-not-leak",
            }

        return FakeConnection(response)

    with pytest.raises(ActivationProtocolError, match="协议校验失败"):
        activate_existing("pipe", lambda _url: None, connector=connector)


def test_real_windows_named_pipe_activation_round_trip() -> None:
    address = rf"\\.\pipe\XYQQuiz-test-{uuid4().hex}"
    url = "http://127.0.0.1:8765/#token=real-pipe"
    opened: list[str] = []

    with ActivationServer(address, lambda: url):
        assert activate_existing(address, opened.append, timeout=1.0) is True

    assert opened == [url]


def test_blocked_focus_callback_does_not_wedge_server_or_stop() -> None:
    address = rf"\\.\pipe\XYQQuiz-test-{uuid4().hex}"
    callback_started = threading.Event()
    release_callback = threading.Event()

    def blocked_focus() -> bool:
        callback_started.set()
        release_callback.wait()
        return True

    server = ActivationServer(
        address,
        focus_callback=blocked_focus,
        callback_timeout=0.05,
    )
    server.start()
    try:
        with pytest.raises(ActivationTimeoutError, match="超时"):
            activate_existing(address, lambda _url: None, response_timeout=0.5)
        assert callback_started.is_set()
        started = time.monotonic()
        server.stop(timeout=0.5)
        assert time.monotonic() - started < 0.5
    finally:
        release_callback.set()
        server.stop(timeout=0.5)


def test_silent_authenticated_client_does_not_wedge_next_activation() -> None:
    address = rf"\\.\pipe\XYQQuiz-test-{uuid4().hex}"
    url = "http://127.0.0.1:8765/#token=after-silent-client"
    opened: list[str] = []

    with ActivationServer(address, lambda: url, request_timeout=0.05):
        with Client(
            address,
            family="AF_PIPE",
            authkey=activation_module._AUTHKEY,
        ):
            time.sleep(0.08)
            assert activate_existing(
                address,
                opened.append,
                response_timeout=0.5,
            )

    assert opened == [url]
