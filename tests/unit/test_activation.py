from __future__ import annotations

import pytest
from uuid import uuid4

from xyq_quiz.runtime.activation import (
    ACTIVATION_PROTOCOL,
    ActivationProtocolError,
    InstanceNames,
    activate_existing,
    handle_activation_request,
    ActivationServer,
)
from xyq_quiz.web.security import APP_ID


class FakeConnection:
    def __init__(self, response: object) -> None:
        self.response = response
        self.request: object = None

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


def test_activate_existing_missing_pipe_is_not_an_error() -> None:
    def connector(*_args, **_kwargs):
        raise FileNotFoundError

    assert activate_existing("pipe", lambda _url: None, connector=connector) is False


def test_activate_existing_rejects_impostor_response() -> None:
    def connector(*_args, **_kwargs):
        return FakeConnection({"ok": True, "url": "http://attacker"})

    with pytest.raises(ActivationProtocolError, match="协议校验失败"):
        activate_existing("pipe", lambda _url: None, connector=connector)


def test_real_windows_named_pipe_activation_round_trip() -> None:
    address = rf"\\.\pipe\XYQQuiz-test-{uuid4().hex}"
    url = "http://127.0.0.1:8765/#token=real-pipe"
    opened: list[str] = []

    with ActivationServer(address, lambda: url):
        assert activate_existing(address, opened.append, timeout=1.0) is True

    assert opened == [url]
