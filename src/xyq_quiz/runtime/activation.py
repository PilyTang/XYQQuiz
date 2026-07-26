from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import getpass
import hashlib
import logging
import os
import secrets
import threading
import time
from typing import Any

from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener

from xyq_quiz.web.security import APP_ID


ACTIVATION_PROTOCOL = 1
_AUTHKEY = b"XYQQuiz-Activation-v1"
_LOGGER = logging.getLogger(__name__)
_NO_REQUEST = object()


class ActivationProtocolError(RuntimeError):
    pass


class ActivationNotReadyError(ActivationProtocolError):
    """The existing instance is authentic but cannot activate its UI yet."""


class ActivationFocusError(ActivationProtocolError):
    """The existing desktop instance could not focus or restore its window."""


class ActivationTimeoutError(ActivationProtocolError):
    """The existing instance did not complete activation within the time limit."""


@dataclass(frozen=True, slots=True)
class InstanceNames:
    mutex: str
    pipe: str

    @classmethod
    def for_identity(cls, user_sid: str, session_id: int) -> InstanceNames:
        identity = f"{user_sid}|{session_id}".encode("utf-8")
        suffix = hashlib.sha256(identity).hexdigest()[:24]
        return cls(
            mutex=f"Local\\XYQQuizBackend-{suffix}",
            pipe=rf"\\.\pipe\XYQQuiz-{suffix}",
        )


def current_instance_names() -> InstanceNames:
    return InstanceNames.for_identity(_current_user_sid(), _current_session_id())


def handle_activation_request(
    request: object,
    url_factory: Callable[[], str | None] | None = None,
    *,
    focus_callback: Callable[[], bool | None] | None = None,
) -> dict[str, object]:
    if not isinstance(request, dict):
        return {"ok": False, "error": "invalid_request"}
    challenge = request.get("challenge")
    if (
        request.get("protocol") != ACTIVATION_PROTOCOL
        or request.get("app_id") != APP_ID
        or request.get("action") != "ACTIVATE"
        or not isinstance(challenge, str)
        or not challenge
    ):
        return {"ok": False, "error": "protocol_mismatch"}
    common_response: dict[str, object] = {
        "protocol": ACTIVATION_PROTOCOL,
        "app_id": APP_ID,
        "challenge": challenge,
    }
    if focus_callback is not None:
        try:
            focused = focus_callback()
        except Exception:
            _LOGGER.exception("桌面窗口激活回调执行失败")
            return {"ok": False, "error": "focus_failed", **common_response}
        if focused is None:
            return {"ok": False, "error": "not_ready", **common_response}
        if focused is not True:
            return {"ok": False, "error": "focus_failed", **common_response}
        return {"ok": True, **common_response, "activation": "desktop"}

    if url_factory is None:
        return {"ok": False, "error": "not_ready", **common_response}
    try:
        url = url_factory()
    except Exception:
        _LOGGER.exception("浏览器激活 URL 生成失败")
        url = None
    if url is None:
        return {"ok": False, "error": "not_ready", **common_response}
    return {
        "ok": True,
        **common_response,
        "url": url,
    }


def _activation_timeout_response(request: object) -> dict[str, object]:
    if not isinstance(request, dict):
        return {"ok": False, "error": "invalid_request"}
    challenge = request.get("challenge")
    if not isinstance(challenge, str) or not challenge:
        return {"ok": False, "error": "protocol_mismatch"}
    return {
        "ok": False,
        "error": "activation_timeout",
        "protocol": ACTIVATION_PROTOCOL,
        "app_id": APP_ID,
        "challenge": challenge,
    }


class ActivationServer:
    def __init__(
        self,
        address: str,
        url_factory: Callable[[], str | None] | None = None,
        *,
        focus_callback: Callable[[], bool | None] | None = None,
        request_timeout: float = 1.0,
        callback_timeout: float = 1.0,
        listener_factory: Callable[..., Any] = Listener,
        connector: Callable[..., Any] = Client,
    ) -> None:
        if (url_factory is None) == (focus_callback is None):
            raise ValueError(
                "activation server requires exactly one of url_factory or focus_callback"
            )
        if request_timeout <= 0 or callback_timeout <= 0:
            raise ValueError("activation server timeouts must be positive")
        self.address = address
        self.url_factory = url_factory
        self.focus_callback = focus_callback
        self.request_timeout = request_timeout
        self.callback_timeout = callback_timeout
        self._listener_factory = listener_factory
        self._connector = connector
        self._listener: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._focus_lock = threading.Lock()
        self._callback_gate = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("activation server already started")
        self._listener = self._listener_factory(
            self.address,
            family="AF_PIPE",
            authkey=_AUTHKEY,
        )
        self._thread = threading.Thread(
            target=self._serve,
            name="xyq-quiz-activation",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        wakeup = threading.Thread(
            target=self._wake_listener,
            name="xyq-quiz-activation-stop",
            daemon=True,
        )
        wakeup.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError("activation server did not stop")
        self._thread = None
        self._listener = None

    def __enter__(self) -> ActivationServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def _serve(self) -> None:
        listener = self._listener
        try:
            while not self._stop.is_set():
                try:
                    connection = listener.accept()
                except (OSError, EOFError, AuthenticationError):
                    if self._stop.is_set():
                        break
                    continue
                with connection:
                    try:
                        request = self._receive_request(connection)
                        if request is _NO_REQUEST:
                            continue
                        if request == {"action": "STOP"} and self._stop.is_set():
                            break
                        response = self._dispatch_request(request)
                        if response is not None:
                            connection.send(response)
                    except (EOFError, OSError):
                        continue
        finally:
            listener.close()

    def _wake_listener(self) -> None:
        try:
            with self._connector(
                self.address,
                family="AF_PIPE",
                authkey=_AUTHKEY,
            ) as connection:
                connection.send({"action": "STOP"})
        except (OSError, EOFError, AuthenticationError):
            pass

    def _receive_request(self, connection: Any) -> object:
        deadline = time.monotonic() + self.request_timeout
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _NO_REQUEST
            if connection.poll(min(0.05, remaining)):
                return connection.recv()
        return _NO_REQUEST

    def _dispatch_request(self, request: object) -> dict[str, object] | None:
        if not self._callback_gate.acquire(blocking=False):
            return _activation_timeout_response(request)
        completed = threading.Event()
        result: list[dict[str, object]] = []

        def invoke() -> None:
            try:
                result.append(
                    handle_activation_request(
                        request,
                        self.url_factory,
                        focus_callback=(
                            self._invoke_focus
                            if self.focus_callback is not None
                            else None
                        ),
                    )
                )
            finally:
                self._callback_gate.release()
                completed.set()

        threading.Thread(
            target=invoke,
            name="xyq-quiz-activation-callback",
            daemon=True,
        ).start()
        deadline = time.monotonic() + self.callback_timeout
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _activation_timeout_response(request)
            if completed.wait(min(0.05, remaining)):
                return result[0]
        return None

    def _invoke_focus(self) -> bool | None:
        callback = self.focus_callback
        if callback is None:
            return None
        # A named-pipe activation arrives on a background thread. Serializing the
        # callback lets the desktop adapter safely marshal focus requests to its
        # GUI thread without overlapping restore/activate operations.
        with self._focus_lock:
            return callback()


def activate_existing(
    address: str,
    opener: Callable[[str], object],
    *,
    timeout: float = 0.0,
    response_timeout: float = 2.0,
    connector: Callable[..., Any] = Client,
    clock: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
) -> bool:
    if response_timeout <= 0:
        raise ValueError("activation response timeout must be positive")
    challenge = secrets.token_urlsafe(18)
    retry_timeout = max(0.0, timeout)
    deadline = clock() + retry_timeout
    while True:
        try:
            with connector(address, family="AF_PIPE", authkey=_AUTHKEY) as connection:
                connection.send(
                    {
                        "protocol": ACTIVATION_PROTOCOL,
                        "app_id": APP_ID,
                        "action": "ACTIVATE",
                        "challenge": challenge,
                    }
                )
                response_deadline = (
                    deadline if retry_timeout > 0 else clock() + response_timeout
                )
                remaining = max(0.0, response_deadline - clock())
                if not connection.poll(remaining):
                    raise ActivationTimeoutError("现有实例未在时限内返回激活响应")
                response = connection.recv()
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            if clock() >= deadline:
                return False
            wait(min(0.05, max(0.0, deadline - clock())))
            continue

        if not isinstance(response, dict):
            raise ActivationProtocolError("现有实例返回了无效的激活响应")
        if (
            response.get("protocol") != ACTIVATION_PROTOCOL
            or response.get("app_id") != APP_ID
            or response.get("challenge") != challenge
        ):
            raise ActivationProtocolError("现有实例激活协议校验失败")

        error = response.get("error")
        if error == "not_ready":
            if clock() >= deadline:
                raise ActivationNotReadyError("现有实例仍在启动，暂时无法激活界面")
            wait(min(0.05, max(0.0, deadline - clock())))
            continue
        if error == "focus_failed":
            raise ActivationFocusError("现有实例已运行，但无法恢复或聚焦桌面窗口")
        if error == "activation_timeout":
            raise ActivationTimeoutError("现有实例激活操作超时")

        if (
            response.get("ok") is not True
        ):
            raise ActivationProtocolError("现有实例激活协议校验失败")

        if response.get("activation") == "desktop":
            if "url" in response:
                raise ActivationProtocolError("现有实例激活协议校验失败")
            return True

        if "activation" in response or not isinstance(response.get("url"), str):
            raise ActivationProtocolError("现有实例激活协议校验失败")
        opener(response["url"])
        return True


def _current_user_sid() -> str:
    if os.name != "nt":
        return f"user:{getpass.getuser()}"
    import ctypes
    from ctypes import wintypes

    TOKEN_QUERY = 0x0008
    TokenUser = 1

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]

    class TOKEN_USER(ctypes.Structure):
        _fields_ = [("User", SID_AND_ATTRIBUTES)]

    kernel32 = ctypes.windll.kernel32
    advapi32 = ctypes.windll.advapi32
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_uint,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)
    ):
        raise ctypes.WinError()
    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, TokenUser, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token,
            TokenUser,
            buffer,
            needed,
            ctypes.byref(needed),
        ):
            raise ctypes.WinError()
        token_user = ctypes.cast(buffer, ctypes.POINTER(TOKEN_USER)).contents
        sid_text = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(token_user.User.Sid, ctypes.byref(sid_text)):
            raise ctypes.WinError()
        try:
            return sid_text.value
        finally:
            kernel32.LocalFree(sid_text)
    finally:
        kernel32.CloseHandle(token)


def _current_session_id() -> int:
    if os.name != "nt":
        return 0
    import ctypes
    from ctypes import wintypes

    session_id = wintypes.DWORD()
    if not ctypes.windll.kernel32.ProcessIdToSessionId(
        os.getpid(), ctypes.byref(session_id)
    ):
        raise ctypes.WinError()
    return int(session_id.value)


__all__ = [
    "ACTIVATION_PROTOCOL",
    "ActivationFocusError",
    "ActivationNotReadyError",
    "ActivationProtocolError",
    "ActivationTimeoutError",
    "ActivationServer",
    "InstanceNames",
    "activate_existing",
    "current_instance_names",
    "handle_activation_request",
]
