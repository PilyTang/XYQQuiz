from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import getpass
import hashlib
import os
import secrets
import threading
import time
from typing import Any

from multiprocessing.connection import Client, Listener

from xyq_quiz.web.security import APP_ID


ACTIVATION_PROTOCOL = 1
_AUTHKEY = b"XYQQuiz-Activation-v1"


class ActivationProtocolError(RuntimeError):
    pass


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
    url_factory: Callable[[], str | None],
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
    url = url_factory()
    if url is None:
        return {
            "ok": False,
            "error": "not_ready",
            "protocol": ACTIVATION_PROTOCOL,
            "app_id": APP_ID,
            "challenge": challenge,
        }
    return {
        "ok": True,
        "protocol": ACTIVATION_PROTOCOL,
        "app_id": APP_ID,
        "challenge": challenge,
        "url": url,
    }


class ActivationServer:
    def __init__(
        self,
        address: str,
        url_factory: Callable[[], str | None],
        *,
        listener_factory: Callable[..., Any] = Listener,
        connector: Callable[..., Any] = Client,
    ) -> None:
        self.address = address
        self.url_factory = url_factory
        self._listener_factory = listener_factory
        self._connector = connector
        self._listener: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

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
        try:
            with self._connector(
                self.address,
                family="AF_PIPE",
                authkey=_AUTHKEY,
            ) as connection:
                connection.send({"action": "STOP"})
        except OSError:
            pass
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
                except (OSError, EOFError):
                    if self._stop.is_set():
                        break
                    continue
                with connection:
                    try:
                        request = connection.recv()
                        if request == {"action": "STOP"} and self._stop.is_set():
                            break
                        connection.send(
                            handle_activation_request(request, self.url_factory)
                        )
                    except (EOFError, OSError):
                        continue
        finally:
            listener.close()


def activate_existing(
    address: str,
    opener: Callable[[str], object],
    *,
    timeout: float = 0.0,
    connector: Callable[..., Any] = Client,
    clock: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
) -> bool:
    challenge = secrets.token_urlsafe(18)
    deadline = clock() + max(0.0, timeout)
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
                response = connection.recv()
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            if clock() >= deadline:
                return False
            wait(min(0.05, max(0.0, deadline - clock())))
            continue

        if not isinstance(response, dict):
            raise ActivationProtocolError("现有实例返回了无效的激活响应")
        if response.get("error") == "not_ready":
            if clock() >= deadline:
                raise ActivationProtocolError("现有实例仍在启动，暂时无法打开页面")
            wait(min(0.05, max(0.0, deadline - clock())))
            continue
        if (
            response.get("ok") is not True
            or response.get("protocol") != ACTIVATION_PROTOCOL
            or response.get("app_id") != APP_ID
            or response.get("challenge") != challenge
            or not isinstance(response.get("url"), str)
        ):
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
    "ActivationProtocolError",
    "ActivationServer",
    "InstanceNames",
    "activate_existing",
    "current_instance_names",
    "handle_activation_request",
]
