from __future__ import annotations

from dataclasses import dataclass
import secrets
import threading
import time
from typing import Callable
from urllib.parse import quote


APP_ID = "xyq-quiz"
TOKEN_HEADER = "X-XYQQuiz-Token"
SESSION_COOKIE = "xyq_quiz_browser_session"


@dataclass(frozen=True, slots=True)
class BoundaryDecision:
    allowed: bool
    status_code: int = 200
    message: str = ""


class LocalWebSecurity:
    """Process-local browser authentication for the loopback web UI."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        process_token: str | None = None,
        bootstrap_ttl: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("the production web boundary must use 127.0.0.1")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if bootstrap_ttl <= 0:
            raise ValueError("bootstrap_ttl must be positive")
        self.host = host
        self.port = port
        self.expected_host = f"{host}:{port}"
        self.expected_origin = f"http://{host}:{port}"
        self._process_token = process_token or secrets.token_urlsafe(32)
        self._bootstrap_ttl = bootstrap_ttl
        self._clock = clock
        self._bootstrap_tokens: dict[str, float] = {}
        self._browser_sessions: set[str] = set()
        self._lock = threading.Lock()

    def issue_browser_url(self, base_url: str) -> str:
        if base_url.rstrip("/") != self.expected_origin:
            raise ValueError("browser URL does not match the configured loopback origin")
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._purge_expired_locked()
            self._bootstrap_tokens[token] = self._clock() + self._bootstrap_ttl
        return f"{self.expected_origin}/#token={quote(token, safe='')}"

    def consume_bootstrap(self, token: object) -> str | None:
        if not isinstance(token, str) or not token:
            return None
        with self._lock:
            self._purge_expired_locked()
            expires_at = self._bootstrap_tokens.pop(token, None)
        if expires_at is None or expires_at < self._clock():
            return None
        return self._process_token

    def validate_process_token(self, token: object) -> bool:
        return isinstance(token, str) and secrets.compare_digest(
            token,
            self._process_token,
        )

    def issue_browser_session(self) -> str:
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._browser_sessions.add(token)
        return token

    def restore_browser_session(self, token: object) -> str | None:
        if not isinstance(token, str) or not token:
            return None
        with self._lock:
            valid = any(
                secrets.compare_digest(token, candidate)
                for candidate in self._browser_sessions
            )
        return self._process_token if valid else None

    def validate_host(self, host: object) -> BoundaryDecision:
        if not isinstance(host, str) or not secrets.compare_digest(
            host.lower(),
            self.expected_host,
        ):
            return BoundaryDecision(False, 400, "无效的本机服务 Host")
        return BoundaryDecision(True)

    def validate_origin(self, origin: object) -> BoundaryDecision:
        if not isinstance(origin, str) or not secrets.compare_digest(
            origin,
            self.expected_origin,
        ):
            return BoundaryDecision(False, 403, "请求来源不受信任")
        return BoundaryDecision(True)

    def authorize_http(
        self,
        *,
        host: object,
        origin: object,
        token: object,
        bootstrap: bool = False,
    ) -> BoundaryDecision:
        host_decision = self.validate_host(host)
        if not host_decision.allowed:
            return host_decision
        origin_decision = self.validate_origin(origin)
        if not origin_decision.allowed:
            return origin_decision
        if not bootstrap and not self.validate_process_token(token):
            return BoundaryDecision(False, 403, "本机会话令牌无效")
        return BoundaryDecision(True)

    def authorize_websocket(
        self,
        *,
        host: object,
        origin: object,
    ) -> BoundaryDecision:
        host_decision = self.validate_host(host)
        if not host_decision.allowed:
            return host_decision
        return self.validate_origin(origin)

    def _purge_expired_locked(self) -> None:
        now = self._clock()
        expired = [
            token
            for token, expires_at in self._bootstrap_tokens.items()
            if expires_at < now
        ]
        for token in expired:
            self._bootstrap_tokens.pop(token, None)


__all__ = [
    "APP_ID",
    "BoundaryDecision",
    "LocalWebSecurity",
    "SESSION_COOKIE",
    "TOKEN_HEADER",
]
