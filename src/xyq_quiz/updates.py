"""Optional, credential-free update discovery; never installs or executes files."""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from uuid import uuid4

import httpx

from xyq_quiz import __version__

REPOSITORY = "https://cnb.cool/pilytang/XYQQuiz"
MANIFEST_URL = REPOSITORY + "/-/git/raw/main/latest.json"
MAX_BYTES = 32768
INTERVAL = 86400


def version_tuple(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,5}\.\d{1,5}\.\d{1,5}", value):
        raise ValueError("仅接受正式版版本号")
    return tuple(map(int, value.split(".")))


def validate_manifest(value: object) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("版本信息格式不正确")
    version = value.get("version")
    version_tuple(version)
    filename = f"XYQQuiz-v{version}-win10-win11-x64.zip"
    expected = f"{REPOSITORY}/-/releases/download/v{version}/{filename}"
    if value.get("download_url") != expected:
        raise ValueError("下载地址不属于项目发行版")
    digest = value.get("sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("校验值格式不正确")
    notes = value.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 8000:
        raise ValueError("更新说明过长")
    return {"schema_version": 1, "version": version, "download_url": expected,
            "sha256": digest, "notes": notes}


class UpdateChecker:
    def __init__(self, state_path: Path, *, transport=None, clock=time.time):
        self.state_path = state_path
        self.transport = transport
        self.clock = clock
        self.lock = asyncio.Lock()
        self.enabled = True
        self.attempted_at = 0.0
        self.checked_at = 0.0
        self.latest = None
        self.error = ""
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if type(state.get("enabled")) is bool:
                self.enabled = state["enabled"]
            for key in ("attempted_at", "checked_at"):
                stamp = state.get(key, 0)
                if isinstance(stamp, (int, float)) and 0 <= stamp <= self.clock():
                    setattr(self, key, stamp)
            if state.get("latest") is not None:
                self.latest = validate_manifest(state["latest"])
            if state.get("error"):
                self.error = "上次检查未成功，可手动重试"
        except (OSError, ValueError, TypeError, AttributeError):
            self.latest = None

    def snapshot(self) -> dict:
        available = bool(self.latest and version_tuple(self.latest["version"]) > version_tuple(__version__))
        return {"current_version": __version__, "enabled": self.enabled,
                "checked_at": self.checked_at, "attempted_at": self.attempted_at,
                "latest": self.latest, "available": available, "error": self.error,
                "status": "unavailable" if self.error else (
                    "available" if available else "current" if self.latest else "unchecked")}

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".update-{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps({"enabled": self.enabled,
                "attempted_at": self.attempted_at, "checked_at": self.checked_at,
                "latest": self.latest, "error": self.error}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    async def set_enabled(self, enabled: bool) -> dict:
        async with self.lock:
            previous = self.enabled
            self.enabled = enabled
            try:
                self._save()
            except OSError:
                self.enabled = previous
                raise
            return self.snapshot()

    async def _fetch(self) -> dict:
        async with httpx.AsyncClient(transport=self.transport, timeout=6,
                                    follow_redirects=True, max_redirects=3) as client:
            async with client.stream("GET", MANIFEST_URL, headers={
                "Accept": "application/json", "User-Agent": "XYQQuiz-update-check",
            }) as response:
                response.raise_for_status()
                if response.url.scheme != "https":
                    raise ValueError("不安全的版本信息地址")
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_BYTES:
                        raise ValueError("版本信息过大")
                return validate_manifest(json.loads(content))

    async def check(self, *, manual: bool = False) -> dict:
        async with self.lock:
            now = self.clock()
            if not manual and (not self.enabled or (
                self.attempted_at and 0 <= now - self.attempted_at < INTERVAL
            )):
                return self.snapshot()
            self.attempted_at = now
            try:
                latest = await asyncio.wait_for(self._fetch(), timeout=12)
                if self.latest and version_tuple(latest["version"]) < version_tuple(self.latest["version"]):
                    raise ValueError("版本源暂未同步")
                self.latest = latest
                self.checked_at = self.clock()
                self.error = ""
            except (httpx.HTTPError, ValueError, TypeError, TimeoutError):
                self.error = "暂时无法检查更新，请稍后重试"
            try:
                self._save()
            except OSError:
                self.error = "无法保存更新检查记录，请检查目录写入权限"
            return self.snapshot()
