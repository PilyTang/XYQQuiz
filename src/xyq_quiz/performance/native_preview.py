from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class NativePreviewSelfTest:
    adapter_id: int
    adapter_name: str
    encoder_name: str
    feature_level: int
    encoded_bytes: int


def locate_native_preview_helper(
    *,
    app_root: Path,
    resource_root: Path,
    frozen: bool,
) -> Path:
    candidates = (
        (resource_root / "native" / "XYQPreviewHelper.exe",)
        if frozen
        else (
            app_root / "build" / "native" / "XYQPreviewHelper.exe",
            resource_root / "native" / "XYQPreviewHelper.exe",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def run_native_preview_self_test(
    helper_path: Path,
    adapter_id: int,
    *,
    timeout: float = 15.0,
    runner: Callable[..., Any] = subprocess.run,
) -> NativePreviewSelfTest:
    if adapter_id < 0:
        raise ValueError("adapter_id must be non-negative")
    helper = Path(helper_path).resolve()
    if not helper.is_file():
        raise FileNotFoundError(f"硬件预览辅助进程不存在：{helper}")
    creation_flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creation_flags = subprocess.CREATE_NO_WINDOW
    completed = runner(
        [str(helper), "--self-test", "--adapter", str(adapter_id)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        creationflags=creation_flags,
    )
    stdout = str(getattr(completed, "stdout", ""))
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        stderr = str(getattr(completed, "stderr", "")).strip()
        raise RuntimeError(
            "硬件预览自检没有返回结果"
            + (f"：{stderr}" if stderr else "")
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError("硬件预览自检返回了无效 JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("硬件预览自检返回值不是对象")
    if getattr(completed, "returncode", 1) != 0 or payload.get("ok") is not True:
        reason = payload.get("error")
        raise RuntimeError(
            str(reason) if isinstance(reason, str) else "硬件预览自检失败"
        )
    result = NativePreviewSelfTest(
        adapter_id=_required_int(payload, "adapter_id"),
        adapter_name=_required_string(payload, "adapter_name"),
        encoder_name=_required_string(payload, "encoder_name"),
        feature_level=_required_int(payload, "feature_level"),
        encoded_bytes=_required_int(payload, "encoded_bytes"),
    )
    if result.adapter_id != adapter_id:
        raise RuntimeError("硬件预览自检返回了错误的设备 ID")
    if result.encoded_bytes <= 0:
        raise RuntimeError("硬件预览自检没有生成 H.264 数据")
    return result


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"硬件预览自检缺少 {field}")
    return value


def _required_int(payload: dict[str, object], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"硬件预览自检缺少 {field}")
    return value


__all__ = [
    "NativePreviewSelfTest",
    "locate_native_preview_helper",
    "run_native_preview_self_test",
]
