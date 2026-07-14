from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import struct
import time
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from xyq_quiz import __version__
from xyq_quiz.config import AppConfig
from xyq_quiz.capture.wgc import WGCCapture
from xyq_quiz.knowledge.updater import load_current_generation
from xyq_quiz.recognition.ocr import RapidOCREngine
from xyq_quiz.runtime.paths import RuntimePaths
from xyq_quiz.web.security import LocalWebSecurity


CheckStatus = Literal["PASS", "FAIL", "SKIPPED"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    duration_ms: float


@dataclass(frozen=True, slots=True)
class SelfTestReport:
    schema_version: int
    app_version: str
    generated_at: str
    headless: bool
    ok: bool
    checks: tuple[CheckResult, ...]


def run_self_test(
    paths: RuntimePaths,
    *,
    config_path: Path,
    report_dir: Path,
    headless: bool,
    checks: Sequence[tuple[str, Callable[[], str]]] | None = None,
) -> tuple[SelfTestReport, Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(checks or _default_checks(paths, config_path, headless))
    results: list[CheckResult] = []
    for name, check in selected:
        started = time.perf_counter_ns()
        try:
            detail = check()
        except _SkipCheck as skipped:
            status: CheckStatus = "SKIPPED"
            detail = str(skipped)
        except Exception as error:
            status = "FAIL"
            detail = f"{type(error).__name__}: {error}"
        else:
            status = "PASS"
        results.append(
            CheckResult(
                name=name,
                status=status,
                detail=detail,
                duration_ms=(time.perf_counter_ns() - started) / 1_000_000,
            )
        )

    report = SelfTestReport(
        schema_version=1,
        app_version=__version__,
        generated_at=datetime.now(timezone.utc).isoformat(),
        headless=headless,
        ok=all(result.status != "FAIL" for result in results),
        checks=tuple(results),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    json_path = report_dir / f"self-test-{stamp}.json"
    html_path = report_dir / f"self-test-{stamp}.html"
    _write_atomic(json_path, _report_json(report))
    _write_atomic(html_path, _report_html(report).encode("utf-8"))
    return report, json_path, html_path


def version_payload(paths: RuntimePaths) -> dict[str, object]:
    manifest_path = paths.resource_root / "build-manifest.json"
    commit = "source-tree"
    signed = False
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict):
            commit = str(manifest.get("git_commit", "unknown"))
            signed = bool(manifest.get("signed", False))
    return {
        "schema_version": 1,
        "app_id": "xyq-quiz",
        "version": __version__,
        "commit": commit,
        "target": "Windows 10 1903+ / Windows 11 x64",
        "signed": signed,
    }


def write_version_report(paths: RuntimePaths, report_dir: Path) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "version.json"
    payload = (json.dumps(version_payload(paths), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    _write_atomic(path, payload)
    return path


def verify_build_manifest(
    package_root: Path,
    *,
    manifest_path: Path | None = None,
    strict_extra: bool = True,
) -> str:
    package_root = Path(package_root).resolve()
    manifest_path = Path(manifest_path or (package_root / "build-manifest.json")).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"构建清单缺失：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("build-manifest.json schema 无效")
    required_strings = ("app_version", "git_commit", "built_at", "build_system", "target")
    for name in required_strings:
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"build-manifest.json 缺少字段：{name}")
    if not isinstance(manifest.get("signed"), bool):
        raise ValueError("build-manifest.json signed 必须是布尔值")
    if not isinstance(manifest.get("dependencies"), dict):
        raise ValueError("build-manifest.json dependencies 必须是对象")
    if not isinstance(manifest.get("verified_platforms"), list):
        raise ValueError("build-manifest.json verified_platforms 必须是数组")
    question_bank = manifest.get("question_bank")
    if not isinstance(question_bank, dict):
        raise ValueError("build-manifest.json question_bank 必须是对象")
    for name in ("generation_id", "source_url", "retrieved_at", "sha256"):
        if not isinstance(question_bank.get(name), str) or not question_bank[name]:
            raise ValueError(f"build-manifest.json question_bank 缺少字段：{name}")
    record_count = question_bank.get("record_count")
    if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 1:
        raise ValueError("build-manifest.json question_bank record_count 无效")
    if not re.fullmatch(r"[0-9a-f]{64}", question_bank["sha256"]):
        raise ValueError("build-manifest.json question_bank sha256 无效")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("build-manifest.json files 必须是数组")
    seen: set[str] = set()
    listed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("构建清单文件项必须是对象")
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise ValueError("构建清单路径必须是字符串")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative != pure.as_posix():
            raise ValueError(f"构建清单包含不安全路径：{relative}")
        folded = relative.casefold()
        if folded in seen:
            raise ValueError(f"构建清单包含重复路径：{relative}")
        seen.add(folded)
        listed.add(relative)
        path = package_root.joinpath(*pure.parts)
        payload = path.read_bytes()
        if entry.get("size") != len(payload):
            raise ValueError(f"文件大小不匹配：{relative}")
        digest = hashlib.sha256(payload).hexdigest()
        if entry.get("sha256") != digest:
            raise ValueError(f"文件 SHA-256 不匹配：{relative}")

    if strict_extra:
        mutable_roots = {"config.json", "data", "logs", "diagnostics", "state-backups"}
        actual = set()
        for path in package_root.rglob("*"):
            if not path.is_file() or path == manifest_path:
                continue
            relative = path.relative_to(package_root)
            if relative.parts and relative.parts[0] in mutable_roots:
                continue
            actual.add(relative.as_posix())
        extras = sorted(actual - listed)
        if extras:
            raise ValueError(f"_internal 存在未声明文件：{extras[0]}")
    return f"已校验 {len(entries)} 个不可变文件"


class _SkipCheck(RuntimeError):
    pass


def _default_checks(
    paths: RuntimePaths,
    config_path: Path,
    headless: bool,
) -> tuple[tuple[str, Callable[[], str]], ...]:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("platform", _check_platform),
        ("portable-write", lambda: _check_writable(paths.app_root)),
        (
            "build-manifest",
            lambda: _check_manifest_for_runtime(paths),
        ),
        ("portable-state", lambda: _check_state(config_path)),
        ("native-imports", _check_native_imports),
        ("ocr-inference", _check_ocr_inference),
        ("local-web-security", _check_web_security),
    ]
    if headless:
        checks.append(("synthetic-wgc", lambda: _skip("headless 模式无交互桌面")))
    else:
        checks.append(("synthetic-wgc", _check_synthetic_wgc))
    return tuple(checks)


def _check_platform() -> str:
    if platform.system() != "Windows":
        raise RuntimeError("只支持 Windows")
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("只支持 64 位进程")
    build = int(platform.version().split(".")[-1])
    if build < 18362:
        raise RuntimeError(f"Windows build {build} 低于 18362")
    return f"Windows build {build}, x64"


def _check_writable(root: Path) -> str:
    probe = Path(root) / f".xyqquiz-write-probe-{uuid4().hex}.tmp"
    try:
        with probe.open("xb") as handle:
            handle.write(b"XYQQuiz")
            handle.flush()
            os.fsync(handle.fileno())
        if probe.read_bytes() != b"XYQQuiz":
            raise RuntimeError("写入探针内容不一致")
    finally:
        probe.unlink(missing_ok=True)
    return "便携根目录可创建、刷新、读取和删除文件"


def _check_manifest_for_runtime(paths: RuntimePaths) -> str:
    if not paths.frozen:
        raise _SkipCheck("源码运行不要求 PyInstaller 构建清单")
    return verify_build_manifest(
        paths.app_root,
        manifest_path=paths.resource_root / "build-manifest.json",
    )


def _check_state(config_path: Path) -> str:
    config = AppConfig.load(config_path)
    generation = load_current_generation(config.data_dir)
    return f"配置与题库可用，generation={generation.generation_id}, records={generation.question_bank.count}"


def _check_native_imports() -> str:
    modules = ("cv2", "onnxruntime", "rapidocr", "windows_capture", "websockets")
    for name in modules:
        importlib.import_module(name)
    return "原生与 WebSocket 运行依赖均可导入"


def _check_ocr_inference() -> str:
    image = np.full((96, 320, 3), 255, dtype=np.uint8)
    result = RapidOCREngine().recognize(image)
    return f"RapidOCR 推理完成，text_length={len(result.text)}"


def _check_web_security() -> str:
    security = LocalWebSecurity("127.0.0.1", 8765, process_token="self-test-token")
    url = security.issue_browser_url(security.expected_origin)
    bootstrap = url.split("#token=", 1)[1]
    process = security.consume_bootstrap(bootstrap)
    if process != "self-test-token" or security.consume_bootstrap(bootstrap) is not None:
        raise RuntimeError("一次性引导令牌没有正确失效")
    if not security.validate_process_token(process):
        raise RuntimeError("进程令牌校验失败")
    browser_session = security.issue_browser_session()
    if security.restore_browser_session(browser_session) != "self-test-token":
        raise RuntimeError("浏览器会话恢复失败")
    restarted = LocalWebSecurity(
        "127.0.0.1",
        8765,
        process_token="restarted-token",
    )
    if restarted.restore_browser_session(browser_session) is not None:
        raise RuntimeError("后台重启后旧浏览器会话没有失效")
    return "一次性引导、浏览器恢复与进程令牌边界可用"


def _check_synthetic_wgc() -> str:
    if os.name != "nt":
        raise _SkipCheck("WGC 只在 Windows 桌面可用")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.UpdateWindow.argtypes = [wintypes.HWND]
    user32.UpdateWindow.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.FillRect.argtypes = [
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.HBRUSH,
    ]
    user32.FillRect.restype = ctypes.c_int
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.PeekMessageW.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.TranslateMessage.restype = wintypes.BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = ctypes.c_ssize_t
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
    gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL

    WS_OVERLAPPEDWINDOW = 0x00CF0000
    WS_VISIBLE = 0x10000000
    SW_SHOW = 5
    PM_REMOVE = 0x0001
    hwnd = user32.CreateWindowExW(
        0,
        "STATIC",
        "XYQQuiz WGC Self-Test",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        100,
        100,
        420,
        260,
        None,
        None,
        None,
        None,
    )
    if not hwnd:
        raise ctypes.WinError()
    capture = WGCCapture()
    try:
        user32.ShowWindow(hwnd, SW_SHOW)
        user32.UpdateWindow(hwnd)
        if not capture.start(int(hwnd)):
            raise RuntimeError("WGC 未能启动合成窗口捕获")
        deadline = time.monotonic() + 8.0
        message = wintypes.MSG()
        while time.monotonic() < deadline:
            while user32.PeekMessageW(
                ctypes.byref(message), None, 0, 0, PM_REMOVE
            ):
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
            rect = wintypes.RECT()
            if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                dc = user32.GetDC(hwnd)
                brush = gdi32.CreateSolidBrush(0x000000FF)
                try:
                    user32.FillRect(dc, ctypes.byref(rect), brush)
                finally:
                    gdi32.DeleteObject(brush)
                    user32.ReleaseDC(hwnd, dc)
            frame = capture.latest()
            if frame is not None:
                height, width = frame.bgr.shape[:2]
                red_peak = int(frame.bgr[:, :, 2].max())
                if width >= 300 and height >= 150 and red_peak >= 180:
                    return f"WGC 捕获 {width}x{height}，已检测到合成红色区域"
            time.sleep(0.05)
        raise RuntimeError("8 秒内未取得有效的合成 WGC 彩色帧")
    finally:
        capture.close()
        user32.DestroyWindow(hwnd)


def _skip(message: str) -> str:
    raise _SkipCheck(message)


def _report_json(report: SelfTestReport) -> bytes:
    return (
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _report_html(report: SelfTestReport) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(check.name)}</td>"
        f"<td>{check.status}</td>"
        f"<td>{check.duration_ms:.1f} ms</td>"
        f"<td>{html.escape(check.detail)}</td>"
        "</tr>"
        for check in report.checks
    )
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<title>XYQQuiz 自检报告</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:.45rem;text-align:left}</style>"
        f"<h1>XYQQuiz 自检：{'通过' if report.ok else '失败'}</h1>"
        f"<p>版本 {html.escape(report.app_version)} · {html.escape(report.generated_at)}</p>"
        "<table><thead><tr><th>项目</th><th>结果</th><th>耗时</th><th>说明</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></html>"
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CheckResult",
    "SelfTestReport",
    "run_self_test",
    "verify_build_manifest",
    "version_payload",
    "write_version_report",
]
