from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import ctypes
from ctypes import wintypes
from functools import partial
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import TracebackType
from typing import Any
from urllib.request import urlopen
import webbrowser


_NULL_STREAMS: list[Any] = []


def bootstrap_frozen_stdio() -> None:
    if not bool(getattr(sys, "frozen", False)):
        return
    if sys.stdin is None:
        stream = open(os.devnull, "r", encoding="utf-8")
        _NULL_STREAMS.append(stream)
        sys.stdin = stream
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            stream = open(os.devnull, "w", encoding="utf-8")
            _NULL_STREAMS.append(stream)
            setattr(sys, name, stream)


bootstrap_frozen_stdio()

import uvicorn

from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.service import CaptureService
from xyq_quiz.config import AppConfig
from xyq_quiz.diagnostics import DiagnosticWriter, EnvironmentDiagnosticWriter
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.updater import QuestionBankUpdater, load_current_generation
from xyq_quiz.recognition.layout import (
    LayoutProfile,
    build_layout_detector,
    validate_anchor_templates,
)
from xyq_quiz.recognition.ocr import RapidOCREngine
from xyq_quiz.recognition.pipeline import RecognitionPipeline
from xyq_quiz.runtime.coordinator import RecognitionCoordinator
from xyq_quiz.runtime.activation import (
    ActivationProtocolError,
    ActivationServer,
    activate_existing,
    current_instance_names,
)
from xyq_quiz.runtime.paths import RuntimePaths, initialize_portable_state
from xyq_quiz.runtime.portable import migrate_portable_state
from xyq_quiz.runtime.state import RuntimeStore
from xyq_quiz.selftest import run_self_test, version_payload, write_version_report
from xyq_quiz.web.app import Services, create_app
from xyq_quiz.web.security import APP_ID, LocalWebSecurity


SW_SHOWNORMAL = 1
ERROR_ALREADY_EXISTS = 183
DEFAULT_MUTEX_NAME = "Local\\XYQQuizBackend"


class ElevationError(RuntimeError):
    pass


class AlreadyRunningError(RuntimeError):
    pass


class StartupAssetError(RuntimeError):
    pass


class PortConflictError(RuntimeError):
    pass


def validate_recognition_assets(layout_path: Path) -> LayoutProfile:
    path = Path(layout_path)
    if not path.is_file():
        raise StartupAssetError(
            f"尚未生成真实科举布局：{path}。请等待真实截图并运行 xyq-quiz-calibrate；不会生成占位 profile/anchor。"
        )
    try:
        profile = LayoutProfile.load(path)
    except (OSError, ValueError) as exc:
        raise StartupAssetError(f"真实科举布局不可用：{path}：{exc}") from exc
    missing = [anchor.template_path for anchor in profile.anchors if not anchor.template_path.is_file()]
    if missing:
        raise StartupAssetError(
            "真实科举 anchor 缺失：" + ", ".join(str(item) for item in missing)
        )
    try:
        validate_anchor_templates(profile)
    except ValueError as exc:
        raise StartupAssetError(str(exc)) from exc
    return profile


def validate_recognition_asset_bundle(
    layout_paths: Sequence[Path],
) -> tuple[LayoutProfile, ...]:
    if not layout_paths:
        raise StartupAssetError("尚未配置真实科举布局。")
    return tuple(validate_recognition_assets(path) for path in layout_paths)


def ensure_elevated(
    argv: Sequence[str],
    shell_execute: Callable[[Any, str, str, str, str, int], int] | None = None,
    *,
    is_admin: Callable[[], bool] | None = None,
) -> bool:
    """Continue as admin, or relaunch this module elevated and stop the parent."""
    admin_check = is_admin or _is_user_an_admin
    if admin_check():
        return True
    execute = shell_execute or ctypes.windll.shell32.ShellExecuteW
    if bool(getattr(sys, "frozen", False)):
        executable = sys.executable
        relaunch_args = list(argv)
    else:
        executable = sys.executable
        relaunch_args = ["-m", "xyq_quiz.launcher", *argv]
    parameters = subprocess.list2cmdline(relaunch_args)
    result = int(
        execute(
            None,
            "runas",
            executable,
            parameters,
            str(Path.cwd()),
            SW_SHOWNORMAL,
        )
    )
    if result <= 32:
        raise ElevationError(
            f"管理员启动失败：ShellExecuteW 返回 {result}（需要允许 UAC 提权）"
        )
    return False


def _is_user_an_admin() -> bool:
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


class SingleInstance:
    def __init__(self, name: str = DEFAULT_MUTEX_NAME, *, kernel32: Any = None) -> None:
        self.name = name
        self._kernel32 = kernel32
        self._handle: Any = None

    def __enter__(self) -> SingleInstance:
        kernel32 = self._kernel32 or ctypes.windll.kernel32
        create_mutex = kernel32.CreateMutexW
        if hasattr(create_mutex, "argtypes"):
            create_mutex.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
            create_mutex.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        if hasattr(close_handle, "argtypes"):
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
        handle = create_mutex(None, False, self.name)
        error = int(kernel32.GetLastError())
        if not handle:
            raise OSError(error, f"创建单实例互斥量失败（Windows error {error}）")
        self._kernel32 = kernel32
        self._handle = handle
        if error == ERROR_ALREADY_EXISTS:
            self.close()
            raise AlreadyRunningError("XYQ Quiz backend is already running")
        return self

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            if not self._kernel32.CloseHandle(handle):
                error = int(self._kernel32.GetLastError())
                raise OSError(
                    error,
                    f"关闭单实例互斥量失败（Windows error {error}）",
                )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def configure_logging(log_path: Path) -> RotatingFileHandler:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return handler


def reserve_loopback_port(
    host: str,
    port: int,
    *,
    socket_factory: Callable[..., socket.socket] = socket.socket,
) -> socket.socket:
    if host != "127.0.0.1":
        raise ValueError("web host must be 127.0.0.1")
    reserved = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
    try:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            reserved.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        reserved.bind((host, port))
        return reserved
    except OSError as error:
        reserved.close()
        if error.errno in {98, 10048} or getattr(error, "winerror", None) == 10048:
            raise PortConflictError(
                f"本机端口 {port} 已被其他程序占用。\n\n"
                "请关闭占用该端口的程序后重试，或修改 config.json 中的 web.port。"
            ) from error
        raise


def wait_for_health_and_open(
    server: Any,
    url: str,
    cancel: threading.Event,
    gate: threading.Lock | None = None,
    *,
    health_check: Callable[[str], bool] | None = None,
    opener: Callable[[str], Any] = webbrowser.open,
    wait: Callable[[float], bool] | None = None,
    timeout: float = 30.0,
    browser_url_factory: Callable[[], str] | None = None,
) -> bool:
    check = health_check or _http_is_ready
    wait_for_cancel = wait or cancel.wait
    open_gate = gate or threading.Lock()
    open_claimed = False
    deadline = time.monotonic() + timeout
    while not cancel.is_set():
        if server.should_exit:
            return False
        if server.started and check(url):
            if not open_gate.acquire(timeout=0.1):
                return False
            try:
                if (
                    open_claimed
                    or cancel.is_set()
                    or server.should_exit
                    or not server.started
                ):
                    return False
                open_claimed = True
            finally:
                open_gate.release()
            opener(browser_url_factory() if browser_url_factory is not None else url)
            return True
        if time.monotonic() >= deadline:
            return False
        if wait_for_cancel(0.1):
            return False
    return False


def run_server_with_browser(
    server: Any,
    url: str,
    *,
    waiter: Callable[..., bool] = wait_for_health_and_open,
    thread_factory: Callable[..., Any] = threading.Thread,
    join_timeout: float = 1.0,
    browser_url_factory: Callable[[], str] | None = None,
    sockets: list[socket.socket] | None = None,
) -> None:
    cancel = threading.Event()
    gate = threading.Lock()
    thread_target = (
        waiter
        if browser_url_factory is None
        else partial(waiter, browser_url_factory=browser_url_factory)
    )
    browser_thread = thread_factory(
        target=thread_target,
        args=(server, url, cancel, gate),
        name="xyq-quiz-browser",
        daemon=True,
    )
    browser_thread.start()
    try:
        if sockets is None:
            server.run()
        else:
            server.run(sockets=sockets)
    finally:
        cancel.set()
        browser_thread.join(timeout=join_timeout)


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


def build_services(
    config: AppConfig,
    *,
    runtime_paths: RuntimePaths | None = None,
) -> Services:
    """Build one fresh, single-use service graph for one Uvicorn lifespan."""
    layout_profiles = validate_recognition_asset_bundle(
        config.effective_layout_paths
    )
    current = load_current_generation(config.data_dir)
    match = config.match
    matcher = QuestionMatcher(
        current.question_bank,
        match.question_score,
        match.question_gap,
        match.option_score,
    )
    layout_detector = build_layout_detector(layout_profiles)
    pipeline = RecognitionPipeline(
        layout_detector,
        RapidOCREngine(),
        matcher,
    )
    hub = LatestFrameHub()
    runtime = RuntimeStore()
    capture = CaptureService(config, hub)
    coordinator = RecognitionCoordinator(
        capture,
        hub,
        layout_detector,
        pipeline,
        runtime,
    )
    paths = runtime_paths or RuntimePaths.discover()
    return Services(
        hub=hub,
        runtime=runtime,
        capture=capture,
        coordinator=coordinator,
        pipeline=pipeline,
        updater=QuestionBankUpdater(config.data_dir),
        match_config=config.match,
        diagnostic_writer=DiagnosticWriter(
            config.diagnostics_dir,
            log_path=config.log_path,
            app_root=paths.app_root,
        ),
        environment_diagnostic_writer=EnvironmentDiagnosticWriter(
            config.diagnostics_dir,
            app_root=paths.app_root,
            resource_root=paths.resource_root,
            log_path=config.log_path,
        ),
        diagnostic_config=config,
        diagnostic_metadata=current.metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="启动梦幻西游科举答题助手")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-dialog", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--elevated-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(raw_args)

    if args.headless and not args.self_test:
        parser.error("--headless 只能与 --self-test 一起使用")
    if args.no_dialog and not args.self_test:
        parser.error("--no-dialog 只能与 --self-test 一起使用")
    paths = RuntimePaths.discover()
    if args.version:
        if args.report_dir is not None:
            write_version_report(paths, args.report_dir)
        else:
            payload = version_payload(paths)
            _show_message(
                "XYQQuiz 版本",
                "\n".join(
                    (
                        f"版本：{payload['version']}",
                        f"Commit：{payload['commit']}",
                        f"目标：{payload['target']}",
                        f"签名：{'是' if payload['signed'] else '否'}",
                    )
                ),
            )
        return 0
    if args.self_test:
        if args.report_dir is None:
            _show_message("XYQQuiz 自检", "--self-test 必须同时指定 --report-dir", error=True)
            return 2
        try:
            if paths.frozen:
                initialize_portable_state(paths)
                migrate_portable_state(paths.config_path, paths.data_dir)
            config_path = args.config
            if config_path is None:
                config_path = paths.config_path if paths.config_path.is_file() else paths.default_config_path
            report, _json_path, html_path = run_self_test(
                paths,
                config_path=config_path,
                report_dir=args.report_dir,
                headless=args.headless,
            )
        except Exception as error:
            _show_message("XYQQuiz 自检失败", str(error), error=True)
            return 1
        if not args.headless and not args.no_dialog:
            _show_message(
                "XYQQuiz 自检",
                f"{'通过' if report.ok else '失败'}\n报告：{html_path}",
                error=not report.ok,
            )
        return 0 if report.ok else 1

    names = current_instance_names()
    try:
        if activate_existing(names.pipe, webbrowser.open):
            return 0
    except ActivationProtocolError as error:
        _show_message("XYQQuiz 启动失败", str(error), error=True)
        return 3

    if args.elevated_child and not _is_user_an_admin():
        _show_message("XYQQuiz 启动失败", "提权子进程没有管理员权限", error=True)
        return 1
    try:
        elevated_args = list(raw_args)
        if "--elevated-child" not in elevated_args:
            elevated_args.append("--elevated-child")
        if not ensure_elevated(elevated_args):
            return 0
    except ElevationError as error:
        _show_message("XYQQuiz 启动失败", str(error), error=True)
        return 1

    if paths.frozen:
        initialize_portable_state(paths)
        migrate_portable_state(paths.config_path, paths.data_dir)
    config_path = args.config
    if config_path is None and paths.frozen:
        config_path = paths.config_path
    config = AppConfig.load(config_path)
    log_handler = configure_logging(config.log_path)
    try:
        with SingleInstance(names.mutex):
            security = LocalWebSecurity(config.web.host, config.web.port)
            base_url = f"http://127.0.0.1:{config.web.port}"
            server_holder: dict[str, Any] = {}

            def activation_url() -> str | None:
                server = server_holder.get("server")
                if server is None or not server.started or server.should_exit:
                    return None
                return security.issue_browser_url(base_url)

            with reserve_loopback_port(config.web.host, config.web.port) as web_socket:
                activation_server = ActivationServer(names.pipe, activation_url)
                activation_server.start()
                try:
                    services = build_services(config, runtime_paths=paths)
                    app = create_app(services, security)
                    health_url = f"{base_url}/api/health"
                    server = uvicorn.Server(
                        uvicorn.Config(
                            app,
                            host=config.web.host,
                            port=config.web.port,
                            log_config=None,
                        )
                    )
                    server_holder["server"] = server
                    services.shutdown = lambda: setattr(server, "should_exit", True)
                    run_server_with_browser(
                        server,
                        health_url,
                        browser_url_factory=lambda: security.issue_browser_url(base_url),
                        sockets=[web_socket],
                    )
                finally:
                    activation_server.stop()
            return 0
    except PortConflictError as error:
        logging.getLogger(__name__).error("%s", error)
        _show_message("XYQQuiz 端口冲突", str(error), error=True)
        return 4
    except AlreadyRunningError as error:
        logging.getLogger(__name__).info("%s", error)
        try:
            return 0 if activate_existing(
                names.pipe,
                webbrowser.open,
                timeout=10.0,
            ) else 3
        except ActivationProtocolError:
            logging.getLogger(__name__).exception("现有实例激活失败")
            return 3
    except StartupAssetError as error:
        logging.getLogger(__name__).error("%s", error)
        _show_message("XYQQuiz 启动资源错误", str(error), error=True)
        return 2
    except Exception:
        logging.getLogger(__name__).exception("XYQ Quiz 启动失败")
        _show_message(
            "XYQQuiz 启动失败",
            "程序启动失败，详细原因已写入 logs\\app.log。",
            error=True,
        )
        return 1
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


def _show_message(title: str, message: str, *, error: bool = False) -> None:
    if os.name == "nt" and bool(getattr(sys, "frozen", False)):
        flags = 0x00000010 if error else 0x00000040
        ctypes.windll.user32.MessageBoxW(None, message, title, flags)
        return
    stream = sys.stderr if error else sys.stdout
    print(message, file=stream)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AlreadyRunningError",
    "ElevationError",
    "PortConflictError",
    "SingleInstance",
    "StartupAssetError",
    "build_services",
    "bootstrap_frozen_stdio",
    "configure_logging",
    "ensure_elevated",
    "main",
    "run_server_with_browser",
    "reserve_loopback_port",
    "wait_for_health_and_open",
    "validate_recognition_assets",
    "validate_recognition_asset_bundle",
]
