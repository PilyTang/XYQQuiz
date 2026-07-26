from __future__ import annotations

from contextlib import nullcontext
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

import xyq_quiz.launcher as launcher_module
from xyq_quiz.config import AppConfig
from xyq_quiz.desktop.webview import DesktopRunMode, DesktopUnavailable
from xyq_quiz.launcher import (
    AlreadyRunningError,
    ElevationError,
    PortConflictError,
    RestartError,
    SingleInstance,
    StartupAssetError,
    configure_logging,
    ensure_elevated,
    reserve_loopback_port,
    run_server_with_browser,
    run_server_with_desktop,
    spawn_restart_process,
    wait_for_health_and_open,
    validate_recognition_assets,
    validate_recognition_asset_bundle,
    build_services,
)
from xyq_quiz.runtime.paths import RuntimePaths
from xyq_quiz.performance.settings import save_performance_settings


def test_reserve_loopback_port_reports_conflict_and_releases_socket() -> None:
    with reserve_loopback_port("127.0.0.1", 0) as first:
        port = first.getsockname()[1]
        with pytest.raises(PortConflictError, match=rf"端口 {port} 已被其他程序占用"):
            reserve_loopback_port("127.0.0.1", port)

    with reserve_loopback_port("127.0.0.1", port) as second:
        assert second.getsockname() == ("127.0.0.1", port)


def test_reserve_loopback_port_rejects_non_loopback_host() -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        reserve_loopback_port("0.0.0.0", 8765)


def test_source_restart_loads_saved_performance_config(tmp_path: Path) -> None:
    discovered = RuntimePaths.discover()
    paths = RuntimePaths(
        app_root=tmp_path,
        resource_root=discovered.resource_root,
        defaults_root=discovered.defaults_root,
        config_path=tmp_path / "config.json",
        data_dir=tmp_path / "data",
        user_data_dir=tmp_path / "user-data",
        logs_dir=tmp_path / "logs",
        diagnostics_dir=tmp_path / "diagnostics",
        frozen=False,
    )
    assert launcher_module._select_runtime_config_path(None, paths) is None

    save_performance_settings(
        paths.config_path,
        ocr_backend="directml:0",
        preview_backend="windows_hardware:0",
        fallback_config=AppConfig(),
    )

    selected = launcher_module._select_runtime_config_path(None, paths)
    restarted = AppConfig.load(selected)
    assert selected == paths.config_path
    assert restarted.performance.ocr_backend == "directml:0"
    assert restarted.performance.preview_backend == "windows_hardware:auto"

    explicit = tmp_path / "custom.json"
    assert launcher_module._select_runtime_config_path(explicit, paths) == explicit


def test_spawn_restart_process_uses_argument_array_and_waits_for_old_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return object()

    monkeypatch.delattr(launcher_module.sys, "frozen", raising=False)

    spawn_restart_process(
        ["--config", "D:\\Some Folder\\config.json", "--elevated-child"],
        process_id=4321,
        popen=fake_popen,
    )

    assert calls[0][0] == [
        launcher_module.sys.executable,
        "-m",
        "xyq_quiz.launcher",
        "--config",
        "D:\\Some Folder\\config.json",
        "--elevated-child",
        "--restart-after-pid",
        "4321",
    ]
    assert calls[0][1]["close_fds"] is True


def test_spawn_restart_process_reports_launch_failure() -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise OSError("blocked")

    with pytest.raises(RestartError, match="无法启动"):
        spawn_restart_process([], process_id=1, popen=fail)


def test_uvicorn_serves_through_prebound_socket() -> None:
    async def app(scope, receive, send) -> None:
        del receive
        assert scope["type"] == "http"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ready"})

    with reserve_loopback_port("127.0.0.1", 0) as reserved:
        port = reserved.getsockname()[1]
        server = launcher_module.uvicorn.Server(
            launcher_module.uvicorn.Config(
                app,
                log_level="critical",
                lifespan="off",
            )
        )
        server_thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [reserved]},
            daemon=True,
        )
        server_thread.start()
        deadline = time.monotonic() + 3
        while not server.started and server_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            assert server.started
            with launcher_module.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                assert response.status == 200
                assert response.read() == b"ready"
        finally:
            server.should_exit = True
            server_thread.join(timeout=3)

        assert not server_thread.is_alive()


def test_main_reports_port_conflict_with_dedicated_exit_code(
    tmp_path: Path,
    monkeypatch,
) -> None:
    messages: list[tuple[str, str, bool]] = []
    requested_ports: list[int] = []
    config = AppConfig(log_path=tmp_path / "app.log")
    monkeypatch.setattr(
        launcher_module,
        "current_instance_names",
        lambda: SimpleNamespace(pipe="test-pipe", mutex="test-mutex"),
    )
    monkeypatch.setattr(launcher_module, "activate_existing", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(launcher_module, "_is_user_an_admin", lambda: True)
    monkeypatch.setattr(launcher_module, "ensure_elevated", lambda _argv: True)
    monkeypatch.setattr(launcher_module.AppConfig, "load", lambda _path: config)
    monkeypatch.setattr(launcher_module, "SingleInstance", lambda _name: nullcontext())

    def conflict(_host: str, port: int):
        requested_ports.append(port)
        raise PortConflictError("本机端口 8765 已被其他程序占用。")

    monkeypatch.setattr(
        launcher_module,
        "reserve_loopback_port",
        conflict,
    )
    monkeypatch.setattr(
        launcher_module,
        "_show_message",
        lambda title, message, *, error=False: messages.append((title, message, error)),
    )

    assert launcher_module.main(["--elevated-child", "--external-browser"]) == 4
    assert requested_ports == [8765]
    assert messages == [
        ("XYQQuiz 端口冲突", "本机端口 8765 已被其他程序占用。", True)
    ]


def test_build_services_uses_fixed_single_ocr_worker() -> None:
    config = AppConfig(
        layout_paths=[
            Path("data/layouts/keju-default.json"),
            Path("data/layouts/keju-picture.json"),
        ]
    )

    services = build_services(config)
    try:
        assert services.pipeline._executor._max_workers == 1
        assert services.coordinator._scan_fps == config.recognition.scan_fps
    finally:
        services.pipeline.close()


def test_build_services_merges_portable_local_questions_without_diagnostic_text(
    tmp_path: Path,
) -> None:
    user_data = tmp_path / "user-data"
    user_data.mkdir()
    (user_data / "questions.json").write_text(
        '{"schema_version":1,"records":[{"id":"startup-local",'
        '"question":"启动时本地题目","answer":"启动时本地答案",'
        '"mode":"supplement","enabled":true,"answer_aliases":[]}]}'
        "\n",
        encoding="utf-8",
    )
    discovered = RuntimePaths.discover()
    paths = RuntimePaths(
        app_root=tmp_path,
        resource_root=discovered.resource_root,
        defaults_root=discovered.defaults_root,
        config_path=tmp_path / "config.json",
        data_dir=discovered.data_dir,
        user_data_dir=user_data,
        logs_dir=tmp_path / "logs",
        diagnostics_dir=tmp_path / "diagnostics",
        frozen=False,
    )
    config = AppConfig(
        layout_paths=[
            Path("data/layouts/keju-default.json"),
            Path("data/layouts/keju-picture.json"),
        ]
    )

    services = build_services(config, runtime_paths=paths)
    try:
        match = services.pipeline._matcher.match_question("启动时本地题目")
        assert match is not None
        assert match.record.answer == "启动时本地答案"
        metadata = services.snapshot_diagnostic_metadata()
        assert metadata["local_questions"]["record_count"] == 1
        assert "启动时本地题目" not in str(metadata)
        assert "启动时本地答案" not in str(metadata)
    finally:
        services.pipeline.close()


def test_missing_real_layout_is_clear_and_creates_no_placeholder(tmp_path: Path) -> None:
    layout = tmp_path / "layouts" / "keju-default.json"

    with pytest.raises(StartupAssetError, match="尚未生成真实科举布局"):
        validate_recognition_assets(layout)

    assert not layout.exists()
    assert not layout.parent.exists()


def test_unreadable_anchor_is_rejected_during_startup_validation(tmp_path: Path) -> None:
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    (anchors / "one.png").write_bytes(b"broken")
    (anchors / "two.png").write_bytes(b"broken")
    layout = tmp_path / "layout.json"
    import json
    layout.write_text(
        json.dumps(
            {
                "reference_size": [10, 10],
                "question_rect": [0, 0, 0.2, 0.2],
                "option_rects": [[0, 0.2, 0.2, 0.2]] * 4,
                "anchors": [
                    {"search_rect": [0, 0, 0.5, 0.5], "template_path": "anchors/one.png", "threshold": 0.8},
                    {"search_rect": [0.5, 0, 0.5, 0.5], "template_path": "anchors/two.png", "threshold": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StartupAssetError, match="anchor 不可读"):
        validate_recognition_assets(layout)


def test_startup_bundle_rejects_one_corrupted_profile(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(StartupAssetError, match="missing.json"):
        validate_recognition_asset_bundle((missing,))


class FakeShell:
    def __init__(self, result: int = 42) -> None:
        self.result = result
        self.calls = 0
        self.verb = ""
        self.executable = ""
        self.parameters = ""
        self.cwd = ""
        self.show = 0

    def __call__(
        self,
        hwnd: object,
        verb: str,
        executable: str,
        parameters: str,
        cwd: str,
        show: int,
    ) -> int:
        del hwnd
        self.calls += 1
        self.verb = verb
        self.executable = executable
        self.parameters = parameters
        self.cwd = cwd
        self.show = show
        return self.result


class FakeFunction:
    def __init__(self, callback) -> None:
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeKernel32:
    def __init__(
        self,
        handle: int,
        last_error: int = 0,
        *,
        close_result: bool = True,
    ) -> None:
        self.handle = handle
        self.last_error = last_error
        self.close_result = close_result
        self.names: list[str] = []
        self.closed: list[int] = []
        self.CreateMutexW = FakeFunction(self._create_mutex)
        self.CloseHandle = FakeFunction(self._close_handle)

    def _create_mutex(self, security: object, initial_owner: bool, name: str) -> int:
        del security, initial_owner
        self.names.append(name)
        return self.handle

    def GetLastError(self) -> int:
        return self.last_error

    def _close_handle(self, handle: int) -> bool:
        self.closed.append(handle)
        return self.close_result


def test_non_admin_relaunches_full_module_command_with_runas() -> None:
    fake_shell = FakeShell()

    assert ensure_elevated(
        ["--config", "x y.json"],
        fake_shell,
        is_admin=lambda: False,
    ) is False

    assert fake_shell.calls == 1
    assert fake_shell.verb == "runas"
    assert fake_shell.parameters.startswith("-m xyq_quiz.launcher ")
    assert '"x y.json"' in fake_shell.parameters
    assert fake_shell.cwd == str(Path.cwd())
    assert fake_shell.show == 1


def test_admin_continues_without_shell_execute() -> None:
    fake_shell = FakeShell()

    assert ensure_elevated([], fake_shell, is_admin=lambda: True) is True

    assert fake_shell.calls == 0


def test_elevation_shell_failure_is_clear() -> None:
    with pytest.raises(ElevationError, match="32"):
        ensure_elevated(
            [],
            FakeShell(result=32),
            is_admin=lambda: False,
        )


def test_successful_elevated_relaunch_makes_parent_exit_immediately(
    monkeypatch,
) -> None:
    monkeypatch.setattr(launcher_module, "ensure_elevated", lambda _argv: False)
    monkeypatch.setattr(
        launcher_module.AppConfig,
        "load",
        lambda _path: (_ for _ in ()).throw(AssertionError("parent kept running")),
    )

    assert launcher_module.main(["--config", "missing.json"]) == 0


def test_version_report_runs_before_activation_or_elevation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = launcher_module.RuntimePaths.discover(
        executable=tmp_path / "XYQQuiz.exe",
        bundle_root=tmp_path / "_internal",
        frozen=True,
    )
    monkeypatch.setattr(launcher_module.RuntimePaths, "discover", lambda: paths)
    monkeypatch.setattr(
        launcher_module,
        "current_instance_names",
        lambda: pytest.fail("version queried instance state"),
    )
    monkeypatch.setattr(
        launcher_module,
        "ensure_elevated",
        lambda _argv: pytest.fail("version requested elevation"),
    )

    assert launcher_module.main(
        ["--version", "--report-dir", str(tmp_path / "reports")]
    ) == 0

    payload = (tmp_path / "reports" / "version.json").read_text("utf-8")
    assert '"app_id": "xyq-quiz"' in payload


def test_headless_no_dialog_self_test_failure_never_blocks_on_message_box(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        launcher_module,
        "run_self_test",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken bundle")),
    )
    monkeypatch.setattr(
        launcher_module,
        "_show_message",
        lambda *_args, **_kwargs: pytest.fail("no-dialog self-test showed a dialog"),
    )

    assert launcher_module.main(
        [
            "--self-test",
            "--headless",
            "--no-dialog",
            "--report-dir",
            str(tmp_path / "report"),
        ]
    ) == 1


def test_single_instance_releases_mutex_on_normal_exit() -> None:
    handle = 0x1234_5678_9ABC_DEF0
    kernel32 = FakeKernel32(handle=handle)

    with SingleInstance("Local\\XYQQuizBackend", kernel32=kernel32):
        assert kernel32.closed == []

    assert kernel32.names == ["Local\\XYQQuizBackend"]
    assert kernel32.closed == [handle]
    assert kernel32.CloseHandle.argtypes == [launcher_module.wintypes.HANDLE]
    assert kernel32.CloseHandle.restype is launcher_module.wintypes.BOOL


def test_existing_instance_closes_new_handle_without_killing_owner() -> None:
    kernel32 = FakeKernel32(handle=202, last_error=183)

    with pytest.raises(AlreadyRunningError, match="already running"):
        with SingleInstance("Local\\XYQQuizBackend", kernel32=kernel32):
            raise AssertionError("context body must not run")

    assert kernel32.closed == [202]


def test_failed_mutex_creation_reports_error_and_has_no_handle_to_close() -> None:
    kernel32 = FakeKernel32(handle=0, last_error=5)

    with pytest.raises(OSError, match="5"):
        with SingleInstance("Local\\XYQQuizBackend", kernel32=kernel32):
            pass

    assert kernel32.closed == []


def test_mutex_close_failure_is_reported_once() -> None:
    kernel32 = FakeKernel32(handle=303, last_error=6, close_result=False)

    with pytest.raises(OSError, match="6"):
        with SingleInstance("Local\\XYQQuizBackend", kernel32=kernel32):
            pass

    assert kernel32.closed == [303]


def test_configure_logging_uses_bounded_utf8_rotating_file(tmp_path: Path) -> None:
    handler = configure_logging(tmp_path / "app.log")
    try:
        assert handler.maxBytes == 2_000_000
        assert handler.backupCount == 3
        assert handler.encoding.lower().replace("-", "") == "utf8"
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_browser_opens_only_after_health_is_ready() -> None:
    events: list[str] = []
    readiness = iter([False, False, True])

    assert wait_for_health_and_open(
        type("Server", (), {"started": True, "should_exit": False})(),
        "http://127.0.0.1:8765/",
        threading.Event(),
        health_check=lambda _url: events.append("health") or next(readiness),
        opener=lambda url: events.append(f"open:{url}") or True,
        wait=lambda _seconds: events.append("sleep") or False,
        timeout=1,
    ) is True

    assert events == [
        "health",
        "sleep",
        "health",
        "sleep",
        "health",
        "open:http://127.0.0.1:8765/",
    ]


def test_other_http_service_never_opens_before_current_uvicorn_started() -> None:
    server = type("Server", (), {"started": False, "should_exit": False})()
    cancel = threading.Event()
    opens: list[str] = []
    health_calls = 0

    def wait(_seconds: float) -> bool:
        cancel.set()
        return True

    def health(_url: str) -> bool:
        nonlocal health_calls
        health_calls += 1
        return True

    assert wait_for_health_and_open(
        server,
        "http://127.0.0.1:8765/",
        cancel,
        health_check=health,
        opener=lambda url: opens.append(url),
        wait=wait,
    ) is False
    assert health_calls == 0
    assert opens == []


def test_exited_server_never_checks_health_or_opens() -> None:
    health_calls: list[str] = []
    opens: list[str] = []

    assert wait_for_health_and_open(
        type("Server", (), {"started": True, "should_exit": True})(),
        "http://127.0.0.1:8765/",
        threading.Event(),
        health_check=lambda url: health_calls.append(url) or True,
        opener=lambda url: opens.append(url),
    ) is False

    assert health_calls == []
    assert opens == []


def test_watcher_never_waits_unbounded_for_decision_gate() -> None:
    gate = threading.Lock()
    gate.acquire()
    started = time.monotonic()
    try:
        assert wait_for_health_and_open(
            type("Server", (), {"started": True, "should_exit": False})(),
            "http://127.0.0.1:8765/",
            threading.Event(),
            gate,
            health_check=lambda _url: True,
            opener=lambda _url: pytest.fail("contended gate must not open"),
        ) is False
    finally:
        gate.release()

    assert time.monotonic() - started < 0.5


def test_server_run_return_cancels_and_joins_browser_watcher() -> None:
    events: list[str] = []

    class FakeServer:
        started = False
        should_exit = False

        def run(self) -> None:
            events.append("run")

    class FakeThread:
        def __init__(self, *, target, args, name, daemon) -> None:
            del target, name
            assert daemon is True
            self.cancel = args[2]

        def start(self) -> None:
            events.append("start")

        def join(self, timeout: float) -> None:
            events.append(f"join:{timeout}")
            assert self.cancel.is_set()

    run_server_with_browser(
        FakeServer(),
        "http://127.0.0.1:8765/",
        thread_factory=FakeThread,
        join_timeout=0.5,
    )

    assert events == ["start", "run", "join:0.5"]


def test_desktop_runner_uses_existing_browser_path_when_pywebview_is_missing() -> None:
    server = object()
    reserved = object()
    events: list[object] = []

    class MissingDesktop:
        def run(self, *_args, **_kwargs):
            raise DesktopUnavailable("pywebview missing", server_started=False)

        def enable_browser_fallback(self, _opener, _url_factory) -> None:
            events.append("fallback-enabled")

        def disable_browser_fallback(self) -> None:
            events.append("fallback-disabled")

    def browser_runner(current_server, health_url, **kwargs) -> None:
        events.append((current_server, health_url, kwargs))

    mode = run_server_with_desktop(
        MissingDesktop(),  # type: ignore[arg-type]
        server,
        "http://127.0.0.1:12345/api/health",
        lambda: "http://127.0.0.1:12345/#token=once",
        sockets=[reserved],  # type: ignore[list-item]
        browser_runner=browser_runner,
    )

    assert mode is DesktopRunMode.BROWSER_FALLBACK
    assert events[0] == "fallback-enabled"
    assert events[1][0] is server  # type: ignore[index]
    assert events[1][2]["sockets"] == [reserved]  # type: ignore[index]
    assert events[2] == "fallback-disabled"


def test_desktop_runner_never_restarts_server_after_post_start_failure() -> None:
    browser_calls: list[object] = []

    class FailedDesktop:
        def run(self, *_args, **_kwargs):
            raise DesktopUnavailable("WebView2 failed", server_started=True)

    with pytest.raises(DesktopUnavailable, match="WebView2 failed"):
        run_server_with_desktop(
            FailedDesktop(),  # type: ignore[arg-type]
            object(),
            "http://127.0.0.1:12345/api/health",
            lambda: "http://127.0.0.1:12345/#token=once",
            browser_runner=lambda *_args, **_kwargs: browser_calls.append(object()),
        )

    assert browser_calls == []


def test_prebound_socket_is_forwarded_to_uvicorn_server() -> None:
    reserved = object()
    calls: list[object] = []

    class FakeServer:
        started = False
        should_exit = False

        def run(self, *, sockets) -> None:
            calls.extend(sockets)

    class FakeThread:
        def __init__(self, *, target, args, name, daemon) -> None:
            del target, name, daemon
            self.cancel = args[2]

        def start(self) -> None:
            pass

        def join(self, timeout: float) -> None:
            del timeout
            assert self.cancel.is_set()

    run_server_with_browser(
        FakeServer(),
        "http://127.0.0.1:8765/",
        thread_factory=FakeThread,
        sockets=[reserved],
    )

    assert calls == [reserved]


def test_fast_server_failure_cannot_open_a_later_service() -> None:
    opens: list[str] = []

    class FastFailServer:
        started = False
        should_exit = False

        def run(self) -> None:
            return

    server = FastFailServer()

    def waiter(current_server, url, cancel, gate) -> bool:
        return wait_for_health_and_open(
            current_server,
            url,
            cancel,
            gate,
            health_check=lambda _url: True,
            opener=lambda opened_url: opens.append(opened_url),
        )

    run_server_with_browser(
        server,
        "http://127.0.0.1:8765/",
        waiter=waiter,
    )
    server.started = True
    time.sleep(0.05)

    assert opens == []


def test_server_run_exception_still_cancels_and_joins_watcher() -> None:
    events: list[str] = []

    class FailingServer:
        started = False
        should_exit = False

        def run(self) -> None:
            raise RuntimeError("bind failed")

    class FakeThread:
        def __init__(self, *, target, args, name, daemon) -> None:
            del target, name, daemon
            self.cancel = args[2]

        def start(self) -> None:
            events.append("start")

        def join(self, timeout: float) -> None:
            assert self.cancel.is_set()
            events.append(f"join:{timeout}")

    with pytest.raises(RuntimeError, match="bind failed"):
        run_server_with_browser(
            FailingServer(),
            "http://127.0.0.1:8765/",
            thread_factory=FakeThread,
            join_timeout=0.25,
        )

    assert events == ["start", "join:0.25"]


@pytest.mark.parametrize("server_raises", [False, True])
def test_blocked_opener_cannot_block_server_return_or_original_error(
    server_raises: bool,
) -> None:
    opener_entered = threading.Event()
    opener_release = threading.Event()
    completed = threading.Event()
    errors: list[BaseException] = []

    class Server:
        started = True
        should_exit = False

        def run(self) -> None:
            assert opener_entered.wait(timeout=1)
            if server_raises:
                raise RuntimeError("original server failure")

    def blocked_opener(_url: str) -> None:
        opener_entered.set()
        opener_release.wait()

    def waiter(server, url, cancel, gate) -> bool:
        return wait_for_health_and_open(
            server,
            url,
            cancel,
            gate,
            health_check=lambda _url: True,
            opener=blocked_opener,
        )

    def exercise() -> None:
        try:
            run_server_with_browser(
                Server(),
                "http://127.0.0.1:8765/",
                waiter=waiter,
                join_timeout=0.05,
            )
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    exercise_thread = threading.Thread(target=exercise)
    exercise_thread.start()
    try:
        assert opener_entered.wait(timeout=1)
        assert completed.wait(timeout=0.3)
        if server_raises:
            assert len(errors) == 1
            assert isinstance(errors[0], RuntimeError)
            assert str(errors[0]) == "original server failure"
        else:
            assert errors == []
    finally:
        opener_release.set()
        exercise_thread.join(timeout=1)

    assert not exercise_thread.is_alive()
