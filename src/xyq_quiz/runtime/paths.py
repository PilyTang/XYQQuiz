from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    app_root: Path
    resource_root: Path
    defaults_root: Path
    config_path: Path
    data_dir: Path
    logs_dir: Path
    diagnostics_dir: Path
    frozen: bool

    @classmethod
    def discover(
        cls,
        *,
        executable: Path | None = None,
        bundle_root: Path | None = None,
        module_file: Path | None = None,
        frozen: bool | None = None,
    ) -> RuntimePaths:
        is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
        if is_frozen:
            exe = Path(executable or sys.executable).resolve()
            app_root = exe.parent
            detected_bundle = bundle_root or getattr(sys, "_MEIPASS", None)
            resource_root = Path(detected_bundle or (app_root / "_internal")).resolve()
            defaults_root = resource_root / "defaults"
        else:
            source = Path(module_file or __file__).resolve()
            app_root = source.parents[3]
            resource_root = app_root
            defaults_root = app_root
        return cls(
            app_root=app_root,
            resource_root=resource_root,
            defaults_root=defaults_root,
            config_path=app_root / "config.json",
            data_dir=app_root / "data",
            logs_dir=app_root / "logs",
            diagnostics_dir=app_root / "diagnostics",
            frozen=is_frozen,
        )

    @property
    def default_config_path(self) -> Path:
        name = "config.json" if self.frozen else "config.example.json"
        return self.defaults_root / name

    @property
    def default_data_dir(self) -> Path:
        return self.defaults_root / "data"


def initialize_portable_state(paths: RuntimePaths) -> None:
    """Atomically seed missing mutable state without replacing user state."""

    paths.app_root.mkdir(parents=True, exist_ok=True)
    if not paths.config_path.exists():
        if not paths.default_config_path.is_file():
            raise FileNotFoundError(f"默认配置缺失：{paths.default_config_path}")
        temporary = paths.app_root / f".config-{uuid4().hex}.tmp"
        try:
            shutil.copy2(paths.default_config_path, temporary)
            _fsync_file(temporary)
            os.replace(temporary, paths.config_path)
        finally:
            temporary.unlink(missing_ok=True)

    if not paths.data_dir.exists():
        if not paths.default_data_dir.is_dir():
            raise FileNotFoundError(f"默认题库缺失：{paths.default_data_dir}")
        temporary = paths.app_root / f".data-{uuid4().hex}.tmp"
        try:
            shutil.copytree(paths.default_data_dir, temporary)
            os.replace(temporary, paths.data_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


__all__ = ["RuntimePaths", "initialize_portable_state"]
