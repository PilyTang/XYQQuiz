from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys

from xyq_quiz.config import AppConfig, PerformanceConfig
from xyq_quiz.performance.devices import enumerate_graphics_adapters
from xyq_quiz.performance.settings import _atomic_write_json

_LOGGER = logging.getLogger(__name__)
_GIB = 1024 ** 3


@dataclass(frozen=True)
class HardwareFacts:
    logical_processors: int | None
    installed_memory_bytes: int | None
    hardware_graphics: bool | None


def _installed_memory_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetPhysicallyInstalledSystemMemory
    query.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    query.restype = wintypes.BOOL
    memory_kib = ctypes.c_ulonglong()
    if not query(ctypes.byref(memory_kib)) or not memory_kib.value:
        return None
    return int(memory_kib.value) * 1024


def probe_hardware() -> HardwareFacts:
    try:
        processors = os.cpu_count()
    except Exception:
        processors = None
    try:
        memory = _installed_memory_bytes()
    except Exception:
        memory = None
    try:
        # Known physical GPU vendors; virtual/remote adapters must not make
        # an otherwise unverified system qualify for standard mode.
        graphics = any(not a.is_software and a.vendor_id in {0x10DE,0x1002,0x8086,0x5143}
                       for a in enumerate_graphics_adapters())
    except Exception:
        graphics = None
    return HardwareFacts(processors, memory, graphics)


def choose_resource_profile(facts: HardwareFacts) -> tuple[bool, str]:
    standard = (facts.logical_processors is not None and facts.logical_processors >= 12
                and facts.installed_memory_bytes is not None and facts.installed_memory_bytes >= 16 * _GIB
                and facts.hardware_graphics is True)
    cpu = f"{facts.logical_processors} 个逻辑线程" if facts.logical_processors else "CPU 信息未取得"
    memory = (f"{facts.installed_memory_bytes / _GIB:g} GiB 已安装内存"
              if facts.installed_memory_bytes else "内存信息未取得")
    gpu = {True:"检测到硬件图形适配器",False:"未检测到硬件图形适配器",None:"显卡信息未取得"}[facts.hardware_graphics]
    mode = "标准模式" if standard else "低配模式"
    return not standard, f"首次自动选择{mode}：{cpu}，{memory}，{gpu}。可手动调整。"


def initialize_resource_profile(config: AppConfig, config_path: Path, *, probe=probe_hardware) -> AppConfig:
    """Evaluate once, atomically persist, and never replace a saved choice."""
    if config.performance.resource_profile_origin != "pending":
        return config
    path = Path(config_path)
    document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else config.model_dump(mode="json")
    existing = PerformanceConfig.model_validate(document.get("performance", {}))
    if existing.resource_profile_origin != "pending":
        return config.model_copy(update={"performance": existing})
    try:
        facts = probe()
    except Exception:
        facts = HardwareFacts(None,None,None)
    low, reason = choose_resource_profile(facts)
    selected = config.performance.model_copy(update={
        "low_resource_mode":low, "resource_profile_origin":"automatic", "resource_profile_reason":reason,
    })
    document["performance"] = selected.model_dump(mode="json")
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        _atomic_write_json(path,document)
    except OSError:
        _LOGGER.warning("首次硬件选择未能保存，本次仍使用检测结果；下次启动会重试",exc_info=True)
    _LOGGER.info("%s",reason)
    return config.model_copy(update={"performance": selected})
