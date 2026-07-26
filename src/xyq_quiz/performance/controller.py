from __future__ import annotations

from dataclasses import asdict
import gc
import logging
from pathlib import Path
import threading
from typing import Callable

import cv2
import numpy as np

from xyq_quiz.config import AppConfig, PerformanceConfig
from xyq_quiz.performance.devices import enumerate_graphics_adapters
from xyq_quiz.performance.models import (
    BackendCapability,
    BackendOption,
    BackendRuntimeState,
    GraphicsAdapter,
    PerformanceSnapshot,
)
from xyq_quiz.performance.native_preview import run_native_preview_self_test
from xyq_quiz.performance.settings import (
    SavedPerformanceSettings,
    save_performance_settings,
)
from xyq_quiz.recognition.models import OCRText
from xyq_quiz.recognition.ocr import OCRRole, RapidOCREngine


_LOGGER = logging.getLogger(__name__)


def directml_provider_available() -> tuple[bool, str | None]:
    try:
        import onnxruntime as ort

        providers = tuple(ort.get_available_providers())
    except Exception as error:
        return False, f"ONNX Runtime 不可用：{error}"
    if "DmlExecutionProvider" not in providers:
        return False, "当前 ONNX Runtime 未包含 DirectML 执行提供程序"
    return True, None


class _ResilientOCREngine:
    """Permanently fall back to CPU after one accelerated runtime failure."""

    def __init__(
        self,
        primary: RapidOCREngine,
        fallback: RapidOCREngine,
        *,
        on_success: Callable[[], None],
        on_failure: Callable[[str], None],
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._on_success = on_success
        self._on_failure = on_failure
        self._lock = threading.Lock()
        self._failed = False
        self._success_reported = False

    def recognize(self, image: np.ndarray) -> OCRText:
        return self._invoke("recognize", image)

    def recognize_region(
        self,
        image: np.ndarray,
        role: OCRRole,
        *,
        fallback_image: np.ndarray,
    ) -> OCRText:
        return self._invoke(
            "recognize_region",
            image,
            role,
            fallback_image=fallback_image,
        )

    def _invoke(self, method: str, *args: object, **kwargs: object) -> OCRText:
        with self._lock:
            use_fallback = self._failed
        if not use_fallback:
            try:
                result = getattr(self._primary, method)(*args, **kwargs)
            except Exception as error:
                reason = f"DirectML OCR 运行失败：{error}"
                with self._lock:
                    first_failure = not self._failed
                    self._failed = True
                if first_failure:
                    _LOGGER.exception("%s；本次运行回退 CPU", reason)
                    self._on_failure(reason)
            else:
                providers = self._primary.execution_providers()
                if any(
                    current[0] != "DmlExecutionProvider"
                    for current in providers
                ):
                    reason = f"DirectML OCR 实际 provider 校验失败：{providers}"
                    with self._lock:
                        first_failure = not self._failed
                        self._failed = True
                    if first_failure:
                        self._on_failure(reason)
                    return getattr(self._fallback, method)(*args, **kwargs)
                with self._lock:
                    first_success = not self._success_reported
                    self._success_reported = True
                if first_success:
                    self._on_success()
                return result
        return getattr(self._fallback, method)(*args, **kwargs)


class PerformanceController:
    def __init__(
        self,
        preferences: PerformanceConfig,
        *,
        config_path: Path,
        fallback_config: AppConfig,
        desktop_mode: bool,
        adapter_provider: Callable[[], tuple[GraphicsAdapter, ...]] = (
            enumerate_graphics_adapters
        ),
        dml_provider_probe: Callable[[], tuple[bool, str | None]] = (
            directml_provider_available
        ),
        dml_engine_factory: Callable[[int], RapidOCREngine] | None = None,
        cpu_engine_factory: Callable[[], RapidOCREngine] = RapidOCREngine,
        native_preview_helper: Path | None = None,
        native_preview_probe: Callable[[Path, int], object] = (
            run_native_preview_self_test
        ),
    ) -> None:
        self._current_preferences = preferences
        self._pending_preferences = preferences
        self._config_path = Path(config_path)
        self._fallback_config = fallback_config
        self._desktop_mode = desktop_mode
        self._adapter_provider = adapter_provider
        self._dml_provider_probe = dml_provider_probe
        self._dml_engine_factory = dml_engine_factory or (
            lambda device_id: RapidOCREngine(
                backend="directml",
                device_id=device_id,
            )
        )
        self._cpu_engine_factory = cpu_engine_factory
        self._native_preview_helper = (
            Path(native_preview_helper).resolve()
            if native_preview_helper is not None
            else None
        )
        self._native_preview_probe = native_preview_probe
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._probe_thread: threading.Thread | None = None
        self._adapters = self._safe_adapters()
        self._dml_provider_ok, self._dml_provider_reason = self._safe_dml_provider()
        self._verified_dml: set[int] = set()
        self._failed_dml: dict[int, str] = {}
        self._verified_preview: set[int] = set()
        self._failed_preview: dict[int, str] = {}
        self._canvas_fps: float | None = None
        self._benchmark_status = "idle"

        self._ocr_state = self._initial_ocr_state(preferences.ocr_backend)
        self._preview_state = self._initial_preview_state(
            preferences.preview_backend
        )

    def create_ocr_engine(self):
        requested = self._current_preferences.ocr_backend
        if requested.startswith("directml:"):
            device_id = int(requested.partition(":")[2])
            adapter = self._adapter(device_id)
            if not self._dml_provider_ok or adapter is None:
                return self._cpu_engine_factory()
            return _ResilientOCREngine(
                self._dml_engine_factory(device_id),
                self._cpu_engine_factory(),
                on_success=lambda: self._mark_dml_success(device_id),
                on_failure=lambda reason: self._mark_dml_failure(device_id, reason),
            )
        # Without a valid benchmark cache Auto deliberately starts on CPU.
        return self._cpu_engine_factory()

    def start(self) -> None:
        with self._lock:
            if self._probe_thread is not None:
                return
            can_probe_dml = self._dml_provider_ok and bool(self._adapters)
            can_probe_preview = (
                self._desktop_mode
                and bool(self._adapters)
                and self._native_preview_helper is not None
                and self._native_preview_helper.is_file()
            )
            if not can_probe_dml and not can_probe_preview:
                return
            self._benchmark_status = "probing"
            thread = threading.Thread(
                target=self._probe_available_backends,
                name="xyq-quiz-backend-probe",
                daemon=True,
            )
            self._probe_thread = thread
        thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.wait_for_probe(timeout)

    def wait_for_probe(self, timeout: float = 5.0) -> bool:
        with self._lock:
            thread = self._probe_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                return False
        return True

    def snapshot(self) -> PerformanceSnapshot:
        with self._lock:
            probing = self._probe_thread is not None and self._probe_thread.is_alive()
            return PerformanceSnapshot(
                ocr=self._ocr_state,
                preview=self._preview_state,
                pending_ocr=self._pending_preferences.ocr_backend,
                pending_preview=self._pending_preferences.preview_backend,
                ocr_options=self._ocr_options(),
                preview_options=self._preview_options(),
                probing=probing,
                benchmark_status=self._benchmark_status,
                canvas_fps=self._canvas_fps,
            )

    def payload(self) -> dict[str, object]:
        return asdict(self.snapshot())

    def record_canvas_fps(self, value: float) -> None:
        if not np.isfinite(value) or value < 0 or value > 240:
            raise ValueError("canvas FPS must be between 0 and 240")
        with self._lock:
            self._canvas_fps = round(float(value), 1)

    def save(
        self,
        *,
        ocr_backend: str,
        preview_backend: str,
    ) -> SavedPerformanceSettings:
        validated = PerformanceConfig.model_validate(
            {
                "ocr_backend": ocr_backend,
                "preview_backend": preview_backend,
            }
        )
        snapshot = self.snapshot()
        self._require_selectable(validated.ocr_backend, snapshot.ocr_options)
        self._require_selectable(
            validated.preview_backend,
            snapshot.preview_options,
        )
        saved = save_performance_settings(
            self._config_path,
            ocr_backend=validated.ocr_backend,
            preview_backend=validated.preview_backend,
            fallback_config=self._fallback_config,
        )
        with self._lock:
            self._pending_preferences = validated
        return saved

    def _probe_available_backends(self) -> None:
        try:
            self._probe_directml()
            self._probe_native_preview()
            self._migrate_duplicate_adapter_preferences()
        finally:
            with self._lock:
                if not self._stop.is_set():
                    self._benchmark_status = "ready"

    def _probe_directml(self) -> None:
        if not self._dml_provider_ok:
            return
        for adapter in self._adapters:
            if self._stop.is_set():
                return
            with self._lock:
                if adapter.device_id in self._verified_dml:
                    continue
            engine: RapidOCREngine | None = None
            try:
                engine = self._dml_engine_factory(adapter.device_id)
                image = np.full((96, 384, 3), 255, dtype=np.uint8)
                cv2.putText(
                    image,
                    "backend selftest",
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                engine.recognize(image)
                providers = engine.execution_providers()
                if any(current[0] != "DmlExecutionProvider" for current in providers):
                    raise RuntimeError(f"实际 provider 不是 DirectML：{providers}")
            except Exception as error:
                self._mark_dml_failure(
                    adapter.device_id,
                    f"DirectML 最小自检失败：{error}",
                    affect_runtime=False,
                )
            else:
                self._mark_dml_success(adapter.device_id)
            finally:
                del engine
                gc.collect()

    def _probe_native_preview(self) -> None:
        helper = self._native_preview_helper
        if not self._desktop_mode or helper is None or not helper.is_file():
            return
        for adapter in self._adapters:
            if self._stop.is_set():
                return
            try:
                self._native_preview_probe(helper, adapter.device_id)
            except Exception as error:
                with self._lock:
                    self._verified_preview.discard(adapter.device_id)
                    self._failed_preview[adapter.device_id] = (
                        f"Windows 硬件预览自检失败：{error}"
                    )
            else:
                with self._lock:
                    self._failed_preview.pop(adapter.device_id, None)
                    self._verified_preview.add(adapter.device_id)

    def _mark_dml_success(self, device_id: int) -> None:
        with self._lock:
            self._failed_dml.pop(device_id, None)
            self._verified_dml.add(device_id)
            value = f"directml:{device_id}"
            if self._ocr_state.requested == value:
                adapter = self._adapter(device_id)
                self._ocr_state = BackendRuntimeState(
                    requested=value,
                    effective=value,
                    label=self._dml_label(adapter, device_id),
                )

    def _mark_dml_failure(
        self,
        device_id: int,
        reason: str,
        *,
        affect_runtime: bool = True,
    ) -> None:
        with self._lock:
            self._verified_dml.discard(device_id)
            self._failed_dml[device_id] = reason
            value = f"directml:{device_id}"
            if affect_runtime and self._ocr_state.requested == value:
                self._ocr_state = BackendRuntimeState(
                    requested=value,
                    effective="cpu",
                    label="OCR CPU",
                    fallback_reason=reason,
                )

    def _initial_ocr_state(self, requested: str) -> BackendRuntimeState:
        if not requested.startswith("directml:"):
            return BackendRuntimeState(requested, "cpu", "OCR CPU")
        device_id = int(requested.partition(":")[2])
        adapter = self._adapter(device_id)
        if not self._dml_provider_ok:
            reason = self._dml_provider_reason or "DirectML 不可用"
            self._failed_dml[device_id] = reason
            return BackendRuntimeState(
                requested,
                "cpu",
                "OCR CPU",
                reason,
            )
        if adapter is None:
            reason = f"已保存的 DirectML 设备 {device_id} 当前不存在"
            self._failed_dml[device_id] = reason
            return BackendRuntimeState(requested, "cpu", "OCR CPU", reason)
        return BackendRuntimeState(
            requested,
            requested,
            self._dml_label(adapter, device_id),
        )

    def _initial_preview_state(self, requested: str) -> BackendRuntimeState:
        if requested == "windows_hardware:auto":
            if not self._desktop_mode:
                return BackendRuntimeState(
                    requested,
                    "cpu",
                    "预览 CPU",
                    "外部浏览器模式不支持 Windows 硬件预览",
                )
            if self._native_preview_helper is None or not self._native_preview_helper.is_file():
                return BackendRuntimeState(
                    requested,
                    "cpu",
                    "预览 CPU",
                    "Windows 硬件预览辅助进程不存在",
                )
            return BackendRuntimeState(
                requested,
                requested,
                self._preview_label(),
            )
        return BackendRuntimeState(requested, "cpu", "预览 CPU")

    def _ocr_options(self) -> tuple[BackendOption, ...]:
        options = [
            BackendOption("auto", "自动", BackendCapability.OCR, True, True),
            BackendOption("cpu", "CPU", BackendCapability.OCR, True, True),
        ]
        requested = self._pending_preferences.ocr_backend
        requested_id = self._requested_device_id(requested, "directml")
        for adapter in self._representative_adapters(
            self._verified_dml,
            requested_id,
        ):
            device_id = adapter.device_id
            value = f"directml:{device_id}"
            verified = device_id in self._verified_dml
            failed_reason = self._failed_dml.get(device_id)
            reason = None
            if not self._dml_provider_ok:
                reason = self._dml_provider_reason
            elif failed_reason is not None:
                reason = failed_reason
            elif not verified:
                reason = "正在进行 DirectML 最小自检"
            options.append(
                BackendOption(
                    value,
                    self._dml_label(adapter, device_id),
                    BackendCapability.OCR,
                    verified,
                    verified,
                    reason,
                    device_id,
                )
            )
        return tuple(options)

    def _preview_options(self) -> tuple[BackendOption, ...]:
        options = [
            BackendOption("auto", "自动", BackendCapability.PREVIEW, True, True),
            BackendOption("cpu", "CPU", BackendCapability.PREVIEW, True, True),
        ]
        if self._desktop_mode and self._verified_preview:
            options.append(
                BackendOption(
                    "windows_hardware:auto",
                    self._preview_label(),
                    BackendCapability.PREVIEW,
                    True,
                    True,
                )
            )
        return tuple(options)

    def preview_device_id(self) -> int | None:
        effective = self.snapshot().preview.effective
        return -1 if effective == "windows_hardware:auto" else None

    def mark_preview_success(self, device_id: int) -> None:
        with self._lock:
            self._verified_preview.add(device_id)
            self._failed_preview.pop(device_id, None)
            requested = "windows_hardware:auto"
            if self._preview_state.requested == requested:
                self._preview_state = BackendRuntimeState(
                    requested,
                    requested,
                    self._preview_label(),
                )

    def mark_preview_failure(self, device_id: int, reason: str) -> None:
        failed_value = "windows_hardware:auto"
        with self._lock:
            self._failed_preview[device_id] = reason
            if self._preview_state.requested == failed_value:
                self._preview_state = BackendRuntimeState(
                    failed_value,
                    "cpu",
                    "预览 CPU",
                    reason,
                )
        _LOGGER.warning("硬件预览运行失败，本次运行回退 CPU；已保留用户选择")

    def _adapter(self, device_id: int) -> GraphicsAdapter | None:
        return next(
            (adapter for adapter in self._adapters if adapter.device_id == device_id),
            None,
        )

    @staticmethod
    def _adapter_identity(adapter: GraphicsAdapter) -> tuple[object, ...]:
        return (
            adapter.name.casefold(),
            adapter.vendor_id,
            adapter.device_id_hex,
            adapter.dedicated_video_memory,
        )

    def _representative_adapters(
        self,
        verified: set[int],
        requested_id: int | None,
        *,
        verified_only: bool = False,
    ) -> tuple[GraphicsAdapter, ...]:
        groups: dict[tuple[object, ...], list[GraphicsAdapter]] = {}
        for adapter in self._adapters:
            groups.setdefault(self._adapter_identity(adapter), []).append(adapter)

        representatives: list[GraphicsAdapter] = []
        for candidates in groups.values():
            verified_candidates = [
                item for item in candidates if item.device_id in verified
            ]
            if verified_only and not verified_candidates:
                continue
            requested = next(
                (
                    item
                    for item in candidates
                    if item.device_id == requested_id
                    and (not verified_only or item.device_id in verified)
                ),
                None,
            )
            representatives.append(
                requested
                or (verified_candidates[0] if verified_candidates else candidates[0])
            )
        return tuple(representatives)

    @staticmethod
    def _requested_device_id(value: str, prefix: str) -> int | None:
        marker = f"{prefix}:"
        if not value.startswith(marker):
            return None
        suffix = value.partition(":")[2]
        return int(suffix) if suffix.isascii() and suffix.isdecimal() else None

    def _migrate_duplicate_adapter_preferences(self) -> None:
        with self._lock:
            pending = self._pending_preferences
            migrated_ocr = self._migrated_backend_value(
                pending.ocr_backend,
                "directml",
                self._verified_dml,
            )
            migrated_preview = self._migrated_backend_value(
                pending.preview_backend,
                "windows_hardware",
                self._verified_preview,
            )
        if (
            migrated_ocr == pending.ocr_backend
            and migrated_preview == pending.preview_backend
        ):
            return
        try:
            saved = save_performance_settings(
                self._config_path,
                ocr_backend=migrated_ocr,
                preview_backend=migrated_preview,
                fallback_config=self._fallback_config,
            )
        except Exception:
            _LOGGER.exception("重复图形适配器配置自动迁移失败")
            return
        with self._lock:
            self._pending_preferences = PerformanceConfig(
                ocr_backend=saved.ocr_backend,
                preview_backend=saved.preview_backend,
            )
        _LOGGER.info(
            "重复图形适配器配置已迁移：OCR %s，预览 %s",
            saved.ocr_backend,
            saved.preview_backend,
        )

    def _migrated_backend_value(
        self,
        value: str,
        prefix: str,
        verified: set[int],
    ) -> str:
        device_id = self._requested_device_id(value, prefix)
        if device_id is None or device_id in verified:
            return value
        adapter = self._adapter(device_id)
        if adapter is None:
            return value
        identity = self._adapter_identity(adapter)
        replacement = next(
            (
                item
                for item in self._adapters
                if item.device_id in verified
                and item.device_id != device_id
                and self._adapter_identity(item) == identity
            ),
            None,
        )
        if replacement is None:
            return value
        return f"{prefix}:{replacement.device_id}"

    @staticmethod
    def _dml_label(adapter: GraphicsAdapter | None, device_id: int) -> str:
        name = adapter.name if adapter is not None else f"设备 {device_id}"
        return f"DirectML — {name}"

    @staticmethod
    def _preview_label() -> str:
        return "Windows 硬件预览（自动选择显示器显卡）"

    def _safe_adapters(self) -> tuple[GraphicsAdapter, ...]:
        try:
            return self._adapter_provider()
        except Exception:
            _LOGGER.exception("DXGI adapter enumeration failed")
            return ()

    def _safe_dml_provider(self) -> tuple[bool, str | None]:
        try:
            return self._dml_provider_probe()
        except Exception as error:
            return False, f"DirectML 探测失败：{error}"

    @staticmethod
    def _require_selectable(
        value: str,
        options: tuple[BackendOption, ...],
    ) -> None:
        option = next((item for item in options if item.value == value), None)
        if option is None or not option.selectable:
            reason = option.reason if option is not None else "后端不存在"
            raise ValueError(f"后端 {value} 当前不可选择：{reason}")


__all__ = ["PerformanceController", "directml_provider_available"]
