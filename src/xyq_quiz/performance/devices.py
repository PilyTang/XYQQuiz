from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
from uuid import UUID

from xyq_quiz.performance.models import GraphicsAdapter


_DXGI_ERROR_NOT_FOUND = 0x887A0002
_MICROSOFT_VENDOR_ID = 0x1414
_MICROSOFT_BASIC_RENDER_DEVICE_ID = 0x008C


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    )

    @classmethod
    def from_uuid(cls, value: str) -> _GUID:
        raw = UUID(value).bytes_le
        return cls.from_buffer_copy(raw)


class _LUID(ctypes.Structure):
    _fields_ = (("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG))


class _DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = (
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
    )


def enumerate_graphics_adapters() -> tuple[GraphicsAdapter, ...]:
    """Return DXGI adapter order used by DirectML's ``device_id`` option."""
    if sys.platform != "win32":
        return ()

    dxgi = ctypes.WinDLL("dxgi", use_last_error=True)
    create_factory = dxgi.CreateDXGIFactory1
    create_factory.argtypes = (ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))
    create_factory.restype = ctypes.c_long

    iid_factory1 = _GUID.from_uuid("770AAE78-F26F-4DBA-A829-253C83D1B387")
    factory = ctypes.c_void_p()
    _check_hresult(create_factory(ctypes.byref(iid_factory1), ctypes.byref(factory)))
    try:
        adapters: list[GraphicsAdapter] = []
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            result = _com_call(
                factory,
                # DirectML defines device_id using IDXGIFactory::EnumAdapters
                # order. Do not substitute a WMI or preference-sorted list.
                7,
                ctypes.c_long,
                wintypes.UINT,
                ctypes.POINTER(ctypes.c_void_p),
            )(factory, index, ctypes.byref(adapter))
            if ctypes.c_ulong(result).value == _DXGI_ERROR_NOT_FOUND:
                break
            _check_hresult(result)
            try:
                description = _DXGI_ADAPTER_DESC()
                _check_hresult(
                    _com_call(
                        adapter,
                        8,
                        ctypes.c_long,
                        ctypes.POINTER(_DXGI_ADAPTER_DESC),
                    )(adapter, ctypes.byref(description))
                )
                name = str(description.Description).rstrip("\x00").strip()
                adapters.append(
                    GraphicsAdapter(
                        device_id=index,
                        name=name or f"图形适配器 {index}",
                        vendor_id=int(description.VendorId),
                        device_id_hex=int(description.DeviceId),
                        dedicated_video_memory=int(description.DedicatedVideoMemory),
                        is_software=(
                            int(description.VendorId) == _MICROSOFT_VENDOR_ID
                            and int(description.DeviceId)
                            == _MICROSOFT_BASIC_RENDER_DEVICE_ID
                        ),
                    )
                )
            finally:
                _release(adapter)
            index += 1
        return tuple(adapter for adapter in adapters if not adapter.is_software)
    finally:
        _release(factory)


def _com_call(pointer: ctypes.c_void_p, index: int, restype: object, *argtypes: object):
    vtable = ctypes.cast(
        pointer,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
    ).contents
    function_address = vtable[index]
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(function_address)


def _release(pointer: ctypes.c_void_p) -> None:
    if pointer:
        _com_call(pointer, 2, wintypes.ULONG)(pointer)


def _check_hresult(result: int) -> None:
    if result < 0:
        raise OSError(
            ctypes.c_ulong(result).value,
            f"DXGI call failed with HRESULT 0x{ctypes.c_ulong(result).value:08X}",
        )


__all__ = ["enumerate_graphics_adapters"]
