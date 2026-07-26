# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


PROJECT_ROOT = Path(SPECPATH).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"

datas = [
    (str(PROJECT_ROOT / "data"), "defaults/data"),
    (str(SOURCE_ROOT / "xyq_quiz" / "web" / "static"), "xyq_quiz/web/static"),
    (str(PROJECT_ROOT / "packaging" / "state-schema.json"), "."),
]
binaries = []
native_helper = PROJECT_ROOT / "build" / "native" / "XYQPreviewHelper.exe"
if not native_helper.is_file():
    raise FileNotFoundError(
        "Native preview helper is missing; run scripts/build-native-helper.ps1 first"
    )
binaries.append((str(native_helper), "native"))
hiddenimports = []
for package in (
    "rapidocr",
    "onnxruntime",
    "cv2",
    "windows_capture",
    "uvicorn",
    "websockets",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# pywebview selects its platform module at runtime. Keep the Windows
# WinForms/WebView2 path explicit and reject every alternate renderer so a
# developer machine with Qt, GTK, or CEF installed cannot silently inflate or
# change the portable build. The package's own PyInstaller hook collects its
# JavaScript and WebView2 interop/loader files; do not bulk-collect webview.
hiddenimports += [
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
]
webview_excludes = [
    "webview.platforms.android",
    "webview.platforms.cef",
    "webview.platforms.cocoa",
    "webview.platforms.gtk",
    "webview.platforms.mshtml",
    "webview.platforms.qt",
    "cefpython3",
    "gi",
    "jnius",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "qtpy",
]

a = Analysis(
    [str(SOURCE_ROOT / "xyq_quiz" / "launcher.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", *webview_excludes],
    noarchive=False,
    optimize=0,
)

# pywebview's cross-platform hook deliberately ships every loader. Keep all
# three WebView2 loader directories: pywebview 6.2.1 resolves win-arm64,
# win-x64, and win-x86 unconditionally while importing EdgeChromium, even in
# an x64 process. The extra loaders are runtime resources, not alternate app
# architectures. Legacy MSHTML interop and Android payloads remain unused.
unused_webview_payloads = (
    "webview/lib/pywebview-android.jar",
    "webview/lib/webbrowserinterop.x64.dll",
    "webview/lib/webbrowserinterop.x86.dll",
)


def keep_target_webview_payload(entry):
    destination = str(entry[0]).replace("\\", "/").lower()
    return not any(
        destination == payload or destination.startswith(payload)
        for payload in unused_webview_payloads
    )


a.datas = [entry for entry in a.datas if keep_target_webview_payload(entry)]
a.binaries = [entry for entry in a.binaries if keep_target_webview_payload(entry)]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="XYQQuiz",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",
    codesign_identity=None,
    entitlements_file=None,
    manifest=str(PROJECT_ROOT / "packaging" / "XYQQuiz.manifest"),
    version=str(PROJECT_ROOT / "packaging" / "version_info.txt"),
    uac_admin=False,
    contents_directory="_internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="XYQQuiz",
)
