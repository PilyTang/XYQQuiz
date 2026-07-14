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

a = Analysis(
    [str(SOURCE_ROOT / "xyq_quiz" / "launcher.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest"],
    noarchive=False,
    optimize=0,
)
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
