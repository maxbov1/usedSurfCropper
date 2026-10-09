# PyInstaller build specification for the double-clickable macOS app.
import os
from pathlib import Path

# PyInstaller exposes SPECPATH as the directory containing this spec file.
ROOT = Path(SPECPATH).parent
datas = [
    (str(ROOT / "templates"), "templates"),
    (str(ROOT / "static"), "static"),
    (str(ROOT / "fixtures"), "fixtures"),
    (str(ROOT / "sample-pack" / "usedsurf-crop-sample-pack"), "sample-pack/usedsurf-crop-sample-pack"),
]
model = ROOT / "yolo11n.pt"
if model.exists():
    datas.append((str(model), "models"))
build_info_value = os.environ.get("USED_SURF_BUILD_INFO_PATH", "")
if build_info_value:
    build_info = Path(build_info_value)
    if build_info.is_file():
        datas.append((str(build_info), "."))

hiddenimports = [
    "pytesseract",
    "pytesseract.pytesseract",
    "ultralytics",
    "torch",
    "torch.nn",
    "torch.backends",
    "torch.backends.mps",
    "scripts.run_batch_crop",
    "scripts.run_true_crop_eval",
]

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tensorboard", "torch.utils.tensorboard", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UsedSurf",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="UsedSurf",
)
app = BUNDLE(
    coll,
    name="UsedSurf.app",
    bundle_identifier="com.usedsurf.cropper",
    info_plist={
        "CFBundleDisplayName": "UsedSurf Cropper",
        "CFBundleName": "UsedSurf Cropper",
        "NSHighResolutionCapable": True,
    },
)
