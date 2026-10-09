# PyInstaller build specification for the double-clickable macOS app.
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

hiddenimports = [
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
    a.binaries,
    a.datas,
    [],
    name="UsedSurf",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
app = BUNDLE(
    exe,
    name="UsedSurf.app",
    bundle_identifier="com.usedsurf.cropper",
    info_plist={
        "CFBundleDisplayName": "UsedSurf Cropper",
        "CFBundleName": "UsedSurf Cropper",
        "NSHighResolutionCapable": True,
    },
)
