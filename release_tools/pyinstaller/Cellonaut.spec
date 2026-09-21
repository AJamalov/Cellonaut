# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)

project_dir = Path(SPECPATH).resolve().parents[1]
release_tools_dir = project_dir / "release_tools"
package_data_dir = project_dir / "cellonaut" / "data"
assets_dir = package_data_dir / "assets"
offline_assets_dir = project_dir / ".local" / "release_assets" / "offline"
if not sys.platform.startswith("win"):
    raise RuntimeError("Cellonaut release packages are currently built for Windows only.")
icon_file = assets_dir / "icon.ico"

block_cipher = None

# Collect submodules only for packages that load them dynamically.
hiddenimports = [
    "imagej",
    "imglyb",
    "scyjava",
    "jpype",
    "jpype.imports",
    "jgo",
    "cjdk",
    "labeling",
    "skimage.measure",
    "skimage.filters",
    "skimage.segmentation",
    "skimage.morphology",
    "skimage.feature",
    "scipy.ndimage",
    "PIL._tkinter_finder",
]

for pkg in [
    "cellpose",
    "torch",
    "torchvision",
    "triton",
    "nd2",
]:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception as exc:
        print(f"WARNING: Could not collect submodules for {pkg}: {exc}", file=sys.stderr)

hiddenimports = list(dict.fromkeys(hiddenimports))

datas = []
binaries = []

required_offline_assets = [
    offline_assets_dir / "Fiji.app",
    offline_assets_dir / "cellpose_models",
    offline_assets_dir / "OFFLINE_ASSETS.json",
]
missing_offline_assets = [str(path) for path in required_offline_assets if not path.exists()]
if missing_offline_assets:
    raise RuntimeError(
        "Offline release assets are missing. Run the release build preparation first: "
        + ", ".join(missing_offline_assets)
    )
datas.extend(
    [
        (str(offline_assets_dir / "Fiji.app"), "offline/Fiji.app"),
        (str(offline_assets_dir / "cellpose_models"), "offline/cellpose_models"),
        (str(offline_assets_dir / "OFFLINE_ASSETS.json"), "offline"),
    ]
)

# Bundle the package's read-only resources.
for folder_name in ("assets", "licenses", "presets"):
    folder = package_data_dir / folder_name
    if folder.exists():
        datas.append((str(folder), f"cellonaut/data/{folder_name}"))

for pkg in [
    "cellpose",
    "tifffile",
    "nd2",
    "imagej",
    "imglyb",
    "scyjava",
    "jgo",
    "cjdk",
    "matplotlib",
    "skimage",
    "scipy",
    "PIL",
    "triton",
]:
    try:
        if pkg == "triton":
            datas += collect_data_files(pkg, include_py_files=True)
        else:
            datas += collect_data_files(pkg)
    except Exception as exc:
        print(f"WARNING: Could not collect data files for {pkg}: {exc}", file=sys.stderr)

# Qt 6.10 PySide wheels on Windows import the unsuffixed Windows ICU symbols
# from System32. Do not bundle conda ICU DLLs here; they export suffixed symbols
# such as ucnv_open_73 and cause "specified procedure could not be found".
library_bin = Path(sys.executable).resolve().parent.parent / "Library" / "bin"
if library_bin.exists():
    for dll in library_bin.glob("zstd*.dll"):
        binaries.append((str(dll), "."))

binaries = [item for item in binaries if not Path(item[0]).name.lower().startswith("icu")]
datas = [item for item in datas if not Path(item[0]).name.lower().startswith("icu")]

for pkg in [
    "numpy",
    "scipy",
    "skimage",
    "torch",
    "torchvision",
    "triton",
    "jpype",
]:
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception as exc:
        print(f"WARNING: Could not collect dynamic libraries for {pkg}: {exc}", file=sys.stderr)

# imagej and scyjava read package metadata at runtime.
for pkg in [
    "pyimagej",
    "imagej",
    "imglyb",
    "scyjava",
    "jpype1",
    "jgo",
    "cjdk",
    "labeling",
    "requests",
    "urllib3",
    "nd2",
    "cellpose",
    "tifffile",
    "numpy",
    "pandas",
    "scipy",
    "scikit-image",
    "matplotlib",
    "pillow",
    "pytorch-triton-rocm",
    "triton",
]:
    try:
        datas += copy_metadata(pkg)
    except Exception as exc:
        print(f"WARNING: Could not copy metadata for {pkg}: {exc}", file=sys.stderr)

runtime_hooks = []
runtime_hook = release_tools_dir / "pyinstaller" / "pyi_rth_cellonaut_env.py"
if runtime_hook.exists():
    runtime_hooks.append(str(runtime_hook))

analysis = Analysis(
    [str(project_dir / "cellonaut_app.py")],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        "matplotlib": {
            "backends": ["Agg"],
        },
    },
    runtime_hooks=runtime_hooks,
    excludes=[
        "matplotlib.tests",
        "numpy.tests",
        "pandas.tests",
        "scipy.tests",
        "pytest",
        "cupy",
        "tensorboard",
        "tkinter",
        "_tkinter",
        "brotli",
        "brotlicffi",
        "PySide6.scripts.deploy_lib",
        # Cellonaut uses Qt Widgets; exclude optional Qt stacks imported by
        # scientific dependencies.
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DRender",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtDesigner",
        "PySide6.QtGraphs",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQuickControls2",
        "PySide6.QtQuickTest",
        "PySide6.QtQuickWidgets",
        "PySide6.QtWebChannel",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtWebEngineWidgets",
        "skimage.io",
        "skimage.io._plugins",
        "skimage.viewer",
        "skimage.future.graph",
        "skimage.feature._orb_descriptor_positions",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

# QtGui collects plugins that pull Qt Quick, QML, and PDF into the build.
# Remove them after Analysis classifies the hook output.
unused_qt_binary_keys = {
    "qt6pdf",
    "qt6qml",
    "qt6qmlmeta",
    "qt6qmlmodels",
    "qt6qmlworkerscript",
    "qt6quick",
    "qt6virtualkeyboard",
}
unused_qt_plugin_keys = {
    "qpdf",
    "qtuiotouchplugin",
    "qtvirtualkeyboardplugin",
}


def _binary_key(destination):
    name = Path(destination).name.casefold().removeprefix("lib")
    return name.split(".dll", 1)[0]


def _is_unused_qt_binary(item):
    destination = str(item[0]).replace("\\", "/").casefold()
    key = _binary_key(destination)
    return key in unused_qt_binary_keys or key in unused_qt_plugin_keys


def _is_unused_qt_data(item):
    if _is_unused_qt_binary(item):
        return True
    destination = str(item[0]).replace("\\", "/").casefold()
    # Cellonaut does not install a QTranslator, so omit Qt translations.
    return destination.startswith("pyside6/") and "/translations/" in destination


analysis.binaries = [item for item in analysis.binaries if not _is_unused_qt_binary(item)]
analysis.datas = [item for item in analysis.datas if not _is_unused_qt_data(item)]

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Cellonaut",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(icon_file) if icon_file.exists() else None,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Cellonaut",
)
