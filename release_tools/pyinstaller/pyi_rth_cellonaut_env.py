import os
import sys
from pathlib import Path

# Use a noninteractive plotting backend and limit native worker threads.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

if getattr(sys, "frozen", False):
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    app_dir = Path(sys.executable).resolve().parent
    pyside_dir = bundle_root / "PySide6"
    shiboken_dir = bundle_root / "shiboken6"
    offline_root = bundle_root / "offline"
    cellpose_models = offline_root / "cellpose_models"
    os.environ["CELLONAUT_OFFLINE"] = "1"
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(cellpose_models)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dll_dirs = [p for p in (bundle_root, pyside_dir, shiboken_dir, app_dir) if p.exists()]
    for dll_dir in dll_dirs:
        try:
            os.add_dll_directory(str(dll_dir))  # pyright: ignore[reportAttributeAccessIssue]
        except OSError:
            pass

    if dll_dirs:
        bundled_path = os.pathsep.join(str(p) for p in dll_dirs)
        # Defensive cleanup only: Cellonaut does not require Anaconda, and
        # frozen lab builds should not depend on Anaconda Distribution DLLs.
        existing_paths = [
            part
            for part in os.environ.get("PATH", "").split(os.pathsep)
            if part
            and not (
                "anaconda" in part.lower()
                and (
                    "site-packages\\pyside6" in part.lower()
                    or "library\\bin" in part.lower()
                    or part.lower().endswith("\\anaconda3")
                    or "\\anaconda3\\scripts" in part.lower()
                    or "\\anaconda3\\envs\\" in part.lower()
                )
            )
        ]
        os.environ["PATH"] = os.pathsep.join([bundled_path, *existing_paths])

    plugins_dir = pyside_dir / "plugins"
    if plugins_dir.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins_dir))
