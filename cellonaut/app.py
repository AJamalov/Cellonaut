"""Application entry point for the Cellonaut desktop GUI."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

from cellonaut.version import APP_NAME, __version__, format_app_version


# Windows otherwise groups the packaged app under the Python launcher and may
# show Python's icon in the taskbar instead of Cellonaut's. Failure here is
# cosmetic, so unsupported Windows variants must still be allowed to start.
def configure_windows_app_identity() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        app_id = f"Cellonaut.{APP_NAME}.{__version__}"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)  # pyright: ignore[reportAttributeAccessIssue]
    except (AttributeError, OSError):
        pass


# GUI imports stay out of the command-line entry path so lightweight options
# such as --version work before Qt is loaded. Relaunching first also keeps Qt
# and the scientific packages in the same Python environment.
def _run_gui(argv: list[str]) -> int:
    from cellonaut.gui.runtime import install_crash_logging, relaunch_with_project_venv

    relaunch_with_project_venv()

    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
        from cellonaut.config.defaults import DEFAULT_THEME
        from cellonaut.gui.main_window import CellonautMainWindow
        from cellonaut.gui.theme import apply_theme
        from cellonaut.resources import APP_ICON_FILE, get_resource_path
    except ImportError as exc:
        raise RuntimeError("Cellonaut GUI dependencies could not be loaded.") from exc

    install_crash_logging()
    configure_windows_app_identity()
    app = QApplication([sys.argv[0], *argv])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)

    apply_theme(app, DEFAULT_THEME)

    icon_path = get_resource_path(APP_ICON_FILE)
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = CellonautMainWindow()
    window.finish_startup_initialization()
    window.ensurePolished()
    window.showMaximized()
    return app.exec()


def _run_release_smoke(output_path: Path, *, skip_heavy: bool = False) -> int:
    """Exercise bundled runtime imports and persist a machine-readable result."""
    from cellonaut.release_checks.smoke import (
        BASE_IMPORTS,
        HEAVY_IMPORTS,
        import_modules,
        report_failures,
        validate_packaged_offline_assets,
    )

    failures = validate_packaged_offline_assets()
    failures.extend(import_modules(BASE_IMPORTS))
    if not skip_heavy:
        failures.extend(import_modules(HEAVY_IMPORTS))
    report_failures(failures)
    result = {"app_version": __version__, "failures": failures}
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


def _run_release_runtime_check(output_path: Path) -> int:
    """Check Fiji and Cellpose, including CUDA inference when hardware permits."""
    failures: list[str] = []
    try:
        from cellonaut.io.imagej_runtime import get_ij
        from cellonaut.resources import get_bundled_fiji_path

        fiji_path = get_bundled_fiji_path()
        if fiji_path is None:
            raise RuntimeError("bundled Fiji.app could not be resolved")
        imagej_runtime = get_ij(fiji_path, log_func=lambda _message: None)
        if imagej_runtime is None:
            raise RuntimeError("PyImageJ returned no initialized runtime")
    except Exception as exc:
        failures.append(f"Fiji runtime: {exc}")

    try:
        from cellonaut.cell_segmentation.core import (
            CellSegmentationConfig,
            clear_cellpose_model_cache,
            get_cellpose_model,
        )
        from cellonaut.config.defaults import CELLPOSE_MODEL_OPTIONS

        for model_name in CELLPOSE_MODEL_OPTIONS:
            try:
                get_cellpose_model(CellSegmentationConfig(model_type=model_name, use_gpu=False))
            except Exception as exc:
                failures.append(f"Cellpose model {model_name}: {exc}")
            finally:
                clear_cellpose_model_cache()
    except Exception as exc:
        failures.append(f"Cellpose runtime: {exc}")

    from cellonaut.release_checks.smoke import report_failures, validate_windows_cuda_inference

    cuda_inference, cuda_failures = validate_windows_cuda_inference()
    failures.extend(cuda_failures)

    report_failures(failures)
    result = {"app_version": __version__, "cuda_inference": cuda_inference, "failures": failures}
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


# Installed launchers can call this function without passing through
# `cellonaut.__main__`, so multiprocessing setup also belongs here. Unknown
# arguments are preserved because they may be platform-specific Qt flags.
def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Launch the Cellonaut desktop GUI.")
    parser.add_argument("--version", action="version", version=format_app_version())
    parser.add_argument("--release-smoke", type=Path, metavar="OUTPUT", help=argparse.SUPPRESS)
    parser.add_argument("--release-smoke-skip-heavy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--release-runtime-check", type=Path, metavar="OUTPUT", help=argparse.SUPPRESS)
    args, qt_args = parser.parse_known_args(argv)
    if args.release_smoke is not None:
        return _run_release_smoke(args.release_smoke, skip_heavy=args.release_smoke_skip_heavy)
    if args.release_runtime_check is not None:
        return _run_release_runtime_check(args.release_runtime_check)
    return _run_gui(qt_args)


if __name__ == "__main__":
    sys.exit(main())
