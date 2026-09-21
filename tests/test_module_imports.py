from __future__ import annotations

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_refactored_pipeline_modules_import_cleanly():
    module_names = [
        "cellonaut.pipeline.models",
        "cellonaut.pipeline.planning",
        "cellonaut.io.fiji_tiff",
        "cellonaut.io.fiji_installation",
        "cellonaut.io.image_io",
        "cellonaut.io.imagej_runtime",
        "cellonaut.io.java_runtime",
        "cellonaut.masks.roi_processing",
        "cellonaut.masks.cellpose",
        "cellonaut.measurement.execution",
        "cellonaut.masks.overlay_exports",
        "cellonaut.pipeline.runner",
    ]

    imported = [importlib.import_module(module_name).__name__ for module_name in module_names]

    assert imported == module_names


def test_python_packaging_library_is_not_shadowed_by_release_tools():
    module = importlib.import_module("packaging.version")
    module_file = module.__file__
    assert module_file is not None
    module_path = Path(module_file).resolve()
    project_packaging_dir = PROJECT_ROOT / "packaging"

    assert project_packaging_dir not in module_path.parents
