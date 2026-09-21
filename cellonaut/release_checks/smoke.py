"""Verify that packaged runtime modules can be imported before distribution."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


BASE_IMPORTS = [
    "numpy",
    "pandas",
    "tifffile",
    "skimage",
    "scipy",
    "packaging.version",
    "xarray",
    "PySide6",
    "cellonaut.pipeline.models",
    "cellonaut.pipeline.planning",
    "cellonaut.dependency_versions",
    "cellonaut.masks.cellpose",
    "cellonaut.io.fiji_tiff",
    "cellonaut.io.image_io",
    "cellonaut.io.imagej_runtime",
    "cellonaut.measurement.execution",
    "cellonaut.masks.overlay_exports",
    "cellonaut.pipeline.runner",
    "cellonaut.resources",
    "cellonaut.results.layout",
    "cellonaut.masks.roi_processing",
]

HEAVY_IMPORTS = [
    "imagej",
    "scyjava",
    "jpype",
    "nd2",
    "torch",
    "torchvision",
    "cellpose",
]


def validate_packaged_offline_assets() -> list[str]:
    """Confirm a frozen offline app resolves the Fiji and Cellpose payloads."""
    if os.environ.get("CELLONAUT_OFFLINE", "").strip() != "1":
        return []

    from cellonaut.io.fiji_installation import scan_fiji_installation
    from cellonaut.release_checks.offline_assets import (
        CELLPOSE_MODEL_NAMES,
        REMOVED_CELLPOSE_MODEL_NAMES,
        _sha256,
    )
    from cellonaut.resources import get_bundled_fiji_path, get_resource_path

    failures: list[str] = []
    fiji_path = get_bundled_fiji_path()
    if fiji_path is None:
        failures.append("offline assets: bundled Fiji.app could not be resolved")
    else:
        status = scan_fiji_installation(fiji_path)
        if not status.ready:
            missing = ", ".join(item.label for item in status.missing_required) or "Fiji runtime"
            failures.append(f"offline assets: bundled Fiji is incomplete ({missing})")

    model_root = Path(os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", ""))
    manifest_path = get_resource_path("offline/OFFLINE_ASSETS.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_manifest = manifest["cellpose_models"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        failures.append(f"offline assets: invalid or missing manifest ({exc})")
        model_manifest = {}

    for name in CELLPOSE_MODEL_NAMES:
        model_path = model_root / name
        metadata = model_manifest.get(name, {}) if isinstance(model_manifest, dict) else {}
        expected_bytes = metadata.get("bytes") if isinstance(metadata, dict) else None
        expected_sha256 = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not model_path.is_file():
            failures.append(f"offline assets: Cellpose model is missing ({name})")
        elif not isinstance(expected_bytes, int) or model_path.stat().st_size != expected_bytes:
            failures.append(f"offline assets: Cellpose model size does not match manifest ({name})")
        elif not isinstance(expected_sha256, str) or _sha256(model_path) != expected_sha256:
            failures.append(f"offline assets: Cellpose model SHA-256 does not match manifest ({name})")
    for name in REMOVED_CELLPOSE_MODEL_NAMES:
        if (model_root / name).exists():
            failures.append(f"offline assets: unsupported DINO model is present ({name})")

    return failures


def validate_windows_cuda_inference(
    *,
    windows_release: bool | None = None,
    torch_module=None,
    model_factory=None,
) -> tuple[str, list[str]]:
    """Exercise the bundled CUDA model on an NVIDIA host; report an honest skip elsewhere."""
    if windows_release is None:
        windows_release = False
        if sys.platform.startswith("win") and getattr(sys, "frozen", False):
            profile_marker = Path(sys.executable).resolve().parent / "BUILD_PROFILE.txt"
            try:
                windows_release = profile_marker.read_text(encoding="ascii").strip().casefold() == "cuda126"
            except OSError:
                pass
    if not windows_release:
        return "not_applicable", []

    try:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]
        cuda_version = str(getattr(torch_module.version, "cuda", "") or "")
        if cuda_version != "12.6":
            return "failed", [f"CUDA runtime: expected CUDA 12.6, found {cuda_version or 'none'}"]
        if not torch_module.cuda.is_available():
            return "no_compatible_gpu", []

        import numpy as np
        from cellonaut.cell_segmentation.core import (
            CellSegmentationConfig,
            clear_cellpose_model_cache,
            get_cellpose_model,
        )
        from cellonaut.config.defaults import CELLPOSE_MODEL_OPTIONS

        try:
            model = (
                model_factory()
                if model_factory is not None
                else get_cellpose_model(CellSegmentationConfig(model_type=CELLPOSE_MODEL_OPTIONS[0], use_gpu=True))
            )
            if getattr(getattr(model, "device", None), "type", "") != "cuda":
                raise RuntimeError("Cellpose did not select a CUDA device")
            image = np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96)
            masks, _, _ = model.eval(image, diameter=30, min_size=5)
            if np.asarray(masks).shape != image.shape:
                raise RuntimeError("Cellpose returned an unexpected mask shape")
            torch_module.cuda.synchronize()
        finally:
            clear_cellpose_model_cache()
    except Exception as exc:
        return "failed", [f"CUDA inference: {exc}"]
    return "passed", []


def report_failures(failures: list[str]) -> None:
    """Write release-check failures to captured build logs as well as JSON."""
    for failure in failures:
        print(f"FAIL {failure}", flush=True)


def import_modules(module_names: list[str]) -> list[str]:
    """Import every requested module and report all failures in one pass."""
    failures = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            print(f"OK import {module_name}")
        except Exception as exc:
            failures.append(f"{module_name}: {exc}")
            print(f"FAIL import {module_name}: {exc}")
    return failures


# Keep heavy Fiji and Cellpose imports optional for quick packaging iteration,
# while making the complete runtime check the default release behavior.
def main(argv: list[str] | None = None) -> int:
    """Run the packaged-import smoke gate and return a process-friendly result."""
    parser = argparse.ArgumentParser(description="Smoke-test Cellonaut packaging imports.")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip Fiji/Cellpose/Torch import checks.")
    args = parser.parse_args(argv)

    failures = import_modules(BASE_IMPORTS)
    if not args.skip_heavy:
        failures.extend(import_modules(HEAVY_IMPORTS))

    if failures:
        print("\nPackaging smoke test failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPackaging smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
