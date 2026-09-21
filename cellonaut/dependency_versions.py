"""Collect the focused dependency versions recorded in run summaries."""

from __future__ import annotations

from importlib import metadata

KEY_PACKAGE_NAMES = (
    "PySide6",
    "numpy",
    "pandas",
    "tifffile",
    "scikit-image",
    "scipy",
    "pillow",
    "matplotlib",
    "packaging",
    "nd2",
    "pyimagej",
    "scyjava",
    "jpype1",
    "cellpose",
    "torch",
    "torchvision",
)


# Diagnostics should remain available when an optional package is absent or
# its metadata is damaged, so version lookup never blocks startup.
def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"
    except Exception:
        return "unknown"


# Report only packages that affect the GUI, image pipeline, external runtimes,
# or scientific backends instead of dumping the full Python environment.
def collect_key_package_versions() -> dict[str, str]:
    return {name: _package_version(name) for name in KEY_PACKAGE_NAMES}
