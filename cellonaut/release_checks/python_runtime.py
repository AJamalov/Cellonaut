"""Validate the exact Python interpreter used for official release builds."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ".python-version"


def expected_release_python(project_root: Path = PROJECT_ROOT) -> str:
    """Read the single release interpreter version recorded by the repository."""
    version_path = Path(project_root) / VERSION_FILE
    try:
        version = version_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read release Python version from {version_path}: {exc}") from exc
    if not version or any(part == "" or not part.isdigit() for part in version.split(".")):
        raise RuntimeError(f"Invalid release Python version in {version_path}: {version!r}")
    return version


def validate_release_python(project_root: Path = PROJECT_ROOT, *, actual: str | None = None) -> str:
    """Reject an interpreter that differs from the pinned release version."""
    expected = expected_release_python(project_root)
    actual = actual or platform.python_version()
    if actual != expected:
        raise RuntimeError(f"Official release builds require Python {expected}; this interpreter is Python {actual}.")
    return expected


def main(argv: list[str] | None = None) -> int:
    """Validate the active interpreter from a build script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        version = validate_release_python(args.project_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Release Python: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
