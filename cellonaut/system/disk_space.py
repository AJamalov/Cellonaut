"""Disk-capacity estimates used before analysis writes large output trees."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Iterable


MIB = 1024**2
GIB = 1024**3
HARD_MINIMUM_FREE_BYTES = 256 * MIB


@dataclass(frozen=True, slots=True)
class DiskSpaceEstimate:
    destination: Path
    free_bytes: int
    recommended_bytes: int
    source_bytes: int

    @property
    def critically_low(self) -> bool:
        return self.free_bytes < HARD_MINIMUM_FREE_BYTES

    @property
    def below_recommended(self) -> bool:
        return self.free_bytes < self.recommended_bytes


def format_bytes(value: int) -> str:
    """Return a compact binary-size label suitable for preflight dialogs."""
    size = max(0, int(value))
    if size >= GIB:
        return f"{size / GIB:.1f} GiB"
    if size >= MIB:
        return f"{size / MIB:.0f} MiB"
    return f"{size / 1024:.0f} KiB"


def nearest_existing_path(path: Path) -> Path:
    """Find the existing ancestor whose volume will contain a destination."""
    candidate = Path(path).expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise OSError(f"Could not resolve a storage volume for {path}")
    return candidate


def unique_file_bytes(paths: Iterable[Path]) -> int:
    """Sum readable files once; inaccessible and disappearing files are ignored."""
    total = 0
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        try:
            key = os.path.normcase(str(path.resolve()))
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            total += max(0, int(path.stat().st_size))
        except OSError:
            continue
    return total


def estimate_disk_space(
    destination: Path,
    *,
    source_bytes: int,
    output_multiplier: float,
    reserve_bytes: int,
) -> DiskSpaceEstimate:
    """Compare free space with a conservative source-derived recommendation."""
    volume_path = nearest_existing_path(destination)
    usage = shutil.disk_usage(volume_path)
    source_size = max(0, int(source_bytes))
    recommended = max(0, int(reserve_bytes)) + max(0, int(source_size * max(0.0, output_multiplier)))
    return DiskSpaceEstimate(
        destination=Path(destination),
        free_bytes=max(0, int(usage.free)),
        recommended_bytes=recommended,
        source_bytes=source_size,
    )


def readiness_source_files(readiness: dict, *, first_ready_only: bool = False) -> list[Path]:
    """Extract the resolved TIFF inputs already found by pipeline readiness checks."""
    paths: list[Path] = []
    for sample in readiness.get("samples", []) or []:
        if not isinstance(sample, dict) or not sample.get("ready"):
            continue
        files = sample.get("files", {}) or {}
        if isinstance(files, dict):
            paths.extend(Path(str(value)) for value in files.values() if str(value or "").strip())
        if first_ready_only:
            break
    return paths
