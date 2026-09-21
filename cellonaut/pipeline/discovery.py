"""Input-folder discovery and sample grouping rules.

Cellonaut accepts several TIFF layouts. This module detects those layouts,
finds matching channel files, and builds stable sample labels without relying on
GUI state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from cellonaut.artifact_naming import relative_artifact_stem
from cellonaut.config.defaults import (
    INPUT_STRUCTURE_FLAT_TIFFS,
    INPUT_STRUCTURE_GROUPED_BY_PROTEIN,
    INPUT_STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    INPUT_STRUCTURE_SAMPLES_DIRECTLY,
)
from cellonaut.results.layout import build_results_layout

STRUCTURE_SAMPLES_DIRECTLY = INPUT_STRUCTURE_SAMPLES_DIRECTLY
STRUCTURE_GROUPED_BY_PROTEIN = INPUT_STRUCTURE_GROUPED_BY_PROTEIN
STRUCTURE_FLAT_TIFFS = INPUT_STRUCTURE_FLAT_TIFFS
STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS = INPUT_STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS

TIFF_SUFFIXES = {".tif", ".tiff"}
TiffPresenceCache = dict[tuple[str, bool], bool]
IGNORED_SCAN_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "installer_dist",
}


def safe_iterdir(folder: Path) -> list[Path]:
    """List a directory, treating inaccessible locations as empty."""

    try:
        return list(folder.iterdir())
    except (OSError, PermissionError):
        return []


def limited_dir_entries(
    folder: Path,
    *,
    max_entries: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[list[Path], bool]:
    """Read at most ``max_entries`` entries and report cancellation or truncation."""
    entries: list[Path] = []
    try:
        for path in folder.iterdir():
            if (cancel_requested is not None and cancel_requested()) or len(entries) >= max_entries:
                return entries, True
            entries.append(path)
    except (OSError, PermissionError):
        return [], False
    return entries, False


def directory_scan_key(folder: Path) -> str | None:
    """Return a resolved identity used to prevent recursive alias cycles."""

    try:
        resolved = Path(folder).resolve()
    except RuntimeError:
        return None
    except OSError:
        resolved = Path(folder).absolute()
    return os.path.normcase(str(resolved))


def is_ignored_scan_dir(path: Path) -> bool:
    """Return whether a directory is excluded from microscopy discovery."""

    return path.name.lower() in IGNORED_SCAN_DIRS


def scan_matching_files(
    folder: Path,
    *,
    recursive: bool,
    predicate: Callable[[Path], bool],
    iterdir_func: Callable[[Path], list[Path]] | None = None,
    ignore_dir_func: Callable[[Path], bool] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Find matching files with optional recursion and cycle safety; callers choose sorting."""

    if not folder.exists() or not folder.is_dir() or (limit is not None and limit <= 0):
        return []

    list_entries = iterdir_func or safe_iterdir
    ignore_dir = ignore_dir_func or is_ignored_scan_dir
    if not recursive:
        files: list[Path] = []
        for path in list_entries(folder):
            if not predicate(path):
                continue
            files.append(path)
            if limit is not None and len(files) >= limit:
                break
        return files

    files = []
    pending = [folder]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        key = directory_scan_key(current)
        if key is None or key in visited:
            continue
        visited.add(key)
        for path in list_entries(current):
            if path.is_dir() and not ignore_dir(path):
                pending.append(path)
            elif predicate(path):
                files.append(path)
                if limit is not None and len(files) >= limit:
                    return files
    return files


def scan_matching_files_bounded(
    folder: Path,
    *,
    predicate: Callable[[Path], bool],
    max_dirs: int,
    max_entries: int,
    cancel_requested: Callable[[], bool] | None = None,
    iterdir_func: Callable[..., tuple[list[Path], bool]] | None = None,
    ignore_dir_func: Callable[[Path], bool] | None = None,
    directory_key_func: Callable[[Path], str | None] | None = None,
) -> tuple[list[Path], bool]:
    """Recursively scan with shared directory, entry, cancellation, and cycle limits."""
    list_entries = iterdir_func or limited_dir_entries
    ignore_dir = ignore_dir_func or is_ignored_scan_dir
    scan_key = directory_key_func or directory_scan_key
    max_dirs = max(1, int(max_dirs))
    max_entries = max(1, int(max_entries))
    files: list[Path] = []
    pending = [folder]
    visited: set[str] = set()
    dirs_seen = 0
    entries_seen = 0
    truncated = False

    while pending:
        if cancel_requested is not None and cancel_requested():
            return files, True
        if dirs_seen >= max_dirs or entries_seen >= max_entries:
            return files, True

        current = pending.pop()
        key = scan_key(current)
        if key is None or key in visited:
            continue
        visited.add(key)
        dirs_seen += 1

        entries, entry_truncated = list_entries(
            current,
            max_entries=max_entries - entries_seen,
            cancel_requested=cancel_requested,
        )
        entries_seen += len(entries)
        truncated = truncated or entry_truncated
        for path in entries:
            try:
                if path.is_dir() and not ignore_dir(path):
                    pending.append(path)
                elif predicate(path):
                    files.append(path)
            except OSError:
                continue
        if truncated:
            break

    return files, truncated


def scan_child_dirs(folder: Path) -> list[Path]:
    """Return accessible child directories in reproducible alphabetical order."""

    return sorted(
        [path for path in safe_iterdir(folder) if path.is_dir() and not is_ignored_scan_dir(path)],
        key=lambda path: path.name.lower(),
    )


def scan_child_dirs_in_folder_order(folder: Path) -> list[Path]:
    """Order child directories by st_ctime, then name, as a metadata fallback.

    st_ctime represents creation time on Windows and metadata-change time on Unix.
    """

    paths = [path for path in safe_iterdir(folder) if path.is_dir() and not is_ignored_scan_dir(path)]
    try:
        return sorted(paths, key=lambda path: (path.stat().st_ctime_ns, path.name))
    except OSError:
        return paths


def is_supported_tiff_file(path: Path) -> bool:
    """Return whether a path has a supported TIFF filename extension."""

    if not path.is_file():
        return False
    if path.name.startswith("._"):
        return False
    return path.suffix.lower() in TIFF_SUFFIXES


# Stop after finding one TIFF and cache the answer only for this detection pass.
def contains_supported_tiff(
    folder: Path,
    recursive: bool = False,
    *,
    cache: TiffPresenceCache | None = None,
) -> bool:
    """Check for a TIFF with early exit and optional per-scan caching."""

    folder_key = directory_scan_key(folder)
    cache_key = (folder_key, recursive) if folder_key is not None else None
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]

    found = bool(
        scan_matching_files(
            folder,
            recursive=recursive,
            predicate=is_supported_tiff_file,
            iterdir_func=safe_iterdir,
            limit=1,
        )
    )
    if cache is not None and cache_key is not None:
        cache[cache_key] = found
    return found


def iter_supported_tiff_files(folder: Path, recursive: bool = False) -> list[Path]:
    """Return supported TIFF files in stable relative-path order."""

    files = scan_matching_files(
        folder,
        recursive=recursive,
        predicate=is_supported_tiff_file,
        iterdir_func=safe_iterdir,
    )
    if recursive:
        return sorted(files, key=lambda path: relative_path_sort_key(folder, path))
    return sorted(files, key=lambda path: path.name.lower())


def relative_path_sort_key(root: Path, path: Path) -> str:
    """Return a case-insensitive relative key for stable nested ordering."""

    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix().lower()
    except (OSError, ValueError):
        return Path(path).as_posix().lower()


def sample_relative_parent(
    input_dir: Path,
    sample_path: Path,
    input_structure: str,
) -> Path:
    """Return the source subfolder that recursive flat-TIFF exports must mirror."""

    if input_structure != STRUCTURE_FLAT_TIFFS:
        return Path()
    try:
        relative = Path(sample_path).resolve().parent.relative_to(Path(input_dir).resolve())
    except (OSError, ValueError):
        return Path()
    return Path() if str(relative) == "." else relative


def safe_result_name_from_relative_path(path: Path) -> str:
    """Convert a relative source path into one filesystem-safe result name."""

    return relative_artifact_stem(path)


def find_first_image_file(
    folder: Path, log_func: Callable[[str], None],
    ambiguity_func: Callable[[str], None] | None = None,
) -> Path | None:
    """Choose one TIFF deterministically from a channel folder."""

    if not folder.is_dir():
        return None
    files = sorted([p for p in safe_iterdir(folder) if is_supported_tiff_file(p)])
    if not files:
        return None
    if len(files) > 1:
        message = f"[WARN] Multiple TIFF files found in {folder}, using: {files[-1].name}"
        log_func(message)
        if ambiguity_func is not None:
            ambiguity_func(message)
    return files[-1]


def find_matching_image_file_by_stem(folder: Path, sample_stem: str) -> Path | None:
    """Find a channel-folder TIFF by stem regardless of TIFF extension spelling."""

    if not folder.is_dir():
        return None
    for path in sorted([p for p in safe_iterdir(folder) if is_supported_tiff_file(p)]):
        if path.stem == sample_stem:
            return path
    return None


def get_subfolder(parent: Path, name: str) -> Path | None:
    """Return an existing child directory, or None when the channel is absent."""

    path = parent / name
    return path if path.exists() and path.is_dir() else None


def get_image_folder_flat_tiff_sample_stems(input_dir: Path, image_folder_names: list[str]) -> list[str]:
    """Return the union of sample stems across flat channel folders."""

    stems: set[str] = set()
    for folder_name in image_folder_names:
        folder = input_dir / folder_name
        if not folder.is_dir():
            continue
        for path in safe_iterdir(folder):
            if is_supported_tiff_file(path):
                stems.add(path.stem)
    return sorted(stems)


def get_sample_paths_for_structure(
    input_dir: Path,
    input_structure: str,
    image_folder_names: list[str] | None = None,
    *,
    tiff_presence_cache: TiffPresenceCache | None = None,
) -> list[Path]:
    """Enumerate one stable source path per logical sample for an input layout."""

    if not input_dir.is_dir():
        return []

    if input_structure == STRUCTURE_SAMPLES_DIRECTLY:
        return scan_child_dirs(input_dir)

    if input_structure == STRUCTURE_GROUPED_BY_PROTEIN:
        sample_list: list[Path] = []
        for protein_folder in scan_child_dirs(input_dir):
            sample_list.extend(scan_child_dirs(protein_folder))
        return sample_list

    if input_structure == STRUCTURE_FLAT_TIFFS:
        return iter_supported_tiff_files(input_dir, recursive=True)

    if input_structure == STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS:
        folders = image_folder_names or [
            p.name
            for p in scan_child_dirs(input_dir)
            if contains_supported_tiff(p, recursive=False, cache=tiff_presence_cache)
        ]
        return [input_dir / f"{stem}.tif" for stem in get_image_folder_flat_tiff_sample_stems(input_dir, folders)]

    return []


def get_measurement_sample_paths(cfg: Any) -> list[Path]:
    """Enumerate samples from the same structure used by pipeline execution."""

    return get_sample_paths_for_structure(
        cfg.input_dir,
        cfg.input_structure,
        [img.folder_name for img in cfg.images],
    )


# Folder names remain the fallback channel labels when TIFF metadata is unavailable.
def _folder_names_from_defs(active_image_defs: list[dict[str, Any]] | None) -> list[str]:
    names = []
    for img in active_image_defs or []:
        name = str(img.get("folder", "") or img.get("name", "") or "").strip()
        if name:
            names.append(name)
    return names


# Scores favor channel folders repeated consistently across many samples.
def _channel_score_for_samples(
    sample_dirs: list[Path],
    *,
    tiff_presence_cache: TiffPresenceCache | None = None,
) -> dict[str, int]:
    scores: dict[str, int] = {}
    for sample in sample_dirs:
        for channel_dir in scan_child_dirs_in_folder_order(sample):
            has_images = contains_supported_tiff(
                channel_dir,
                recursive=False,
                cache=tiff_presence_cache,
            )
            has_nested_images = has_images or contains_supported_tiff(
                channel_dir,
                recursive=True,
                cache=tiff_presence_cache,
            )
            if has_images or has_nested_images:
                scores[channel_dir.name] = scores.get(channel_dir.name, 0) + 1
    return scores


def detect_image_folders_from_input(
    root: Path,
    expected_folders: list[str] | None = None,
    active_image_defs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Detect likely input structure and channel folders with supporting evidence."""

    if not root.exists() or not root.is_dir():
        return {"folders": [], "structure": "", "sample_count": 0, "scores": {}}

    tiff_presence_cache: TiffPresenceCache = {}
    flat_samples = iter_supported_tiff_files(root, recursive=False)
    if flat_samples:
        active_names = _folder_names_from_defs(active_image_defs)
        first_name = active_names[0] if active_names else "Channel 1"
        return {
            "folders": [first_name],
            "structure": STRUCTURE_FLAT_TIFFS,
            "sample_count": len(flat_samples),
            "scores": {first_name: len(flat_samples)},
        }

    folders = expected_folders or [
        p.name
        for p in scan_child_dirs_in_folder_order(root)
        if contains_supported_tiff(p, recursive=False, cache=tiff_presence_cache)
    ]
    image_folder_samples = get_sample_paths_for_structure(
        root,
        STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
        folders,
        tiff_presence_cache=tiff_presence_cache,
    )
    if image_folder_samples:
        detected_folders = [
            p.name
            for p in scan_child_dirs_in_folder_order(root)
            if contains_supported_tiff(p, recursive=False, cache=tiff_presence_cache)
        ]
        if expected_folders:
            detected_folders = [name for name in expected_folders if (root / name).is_dir()]
        return {
            "folders": detected_folders,
            "structure": STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
            "sample_count": len(image_folder_samples),
            "scores": {name: len(image_folder_samples) for name in detected_folders},
        }

    direct_samples = get_sample_paths_for_structure(root, STRUCTURE_SAMPLES_DIRECTLY)
    grouped_samples = get_sample_paths_for_structure(root, STRUCTURE_GROUPED_BY_PROTEIN)

    direct_scores = _channel_score_for_samples(
        direct_samples,
        tiff_presence_cache=tiff_presence_cache,
    )
    grouped_scores = _channel_score_for_samples(
        grouped_samples,
        tiff_presence_cache=tiff_presence_cache,
    )

    direct_total = sum(direct_scores.values())
    grouped_total = sum(grouped_scores.values())

    if grouped_total > direct_total:
        structure = STRUCTURE_GROUPED_BY_PROTEIN
        scores = grouped_scores
        sample_count = len(grouped_samples)
    else:
        structure = STRUCTURE_SAMPLES_DIRECTLY
        scores = direct_scores
        sample_count = len(direct_samples)

    folders = list(scores.keys())

    return {
        "folders": folders,
        "structure": structure,
        "sample_count": sample_count,
        "scores": scores,
    }


def get_protein_name_for_sample(sample_folder: Path, input_structure: str) -> str:
    """Return the experimental group for grouped input, otherwise ROOT."""

    if input_structure == STRUCTURE_GROUPED_BY_PROTEIN and sample_folder.parent is not None:
        return sample_folder.parent.name
    return "ROOT"


def build_sample_label(sample_folder: Path, input_structure: str) -> str:
    """Build a human-facing sample label that disambiguates grouped samples."""

    sample = (
        sample_folder.stem
        if input_structure in {STRUCTURE_FLAT_TIFFS, STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS}
        else sample_folder.name
    )
    protein = get_protein_name_for_sample(sample_folder, input_structure)
    return sample if protein == "ROOT" else f"{protein}_{sample}"


def build_result_id(
    sample_folder: Path,
    input_structure: str,
    input_dir: Path | None = None,
) -> str:
    """Build a portable result identifier that preserves nested sample context."""

    if input_structure in {STRUCTURE_FLAT_TIFFS, STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS}:
        if input_structure == STRUCTURE_FLAT_TIFFS and input_dir is not None:
            try:
                relative = Path(sample_folder).resolve().relative_to(Path(input_dir).resolve()).with_suffix("")
                return safe_result_name_from_relative_path(relative)
            except (OSError, ValueError):
                pass
        return safe_result_name_from_relative_path(Path(sample_folder.stem))

    return safe_result_name_from_relative_path(Path(build_sample_label(sample_folder, input_structure)))


def validate_unique_result_ids(
    sample_paths: list[Path],
    input_structure: str,
    *,
    input_dir: Path,
) -> None:
    """Reject sample sets whose portable result identifiers would collide."""

    seen: dict[str, tuple[str, Path]] = {}
    for sample_path in sample_paths:
        result_id = build_result_id(sample_path, input_structure, input_dir=input_dir)
        key = result_id.casefold()
        previous = seen.get(key)
        if previous is not None and previous[1] != sample_path:
            raise ValueError(
                "Two input samples would use the same output name after filename sanitization: "
                f"{previous[1]} and {sample_path} -> {result_id}. "
                "Rename one source sample before running the pipeline."
            )
        seen[key] = (result_id, sample_path)


def nest_export_dirs_for_sample(
    export_dirs: dict[str, Path],
    input_dir: Path,
    sample_folder: Path,
    input_structure: str,
) -> dict[str, Path]:
    """Mirror a recursive flat-TIFF source folder beneath every export category."""

    relative_parent = sample_relative_parent(input_dir, sample_folder, input_structure)
    if relative_parent == Path():
        return dict(export_dirs)
    return {key: path / relative_parent for key, path in export_dirs.items()}


def build_common_results_export_dirs(output_dir: Path) -> dict[str, Path]:
    """Return the shared export-directory map used by all pipeline stages."""

    layout = build_results_layout(output_dir, create_root=False)
    return {
        "masks": layout["masks"],
        "weka_probability_maps": layout["weka_probability_maps"],
        "weka_threshold_masks": layout["weka_threshold_masks"],
        "final_binary_masks": layout["final_binary_masks"],
        "mask_skeletons": layout["mask_skeletons"],
        "mask_overlays": layout["mask_overlays"],
        "qc_overlay_pngs": layout["qc_overlay_pngs"],
        "cell_segmentation_labels": layout["cell_segmentation_labels"],
        "cell_segmentation_outlines": layout["cell_segmentation_outlines"],
        "cell_segmentation_qc_pngs": layout["cell_segmentation_qc_pngs"],
        "cell_segmentation_tables": layout["cell_segmentation_tables"],
        "cell_signal_tables": layout["cell_signal_tables"],
        "cell_geometry_tables": layout["cell_geometry_tables"],
        "processing_montages": layout["processing_montages"],
    }
