"""File-browser helpers and bounded input-folder scanning for the GUI.

Input folders can be large network shares or raw microscopy exports, so this
module keeps layout detection, TIFF metadata checks, and ND2 hints in bounded
background scans instead of blocking the main window.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QModelIndex, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileSystemModel, QHeaderView, QMessageBox

import cellonaut.io.nd2_import as ndi
from cellonaut.runtime import PipelineStage, stage_update
from cellonaut.pipeline.discovery import (
    STRUCTURE_FLAT_TIFFS,
    STRUCTURE_GROUPED_BY_PROTEIN,
    STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    STRUCTURE_SAMPLES_DIRECTLY,
    detect_image_folders_from_input,
    directory_scan_key,
    is_ignored_scan_dir,
    iter_supported_tiff_files,
    is_supported_tiff_file,
    limited_dir_entries,
    scan_matching_files_bounded,
)
from cellonaut.config.relationships import (
    image_uses_combined_mask,
    remap_analysis_references,
    seed_first_classifier_self_relationship,
)
from cellonaut.gui.detected_channels import apply_detected_image_definitions
from cellonaut.io.image_io import get_tiff_stack_info
from cellonaut.io.path_keys import input_path_key
from cellonaut.gui.mixin import GuiMixin


def open_folder_in_system_browser(path: Path) -> None:
    target = path.parent if path.is_file() else path
    os.startfile(str(target))  # pyright: ignore[reportAttributeAccessIssue]


INPUT_SCAN_ND2_MAX_DIRS = 2000
INPUT_SCAN_ND2_MAX_ENTRIES = 50000
INPUT_SCAN_LAYOUT_MAX_DIRS = 600
INPUT_SCAN_LAYOUT_MAX_ENTRIES = 20000
INPUT_SCAN_STATUS_TIMEOUT_MS = 15000


# Every scan returns the same keys so GUI code does not need partial-result branches.
def _empty_input_scan_result(folder_path: str) -> dict:
    path = str(folder_path or "")
    return {
        "path": path,
        "path_key": input_path_key(path),
        "path_exists": False,
        "path_is_dir": False,
        "detection": {"folders": [], "structure": "", "sample_count": 0, "scores": {}},
        "first_tiff": "",
        "tiff_info": None,
        "nd2_count": 0,
        "nd2_first": "",
        "nd2_channels": [],
        "nd2_channel_colors": [],
        "nd2_sizes": {},
        "nd2_z_summary": {},
        "nd2_dataset_summary": {},
        "nd2_error": "",
        "nd2_scan_truncated": False,
        "layout_scan_truncated": False,
        "scan_cancelled": False,
        "scan_error": "",
    }


# Bound a single directory read because even a non-recursive network listing can be very large.
def _limited_dir_entries(
    folder: Path,
    *,
    max_entries: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[list[Path], bool]:
    return limited_dir_entries(folder, max_entries=max_entries, cancel_requested=cancel_requested)


# ND2 hints should never make path entry wait on an unrestricted recursive walk.
def _limited_nd2_files(
    root: Path,
    *,
    max_dirs: int = INPUT_SCAN_ND2_MAX_DIRS,
    max_entries: int = INPUT_SCAN_ND2_MAX_ENTRIES,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[list[Path], bool]:
    files, truncated = scan_matching_files_bounded(
        root,
        predicate=lambda path: path.is_file() and path.suffix.lower() == ".nd2",
        max_dirs=max_dirs,
        max_entries=max_entries,
        cancel_requested=cancel_requested,
        iterdir_func=_limited_dir_entries,
        ignore_dir_func=is_ignored_scan_dir,
        directory_key_func=directory_scan_key,
    )
    return sorted(files, key=lambda path: str(path).lower()), truncated


class _BackgroundScanBudget:
    """Share one directory and entry allowance across the whole layout heuristic."""

    def __init__(
        self,
        *,
        max_dirs: int,
        max_entries: int,
        cancel_requested: Callable[[], bool] | None = None,
    ):
        self.max_dirs = max(1, int(max_dirs))
        self.max_entries = max(1, int(max_entries))
        self.cancel_requested = cancel_requested
        self.dirs_seen = 0
        self.entries_seen = 0
        self.truncated = False
        self._entry_cache: dict[str, list[Path]] = {}

    @property
    def cancelled(self) -> bool:
        return bool(self.cancel_requested is not None and self.cancel_requested())

    # Cache directory listings because the heuristic asks separate TIFF and
    # child-folder questions about the same root and channel directories.
    def entries(self, folder: Path) -> list[Path]:
        key = directory_scan_key(folder)
        if key is None:
            return []
        if key in self._entry_cache:
            return self._entry_cache[key]
        if self.cancelled or self.dirs_seen >= self.max_dirs or self.entries_seen >= self.max_entries:
            self.truncated = True
            return []

        self.dirs_seen += 1
        remaining = self.max_entries - self.entries_seen
        entries, truncated = _limited_dir_entries(
            folder,
            max_entries=remaining,
            cancel_requested=self.cancel_requested,
        )
        self.entries_seen += len(entries)
        self.truncated = self.truncated or truncated or self.cancelled
        self._entry_cache[key] = entries
        return entries

    def direct_tiffs(self, folder: Path) -> list[Path]:
        return sorted(
            [path for path in self.entries(folder) if self.is_supported_tiff(path)],
            key=lambda path: path.name.lower(),
        )

    def direct_tiff_count(self, folder: Path) -> int:
        return sum(1 for path in self.entries(folder) if self.is_supported_tiff(path))

    @staticmethod
    def is_supported_tiff(path: Path) -> bool:
        try:
            return is_supported_tiff_file(path)
        except OSError:
            return False

    def child_dirs(self, folder: Path) -> list[Path]:
        dirs = []
        for path in self.entries(folder):
            try:
                if path.is_dir() and not is_ignored_scan_dir(path):
                    dirs.append(path)
            except OSError:
                continue
        if len(dirs) > self.max_dirs:
            dirs = dirs[: self.max_dirs]
            self.truncated = True
        try:
            return sorted(dirs, key=lambda path: (path.stat().st_ctime_ns, path.name))
        except OSError:
            return dirs


def _flat_tiff_detection(
    budget: _BackgroundScanBudget,
    root: Path,
    active_image_defs: list[dict],
) -> dict | None:
    direct_tiffs = budget.direct_tiffs(root)
    if not direct_tiffs:
        return None
    active_names = [
        str(item.get("folder", "") or item.get("name", "") or "").strip() for item in active_image_defs or []
    ]
    first_name = next((name for name in active_names if name), "Channel 1")
    return {
        "folders": [first_name],
        "structure": STRUCTURE_FLAT_TIFFS,
        "sample_count": len(direct_tiffs),
        "scores": {first_name: len(direct_tiffs)},
    }


def _image_folder_detection(
    budget: _BackgroundScanBudget,
    root: Path,
    child_dirs: list[Path],
    expected_folders: list[str],
    max_dirs: int,
) -> dict | None:
    folder_counts: dict[str, int] = {}
    candidate_groups = []
    if expected_folders:
        candidate_groups.append([root / name for name in expected_folders if str(name or "").strip()])
    candidate_groups.append(child_dirs)
    for candidates in candidate_groups:
        for folder in candidates[:max_dirs]:
            if budget.cancelled:
                break
            if not folder.is_dir():
                continue
            count = budget.direct_tiff_count(folder)
            if count:
                folder_counts[folder.name] = count
        if folder_counts:
            break

    if not folder_counts:
        return None
    return {
        "folders": list(folder_counts.keys()),
        "structure": STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
        "sample_count": max(folder_counts.values()),
        "scores": folder_counts,
    }


def _nested_tiff_detection(
    budget: _BackgroundScanBudget,
    child_dirs: list[Path],
    max_dirs: int,
) -> dict:
    direct_scores: dict[str, int] = {}
    grouped_scores: dict[str, int] = {}
    direct_sample_count = 0
    grouped_sample_count = 0
    for child in child_dirs[:max_dirs]:
        if budget.cancelled:
            break
        channel_dirs = budget.child_dirs(child)
        child_has_channel = False
        for channel in channel_dirs:
            count = budget.direct_tiff_count(channel)
            if count:
                direct_scores[channel.name] = direct_scores.get(channel.name, 0) + 1
                child_has_channel = True
        if child_has_channel:
            direct_sample_count += 1
            continue

        for grandchild in channel_dirs[:max_dirs]:
            if budget.cancelled:
                break
            grandchild_channels = budget.child_dirs(grandchild)
            grandchild_has_channel = False
            for channel in grandchild_channels:
                count = budget.direct_tiff_count(channel)
                if count:
                    grouped_scores[channel.name] = grouped_scores.get(channel.name, 0) + 1
                    grandchild_has_channel = True
            if grandchild_has_channel:
                grouped_sample_count += 1

    if sum(grouped_scores.values()) > sum(direct_scores.values()):
        return {
            "folders": list(grouped_scores.keys()),
            "structure": STRUCTURE_GROUPED_BY_PROTEIN,
            "sample_count": grouped_sample_count,
            "scores": grouped_scores,
        }

    return {
        "folders": list(direct_scores.keys()),
        "structure": STRUCTURE_SAMPLES_DIRECTLY if direct_scores else "",
        "sample_count": direct_sample_count,
        "scores": direct_scores,
    }


# Use a limited scan for quick folder detection; it may miss some files.
def _limited_background_input_detection(
    root: Path,
    *,
    expected_folders: list[str],
    active_image_defs: list[dict],
    max_dirs: int = INPUT_SCAN_LAYOUT_MAX_DIRS,
    max_entries: int = INPUT_SCAN_LAYOUT_MAX_ENTRIES,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[dict, bool]:
    budget = _BackgroundScanBudget(
        max_dirs=max_dirs,
        max_entries=max_entries,
        cancel_requested=cancel_requested,
    )
    detection = _flat_tiff_detection(budget, root, active_image_defs)
    if detection is None:
        child_dirs = budget.child_dirs(root)
        detection = _image_folder_detection(budget, root, child_dirs, expected_folders, max_dirs)
        if detection is None:
            detection = _nested_tiff_detection(budget, child_dirs, max_dirs)
    return detection, budget.truncated


# Slow metadata inspection stays independent of Qt so it is safe to run on a worker thread.
def _scan_cancelled(cancel_requested: Callable[[], bool] | None) -> bool:
    return bool(cancel_requested is not None and cancel_requested())


def _inspect_first_flat_tiff(root: Path, result: dict) -> None:
    tiffs = iter_supported_tiff_files(root, recursive=False)
    if not tiffs:
        return
    first_tiff = tiffs[0]
    result["first_tiff"] = str(first_tiff)
    try:
        result["tiff_info"] = get_tiff_stack_info(first_tiff)
    except Exception as exc:
        result["tiff_error"] = f"{type(exc).__name__}: {exc}"


def _summarize_nd2_dataset(
    root: Path,
    nd2_files: list[Path],
    first_info: dict,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[dict, dict]:
    z_depths: list[int] = []
    issues: list[str] = []
    issue_count = 0
    metadata_error_count = 0
    reference_channels = list(first_info.get("channel_names", []) or [])

    for index, nd2_path in enumerate(nd2_files):
        if _scan_cancelled(cancel_requested):
            break
        try:
            current_info = first_info if index == 0 else ndi.inspect_nd2_file(nd2_path)
            current_sizes = dict(current_info.get("sizes", {}) or {})
            current_channels = list(current_info.get("channel_names", []) or [])
            z_depths.append(max(1, int(current_sizes.get("Z", 1) or 1)))
            relative_name = str(nd2_path.relative_to(root))
            if current_channels != reference_channels:
                issue_count += 1
                if len(issues) < 8:
                    issues.append(f"{relative_name}: channels {current_channels} do not match {reference_channels}.")
            unsupported_axes = [
                str(axis)
                for axis, size in current_sizes.items()
                if str(axis) not in {"T", "P", "Z", "C", "Y", "X", "S"} and int(size or 0) > 1
            ]
            if unsupported_axes:
                issue_count += 1
                if len(issues) < 8:
                    issues.append(f"{relative_name}: unsupported dimensions {', '.join(unsupported_axes)}.")
            if int(current_sizes.get("Y", 0) or 0) < 1 or int(current_sizes.get("X", 0) or 0) < 1:
                issue_count += 1
                if len(issues) < 8:
                    issues.append(f"{relative_name}: missing a valid X/Y image plane.")
        except Exception as exc:
            issue_count += 1
            metadata_error_count += 1
            if len(issues) < 8:
                issues.append(f"{nd2_path.name}: could not read metadata ({type(exc).__name__}: {exc}).")

    stack_depths = [depth for depth in z_depths if depth > 1]
    inspected_all = len(z_depths) == len(nd2_files)
    return (
        {
            "inspected_count": len(z_depths),
            "stack_count": len(stack_depths),
            "single_plane_count": sum(depth == 1 for depth in z_depths),
            "min_stack_depth": min(stack_depths, default=0),
            "max_stack_depth": max(stack_depths, default=0),
            "error_count": metadata_error_count,
            "complete": inspected_all and metadata_error_count == 0,
        },
        {
            "inspected_count": len(z_depths),
            "issue_count": issue_count,
            "issues": issues,
            "compatible": inspected_all and issue_count == 0,
        },
    )


def _inspect_nd2_input(
    root: Path,
    result: dict,
    *,
    max_dirs: int,
    max_entries: int,
    cancel_requested: Callable[[], bool] | None,
    inspect_all_z: bool,
) -> None:
    try:
        nd2_files, truncated = _limited_nd2_files(
            root,
            max_dirs=max_dirs,
            max_entries=max_entries,
            cancel_requested=cancel_requested,
        )
    except OSError as exc:
        result["nd2_error"] = f"{type(exc).__name__}: {exc}"
        return

    result["nd2_count"] = len(nd2_files)
    result["nd2_scan_truncated"] = bool(truncated)
    if not nd2_files:
        return

    first_nd2 = nd2_files[0]
    result["nd2_first"] = str(first_nd2)
    try:
        info = ndi.inspect_nd2_file(first_nd2)
        result["nd2_channels"] = list(info.get("channel_names", []))
        result["nd2_channel_colors"] = list(info.get("channel_colors", []))
        result["nd2_sizes"] = dict(info.get("sizes", {}) or {})
        if inspect_all_z:
            z_summary, dataset_summary = _summarize_nd2_dataset(
                root,
                nd2_files,
                info,
                cancel_requested,
            )
            result["nd2_z_summary"] = z_summary
            result["nd2_dataset_summary"] = dataset_summary
    except Exception as exc:
        result["nd2_error"] = f"{type(exc).__name__}: {exc}"


def scan_input_path(
    folder_path: str,
    *,
    expected_folders: list[str],
    active_image_defs: list[dict],
    max_nd2_dirs: int = INPUT_SCAN_ND2_MAX_DIRS,
    max_nd2_entries: int = INPUT_SCAN_ND2_MAX_ENTRIES,
    cancel_requested: Callable[[], bool] | None = None,
    inspect_all_nd2_z: bool = False,
) -> dict:
    root = Path(str(folder_path or "").strip())
    result = _empty_input_scan_result(str(root))
    if _scan_cancelled(cancel_requested):
        result["scan_cancelled"] = True
        return result
    if not root.exists() or not root.is_dir():
        return result
    result["path_exists"] = True
    result["path_is_dir"] = True

    # Layout detection and ND2 inspection are intentionally bounded. A user may
    # point Cellonaut at a large drive or network share while browsing.
    detection, layout_truncated = _limited_background_input_detection(
        root,
        expected_folders=expected_folders,
        active_image_defs=active_image_defs,
        cancel_requested=cancel_requested,
    )
    result["detection"] = detection
    result["layout_scan_truncated"] = bool(layout_truncated)
    if _scan_cancelled(cancel_requested):
        result["scan_cancelled"] = True
        return result

    if detection.get("structure") == STRUCTURE_FLAT_TIFFS:
        _inspect_first_flat_tiff(root, result)

    _inspect_nd2_input(
        root,
        result,
        max_dirs=max_nd2_dirs,
        max_entries=max_nd2_entries,
        cancel_requested=cancel_requested,
        inspect_all_z=inspect_all_nd2_z,
    )
    result["scan_cancelled"] = _scan_cancelled(cancel_requested)
    return result


class InputPathScanWorker(QObject):
    """Background scanner for folder layout, TIFF stack, and ND2 metadata."""

    done_signal = Signal(dict)

    # Copy mutable definitions before crossing threads so later GUI edits cannot change an active scan.
    def __init__(
        self,
        folder_path: str,
        expected_folders: list[str],
        active_image_defs: list[dict],
        *,
        inspect_all_nd2_z: bool = False,
    ):
        super().__init__()
        self.folder_path = folder_path
        self.expected_folders = list(expected_folders)
        self.active_image_defs = [dict(item) for item in active_image_defs]
        self.inspect_all_nd2_z = bool(inspect_all_nd2_z)
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    # Convert unexpected scanner failures into data because exceptions cannot cross the Qt signal boundary safely.
    @Slot()
    def run(self):
        try:
            result = scan_input_path(
                self.folder_path,
                expected_folders=self.expected_folders,
                active_image_defs=self.active_image_defs,
                cancel_requested=self._cancel_event.is_set,
                inspect_all_nd2_z=self.inspect_all_nd2_z,
            )
        except Exception as exc:
            result = _empty_input_scan_result(self.folder_path)
            result["scan_error"] = f"{type(exc).__name__}: {exc}"
        self.done_signal.emit(result)


# Discovery and navigation share one owner because background scan results must be
# matched to the exact path currently represented by the Files and Setup views.
class CellonautGuiFileBrowserMixin(GuiMixin):
    """Handle input selection, background discovery, and recent-folder state."""

    # A scan result is trustworthy only for the path it inspected; using an
    # older folder's layout would make sample discovery silently target the
    # wrong directory depth.
    def get_current_input_structure(self, *, require_ready: bool = True) -> str:
        current_path = self.input_dir.get().strip() if hasattr(self, "input_dir") else ""
        scan_result = getattr(self, "_last_input_scan_result", {}) or {}
        scanned_path = str(scan_result.get("path", "") or "").strip()
        current_key = input_path_key(current_path)
        if input_path_key(scanned_path) != current_key:
            pending_path = str(getattr(self, "_pending_input_scan_path", "") or "").strip()
            active_path = str(getattr(self, "_active_input_scan_path", "") or "").strip()
            if require_ready and current_key and current_key in {input_path_key(pending_path), input_path_key(active_path)}:
                raise ValueError(
                    "The input folder is still being scanned. Wait for the scan to finish before running or previewing."
                )
            return STRUCTURE_FLAT_TIFFS

        structure = str((scan_result.get("detection") or {}).get("structure", "") or "").strip()
        supported = {
            STRUCTURE_SAMPLES_DIRECTLY,
            STRUCTURE_GROUPED_BY_PROTEIN,
            STRUCTURE_FLAT_TIFFS,
            STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
        }
        return structure if structure in supported else STRUCTURE_FLAT_TIFFS

    # Configure the model after widgets exist, keeping UI construction separate from filesystem access.
    def setup_file_browser(self):
        self.fs_model = QFileSystemModel(self.as_qobject())
        self.fs_model.setRootPath("")

        self.file_tree.setModel(self.fs_model)
        self.file_tree.setRootIndex(QModelIndex())

        self.file_tree.setColumnWidth(0, 280)
        self.file_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.file_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.file_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.file_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        self.file_tree.activated.connect(self.on_file_tree_activated)

        selection_model = self.file_tree.selectionModel()
        if selection_model is not None:
            selection_model.currentChanged.connect(self.on_file_tree_current_changed)

        self.browser_path_entry.clear()
        self._browser_root_path = None
        self._browser_history = []
        self._browser_forward_history = []
        self.update_browser_navigation_buttons()
        for path_row in (self.input_dir, self.output_dir, self.nd2_output_dir):
            path_row.edit.textChanged.connect(self.update_browser_navigation_buttons)

    # Debouncing avoids starting a folder scan for every keystroke in a pasted path.
    def setup_path_drop_helpers(self):
        if hasattr(self, "input_dir"):
            self._input_scan_debounce = QTimer(self.as_qobject())
            self._input_scan_debounce.setSingleShot(True)
            self._input_scan_debounce.setInterval(600)
            self._input_scan_debounce.timeout.connect(self.start_pending_input_path_scan)
            self._input_scan_status_timeout = QTimer(self.as_qobject())
            self._input_scan_status_timeout.setSingleShot(True)
            self._input_scan_status_timeout.setInterval(INPUT_SCAN_STATUS_TIMEOUT_MS)
            self._input_scan_status_timeout.timeout.connect(self.on_input_scan_status_timeout)
            self._input_scan_thread = None
            self._input_scan_worker = None
            self._pending_input_scan_path = ""
            self._active_input_scan_path = ""
            self.input_dir.edit.textChanged.connect(self.on_pipeline_input_path_changed)
            self.input_dir.edit.pathDropped.connect(self.on_pipeline_input_path_dropped)

    # Suggestions are non-destructive so an explicitly chosen output folder always wins.
    def suggest_output_folder_if_empty(self, output_widget, input_path: str, suffix: str):
        if output_widget is None:
            return

        current = output_widget.get().strip() if hasattr(output_widget, "get") else output_widget.text().strip()
        if current:
            return

        source = Path(str(input_path or "").strip())
        if not source.name:
            return

        suggested = source.parent / f"{source.name}_{suffix}"
        if hasattr(output_widget, "set"):
            output_widget.set(str(suggested))
        else:
            output_widget.setText(str(suggested))

    def on_pipeline_input_path_changed(self, text: str):
        if getattr(self, "_suppress_input_path_scan", False):
            return
        self.suggest_output_folder_if_empty(self.output_dir, text, "Cellonaut_output")
        self.schedule_input_path_scan(text)

    def on_pipeline_input_path_dropped(self, text: str):
        self.schedule_input_path_scan(text, immediate=True)

    # Combine duplicate scan requests and keep the newest requested path.
    def schedule_input_path_scan(self, text: str, immediate: bool = False):
        if getattr(self, "_closing_requested", False):
            return
        requested_path = str(text or "").strip()
        active_path = str(getattr(self, "_active_input_scan_path", "") or "").strip()
        if input_path_key(requested_path) == input_path_key(active_path):
            self._pending_input_scan_path = ""
            timer = getattr(self, "_input_scan_debounce", None)
            if timer is not None:
                timer.stop()
            return

        completed_path = str((getattr(self, "_last_input_scan_result", {}) or {}).get("path", "") or "").strip()
        if input_path_key(requested_path) == input_path_key(completed_path) and not immediate:
            self._pending_input_scan_path = ""
            timer = getattr(self, "_input_scan_debounce", None)
            if timer is not None:
                timer.stop()
            return

        self._pending_input_scan_path = requested_path
        active_worker = getattr(self, "_input_scan_worker", None)
        if active_path and input_path_key(requested_path) != input_path_key(active_path):
            request_cancel = getattr(active_worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        timer = getattr(self, "_input_scan_debounce", None)
        if timer is None:
            return
        timer.start(0 if immediate else 600)

    # Background discovery must not overwrite progress from a pipeline, preview, or conversion worker.
    def input_scan_can_use_global_status(self) -> bool:
        if getattr(self, "running", False):
            return False
        for attr in ("worker_thread", "preview_worker_thread", "nd2_worker_thread"):
            thread = getattr(self, attr, None)
            if thread is not None and thread.isRunning():
                return False
        return True

    # Only one scanner thread runs at a time; a replacement asks the current
    # scanner to stop between directory reads before it starts.
    def start_pending_input_path_scan(self):
        if getattr(self, "_closing_requested", False):
            return
        folder_path = str(getattr(self, "_pending_input_scan_path", "") or "").strip()
        thread = getattr(self, "_input_scan_thread", None)
        if thread is not None and thread.isRunning():
            if input_path_key(folder_path) == input_path_key(
                str(getattr(self, "_active_input_scan_path", "") or "").strip()
            ):
                self._pending_input_scan_path = ""
            return
        if not folder_path:
            self.apply_input_path_scan_result({"path": "", "detection": {}, "nd2_count": 0})
            return

        self._pending_input_scan_path = ""
        self._active_input_scan_path = folder_path
        self.log(
            "[SCAN] Checking folder access, TIFF layout/channels, and ND2 metadata in background: " f"{folder_path}"
        )
        timeout_timer = getattr(self, "_input_scan_status_timeout", None)
        if timeout_timer is not None:
            timeout_timer.start()
        if self.input_scan_can_use_global_status():
            self.update_worker_status(stage_update(PipelineStage.SCANNING, "Scanning input folder"))

        expected = self.get_expected_image_folders()
        active_defs = self.get_active_image_definitions()
        worker = InputPathScanWorker(folder_path, expected, active_defs)
        thread, worker = self.prepare_worker_thread(
            worker,
            terminal_signal=worker.done_signal,
            result_callback=self.input_scan_result_on_gui.emit,
            finished_callback=self.input_scan_finished_on_gui.emit,
            thread_factory=QThread,
        )
        self._input_scan_worker = worker
        self._input_scan_thread = thread
        thread.start()

    # Stop a scan that exceeds the interactive time limit. A single operating-
    # system network read cannot be interrupted, but traversal stops afterward.
    @Slot()
    def on_input_scan_status_timeout(self):
        thread = getattr(self, "_input_scan_thread", None)
        if thread is None or not thread.isRunning():
            return
        active_path = str(getattr(self, "_active_input_scan_path", "") or "")
        self.log(
            "[SCAN][WARN] Background folder scan exceeded the time limit; cancellation was requested. "
            "A blocked network directory read may still take time to return. "
            f"Path: {active_path}"
        )
        worker = getattr(self, "_input_scan_worker", None)
        request_cancel = getattr(worker, "request_cancel", None)
        if callable(request_cancel):
            request_cancel()
        if getattr(self, "_worker_status_stage", None) == PipelineStage.SCANNING:
            stop_heartbeat = getattr(self, "stop_worker_status_heartbeat", None)
            if callable(stop_heartbeat):
                stop_heartbeat()
            if self.input_scan_can_use_global_status():
                self.update_worker_status("Ready")

    # Reject results for old paths so delayed network scans cannot replace the current channel setup.
    @Slot(dict)
    def apply_input_path_scan_result(self, result: dict):
        scanned_path = str(result.get("path", "") or "")
        current_path = self.input_dir.get().strip()
        if input_path_key(scanned_path) != input_path_key(current_path):
            timeout_timer = getattr(self, "_input_scan_status_timeout", None)
            if timeout_timer is not None:
                timeout_timer.stop()
            if getattr(self, "_worker_status_stage", None) == PipelineStage.SCANNING:
                stop_heartbeat = getattr(self, "stop_worker_status_heartbeat", None)
                if callable(stop_heartbeat):
                    stop_heartbeat()
                if self.input_scan_can_use_global_status():
                    self.update_worker_status("Ready")
            self.log(f"[SCAN] Ignored stale scan result for: {scanned_path}")
            return

        self.commit_gui_edits()
        preset_was_clean = not self.preset_has_unsaved_changes()

        timeout_timer = getattr(self, "_input_scan_status_timeout", None)
        if timeout_timer is not None:
            timeout_timer.stop()

        if result.get("scan_cancelled"):
            self.log(f"[SCAN] Background folder scan cancelled: {scanned_path}")
            return

        self._last_input_scan_result = dict(result)
        if self.input_scan_can_use_global_status():
            self.update_worker_status("Ready")

        if result.get("scan_error"):
            self.log(f"[SCAN][WARN] Background folder scan failed: {result['scan_error']}")
            return
        if hasattr(self, "validate_all_fields"):
            self.validate_all_fields()

        nd2_count = int(result.get("nd2_count", 0) or 0)
        self.apply_nd2_scan_result(result)
        if result.get("layout_scan_truncated"):
            self.log(
                "[SCAN][WARN] Background TIFF layout scan stopped early after reaching the scan budget. "
                "Use Check Setup for a full validation pass."
            )
        if result.get("nd2_scan_truncated"):
            self.log(
                "[SCAN][WARN] ND2 background scan stopped early after reaching the scan budget. "
                "Open Check Setup or start ND2 conversion for a full scan."
            )
        if nd2_count:
            self.suggest_output_folder_if_empty(self.nd2_output_dir, scanned_path, "converted")
            self.remember_automatic_preset_state_if_clean(preset_was_clean)
            return

        detection = dict(result.get("detection") or {})
        self.log(
            "[SCAN] Finished: "
            f"structure={detection.get('structure') or 'unknown'}, "
            f"samples={int(detection.get('sample_count', 0) or 0)}, "
            f"ND2 files={nd2_count}"
        )
        self.apply_detected_image_folders_from_input(
            scanned_path,
            detection=detection,
            first_tiff=Path(result["first_tiff"]) if result.get("first_tiff") else None,
            tiff_info=result.get("tiff_info"),
        )
        self.remember_automatic_preset_state_if_clean(preset_was_clean)

    # Start the newest queued path only after Qt has destroyed the previous worker thread.
    @Slot()
    def cleanup_input_path_scan(self):
        timeout_timer = getattr(self, "_input_scan_status_timeout", None)
        if timeout_timer is not None:
            timeout_timer.stop()
        self._input_scan_worker = None
        self._input_scan_thread = None
        self._active_input_scan_path = ""
        # Clear the scan indicator when the thread finishes, even if its result was not delivered.
        if getattr(self, "_worker_status_stage", None) == PipelineStage.SCANNING:
            stop_heartbeat = getattr(self, "stop_worker_status_heartbeat", None)
            if callable(stop_heartbeat):
                stop_heartbeat()
            if self.input_scan_can_use_global_status():
                self.update_worker_status("Ready")
        pending = str(getattr(self, "_pending_input_scan_path", "") or "").strip()
        current = self.input_dir.get().strip()
        completed = str((getattr(self, "_last_input_scan_result", {}) or {}).get("path", "") or "").strip()
        pending_key = input_path_key(pending)
        current_key = input_path_key(current)
        completed_key = input_path_key(completed)
        if pending and pending_key == current_key and pending_key != completed_key:
            QTimer.singleShot(0, self.start_pending_input_path_scan)
        elif pending_key == completed_key:
            self._pending_input_scan_path = ""

    # Apply detected names only to untouched defaults; configured presets may
    # intentionally include optional channels that are absent from this dataset.
    def apply_detected_image_folders_from_input(
        self,
        folder_path: str,
        detection: dict | None = None,
        first_tiff: Path | None = None,
        tiff_info: dict | None = None,
    ):
        root = Path(str(folder_path or "").strip())
        detection = detection or detect_image_folders_from_input(
            root,
            expected_folders=self.get_expected_image_folders(),
            active_image_defs=self.get_active_image_definitions(),
        )
        folders = detection.get("folders", [])
        if not folders:
            return

        structure = detection.get("structure")
        if structure == STRUCTURE_FLAT_TIFFS and self.apply_detected_tiff_stack_channels(
            root,
            detection,
            first_tiff=first_tiff,
            tiff_info=tiff_info,
        ):
            return

        if structure in {
            STRUCTURE_SAMPLES_DIRECTLY,
            STRUCTURE_GROUPED_BY_PROTEIN,
            STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
        }:
            self.apply_detected_folder_channels(list(folders), detection)

    # Folder names are safe defaults only while the current rows still have
    # generic names and no user-supplied mask source configuration.
    def apply_detected_folder_channels(self, folders: list[str], detection: dict) -> bool:
        detected_folders = [str(name or "").strip() for name in folders if str(name or "").strip()]
        if not detected_folders:
            return False

        old_defs = self.get_active_image_definitions()
        physical_defs = [
            dict(image_def)
            for image_def in old_defs
            if not bool(image_def.get("is_mask_only", False))
            and not image_uses_combined_mask(image_def)
        ]
        mask_defs = [
            dict(image_def)
            for image_def in old_defs
            if bool(image_def.get("is_mask_only", False))
            or image_uses_combined_mask(image_def)
        ]
        generic_channels = all(
            str(image_def.get("name", "") or "").strip().startswith(("Channel ", "Image"))
            for image_def in physical_defs
        )
        configured_masks = any(
            str(image_def.get("classifier", "") or "").strip() or image_uses_combined_mask(image_def)
            for image_def in mask_defs
        )
        if not generic_channels or configured_masks:
            self.log("[AUTO] Detected folder-based TIFF input. Kept the configured channel names and folder mappings.")
            return False

        source_name_map: dict[str, str] = {}
        new_channels: list[dict] = []
        for index, folder_name in enumerate(detected_folders):
            image_def = (
                dict(physical_defs[index])
                if index < len(physical_defs)
                else self.create_default_image_definition(index)
            )
            old_name = str(image_def.get("name", "") or f"Channel {index + 1}").strip()
            source_name_map[old_name] = folder_name
            image_def["name"] = folder_name
            image_def["folder"] = folder_name
            image_def["stack_channel_index"] = ""
            new_channels.append(image_def)

        mask_name_map: dict[str, str] = {}
        new_masks: list[dict] = []
        for mask_def in mask_defs:
            source_name = str(mask_def.get("mask_source_channel", "") or "").strip()
            if source_name not in source_name_map:
                continue
            new_source = source_name_map[source_name]
            old_mask_name = str(mask_def.get("name", "") or "").strip()
            if old_mask_name == f"{source_name} mask":
                mask_def["name"] = f"{new_source} mask"
                mask_name_map[old_mask_name] = str(mask_def["name"])
            mask_def["mask_source_channel"] = new_source
            mask_def["folder"] = new_source
            new_masks.append(mask_def)

        new_defs = [*new_channels, *new_masks]
        final_names = [str(image_def.get("name", "") or "").strip() for image_def in new_defs]
        rename_map = {**source_name_map, **mask_name_map}
        remapped_defs = remap_analysis_references(
            new_defs,
            [*rename_map, *final_names],
            [*rename_map.values(), *final_names],
        )
        remapped_defs = seed_first_classifier_self_relationship(remapped_defs)
        apply_detected_image_definitions(self, remapped_defs, final_names, final_names)

        sample_count = int(detection.get("sample_count", 0) or 0)
        self.log(
            "[AUTO] Detected folder-based TIFF input: "
            f"{sample_count} sample(s), {len(detected_folders)} channel folder(s)."
        )
        return True

    # Recursive lookup supports converted ND2 datasets whose source subfolders were preserved.
    def get_first_tiff_in_folder(self, root: Path) -> Path | None:
        if not root.exists() or not root.is_dir():
            return None
        tiffs = iter_supported_tiff_files(root, recursive=True)
        return tiffs[0] if tiffs else None

    # Preserve user configuration by matching existing rows by stack layer before creating defaults.
    def apply_detected_tiff_stack_channels(
        self,
        root: Path,
        detection: dict | None = None,
        *,
        first_tiff: Path | None = None,
        tiff_info: dict | None = None,
    ) -> bool:
        first_tiff = first_tiff or self.get_first_tiff_in_folder(root)
        if first_tiff is None:
            return False

        cache_key = str(first_tiff.resolve()) if first_tiff.exists() else str(first_tiff)
        if getattr(self, "_last_auto_stack_key", "") == cache_key:
            return True

        info = tiff_info
        if info is None:
            try:
                info = get_tiff_stack_info(first_tiff)
            except Exception as exc:
                self.log(f"[AUTO] Could not inspect TIFF stack {first_tiff.name}: {exc}")
                return False

        channel_count = max(1, int(info.get("channel_count", 1) or 1))
        channel_names = list(info.get("channel_names", []) or [])
        channel_colors = list(info.get("channel_colors", []) or [])
        old_defs = self.get_active_image_definitions()
        old_channel_defs = [dict(image_def) for image_def in old_defs if self.is_physical_channel_definition(image_def)]
        mask_defs = [
            dict(image_def)
            for image_def in old_defs
            if bool(image_def.get("is_mask_only", False))
            or image_uses_combined_mask(image_def)
        ]
        old_names = [str(img.get("name", "") or f"Image{i + 1}").strip() for i, img in enumerate(old_defs)]

        old_by_layer = {
            str(img.get("stack_channel_index", "") or "").strip(): dict(img)
            for img in old_channel_defs
            if str(img.get("stack_channel_index", "") or "").strip()
        }

        new_defs = []
        for idx in range(channel_count):
            layer = str(idx + 1)
            detected_name = (
                str(channel_names[idx]).strip()
                if idx < len(channel_names) and str(channel_names[idx]).strip()
                else f"Channel {idx + 1}"
            )
            if layer in old_by_layer:
                image_def = dict(old_by_layer[layer])
            elif idx < len(old_channel_defs) and str(
                old_channel_defs[idx].get("stack_channel_index", "") or ""
            ).strip() in {"", layer}:
                image_def = dict(old_channel_defs[idx])
            else:
                image_def = self.create_default_image_definition(idx)

            current_name = str(image_def.get("name", "") or "").strip()
            if (
                not current_name
                or current_name.startswith(("Image", "Channel "))
                or str(image_def.get("stack_channel_index", "") or "").strip() not in {"", layer}
            ):
                image_def["name"] = detected_name
            image_def["folder"] = detected_name
            image_def["display_color"] = (
                str(channel_colors[idx]) if idx < len(channel_colors) else str(image_def.get("display_color", "") or "")
            )
            image_def["stack_channel_index"] = layer if channel_count > 1 else ""
            image_def.setdefault("classifier", "")
            new_defs.append(image_def)

        new_defs.extend(mask_defs)
        new_names = [str(img.get("name", "") or f"Channel {i + 1}") for i, img in enumerate(new_defs)]
        apply_detected_image_definitions(self, new_defs, old_names, new_names)

        self._last_auto_stack_key = cache_key
        sample_count = int((detection or {}).get("sample_count", 0) or 0)
        self.log(
            "[AUTO] Detected TIFF stack input: "
            f"{sample_count} TIFF sample(s), {channel_count} channel(s) from {first_tiff.name}. "
            f"Applied {len(new_defs)} channel/mask row(s)" + (" using OME channel names." if channel_names else ".")
        )
        return True

    # Renamed channels must carry their mask and measurement references with them.
    def remap_analysis_references_for_detected_folders(
        self,
        image_defs: list[dict],
        old_names: list[str],
        new_names: list[str],
    ):
        remapped_defs = remap_analysis_references(image_defs, old_names, new_names)
        remapped_defs = seed_first_classifier_self_relationship(remapped_defs)
        image_defs[:] = remapped_defs

    def on_file_tree_current_changed(self, current, previous):
        if not current or not current.isValid():
            return

        path = self.fs_model.filePath(current)
        if not path:
            return

        self.browser_path_entry.setText(path)

        p = Path(path)
        if p.is_file():
            self.open_file_in_right_panel(path)

    def on_file_tree_activated(self, index):
        if not index or not index.isValid():
            return

        path = self.fs_model.filePath(index)
        p = Path(path)

        if p.is_dir():
            self.set_browser_root(path)
        elif p.is_file():
            self.open_file_in_right_panel(path)

    def set_browser_root(self, folder_path: str):
        self._set_browser_root_internal(folder_path, push_history=True)

    # Keep navigation availability synchronized with history so icon-only controls
    # communicate clearly when backward or forward movement is possible.
    def update_browser_navigation_buttons(self, *_args):
        if hasattr(self, "browser_back_button"):
            self.browser_back_button.setEnabled(bool(self._browser_history))
        if hasattr(self, "browser_forward_button"):
            self.browser_forward_button.setEnabled(bool(self._browser_forward_history))
        if hasattr(self, "browser_up_button"):
            self.browser_up_button.setEnabled(self._browser_root_path is not None)
        if hasattr(self, "browser_input_button"):
            self.browser_input_button.setEnabled(bool(self.input_dir.get().strip()))
        if hasattr(self, "browser_output_button"):
            self.browser_output_button.setEnabled(bool(self.output_dir.get().strip()))
        if hasattr(self, "browser_open_location_button"):
            current_path = self.browser_path_entry.text().strip() or str(self._browser_root_path or "").strip()
            self.browser_open_location_button.setEnabled(bool(current_path))
        if hasattr(self, "browser_go_button"):
            self.browser_go_button.setEnabled(bool(self.browser_path_entry.text().strip()))

    # The model's invalid root index is Qt's portable representation of the computer view.
    def _show_browser_computer_view(self):
        self.file_tree.setRootIndex(QModelIndex())
        self.browser_path_entry.clear()
        self._browser_root_path = None
        self.update_browser_navigation_buttons()

    # Record the current folder before using the Computer shortcut so Back works
    # the same way it does after any other deliberate navigation.
    def browser_go_to_computer(self):
        if self._browser_root_path is not None:
            self._browser_history.append(self._browser_root_path)
            self._browser_forward_history.clear()
        self._show_browser_computer_view()

    def browser_go_back(self):
        if not self._browser_history:
            return
        previous = self._browser_history.pop()
        self._browser_forward_history.append(self._browser_root_path)
        if previous is None:
            self._show_browser_computer_view()
        else:
            self._set_browser_root_internal(previous, push_history=False, clear_forward=False)

    # Forward mirrors Back with its own stack so new navigation can invalidate
    # only the abandoned forward branch, as users expect from file explorers.
    def browser_go_forward(self):
        if not self._browser_forward_history:
            return
        next_path = self._browser_forward_history.pop()
        self._browser_history.append(self._browser_root_path)
        if next_path is None:
            self._show_browser_computer_view()
        else:
            self._set_browser_root_internal(next_path, push_history=False, clear_forward=False)

    def browser_go_up(self):
        raw = self.browser_path_entry.text().strip()
        current: Path | None

        if raw:
            p = Path(raw)
            if p.exists():
                current = p.parent if p.is_file() else p
            else:
                current = Path(self._browser_root_path) if self._browser_root_path else None
        else:
            current = Path(self._browser_root_path) if self._browser_root_path else None

        if current is None:
            self.browser_go_to_computer()
            return

        parent = current.parent
        if parent == current or str(parent) == "":
            self.browser_go_to_computer()
            return

        if parent.exists():
            self._set_browser_root_internal(str(parent), push_history=True)
        else:
            self.browser_go_to_computer()

    # Centralize root changes so validation and history remain identical for every navigation control.
    def _set_browser_root_internal(
        self,
        folder_path: str,
        push_history: bool = True,
        clear_forward: bool = True,
    ):
        if not folder_path:
            return

        p = Path(folder_path)
        if not p.exists():
            QMessageBox.information(self, "Browser", f"Path does not exist:\n{folder_path}")
            return

        idx = self.fs_model.index(str(p))
        if not idx.isValid():
            QMessageBox.information(self, "Browser", f"Could not open:\n{folder_path}")
            return

        if input_path_key(self._browser_root_path or "") == input_path_key(str(p)):
            self.browser_path_entry.setText(str(p))
            self.update_browser_navigation_buttons()
            return

        if push_history:
            self._browser_history.append(self._browser_root_path)
        if clear_forward:
            self._browser_forward_history.clear()

        self.file_tree.setRootIndex(idx)
        self.browser_path_entry.setText(str(p))
        self._browser_root_path = str(p)
        self.update_browser_navigation_buttons()

    def browse_to_input_dir(self):
        self.set_browser_root(self.input_dir.get())

    def browse_to_output_dir(self):
        self.set_browser_root(self.output_dir.get())

    def browser_go_to_path(self):
        raw = self.browser_path_entry.text().strip()
        if not raw:
            return

        p = Path(raw)

        if not p.exists():
            QMessageBox.information(self, "Browser", f"Path does not exist:\n{raw}")
            return

        if p.is_file():
            parent = p.parent
            self.set_browser_root(str(parent))
            self.open_file_in_right_panel(str(p))
            return

        self.set_browser_root(str(p))
