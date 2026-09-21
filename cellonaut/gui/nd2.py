"""ND2 import workflow and metadata binding for the desktop GUI.

The actual ND2 export lives in the IO layer. This mixin translates GUI choices
into backend settings, applies metadata found by the input scanner, and keeps
channel/Z controls understandable before conversion starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QMessageBox, QWidget
import cellonaut.io.nd2_import as ndi
from cellonaut.exceptions import SetupError, SetupErrorCode
from cellonaut.gui.file_browser import InputPathScanWorker, open_folder_in_system_browser
from cellonaut.io.path_keys import input_path_key
from cellonaut.pipeline.discovery import safe_iterdir
from cellonaut.gui.mixin import GuiMixin
from cellonaut.io.nd2_import import ND2ImportConfig
from cellonaut.system.disk_space import GIB
from cellonaut.workers import ND2ImportWorker


class CellonautGuiNd2Mixin(GuiMixin):
    """GUI-side ND2 state and conversion actions."""

    def nd2_source_folder_text(self) -> str:
        source_row = getattr(self, "nd2_source_dir", None)
        if source_row is not None:
            try:
                return str(source_row.get() or "").strip()
            except (AttributeError, RuntimeError):
                pass
        input_row = getattr(self, "input_dir", None)
        if input_row is None:
            return ""
        try:
            return str(input_row.get() or "").strip()
        except (AttributeError, RuntimeError):
            return ""

    def set_nd2_source_folder(self, path_text: str) -> None:
        source_row = getattr(self, "nd2_source_dir", None)
        if source_row is None:
            return
        self._syncing_nd2_source = True
        try:
            source_row.set(str(path_text or ""))
        finally:
            self._syncing_nd2_source = False

    def open_nd2_import_dialog(self) -> None:
        dialog = getattr(self, "nd2_dialog", None)
        if dialog is None:
            return
        if not self.nd2_source_folder_text():
            input_row = getattr(self, "input_dir", None)
            if input_row is not None:
                self.set_nd2_source_folder(input_row.get())
        self.suggest_output_folder_if_empty(
            getattr(self, "nd2_output_dir", None), self.nd2_source_folder_text(), "converted"
        )
        source_path = Path(self.nd2_source_folder_text())
        timer = getattr(self, "nd2_source_scan_timer", None)
        if (
            timer is not None
            and not getattr(self, "nd2_detected_channel_names", [])
            and source_path.exists()
            and source_path.is_dir()
        ):
            timer.start(0)
        dialog.show()
        QTimer.singleShot(0, self.resize_nd2_dialog_to_content)
        dialog.raise_()
        dialog.activateWindow()

    def resize_nd2_dialog_to_content(self) -> None:
        dialog = getattr(self, "nd2_dialog", None)
        if dialog is None:
            return
        dialog.adjustSize()

    def on_nd2_source_path_changed(self, text: str) -> None:
        if getattr(self, "_syncing_nd2_source", False):
            return
        self.nd2_detected_channel_names = []
        self.nd2_channel_folder_state = {}
        if hasattr(self, "nd2_z_mode"):
            self.nd2_z_mode.set("Max projection")
        self.remember_nd2_scan_summary(nd2_count=0, first_display="", sizes={})
        self.suggest_output_folder_if_empty(getattr(self, "nd2_output_dir", None), text, "converted")
        self.rebuild_nd2_channel_rows()
        self.set_nd2_conversion_controls_visible(False)
        progress_panel = getattr(self, "nd2_progress_panel", None)
        if progress_panel is not None:
            progress_panel.setVisible(False)
        label = getattr(self, "nd2_detected_channels_label", None)
        if label is not None:
            label.setText("Waiting for files...")
        worker = getattr(self, "_nd2_scan_worker", None)
        if worker is not None:
            worker.request_cancel()
        source_dir = Path(str(text or "").strip())
        timer = getattr(self, "nd2_source_scan_timer", None)
        if timer is not None:
            timer.stop()
            if source_dir.exists() and source_dir.is_dir():
                timer.start()

    def inspect_nd2_source(self) -> None:
        thread = getattr(self, "_nd2_scan_thread", None)
        if thread is not None and thread.isRunning():
            return

        source_text = self.nd2_source_folder_text()
        source_dir = Path(source_text)
        if not source_text or not source_dir.exists() or not source_dir.is_dir():
            return

        self.set_nd2_conversion_controls_visible(False)
        self.nd2_detected_channels_label.setText("Inspecting ND2 files and reading channel metadata...")
        source_row = getattr(self, "nd2_source_dir", None)
        if source_row is not None:
            source_row.setEnabled(False)

        worker = InputPathScanWorker(source_text, [], [], inspect_all_nd2_z=True)
        thread, worker = self.prepare_worker_thread(
            worker,
            terminal_signal=worker.done_signal,
            result_callback=self.apply_nd2_scan_result,
            finished_callback=self.cleanup_nd2_scan_worker,
        )
        self._nd2_scan_thread = thread
        self._nd2_scan_worker = worker
        thread.start()

    @Slot()
    def cleanup_nd2_scan_worker(self) -> None:
        scanned_source = str(getattr(getattr(self, "_nd2_scan_worker", None), "folder_path", "") or "")
        self._nd2_scan_thread = None
        self._nd2_scan_worker = None
        source_row = getattr(self, "nd2_source_dir", None)
        if source_row is not None:
            source_row.setEnabled(not bool(getattr(self, "_primary_task_active", False)))
        current_source = self.nd2_source_folder_text()
        if input_path_key(scanned_source) != input_path_key(current_source):
            current_path = Path(current_source)
            timer = getattr(self, "nd2_source_scan_timer", None)
            if timer is not None and current_path.exists() and current_path.is_dir():
                timer.start(0)

    # Convert display labels to the values expected by the ND2 reader.
    def nd2_backend_z_mode(self) -> str:
        label = self.nd2_z_mode.get() if hasattr(self, "nd2_z_mode") else "Max projection"
        return "single_z" if str(label) == "Single Z slice" else "max_projection"

    # Convert the one-based control to NumPy's zero-based index once to avoid off-by-one logic downstream.
    def nd2_backend_z_index(self) -> int:
        spin = getattr(self, "nd2_z_index_spin", None)
        value = int(spin.value()) if spin is not None else 1
        return max(0, value - 1)

    def nd2_z_stack_summary(self) -> dict[str, int | bool]:
        return dict(getattr(self, "nd2_z_summary", {}) or {})

    # Keep the slice control bounded by detected metadata while allowing entry before inspection finishes.
    def update_nd2_z_controls(self):
        spin = getattr(self, "nd2_z_index_spin", None)
        row = getattr(self, "nd2_z_index_row", None)
        if spin is None:
            return

        summary = self.nd2_z_stack_summary()
        stack_count = int(summary.get("stack_count", 0) or 0)
        min_stack_depth = int(summary.get("min_stack_depth", 0) or 0)
        inspection_complete = bool(summary.get("complete", False))
        has_stacks = stack_count > 0
        max_z = max(1, min_stack_depth) if has_stacks else 1
        spin.setRange(1, max_z)
        single_z = self.nd2_backend_z_mode() == "single_z"
        if not inspection_complete and single_z:
            self.nd2_z_mode.set("Max projection")
            single_z = False
        spin.setEnabled(has_stacks and inspection_complete and single_z)
        if row is not None:
            row.setVisible(has_stacks and single_z)
            row.setEnabled(has_stacks and inspection_complete and single_z)
        mode = getattr(self, "nd2_z_mode", None)
        if mode is not None:
            mode.setVisible(has_stacks)
            mode.setEnabled(has_stacks and inspection_complete)
            if has_stacks and not inspection_complete:
                mode.setToolTip("Max projection is required because Z metadata could not be read from every ND2 file.")
            else:
                mode.setToolTip("Choose how detected ND2 Z stacks are reduced during TIFF conversion.")
        spin.setToolTip(f"Slice to keep in every stack, starting at 1. Available: 1-{max_z}.")
        QTimer.singleShot(0, self.resize_nd2_dialog_to_content)

    def on_nd2_z_mode_changed(self, *_args):
        self.update_nd2_z_controls()
        self.refresh_nd2_status_summary()

    def on_nd2_z_index_changed(self, *_args):
        self.refresh_nd2_status_summary()

    def remember_nd2_scan_summary(
        self,
        *,
        nd2_count: int,
        first_display,
        sizes: dict | None = None,
        z_summary: dict | None = None,
        dataset_summary: dict | None = None,
    ):
        self.nd2_detected_file_count = int(nd2_count or 0)
        self.nd2_detected_first_display = str(first_display or "")
        self.nd2_detected_sizes = dict(sizes or {})
        self.nd2_z_summary = dict(z_summary or {})
        self.nd2_dataset_summary = dict(dataset_summary or {})
        self.update_nd2_z_controls()

    def inspect_nd2_z_summary(self, nd2_files: list[Path], first_info: dict) -> dict[str, int | bool]:
        z_depths: list[int] = []
        error_count = 0
        metadata_error_count = 0
        issues: list[str] = []
        reference_channels = list(first_info.get("channel_names", []) or [])
        for index, nd2_path in enumerate(nd2_files):
            try:
                info = first_info if index == 0 else ndi.inspect_nd2_file(nd2_path)
                sizes = dict(info.get("sizes", {}) or {})
                channels = list(info.get("channel_names", []) or [])
                z_depths.append(max(1, int(sizes.get("Z", 1) or 1)))
                if channels != reference_channels:
                    error_count += 1
                    issues.append(f"{nd2_path.name}: channels {channels} do not match {reference_channels}.")
            except Exception as exc:
                error_count += 1
                metadata_error_count += 1
                issues.append(f"{nd2_path.name}: could not read metadata ({type(exc).__name__}: {exc}).")
                self.log(f"[ND2][WARN] Could not inspect Z metadata for {nd2_path.name}: {exc}")
        stack_depths = [depth for depth in z_depths if depth > 1]
        z_complete = len(z_depths) == len(nd2_files) and metadata_error_count == 0
        dataset_compatible = len(z_depths) == len(nd2_files) and error_count == 0
        self.nd2_dataset_summary = {
            "inspected_count": len(z_depths),
            "issue_count": error_count,
            "issues": issues[:8],
            "compatible": dataset_compatible,
        }
        return {
            "inspected_count": len(z_depths),
            "stack_count": len(stack_depths),
            "single_plane_count": sum(depth == 1 for depth in z_depths),
            "min_stack_depth": min(stack_depths, default=0),
            "max_stack_depth": max(stack_depths, default=0),
            "error_count": metadata_error_count,
            "complete": z_complete,
        }

    # Retain source indices so skipped channels still receive their own metadata colors.
    def nd2_selected_channels(self) -> list[tuple[int, str]]:
        state = dict(getattr(self, "nd2_channel_folder_state", {}) or {})
        detected = list(getattr(self, "nd2_detected_channel_names", []) or [])
        selected: list[tuple[int, str]] = []
        for index, channel_name in enumerate(detected):
            output_name = str(state.get(channel_name, channel_name) or "").strip()
            if output_name:
                selected.append((index, output_name))
        return selected

    def nd2_selected_channel_names(self) -> list[str]:
        return [name for _index, name in self.nd2_selected_channels()]

    # Compare case-insensitively because channel references elsewhere in the GUI are case-insensitive.
    def nd2_duplicate_output_names(self) -> list[str]:
        duplicates: list[str] = []
        seen: set[str] = set()
        for name in self.nd2_selected_channel_names():
            key = name.casefold()
            if key in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(key)
        return duplicates

    # Read through the PathRow wrapper so tests and the full Qt widget share this formatting path.
    def nd2_output_folder_text(self) -> str:
        output_row = getattr(self, "nd2_output_dir", None)
        if output_row is None:
            return ""
        try:
            return str(output_row.get() or "").strip()
        except (AttributeError, RuntimeError):
            return ""

    # State the missing action directly instead of presenting an empty destination line.
    def nd2_export_destination_text(self) -> str:
        output_text = self.nd2_output_folder_text()
        if output_text:
            return f"TIFF output: {output_text}"
        return "TIFF output: choose a converted TIFF folder before converting."

    # Gather scan, mapping, axis, and destination decisions into one reviewable status block.
    def nd2_status_summary_text(self, *, error: str = "") -> str:
        count = int(getattr(self, "nd2_detected_file_count", 0) or 0)
        first_display = str(getattr(self, "nd2_detected_first_display", "") or "")
        channel_names = list(getattr(self, "nd2_detected_channel_names", []) or [])
        channel_count = len(channel_names)
        sizes = dict(getattr(self, "nd2_detected_sizes", {}) or {})

        if count <= 0:
            return "Waiting for files..."

        lines = [f"ND2 files detected: {count}"]
        if first_display:
            lines.append(f"First file: {first_display}")
        lines.append(self.nd2_export_destination_text())

        if error:
            lines.append(f"Could not inspect ND2 channels: {error}")
            lines.append("Try a different ND2 file, check that the file is readable, or install the ND2 backend.")
            return "\n".join(lines)

        lines.append(f"Channels: {channel_count}")
        dataset_summary = dict(getattr(self, "nd2_dataset_summary", {}) or {})
        if dataset_summary:
            inspected_count = int(dataset_summary.get("inspected_count", 0) or 0)
            issue_count = int(dataset_summary.get("issue_count", 0) or 0)
            if bool(dataset_summary.get("compatible", False)):
                lines.append(f"All {inspected_count} file(s) have compatible channel metadata.")
            elif issue_count:
                lines.append(f"Conversion blocked: {issue_count} compatibility issue(s) found.")
                lines.extend(str(issue) for issue in list(dataset_summary.get("issues", []) or []))

        z_summary = self.nd2_z_stack_summary()
        stack_count = int(z_summary.get("stack_count", 0) or 0)
        single_plane_count = int(z_summary.get("single_plane_count", 0) or 0)
        min_depth = int(z_summary.get("min_stack_depth", 0) or 0)
        max_depth = int(z_summary.get("max_stack_depth", 0) or 0)
        if stack_count:
            depth_text = f"{min_depth} slices" if min_depth == max_depth else f"{min_depth}-{max_depth} slices"
            stack_text = f"Z stacks: {stack_count} file(s), {depth_text}"
            if single_plane_count:
                stack_text += f"; {single_plane_count} single-plane file(s)"
            if self.nd2_backend_z_mode() == "single_z":
                spin = getattr(self, "nd2_z_index_spin", None)
                slice_number = int(spin.value()) if spin is not None else 1
                stack_text += f"; exporting slice {slice_number}"
            else:
                stack_text += "; exporting max projection"
            lines.append(stack_text)
            if not bool(z_summary.get("complete", False)):
                lines.append("Z metadata could not be read from every file; max projection is required.")

        for key, label, export_text in (
            ("T", "Timepoints", "exporting first"),
            ("P", "Positions", "exporting first"),
            ("S", "RGB components", "exporting first component"),
        ):
            try:
                value = int(sizes.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 1:
                lines.append(f"{label}: {value}; {export_text}")

        return "\n".join(lines)

    def refresh_nd2_status_summary(self, *, error: str = ""):
        label = getattr(self, "nd2_detected_channels_label", None)
        if label is not None:
            label.setText(self.nd2_status_summary_text(error=error))

    # Keep the complete workflow visible and disable only actions that require inspected ND2 metadata.
    def set_nd2_conversion_controls_visible(self, has_nd2: bool):
        has_nd2 = bool(has_nd2)
        if hasattr(self, "nd2_group"):
            self.nd2_group.setVisible(True)
        for attr in (
            "nd2_channel_rows_container",
            "nd2_mapping_help",
            "nd2_convert_button",
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                set_visible = getattr(widget, "setVisible", None)
                if callable(set_visible):
                    set_visible(has_nd2)
                set_enabled = getattr(widget, "setEnabled", None)
                if callable(set_enabled):
                    set_enabled(has_nd2)
        self.update_nd2_z_controls()

    # Apply background-scan results on the GUI thread and preserve user channel names for matching channels.
    def apply_nd2_scan_result(self, result: dict):
        nd2_count = int(result.get("nd2_count", 0) or 0)
        first_text = str(result.get("nd2_first", "") or "")
        channel_names = list(result.get("nd2_channels", []) or [])
        sizes = dict(result.get("nd2_sizes", {}) or {})
        z_summary = dict(result.get("nd2_z_summary", {}) or {})
        dataset_summary = dict(result.get("nd2_dataset_summary", {}) or {})
        error = str(result.get("nd2_error", "") or "")
        source_text = str(result.get("path", "") or "")
        source_dir = Path(source_text)
        current_source = self.nd2_source_folder_text()
        if current_source and input_path_key(source_text) != input_path_key(current_source):
            return
        if source_text:
            self.set_nd2_source_folder(source_text)

        has_nd2 = nd2_count > 0
        self.set_nd2_conversion_controls_visible(False)

        if not has_nd2:
            self.nd2_detected_channel_names = []
            self.remember_nd2_scan_summary(nd2_count=0, first_display="", sizes={})
            self.nd2_detected_channels_label.setText("Waiting for files...")
            self.rebuild_nd2_channel_rows()
            return

        first_path = Path(first_text) if first_text else Path()
        try:
            first_display = first_path.relative_to(source_dir)
        except (ValueError, OSError):
            first_display = first_path

        if error:
            self.nd2_detected_channel_names = []
            self.remember_nd2_scan_summary(
                nd2_count=nd2_count,
                first_display=first_display,
                sizes=sizes,
                z_summary=z_summary,
                dataset_summary=dataset_summary,
            )
            self.refresh_nd2_status_summary(error=error)
            self.rebuild_nd2_channel_rows()
            return

        self.nd2_detected_channel_names = channel_names
        self.remember_nd2_scan_summary(
            nd2_count=nd2_count,
            first_display=first_display,
            sizes=sizes,
            z_summary=z_summary,
            dataset_summary=dataset_summary,
        )
        previous_folders = cast(dict[str, str], getattr(self, "nd2_channel_folder_state", {}))
        self.nd2_channel_folder_state = {name: previous_folders.get(name, name) for name in channel_names}
        self.refresh_nd2_status_summary()
        self.rebuild_nd2_channel_rows()
        compatible = bool(dataset_summary.get("compatible", True))
        self.set_nd2_conversion_controls_visible(bool(channel_names) and compatible)
        self.log(f"[ND2] Detected channels from {first_path.name}: {channel_names}")

    # Trust the recursive scanner cache first; the shallow fallback avoids freezing the GUI on large trees.
    def input_dir_contains_nd2_files(self) -> bool:
        try:
            input_text = self.input_dir.get().strip()
        except (AttributeError, RuntimeError):
            return False
        root = Path(input_text)

        cached_scan = getattr(self, "_last_input_scan_result", {}) or {}
        if input_path_key(str(cached_scan.get("path", "") or "")) == input_path_key(input_text):
            return int(cached_scan.get("nd2_count", 0) or 0) > 0

        if not root.exists() or not root.is_dir():
            return False

        # Keep this fallback shallow. Recursive ND2 metadata scans run in the
        # background input scanner so large folders do not block the GUI.
        return any(path.is_file() and path.suffix.lower() == ".nd2" for path in safe_iterdir(root))

    def update_nd2_visibility(self):
        ready = self.input_dir_contains_nd2_files() and bool(self.nd2_detected_channel_names)
        self.set_nd2_conversion_controls_visible(ready)

    # Intercept a normal pipeline run on raw ND2 input so users cannot accidentally bypass conversion.
    def run_pipeline_from_nd2_input_if_needed(self) -> bool:
        if not self.input_dir_contains_nd2_files():
            return False
        if self.nd2_worker_thread is not None and self.nd2_worker_thread.isRunning():
            return True

        input_text = self.input_dir.get().strip()
        self.set_nd2_source_folder(input_text)
        self.suggest_output_folder_if_empty(getattr(self, "nd2_output_dir", None), input_text, "converted")
        self.open_nd2_import_dialog()
        self.nd2_detected_channels_label.setText(
            "Raw ND2 files cannot run directly. Review the automatically inspected mapping, then choose "
            "Convert ND2 to TIFF. After conversion, select the TIFF folder as the pipeline input."
        )
        return True

    # Show both metadata name and zero-based backend index to make channel mapping unambiguous.
    def get_nd2_channel_label(self, channel_name: str, index: int) -> str:
        return f"{channel_name} (Index {index})"

    # Persist edits before rebuilding rows because Qt deletes the old line-edit widgets.
    def sync_nd2_channel_folder_state_from_ui(self):
        rows = list(getattr(self, "nd2_channel_rows", []) or [])
        if not rows:
            return

        state = {}
        for row in rows:
            channel_name = row["channel_name"]
            folder_name = cast(QLineEdit, row["folder"]).text().strip()
            state[channel_name] = folder_name

        self.nd2_channel_folder_state = state

    def rebuild_nd2_channel_rows(self):
        if not hasattr(self, "nd2_channel_rows_container_layout"):
            return

        self.sync_nd2_channel_folder_state_from_ui()

        while self.nd2_channel_rows_container_layout.count():
            item = self.nd2_channel_rows_container_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.nd2_channel_rows = []

        if not self.nd2_detected_channel_names:
            placeholder = QLabel("ND2 channel mapping will appear after channel inspection.")
            placeholder.setWordWrap(True)
            self.nd2_channel_rows_container_layout.addWidget(placeholder)
            self.update_nd2_channel_summary()
            QTimer.singleShot(0, self.resize_nd2_dialog_to_content)
            return

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        source_header = QLabel("ND2 channel")
        source_header.setProperty("muted", "true")
        source_header.setProperty("uiRole", "mutedLabel")
        source_header.setMinimumWidth(220)
        target_header = QLabel("TIFF channel name")
        target_header.setProperty("muted", "true")
        target_header.setProperty("uiRole", "mutedLabel")
        header_layout.addWidget(source_header)
        header_layout.addWidget(target_header, 1)
        self.nd2_channel_rows_container_layout.addWidget(header_widget)

        for index, channel_name in enumerate(self.nd2_detected_channel_names):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            label = QLabel(self.get_nd2_channel_label(channel_name, index))
            label.setMinimumWidth(220)

            default_folder = self.nd2_channel_folder_state.get(channel_name, channel_name)
            folder_entry = QLineEdit(default_folder)
            folder_entry.setPlaceholderText("Channel name (leave empty to skip)")
            folder_entry.textChanged.connect(self.on_nd2_channel_folder_changed)

            row_layout.addWidget(label)
            row_layout.addWidget(folder_entry, 1)

            self.nd2_channel_rows_container_layout.addWidget(row_widget)

            self.nd2_channel_rows.append(
                {
                    "channel_name": channel_name,
                    "index": index,
                    "label": label,
                    "folder": folder_entry,
                    "widget": row_widget,
                }
            )

        self.update_nd2_channel_summary()
        QTimer.singleShot(0, self.resize_nd2_dialog_to_content)

    def on_nd2_channel_folder_changed(self, *_args):
        self.sync_nd2_channel_folder_state_from_ui()
        self.update_nd2_channel_summary()

    def set_nd2_action_buttons_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled) and not bool(getattr(self, "_primary_task_active", False))
        for attr in ("nd2_convert_button",):
            button = getattr(self, attr, None)
            if button is not None:
                button.setEnabled(enabled)

    # Surface skipped and duplicate mappings before the user reaches the conversion button.
    def update_nd2_channel_summary(self):
        if not hasattr(self, "nd2_mapping_help"):
            return
        selected_names = self.nd2_selected_channel_names()
        total = len(list(getattr(self, "nd2_detected_channel_names", []) or []))
        if total <= 0:
            self.nd2_mapping_help.setText("ND2 channel mapping will appear after channel inspection.")
            self.set_nd2_action_buttons_enabled(False)
            self.refresh_nd2_status_summary()
            return
        if not selected_names:
            self.nd2_mapping_help.setText(
                "No ND2 channels are selected for conversion. Enter at least one TIFF channel name."
            )
            self.set_nd2_action_buttons_enabled(False)
            self.refresh_nd2_status_summary()
            return
        duplicate_names = self.nd2_duplicate_output_names()
        if duplicate_names:
            self.nd2_mapping_help.setText(
                f"Duplicate TIFF channel names: {', '.join(duplicate_names)}. Use a unique name for each channel."
            )
            self.set_nd2_action_buttons_enabled(False)
            self.refresh_nd2_status_summary()
            return
        self.nd2_mapping_help.setText(
            f"TIFF channels: {', '.join(selected_names)}. Empty channel names are skipped."
        )
        self.set_nd2_action_buttons_enabled(True)
        self.refresh_nd2_status_summary()

    # Add a concrete corrective action to backend and validation errors without hiding their technical cause.
    def format_nd2_settings_error(self, error: Exception | str) -> str:
        message = str(error).strip() or "ND2 conversion cannot start."
        lines = ["Convert ND2 to TIFF is blocked.", "", message]
        code = error.code if isinstance(error, SetupError) else None
        if code in {SetupErrorCode.ND2_INPUT, SetupErrorCode.INPUT_MISSING}:
            lines.extend(
                [
                    "",
                    "Choose an input folder that contains .nd2 files, including nested subfolders if needed.",
                ]
            )
        elif code == SetupErrorCode.ND2_CHANNELS:
            lines.extend(
                [
                    "",
                    "Check the channel mapping in Import ND2. Empty TIFF channel names are skipped, and names must be unique.",
                ]
            )
        elif code == SetupErrorCode.ND2_OUTPUT or isinstance(error, PermissionError):
            lines.extend(
                [
                    "",
                    "Choose a writable converted TIFF folder.",
                ]
            )
        elif code == SetupErrorCode.ND2_BACKEND:
            lines.extend(
                [
                    "",
                    "Install the ND2 backend in this environment, then restart Cellonaut if needed.",
                ]
            )
        return "\n".join(lines)

    # Build and validate the backend configuration from the latest row edits at the moment conversion starts.
    def collect_nd2_config(self) -> ND2ImportConfig:
        self.sync_nd2_channel_folder_state_from_ui()

        source_dir = Path(self.nd2_source_folder_text())
        if not source_dir.exists() or not source_dir.is_dir():
            raise SetupError(SetupErrorCode.INPUT_MISSING, "Input folder must be an existing folder containing ND2 files.")

        if not self.nd2_output_dir.get().strip():
            raise SetupError(SetupErrorCode.ND2_OUTPUT, "Converted TIFF folder is required before conversion.")

        if not self.nd2_detected_channel_names:
            self.detect_nd2_channels_from_input()
        dataset_summary = dict(getattr(self, "nd2_dataset_summary", {}) or {})
        if dataset_summary and not bool(dataset_summary.get("compatible", False)):
            raise SetupError(SetupErrorCode.ND2_CHANNELS, "The selected ND2 files have incompatible channel metadata. Choose a consistent dataset.")

        if not self.nd2_detected_channel_names:
            raise self.nd2_channel_detection_error(source_dir)

        channel_map: dict[str, int | str] = {}
        used_folder_names: set[str] = set()

        for channel_index, channel_name in enumerate(self.nd2_detected_channel_names):
            folder_name = str(self.nd2_channel_folder_state.get(channel_name, "") or "").strip()

            # Empty = skip this channel
            if not folder_name:
                continue

            folder_key = folder_name.casefold()
            if folder_key in used_folder_names:
                raise SetupError(SetupErrorCode.ND2_CHANNELS,
                    f"Duplicate channel name: {folder_name!r}. " "Each converted ND2 channel must have a unique name."
                )

            used_folder_names.add(folder_key)
            channel_map[folder_name] = channel_index

        if not channel_map:
            raise SetupError(SetupErrorCode.ND2_CHANNELS, "No ND2 channels are selected for conversion. Enter at least one TIFF channel name.")

        cfg = ND2ImportConfig(
            source_dir=source_dir,
            output_dir=Path(self.nd2_output_dir.get()),
            channel_map=channel_map,
            z_mode=self.nd2_backend_z_mode(),
            z_index=self.nd2_backend_z_index(),
            strip_after_last_underscore=False,
        )

        if not cfg.source_dir.exists():
            raise SetupError(SetupErrorCode.INPUT_MISSING, "ND2 source folder does not exist.")

        try:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SetupError(SetupErrorCode.ND2_OUTPUT, f"Could not create converted TIFF folder: {cfg.output_dir}\n{exc}") from exc
        return cfg

    # Distinguish "files exist but inspection failed" from an actually empty folder.
    def nd2_channel_detection_error(self, source_dir: Path) -> SetupError:
        nd2_files = []
        try:
            nd2_files = ndi.list_nd2_files(source_dir)
        except Exception:
            pass

        if nd2_files:
            first_file = nd2_files[0]
            try:
                rel_first = first_file.relative_to(source_dir)
            except ValueError:
                rel_first = first_file
            return SetupError(SetupErrorCode.ND2_CHANNELS,
                f"Found {len(nd2_files)} ND2 file(s), including:\n{rel_first}\n\n"
                "But ND2 channel names have not been detected yet. "
                "Cellonaut tried to rescan before conversion, so the first ND2 file "
                "may not be inspectable."
            )

        return SetupError(SetupErrorCode.ND2_INPUT, "No ND2 files were detected in the selected input folder or its subfolders.")

    # Keep a synchronous fallback for explicit conversion clicks when no background scan result is available.
    def detect_nd2_channels_from_input(self):
        source_text = self.nd2_source_folder_text()
        if not source_text:
            self.nd2_detected_channel_names = []
            self.update_nd2_visibility()
            return
        source_dir = Path(source_text)

        if not source_dir.exists() or not source_dir.is_dir():
            self.nd2_detected_channel_names = []
            self.update_nd2_visibility()
            return

        try:
            nd2_files = ndi.list_nd2_files(source_dir)
        except OSError as exc:
            self.nd2_detected_channel_names = []
            self.nd2_detected_channels_label.setText(f"Could not scan input folder for ND2 files: {exc}")
            self.rebuild_nd2_channel_rows()
            self.update_nd2_visibility()
            return

        self.update_nd2_visibility()

        first_display = ""
        if nd2_files:
            try:
                first_display = str(nd2_files[0].relative_to(source_dir))
            except ValueError:
                first_display = str(nd2_files[0])

        if not nd2_files:
            self.nd2_detected_channel_names = []
            self.remember_nd2_scan_summary(nd2_count=0, first_display="", sizes={})
            self.nd2_detected_channels_label.setText("Waiting for files...")
            self.rebuild_nd2_channel_rows()
            return

        try:
            info = ndi.inspect_nd2_file(nd2_files[0])
        except Exception as exc:
            self.nd2_detected_channel_names = []
            self.remember_nd2_scan_summary(nd2_count=len(nd2_files), first_display=first_display, sizes={})
            self.refresh_nd2_status_summary(error=str(exc))
            self.rebuild_nd2_channel_rows()
            return

        channel_names = list(info.get("channel_names", []))
        self.nd2_detected_channel_names = channel_names
        z_summary = self.inspect_nd2_z_summary(nd2_files, info)
        dataset_summary = dict(getattr(self, "nd2_dataset_summary", {}) or {})
        self.remember_nd2_scan_summary(
            nd2_count=len(nd2_files),
            first_display=first_display,
            sizes=dict(info.get("sizes", {}) or {}),
            z_summary=z_summary,
            dataset_summary=dataset_summary,
        )

        new_state = {}
        for ch_name in channel_names:
            new_state[ch_name] = self.nd2_channel_folder_state.get(ch_name, ch_name)
        self.nd2_channel_folder_state = new_state

        self.refresh_nd2_status_summary()
        self.rebuild_nd2_channel_rows()
        self.set_nd2_conversion_controls_visible(
            bool(channel_names) and bool(dataset_summary.get("compatible", False))
        )
        self.log(f"[ND2] Detected channels from {nd2_files[0].name}: {channel_names}")

    # Ignore repeated clicks while a worker owns the conversion destination.
    def convert_nd2_folder_clicked(self):
        if self.has_active_processing_task():
            return

        try:
            if not self.nd2_detected_channel_names:
                self.detect_nd2_channels_from_input()
            cfg = self.collect_nd2_config()
        except Exception as e:
            QMessageBox.critical(self, "Invalid ND2 settings", self.format_nd2_settings_error(e))
            return

        try:
            source_files = ndi.list_nd2_files(cfg.source_dir)
        except OSError:
            source_files = []
        if not self.confirm_disk_space_for_files(
            cfg.output_dir,
            source_files,
            operation="ND2 conversion",
            output_multiplier=3.0,
            reserve_bytes=2 * GIB,
        ):
            return

        self.start_nd2_import(cfg)

    def show_nd2_progress(self, message: str, *, progress: int = 0, running: bool = True) -> None:
        panel = getattr(self, "nd2_progress_panel", None)
        if panel is not None:
            panel.setVisible(True)
        label = getattr(self, "nd2_progress_label", None)
        if label is not None:
            label.setText(message)
        progress_bar = getattr(self, "nd2_progress_bar", None)
        if progress_bar is not None:
            progress_bar.setValue(max(0, min(100, int(progress))))
        cancel_button = getattr(self, "nd2_cancel_button", None)
        if cancel_button is not None:
            cancel_button.setVisible(running)
            cancel_button.setEnabled(running)
        open_button = getattr(self, "nd2_open_output_button", None)
        if open_button is not None:
            open_button.setVisible(not running)
        QTimer.singleShot(0, self.resize_nd2_dialog_to_content)

    def on_nd2_current_file_changed(self, message: str) -> None:
        self.current_sample_label.setText(message)
        label = getattr(self, "nd2_progress_label", None)
        if label is not None:
            label.setText(message)

    def on_nd2_progress_changed(self, value: int) -> None:
        self.progress_bar.setValue(value)
        progress_bar = getattr(self, "nd2_progress_bar", None)
        if progress_bar is not None:
            progress_bar.setValue(value)

    def cancel_nd2_conversion_clicked(self) -> None:
        worker = getattr(self, "nd2_worker", None)
        if worker is None:
            return
        worker.request_cancel()
        self.show_nd2_progress("Stopping ND2 conversion...", progress=self.progress_bar.value())
        button = getattr(self, "nd2_cancel_button", None)
        if button is not None:
            button.setEnabled(False)

    def open_nd2_output_folder(self) -> None:
        output_path = Path(self.nd2_output_folder_text())
        if output_path.exists():
            open_folder_in_system_browser(output_path)

    # Run conversion outside the GUI process so cancellation can stop a blocked native ND2 read promptly.
    def start_nd2_import(self, cfg: ND2ImportConfig):
        if self.has_active_processing_task():
            return
        self.stop_worker_status_heartbeat()
        self.set_pipeline_busy(True)
        self.set_primary_task_active(True)
        self.status_label.setText("Preparing ND2 conversion...")
        self.set_status_style("Running")
        self.progress_bar.setValue(0)
        self.current_sample_label.setText("Preparing converted TIFF files...")
        self.show_nd2_progress("Preparing ND2 conversion...", progress=0, running=True)
        self.log(f"[ND2] Converting from {cfg.source_dir} to {cfg.output_dir}")
        self.log(f"[ND2] Channels: {list(cfg.channel_map.keys())}")

        self.nd2_worker = ND2ImportWorker(cfg)
        self.nd2_worker.current_file_signal.connect(
            self.on_nd2_current_file_changed,
            Qt.ConnectionType.QueuedConnection,
        )
        self.nd2_worker.progress_signal.connect(
            self.on_nd2_progress_changed,
            Qt.ConnectionType.QueuedConnection,
        )
        self.nd2_worker.terminal_signal.connect(self.nd2_worker_finished_on_gui.emit)

        self.nd2_worker_thread, self.nd2_worker = self.run_worker(
            self.nd2_worker,
            self.cleanup_nd2_worker_refs,
        )

    # Restore controls before opening the modal error dialog so the application never appears stuck afterward.
    @Slot(str)
    def on_nd2_import_error(self, msg: str):
        self.status_label.setText("ND2 conversion error")
        self.set_status_style("Error")
        self.current_sample_label.setText("ND2 conversion failed")
        self.show_nd2_progress("Conversion failed. See the error details for what to fix.", running=False)
        self.set_primary_task_active(False)
        self.set_pipeline_busy(False)
        QMessageBox.critical(self, "ND2 conversion error", msg)

    @Slot()
    def cleanup_nd2_worker_refs(self):
        self.nd2_worker = None
        self.nd2_worker_thread = None

    # Update the interface after conversion succeeds, partly succeeds, fails, or is cancelled.
    @Slot(dict)
    def on_nd2_worker_finished(self, outcome: dict):
        state = str(outcome.get("state", "error"))
        if state == "cancelled":
            self.show_nd2_progress("Conversion cancelled.", running=False)
            self.on_nd2_import_cancelled()
            return
        if state == "error":
            self.on_nd2_import_error(str(outcome.get("error") or "ND2 conversion failed."))
            return
        self.on_nd2_import_done(
            outcome.get("result") or {},
            completed_with_errors=(state == "partial"),
        )

    # Finish conversion without changing pipeline settings or launching another workflow.
    def on_nd2_import_done(
        self,
        summary: dict,
        *,
        completed_with_errors: bool = False,
    ):
        if completed_with_errors:
            self.status_label.setText("ND2 conversion completed with errors")
            self.set_status_style("Completed with errors")
        else:
            self.status_label.setText("ND2 conversion done")
            self.set_status_style("Done")
        self.current_sample_label.setText(
            f"Processed: {summary.get('processed', 0)} / {summary.get('total', 0)} | Failed: {summary.get('failed', 0)}"
        )
        self.progress_bar.setValue(100)
        processed = int(summary.get("processed", 0) or 0)
        total = int(summary.get("total", 0) or 0)
        failed = int(summary.get("failed", 0) or 0)
        skipped = int(summary.get("skipped", max(0, total - processed - failed)) or 0)
        result_heading = "Conversion finished with errors." if completed_with_errors else "Conversion complete."
        progress_text = (
            f"{result_heading}\n"
            f"Converted: {processed} | Skipped: {skipped} | Failed: {failed}\n"
            f"Saved to: {self.nd2_output_folder_text()}\n"
            "Next: select this converted TIFF folder as the pipeline Input directory."
        )
        self.show_nd2_progress(progress_text, progress=100, running=False)

        self.log(f"[ND2] Conversion summary: {summary}")
        if completed_with_errors:
            self.log("[ND2][WARN] One or more files failed to convert.")
        self.set_browser_root(self.nd2_output_dir.get())

        self.set_primary_task_active(False)
        self.set_pipeline_busy(False)
