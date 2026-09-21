"""Preset selection, editing, and transfer behavior for the main window."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from cellonaut.config.defaults import coerce_bool, default_measurement_options
from cellonaut.config.preset_store import (
    DEFAULT_PRESET_NAME,
    is_default_preset,
    preset_path,
    preset_state_token,
    privacy_safe_preset,
    read_json_object,
    sanitize_preset_name,
    write_json_object,
)
from cellonaut.gui.dialogs import PresetInfoDialog
from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.panel_notes import normalize_panel_notes
from cellonaut.io.writers import write_text
from cellonaut.resources import PRESETS_DIR, ensure_user_app_dirs, get_resource_path


class CellonautGuiPresetMixin(GuiMixin):
    """Coordinate preset state, persistence, and user actions."""

    # The default baseline must be portable between computers, so remove local
    # folders and classifier paths while preserving workflow settings.
    def get_default_preset_dict(self):
        data = self.get_preset_dict()

        for key in [
            "input_dir",
            "output_dir",
            "mask_source_dir",
        ]:
            if key in data:
                data[key] = ""

        cleaned_defs = []
        for img in data.get("image_definitions", []):
            img_copy = dict(img)
            img_copy["classifier"] = ""
            cleaned_defs.append(img_copy)
        data["image_definitions"] = cleaned_defs

        return data

    # The Default preset is the recovery baseline. Recreate it when missing or
    # unreadable so startup is never left without a valid configuration.
    def ensure_default_preset(self):
        self.ensure_presets_dir()
        path = self.preset_path_from_name(DEFAULT_PRESET_NAME)
        if path.exists():
            try:
                self._read_json_file(path)
                return
            except Exception as exc:
                self.log(f"[PRESETS][WARN] Default preset was unreadable and will be recreated: {exc}")

        try:
            bundled_path = get_resource_path(Path("presets") / "Default.json")
            try:
                default_data = self._read_json_file(bundled_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self.log(f"[PRESETS][WARN] Could not read the bundled Default preset: {exc}")
                default_data = self.get_default_preset_dict()
            self._write_json_file(path, default_data)
            self.log(f"Created default preset: {path}")
        except Exception as e:
            self.log(f"Could not create default preset: {e}")

    # The default preset contains pipeline data only, so loading it cannot
    # overwrite appearance, internal Fiji resolution, or ND2 import state.
    def load_default_preset_on_startup(self):
        self.ensure_default_preset()
        path = self.preset_path_from_name(DEFAULT_PRESET_NAME)
        if not path.exists():
            return

        try:
            settings = self._read_json_file(path)
            self.apply_preset_dict(settings)
            self.refresh_presets_list(select_name=DEFAULT_PRESET_NAME)
            self.remember_preset_state(DEFAULT_PRESET_NAME, self.get_preset_dict())
            self.log(f"Loaded startup preset: {path}")
        except Exception as e:
            self.log(f"Could not load startup preset: {e}")

    # Route directory creation through the resource layer so every platform uses
    # the same writable application-data location.
    def ensure_presets_dir(self):
        ensure_user_app_dirs()

    # Preset names become filenames, so restrict them to characters that behave
    # consistently in source runs and the packaged Windows application.
    def sanitize_preset_name(self, name: str) -> str:
        return sanitize_preset_name(name)

    # Build preset paths in one place so UI labels can never escape the preset
    # directory or disagree with list discovery.
    def preset_path_from_name(self, name: str) -> Path:
        return preset_path(PRESETS_DIR, name)

    def preset_state_token(self, data: dict) -> str:
        return preset_state_token(data)

    def remember_preset_state(self, name: str, data: dict) -> None:
        self._active_preset_name = str(name or "").strip()
        self._active_preset_snapshot = self.preset_state_token(data)
        self.set_preset_dirty_state(False)

    # Keep the visual marker and Save availability tied to the same computed
    # state so the interface cannot claim a clean preset while enabling Save.
    def set_preset_dirty_state(self, dirty: bool) -> None:
        self._preset_dirty = bool(dirty)
        refresh_summaries = getattr(self, "refresh_pipeline_section_summaries", None)
        if callable(refresh_summaries) and hasattr(self, "project_section"):
            refresh_summaries()
        label = getattr(self, "preset_dirty_label", None)
        if label is not None:
            label.setVisible(self._preset_dirty)
        refresh_controls = getattr(self, "refresh_preset_control_states", None)
        if callable(refresh_controls):
            refresh_controls()

    # User input is debounced by the main window; comparing the complete preset
    # here keeps dynamic tables covered without wiring every generated editor.
    def refresh_preset_dirty_indicator(self) -> None:
        # Renames rebuild dependent selectors. While the name editor still has
        # focus, report the pending change without destroying that editor on the
        # debounce timer. editingFinished and Save still commit it explicitly.
        for index, row in enumerate(getattr(self, "image_rows", [])):
            if row is None or row.name is None or not row.name.edit.hasFocus():
                continue
            if row.name.get().strip() != self.image_definitions[index]["name"]:
                self.set_preset_dirty_state(True)
                return
        self.commit_gui_edits()
        self.set_preset_dirty_state(self.preset_has_unsaved_changes())

    # Dataset scans may fill generic channel rows after a preset loads. Treat
    # that automatic discovery as the new baseline only when no user edit would
    # be hidden by doing so.
    def remember_automatic_preset_state_if_clean(self, was_clean: bool) -> None:
        name = str(getattr(self, "_active_preset_name", "") or "").strip()
        if was_clean and name:
            self.remember_preset_state(name, self.get_preset_dict())

    # Compare committed data. UI actions commit before asking this question.
    def preset_has_unsaved_changes(self) -> bool:
        baseline = getattr(self, "_active_preset_snapshot", None)
        if baseline is None:
            return False
        try:
            return self.preset_state_token(self.get_preset_dict()) != baseline
        except Exception as exc:
            self.log(f"[PRESETS][WARN] Could not check for unsaved changes: {exc}")
            return True

    # Signal blocking prevents restoring a cancelled selection from recursively
    # opening another save prompt.
    def restore_active_preset_selection(self) -> None:
        active_name = str(getattr(self, "_active_preset_name", "") or "").strip()
        if not active_name:
            return
        index = self.preset_combo.findText(active_name)
        if index < 0:
            return
        previous = self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(index)
        self.preset_combo.blockSignals(previous)

    # Save the complete current workflow when switching presets.
    def save_active_preset_changes(self) -> bool:
        name = str(getattr(self, "_active_preset_name", "") or "").strip()
        if not name:
            return True
        if is_default_preset(name):
            return self.save_preset_as()
        path = self.preset_path_from_name(name)
        if not path.exists():
            QMessageBox.warning(self, "Save Preset", f"Preset not found:\n{path}")
            return False

        try:
            self._save_preset_to_path(
                path,
                name,
                created=False,
                notify=False,
                refresh=False,
            )
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Preset save error", str(exc))
            return False

    # Selection itself now performs loading; prompt only when switching would
    # otherwise discard a workflow that differs from the active preset baseline.
    def on_preset_selection_changed(self, _index: int) -> None:
        if getattr(self, "_handling_preset_selection", False):
            return
        selected_name = self.preset_combo.currentText().strip()
        active_name = str(getattr(self, "_active_preset_name", "") or "").strip()
        if not selected_name or selected_name == active_name:
            return

        self._handling_preset_selection = True
        try:
            self.commit_gui_edits()
            has_unsaved_changes = bool(active_name) and self.preset_has_unsaved_changes()
            self.set_preset_dirty_state(has_unsaved_changes)
            if has_unsaved_changes:
                save_prompt = (
                    "Save these changes as a new preset before loading "
                    f"'{selected_name}'?"
                    if is_default_preset(active_name)
                    else f"Save changes to preset '{active_name}' before loading '{selected_name}'?"
                )
                reply = QMessageBox.question(
                    self,
                    "Unsaved Preset Changes",
                    save_prompt,
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Discard
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Save,
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    self.restore_active_preset_selection()
                    return
                if reply == QMessageBox.StandardButton.Save:
                    if not self.save_active_preset_changes():
                        self.restore_active_preset_selection()
                        return
                    target_index = self.preset_combo.findText(selected_name)
                    if target_index >= 0:
                        previous = self.preset_combo.blockSignals(True)
                        self.preset_combo.setCurrentIndex(target_index)
                        self.preset_combo.blockSignals(previous)

            if not self.load_selected_preset():
                self.restore_active_preset_selection()
        finally:
            self._handling_preset_selection = False

    # Block combo-box signals while rebuilding the list to avoid loading a preset
    # merely because its row was temporarily selected during insertion.
    def refresh_presets_list(self, select_name: str | None = None):
        self.ensure_default_preset()

        current = select_name or self.preset_combo.currentText()

        previous = self.preset_combo.blockSignals(True)
        try:
            self.preset_combo.clear()

            preset_files = sorted(
                PRESETS_DIR.glob("*.json"),
                key=lambda p: (p.stem.lower() != "default", p.stem.lower()),
            )
            for p in preset_files:
                if is_default_preset(p.stem) and callable(getattr(self, "shell_icon", None)):
                    self.preset_combo.addItem(self.shell_icon("lock"), p.stem)
                else:
                    self.preset_combo.addItem(p.stem)
                if is_default_preset(p.stem):
                    idx = self.preset_combo.count() - 1
                    self.preset_combo.setItemData(
                        idx,
                        "Default is locked. Save changes as a new preset.",
                        Qt.ItemDataRole.ToolTipRole,
                    )

            if current:
                idx = self.preset_combo.findText(current)
                if idx >= 0:
                    self.preset_combo.setCurrentIndex(idx)
        finally:
            self.preset_combo.blockSignals(previous)
        if hasattr(self, "refresh_preset_control_states"):
            self.refresh_preset_control_states()

    # Preset writes are routine and should not interrupt setup with a modal
    # acknowledgement; show brief feedback in the existing status area instead.
    def show_preset_saved_feedback(self, preset_name: str, *, created: bool) -> None:
        if not hasattr(self, "status_label"):
            return
        action = "created" if created else "saved"
        self.status_label.setText(f"Preset {action}: {preset_name}")
        self.set_status_style("Done")

        timer = getattr(self, "_preset_feedback_timer", None)
        if timer is None:
            timer = QTimer(self.as_qobject())
            timer.setSingleShot(True)
            timer.setInterval(2500)
            timer.timeout.connect(self.clear_preset_saved_feedback)
            self._preset_feedback_timer = timer
        timer.start()

    def show_preset_reverted_feedback(self, preset_name: str) -> None:
        if not hasattr(self, "status_label"):
            return
        self.status_label.setText(f"Preset reverted: {preset_name}")
        self.set_status_style("Done")
        timer = getattr(self, "_preset_feedback_timer", None)
        if timer is None:
            timer = QTimer(self.as_qobject())
            timer.setSingleShot(True)
            timer.setInterval(2500)
            timer.timeout.connect(self.clear_preset_saved_feedback)
            self._preset_feedback_timer = timer
        timer.start()

    # Do not overwrite a newer worker status when delayed preset feedback ends.
    def clear_preset_saved_feedback(self) -> None:
        if self.has_active_processing_task():
            return
        if self.status_label.text().startswith("Preset "):
            self.status_label.setText("Ready")
            self.set_status_style("Ready")

    # Centralize UTF-8 and formatting so every saved settings file is readable
    # by people and behaves identically across platforms.
    def _write_json_file(self, path: Path, data: dict):
        write_json_object(path, data, writer=write_text)

    # A valid JSON array is still invalid preset data; reject it here before
    # callers attempt dictionary lookups and produce less useful errors.
    def _read_json_file(self, path: Path) -> dict:
        return read_json_object(path)

    # Creation and update share the same write and refresh sequence so only their
    # user-facing confirmation differs.
    def _save_preset_to_path(
        self,
        path: Path,
        preset_name: str,
        created: bool,
        *,
        notify: bool = True,
        refresh: bool = True,
    ):
        if is_default_preset(preset_name) or is_default_preset(path.stem):
            raise ValueError("The Default preset is locked. Save the changes as a new preset.")
        self.commit_gui_edits()
        data = self.get_preset_dict()

        self._write_json_file(path, data)
        if refresh:
            self.refresh_presets_list(select_name=preset_name)
        self.remember_preset_state(preset_name, data)

        if created and notify:
            self.show_preset_saved_feedback(preset_name, created=True)
            self.log(f"Saved preset: {path}")
        elif notify:
            self.show_preset_saved_feedback(preset_name, created=False)
            self.log(f"Updated preset: {path}")
        else:
            self.log(f"Saved changes to preset: {path}")

    # Creating a preset may replace an existing file, so require an explicit
    # overwrite decision after the name has been sanitized.
    def save_preset_as(self) -> bool:
        self.ensure_presets_dir()

        dialog = PresetInfoDialog(parent=self)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        name = self.sanitize_preset_name(dialog.get_name())

        if not name:
            QMessageBox.information(self, "Preset", "Preset name cannot be empty.")
            return False
        if is_default_preset(name):
            QMessageBox.information(
                self,
                "Preset",
                "The name Default is reserved for the locked recovery preset. Choose another name.",
            )
            return False

        path = self.preset_path_from_name(name)

        if path.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite Preset",
                f"Preset '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        try:
            self._save_preset_to_path(path, name, created=True)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Preset save error", str(e))
            return False

    # Save the current workflow under the selected preset name.
    def save_to_selected_preset(self):
        self.ensure_presets_dir()

        name = self.preset_combo.currentText().strip()
        if not name:
            self.save_preset_as()
            return
        if is_default_preset(name):
            self.save_preset_as()
            return

        path = self.preset_path_from_name(name)

        if not path.exists():
            QMessageBox.information(self, "Preset", f"Preset not found:\n{path}")
            self.refresh_presets_list()
            return

        try:
            self._save_preset_to_path(path, name, created=False)
        except Exception as e:
            QMessageBox.critical(self, "Preset save error", str(e))

    def export_selected_preset(self) -> None:
        """Export the visible workflow with parent directories removed from known path fields."""
        name = self.preset_combo.currentText().strip()
        if not name:
            QMessageBox.information(self, "Export Preset", "No preset is selected.")
            return

        self.commit_gui_edits()
        data = self.get_preset_dict()
        data["_preset_name"] = name
        exported = privacy_safe_preset(data)

        destination_text, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Preset",
            str(Path.home() / f"{self.sanitize_preset_name(name)}.json"),
            "Cellonaut presets (*.json)",
        )
        if not destination_text:
            return
        destination = Path(destination_text)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        try:
            self._write_json_file(destination, exported)
        except Exception as exc:
            QMessageBox.critical(self, "Preset export error", str(exc))
            return
        self.log(f"[PRESET] Exported preset: {destination}")
        QMessageBox.information(
            self,
            "Preset Exported",
            f"Saved:\n{destination}\n\nParent folders were removed from known path fields. Filenames, panel notes, and other text remain; review them before sharing. The recipient must reselect local paths.",
        )

    def import_preset(self) -> None:
        """Import one JSON preset, sanitize paths, and make it available immediately."""
        source_text, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import Cellonaut Preset",
            str(Path.home()),
            "Cellonaut presets (*.json)",
        )
        if not source_text:
            return
        source = Path(source_text)
        try:
            imported = self._read_json_file(source)
            if not isinstance(imported.get("image_definitions"), list):
                raise ValueError("The selected file does not contain Cellonaut image definitions.")
            imported = privacy_safe_preset(imported)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Preset import error", str(exc))
            return

        proposed_name = self.sanitize_preset_name(str(imported.pop("_preset_name", "") or source.stem))
        imported = {key: value for key, value in imported.items() if not str(key).startswith("_preset_")}
        if not proposed_name:
            QMessageBox.critical(self, "Preset import error", "The imported preset has no usable name.")
            return
        if is_default_preset(proposed_name):
            proposed_name = "Default imported"

        self.ensure_presets_dir()
        destination = self.preset_path_from_name(proposed_name)
        if destination.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite Imported Preset",
                f"Preset '{proposed_name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            self._write_json_file(destination, imported)
            self.refresh_presets_list(select_name=proposed_name)
            self.load_selected_preset()
        except Exception as exc:
            QMessageBox.critical(self, "Preset import error", str(exc))
            return
        self.log(f"[PRESET] Imported preset: {source} -> {destination}")
        QMessageBox.information(
            self,
            "Preset Imported",
            f"Imported '{proposed_name}'.\n\nParent folders were removed from known path fields. Filenames, panel notes, and other text remain. Reselect local folders and model files before use.",
        )

    # Revert is intentionally explicit because it discards every unsaved edit
    # in the selected workflow without changing or deleting the preset file.
    def revert_selected_preset(self):
        self.commit_gui_edits()
        name = self.preset_combo.currentText().strip()
        if not name or not self.preset_has_unsaved_changes():
            return
        reply = QMessageBox.question(
            self,
            "Revert Preset Changes",
            f"Discard unsaved changes and restore preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if self.load_selected_preset():
            self.show_preset_reverted_feedback(name)

    # Workflow presets should not resize the current window, especially when a
    # preset was created on a different display or operating system.
    def load_selected_preset(self) -> bool:
        name = self.preset_combo.currentText().strip()
        if not name:
            QMessageBox.information(self, "Preset", "No preset selected.")
            return False

        path = self.preset_path_from_name(name)
        if not path.exists():
            QMessageBox.information(self, "Preset", f"Preset not found:\n{path}")
            self.refresh_presets_list()
            return False

        try:
            settings = self._read_json_file(path)
            self.apply_preset_dict(settings)
            self.remember_preset_state(name, self.get_preset_dict())

            self.log(f"[PRESET] Loaded preset: {name} ({path})")
            if hasattr(self, "current_sample_label"):
                self.current_sample_label.setText(f"Preset loaded: {name}")
            if hasattr(self, "status_label"):
                self.status_label.setText("Preset loaded")
                self.set_status_style("Done")
            return True
        except (OSError, TypeError, ValueError, RuntimeError) as e:
            QMessageBox.critical(self, "Preset load error", str(e))
            return False

    # Protect Default because startup depends on it as a known recovery state;
    # all other presets require confirmation before deletion.
    def delete_selected_preset(self):
        name = self.preset_combo.currentText().strip()
        if not name:
            QMessageBox.information(self, "Preset", "No preset selected.")
            return

        if is_default_preset(name):
            QMessageBox.information(self, "Preset", "The Default preset cannot be deleted.")
            return

        path = self.preset_path_from_name(name)
        if not path.exists():
            QMessageBox.information(self, "Preset", f"Preset not found:\n{path}")
            self.refresh_presets_list()
            return

        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            path.unlink()
            self.refresh_presets_list(select_name=DEFAULT_PRESET_NAME)
            self.load_selected_preset()
            self.log(f"[PRESET] Deleted preset: {name} ({path})")
        except Exception as e:
            QMessageBox.critical(self, "Preset delete error", str(e))

    # Presets contain reproducible pipeline choices. Fiji belongs to internal
    # runtime state, while ND2 conversion remains in its dedicated importer.
    def get_preset_dict(self):
        """Serialize committed state; save/export actions commit before calling."""
        state = self.configuration_state
        return {
            "input_dir": state.input_dir,
            "output_dir": state.output_dir,
            "mask_source_dir": state.mask_source_dir,
            "reuse_existing_masks": state.reuse_existing_masks,
            "measurement_options": dict(state.measurement_options),
            "image_definitions": deepcopy(state.image_definitions),
            "panel_notes": dict(state.panel_notes),
        }

    def apply_preset_dict(self, s, *, schedule_scan: bool = True):
        """Normalize preset data into editable state, then render without committing old widgets.

        Legacy migration occurs before rendering. Suppress path scans while
        rebuilding controls, then optionally schedule one scan of the final path.
        ND2 import settings remain independent of the analysis preset.
        """
        state = self.configuration_state
        definitions = self.normalize_image_definitions(s.get("image_definitions", state.image_definitions))
        options = default_measurement_options()
        loaded_measurements = s.get("measurement_options", {})
        if isinstance(loaded_measurements, dict):
            for key in options:
                if key in loaded_measurements:
                    options[key] = coerce_bool(loaded_measurements[key], options[key])
        state.image_definitions = definitions
        state.measurement_options = options
        state.reuse_existing_masks = coerce_bool(s.get("reuse_existing_masks", False))
        for key in ("input_dir", "output_dir", "mask_source_dir"):
            if key in s:
                setattr(state, key, str(s[key] or "").strip())

        self._loading_gui_configuration = True
        self._suppress_input_path_scan = True
        try:
            for key in ("input_dir", "output_dir", "mask_source_dir"):
                widget = getattr(self, key, None)
                if widget is not None:
                    widget.set(getattr(state, key))
            if hasattr(self, "reuse_existing_masks_checkbox"):
                self.reuse_existing_masks_checkbox.setChecked(state.reuse_existing_masks)
            self.pipeline_panel_notes = normalize_panel_notes(s.get("panel_notes", {}))
            active_panel_index = getattr(self, "_active_pipeline_section_index", None)
            if active_panel_index is not None and hasattr(self, "show_pipeline_panel_note"):
                self.show_pipeline_panel_note(active_panel_index)

            self.rebuild_image_rows(sync_from_ui=False)

            # Loading an analysis preset must not reset or hide the separate ND2
            # import workflow, so refresh only pipeline layout state here.
            self.apply_standard_layout_spacing()
            self.set_status_style(self.status_label.text())

        finally:
            self._loading_gui_configuration = False
            self._suppress_input_path_scan = False
        state.input_structure = self.get_current_input_structure(require_ready=False)

        # Path signals were suppressed while the preset was reconstructed, so
        # start one scan now with the final path and channel definitions.
        input_path = self.input_dir.get().strip() if hasattr(self, "input_dir") else ""
        if schedule_scan and input_path and hasattr(self, "schedule_input_path_scan"):
            self.schedule_input_path_scan(input_path, immediate=True)
