"""Image and mask processing settings table controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QWidget

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_SCOPE_MEASUREMENT,
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    normalize_qc_filter_mode,
)
from cellonaut.config.relationships import image_display_names
from cellonaut.gui.image_processing_settings import CellonautGuiImageProcessingSettingsMixin
from cellonaut.gui.mask_processing_settings import CellonautGuiMaskProcessingSettingsMixin
from cellonaut.gui.widgets import fit_processing_table_columns, fit_table_height


# Share row enablement, ordering, and editor-state behavior between image and
# mask recipes so both panels persist and validate steps identically.
class CellonautGuiProcessingSettingsMixin(
    CellonautGuiImageProcessingSettingsMixin,
    CellonautGuiMaskProcessingSettingsMixin,
):
    # The first recipe-table column contains every row action.  Keep this
    # comfortably wider than the combined minimums so Qt never clips the
    # square delete or enable controls at the cell boundary.
    PROCESSING_STEP_COLUMN_WIDTH = 360

    def on_processing_step_type_changed(self, *_args) -> None:
        """Persist and rebuild either processing recipe after its operation changes."""
        if self._rebuilding_image_tabs:
            return
        self.commit_gui_edits()
        self.rebuild_mask_processing_table()

    def processing_child_with_role(
        self, parent: QWidget | None, role: str, widget_type: type[QWidget] = QWidget
    ) -> QWidget | None:
        """Return the first child control assigned a processing-table role."""
        if not isinstance(parent, QWidget):
            return None
        return next(
            (child for child in parent.findChildren(widget_type) if child.property("processingRole") == role),
            None,
        )

    def activate_processing_step_selector(self, button, step_type: str, label: str, refresh_style) -> None:
        """Select an operation and activate its shared recipe row."""
        button.setProperty("stepType", step_type)
        button.setText(label)
        refresh_style(button)
        table = button.parentWidget()
        while table is not None and not isinstance(table, QTableWidget):
            table = table.parentWidget()
        enabled_button = None
        if isinstance(table, QTableWidget):
            for row_idx in range(table.rowCount()):
                step_cell = table.cellWidget(row_idx, 0)
                if step_cell is not None and (step_cell is button.parentWidget() or step_cell.isAncestorOf(button)):
                    enabled_button = self.processing_child_with_role(
                        table.cellWidget(row_idx, 1), "enabled", QPushButton
                    )
                    break
        if isinstance(enabled_button, QPushButton):
            enabled_button.setChecked(True)
        self.on_processing_step_type_changed()

    # Keep inactive Cellpose settings visible so users can review them before enabling segmentation.
    def set_cellpose_row_active_state(self, table, row_idx: int, active: bool):
        state = "active" if active else "inactive"
        for col_idx in range(table.columnCount()):
            widget = table.cellWidget(row_idx, col_idx)
            if widget is None:
                continue

            widgets = [widget, *widget.findChildren(QWidget)]
            if col_idx == 0:
                widget.setEnabled(True)
                for current_widget in widgets:
                    role = current_widget.property("processingRole")
                    current_widget.setProperty("processingRowState", "" if role == "enabled" else state)
                    current_widget.setEnabled(True)
                    current_widget.style().unpolish(current_widget)
                    current_widget.style().polish(current_widget)
                self.set_widget_muted(widget, False)
                continue

            widget.setEnabled(bool(active))
            for current_widget in widgets:
                current_widget.setProperty("processingRowState", state)
                current_widget.style().unpolish(current_widget)
                current_widget.style().polish(current_widget)
            self.set_widget_muted(widget, not active)

    def refresh_cellpose_settings_table_active_states(self):
        if not hasattr(self, "cellpose_settings_table"):
            return
        table = self.cellpose_settings_table
        for row_idx in range(table.rowCount()):
            enable_widget = table.cellWidget(row_idx, 0)
            self.set_cellpose_row_active_state(
                table,
                row_idx,
                self.processing_row_enabled_from_header(enable_widget, fallback=False),
            )

    # A row remains relevant when either cell segmentation or any mask relationship uses it.
    def analysis_row_has_active_mask(self, image_def: dict) -> bool:
        if list(image_def.get("analysis_cellpose_mask_sources", []) or []):
            return True
        if str(image_def.get("analysis_cellpose_mask_source", "") or "").strip():
            return True
        relationships = dict(image_def.get("mask_relationships", {}) or {})
        return any(bool(value) for value in relationships.values())

    def display_filter_mode(self, mode: str) -> str:
        return normalize_qc_filter_mode(mode)

    def make_processing_header_cell(self) -> QWidget:
        widget = QWidget()
        widget.setProperty("uiRole", "tableHeaderCell")
        return widget

    def refresh_processing_enabled_button_style(self, button: QPushButton) -> None:
        button.setText("")
        button.setProperty("enabledState", "on" if button.isChecked() else "off")
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    # Use one compact enable control for Cellpose, image processing, and mask processing rows.
    def make_processing_enabled_button(self, enabled: bool, tooltip: str) -> QPushButton:
        button = QPushButton()
        button.setCheckable(True)
        button.setChecked(bool(enabled))
        button.setProperty("processingRole", "enabled")
        button.setProperty("uiRole", "processingEnableButton")
        button.setFixedSize(18, 18)
        button.setStyleSheet("padding: 0; min-width: 16px; max-width: 16px; min-height: 16px; max-height: 16px;")
        button.setToolTip(tooltip)
        button.toggled.connect(lambda _checked=False, btn=button: self.refresh_processing_enabled_button_style(btn))
        self.refresh_processing_enabled_button_style(button)
        return button

    def make_processing_remove_button(
        self, step_index: int, tooltip: str, callback, *, removable: bool = True
    ) -> QPushButton:
        """Create the shared square delete action used by both recipe tables."""
        button = QPushButton()
        button.setProperty("processingRole", "remove")
        button.setProperty("uiRole", "processingActionButton")
        button.setProperty("cellonautIconName", "trash-2")
        icon_factory = cast(Callable[[str, int], QIcon] | None, getattr(self, "shell_icon", None))
        if callable(icon_factory):
            button.setIcon(icon_factory("trash-2", 13))
        button.setIconSize(QSize(13, 13))
        button.setFixedSize(22, 22)
        button.setStyleSheet("padding: 0; min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px;")
        button.setEnabled(removable)
        button.setToolTip(tooltip)
        button.setAccessibleName("Remove processing step")
        button.clicked.connect(callback)
        return button

    def processing_row_enabled_from_header(self, header_widget: QWidget | None, fallback: bool = True) -> bool:
        enabled_button = self.processing_child_with_role(header_widget, "enabled", QPushButton)
        if isinstance(enabled_button, QPushButton):
            return bool(enabled_button.isChecked())
        return bool(fallback)

    # Make inactive values read-only rather than clearing them so toggling a row is reversible.
    def set_processing_table_row_active_state(self, table, row_idx: int, active: bool) -> None:
        state = "active" if active else "inactive"
        for col_idx in range(table.columnCount()):
            widget = table.cellWidget(row_idx, col_idx)
            if widget is None:
                continue
            widgets = [widget, *widget.findChildren(QWidget)]
            for current_widget in widgets:
                role = current_widget.property("processingRole")
                if role in {"enabled", "remove"}:
                    current_widget.setProperty("processingRowState", "")
                else:
                    current_widget.setProperty("processingRowState", state)

                if col_idx > 0 and role not in {"enabled", "remove"}:
                    if isinstance(current_widget, QLineEdit):
                        if current_widget.property("processingBaseReadOnly") is None:
                            current_widget.setProperty("processingBaseReadOnly", bool(current_widget.isReadOnly()))
                        current_widget.setReadOnly(
                            bool((not active) or current_widget.property("processingBaseReadOnly"))
                        )

                current_widget.style().unpolish(current_widget)
                current_widget.style().polish(current_widget)
                current_widget.update()

    # Both recipe tables use the same corner label, source headings, and add row.
    def prepare_processing_table(
        self,
        table,
        rows: list[tuple[int, dict]],
        names: list[str],
        step_count: int,
        step_header: str,
    ) -> None:
        table.blockSignals(True)
        table.clear()
        table.clearSpans()
        table.setRowCount(step_count + 1)
        table.setColumnCount(len(rows) + 2)
        table.setHorizontalHeaderLabels(
            [step_header, "Enable", *[names[source_index] for source_index, _ in rows]]
        )
        table.horizontalHeader().setVisible(True)
        table.verticalHeader().setVisible(False)

    # Shared recipes are active when any source retains the corresponding step.
    def merged_processing_step_enabled(
        self,
        rows: list[tuple[int, dict]],
        steps_by_source: dict[int, list[dict]],
        step_index: int,
        fallback: dict,
    ) -> bool:
        present = []
        for source_index, _image_def in rows:
            steps = steps_by_source[source_index]
            if step_index < len(steps):
                present.append(steps[step_index])
        if present and "row_enabled" in present[0]:
            return bool(present[0].get("row_enabled", True))
        return any(bool(step.get("enabled", True)) for step in present) if present else bool(
            fallback.get("row_enabled", fallback.get("enabled", True))
        )

    def make_processing_enable_cell(self, button: QPushButton) -> QWidget:
        """Center a compact row toggle in the dedicated Enable column."""
        widget = QWidget()
        widget.setProperty("uiRole", "matrixActionCell")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(button)
        return widget

    # The final row differs only in its label and callback between recipe types.
    def add_processing_table_button(self, table, row_idx: int, tooltip: str, callback) -> None:
        header = self.make_processing_header_cell()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button = QPushButton("Add step")
        button.setToolTip(tooltip)
        button.setFixedWidth(120)
        button.clicked.connect(lambda _checked=False: callback())
        layout.addStretch(1)
        layout.addWidget(button)
        layout.addStretch(1)
        table.setCellWidget(row_idx, 0, header)

    # Persist every channel in one pass so shared row order cannot drift between definitions.
    def save_image_processing_table_to_definitions(self, active_defs: list[dict]) -> None:
        if not hasattr(self, "image_processing_table"):
            return
        step_definitions = self.image_processing_step_definitions_by_type()
        table_specs = (
            (
                self.image_processing_table,
                list(getattr(self, "_image_processing_channel_refs", [])),
                list(getattr(self, "_image_processing_step_refs", [])),
                IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            ),
            (
                getattr(self, "measurement_processing_table", None),
                list(getattr(self, "_measurement_processing_channel_refs", [])),
                list(getattr(self, "_measurement_processing_step_refs", [])),
                IMAGE_PROCESSING_SCOPE_MEASUREMENT,
            ),
        )
        table_specs = tuple(
            spec for spec in table_specs
            if spec[0] is not None and spec[1] and spec[0].rowCount() > 0
        )
        active_scopes = {scope for _table, _channels, _steps, scope in table_specs}
        saved_by_source: dict[int, list[dict]] = {
            source_index: [
                step
                for step in self.normalized_image_processing_steps_for_definition(image_def)
                if str(step.get("scope", "") or "") not in active_scopes
            ]
            for source_index, image_def in enumerate(active_defs)
            if self.is_physical_channel_definition(image_def)
        }
        for table, channel_refs, step_refs, step_scope in table_specs:
            if table is None:
                continue
            for channel_ref in channel_refs:
                source_index = int(channel_ref.get("source_index", -1))
                saved_by_source.setdefault(source_index, [])
            for step_ref in step_refs:
                row_idx = int(step_ref.get("row_idx", -1))
                header_widget = table.cellWidget(row_idx, 0)
                enable_widget = table.cellWidget(row_idx, 1)
                step_selector = self.processing_child_with_role(header_widget, "step_type")
                step_type = self.image_processing_step_type_from_widget(
                    step_selector, str(step_ref.get("step_type", "") or "")
                )
                if not step_type:
                    if bool(step_ref.get("synthetic", False)):
                        continue
                    for channel_ref in channel_refs:
                        source_index = int(channel_ref.get("source_index", -1))
                        if source_index >= 0:
                            saved_by_source.setdefault(source_index, []).append(
                                {
                                    "type": "",
                                    "enabled": False,
                                    "row_enabled": False,
                                    "scope": step_scope,
                                    "params": {},
                                }
                            )
                    continue
                step_definition = step_definitions.get(step_type)
                if step_definition is None or step_scope not in step_definition.get("allowed_scopes", []):
                    continue
                param_key = str(step_definition.get("param_key", "") or "")
                step_enabled = self.processing_row_enabled_from_header(
                    enable_widget, bool(step_ref.get("enabled", True))
                )
                for channel_ref in channel_refs:
                    source_index = int(channel_ref.get("source_index", -1))
                    col_idx = int(channel_ref.get("col_idx", -1))
                    if source_index < 0 or source_index >= len(active_defs) or col_idx < 0:
                        continue
                    value_widget = table.cellWidget(row_idx, col_idx)
                    cell_enabled = self.image_processing_cell_enabled(value_widget)
                    value = ""
                    options = dict(value_widget.property("fijiOptions") or {}) if value_widget else {}
                    if param_key and cell_enabled:
                        value = self.image_processing_value_from_widget(
                            value_widget, str(active_defs[source_index].get(param_key, "") or "")
                        )
                    saved_by_source[source_index].append(
                        {
                            "type": step_type,
                            "enabled": bool(step_enabled and cell_enabled),
                            "row_enabled": bool(step_enabled),
                            "scope": step_scope,
                            "params": ({param_key: value, **options} if param_key else {}),
                        }
                    )
                    if param_key:
                        active_defs[source_index][param_key] = value

        for source_index, steps in saved_by_source.items():
            if 0 <= source_index < len(active_defs):
                active_defs[source_index]["image_processing_steps"] = steps

    def _commit_processing_recipe_change(self, active_defs: list[dict]) -> None:
        self.image_definitions = self.normalize_image_definitions(active_defs)
        self.rebuild_mask_processing_table()

    def _change_shared_processing_recipe(
        self,
        active_defs: list[dict],
        *,
        masks: bool,
        action: str,
        step_index: int = 0,
        target_step: int = 0,
        new_step: dict | None = None,
    ) -> bool:
        """Apply one row edit to every channel or mask in its shared recipe."""
        matches = self.is_mask_only_definition if masks else self.is_physical_channel_definition
        indices = [index for index, image_def in enumerate(active_defs) if matches(image_def)]
        if not indices:
            return False
        storage_key = "mask_processing_steps" if masks else "image_processing_steps"
        normalize = (
            self.normalized_mask_processing_steps_for_definition
            if masks
            else self.normalized_image_processing_steps_for_definition
        )
        order = None
        fallback: list[dict] = []
        if action == "reorder":
            base_steps = normalize(active_defs[indices[0]])
            if not 0 <= step_index < len(base_steps):
                return False
            target_step = max(0, min(target_step, len(base_steps) - 1))
            if step_index == target_step:
                return False
            order = list(range(len(base_steps)))
            order.insert(target_step, order.pop(step_index))
            fallback = [base_steps[index] for index in order]

        changed = False
        for definition_index in indices:
            steps = normalize(active_defs[definition_index])
            if order is not None:
                steps = [steps[index] for index in order] if len(steps) == len(order) else list(fallback)
            elif action == "add":
                added_step = dict(new_step or {})
                if steps and not str(steps[0].get("type", "") or "") and str(added_step.get("type", "") or ""):
                    steps[0] = added_step
                else:
                    steps.append(added_step)
            elif 0 <= step_index < len(steps):
                steps.pop(step_index)
            else:
                continue
            active_defs[definition_index][storage_key] = steps
            changed = True
        return changed

    def _change_image_scope_recipe(
        self, scope: str, action: str, step_index: int = 0, target_step: int = 0, step_type: str = ""
    ) -> None:
        if self._rebuilding_image_tabs:
            return
        self.commit_gui_edits()
        active_defs = self.get_active_image_definitions()
        definition = self.image_processing_step_definitions_by_type().get(step_type, {})
        if step_type and scope not in definition.get("allowed_scopes", []):
            return
        param_key = str(definition.get("param_key", "") or "")
        for image_def in active_defs:
            if not self.is_physical_channel_definition(image_def):
                continue
            all_steps = self.normalized_image_processing_steps_for_definition(image_def)
            scoped = [step for step in all_steps if str(step.get("scope", "") or "") == scope]
            other = [step for step in all_steps if str(step.get("scope", "") or "") != scope]
            if action == "reorder" and 0 <= step_index < len(scoped):
                destination = max(0, min(target_step, len(scoped) - 1))
                scoped.insert(destination, scoped.pop(step_index))
            elif action == "remove" and 0 <= step_index < len(scoped):
                scoped.pop(step_index)
            elif action == "add":
                if not scoped and not step_type:
                    scoped.append(
                        {
                            "type": "",
                            "enabled": False,
                            "row_enabled": False,
                            "scope": scope,
                            "params": {},
                        }
                    )
                scoped.append(
                    {
                        "type": step_type,
                        "enabled": bool(step_type),
                        "row_enabled": bool(step_type),
                        "scope": scope,
                        "params": ({param_key: ""} if param_key else {}),
                    }
                )
            image_def["image_processing_steps"] = [*other, *scoped]
        self._commit_processing_recipe_change(active_defs)

    # Unqualified image-processing actions edit the segmentation-input recipe.
    def reorder_image_processing_recipe(self, source_step: int, target_step: int):
        self._change_image_scope_recipe(
            IMAGE_PROCESSING_SCOPE_SEGMENTATION, "reorder", source_step, target_step
        )

    def reorder_measurement_processing_recipe(self, source_step: int, target_step: int):
        refs = list(getattr(self, "_measurement_processing_step_refs", []) or [])
        if 0 <= source_step < len(refs) and refs[source_step].get("step_type") == "rolling_ball_background":
            return
        self._change_image_scope_recipe(
            IMAGE_PROCESSING_SCOPE_MEASUREMENT, "reorder", source_step, target_step
        )

    # Add the row to every channel so their steps stay aligned.
    def add_image_processing_step(self, step_type: str | None = None):
        self._change_image_scope_recipe(
            IMAGE_PROCESSING_SCOPE_SEGMENTATION, "add", step_type="" if step_type is None else str(step_type)
        )

    def add_measurement_processing_step(self, step_type: str | None = None):
        self._change_image_scope_recipe(
            IMAGE_PROCESSING_SCOPE_MEASUREMENT, "add", step_type="" if step_type is None else str(step_type)
        )

    # Remove a shared row from every channel to preserve index-based alignment.
    def remove_image_processing_step(self, step_index: int):
        if self._rebuilding_image_tabs:
            return
        self._change_image_scope_recipe(IMAGE_PROCESSING_SCOPE_SEGMENTATION, "remove", step_index)

    def remove_measurement_processing_step(self, step_index: int):
        if self._rebuilding_image_tabs:
            return
        self._change_image_scope_recipe(IMAGE_PROCESSING_SCOPE_MEASUREMENT, "remove", step_index)

    # The first populated mask defines the row order shown across all mask columns.
    def mask_processing_recipe_from_mask_rows(self, mask_rows: list[tuple[int, dict]]) -> list[dict]:
        for _source_index, image_def in mask_rows:
            steps = self.normalized_mask_processing_steps_for_definition(image_def)
            if steps:
                return steps
        return []

    # Keep recipe order identical across masks because each table row represents one operation.
    def reorder_mask_processing_recipe(self, source_step: int, target_step: int):
        if self._rebuilding_image_tabs:
            return
        self.commit_gui_edits()
        active_defs = self.get_active_image_definitions()
        changed = self._change_shared_processing_recipe(
            active_defs,
            masks=True,
            action="reorder",
            step_index=source_step,
            target_step=target_step,
        )
        if changed:
            self._commit_processing_recipe_change(active_defs)

    # Add placeholders to every mask so choosing a step never misaligns later rows.
    def add_mask_processing_step(self, step_type: str | None = None):
        if self._rebuilding_image_tabs:
            return
        self.commit_gui_edits()
        active_defs = self.get_active_image_definitions()
        step_type = "" if step_type is None else str(step_type)
        definition = self.mask_settings_step_definitions_by_type().get(step_type, {})
        param_key = str(definition.get("param_key", "") or "")
        new_step = {
            "type": step_type,
            "enabled": bool(step_type),
            "row_enabled": bool(step_type),
            "params": ({param_key: ""} if param_key else {}),
        }
        if self._change_shared_processing_recipe(
            active_defs,
            masks=True,
            action="add",
            new_step=new_step,
        ):
            self._commit_processing_recipe_change(active_defs)

    # Delete the corresponding row from every mask definition in one operation.
    def remove_mask_processing_step(self, step_index: int):
        if self._rebuilding_image_tabs:
            return
        self.commit_gui_edits()
        active_defs = self.get_active_image_definitions()
        if self._change_shared_processing_recipe(
            active_defs,
            masks=True,
            action="remove",
            step_index=step_index,
        ):
            self._commit_processing_recipe_change(active_defs)

    def _build_image_scope_processing_table(
        self,
        table,
        channel_rows: list[tuple[int, dict]],
        names: list[str],
        scope: str,
        ref_prefix: str,
    ) -> None:
        scoped_steps_by_source = {
            source_index: [
                step
                for step in self.normalized_image_processing_steps_for_definition(image_def)
                if str(step.get("scope", "") or "") == scope
            ]
            for source_index, image_def in channel_rows
        }
        recipe_steps = next((steps for steps in scoped_steps_by_source.values() if steps), [])
        synthetic_placeholder = not recipe_steps
        if not recipe_steps:
            recipe_steps = [
                {"type": "", "enabled": False, "row_enabled": False, "scope": scope, "params": {}}
            ]
        step_definitions = self.image_processing_step_definitions_by_type()
        step_refs = [
            {
                "step_index": step_index,
                "row_idx": step_index,
                "step_type": str(step.get("type", "") or ""),
                "scope": scope,
                "enabled": self.merged_processing_step_enabled(
                    channel_rows, scoped_steps_by_source, step_index, step
                ),
                "synthetic": synthetic_placeholder,
            }
            for step_index, step in enumerate(recipe_steps)
        ]
        channel_refs = [
            {"source_index": source_index, "col_idx": col_idx + 2}
            for col_idx, (source_index, _image_def) in enumerate(channel_rows)
        ]
        setattr(self, f"_{ref_prefix}_step_refs", step_refs)
        setattr(self, f"_{ref_prefix}_channel_refs", channel_refs)
        if ref_prefix == "image_processing":
            self._image_processing_row_refs = []
        else:
            self._measurement_processing_row_refs = []

        header = "Before mask creation" if scope == IMAGE_PROCESSING_SCOPE_SEGMENTATION else "Before measurement"
        self.prepare_processing_table(table, channel_rows, names, len(recipe_steps), header)
        remove_callback = (
            self.remove_image_processing_step
            if scope == IMAGE_PROCESSING_SCOPE_SEGMENTATION
            else self.remove_measurement_processing_step
        )
        add_callback = (
            self.add_image_processing_step
            if scope == IMAGE_PROCESSING_SCOPE_SEGMENTATION
            else self.add_measurement_processing_step
        )
        for step_index, step in enumerate(recipe_steps):
            row_idx = step_index
            step_type = str(step.get("type", "") or "")
            step_enabled = self.merged_processing_step_enabled(
                channel_rows, scoped_steps_by_source, step_index, step
            )
            header_widget = self.make_processing_header_cell()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(4, 2, 4, 2)
            header_layout.setSpacing(4)
            drag_handle = QLabel("|||")
            drag_handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            background_output_row = scope == IMAGE_PROCESSING_SCOPE_MEASUREMENT and step_type == "rolling_ball_background"
            drag_handle.setCursor(
                Qt.CursorShape.ArrowCursor if background_output_row else Qt.CursorShape.SizeAllCursor
            )
            drag_handle.setProperty("processingRole", "drag_handle")
            drag_handle.setProperty("uiRole", "tableHeaderCellLabel")
            drag_handle.setFixedWidth(26)
            drag_handle.setToolTip(
                "Background-corrected results are always calculated after the other image steps."
                if background_output_row
                else "Drag this sidebar up or down to reorder steps in this stage."
            )
            lock_indicator = None
            if background_output_row:
                lock_indicator = QLabel()
                lock_indicator.setProperty("processingRole", "locked_step")
                lock_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lock_indicator.setFixedSize(18, 18)
                lock_indicator.setToolTip(
                    "Locked last: each radius creates an independent background-corrected measurement "
                    "from the image produced by all preceding steps."
                )
                lock_indicator.setAccessibleName("Processing step locked last")
                icon_factory = cast(Callable[[str, int], QIcon] | None, getattr(self, "shell_icon", None))
                if callable(icon_factory):
                    lock_indicator.setPixmap(icon_factory("lock", 13).pixmap(QSize(13, 13)))
                else:
                    lock_indicator.setText("🔒")
            remove_btn = self.make_processing_remove_button(
                step_index,
                "Choose an operation before removing this row."
                if synthetic_placeholder
                else "Remove this processing step.",
                lambda _checked=False, sidx=step_index, callback=remove_callback: callback(sidx),
                removable=bool(step_type) or not synthetic_placeholder,
            )
            step_combo = self.make_image_processing_step_menu_button(step_type, scope)
            warning_label = self.make_image_processing_warning_label(step_type, scope)
            enabled_btn = self.make_processing_enabled_button(
                step_enabled,
                "Enable or disable this processing row for all image channels.",
            )
            enabled_btn.toggled.connect(
                lambda checked=False, tbl=table, ridx=row_idx: self.set_processing_table_row_active_state(
                    tbl, ridx, bool(checked)
                )
            )
            enabled_btn.toggled.connect(self.on_processing_table_changed)
            header_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            header_layout.addWidget(remove_btn, 0, Qt.AlignmentFlag.AlignVCenter)
            header_layout.addWidget(drag_handle, 0, Qt.AlignmentFlag.AlignCenter)
            if lock_indicator is not None:
                header_layout.addWidget(lock_indicator, 0, Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(step_combo, 1, Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(warning_label)
            table.setCellWidget(row_idx, 0, header_widget)
            table.setCellWidget(row_idx, 1, self.make_processing_enable_cell(enabled_btn))

            definition = step_definitions.get(step_type, {})
            param_key = str(definition.get("param_key", "") or "")
            for col_idx, (source_index, image_def) in enumerate(channel_rows, start=2):
                source_steps = scoped_steps_by_source.get(source_index, [])
                source_step = source_steps[step_index] if step_index < len(source_steps) else step
                params = dict(source_step.get("params", {}) or {})
                value = str(params.get(param_key, image_def.get(param_key, "")) or "")
                cell_enabled = bool(source_step.get("enabled", False))
                if param_key and not value:
                    cell_enabled = False
                table.setCellWidget(
                    row_idx,
                    col_idx,
                    self.make_image_processing_value_cell(
                        param_key, value, enabled=cell_enabled, step_type=step_type, params=params, scope=scope
                    ),
                )
                row_refs = getattr(self, f"_{ref_prefix}_row_refs")
                row_refs.append(
                    {
                        "row_idx": row_idx,
                        "source_index": source_index,
                        "col_idx": col_idx,
                        "step_type": step_type,
                        "param_key": param_key,
                    }
                )
            self.set_processing_table_row_active_state(table, row_idx, step_enabled)

        self.add_processing_table_button(table, len(recipe_steps), f"Add a step to {header.lower()}.", add_callback)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        for row_idx in range(len(recipe_steps)):
            table.setRowHeight(row_idx, 44)
        table.setRowHeight(len(recipe_steps), 44)
        fit_processing_table_columns(table, control_width=self.PROCESSING_STEP_COLUMN_WIDTH)
        if any(ref["step_type"] in {"remove_outliers", "bit_depth"} for ref in step_refs):
            for col_idx in range(2, table.columnCount()):
                table.setColumnWidth(col_idx, 210)
        elif any(ref["step_type"] in {"rolling_ball_background", "enhance_contrast"} for ref in step_refs):
            for col_idx in range(2, table.columnCount()):
                table.setColumnWidth(col_idx, 145)
        fit_table_height(table, minimum=112, maximum=288)
        table.blockSignals(False)

    # Rebuild both recipe tables together because channel and mask definitions share one model.
    def rebuild_mask_processing_table(self):
        if not hasattr(self, "mask_processing_table"):
            return

        active_defs = self.get_active_image_definitions()
        names = image_display_names(active_defs)
        channel_rows = [
            (index, image_def)
            for index, image_def in enumerate(active_defs)
            if self.is_physical_channel_definition(image_def)
        ]
        mask_rows = [
            (index, image_def) for index, image_def in enumerate(active_defs) if self.is_mask_only_definition(image_def)
        ]

        if hasattr(self, "image_processing_table"):
            self._build_image_scope_processing_table(
                self.image_processing_table,
                channel_rows,
                names,
                IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "image_processing",
            )
        if hasattr(self, "measurement_processing_table"):
            self._build_image_scope_processing_table(
                self.measurement_processing_table,
                channel_rows,
                names,
                IMAGE_PROCESSING_SCOPE_MEASUREMENT,
                "measurement_processing",
            )

        table = self.mask_processing_table

        mask_steps = self.mask_processing_recipe_from_mask_rows(mask_rows)
        synthetic_mask_placeholder = not mask_steps
        if not mask_steps:
            mask_steps = [{"type": "", "enabled": False, "row_enabled": False, "params": {}}]
        mask_step_definitions = self.mask_settings_step_definitions_by_type()
        mask_steps_by_source = {
            source_index: self.normalized_mask_processing_steps_for_definition(image_def)
            for source_index, image_def in mask_rows
        }

        self._mask_settings_step_refs = [
            {
                "step_index": step_index,
                "row_idx": step_index,
                "step_type": str(step.get("type", "") or ""),
                "enabled": self.merged_processing_step_enabled(mask_rows, mask_steps_by_source, step_index, step),
            }
            for step_index, step in enumerate(mask_steps)
        ]
        self._mask_settings_mask_refs = [
            {"source_index": source_index, "col_idx": col_idx + 2}
            for col_idx, (source_index, _image_def) in enumerate(mask_rows)
        ]

        self.prepare_processing_table(table, mask_rows, names, len(mask_steps), "Mask processing steps")

        for step_index, step in enumerate(mask_steps):
            row_idx = step_index
            step_type = str(step.get("type", "") or "")
            step_enabled = self.merged_processing_step_enabled(mask_rows, mask_steps_by_source, step_index, step)
            definition = mask_step_definitions.get(step_type, {})
            param_key = str(definition.get("param_key", "") or "")

            mask_header_widget = self.make_mask_settings_step_header(
                step,
                step_index,
                removable=bool(step_type) or not synthetic_mask_placeholder,
            )
            table.setCellWidget(row_idx, 0, mask_header_widget)
            mask_enabled_btn = self.make_processing_enabled_button(
                step_enabled,
                "Enable or disable this processing row for every mask column.",
            )
            mask_enabled_btn.toggled.connect(self.on_processing_table_changed)
            mask_enabled_btn.toggled.connect(
                lambda checked=False, tbl=table, ridx=row_idx: self.set_processing_table_row_active_state(
                    tbl,
                    ridx,
                    bool(checked),
                )
            )
            table.setCellWidget(row_idx, 1, self.make_processing_enable_cell(mask_enabled_btn))
            for col_idx, (source_index, image_def) in enumerate(mask_rows, start=2):
                mask_channel_steps = mask_steps_by_source.get(source_index, [])
                mask_step = mask_channel_steps[step_index] if step_index < len(mask_channel_steps) else step
                params = dict(mask_step.get("params", {}) or {})
                raw_value = params.get(param_key, "") if param_key else None
                value = ("" if raw_value is None else str(raw_value)) if param_key else None
                cell_enabled = bool(mask_step.get("enabled", True))
                table.setCellWidget(
                    row_idx,
                    col_idx,
                    self.make_mask_settings_value_cell(
                        step_type,
                        image_def,
                        value=value,
                        enabled=cell_enabled,
                    ),
                )

            self.set_processing_table_row_active_state(table, row_idx, step_enabled)

        self.add_processing_table_button(
            table, len(mask_steps), "Add a mask processing step.", self.add_mask_processing_step
        )

        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        for row_idx in range(len(mask_steps)):
            table.setRowHeight(row_idx, 44)
        table.setRowHeight(len(mask_steps), 44)
        fit_processing_table_columns(table, control_width=self.PROCESSING_STEP_COLUMN_WIDTH)
        if any(ref["step_type"] in {"binary_erode", "binary_dilate", "binary_open", "binary_close", "analyze_particles", "translate"}
               for ref in self._mask_settings_step_refs):
            for col_idx in range(2, table.columnCount()):
                table.setColumnWidth(col_idx, 245)
        fit_table_height(table, minimum=112, maximum=288)
        table.blockSignals(False)
        self.validate_mask_processing_table()

    @staticmethod
    def _reset_mask_processing_values(image_def):
        image_def["mask_processing_steps"] = []

    def _saved_mask_processing_step(self, table, step_ref, col_idx, definitions):
        row_idx = int(step_ref.get("row_idx", -1))
        header_widget = table.cellWidget(row_idx, 0)
        enable_widget = table.cellWidget(row_idx, 1)
        step_selector = self.processing_child_with_role(header_widget, "mask_step_type")
        previous_type = str(step_ref.get("step_type", "") or "")
        step_type = self.mask_settings_step_type_from_widget(step_selector, previous_type)
        if not step_type:
            return {"type": "", "enabled": False, "row_enabled": False, "params": {}}

        definition = definitions.get(step_type)
        if definition is None:
            return None
        param_key = str(definition.get("param_key", "") or "")
        value_widget = table.cellWidget(row_idx, col_idx)
        row_enabled = self.processing_row_enabled_from_header(
            enable_widget,
            bool(step_ref.get("enabled", True)),
        )
        newly_selected = bool(step_type and not previous_type)
        if newly_selected:
            row_enabled = True
        cell_enabled = self.mask_settings_cell_enabled(value_widget)
        params = {}
        if param_key:
            params[param_key] = self.mask_settings_value_from_widget(
                value_widget,
                str(definition.get("default", "") or ""),
            )
        else:
            self.mask_settings_value_from_widget(value_widget, "false")
        return {
            "type": step_type,
            "enabled": bool(row_enabled and cell_enabled),
            "row_enabled": bool(row_enabled),
            "params": params,
        }

    # Save table edits before rebuilding the controls.
    def _read_processing_table_edits(self, active_defs: list[dict]) -> None:
        if self._rebuilding_image_tabs:
            return
        if not hasattr(self, "mask_processing_table"):
            return

        table = self.mask_processing_table

        if hasattr(self, "image_processing_table"):
            self.save_image_processing_table_to_definitions(active_defs)

        mask_step_refs = list(getattr(self, "_mask_settings_step_refs", []))
        mask_refs = list(getattr(self, "_mask_settings_mask_refs", []))
        mask_step_definitions = self.mask_settings_step_definitions_by_type()

        for mask_ref in mask_refs:
            source_index = int(mask_ref.get("source_index", -1))
            col_idx = int(mask_ref.get("col_idx", -1))
            if source_index < 0 or source_index >= len(active_defs) or col_idx < 0:
                continue
            image_def = active_defs[source_index]
            self._reset_mask_processing_values(image_def)

            for step_ref in mask_step_refs:
                step = self._saved_mask_processing_step(table, step_ref, col_idx, mask_step_definitions)
                if step is None:
                    continue
                image_def["mask_processing_steps"].append(step)

    def on_processing_table_changed(self, *_args):
        if self._rebuilding_image_tabs:
            return
        self.commit_gui_edits()
        self.refresh_channel_name_dependent_ui()
        self.validate_mask_processing_table()
