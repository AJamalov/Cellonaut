"""Inline analysis settings and filtered-result export behavior."""

from __future__ import annotations

import math
from pathlib import Path

from cellonaut.results.artifacts import ArtifactResolver, artifact_root, image_definition

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QMessageBox, QWidget

from cellonaut.config.defaults import (
    CELLPOSE_MEASUREMENT_KEYS,
    CONFIGURED_MASK_MEASUREMENT_KEYS,
    CONFIGURED_MASK_WITHIN_CELLPOSE_KEYS,
    MASK_INTENSITY_SOURCE_LABELS,
    MEASUREMENT_LABELS,
    MEASUREMENT_METADATA,
    QC_FILTER_MODE_LABELS,
)
from cellonaut.gui.mixin import GuiMixin
from cellonaut.io.writers import write_dataframe_csv
from cellonaut.measurement.table_cleanup import drop_derived_ratio_columns
from cellonaut.masks.cell_qc import CELL_FILTER_METRICS, MASK_FILTER_METRICS, cell_qc_rules_to_text
from cellonaut.masks.preview_filter_overlay import export_all_filtered_result_tables, filtered_table_from_current_rules
from cellonaut.pipeline.run_outputs import next_numbered_child


MEASUREMENT_GROUP_KEYS = (
    ("analysis_basic_measurement_checks_layout", CONFIGURED_MASK_MEASUREMENT_KEYS),
    ("analysis_whole_cell_measurement_checks_layout", CELLPOSE_MEASUREMENT_KEYS),
    ("analysis_cell_measurement_checks_layout", CONFIGURED_MASK_WITHIN_CELLPOSE_KEYS),
)

FILTER_DISPLAY = {
    "Area": ("Cell area", "px²", "Area enclosed by the Cellpose cell mask."),
    "Perimeter": ("Cell perimeter", "px", "Length of the Cellpose cell boundary."),
    "Circularity": ("Cell circularity", "", "4*pi*area/perimeter squared. Pixel-based perimeter estimates can give values above 1."),
    "Solidity": ("Cell solidity", "0–1", "Area divided by convex-hull area."),
    "Mean intensity": ("Average cell intensity", "a.u.", "Average Cellpose-source-channel intensity across the cell."),
    "Minimum intensity": ("Lowest cell intensity", "a.u.", "Lowest Cellpose-source-channel intensity inside the cell."),
    "Maximum intensity": ("Highest cell intensity", "a.u.", "Highest Cellpose-source-channel intensity inside the cell."),
    "Raw integrated density": ("Total raw cell intensity", "a.u. ? px?", "Sum of Cellpose-source-channel pixel values inside the cell."),
    "Integrated density": ("Total cell intensity", "a.u. ? px?", "Same pixel sum as RawIntDen in this cell table; background correction is a separate measurement."),
    "Mask area": ("Target mask area", "px²", "Area of the selected target mask inside the cell."),
    "Mask fraction of cell area": ("Cell area covered by target", "% or 0–1", "Fraction of the cell occupied by the selected target mask."),
    "Mean intensity inside mask": ("Average intensity inside target", "a.u.", "Average intensity where the target mask overlaps the cell, using the selected target intensity source."),
    "Mask integrated density": ("Total target intensity", "a.u. ? px?", "Sum of pixel values where the target mask overlaps the cell, using the selected target intensity source."),
    "Mask integrated density / cell area": ("Target intensity per cell area", "a.u.", "Target pixel sum divided by total cell area."),
    "Fraction of cell intensity in mask": ("Cell intensity inside target", "% or ratio", "Target pixel sum divided by whole-cell pixel sum, both from the selected target intensity source. Undefined when the whole-cell sum is zero; values can fall outside 0 to 1 when intensities are negative."),
}

CELL_SHAPE_FILTERS = ("Area", "Perimeter", "Circularity", "Solidity")
CELL_COMMON_INTENSITY_FILTERS = ("Mean intensity",)
CELL_ADVANCED_INTENSITY_FILTERS = tuple(metric for metric in CELL_FILTER_METRICS if metric not in CELL_SHAPE_FILTERS + CELL_COMMON_INTENSITY_FILTERS)
MASK_COMMON_FILTERS = ("Mask fraction of cell area", "Mean intensity inside mask", "Mask integrated density / cell area")
MASK_ADVANCED_FILTERS = tuple(metric for metric in MASK_FILTER_METRICS if metric not in MASK_COMMON_FILTERS)


# Isolate measurement and filter editing because these controls share one global
# option set while retaining per-target relationships and filter rules.
class CellonautGuiDatasetAnalysisMixin(GuiMixin):
    """Manage analysis relationships derived from dataset definitions."""

    # Labels come from the same metadata used by exports and help text so the
    # measurement panel cannot introduce a second naming scheme.
    def measurement_display_label(self, key: str) -> str:
        return str(MEASUREMENT_LABELS.get(key, key))

    # Build tooltips from structured metadata to keep units attached to the
    # correct measurement when labels are changed.
    def measurement_tooltip(self, key: str) -> str:
        metadata = dict(MEASUREMENT_METADATA.get(key, {}) or {})
        label = self.measurement_display_label(key)
        description = str(metadata.get("description", "") or "").strip()
        unit = str(metadata.get("unit", "") or "").strip()
        parts = [label]
        if unit:
            parts.append(f"Unit: {unit}")
        if description:
            parts.append(description)
        return "\n\n".join(parts)

    # Measurement controls are rebuilt from metadata; delete old widgets first so
    # hidden checkboxes and duplicate signal connections cannot survive a reload.
    def clear_inline_analysis_measurements(self):
        for layout_name in (
            "analysis_basic_measurement_checks_layout",
            "analysis_whole_cell_measurement_checks_layout",
            "analysis_cell_measurement_checks_layout",
        ):
            layout = getattr(self, layout_name, None)
            if layout is None:
                continue
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        self._inline_measurement_checks = []

    # Build each category through one path so every checkbox carries the same key,
    # tooltip, and update signal expected by state collection.
    def add_inline_measurement_group(self, layout_name: str, keys: list[str], options: dict):
        layout = getattr(self, layout_name, None)
        if layout is None:
            return
        for index, key in enumerate(keys):
            cb = QCheckBox(self.measurement_display_label(key))
            cb.setProperty("measurementKey", key)
            cb.setToolTip(self.measurement_tooltip(key))
            cb.setChecked(bool(options.get(key, False)))
            cb.toggled.connect(self.on_inline_measurement_changed)
            layout.addWidget(cb, index // 2, index % 2)
            self._inline_measurement_checks.append(cb)

    # Filter rows are capability-dependent and may change between channels, so
    # destroy the previous widgets rather than leaving disabled stale rules behind.
    def clear_inline_filter_rows(self, layout_name: str):
        layout = getattr(self, layout_name, None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_advanced_filter_group_expanded(self, group, expanded: bool) -> None:
        """Collapse advanced filter controls while keeping the group heading available."""
        for child in group.findChildren(QWidget):
            child.setVisible(bool(expanded))

    # Return references to the generated editors because users need to leave either
    # bound blank, which is simpler to represent in widgets than a fixed data model.
    def build_inline_filter_rows(self, layout_name: str, metrics: list[str], rules: dict) -> dict:
        layout = getattr(self, layout_name, None)
        if layout is None:
            return {}
        self.clear_inline_filter_rows(layout_name)
        title_label = QLabel("Metric")
        title_label.setProperty("muted", "true")
        title_label.setProperty("uiRole", "mutedLabel")
        min_header = QLabel("At least")
        min_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        min_header.setProperty("muted", "true")
        min_header.setProperty("uiRole", "mutedLabel")
        max_header = QLabel("At most")
        max_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        max_header.setProperty("muted", "true")
        max_header.setProperty("uiRole", "mutedLabel")
        layout.addWidget(title_label, 0, 0)
        layout.addWidget(min_header, 0, 1)
        layout.addWidget(max_header, 0, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)

        rows = {}
        for row_idx, metric in enumerate(metrics, start=1):
            rule = rules.get(metric, {}) if isinstance(rules.get(metric, {}), dict) else {}
            display_name, unit, description = FILTER_DISPLAY.get(metric, (metric, "", ""))
            label_text = f"{display_name} ({unit})" if unit else display_name
            metric_label = QLabel(label_text)
            metric_label.setMinimumWidth(0)
            metric_label.setWordWrap(True)
            metric_label.setToolTip(description)
            min_edit = QLineEdit()
            min_edit.setPlaceholderText("blank")
            min_edit.setMinimumWidth(64)
            min_edit.setMaximumWidth(160)
            min_edit.setProperty("filter_label", metric)
            min_edit.setProperty("filter_bound", "min")
            min_edit.setToolTip(f"Include values at or above this {display_name.lower()}. Leave blank for no lower bound.\n\n{description}")
            max_edit = QLineEdit()
            max_edit.setPlaceholderText("blank")
            max_edit.setMinimumWidth(64)
            max_edit.setMaximumWidth(160)
            max_edit.setProperty("filter_label", metric)
            max_edit.setProperty("filter_bound", "max")
            max_edit.setToolTip(f"Include values at or below this {display_name.lower()}. Leave blank for no upper bound.\n\n{description}")
            min_value = rule.get("min")
            max_value = rule.get("max")
            if min_value not in (None, ""):
                min_edit.setText(f"{float(min_value):g}")
            if max_value not in (None, ""):
                max_edit.setText(f"{float(max_value):g}")
            min_edit.textChanged.connect(self.on_inline_filter_text_changed)
            max_edit.textChanged.connect(self.on_inline_filter_text_changed)
            layout.addWidget(metric_label, row_idx, 0)
            layout.addWidget(min_edit, row_idx, 1)
            layout.addWidget(max_edit, row_idx, 2)
            rows[metric] = {"min_edit": min_edit, "max_edit": max_edit}
        return rows

    # Editing stores rules immediately; Update groups owns recalculation so
    # several bounds can be changed without repeatedly redrawing a large image.
    def on_inline_filter_text_changed(self, *_args):
        self.on_inline_analysis_settings_changed(refresh_preview=False)

    # Convert text only when rules are consumed so blank bounds remain meaningful
    # and temporary editing states can produce a precise inline error.
    def rules_from_inline_filter_rows(self, rows: dict) -> dict:
        def parse_bound(metric: str, text: str) -> float:
            percentage = text.endswith("%")
            numeric_text = text[:-1].strip() if percentage else text
            try:
                value = float(numeric_text)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{metric}: enter a number or percentage.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{metric}: enter a finite number.")
            return value / 100.0 if percentage else value

        rules = {}
        for metric, row in rows.items():
            min_text = row["min_edit"].text().strip()
            max_text = row["max_edit"].text().strip()
            rule = {}
            if min_text:
                rule["min"] = parse_bound(metric, min_text)
            if max_text:
                rule["max"] = parse_bound(metric, max_text)
            if "min" in rule and "max" in rule and rule["min"] > rule["max"]:
                raise ValueError(f"{metric}: minimum cannot be greater than maximum.")
            if rule:
                rules[metric] = rule
        return rules

    # Count valid rules rather than non-empty text fields; one metric with two
    # bounds is still one filter from the user's perspective.
    def update_inline_filter_count_label(self, cell_rules: dict | None = None, mask_rules: dict | None = None):
        if not hasattr(self, "analysis_active_filter_count_label"):
            return
        if cell_rules is None:
            try:
                cell_rules = self.rules_from_inline_filter_rows(getattr(self, "_inline_cell_filter_rows", {}) or {})
            except ValueError:
                cell_rules = {}
        if mask_rules is None:
            try:
                mask_rules = self.rules_from_inline_filter_rows(getattr(self, "_inline_mask_filter_rows", {}) or {})
            except ValueError:
                mask_rules = {}
        cell_filter_count = len(cell_rules or {})
        mask_count = len(mask_rules or {})
        total = cell_filter_count + mask_count
        self.analysis_active_filter_count_label.setText(
            f"Active conditions: {total}  |  Cell: {cell_filter_count}  |  Target mask: {mask_count}"
        )

    # Hide categories whose required mask source is unavailable instead of
    # allowing users to configure rules the pipeline cannot evaluate.
    def update_inline_filter_category_visibility(self, has_cell_filters: bool, has_mask_filters: bool):
        roles = [
            str(role or "").strip().lower()
            for role in list(self.preview_state.current_layer_roles or [])
        ]
        has_open_cell_mask = "cellpose_mask" in roles
        source_matches_pipeline = bool(self.preview_state.tools_source_matches_pipeline)
        has_cell_filters = bool(has_cell_filters and has_open_cell_mask and source_matches_pipeline)
        has_mask_filters = bool(has_mask_filters and has_open_cell_mask and source_matches_pipeline)
        has_any_filters = bool(has_cell_filters or has_mask_filters)
        if hasattr(self, "analysis_cell_filter_host"):
            self.analysis_cell_filter_host.setVisible(bool(has_cell_filters))
        if hasattr(self, "analysis_mask_filter_host"):
            self.analysis_mask_filter_host.setVisible(bool(has_mask_filters))
        if hasattr(self, "analysis_mask_intensity_source_combo"):
            self.analysis_mask_intensity_source_combo.setEnabled(bool(has_mask_filters))
        for widget_name in (
            "analysis_filters_title",
            "analysis_filter_source_label",
            "analysis_population_controls",
            "analysis_filter_controls_group",
            "analysis_filter_export_group",
            "analysis_active_filter_count_label",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(has_any_filters)
        if not has_any_filters and hasattr(self, "analysis_filters_hint"):
            self.analysis_filters_hint.setVisible(False)
        if hasattr(self, "analysis_filter_empty_label"):
            self.analysis_filter_empty_label.setVisible(not has_any_filters)
            if not has_any_filters:
                self.analysis_filter_empty_label.setText(
                    "No cell masks are available for this measured channel in both the open image and the current pipeline."
                )

    # The panel title, filter capabilities, and saved rule updates all describe
    # the same channel relationship; derive and render that context in one place.
    def update_inline_filter_source_context(self, image_def: dict, row_idx: int) -> tuple[str, str, str]:
        source_label = str(
            self.preview_state.tools_source_label_override
            or image_def.get("name", "")
            or f"Image {row_idx + 1}"
        )
        selected_cellpose = str(image_def.get("analysis_cellpose_mask_source", "") or "").strip()
        active_definitions = self.get_active_image_definitions()
        try:
            resolver = ArtifactResolver.load(Path(self.preview_state.file_path or ""))
            if resolver is not None:
                record = resolver.record(Path(self.preview_state.file_path or ""))
                mask_definition = image_definition(
                    {"target": record.get("cell_mask", ""), "label": ""}, active_definitions
                )
                selected_cellpose = str((mask_definition or {}).get("name", "") or selected_cellpose).strip()
        except Exception:
            pass
        definitions_by_name = {
            str(item.get("name", "") or "").strip(): item for item in active_definitions
        }
        cellpose_provider = definitions_by_name.get(selected_cellpose, {})
        cell_source_label = str(
            cellpose_provider.get("analysis_cell_segmentation_source", selected_cellpose or source_label)
            or selected_cellpose
            or source_label
        )
        selected_masks = [
            name
            for name, checked in dict(image_def.get("mask_relationships", {}) or {}).items()
            if checked
        ]
        configured_target = str(image_def.get("cell_group_mask_source", "") or "").strip()
        selected_mask_label = (
            configured_target
            if configured_target in selected_masks
            else (selected_masks[0] if selected_masks else "")
        )
        image_def["cell_group_mask_source"] = selected_mask_label
        target_combo = getattr(self, "analysis_filter_target_mask_combo", None)
        if target_combo is not None:
            target_combo.blockSignals(True)
            target_combo.clear()
            if selected_masks:
                for mask_name in selected_masks:
                    target_combo.addItem(mask_name, mask_name)
                target_combo.setCurrentIndex(max(0, target_combo.findData(selected_mask_label)))
                target_combo.setEnabled(True)
            else:
                target_combo.addItem("No measurement mask enabled", "")
                target_combo.setEnabled(False)
            target_combo.blockSignals(False)
        shown_mask_label = selected_mask_label or "(none selected)"
        if hasattr(self, "analysis_filter_source_label"):
            self.analysis_filter_source_label.setText(
                f"Cellpose source: {cell_source_label} · Cell-group target: {shown_mask_label}"
            )
        if hasattr(self, "analysis_mask_filter_host"):
            target_name = "Target mask" if not selected_mask_label else selected_mask_label
            self.analysis_mask_filter_host.setTitle(f"{target_name} within cell")
        self.update_inline_filter_category_visibility(
            bool(selected_cellpose),
            bool(selected_mask_label),
        )
        return source_label, cell_source_label, selected_mask_label

    def set_inline_analysis_panel_visible(self, panel: str) -> None:
        self.analysis_measurements_panel.setVisible(True)
        tabs = getattr(self, "preview_tools_tabs", None)
        if tabs is not None and panel in {"filters", "mask_adjust"}:
            self._preview_tools_switching = True
            tabs.setCurrentWidget(
                self.analysis_filters_panel if panel == "filters" else self.analysis_mask_adjust_panel
            )
            self._preview_tools_switching = False
        elif tabs is None:
            self.analysis_filters_panel.setVisible(panel == "filters")
            if hasattr(self, "analysis_mask_adjust_panel"):
                self.analysis_mask_adjust_panel.setVisible(panel == "mask_adjust")

    def _resolve_inline_analysis_panel(self, row_idx, focus, active_defs):
        focus_text = str(focus).lower()
        panel = "measurements" if focus_text.startswith("measure") else (
            "mask_adjust" if focus_text.startswith("mask") else "filters"
        )
        if panel == "measurements":
            return panel, None, self.analysis_measurements_panel.isVisible()
        if row_idx is None or not 0 <= int(row_idx) < len(active_defs):
            return None

        normalized_row_idx = int(row_idx)
        tabs = getattr(self, "preview_tools_tabs", None)
        target_widget = self.analysis_mask_adjust_panel if panel == "mask_adjust" else self.analysis_filters_panel
        if tabs is None:
            panel_visible = target_widget.isVisible()
        else:
            panel_visible = bool(
                tabs.currentWidget() is target_widget
                and getattr(self, "_active_pipeline_section_index", None)
                == self.pipeline_sections.index(self.preview_tools_section)
            )
        return panel, normalized_row_idx, panel_visible

    def _populate_inline_measurement_settings(self) -> None:
        self.clear_inline_analysis_measurements()
        options = dict(self.measurement_options or {})
        self._inline_measurement_checks = []
        for layout_name, keys in MEASUREMENT_GROUP_KEYS:
            self.add_inline_measurement_group(layout_name, list(keys), options)
        self.analysis_measurements_title.setText("Measurement settings")
        self.analysis_measurements_hint.setText(
            "These choices apply to every measured channel. Cellpose whole-cell measurements require a selected "
            "Cellpose mask; measurements of configured masks within Cellpose cells also require an assigned Weka or "
            "combined mask."
        )
    def _populate_inline_filter_settings(self, image_def, row_idx, active_defs) -> None:
        if hasattr(self, "update_preview_tools_source_options"):
            self.update_preview_tools_source_options(active_defs, row_idx)
        source_label, _cell_source_label, _selected_mask_label = self.update_inline_filter_source_context(
            image_def, row_idx
        )
        self.analysis_filters_title.setText(f"{source_label} cell groups")
        self.analysis_filters_hint.setText("")
        self.analysis_filters_hint.setVisible(False)
        if hasattr(self, "analysis_mask_adjust_title"):
            self.analysis_mask_adjust_title.setText(f"{source_label} mask adjustments")
        if hasattr(self, "analysis_mask_adjust_hint"):
            self.analysis_mask_adjust_hint.setText(
                f"Adjust one mask layer for the open preview of {source_label}. "
                "Use Preview adjustment to compare the temporary result without changing pipeline settings."
            )
        self.populate_analysis_population_combo(image_def)
        self.load_analysis_population_editor(image_def)
        self.update_inline_filter_count_label()

    def _focus_inline_analysis_panel(self, panel: str) -> None:
        if panel == "measurements":
            if self._inline_measurement_checks:
                self._inline_measurement_checks[0].setFocus()
        elif panel == "mask_adjust":
            if hasattr(self, "load_persisted_mask_adjustments"):
                self.load_persisted_mask_adjustments()
            if hasattr(self, "update_mask_adjustment_status"):
                self.update_mask_adjustment_status()
        else:
            first_row = next(iter(getattr(self, "_inline_cell_filter_rows", {}).values()), None)
            if first_row:
                first_row["min_edit"].setFocus()

    # The three editors reuse one area beneath the matrix. Save the previous row
    # first, then rebuild the shared controls for the newly selected owner.
    def show_analysis_settings_panel(self, row_idx: int | None = None, focus: str = "filters"):
        if not hasattr(self, "analysis_measurements_panel") or not hasattr(self, "analysis_filters_panel"):
            return
        self.commit_gui_edits()
        active_defs = self.get_active_image_definitions()

        selection = self._resolve_inline_analysis_panel(row_idx, focus, active_defs)
        if selection is None:
            return
        panel, normalized_row_idx, panel_visible = selection

        if panel == "measurements" and (
            getattr(self, "_inline_analysis_settings_row", None) == normalized_row_idx
            and getattr(self, "_inline_analysis_settings_panel", "") == panel
            and panel_visible
        ):
            self.set_inline_analysis_panel_visible("")
            self._inline_analysis_settings_row = None
            self._inline_analysis_settings_panel = ""
            return

        self._inline_analysis_settings_updating = True
        self._inline_analysis_settings_row = normalized_row_idx
        self._inline_analysis_settings_panel = panel

        self._populate_inline_measurement_settings()

        if panel != "measurements":
            if normalized_row_idx is None:
                return
            self._populate_inline_filter_settings(active_defs[normalized_row_idx], normalized_row_idx, active_defs)

        self.set_inline_analysis_panel_visible(panel)
        self._inline_analysis_settings_updating = False
        if panel in {"filters", "mask_adjust"} and hasattr(self, "preview_tools_section"):
            self.show_left_page(self.pipeline_tab)
            self.show_pipeline_section(self.pipeline_sections.index(self.preview_tools_section))
            self.update_preview_tools_context()
        self._focus_inline_analysis_panel(panel)

    def populate_analysis_population_combo(self, image_def):
        populations = image_def.setdefault("cell_populations", [])
        combo = self.analysis_population_combo
        selected = max(0, int(combo.currentIndex()))
        combo.blockSignals(True)
        combo.clear()
        for population in populations:
            combo.addItem(str(population.get("name", "Cell group") or "Cell group"))
        combo.setCurrentIndex(min(selected, combo.count() - 1))
        combo.blockSignals(False)

    def current_analysis_population_settings(self, image_def):
        index = max(0, int(self.analysis_population_combo.currentIndex()))
        populations = image_def.setdefault("cell_populations", [])
        return populations[index] if index < len(populations) else populations[0]

    def configure_analysis_filter_mode_labels(self, _excluded_group: bool = False) -> None:
        combo = self.analysis_filter_mode_combo
        current_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for stored, label in QC_FILTER_MODE_LABELS.items():
            combo.addItem(label, stored)
        selected = combo.findData(current_data)
        combo.setCurrentIndex(max(0, selected))
        combo.blockSignals(False)

    def load_analysis_population_editor(self, image_def):
        # The combo may already point at the next group when its change signal
        # fires. Retain the owner of the currently rendered controls until they
        # have been committed, just as the inline panel retains its channel row.
        self._inline_analysis_population_index = max(0, self.analysis_population_combo.currentIndex())
        settings = self.current_analysis_population_settings(image_def)
        self.configure_analysis_filter_mode_labels()
        self.analysis_population_name_edit.setText(
            str(settings.get("name", "Cell group") or "Cell group")
        )
        self.analysis_population_name_edit.setEnabled(True)
        self.analysis_remove_population_button.setEnabled(
            len(image_def.get("cell_populations", []) or []) > 1
        )
        self.analysis_exclude_filtered_checkbox.setVisible(True)
        self.analysis_exclude_filtered_checkbox.setChecked(bool(settings.get("exclude_from_csv", False)))
        stored_mode = self.display_filter_mode(
            str(settings.get("qc_filter_mode", "Exclude if any filter fails") or "Exclude if any filter fails")
        )
        mode_index = self.analysis_filter_mode_combo.findData(stored_mode)
        self.analysis_filter_mode_combo.setCurrentIndex(max(0, mode_index))
        intensity_source = str(settings.get("mask_qc_intensity_source", "Measured image") or "Measured image")
        self.analysis_mask_intensity_source_combo.setCurrentText(
            MASK_INTENSITY_SOURCE_LABELS.get(intensity_source, "Measured channel")
        )
        cell_rules = self.parse_cell_qc_limits_text(str(settings.get("cell_qc_limits", "") or ""))
        mask_rules = self.parse_cell_qc_limits_text(str(settings.get("mask_qc_limits", "") or ""))
        self._inline_cell_filter_rows = {}
        for layout_name, metrics in (
            ("analysis_cell_shape_filter_layout", CELL_SHAPE_FILTERS),
            ("analysis_cell_intensity_filter_layout", CELL_COMMON_INTENSITY_FILTERS),
            ("analysis_cell_advanced_filter_layout", CELL_ADVANCED_INTENSITY_FILTERS),
        ):
            self._inline_cell_filter_rows.update(
                self.build_inline_filter_rows(layout_name, list(metrics), cell_rules)
            )
        self._inline_mask_filter_rows = self.build_inline_filter_rows(
            "analysis_mask_filter_layout", list(MASK_COMMON_FILTERS), mask_rules
        )
        self._inline_mask_filter_rows.update(
            self.build_inline_filter_rows(
                "analysis_mask_advanced_filter_layout", list(MASK_ADVANCED_FILTERS), mask_rules
            )
        )
        self.analysis_cell_advanced_filter_group.setChecked(
            any(metric in cell_rules for metric in CELL_ADVANCED_INTENSITY_FILTERS)
        )
        self.analysis_mask_advanced_filter_group.setChecked(
            any(metric in mask_rules for metric in MASK_ADVANCED_FILTERS)
        )
        self.set_advanced_filter_group_expanded(
            self.analysis_cell_advanced_filter_group,
            self.analysis_cell_advanced_filter_group.isChecked(),
        )
        self.set_advanced_filter_group_expanded(
            self.analysis_mask_advanced_filter_group,
            self.analysis_mask_advanced_filter_group.isChecked(),
        )

    def on_analysis_population_changed(self, _index):
        if getattr(self, "_inline_analysis_settings_updating", False):
            return
        self.commit_gui_edits()
        row = getattr(self, "_inline_analysis_settings_row", None)
        defs = self.get_active_image_definitions()
        if row is None or not 0 <= int(row) < len(defs):
            return
        self._inline_analysis_settings_updating = True
        self.load_analysis_population_editor(defs[int(row)])
        self.update_inline_filter_count_label()
        self._inline_analysis_settings_updating = False

    def refresh_analysis_filter_panel_for_open_preview(self, row_idx: int) -> None:
        """Reload the visible filter editor when a newly opened image changes its source."""
        if str(getattr(self, "_inline_analysis_settings_panel", "") or "") != "filters":
            return
        active_defs = self.get_active_image_definitions()
        if not 0 <= int(row_idx) < len(active_defs):
            return

        image_def = active_defs[int(row_idx)]
        self._inline_analysis_settings_updating = True
        self._inline_analysis_settings_row = int(row_idx)
        source_label, _cell_source, _mask_source = self.update_inline_filter_source_context(
            image_def, int(row_idx)
        )
        self.analysis_filters_title.setText(f"{source_label} cell groups")
        self.populate_analysis_population_combo(image_def)
        self.load_analysis_population_editor(image_def)
        self.update_inline_filter_count_label()
        self._inline_analysis_settings_updating = False

    def add_analysis_population(self):
        self.commit_gui_edits()
        row = getattr(self, "_inline_analysis_settings_row", None)
        defs = self.get_active_image_definitions()
        if row is None or not 0 <= int(row) < len(defs):
            return
        image_def = defs[int(row)]
        populations = image_def.setdefault("cell_populations", [])
        palette = ["#00d7ff", "#ffb000", "#bb70ff", "#41d17d", "#ff62b0", "#65a9ff"]
        populations.append({
            "name": f"Cell group {len(populations) + 1}", "color": palette[len(populations) % len(palette)],
            "cell_qc_limits": "", "mask_qc_limits": "", "qc_filter_mode": "Exclude if any filter fails",
            "mask_qc_intensity_source": "Measured image", "exclude_from_csv": False,
        })
        self.image_definitions = defs
        self._inline_analysis_settings_updating = True
        try:
            self.populate_analysis_population_combo(image_def)
            self.analysis_population_combo.setCurrentIndex(len(populations) - 1)
            self.load_analysis_population_editor(image_def)
            self.update_inline_filter_count_label()
        finally:
            self._inline_analysis_settings_updating = False

    def remove_analysis_population(self):
        self.commit_gui_edits()
        index = self.analysis_population_combo.currentIndex()
        row = getattr(self, "_inline_analysis_settings_row", None)
        defs = self.get_active_image_definitions()
        if index < 0 or row is None or not 0 <= int(row) < len(defs):
            return
        populations = defs[int(row)].setdefault("cell_populations", [])
        if len(populations) <= 1:
            return
        if index < len(populations):
            populations.pop(index)
        self.image_definitions = defs
        self._inline_analysis_settings_updating = True
        try:
            self.populate_analysis_population_combo(defs[int(row)])
            self.analysis_population_combo.setCurrentIndex(max(0, index - 1))
            # Rebuilding the combo may already have selected this index with
            # signals blocked. Render explicitly even if setCurrentIndex emits
            # nothing, so the deleted group's controls cannot be committed back.
            self.load_analysis_population_editor(defs[int(row)])
            self.update_inline_filter_count_label()
        finally:
            self._inline_analysis_settings_updating = False

    def on_analysis_population_name_changed(self):
        if getattr(self, "_inline_analysis_settings_updating", False):
            return
        self.commit_gui_edits()
        row = getattr(self, "_inline_analysis_settings_row", None)
        if row is not None and 0 <= int(row) < len(self.image_definitions):
            self.populate_analysis_population_combo(self.image_definitions[int(row)])

    def _read_measurement_edits(self) -> None:
        if getattr(self, "_inline_analysis_settings_updating", False):
            return
        measurement_options = dict(self.measurement_options or {})
        for cb in getattr(self, "_inline_measurement_checks", []) or []:
            key = str(cb.property("measurementKey") or "")
            if key:
                measurement_options[key] = bool(cb.isChecked())
        self.measurement_options = measurement_options

    def on_inline_measurement_changed(self, *_args):
        if getattr(self, "_inline_analysis_settings_updating", False):
            return
        self.commit_gui_edits()
        self.update_analysis_matrix_warning_label()
        if hasattr(self, "refresh_pipeline_section_summaries"):
            self.refresh_pipeline_section_summaries()

    # Save the row-specific filter controls separately from shared measurements.
    def _read_inline_filter_edits(self, active_defs: list[dict]) -> None:
        if getattr(self, "_inline_analysis_settings_updating", False):
            return
        panel = str(getattr(self, "_inline_analysis_settings_panel", "") or "")
        row_idx = getattr(self, "_inline_analysis_settings_row", None)
        if panel != "filters" or row_idx is None:
            return

        if not 0 <= int(row_idx) < len(active_defs):
            return

        image_def = active_defs[int(row_idx)]
        try:
            cell_rules = self.rules_from_inline_filter_rows(getattr(self, "_inline_cell_filter_rows", {}) or {})
            mask_rules = self.rules_from_inline_filter_rows(getattr(self, "_inline_mask_filter_rows", {}) or {})
        except ValueError as exc:
            if hasattr(self, "analysis_filters_hint"):
                self.analysis_filters_hint.setText(str(exc))
                self.analysis_filters_hint.setVisible(True)
            return

        cell_filters = cell_qc_rules_to_text(cell_rules)
        mask_filters = cell_qc_rules_to_text(mask_rules)
        exclude_flagged = bool(self.analysis_exclude_filtered_checkbox.isChecked())
        filter_mode = self.display_filter_mode(
            str(self.analysis_filter_mode_combo.currentData() or self.analysis_filter_mode_combo.currentText())
        )
        intensity_source_display = str(self.analysis_mask_intensity_source_combo.currentText() or "Measured channel")
        mask_intensity_source = (
            "Cell mask image" if intensity_source_display == "Cellpose source channel" else "Measured image"
        )

        index = getattr(self, "_inline_analysis_population_index", max(0, self.analysis_population_combo.currentIndex()))
        settings = image_def["cell_populations"][index]
        if hasattr(self, "analysis_population_name_edit"):
            settings["name"] = self.analysis_population_name_edit.text().strip() or f"Cell group {index + 1}"
        target_combo = getattr(self, "analysis_filter_target_mask_combo", None)
        image_def["cell_group_mask_source"] = (
            str(target_combo.currentData() or "").strip() if target_combo is not None else ""
        )
        settings["cell_qc_limits"] = cell_filters
        settings["mask_qc_limits"] = mask_filters
        settings["exclude_from_csv"] = exclude_flagged
        settings["qc_filter_mode"] = filter_mode
        settings["mask_qc_intensity_source"] = mask_intensity_source

    def on_inline_analysis_settings_changed(self, *_args, refresh_preview: bool = False):
        if getattr(self, "_inline_analysis_settings_updating", False):
            return
        self.commit_gui_edits()
        self.update_inline_filter_count_label()
        self.update_analysis_matrix_warning_label()
        if refresh_preview and hasattr(self, "refresh_preview_filter_overlay"):
            self.refresh_preview_filter_overlay()

    # Export beside the preview's own CSV data so the file always belongs to the
    # run that supplied the visible cells, and number folders to avoid overwrites.
    def export_current_filter_preview_csv(self):
        data = self.preview_state.filter_data
        if data is None:
            QMessageBox.information(
                self,
                "Export filtered CSV for this sample",
                "Open a generated overlay with linked cell measurements first.",
            )
            return

        row_idx = getattr(self, "_inline_analysis_settings_row", None)
        try:
            self.rules_from_inline_filter_rows(getattr(self, "_inline_cell_filter_rows", {}) or {})
            self.rules_from_inline_filter_rows(getattr(self, "_inline_mask_filter_rows", {}) or {})
        except ValueError as exc:
            QMessageBox.critical(self, "Export filtered CSV for this sample", f"Fix the current filter rule first:\n{exc}")
            return
        self.commit_gui_edits()
        active_defs = self.get_active_image_definitions()
        if row_idx is None or not 0 <= int(row_idx) < len(active_defs):
            QMessageBox.information(
                self,
                "Export filtered CSV for this sample",
                "Open Cell Groups for the measured channel you want to export.",
            )
            return

        image_def = active_defs[int(row_idx)]

        try:
            kept, report, excluded, summary = filtered_table_from_current_rules(
                data.table,
                image_def,
                self.parse_cell_qc_limits_text,
                prepare_table=lambda table, settings: self.attach_preview_mask_filter_metrics(table, data.label_image, settings),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export filtered CSV for this sample", str(exc))
            return

        results_root = artifact_root(data.table_path) or data.table_path.parent
        out_dir = next_numbered_child(results_root / "Image Preview Tools" / "Cell Groups", "export")
        out_dir.mkdir(parents=True, exist_ok=True)
        base = data.table_path.stem
        kept_path = out_dir / f"{base}_filtered_current.csv"
        report_path = out_dir / f"{base}_filter_report_current.csv"

        try:
            write_dataframe_csv(drop_derived_ratio_columns(kept), kept_path, index=False)
            write_dataframe_csv(drop_derived_ratio_columns(report), report_path, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export filtered CSV for this sample", f"Could not save CSV files:\n{exc}")
            return

        total = int(summary.get("total", len(data.table)) or len(data.table))
        QMessageBox.information(
            self,
            "Export filtered CSV for this sample",
            "Saved filtered CSV:\n"
            f"{kept_path}\n\n"
            "Saved filter report:\n"
            f"{report_path}\n\n"
            f"Kept {len(kept)} / {total} cells. Excluded {len(excluded)}.",
        )

    # Prefer the open preview's Results folder over the current output field; the
    # user may have browsed to a result from an earlier run or another directory.
    def export_all_filter_preview_csvs(self):
        try:
            self.rules_from_inline_filter_rows(getattr(self, "_inline_cell_filter_rows", {}) or {})
            self.rules_from_inline_filter_rows(getattr(self, "_inline_mask_filter_rows", {}) or {})
            self.commit_gui_edits()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export filtered CSV for all samples in the pipeline",
                f"Could not prepare the current cell-group settings:\n{type(exc).__name__}: {exc}",
            )
            return

        active_defs = self.get_active_image_definitions()
        if not active_defs:
            QMessageBox.information(
                self,
                "Export filtered CSV for all samples in the pipeline",
                "No measured channels are configured.",
            )
            return

        preview_path = str(self.preview_state.file_path or "").strip()
        preview_file = Path(preview_path) if preview_path else None
        output_path = self.output_dir.get().strip()
        if preview_file is not None and preview_file.is_file():
            results_root = artifact_root(preview_file)
            if results_root is None:
                results_root = Path(output_path) if output_path else None
        else:
            results_root = Path(output_path) if output_path else None

        if results_root is None:
            QMessageBox.information(
                self,
                "Export filtered CSV for all samples in the pipeline",
                "Select an output folder or open a generated overlay first.",
            )
            return

        try:
            result = export_all_filtered_result_tables(
                results_root,
                active_defs,
                self.parse_cell_qc_limits_text,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export filtered CSV for all samples in the pipeline", f"Could not export CSVs:\n{exc}")
            return

        detail = ""
        if result.messages:
            preview_messages = "\n".join(f"- {msg}" for msg in result.messages[:8])
            remaining = len(result.messages) - 8
            if remaining > 0:
                preview_messages += f"\n- ... plus {remaining} more"
            detail = f"\n\nSkipped/details:\n{preview_messages}"

        QMessageBox.information(
            self,
            "Export filtered CSV for all samples in the pipeline",
            "Saved filtered CSV exports to:\n"
            f"{result.export_dir}\n\n"
            f"Processed tables: {result.processed_count}\n"
            f"Skipped tables: {result.skipped_count}\n"
            f"Kept cells: {result.kept_count} / {result.total_count}\n\n"
            "Combined exports:\n"
            f"{result.kept_combined_path or '(none)'}\n"
            f"{result.report_combined_path or '(none)'}\n"
            f"{result.group_membership_path or '(no cell groups)'}\n\n"
            "Regenerated measurements:\n"
            f"{result.all_measurements_path or '(none)'}\n"
            f"{result.readable_measurements_path or '(none)'}\n"
            f"{detail}",
        )
