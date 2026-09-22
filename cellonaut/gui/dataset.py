"""Dataset and analysis-target controls for the main pipeline tab.

This mixin renders channel rows, Cellpose settings and measurement relationships.
Editor values are captured through commit_gui_edits into editable configuration;
Cell Group settings remain in the later Preview/export workflow.
"""

from __future__ import annotations

from cellonaut.runtime import PipelineStage, StageUpdate
from cellonaut.gui.help_content import CELLPOSE_DIAMETER_HELP

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)
from cellonaut.config.defaults import (
    CELLPOSE_MODEL_OPTIONS,
    DEFAULT_CELL_DIAMETER,
    DEFAULT_CELLPOSE_MODEL_TYPE,
    DEFAULT_CELL_MIN_SIZE,
    DEFAULT_CELLPROB_THRESHOLD,
    DEFAULT_FLOW_THRESHOLD,
)
from cellonaut.config.relationships import image_display_names, image_produces_mask
from cellonaut.gui.dataset_analysis import CellonautGuiDatasetAnalysisMixin
from cellonaut.gui.dataset_images import CellonautGuiDatasetImagesMixin
from cellonaut.gui.processing_settings import CellonautGuiProcessingSettingsMixin
from cellonaut.gui.widgets import FormSection, MatrixToggleButton, NoWheelComboBox, fit_table_height


# Coordinate dynamic dataset tables in one owner because channel and mask row
# changes must update processing, Cellpose, and measurement views together.
class CellonautGuiDatasetMixin(
    CellonautGuiProcessingSettingsMixin,
    CellonautGuiDatasetAnalysisMixin,
    CellonautGuiDatasetImagesMixin,
):
    def make_cellpose_custom_model_cell(self, value: object) -> QWidget:
        cell = QWidget()
        cell.setProperty("uiRole", "matrixActionCell")
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        edit = QLineEdit(str(value or ""))
        edit.setObjectName("CellposeCustomModelPath")
        edit.setPlaceholderText("...")
        edit.setToolTip("Optional compatible Cellpose 4 model file. If selected, it overrides the built-in model.")

        browse_button = QPushButton()
        browse_button.setObjectName("CellposeCustomModelBrowse")
        if hasattr(self, "shell_icon"):
            browse_button.setIcon(self.shell_icon("folder-open"))
        browse_button.setFixedSize(28, 28)
        browse_button.setToolTip("Browse for a custom Cellpose model")
        browse_button.setAccessibleName("Browse for custom Cellpose model")
        browse_button.clicked.connect(lambda _checked=False, target=edit: self.browse_cellpose_custom_model(target))

        clear_button = QPushButton()
        clear_button.setObjectName("CellposeCustomModelClear")
        if hasattr(self, "shell_icon"):
            clear_button.setIcon(self.shell_icon("trash-2"))
        clear_button.setFixedSize(28, 28)
        clear_button.setToolTip("Clear the custom model and use the selected built-in model")
        clear_button.setAccessibleName("Clear custom Cellpose model")
        if hasattr(self, "set_button_role"):
            self.set_button_role(clear_button, "danger")
        clear_button.setEnabled(bool(edit.text().strip()))
        clear_button.clicked.connect(edit.clear)
        edit.textChanged.connect(lambda text, button=clear_button: button.setEnabled(bool(text.strip())))
        edit.textChanged.connect(self.on_cellpose_settings_table_changed)

        layout.addWidget(edit, 1)
        layout.addWidget(browse_button)
        layout.addWidget(clear_button)
        return cell

    def browse_cellpose_custom_model(self, edit: QLineEdit) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select custom Cellpose model",
            edit.text().strip(),
            "All files (*.*)",
        )
        if path:
            edit.setText(path)

    # Muting must reach wrapper widgets as well as their controls because matrix
    # cells use both layers for borders and background styling.
    def set_widget_muted(self, widget, muted: bool):
        if widget is None:
            return
        widgets = [widget, *widget.findChildren(QWidget)]
        for current_widget in widgets:
            current_widget.setProperty("mutedState", "true" if muted else "false")
            current_widget.style().unpolish(current_widget)
            current_widget.style().polish(current_widget)
            current_widget.update()

    # Matrix rows and columns must classify definitions identically in every
    # rebuild and save path or toggles can be written back to the wrong channel.
    def analysis_matrix_definition_rows(self, definitions: list[dict]):
        source_rows = [
            (index, image_def)
            for index, image_def in enumerate(definitions)
            if self.is_physical_channel_definition(image_def)
        ]
        target_rows = [
            (index, image_def) for index, image_def in enumerate(definitions) if self.is_mask_only_definition(image_def)
        ]
        return source_rows, target_rows

    # Cellpose settings belong only to physical channels. Rebuild from definitions
    # with signals blocked so table construction is not mistaken for user editing.
    def rebuild_cellpose_settings_table(self):
        if not hasattr(self, "cellpose_settings_table"):
            return

        active_defs = self.get_active_image_definitions()
        source_rows, _target_rows = self.analysis_matrix_definition_rows(active_defs)
        names = [image_def["name"].strip() or f"Channel {source_index + 1}" for source_index, image_def in source_rows]
        table = self.cellpose_settings_table
        table.blockSignals(True)
        table.clear()

        headers = [
            "Enable",
            "Remove border",
            "Cellpose source channel",
            "Cellpose model",
            "Custom model",
            "Diameter (px)",
            "Minimum cell area (px²)",
            "Probability threshold",
            "Flow threshold",
        ]

        table.setRowCount(len(source_rows))
        table.setColumnCount(len(headers))
        table.setVerticalHeaderLabels(names)
        table.setHorizontalHeaderLabels(headers)
        header_help = [
            "Create one reusable Cellpose mask for this channel. It becomes a Channel_Cellpose column in Measurements.",
            "Leave out partial cells that touch the image boundary.",
            "Image channel Cellpose uses to find cell boundaries; it may differ from the measured row.",
            "Built-in Cellpose model used for segmentation.",
            "Optional compatible Cellpose 4 model file; this overrides the built-in model.",
            "Cell diameter in pixels used for rescaling. Blank keeps the original image scale.",
            "Discard detected cells smaller than this area in pixels squared.",
            "Cellpose pixel-inclusion score. Lower finds more or larger cells; higher is more selective.",
            "Cellpose flow-consistency check. Lower positive values reject more shapes; 0 disables the check.",
        ]
        for column, (_header_text, tooltip) in enumerate(zip(headers, header_help, strict=True)):
            header_item = table.horizontalHeaderItem(column)
            if header_item is not None:
                header_item.setToolTip(tooltip)
        minimum_widths = [68, 96, 136, 112, 140, 82, 126, 126, 104]
        header_metrics = table.horizontalHeader().fontMetrics()
        for column, (header_text, minimum_width) in enumerate(zip(headers, minimum_widths, strict=True)):
            table.setColumnWidth(column, max(minimum_width, header_metrics.horizontalAdvance(header_text) + 28))

        for row_idx, (_source_index, image_def) in enumerate(source_rows):
            is_enabled = bool(image_def.get("analysis_cell_segmentation_enabled", False))
            enable_button = self.make_processing_enabled_button(
                is_enabled,
                "Create a reusable Cellpose whole-cell mask owned by this channel.",
            )
            enable_button.toggled.connect(self.on_cellpose_settings_table_changed)
            enable_widget = QWidget()
            enable_widget.setProperty("uiRole", "matrixActionCell")
            enable_layout = QHBoxLayout(enable_widget)
            enable_layout.setContentsMargins(8, 2, 8, 2)
            enable_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            enable_layout.addWidget(enable_button)
            table.setCellWidget(row_idx, 0, enable_widget)
            remove_border_cb = MatrixToggleButton(bool(image_def.get("cell_remove_border", True)))
            remove_border_cb.setMinimumWidth(58)
            remove_border_cb.setToolTip("Remove cells touching the image border when ON.")
            remove_border_cb.clicked.connect(self.on_cellpose_settings_table_changed)
            table.setCellWidget(row_idx, 1, remove_border_cb)

            seg_source_combo = NoWheelComboBox()
            seg_source_combo.addItems(names)
            seg_source_value = str(image_def.get("analysis_cell_segmentation_source", names[row_idx]) or names[row_idx])
            idx = seg_source_combo.findText(seg_source_value)
            seg_source_combo.setCurrentIndex(idx if idx >= 0 else row_idx)
            seg_source_combo.setToolTip(
                "Channel used to create the Cellpose cell mask.\n\n"
                "Cell filter values such as area, mean intensity, and raw integrated density are measured on this channel, "
                "even when the measured channel is different."
            )
            seg_source_combo.currentTextChanged.connect(self.on_cellpose_settings_table_changed)
            table.setCellWidget(row_idx, 2, seg_source_combo)

            model_combo = NoWheelComboBox()
            model_combo.addItems(CELLPOSE_MODEL_OPTIONS)
            model_value = str(
                image_def.get("cellpose_model_type", DEFAULT_CELLPOSE_MODEL_TYPE) or DEFAULT_CELLPOSE_MODEL_TYPE
            )
            idx = model_combo.findText(model_value)
            model_combo.setCurrentIndex(idx if idx >= 0 else 0)
            model_combo.setToolTip(
                "Built-in Cellpose 4 model. cpsam is the default; cpsam_v2 can improve "
                "segmentation on low-contrast images."
            )
            model_combo.currentTextChanged.connect(self.on_cellpose_settings_table_changed)
            table.setCellWidget(row_idx, 3, model_combo)

            table.setCellWidget(
                row_idx,
                4,
                self.make_cellpose_custom_model_cell(image_def.get("cellpose_custom_model_path", "")),
            )

            diameter_value = image_def.get("cell_diameter", DEFAULT_CELL_DIAMETER)
            diameter_edit = QLineEdit("" if diameter_value is None else str(diameter_value))
            diameter_edit.setPlaceholderText("")
            diameter_edit.setToolTip(CELLPOSE_DIAMETER_HELP)
            diameter_edit.textChanged.connect(self.on_cellpose_settings_table_changed)
            table.setCellWidget(row_idx, 5, diameter_edit)

            min_size_edit = QLineEdit(
                str(image_def.get("cell_min_size", DEFAULT_CELL_MIN_SIZE) or DEFAULT_CELL_MIN_SIZE)
            )
            min_size_edit.setToolTip("Smallest accepted cell area. Cellpose uses 15 px² by default.")
            min_size_edit.textChanged.connect(self.on_cellpose_settings_table_changed)
            table.setCellWidget(row_idx, 6, min_size_edit)

            cellprob_edit = QLineEdit(
                str(image_def.get("cellprob_threshold", DEFAULT_CELLPROB_THRESHOLD) or DEFAULT_CELLPROB_THRESHOLD)
            )
            cellprob_edit.setToolTip(
                "Lower to include more pixels and find more or larger cells. Higher is more selective."
            )
            cellprob_edit.textChanged.connect(self.on_cellpose_settings_table_changed)
            table.setCellWidget(row_idx, 7, cellprob_edit)

            flow_edit = QLineEdit(
                str(image_def.get("flow_threshold", DEFAULT_FLOW_THRESHOLD) or DEFAULT_FLOW_THRESHOLD)
            )
            flow_edit.setToolTip(
                "Lower positive values reject more inconsistent cell shapes. Higher keeps more; 0 disables this check."
            )
            flow_edit.textChanged.connect(self.on_cellpose_settings_table_changed)
            table.setCellWidget(row_idx, 8, flow_edit)
            self.set_cellpose_row_active_state(table, row_idx, is_enabled)

        for column in range(len(headers)):
            table.setColumnHidden(column, False)
        table.resizeRowsToContents()
        for row_idx in range(table.rowCount()):
            table.setRowHeight(row_idx, 44)
        fit_table_height(table, minimum=112, maximum=256)

        table.blockSignals(False)
        self.validate_cellpose_settings_table()

    # Save table edits into the shared definitions before rebuilding another view.
    def _read_cellpose_table_edits(self, active_defs: list[dict]) -> None:
        if self._rebuilding_image_tabs:
            return
        if not hasattr(self, "cellpose_settings_table"):
            return

        source_rows, _target_rows = self.analysis_matrix_definition_rows(active_defs)
        table = self.cellpose_settings_table

        for row_idx, (_source_index, image_def) in enumerate(source_rows):
            if row_idx >= table.rowCount():
                continue
            enable_widget = table.cellWidget(row_idx, 0)
            enable_button = enable_widget.findChild(QPushButton) if enable_widget is not None else None
            enable_cb = enable_widget.findChild(QCheckBox) if enable_widget is not None else None
            was_enabled = bool(image_def.get("analysis_cell_segmentation_enabled", False))
            image_def["analysis_cell_segmentation_enabled"] = (
                bool(enable_button.isChecked())
                if enable_button is not None
                else bool(enable_cb.isChecked())
                if enable_cb is not None
                else bool(image_def.get("analysis_cell_segmentation_enabled", False))
            )
            if (
                image_def["analysis_cell_segmentation_enabled"]
                and not was_enabled
                and not list(image_def.get("analysis_cellpose_mask_sources", []) or [])
                and not str(image_def.get("analysis_cellpose_mask_source", "") or "").strip()
            ):
                image_def["analysis_cellpose_mask_source"] = str(image_def.get("name", "") or "").strip()
                image_def["analysis_cellpose_mask_sources"] = [image_def["analysis_cellpose_mask_source"]]

            rb_widget = table.cellWidget(row_idx, 1)
            rb_cb = (
                rb_widget
                if isinstance(rb_widget, MatrixToggleButton)
                else rb_widget.findChild(QPushButton)
                if rb_widget is not None
                else None
            )
            if rb_cb is not None:
                image_def["cell_remove_border"] = bool(rb_cb.isChecked())

            seg_source_combo = table.cellWidget(row_idx, 2)
            image_def["analysis_cell_segmentation_source"] = (
                seg_source_combo.currentText()
                if seg_source_combo is not None
                else str(image_def.get("analysis_cell_segmentation_source", ""))
            )

            model_combo = table.cellWidget(row_idx, 3)
            image_def["cellpose_model_type"] = (
                model_combo.currentText()
                if model_combo is not None
                else str(image_def.get("cellpose_model_type", DEFAULT_CELLPOSE_MODEL_TYPE))
            )

            custom_model_cell = table.cellWidget(row_idx, 4)
            custom_model_edit = custom_model_cell.findChild(QLineEdit) if custom_model_cell is not None else None
            image_def["cellpose_custom_model_path"] = (
                custom_model_edit.text().strip()
                if custom_model_edit is not None
                else str(image_def.get("cellpose_custom_model_path", ""))
            )

            diameter_edit = table.cellWidget(row_idx, 5)
            image_def["cell_diameter"] = (
                diameter_edit.text().strip()
                if diameter_edit is not None
                else str(image_def.get("cell_diameter", DEFAULT_CELL_DIAMETER))
            )

            min_size_edit = table.cellWidget(row_idx, 6)
            image_def["cell_min_size"] = (
                min_size_edit.text().strip()
                if min_size_edit is not None
                else str(image_def.get("cell_min_size", DEFAULT_CELL_MIN_SIZE))
            )

            cellprob_edit = table.cellWidget(row_idx, 7)
            image_def["cellprob_threshold"] = (
                cellprob_edit.text().strip()
                if cellprob_edit is not None
                else str(image_def.get("cellprob_threshold", DEFAULT_CELLPROB_THRESHOLD))
            )

            flow_edit = table.cellWidget(row_idx, 8)
            image_def["flow_threshold"] = (
                flow_edit.text().strip()
                if flow_edit is not None
                else str(image_def.get("flow_threshold", DEFAULT_FLOW_THRESHOLD))
            )

    # One change affects row styling, validation, and available analysis actions;
    # refresh them together after committing the edited definitions.
    def on_cellpose_settings_table_changed(self, *_args):
        if self._rebuilding_image_tabs:
            return
        self.commit_gui_edits()
        self.refresh_cellpose_settings_table_active_states()
        self.refresh_channel_name_dependent_ui()
        self.refresh_analysis_matrix_from_definitions()
        self.validate_cellpose_settings_table()

    def on_analysis_matrix_cellpose_changed(self, row: int, column: int, checked: bool) -> None:
        if self._rebuilding_image_tabs:
            return
        self.on_analysis_matrix_changed()

    # The matrix is derived rather than incrementally patched because channel and
    # mask additions can change both its row and column ownership at once.
    def build_analysis_matrix_for_current_source(self):
        if not hasattr(self, "analysis_matrix_table"):
            return
        if self._rebuilding_image_tabs:
            return

        active_defs = self.get_active_image_definitions()
        names = [img["name"].strip() or f"Channel {i+1}" for i, img in enumerate(active_defs)]
        source_rows, target_rows = self.analysis_matrix_definition_rows(active_defs)
        target_names = [names[index] for index, _image_def in target_rows]
        source_names = [names[index] for index, _image_def in source_rows]
        cellpose_rows = [
            (index, image_def)
            for index, image_def in source_rows
            if bool(image_def.get("analysis_cell_segmentation_enabled", False))
        ]
        cellpose_names = [names[index] for index, _image_def in cellpose_rows]

        self._analysis_relationship_column_count = len(target_rows)
        self._analysis_cellpose_columns = {
            len(target_rows) + offset: name for offset, name in enumerate(cellpose_names)
        }

        table = self.analysis_matrix_table
        table.blockSignals(True)
        table.clear()
        if not getattr(self, "_analysis_matrix_cell_click_connected", False):
            table.cellClicked.connect(self.on_analysis_matrix_cell_clicked)
            self._analysis_matrix_cell_click_connected = True

        table.setRowCount(len(source_rows))
        table.setColumnCount(len(target_rows) + len(cellpose_rows))
        table.setHorizontalHeaderLabels([*target_names, *(f"{name}_Cellpose" for name in cellpose_names)])
        table.setVerticalHeaderLabels(source_names)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.horizontalHeader().setMinimumSectionSize(88)

        for row_idx, (_source_index, source_def) in enumerate(source_rows):
            row_has_mask = self.analysis_row_has_active_mask(source_def)
            relationships = dict(source_def.get("mask_relationships", {}))

            for col_offset, (target_index, target_def) in enumerate(target_rows):
                target_name = names[target_index]
                has_classifier = image_produces_mask(target_def)

                is_on = bool(has_classifier and relationships.get(target_name, False))

                toggle = MatrixToggleButton(is_on)
                toggle.setEnabled(has_classifier)
                self.set_widget_muted(toggle, not has_classifier or not row_has_mask)
                toggle.clicked.connect(self.on_analysis_matrix_changed)

                if not has_classifier:
                    toggle.set_unavailable()
                    toggle.setToolTip(
                        f"{target_name} does not create a mask yet. Complete its mask source in Masks first."
                    )
                else:
                    source_name = str(source_def.get("name", "") or f"Channel {row_idx + 1}")
                    toggle.setToolTip(f"Turn ON to measure {source_name} inside {target_name}.")

                table.setCellWidget(row_idx, col_offset, toggle)

            selected_cellpose = set(source_def.get("analysis_cellpose_mask_sources", []) or [])
            if not selected_cellpose:
                legacy_source = str(source_def.get("analysis_cellpose_mask_source", "") or "").strip()
                selected_cellpose = {legacy_source} if legacy_source else set()
            for offset, cellpose_name in enumerate(cellpose_names):
                column = len(target_rows) + offset
                cellpose_toggle = MatrixToggleButton(cellpose_name in selected_cellpose)
                cellpose_toggle.setToolTip(
                    f"Turn ON to measure {source_names[row_idx]} inside the reusable "
                    f"{cellpose_name}_Cellpose mask. Any number of Cellpose masks can be selected for this channel."
                )
                cellpose_toggle.clicked.connect(
                    lambda checked=False, row=row_idx, col=column: self.on_analysis_matrix_cellpose_changed(
                        row, col, checked
                    )
                )
                table.setCellWidget(row_idx, column, cellpose_toggle)

        for column, cellpose_name in self._analysis_cellpose_columns.items():
            cellpose_header = table.horizontalHeaderItem(column)
            if cellpose_header is not None:
                cellpose_header.setToolTip(
                    f"Cellpose mask configured on {cellpose_name}. It is generated once per sample and can measure any channel row."
                )

        for column in range(table.columnCount()):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        table.resizeRowsToContents()
        for row_idx in range(table.rowCount()):
            table.setRowHeight(row_idx, 40)
        fit_table_height(
            table,
            minimum=112,
            maximum=10_000,
            reserve_horizontal_scrollbar=False,
        )
        table.blockSignals(False)
        self.update_analysis_matrix_warning_label()

    # One relationship drives both overlay display and mask-based measurement,
    # matching the single choice presented by the matrix.
    def _read_analysis_matrix_edits(self, active_defs: list[dict]) -> None:
        if self._rebuilding_image_tabs:
            return
        if not hasattr(self, "analysis_matrix_table"):
            return

        names = image_display_names(active_defs)
        source_rows, target_rows = self.analysis_matrix_definition_rows(active_defs)

        table = self.analysis_matrix_table
        row_count = table.rowCount()
        relationship_count = len(target_rows)

        for row_idx, (_source_index, source_def) in enumerate(source_rows):
            if row_idx >= row_count:
                continue
            relationships = dict(source_def.get("mask_relationships", {}))

            for col_offset, (target_index, _target_def) in enumerate(target_rows):
                target_name = names[target_index]
                toggle = (
                    table.cellWidget(row_idx, col_offset)
                    if row_idx < row_count and col_offset < relationship_count
                    else None
                )

                if isinstance(toggle, QPushButton):
                    relationships[target_name] = bool(toggle.isChecked())

            source_def["mask_relationships"] = relationships
            selected_cellpose = [
                cellpose_name
                for column, cellpose_name in getattr(self, "_analysis_cellpose_columns", {}).items()
                if isinstance(table.cellWidget(row_idx, column), QPushButton)
                and table.cellWidget(row_idx, column).isChecked()
            ]
            represented_cellpose_names = set(getattr(self, "_analysis_cellpose_columns", {}).values())
            previous_cellpose = set(source_def.get("analysis_cellpose_mask_sources", []) or [])
            legacy_source = str(source_def.get("analysis_cellpose_mask_source", "") or "").strip()
            if legacy_source:
                previous_cellpose.add(legacy_source)
            if selected_cellpose or previous_cellpose.intersection(represented_cellpose_names):
                source_def["analysis_cellpose_mask_sources"] = selected_cellpose
                source_def["analysis_cellpose_mask_source"] = selected_cellpose[0] if selected_cellpose else ""

    # Rebuild from committed state; callers capture pending edits explicitly.
    def refresh_analysis_matrix_from_definitions(self):
        if self._rebuilding_image_tabs:
            return
        if not hasattr(self, "analysis_matrix_table"):
            return

        self.build_analysis_matrix_for_current_source()

    # Matrix toggles immediately affect whether row-level filter and adjustment
    # actions are valid, so update those states without rebuilding the whole table.
    def on_analysis_matrix_changed(self, *_args):
        if self._rebuilding_image_tabs:
            return
        self.commit_gui_edits()
        self.refresh_analysis_matrix_row_states()
        self.update_analysis_matrix_warning_label()

    # A row without an active mask cannot offer mask-dependent actions even when
    # mask definitions exist elsewhere in the matrix.
    def refresh_analysis_matrix_row_states(self):
        if not hasattr(self, "analysis_matrix_table"):
            return

        active_defs = self.get_active_image_definitions()
        source_defs = [image_def for image_def in active_defs if self.is_physical_channel_definition(image_def)]
        table = self.analysis_matrix_table
        for row_idx, source_def in enumerate(source_defs):
            row_has_mask = self.analysis_row_has_active_mask(source_def)
            for col_idx in range(table.columnCount()):
                widget = table.cellWidget(row_idx, col_idx)
                if isinstance(widget, MatrixToggleButton):
                    self.set_widget_muted(
                        widget,
                        not widget.isEnabled() or not row_has_mask,
                    )

    # Handle clicks on empty table-cell space as well as child buttons so the
    # compact matrix does not require pixel-perfect clicks.
    def on_analysis_matrix_cell_clicked(self, row: int, col: int):
        if self._rebuilding_image_tabs:
            return
        if not hasattr(self, "analysis_matrix_table"):
            return

        if col >= self.analysis_matrix_table.columnCount():
            return

        toggle = self.analysis_matrix_table.cellWidget(row, col)
        if not isinstance(toggle, MatrixToggleButton) or not toggle.isEnabled():
            return

        toggle.setChecked(not toggle.isChecked())
        if col in getattr(self, "_analysis_cellpose_columns", {}):
            self.on_analysis_matrix_cellpose_changed(row, col, toggle.isChecked())
        else:
            self.on_analysis_matrix_changed()

    # These controls sit inside a scrolling page; requiring focus prevents a
    # stray wheel event from silently changing scientific settings.
    def apply_no_wheel_policy(self, root: QWidget):
        if root is None:
            return

        for combo in root.findChildren(QComboBox):
            combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        for spin in root.findChildren(QSpinBox):
            spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        for dspin in root.findChildren(QDoubleSpinBox):
            dspin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # Channel and mask tabs share the same final + behavior. Keeping the guard and
    # re-entry handling together prevents one tab group from adding duplicate rows.
    def _handle_definition_tab_changed(self, tabs, index: int, add_row):
        if self._rebuilding_image_tabs or getattr(self, "_adding_image_tab", False):
            return
        if tabs is None or tabs.count() <= 0:
            return

        plus_index = tabs.count() - 1
        if index == plus_index:
            self._adding_image_tab = True
            try:
                add_row()
            finally:
                self._adding_image_tab = False
            return

        self.build_analysis_matrix_for_current_source()

    def on_image_tab_changed(self, index: int):
        self._handle_definition_tab_changed(getattr(self, "image_tabs", None), index, self.add_image_row)

    def on_mask_tab_changed(self, index: int):
        self._handle_definition_tab_changed(getattr(self, "mask_tabs", None), index, self.add_mask_row)

    def on_mask_source_tab_changed(self, index: int):
        """Show only masks belonging to the selected image, or global combined masks."""
        if self._rebuilding_image_tabs:
            return
        tabs = getattr(self, "mask_source_tabs", None)
        if tabs is None or index < 0:
            return
        self._selected_mask_source_name = tabs.tabText(index)
        self.rebuild_image_rows(
            rebuild_mask_source_tabs=False,
            rebuild_channel_tabs=False,
            refresh_dependent_views=False,
        )

    def on_mask_tab_clicked(self, index: int):
        """Let the sole mask + tab add a row even though its index cannot change."""
        tabs = getattr(self, "mask_tabs", None)
        if tabs is not None and tabs.count() == 1 and index == 0:
            self._handle_definition_tab_changed(tabs, index, self.add_mask_row)

    # Tabs mirror editable display names; update both channel and mask views when
    # a definition can appear in either collection.
    def update_image_tab_title(self, index: int):
        if not hasattr(self, "image_tabs") and not hasattr(self, "mask_tabs"):
            return
        if index < 0 or index >= len(getattr(self, "image_rows", [])):
            return

        row = self.image_rows[index]
        if row is None or row.name is None:
            return
        title = row.name.get().strip() or row.default_name
        channel_tab_index = row.channel_tab_index
        if hasattr(self, "image_tabs") and 0 <= channel_tab_index < self.image_tabs.count() - 1:
            self.image_tabs.setTabText(channel_tab_index, title)
        mask_tab_index = row.mask_tab_index
        if hasattr(self, "mask_tabs") and 0 <= mask_tab_index < self.mask_tabs.count() - 1:
            self.mask_tabs.setTabText(mask_tab_index, title)

    # Refresh the button style because Qt does not automatically apply property changes.
    def set_button_role(self, button: QPushButton, role: str):
        button.setProperty("role", role)
        button.style().unpolish(button)
        button.style().polish(button)

    # Use consistent colors for similar statuses across tasks.
    def set_status_style(self, status: str | StageUpdate):
        if isinstance(status, dict):
            # Worker state is explicit; local notices still use presentation styling.
            roles: dict[str, str] = {PipelineStage.SUCCESS: "done", PipelineStage.PARTIAL: "warning",
                     PipelineStage.ERROR: "error", PipelineStage.CANCELLED: "idle"}
            role = roles.get(status["stage"], "running")
        else:
            role = self.status_text_role(status)
        self.apply_status_role(role)

    @staticmethod
    def status_text_role(status: str) -> str:
        status_l = (status or "").lower()
        if "completed with errors" in status_l or "warning" in status_l:
            role = "warning"
        elif "error" in status_l:
            role = "error"
        elif "done" in status_l or "ready" in status_l:
            role = "done"
        elif any(
            word in status_l
            for word in [
                "running",
                "importing",
                "generating",
                "cancelling",
                "initializing",
                "checking",
                "copying",
                "scanning",
            ]
        ):
            role = "running"
        else:
            role = "idle"

        return role

    def apply_status_role(self, role: str):
        self.status_label.setProperty("status", role)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        if hasattr(self, "pipeline_spinner_label"):
            self.pipeline_spinner_label.setProperty("status", role)
            self.pipeline_spinner_label.setToolTip(
                {
                    "idle": "Application is idle",
                    "running": "A task is running",
                    "done": "The last task completed successfully",
                    "warning": "The last task completed with warnings",
                    "error": "The last task failed",
                }[role]
            )
            self.pipeline_spinner_label.style().unpolish(self.pipeline_spinner_label)
            self.pipeline_spinner_label.style().polish(self.pipeline_spinner_label)

    def set_pipeline_busy(self, busy: bool):
        if not hasattr(self, "pipeline_spinner_label"):
            return

        self.pipeline_spinner_label.setText("●")
        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(bool(busy))

    def make_section(self, title: str, rows=None) -> FormSection:
        section = FormSection(title)
        if rows:
            section.add_rows(rows)
        return section
