"""Channel and mask tab construction and synchronization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellonaut.config.defaults import (
    DEFAULT_COMBINED_MASK_OPERATION,
    DEFAULT_MASK_SOURCE_MODE,
    DEFAULT_THRESHOLD_METHOD,
    MASK_SOURCE_MODE_COMBINED,
    MASK_SOURCE_MODE_WEKA,
    THRESHOLD_METHOD_OPTIONS,
    default_image_definition,
)
from cellonaut.config.relationships import (
    configured_mask_names,
    image_display_names,
    image_produces_mask,
    image_uses_combined_mask,
    normalize_analysis_relationships,
    remap_analysis_references,
)
from cellonaut.config.state import ImageGuiState
from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.state import ImageDefinitionRowWidgets
from cellonaut.gui.widgets import CollapsibleSection, ComboRow, EntryRow, PathRow
from cellonaut.io.imagej_runtime import inspect_weka_classifier


class CellonautGuiDatasetImagesMixin(GuiMixin):
    """Rebuild channel and mask editors from canonical GUI state."""

    # A pipeline always needs pixel data even when every configured analysis uses masks.
    MINIMUM_CHANNEL_COUNT = 1

    # Sections are assembled in multiple builder modules; this keeps expansion
    # behavior independent of where the content widget was created.
    def _build_collapsible_section(
        self,
        title: str,
        widget: QWidget,
        expanded: bool = False,
        *,
        summary: str = "",
        step_number: int | None = None,
    ) -> CollapsibleSection:
        section = CollapsibleSection(
            title,
            expanded=expanded,
            summary=summary,
            step_number=step_number,
        )
        if isinstance(widget, QGroupBox):
            widget.setTitle("")
            widget.setProperty("embeddedPipelineSection", "true")
        section.addWidget(widget)
        return section

    # Rebuilding source lists should preserve valid selections without emitting
    # intermediate selection changes that rewrite definitions.
    def _set_listwidget_items_with_selection(
        self,
        widget: QListWidget,
        items: list[str],
        selected: set[str] | None = None,
        select_all_if_empty: bool = False,
    ):
        selected = selected or set()

        widget.blockSignals(True)
        widget.clear()

        for text in items:
            item = QListWidgetItem(text)
            widget.addItem(item)
            item.setSelected(text in selected or (select_all_if_empty and not selected))

        widget.blockSignals(False)

    # Expose default construction through the GUI mixin so TIFF and ND2 discovery
    # can create rows without depending directly on configuration internals.
    def create_default_image_definition(self, index: int) -> dict:
        return default_image_definition(index)

    # Combined masks occupy mask tabs even though they are represented by the same
    # definition structure as channels and Weka masks.
    def is_combined_mask_definition(self, image_def: dict) -> bool:
        return image_uses_combined_mask(image_def)

    # Treat combined masks as mask-only regardless of older flag combinations so
    # they never appear as measurable source channels.
    def is_mask_only_definition(self, image_def: dict) -> bool:
        return bool(image_def.get("is_mask_only", False)) or self.is_combined_mask_definition(image_def)

    # Physical channels are the complement of mask-only definitions and are the
    # only rows that may own measurements or Cellpose settings.
    def is_physical_channel_definition(self, image_def: dict) -> bool:
        return not self.is_mask_only_definition(image_def)

    # Removal limits apply to physical inputs rather than the total number of
    # channel and mask definitions shown in both tab groups.
    def physical_channel_count(self) -> int:
        return sum(self.is_physical_channel_definition(image_def) for image_def in self.image_definitions)

    # Mask-only rows may reference only physical image data, so derive assignment
    # choices through the same classification used by the analysis matrix.
    def get_channel_assignment_options(self, definitions: list[dict] | None = None) -> list[str]:
        definitions = definitions if definitions is not None else self.image_definitions
        return [
            str(image_def.get("name", "") or f"Channel {index + 1}").strip()
            for index, image_def in enumerate(definitions)
            if self.is_physical_channel_definition(image_def)
        ]

    # A mask-only row reads pixels from its assigned channel. Copy file and stack
    # metadata so later processing does not need to resolve the GUI relationship.
    def apply_assigned_channel_metadata(
        self,
        image_def: dict,
        channel_name: str,
        definitions: list[dict],
    ) -> None:
        channel_name = str(channel_name or "").strip()
        source_def = next(
            (
                candidate
                for candidate in definitions
                if self.is_physical_channel_definition(candidate)
                and str(candidate.get("name", "") or "").strip() == channel_name
            ),
            None,
        )
        if source_def is None:
            return

        image_def["mask_source_channel"] = channel_name
        image_def["folder"] = str(source_def.get("folder", "") or channel_name)
        image_def["stack_channel_index"] = str(source_def.get("stack_channel_index", "") or "")
        image_def["stack_z_mode"] = str(source_def.get("stack_z_mode", "max_projection") or "max_projection")
        image_def["stack_z_index"] = str(source_def.get("stack_z_index", "") or "")
        image_def["display_color"] = str(source_def.get("display_color", "") or "")

    # Every path into this UI can supply partial dictionaries. Normalize once and
    # then repair channel/mask slots before any widgets are rebuilt.
    def normalize_image_definitions(self, definitions) -> list[dict]:
        if not isinstance(definitions, list):
            definitions = []

        normalized = [ImageGuiState.from_dict(item, i).to_dict() for i, item in enumerate(definitions)]

        while sum(self.is_physical_channel_definition(item) for item in normalized) < self.MINIMUM_CHANNEL_COUNT:
            normalized.append(self.create_default_image_definition(len(normalized)))

        normalized = self.normalize_channel_mask_slots(normalized)

        # Repair channel assignments before rebuilding the editors.
        channel_options = self.get_channel_assignment_options(normalized)
        for image_def in normalized:
            if not self.is_mask_only_definition(image_def) or self.is_combined_mask_definition(image_def):
                continue
            assigned_channel = str(image_def.get("mask_source_channel", "") or "").strip()
            if assigned_channel not in channel_options:
                assigned_channel = channel_options[0] if channel_options else ""
            if assigned_channel:
                self.apply_assigned_channel_metadata(image_def, assigned_channel, normalized)

        return normalize_analysis_relationships(normalized)

    # Display names are also relationship keys, so generated mask names must be
    # unique before they are inserted into matrices or presets.
    def unique_mask_name(self, base_name: str, definitions: list[dict]) -> str:
        existing = {str(image_def.get("name", "") or "").strip() for image_def in definitions}
        name = base_name
        counter = 2
        while name in existing:
            name = f"{base_name} {counter}"
            counter += 1
        return name

    # Each enabled channel mask slot is represented by a separate mask-only row.
    # Reconcile that pair here so adding, removing, or loading cannot orphan one.
    def normalize_channel_mask_slots(self, definitions: list[dict]) -> list[dict]:
        result: list[dict] = []
        pending_masks: list[tuple[str, dict[str, Any]]] = []

        for image_def in definitions:
            image_def = dict(image_def)
            if self.is_physical_channel_definition(image_def) and bool(image_def.get("mask_slot_enabled", True)):
                channel_name = str(image_def.get("name", "") or f"Channel {len(result) + 1}").strip()
                mask_name = self.unique_mask_name(
                    f"{channel_name} mask", [*result, *[mask for _old, mask in pending_masks]]
                )
                mask_def = dict(image_def)
                mask_def.update(
                    {
                        "name": mask_name,
                        "folder": str(image_def.get("folder", "") or channel_name),
                        "is_mask_only": True,
                        "mask_slot_enabled": True,
                        "mask_source_channel": channel_name,
                        "mask_relationships": {},
                        "analysis_cell_segmentation_enabled": False,
                    }
                )
                pending_masks.append((channel_name, mask_def))
                image_def.update(
                    {
                        "classifier": "",
                        "mask_slot_enabled": False,
                        "combined_mask_sources": [],
                    }
                )
            elif self.is_combined_mask_definition(image_def):
                image_def["is_mask_only"] = True
                image_def["mask_slot_enabled"] = True
            result.append(image_def)

        name_map = {old_name: mask_def["name"] for old_name, mask_def in pending_masks}
        if name_map:
            for image_def in result:
                relationships = dict(image_def.get("mask_relationships", {}) or {})
                for old_name, mask_name in name_map.items():
                    if old_name in relationships:
                        relationships[mask_name] = bool(relationships.pop(old_name)) or bool(
                            relationships.get(mask_name, False)
                        )
                image_def["mask_relationships"] = relationships

        result.extend(mask_def for _old_name, mask_def in pending_masks)
        return result

    def get_active_image_definitions(self) -> list[dict]:
        """Read committed definitions, isolated from caller mutations."""
        return deepcopy(self.image_definitions)

    def _read_image_row_edits(self, result: list[dict]) -> None:
        """Capture image widgets without normalization or relationship repair."""
        if not getattr(self, "image_rows", None):
            return
        for idx, row in enumerate(self.image_rows):
            if row is None or idx >= len(result):
                continue

            name_row = row.name
            if name_row is not None:
                display_name = name_row.get().strip() or row.default_name
                result[idx]["name"] = display_name
                result[idx]["folder"] = display_name

            classifier_row = row.classifier
            if classifier_row is not None:
                result[idx]["classifier"] = classifier_row.get().strip()

            mask_source_combo = row.mask_source_mode
            if mask_source_combo is not None:
                result[idx]["mask_source_mode"] = str(mask_source_combo.currentText())

            mask_channel_combo = row.mask_source_channel
            if mask_channel_combo is not None:
                result[idx]["mask_source_channel"] = str(mask_channel_combo.currentText()).strip()

            combined_sources_list = row.combined_mask_sources
            if combined_sources_list is not None:
                result[idx]["combined_mask_sources"] = [item.text() for item in combined_sources_list.selectedItems()]
            combined_operation_combo = row.combined_mask_operation
            if combined_operation_combo is not None:
                operation_value = combined_operation_combo.currentData()
                result[idx]["combined_mask_operation"] = str(
                    operation_value if operation_value is not None else combined_operation_combo.currentText()
                ).strip()

            threshold_method_combo = row.threshold_method
            if threshold_method_combo is not None:
                threshold_value = threshold_method_combo.currentData()
                result[idx]["threshold_method"] = (
                    str(threshold_value if threshold_value is not None else threshold_method_combo.currentText()).strip()
                    or DEFAULT_THRESHOLD_METHOD
                )

            probability_class_row = row.probability_class_index
            if probability_class_row is not None:
                mask_mode = str(result[idx].get("mask_source_mode", DEFAULT_MASK_SOURCE_MODE))
                result[idx]["probability_class_index"] = (
                    probability_class_row.get().strip() or "1"
                    if mask_mode == MASK_SOURCE_MODE_WEKA
                    else "1"
                )

    def commit_gui_edits(self, *, require_ready_input: bool = False) -> None:
        """Capture each editor once, remap names once, then publish coherent state.

        Tables still refer to the old names while an image name is being edited,
        so capture them before reading names and remapping their relationships.
        Builders only render state and must never commit half-built controls.
        """
        if self._rebuilding_image_tabs or getattr(self, "_loading_gui_configuration", False):
            return
        if getattr(self, "_committing_gui_edits", False):
            return
        self._committing_gui_edits = True
        try:
            definitions = self.get_active_image_definitions()
            old_names = image_display_names(definitions)
            self._read_processing_table_edits(definitions)
            self._read_cellpose_table_edits(definitions)
            self._read_analysis_matrix_edits(definitions)
            self._read_inline_filter_edits(definitions)
            self._read_image_row_edits(definitions)
            new_names = image_display_names(definitions)
            if old_names != new_names:
                definitions = remap_analysis_references(definitions, old_names, new_names)
            self.image_definitions = self.normalize_image_definitions(definitions)
            self._read_measurement_edits()

            state = getattr(self, "configuration_state", None)
            if state is not None:
                for name, widget_name in (
                    ("fiji_app_path", "fiji_app"), ("input_dir", "input_dir"),
                    ("output_dir", "output_dir"), ("mask_source_dir", "mask_source_dir"),
                ):
                    widget = getattr(self, widget_name, None)
                    if widget is not None:
                        setattr(state, name, widget.get().strip())
                if hasattr(self, "reuse_existing_masks_checkbox"):
                    state.reuse_existing_masks = self.reuse_existing_masks_checkbox.isChecked()
                if hasattr(self, "get_current_input_structure"):
                    state.input_structure = self.get_current_input_structure(require_ready=require_ready_input)
                if hasattr(self, "sync_pipeline_panel_note_from_editor"):
                    self.sync_pipeline_panel_note_from_editor()
            if old_names != new_names:
                # Saving before editingFinished must also refresh selectors. Otherwise
                # a later commit could write their old display names back into state.
                self.rebuild_image_rows(sync_from_ui=False)
        finally:
            self._committing_gui_edits = False

    def get_image_display_names(self) -> list[str]:
        return image_display_names(self.image_definitions)

    # Rebuild channel and mask tabs together so their indices stay aligned.
    def rebuild_image_rows(
        self,
        sync_from_ui: bool = True,
        *,
        rebuild_mask_source_tabs: bool = True,
        rebuild_channel_tabs: bool = True,
        refresh_dependent_views: bool = True,
    ):
        if not hasattr(self, "image_tabs") or not hasattr(self, "mask_tabs"):
            return

        if self._rebuilding_image_tabs:
            return

        if sync_from_ui:
            self.commit_gui_edits()
        self._rebuilding_image_tabs = True
        update_hosts = [self.mask_tabs]
        if rebuild_channel_tabs:
            update_hosts.append(self.image_tabs)
        if hasattr(self, "mask_source_tabs"):
            update_hosts.append(self.mask_source_tabs)
        for host in update_hosts:
            host.setUpdatesEnabled(False)
        try:
            current_channel_index = self.image_tabs.currentIndex()
            current_mask_index = self.mask_tabs.currentIndex()
            selected_mask_source = str(getattr(self, "_selected_mask_source_name", "") or "")

            if rebuild_channel_tabs:
                self.image_tabs.blockSignals(True)
            self.mask_tabs.blockSignals(True)
            if hasattr(self, "mask_source_tabs") and rebuild_mask_source_tabs:
                self.mask_source_tabs.blockSignals(True)
            if rebuild_channel_tabs:
                while self.image_tabs.count():
                    widget = self.image_tabs.widget(0)
                    if widget is not None:
                        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
                        widget.hide()
                    self.image_tabs.removeTab(0)
                    if widget is not None:
                        widget.deleteLater()
            while self.mask_tabs.count():
                widget = self.mask_tabs.widget(0)
                if widget is not None:
                    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
                    widget.hide()
                self.mask_tabs.removeTab(0)
                if widget is not None:
                    widget.deleteLater()

            channel_names = self.get_channel_assignment_options(self.image_definitions)
            if hasattr(self, "mask_source_tabs") and rebuild_mask_source_tabs:
                while self.mask_source_tabs.count():
                    widget = self.mask_source_tabs.widget(0)
                    if widget is not None:
                        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
                        widget.hide()
                    self.mask_source_tabs.removeTab(0)
                    if widget is not None:
                        widget.deleteLater()
                available_sources = [*channel_names, "Combined masks"]
                for source_name in available_sources:
                    self.mask_source_tabs.addTab(QWidget(self.mask_source_tabs), source_name)
                for tab_index, source_name in enumerate(channel_names):
                    source_masks = [
                        image_def
                        for image_def in self.image_definitions
                        if not self.is_combined_mask_definition(image_def)
                        and bool(image_def.get("is_mask_only", False))
                        and str(image_def.get("mask_source_channel", "") or "").strip() == source_name
                    ]
                    has_runnable_mask = any(
                        Path(str(image_def.get("classifier", "") or "")).is_file()
                        for image_def in source_masks
                        if str(image_def.get("classifier", "") or "").strip()
                    )
                    has_mask_row = bool(source_masks)
                    color = "#69d38a" if has_runnable_mask else "#d9a441" if has_mask_row else "#e8757e"
                    status = (
                        "Classifier file found"
                        if has_runnable_mask
                        else "Mask setup incomplete or classifier unavailable"
                        if has_mask_row
                        else "No mask added"
                    )
                    self.mask_source_tabs.tabBar().setTabTextColor(tab_index, QColor(color))
                    self.mask_source_tabs.setTabToolTip(tab_index, f"{source_name}: {status}")
                if selected_mask_source not in available_sources:
                    selected_mask_source = channel_names[0] if channel_names else "Combined masks"
                self._selected_mask_source_name = selected_mask_source
                self.mask_source_tabs.setCurrentIndex(available_sources.index(selected_mask_source))

            self._build_image_definition_tabs(build_channel_tabs=rebuild_channel_tabs)
            if rebuild_channel_tabs:
                channel_plus_tab = QWidget(self.image_tabs)
                self.image_tabs.addTab(channel_plus_tab, "+")
                self.image_tabs.setTabToolTip(self.image_tabs.count() - 1, "Add a new channel")

                if self.image_tabs.count() > 1:
                    if current_channel_index < 0:
                        current_channel_index = 0
                    current_channel_index = min(current_channel_index, self.image_tabs.count() - 2)
                    self.image_tabs.setCurrentIndex(current_channel_index)

            mask_plus_tab = QWidget(self.mask_tabs)
            self.mask_tabs.addTab(mask_plus_tab, "Add mask")
            self.mask_tabs.setTabToolTip(self.mask_tabs.count() - 1, "Add a mask to the selected image")

            if self.mask_tabs.count() > 1:
                if current_mask_index < 0:
                    current_mask_index = 0
                current_mask_index = min(current_mask_index, self.mask_tabs.count() - 2)
                self.mask_tabs.setCurrentIndex(current_mask_index)

            self.mask_tabs.blockSignals(False)
            if rebuild_channel_tabs:
                self.image_tabs.blockSignals(False)
            if hasattr(self, "mask_source_tabs") and rebuild_mask_source_tabs:
                self.mask_source_tabs.blockSignals(False)

        finally:
            self._rebuilding_image_tabs = False
            for host in update_hosts:
                host.setUpdatesEnabled(True)
                host.update()

        if not refresh_dependent_views:
            return

        self.refresh_channel_name_dependent_ui()
        if hasattr(self, "mask_processing_table"):
            self.rebuild_mask_processing_table()
        if hasattr(self, "cellpose_settings_table"):
            self.rebuild_cellpose_settings_table()
        self.refresh_analysis_matrix_from_definitions()

        # Replaced definitions (preset load, dataset detection, add/remove or
        # rename) must render all editors before the next commit can read them.
        if hasattr(self, "analysis_measurements_panel"):
            self._populate_inline_measurement_settings()
        row = getattr(self, "_inline_analysis_settings_row", None)
        if row is not None:
            if 0 <= int(row) < len(self.image_definitions):
                self.refresh_analysis_filter_panel_for_open_preview(int(row))
            else:
                self._inline_analysis_settings_row = None

    def _build_image_definition_tabs(self, *, build_channel_tabs: bool = True) -> None:
        if build_channel_tabs or len(getattr(self, "image_rows", [])) != len(self.image_definitions):
            self.image_rows = [None for _image_def in self.image_definitions]
        channel_ordinal = 0
        combined_ordinal = 0

        # One definition may represent a physical channel, its mask slot,
        # a Weka mask or a combined mask. The tab labels are derived
        # here so later code can use one shared definition list.
        for i, image_def in enumerate(self.image_definitions):
            is_combined_definition = self.is_combined_mask_definition(image_def)
            is_extra_mask_definition = self.is_mask_only_definition(image_def)
            is_assigned_mask_definition = is_extra_mask_definition and not is_combined_definition
            is_channel_definition = self.is_physical_channel_definition(image_def)
            if is_combined_definition:
                combined_ordinal += 1
                default_name = f"Combined mask {combined_ordinal}"
            elif is_extra_mask_definition:
                default_name = (
                    f"Mask {sum(bool(defn.get('is_mask_only', False)) for defn in self.image_definitions[:i]) + 1}"
                )
            else:
                channel_ordinal += 1
                default_name = f"Channel {channel_ordinal}"

            display_name = str(image_def.get("name", default_name) or default_name)
            existing_row = self.image_rows[i]
            row = (
                existing_row
                if not build_channel_tabs and is_channel_definition and existing_row is not None
                else ImageDefinitionRowWidgets(definition_index=i, default_name=default_name)
            )

            if is_channel_definition and build_channel_tabs:
                channel_tab = QWidget(self.image_tabs)
                channel_layout = QVBoxLayout(channel_tab)
                channel_layout.setContentsMargins(10, 10, 10, 10)
                channel_layout.setSpacing(8)

                header_row = QHBoxLayout()
                header_row.setContentsMargins(0, 0, 0, 0)
                header_row.setSpacing(8)

                title = QLabel(f"Channel {channel_ordinal}")
                title.setProperty("uiRole", "sectionTitle")

                remove_btn = QPushButton("Remove this channel")
                self.set_button_role(remove_btn, "danger")
                can_remove_channel = self.physical_channel_count() > self.MINIMUM_CHANNEL_COUNT
                remove_btn.setEnabled(can_remove_channel)
                if can_remove_channel:
                    remove_btn.clicked.connect(lambda _checked=False, idx=i: self.remove_image_row_at(idx))

                header_row.addWidget(title)
                header_row.addStretch(1)
                header_row.addWidget(remove_btn)

                name_row = EntryRow(
                    "Channel / folder name",
                    display_name,
                    "For folder-based input, this must match the channel folder. It also labels the channel in the UI and results.",
                )
                channel_layout.addLayout(header_row)
                channel_layout.addWidget(name_row)
                channel_layout.addStretch(1)

                row.channel_tab = channel_tab
                row.name = name_row
                row.channel_tab_index = self.image_tabs.count()
                self.image_tabs.addTab(channel_tab, display_name)

                name_row.edit.textChanged.connect(lambda *_args, idx=i: self.update_image_tab_title(idx))
                name_row.edit.editingFinished.connect(self.on_image_definitions_changed)

            if not is_extra_mask_definition:
                self.image_rows[i] = row
                continue

            uses_source_tabs = hasattr(self, "mask_source_tabs")
            selected_mask_source = str(getattr(self, "_selected_mask_source_name", "") or "")
            belongs_to_selected_image = (
                is_assigned_mask_definition
                and str(image_def.get("mask_source_channel", "") or "").strip() == selected_mask_source
            )
            if uses_source_tabs and not (
                belongs_to_selected_image
                or (is_combined_definition and selected_mask_source == "Combined masks")
            ):
                self.image_rows[i] = row
                continue

            mask_tab = QWidget(self.mask_tabs)
            mask_layout = QVBoxLayout(mask_tab)
            mask_layout.setContentsMargins(10, 10, 10, 10)
            mask_layout.setSpacing(8)

            mask_header_row = QHBoxLayout()
            mask_header_row.setContentsMargins(0, 0, 0, 0)
            mask_header_row.setSpacing(8)

            mask_title = QLabel(f"Combined mask {combined_ordinal}" if is_combined_definition else "Mask")
            mask_title.setProperty("uiRole", "sectionTitle")
            mask_header_row.addWidget(mask_title)
            mask_header_row.addStretch(1)

            mask_source_channel_combo = None
            assigned_row = None
            remove_mask_btn = QPushButton("Remove this mask")
            self.set_button_role(remove_mask_btn, "danger")
            remove_mask_btn.clicked.connect(lambda _checked=False, idx=i: self.remove_image_row_at(idx))
            mask_header_row.addWidget(remove_mask_btn)

            mask_kind = "combined mask" if is_combined_definition else "mask"
            name_row = EntryRow(
                "Display name",
                display_name,
                f"Name used for this {mask_kind} in overlays and measurements.",
                label_width=250,
            )
            row.name = name_row
            name_row.edit.textChanged.connect(lambda *_args, idx=i: self.update_image_tab_title(idx))
            name_row.edit.editingFinished.connect(self.on_image_definitions_changed)

            classifier_row = PathRow(
                "Weka classifier",
                default=image_def.get("classifier", ""),
                mode="file",
                file_filter="Weka model (*.model);;All files (*.*)",
                tooltip="A .model file saved by Fiji's Trainable Weka Segmentation. Training-data .arff files are not classifiers.",
                label_width=250,
            )

            threshold_method_value = str(
                image_def.get("threshold_method", DEFAULT_THRESHOLD_METHOD) or DEFAULT_THRESHOLD_METHOD
            )
            threshold_method_host = ComboRow(
                "Probability-map threshold method",
                THRESHOLD_METHOD_OPTIONS,
                threshold_method_value,
                "Uses Fiji/ImageJ's built-in Image > Adjust > Threshold methods on a global 256-bin histogram. "
                "Dark background is fixed ON, so high-probability pixels become foreground.",
                label_width=250,
            )
            threshold_method_combo = threshold_method_host.combo

            probability_class_row = EntryRow(
                "Probability-map class number(s)",
                image_def.get("probability_class_index", "1"),
                "Uses the class order stored in the Fiji classifier: 1 is the first class, 2 is the second, and so on. "
                "Enter 1, 1,3, or 1-3. Each selected class creates a separate mask.",
                label_width=250,
            )
            combined_sources_list = QListWidget()
            combined_sources_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            combined_sources_list.setMaximumHeight(120)
            combined_sources_list.setToolTip(
                "Only completed Weka and combined masks are listed. Select at least two to combine."
            )
            combined_operation_row = ComboRow(
                "Combine using",
                [("OR (Combine)", "OR"), "AND", "XOR"],
                str(image_def.get("combined_mask_operation", DEFAULT_COMBINED_MASK_OPERATION) or DEFAULT_COMBINED_MASK_OPERATION),
                "OR (Combine) keeps all areas covered by any selected mask. AND keeps only the area "
                "shared by every selected mask. XOR keeps areas in one mask or the other, "
                "but not their overlap. With three or more masks, XOR is applied in order: "
                "areas covered by an odd number of masks remain. For a multi-class Weka mask, its selected "
                "classes are united before this operation. Combined masks may be used as later sources.",
                label_width=250,
            )
            combined_operation_combo = combined_operation_row.combo
            selected_combined_sources = set(image_def.get("combined_mask_sources", []) or [])
            configured_masks = configured_mask_names(self.image_definitions)
            available_mask_names = [
                str(candidate.get("name", "") or f"Channel {candidate_index + 1}")
                for candidate_index, candidate in enumerate(self.image_definitions)
                if candidate_index != i
                and str(candidate.get("name", "") or f"Channel {candidate_index + 1}") in configured_masks
            ]
            self._set_listwidget_items_with_selection(
                combined_sources_list,
                available_mask_names,
                selected=selected_combined_sources,
            )
            combined_sources_label = QLabel("Masks to combine")
            combined_sources_label.setProperty("uiRole", "fieldLabel")
            combined_sources_label.setMinimumWidth(250)

            roi_status = QLabel()
            roi_status.setWordWrap(True)
            roi_status.setProperty("muted", "true")
            inspected_model = {"path": "", "message": ""}

            # Mask methods expose different fields, so switch the controls as
            # one group and keep the status text aligned with the visible mode.
            def update_roi_status_label(
                classifier_row=classifier_row,
                probability_class_row=probability_class_row,
                threshold_method_host=threshold_method_host,
                roi_status=roi_status,
                combined_sources_list=combined_sources_list,
                combined_sources_label=combined_sources_label,
                combined_operation_row=combined_operation_row,
                combined_operation_combo=combined_operation_combo,
                assigned_row=assigned_row,
                is_combined_definition=is_combined_definition,
                inspected_model=inspected_model,
            ):
                has_classifier = bool(classifier_row.get().strip())
                classifier_path = Path(classifier_row.get().strip()) if has_classifier else None
                classifier_exists = bool(classifier_path and classifier_path.is_file())
                combined = is_combined_definition
                if assigned_row is not None:
                    assigned_row.setVisible(not combined)
                classifier_row.setVisible(not combined)
                threshold_method_host.setVisible(not combined)
                probability_class_row.setVisible(not combined and has_classifier)
                combined_sources_label.setVisible(combined)
                combined_sources_list.setVisible(combined)
                combined_operation_row.setVisible(combined)
                if combined:
                    count = len(combined_sources_list.selectedItems())
                    operation_data = combined_operation_combo.currentData()
                    operation = str(operation_data if operation_data is not None else combined_operation_combo.currentText())
                    roi_status.setText(
                        f"{count} source mask(s) selected for {operation}. Choose at least two; "
                        "the named result becomes a measurement-matrix column."
                    )
                elif classifier_exists:
                    if inspected_model["path"] == str(classifier_path):
                        roi_status.setText(inspected_model["message"])
                        return
                    class_text = probability_class_row.get().strip() or "1"
                    has_multiple_classes = "," in class_text or ";" in class_text or "-" in class_text
                    if has_multiple_classes:
                        roi_status.setText(
                            "Classifier file found. Each selected Weka class becomes a separate mask output. "
                            "Preview or Run verifies compatibility and the available class count."
                        )
                    else:
                        roi_status.setText(
                            "Classifier file found. Preview or Run verifies compatibility and the available class count."
                        )
                elif has_classifier:
                    roi_status.setText("Classifier file not found. Choose the .model file again.")
                else:
                    roi_status.setText(
                        "This tab does not create a mask yet. Choose a Weka .model classifier. "
                        "To measure cells without this mask, enable Cellpose for the channel."
                    )

            def inspect_selected_classifier(
                *_args,
                classifier_row=classifier_row,
                roi_status=roi_status,
                inspected_model=inspected_model,
                update_status=update_roi_status_label,
            ):
                model_path = Path(classifier_row.get().strip())
                if not model_path.is_file():
                    inspected_model.update(path="", message="")
                    update_status()
                    return
                roi_status.setText("Reading Weka model information...")
                QApplication.processEvents()
                try:
                    fiji_path = self.fiji_app.get() if hasattr(self, "fiji_app") else ""
                    labels = inspect_weka_classifier(model_path, fiji_path)
                    numbered_labels = "; ".join(
                        f"{index}: {label}" for index, label in enumerate(labels, start=1)
                    )
                    message = (
                        f"Weka model loaded successfully: {len(labels)} classes. "
                        f"Probability-map class order: {numbered_labels}."
                    )
                except Exception as exc:
                    message = f"Weka model could not be loaded: {exc}"
                inspected_model.update(path=str(model_path), message=message)
                update_status()

            mask_layout.addLayout(mask_header_row)
            if name_row is not None:
                mask_layout.addWidget(name_row)
            if assigned_row is not None:
                mask_layout.addWidget(assigned_row)
            mask_layout.addWidget(classifier_row)
            mask_layout.addWidget(threshold_method_host)
            mask_layout.addWidget(probability_class_row)
            mask_layout.addWidget(combined_sources_label)
            mask_layout.addWidget(combined_sources_list)
            mask_layout.addWidget(combined_operation_row)
            mask_layout.addWidget(roi_status)
            mask_layout.addStretch(1)

            update_roi_status_label()

            row.mask_tab = mask_tab
            row.classifier = classifier_row
            row.mask_source_mode = None
            row.mask_source_channel = mask_source_channel_combo
            row.probability_class_index = probability_class_row
            row.threshold_method = threshold_method_combo
            row.threshold_method_host = threshold_method_host
            row.combined_mask_sources = combined_sources_list
            row.combined_mask_operation = combined_operation_combo
            row.roi_status = roi_status
            row.mask_tab_index = self.mask_tabs.count()
            self.mask_tabs.addTab(mask_tab, display_name)
            self.image_rows[i] = row

            classifier_row.edit.textChanged.connect(self.on_image_definitions_changed)
            classifier_row.edit.textChanged.connect(lambda *_args, fn=update_roi_status_label: fn())
            classifier_row.edit.editingFinished.connect(inspect_selected_classifier)
            classifier_row.edit.pathDropped.connect(inspect_selected_classifier)
            classifier_row.button.clicked.connect(inspect_selected_classifier)
            probability_class_row.edit.textChanged.connect(self.on_image_definitions_changed)
            probability_class_row.edit.textChanged.connect(lambda *_args, fn=update_roi_status_label: fn())
            threshold_method_combo.currentIndexChanged.connect(self.on_image_definitions_changed)

            if mask_source_channel_combo is not None:
                mask_source_channel_combo.currentTextChanged.connect(self.on_image_definitions_changed)
            combined_sources_list.itemSelectionChanged.connect(self.on_image_definitions_changed)
            combined_sources_list.itemSelectionChanged.connect(update_roi_status_label)
            combined_operation_combo.currentTextChanged.connect(self.on_image_definitions_changed)
            combined_operation_combo.currentTextChanged.connect(lambda *_args, fn=update_roi_status_label: fn())

    # Save current edits before rebuilding the rows.
    def add_image_row(self):
        if self._rebuilding_image_tabs:
            return

        self.commit_gui_edits()
        image_def = self.create_default_image_definition(len(self.image_definitions))
        image_def["mask_slot_enabled"] = False
        self.image_definitions.append(image_def)
        self.rebuild_image_rows(sync_from_ui=False)

        if hasattr(self, "image_tabs") and self.image_tabs.count() >= 2:
            self.image_tabs.blockSignals(True)
            self.image_tabs.setCurrentIndex(self.image_tabs.count() - 2)
            self.image_tabs.blockSignals(False)

    # The selected image tab supplies the source relationship. Combined masks
    # stay global because they can use masks from several images.
    def add_mask_row(self):
        if self._rebuilding_image_tabs:
            return

        self.commit_gui_edits()
        selected_source = str(getattr(self, "_selected_mask_source_name", "") or "")
        combined = selected_source == "Combined masks"
        channel_names = self.get_channel_assignment_options(self.image_definitions)
        assigned_channel = "" if combined else selected_source
        if assigned_channel not in channel_names:
            assigned_channel = channel_names[0] if channel_names else ""
        mask_number = 1 + sum(bool(image_def.get("is_mask_only", False)) for image_def in self.image_definitions)
        image_def = self.create_default_image_definition(len(self.image_definitions))
        image_def.update(
            {
                "name": self.unique_mask_name(
                    "Combined mask" if combined else f"{assigned_channel} mask",
                    self.image_definitions,
                ),
                "folder": assigned_channel or f"Mask {mask_number}",
                "is_mask_only": True,
                "mask_slot_enabled": True,
                "mask_source_channel": assigned_channel,
                "mask_source_mode": MASK_SOURCE_MODE_COMBINED if combined else MASK_SOURCE_MODE_WEKA,
                "classifier": "",
                "mask_relationships": {},
                "analysis_cell_segmentation_enabled": False,
                "combined_mask_sources": [],
                "combined_mask_operation": DEFAULT_COMBINED_MASK_OPERATION,
            }
        )
        if assigned_channel:
            self.apply_assigned_channel_metadata(image_def, assigned_channel, self.image_definitions)
        self.image_definitions.append(image_def)
        self.rebuild_image_rows(sync_from_ui=False)
        if hasattr(self, "mask_tabs") and self.mask_tabs.count() >= 2:
            self.mask_tabs.blockSignals(True)
            self.mask_tabs.setCurrentIndex(self.mask_tabs.count() - 2)
            self.mask_tabs.blockSignals(False)

    # Enforce the physical-channel minimum before rebuilding because mask-only
    # rows do not satisfy the requirement for an input image.
    def remove_image_row_at(self, index: int):
        self.commit_gui_edits()

        if index < 0 or index >= len(self.image_definitions):
            return

        image_def = self.image_definitions[index]
        if (
            self.is_physical_channel_definition(image_def)
            and self.physical_channel_count() <= self.MINIMUM_CHANNEL_COUNT
        ):
            QMessageBox.information(self, "Channels", "At least 1 channel is required.")
            return

        self.image_definitions.pop(index)
        self.image_definitions = self.normalize_image_definitions(self.image_definitions)
        self.rebuild_image_rows(sync_from_ui=False)

    # Names can be keys in open analysis panels and matrices. Remap those
    # references before rebuilding, then restore whichever inline panel was open.
    def on_image_definitions_changed(self, *_args):
        if self._rebuilding_image_tabs:
            return

        for row in getattr(self, "image_rows", []):
            if row is None:
                continue
            prob_row = row.probability_class_index
            source_combo = row.mask_source_mode
            if prob_row is not None and (
                source_combo is None or source_combo.currentText() == MASK_SOURCE_MODE_WEKA
            ):
                self.validate_probability_classes_field(
                    prob_row.edit,
                    field_name="Probability-map class numbers",
                )

        self.commit_gui_edits()
        self.refresh_channel_name_dependent_ui()
        self.refresh_analysis_matrix_from_definitions()

    # Refresh channel labels throughout the interface after names or rows change.
    def refresh_channel_name_dependent_ui(self):
        if self._rebuilding_image_tabs:
            return

        names = self.get_image_display_names()
        if not names:
            names = ["Channel 1"]

        if hasattr(self, "overlay_roi_list"):
            selected = {item.text() for item in self.overlay_roi_list.selectedItems()}
            active_defs = self.get_active_image_definitions()
            mask_names = [img["name"].strip() or "Image" for img in active_defs if image_produces_mask(img)]

            overlay_names = list(mask_names)
            overlay_names.append("Whole cell mask")

            self._set_listwidget_items_with_selection(
                self.overlay_roi_list,
                overlay_names,
                selected=selected,
                select_all_if_empty=False,
            )
