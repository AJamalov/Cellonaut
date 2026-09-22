"""Cell-group membership and display layers for the preview tab."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cellonaut.gui.mixin import GuiMixin
from cellonaut.config.relationships import cellpose_intensity_source_name
from cellonaut.masks.cell_groups import evaluate_cell_groups
from cellonaut.masks.cell_qc import find_matching_cell_qc_column
from cellonaut.masks.preview_filter_overlay import (
    _selected_mask_label,
    find_image_def_for_label,
    find_preview_filter_data,
)


# Coordinate live filter highlights with the loaded artifact because stale
# tables or labels must be rejected before anything is drawn over the preview.
class CellonautGuiPreviewFilterMixin(GuiMixin):
    def pulse_preview_work_busy_icon(self) -> None:
        """Let the static busy indicator paint during synchronous work."""
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    @contextmanager
    def preview_work_busy(self, message: str):
        """Paint a local busy indicator around synchronous preview calculations."""
        indicator = getattr(self, "preview_work_busy_indicator", None)
        label = getattr(self, "preview_work_busy_text", None)
        if label is not None:
            label.setText(message)
        if indicator is not None:
            indicator.setVisible(True)
            self.pulse_preview_work_busy_icon()
        try:
            yield
        finally:
            if indicator is not None:
                indicator.setVisible(False)

    def set_preview_filter_bar_status(self, summary: str, detail: str = "") -> None:
        """Keep the navigation bar concise while retaining diagnostic detail."""
        label = getattr(self, "preview_filter_status_label", None)
        if label is None:
            return
        label.setText(str(summary or ""))
        tooltip = str(detail or summary or "")
        if hasattr(label, "setDetailToolTip"):
            label.setDetailToolTip(tooltip)
        else:
            label.setToolTip(tooltip)

    # Remove scene-owned graphics before dropping references because Qt may keep
    # an otherwise invisible item alive across preview changes.
    def clear_preview_filter_overlay(self):
        layer_list = getattr(self, "preview_layer_list", None)
        if layer_list is not None:
            for row_index in range(layer_list.count() - 1, -1, -1):
                list_item = layer_list.item(row_index)
                if str(list_item.data(Qt.ItemDataRole.UserRole + 2) or "") == "cell_group":
                    row_widget = layer_list.itemWidget(list_item)
                    layer_list.takeItem(row_index)
                    if row_widget is not None:
                        row_widget.deleteLater()
        for population_item in getattr(self, "_preview_population_items", {}).values():
            try:
                if population_item.scene() is not None:
                    population_item.scene().removeItem(population_item)
            except RuntimeError:
                pass
        self._preview_population_items = {}
        self.preview_state.group_layer_display_state = {}
        item = getattr(self, "_preview_filter_item", None)
        if item is not None:
            try:
                if item.scene() is not None:
                    item.scene().removeItem(item)
            except RuntimeError:
                pass
        self._preview_filter_item = None
        self.preview_state.filter_data = None
        self.preview_state.cell_group_result = None
        self.preview_state.filter_excluded_labels = set()
        if hasattr(self, "preview_filter_status_label"):
            self.set_preview_filter_bar_status("No linked filter results")
        if hasattr(self, "analysis_filter_live_status_label"):
            self.analysis_filter_live_status_label.setText("Preview link: no overlay loaded")
        if hasattr(self, "analysis_export_filtered_csv_button"):
            self.analysis_export_filtered_csv_button.setEnabled(False)
            self.analysis_export_filtered_csv_button.setToolTip(
                "Open a generated overlay with linked cell measurements first."
            )
        if hasattr(self, "update_mask_adjustment_status"):
            self.update_mask_adjustment_status()

    # Render selected cell IDs as one transparent layer so changing groups does
    # not require rebuilding the underlying multichannel preview.
    def make_preview_filter_pixmap(self, label_img: np.ndarray, excluded_labels: set[int], color="#ff2828") -> QPixmap:
        labels = np.asarray(label_img)
        if labels.ndim > 2:
            labels = np.squeeze(labels)
        if labels.ndim != 2 or not excluded_labels:
            rgba = np.zeros((1, 1, 4), dtype=np.uint8)
            qimg = QImage(rgba.data, 1, 1, 4, QImage.Format.Format_RGBA8888).copy()
            return QPixmap.fromImage(qimg)

        mask = np.isin(labels.astype(np.int64, copy=False), list(excluded_labels))
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        value = str(color or "#ff2828").lstrip("#")
        try:
            red, green, blue = tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
        except (ValueError, TypeError):
            red, green, blue = 255, 40, 40
        rgba[..., 0] = red
        rgba[..., 1] = green
        rgba[..., 2] = blue
        rgba[..., 3] = np.where(mask, 145, 0).astype(np.uint8)
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(qimg)

    # Reuse the scene item during edits to avoid changing layer order or losing
    # the user's current zoom and pan position.
    def update_preview_filter_overlay_item(self):
        item = getattr(self, "_preview_filter_item", None)
        if not self.preview_state.filter_visible:
            if item is not None:
                item.setVisible(False)
            return

        data = self.preview_state.filter_data
        excluded = set(self.preview_state.filter_excluded_labels or set())
        if data is None or not excluded:
            if item is not None:
                item.setVisible(False)
            return

        pixmap = self.make_preview_filter_pixmap(data.label_image, excluded)
        if pixmap.isNull():
            return

        if item is None:
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(10000)
            self.preview_scene.addItem(item)
            self._preview_filter_item = item
        else:
            item.setPixmap(pixmap)
            if item.scene() is None:
                self.preview_scene.addItem(item)
        item.setZValue(10000)
        item.setVisible(True)

    def rebuild_preview_population_layers(self, populations):
        """Build cell groups as ordinary draggable rows in the main layer list."""
        layer_list = getattr(self, "preview_layer_list", None)
        filter_data = self.preview_state.filter_data
        if layer_list is None or filter_data is None:
            return
        display_state = dict(self.preview_state.group_layer_display_state or {})
        for row_index in range(layer_list.count() - 1, -1, -1):
            list_item = layer_list.item(row_index)
            if str(list_item.data(Qt.ItemDataRole.UserRole + 2) or "") != "cell_group":
                continue
            key = str(list_item.data(Qt.ItemDataRole.UserRole + 3) or "")
            row_widget = layer_list.itemWidget(list_item)
            state = dict(display_state.get(key, {}))
            state["row"] = row_index
            display_state[key] = state
            layer_list.takeItem(row_index)
            if row_widget is not None:
                row_widget.deleteLater()

        old_items = getattr(self, "_preview_population_items", {})
        active_items = {}
        populations = sorted(
            populations,
            key=lambda population: display_state.get(str(population.get("key", "")), {}).get("row", layer_list.count()),
        )
        for index, population in enumerate(populations):
            key = str(population.get("key", index))
            labels = set(population.get("labels", set()))
            state = display_state.setdefault(key, {})
            state.setdefault("visible", self.preview_state.filter_visible)
            state.setdefault("opacity", 100)
            color = str(state.get("color") or population.get("color", "#00d7ff"))
            item = old_items.pop(key, None)
            pixmap = self.make_preview_filter_pixmap(filter_data.label_image, labels, color)
            if item is None:
                item = QGraphicsPixmapItem(pixmap)
                self.preview_scene.addItem(item)
            else:
                item.setPixmap(pixmap)
            item.setZValue(10001 + index)
            item.setVisible(bool(state.get("visible", self.preview_state.filter_visible)))
            item.setOpacity(int(state.get("opacity", 100)) / 100.0)
            active_items[key] = item

            layer_control = QWidget()
            layer_control.setProperty("uiRole", "previewLayerRow")
            layer_control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            outer = QVBoxLayout(layer_control)
            outer.setContentsMargins(4, 4, 4, 4)
            outer.setSpacing(4)
            if index == 0:
                heading = QLabel("CELL GROUPS")
                heading.setProperty("uiRole", "previewLayerGroup")
                outer.addWidget(heading)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            from cellonaut.gui.preview_layers import PreviewLayerDragHandle

            drag_handle = PreviewLayerDragHandle(layer_list)
            visible = QCheckBox()
            visible.setFixedWidth(22)
            visible.setChecked(bool(state.get("visible", self.preview_state.filter_visible)))
            visible.setToolTip("Show or hide this cell-group layer.")
            def set_visible(value, graphics=item, group_state=state):
                group_state["visible"] = bool(value)
                graphics.setVisible(bool(value))
            visible.toggled.connect(set_visible)
            swatch = QPushButton()
            self.set_preview_color_button_style(swatch, color)
            swatch.setToolTip(f"Cell-group color: {color}")
            color_menu = QMenu(swatch)
            for color_name, palette_color in self.preview_color_palette():
                action = color_menu.addAction(color_name)
                def apply_group_color(
                    _checked=False,
                    value=palette_color,
                    graphics=item,
                    cell_labels=labels,
                    button=swatch,
                    group_key=key,
                ):
                    self.preview_state.group_layer_display_state[group_key]["color"] = value
                    graphics.setPixmap(
                        self.make_preview_filter_pixmap(
                            filter_data.label_image, cell_labels, value
                        )
                    )
                    self.set_preview_color_button_style(button, value)
                    button.setToolTip(f"Cell-group color: {value}")
                action.triggered.connect(apply_group_color)
            swatch.setMenu(color_menu)
            name = QLabel(f"{population['name']} ({len(labels)})")
            name.setProperty("uiRole", "previewLayerName")
            name.setWordWrap(True)
            name.setMaximumWidth(108)
            opacity = QSlider(Qt.Orientation.Horizontal)
            opacity.setRange(0, 100)
            opacity.setValue(int(state.get("opacity", 100)))
            opacity.setToolTip(f"Opacity: {opacity.value()}%")
            opacity.valueChanged.connect(lambda value, slider=opacity: slider.setToolTip(f"Opacity: {value}%"))
            opacity.setFixedWidth(96)
            def set_opacity(value, graphics=item, group_state=state):
                group_state["opacity"] = int(value)
                graphics.setOpacity(value / 100.0)
            opacity.valueChanged.connect(set_opacity)
            row.addWidget(drag_handle)
            row.addWidget(visible)
            row.addWidget(swatch)
            row.addWidget(name, 1)
            outer.addLayout(row)
            opacity_row = QHBoxLayout()
            opacity_row.addStretch(1)
            opacity_row.addWidget(opacity)
            outer.addLayout(opacity_row)
            row_height = 76 if index == 0 else 60
            layer_control.setMinimumHeight(row_height)
            list_item = QListWidgetItem()
            list_item.setData(Qt.ItemDataRole.UserRole, -1)
            list_item.setData(Qt.ItemDataRole.UserRole + 1, "Cell groups")
            list_item.setData(Qt.ItemDataRole.UserRole + 2, "cell_group")
            list_item.setData(Qt.ItemDataRole.UserRole + 3, key)
            list_item.setSizeHint(QSize(0, row_height))
            layer_list.insertItem(min(int(state.get("row", layer_list.count())), layer_list.count()), list_item)
            layer_list.setItemWidget(list_item, layer_control)
            drag_handle.list_item = list_item
            state["color"] = color
        for item in old_items.values():
            if item.scene() is not None:
                item.scene().removeItem(item)
        self._preview_population_items = active_items
        self.preview_state.group_layer_display_state = {
            key: display_state[key] for key in active_items
        }
        layer_list.refresh_category_headers()
        self.on_preview_layer_order_changed()
        self.set_preview_layer_panel_visible(True)

    # Reset export controls so they cannot use the previous preview's table.
    def set_preview_filter_unavailable(self, message: str, *, failed: bool = False):
        self.clear_preview_filter_overlay()
        filter_prefix = "Cell-group preview failed" if failed else "Cell-group preview"
        link_prefix = "Preview link failed" if failed else "Preview link"
        if hasattr(self, "preview_filter_status_label"):
            summary = "Cell-group preview unavailable" if not failed else "Cell-group preview failed"
            self.set_preview_filter_bar_status(summary, f"{filter_prefix}: {message}")
        if hasattr(self, "analysis_filter_live_status_label"):
            self.analysis_filter_live_status_label.setText(f"{link_prefix}: {message}")
        if hasattr(self, "analysis_export_filtered_csv_button"):
            self.analysis_export_filtered_csv_button.setEnabled(False)
            self.analysis_export_filtered_csv_button.setToolTip(message)

    # Relink from disk on refresh because users can open artifacts from older
    # runs whose tables and labels differ from the current output directory.
    def refresh_preview_filter_overlay(self):
        with self.preview_work_busy("Updating filters…"):
            self._refresh_preview_filter_overlay()

    def _refresh_preview_filter_overlay(self):
        preview_path = str(self.preview_state.file_path or "").strip()
        preview_file = Path(preview_path) if preview_path else None
        if preview_file is None or not preview_file.is_file():
            self.clear_preview_filter_overlay()
            return

        try:
            if hasattr(self, "commit_gui_edits"):
                self.commit_gui_edits()
            image_defs = self.get_active_image_definitions() if hasattr(self, "get_active_image_definitions") else []
        except Exception as exc:
            message = f"could not read current cell-group settings ({type(exc).__name__}: {exc})"
            self.set_preview_filter_unavailable(message, failed=True)
            log_func = getattr(self, "log", None)
            if callable(log_func):
                log_func(f"[PREVIEW][WARN] {message}")
            return

        try:
            data, message = find_preview_filter_data(preview_file, image_defs)
        except Exception as exc:
            message = f"could not link preview results ({type(exc).__name__}: {exc})"
            self.set_preview_filter_unavailable(message, failed=True)
            log_func = getattr(self, "log", None)
            if callable(log_func):
                log_func(f"[PREVIEW][WARN] {message}")
            return
        self.preview_state.filter_data = data
        self.preview_state.filter_excluded_labels = set()

        if data is None:
            filter_item = getattr(self, "_preview_filter_item", None)
            if filter_item is not None:
                filter_item.setVisible(False)
            self.set_preview_filter_unavailable(message)
            return

        image_def = find_image_def_for_label(image_defs, data.source_label)
        if image_def is None:
            message = f"no current settings found for {data.source_label}"
            self.set_preview_filter_unavailable(message)
            return

        self.preview_state.filter_excluded_labels = set()
        result = evaluate_cell_groups(
            data.table, image_def, self.parse_cell_qc_limits_text,
            prepare_table=lambda table, settings: self.attach_preview_mask_filter_metrics(table, data.label_image, settings),
        )
        self.preview_state.cell_group_result = result
        for group in result.groups:
            if group.error:
                message = f"could not evaluate {group.name} ({group.error})"
                self.set_preview_filter_unavailable(message, failed=True)
                log_func = getattr(self, "log", None)
                if callable(log_func):
                    log_func(f"[PREVIEW][WARN] {message}")
                return
        # Unavailable metrics leave that group's overlay empty. Export checks
        # require_available() on the same result and refuses an incomplete table.
        named_populations = [
            {"key": str(group.index), "name": group.name, "color": group.color, "labels": set(group.labels)}
            for group in result.groups
        ]
        self.rebuild_preview_population_layers(named_populations)

        total = len(result.cell_labels)
        status = f"Cell groups: {len(named_populations)} group(s) evaluated across {total} cells"
        if hasattr(self, "preview_filter_status_label"):
            self.set_preview_filter_bar_status(f"{len(named_populations)} group(s) · {total} cells", status)
        if hasattr(self, "analysis_filter_live_status_label"):
            self.analysis_filter_live_status_label.setText(
                f"Preview link: {total} cells | {len(named_populations)} group(s) | table: {data.table_path.name}"
            )
        if hasattr(self, "analysis_export_filtered_csv_button"):
            self.analysis_export_filtered_csv_button.setEnabled(True)
            self.analysis_export_filtered_csv_button.setToolTip(
                f"Export a filtered copy of {data.table_path.name} and a group-membership report, excluding only groups marked for CSV exclusion."
            )
        if hasattr(self, "update_mask_adjustment_status"):
            self.update_mask_adjustment_status()

    @staticmethod
    def _preview_filter_label_key(label: str) -> str:
        # Sidecars retain configured names; fuzzy matching can select another mask.
        return str(label or "").strip()

    def attach_preview_mask_filter_metrics(self, table, label_image, image_def):
        """Use open overlay pixels when saved tables do not contain mask QC metrics."""
        from cellonaut.masks.preview_filter_overlay import attach_live_mask_metrics

        rules = self.parse_cell_qc_limits_text(str(image_def.get("mask_qc_limits", "") or ""))
        if not rules:
            return table
        if image_def.get("mask_qc_intensity_source") != "Cell mask image" and all(
            find_matching_cell_qc_column(table, metric) is not None for metric in rules
        ):
            return table
        model = self.preview_state.tiff_model or {}
        stack = model.get("data")
        labels = list(self.preview_state.current_labels or [])
        from cellonaut.results.artifacts import ArtifactResolver, current_layer_labels

        resolver = ArtifactResolver.load(Path(self.preview_state.file_path or ""))
        artifact_cell_mask_label = ""
        if resolver is not None:
            labels = current_layer_labels({"layer_labels": labels, "layer_keys": self.preview_state.current_layer_keys}, self.get_active_image_definitions())
            try:
                preview_record = resolver.record(Path(self.preview_state.file_path or ""))
                cell_mask_key = str(preview_record.get("cell_mask", "") or "")
                artifact_cell_mask_label = next(
                    (
                        str(definition.get("name", "") or "")
                        for index, definition in enumerate(self.get_active_image_definitions(), 1)
                        if str(definition.get("key", "") or f"image{index}") == cell_mask_key
                    ),
                    "",
                )
            except Exception:
                artifact_cell_mask_label = ""
        roles = list(self.preview_state.current_layer_roles or [])
        if stack is None or getattr(stack, "ndim", 0) != 5 or not labels:
            if image_def.get("mask_qc_intensity_source") == "Cell mask image":
                raise ValueError("Requested Cellpose-source intensity is unavailable in this overlay.")
            return table

        selected_mask = _selected_mask_label(image_def)
        wanted_mask = self._preview_filter_label_key(selected_mask)
        measured_label = self._preview_filter_label_key(str(image_def.get("name", "") or ""))
        intensity_source = str(image_def.get("mask_qc_intensity_source", "Measured image") or "Measured image")
        if intensity_source == "Cell mask image":
            measured_label = self._preview_filter_label_key(
                cellpose_intensity_source_name(
                    image_def,
                    self.get_active_image_definitions()
                    if hasattr(self, "get_active_image_definitions")
                    else [image_def],
                    artifact_cell_mask_label,
                )
            )

        mask_index = next(
            (
                index
                for index, label in enumerate(labels)
                if index < len(roles)
                and "mask" in str(roles[index]).lower()
                and "cellpose" not in str(roles[index]).lower()
                and "filtered" not in str(roles[index]).lower()
                and self._preview_filter_label_key(label) == wanted_mask
            ),
            None,
        )
        intensity_index = next(
            (
                index
                for index, label in enumerate(labels)
                if (index >= len(roles) or str(roles[index]).lower() == "image")
                and self._preview_filter_label_key(label) == measured_label
            ),
            None,
        )
        if mask_index is None or intensity_index is None:
            if intensity_source == "Cell mask image":
                raise ValueError("Requested Cellpose-source intensity is unavailable in this overlay.")
            return table

        z_count = max(1, int(stack.shape[1]))
        page_index = max(0, min(int(self.preview_state.page_index), int(stack.shape[0]) * z_count - 1))
        t_index, z_index = divmod(page_index, z_count)
        return attach_live_mask_metrics(
            table,
            label_image,
            stack[t_index, z_index, mask_index],
            stack[t_index, z_index, intensity_index],
            replace_existing=intensity_source == "Cell mask image",
        )

    # Visibility is presentation-only; retaining the computed labels makes
    # toggling the layer immediate even for large images.
    def on_preview_filter_visibility_changed(self, checked: bool):
        self.preview_state.filter_visible = bool(checked)
        self.update_preview_filter_overlay_item()
        for item in getattr(self, "_preview_population_items", {}).values():
            item.setVisible(bool(checked))
