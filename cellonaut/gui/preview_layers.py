"""Preview layer metadata, color, opacity, and list controls."""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
import cellonaut.io.nd2_import as ndi
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
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

from cellonaut.colors import normalize_rgb_hex
from cellonaut.gui.preview_colors import (
    DEFAULT_PREVIEW_COLOR,
    DEFAULT_PREVIEW_LAYER_COLORS,
    NEUTRAL_PREVIEW_COLORS,
    PREVIEW_COLOR_PALETTE,
)
from cellonaut.gui.preview_metadata import is_overlay_preview_sidecar


class PreviewLayerDragHandle(QLabel):
    """Small handle that starts an internal move for its layer row."""

    # Start dragging only from the handle so using other layer controls does not reorder rows.
    def __init__(self, layer_list, parent=None):
        super().__init__("||", parent)
        self.layer_list = layer_list
        self.list_item: QListWidgetItem | None = None
        self._press_pos = QPoint()
        self.setToolTip("Drag to change layer order within this group")
        self.setFixedWidth(16)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    # Respect Qt's drag threshold so minor pointer movement does not reorder rows.
    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.MouseButton.LeftButton:
            return
        distance = (event.position().toPoint() - self._press_pos).manhattanLength()
        if distance < QApplication.startDragDistance():
            return
        if self.list_item is not None:
            self.layer_list.setCurrentItem(self.list_item)
            self.layer_list.startDrag(Qt.DropAction.MoveAction)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)


def preview_layer_group_for_role(role: str) -> str:
    normalized_role = str(role or "image").lower()
    if "adjusted" in normalized_role:
        return "Mask adjustments"
    if "mask" in normalized_role:
        return "Masks"
    return "Images"


def preview_layer_default_rank(role: str, *, has_saved_order: bool = False) -> tuple[int, int]:
    normalized_role = str(role or "image").lower()
    group_rank = {"Images": 0, "Masks": 1, "Mask adjustments": 2}
    group = preview_layer_group_for_role(normalized_role)
    mask_rank = 1 if not has_saved_order and normalized_role == "cellpose_mask" else 0
    return group_rank.get(group, 3), mask_rank


# Layer controls and renderer state share one owner so reordering, visibility,
# color, and opacity changes remain aligned with the underlying channel indices.
class CellonautGuiPreviewLayersMixin:
    # Check both sidecar locations because exports store JSON beside the media or
    # in a sibling JSON folder depending on artifact type.
    def load_overlay_sidecar(self, file_path: str) -> dict:
        from cellonaut.results.artifacts import load_artifact_sidecar

        return load_artifact_sidecar(Path(file_path))

    # Use sidecar labels only when their count matches the normalized channel
    # axis; stale metadata is less trustworthy than neutral generated names.
    def get_tiff_channel_labels(self, model: dict, sidecar: dict) -> list[str]:
        data = model["data"]
        c_count = int(data.shape[2])

        labels = list(sidecar.get("layer_labels", []) or [])
        if labels and len(labels) == c_count:
            return labels

        if c_count == 1:
            return ["Image"]

        default_labels = [f"Channel {i + 1}" for i in range(c_count)]
        if model.get("is_overlay"):
            default_labels[0] = "Base"
        return default_labels

    # Preserve duplicate colors only for generated overlays where matching image
    # and mask layers intentionally share a channel identity.
    def get_tiff_channel_colors(self, model: dict, sidecar: dict) -> list[str]:
        data = model["data"]
        c_count = int(data.shape[2])

        colors = list(sidecar.get("layer_colors", []) or [])
        if is_overlay_preview_sidecar(sidecar) and colors:
            return self.preview_layer_colors(c_count, colors, preserve_duplicates=True)
        return self.unique_preview_layer_colors(c_count, colors)

    # Reuse ND2 color inference so TIFF and ND2 previews present familiar channel
    # names with the same defaults.
    def inferred_channel_colors_from_labels(self, labels: list[str]) -> list[str]:
        return [ndi.infer_channel_color_hex(label, label) for label in labels]

    # Treat only a neutral first layer as a grayscale base; a colored first
    # channel must participate in additive compositing like every other layer.
    def preview_layer_is_grayscale_base(self, index: int, colors: list[str]) -> bool:
        if int(index) != 0:
            return False
        color = self.normalize_preview_color_hex(colors[0] if colors else "")
        return color in NEUTRAL_PREVIEW_COLORS

    def preview_color_palette(self) -> list[tuple[str, str]]:
        return list(PREVIEW_COLOR_PALETTE)

    # Order defaults by visual separation so adjacent generated layers remain
    # distinguishable in dense overlays.
    def preview_default_layer_colors(self) -> list[str]:
        return list(DEFAULT_PREVIEW_LAYER_COLORS)

    # Normalize color codes so letter case does not affect duplicate checks.
    def normalize_preview_color_hex(self, value: str) -> str:
        return normalize_rgb_hex(value)

    # Extend beyond the fixed palette with spaced HSV hues so large stacks still
    # receive deterministic, distinct colors.
    def fallback_preview_color_for_index(self, index: int, used: set[str]) -> str:
        palette = self.preview_default_layer_colors()
        for offset in range(len(palette)):
            candidate = palette[(index + offset) % len(palette)].upper()
            if candidate not in used:
                return candidate

        hue = (index * 0.61803398875) % 1.0
        for offset in range(64):
            rgb = colorsys.hsv_to_rgb((hue + offset * 0.071) % 1.0, 0.85, 1.0)
            candidate = "#{:02X}{:02X}{:02X}".format(
                int(round(rgb[0] * 255)),
                int(round(rgb[1] * 255)),
                int(round(rgb[2] * 255)),
            )
            if candidate not in used:
                return candidate
        return DEFAULT_PREVIEW_COLOR

    # Give a manually changed layer priority so its selected color stays fixed
    # while any conflicting layer is reassigned.
    def unique_preview_layer_colors(
        self,
        count: int,
        preferred: list[str] | None = None,
        priority_index: int | None = None,
    ) -> list[str]:
        preferred = list(preferred or [])
        colors = [""] * max(0, int(count))
        used: set[str] = set()

        order: list[int] = []
        if priority_index is not None and 0 <= priority_index < len(colors):
            order.append(priority_index)
        order.extend(index for index in range(len(colors)) if index not in order)

        for index in order:
            candidate = self.normalize_preview_color_hex(preferred[index]) if index < len(preferred) else ""
            if not candidate or candidate in used:
                candidate = self.fallback_preview_color_for_index(index, used)
            colors[index] = candidate
            used.add(candidate)

        return colors

    # Centralize duplicate-color policy because generated overlays and generic
    # multichannel files require different defaults.
    def preview_layer_colors(
        self,
        count: int,
        preferred: list[str] | None = None,
        *,
        preserve_duplicates: bool | None = None,
    ) -> list[str]:
        if preserve_duplicates is None:
            preserve_duplicates = bool(self.preview_state.preserve_layer_colors)
        preferred = list(preferred or [])
        if not preserve_duplicates:
            return self.unique_preview_layer_colors(count, preferred)

        colors: list[str] = []
        used_for_fallbacks: set[str] = set()
        for index in range(max(0, int(count))):
            candidate = self.normalize_preview_color_hex(preferred[index]) if index < len(preferred) else ""
            if not candidate:
                candidate = self.fallback_preview_color_for_index(index, used_for_fallbacks)
            colors.append(candidate)
            used_for_fallbacks.add(candidate)
        return colors

    def set_preview_color_button_style(self, button: QPushButton, color_hex: str):
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {color_hex};
                border: 1px solid #20242D;
                border-radius: 4px;
                min-width: 20px;
                max-width: 20px;
                min-height: 18px;
                max-height: 18px;
                padding: 0;
            }}
            QPushButton::menu-indicator {{
                image: none;
                width: 0;
            }}
            """
        )
        button.setToolTip(f"Layer color: {color_hex}")

    def ensure_preview_overlay_colors(self, count: int):
        colors = list(self.preview_state.overlay_colors or [])
        self.preview_state.overlay_colors = self.preview_layer_colors(count, colors)

    def ensure_preview_overlay_opacities(self, count: int):
        opacities = list(self.preview_state.overlay_opacities or [])
        while len(opacities) < count:
            opacities.append(1.0)
        self.preview_state.overlay_opacities = [max(0.0, min(1.0, float(value))) for value in opacities[:count]]

    # Update scene items in place when possible so recoloring keeps zoom, page,
    # and layer order unchanged.
    def set_preview_channel_color(self, index: int, color_hex: str):
        colors = list(self.preview_state.overlay_colors or [])
        if not 0 <= index < len(colors):
            return

        colors[index] = self.normalize_preview_color_hex(color_hex) or color_hex
        # User-selected colors may be shared by any number of layers.
        self.preview_state.preserve_layer_colors = True
        self.preview_state.overlay_colors = colors

        buttons = getattr(self, "_preview_color_buttons", []) or []
        for button_index, button in enumerate(buttons):
            if 0 <= button_index < len(colors):
                self.set_preview_color_button_style(button, colors[button_index])
        if hasattr(self, "preview_layer_list"):
            self.update_preview_mask_selection_outline(self.preview_state.selected_layer_index)

        if self.should_use_preview_layer_items():
            for item_index in range(len(colors)):
                self.update_preview_layer_item_color(item_index)
            return

        self.on_preview_layer_visibility_changed()

    # Update the layer directly when possible; otherwise schedule a combined-image redraw.
    def set_preview_channel_opacity(self, index: int, value: int):
        opacities = list(self.preview_state.overlay_opacities or [])
        if not 0 <= index < len(opacities):
            return

        opacity = max(0.0, min(1.0, float(value) / 100.0))
        opacities[index] = opacity
        self.preview_state.overlay_opacities = opacities

        sliders = getattr(self, "_preview_opacity_sliders", []) or []
        if 0 <= index < len(sliders):
            sliders[index].setToolTip(f"Opacity: {int(round(opacity * 100))}%")

        if self.update_preview_layer_item_opacity(index, opacity):
            return

        self.on_preview_layer_visibility_changed()

    def make_preview_color_button(self, index: int, label: str) -> QPushButton:
        colors = list(self.preview_state.overlay_colors or [])
        color_hex = colors[index] if 0 <= index < len(colors) else DEFAULT_PREVIEW_COLOR

        button = QPushButton()
        button.setObjectName("PreviewLayerColorButton")
        button.setAccessibleName(f"Color for {label}")
        button.setToolTip(f"Change color for {label}")
        self.set_preview_color_button_style(button, color_hex)

        menu = QMenu(button)
        for color_name, palette_color in self.preview_color_palette():
            swatch = QPixmap(16, 16)
            swatch.fill(Qt.GlobalColor.transparent)
            painter = QPainter(swatch)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(palette_color))
                painter.drawRoundedRect(3, 2, 12, 12, 3, 3)
            finally:
                painter.end()
            action = menu.addAction(QIcon(swatch), color_name)
            action.setIconVisibleInMenu(True)
            action.triggered.connect(
                lambda _checked=False, idx=index, value=palette_color: self.set_preview_channel_color(idx, value)
            )
        button.setMenu(menu)
        return button

    # Store percentages in Qt while retaining normalized floats in renderer state.
    def make_preview_opacity_slider(self, index: int, label: str) -> QSlider:
        opacities = list(self.preview_state.overlay_opacities or [])
        opacity = opacities[index] if 0 <= index < len(opacities) else 1.0

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setObjectName("PreviewLayerOpacitySlider")
        slider.setAccessibleName(f"Opacity for {label}")
        slider.setRange(0, 100)
        slider.setSingleStep(5)
        slider.setPageStep(10)
        slider.setFixedWidth(84)
        slider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        slider.setValue(int(round(max(0.0, min(1.0, opacity)) * 100)))
        slider.setToolTip(f"Opacity: {slider.value()}%")
        slider.valueChanged.connect(lambda value, s=slider: s.setToolTip(f"Opacity: {int(value)}%"))
        slider.valueChanged.connect(lambda value, idx=index: self.set_preview_channel_opacity(idx, value))
        return slider

    def get_overlay_layer_visibility(self) -> list[bool]:
        return list(self.preview_state.layer_visibility)

    def get_overlay_layer_opacities(self) -> list[float]:
        return list(self.preview_state.overlay_opacities)

    def set_preview_channel_visible(self, index: int, visible: bool) -> None:
        self.preview_state.layer_visibility[index] = bool(visible)
        self.on_preview_layer_visibility_changed()

    # Include role and occurrence because image and mask layers can share both a
    # biological label and source key.
    def preview_layer_identity(self, index: int) -> str:
        labels = list(self.preview_state.current_labels or [])
        roles = list(self.preview_state.current_layer_roles or [])
        keys = list(self.preview_state.current_layer_keys or [])
        label = str(labels[index] or "") if 0 <= index < len(labels) else ""
        role = str(roles[index] or "image") if 0 <= index < len(roles) else "image"
        key = str(keys[index] or "") if 0 <= index < len(keys) else ""
        base = key or label or f"__layer_index_{index}"
        occurrence = 0
        for prior_index in range(index):
            prior_key = str(keys[prior_index] or "") if prior_index < len(keys) else ""
            prior_label = str(labels[prior_index] or "") if prior_index < len(labels) else ""
            prior_role = str(roles[prior_index] or "image") if prior_index < len(roles) else "image"
            prior_base = prior_key or prior_label or f"__layer_index_{prior_index}"
            if prior_base == base and prior_role == role:
                occurrence += 1
        return f"{base}::{role}::{occurrence}"

    # Match saved settings to layers even when new layers are inserted.
    def capture_preview_layer_control_state(self) -> dict:
        count = len(self.preview_state.current_labels or [])
        visibility = self.get_overlay_layer_visibility()
        opacities = self.get_overlay_layer_opacities()
        colors = list(self.preview_state.overlay_colors or [])
        order = self.current_preview_layer_order()

        return {
            "visibility": {
                self.preview_layer_identity(index): bool(visibility[index])
                for index in range(min(count, len(visibility)))
            },
            "opacities": {
                self.preview_layer_identity(index): float(opacities[index])
                for index in range(min(count, len(opacities)))
            },
            "colors": {
                self.preview_layer_identity(index): str(colors[index]) for index in range(min(count, len(colors)))
            },
            "order": [self.preview_layer_identity(index) for index in order if 0 <= int(index) < count],
        }

    # Clear widgets and renderer collections together to prevent stale indices
    # from addressing a newly loaded file.
    def clear_preview_layer_controls(self, *, update_mask_targets: bool = True):
        self.clear_preview_mask_selection_outline()
        if hasattr(self, "preview_layer_list"):
            self.preview_layer_list.clear()

        self._preview_overlay_checks: list[QCheckBox] = []
        self._preview_color_buttons: list[QPushButton] = []
        self._preview_opacity_sliders: list[QSlider] = []
        self.preview_state.current_labels = []
        self.preview_state.current_layer_roles = []
        self.preview_state.current_layer_keys = []
        self.preview_state.layer_order = []
        self.preview_state.layer_visibility = []
        self.preview_state.selected_layer_index = None
        self.set_preview_layer_panel_visible(False)
        self.clear_preview_layer_items()
        if update_mask_targets and hasattr(self, "update_mask_adjust_target_options"):
            self.update_mask_adjust_target_options()

    # Build rows directly in saved order because moving them afterward can
    # desynchronize list items from source channel indices.
    def set_preview_layer_controls(self, labels: list[str], preserved_state: dict | None = None):
        roles = list(self.preview_state.current_layer_roles or [])
        keys = list(self.preview_state.current_layer_keys or [])
        self.clear_preview_layer_controls(update_mask_targets=False)
        self.preview_state.current_layer_roles = roles
        self.preview_state.current_layer_keys = keys

        if not labels or len(labels) <= 1:
            self.set_preview_layer_panel_visible(False)
            return

        self.preview_state.current_labels = list(labels)
        self.ensure_preview_overlay_colors(len(labels))
        self.ensure_preview_overlay_opacities(len(labels))

        state = dict(preserved_state or {})
        state_visibility = dict(state.get("visibility") or {})
        state_colors = dict(state.get("colors") or {})
        state_opacities = dict(state.get("opacities") or {})
        for index in range(len(labels)):
            identity = self.preview_layer_identity(index)
            if identity in state_colors:
                self.preview_state.overlay_colors[index] = str(state_colors[identity])
            if identity in state_opacities:
                self.preview_state.overlay_opacities[index] = max(
                    0.0,
                    min(1.0, float(state_opacities[identity])),
                )

        identities = [self.preview_layer_identity(index) for index in range(len(labels))]
        identity_to_index = {identity: index for index, identity in enumerate(identities)}
        order_from_state: list[int] = []
        seen_indices: set[int] = set()
        for identity in list(state.get("order") or []):
            mapped_index = identity_to_index.get(identity)
            if mapped_index is None or mapped_index in seen_indices:
                continue
            order_from_state.append(mapped_index)
            seen_indices.add(mapped_index)
        display_order = order_from_state + [index for index in range(len(labels)) if index not in set(order_from_state)]
        def default_layer_rank(index: int) -> tuple[int, int]:
            role = (
                str(self.preview_state.current_layer_roles[index]).lower()
                if index < len(self.preview_state.current_layer_roles)
                else "image"
            )
            return preview_layer_default_rank(role, has_saved_order=bool(order_from_state))

        display_order.sort(key=default_layer_rank)
        self.preview_state.layer_order = list(display_order)
        self.preview_state.layer_visibility = [
            bool(state_visibility.get(self.preview_layer_identity(index), True)) for index in range(len(labels))
        ]

        overlay_checks: dict[int, QCheckBox] = {}
        color_buttons: dict[int, QPushButton] = {}
        opacity_sliders: dict[int, QSlider] = {}
        previous_group = ""
        for index in display_order:
            label = labels[index]
            layer_control = QWidget()
            layer_control.setProperty("uiRole", "previewLayerRow")
            layer_control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layer_outer_layout = QVBoxLayout(layer_control)
            layer_outer_layout.setContentsMargins(4, 4, 4, 4)
            layer_outer_layout.setSpacing(4)

            role = str(self.preview_state.current_layer_roles[index] or "image") if index < len(self.preview_state.current_layer_roles) else "image"
            group = preview_layer_group_for_role(role)
            starts_group = group != previous_group
            if starts_group:
                group_label = QLabel(group.upper())
                group_label.setProperty("uiRole", "previewLayerGroup")
                layer_outer_layout.addWidget(group_label)
                previous_group = group

            layer_layout = QHBoxLayout()
            layer_layout.setContentsMargins(0, 0, 0, 0)
            layer_layout.setSpacing(8)

            cb = QCheckBox()
            cb.setToolTip(str(label))
            cb.setFixedWidth(22)
            cb.setChecked(self.preview_state.layer_visibility[index])
            cb.toggled.connect(lambda visible, idx=index: self.set_preview_channel_visible(idx, visible))
            color_button = self.make_preview_color_button(index, label)
            opacity_slider = self.make_preview_opacity_slider(index, label)
            display_label = self.preview_layer_display_label(index, label)
            drag_handle = PreviewLayerDragHandle(self.preview_layer_list)
            label_widget = QLabel(display_label)
            label_widget.setProperty("uiRole", "previewLayerName")
            label_widget.setToolTip(display_label)
            label_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            label_widget.setMinimumWidth(0)
            label_widget.setMaximumWidth(108)
            label_widget.setWordWrap(True)
            text_metrics = label_widget.fontMetrics()
            wrapped_height = text_metrics.boundingRect(
                QRect(0, 0, 108, 1000),
                int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                display_label,
            ).height()
            row_height = (76 if starts_group else 60) + max(0, wrapped_height - text_metrics.lineSpacing())
            layer_control.setMinimumHeight(row_height)

            opacity_group = QWidget()
            opacity_group.setProperty("uiRole", "previewLayerOpacity")
            opacity_layout = QHBoxLayout(opacity_group)
            opacity_layout.setContentsMargins(0, 0, 0, 0)
            opacity_layout.setSpacing(0)
            opacity_layout.addStretch(1)
            opacity_layout.addWidget(opacity_slider)

            layer_layout.addWidget(drag_handle)
            layer_layout.addWidget(cb)
            layer_layout.addWidget(color_button)
            layer_layout.addWidget(label_widget, 1)
            layer_outer_layout.addLayout(layer_layout)
            layer_outer_layout.addWidget(opacity_group)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setData(Qt.ItemDataRole.UserRole + 1, group)
            item.setSizeHint(QSize(0, row_height))
            self.preview_layer_list.addItem(item)
            self.preview_layer_list.setItemWidget(item, layer_control)
            drag_handle.list_item = item

            overlay_checks[index] = cb
            color_buttons[index] = color_button
            opacity_sliders[index] = opacity_slider

        self._preview_overlay_checks = [overlay_checks[index] for index in range(len(labels))]
        self._preview_color_buttons = [color_buttons[index] for index in range(len(labels))]
        self._preview_opacity_sliders = [opacity_sliders[index] for index in range(len(labels))]

        self.preview_layer_list.setCurrentRow(0)
        self.set_preview_layer_panel_visible(True)
        self.on_preview_layer_order_changed()
        self.update_mask_adjust_target_options()

    # Pad optional sidecar metadata to channel count before index-based use.
    def set_preview_layer_metadata(self, sidecar: dict, count: int):
        roles = list((sidecar or {}).get("layer_roles", []) or [])
        keys = list((sidecar or {}).get("layer_keys", []) or [])
        self.preview_state.current_layer_roles = [
            str(roles[index]) if index < len(roles) else "image" for index in range(count)
        ]
        self.preview_state.current_layer_keys = [str(keys[index]) if index < len(keys) else "" for index in range(count)]

    def preview_layer_display_label(self, index: int, label: str) -> str:
        """Categories already communicate layer type, so keep names concise."""
        return str(label)

    def clear_preview_mask_selection_outline(self) -> None:
        item = getattr(self, "_preview_mask_selection_item", None)
        if item is not None:
            try:
                if item.scene() is not None:
                    item.scene().removeItem(item)
            except RuntimeError:
                pass
        self._preview_mask_selection_item = None

    def selected_mask_outline_color(self, source_index: int) -> tuple[int, int, int]:
        colors = list(self.preview_state.overlay_colors or [])
        color = QColor(colors[source_index] if source_index < len(colors) else "")
        hue = color.hsvHue() if color.isValid() else -1
        uses_red = color.isValid() and color.hsvSaturationF() >= 0.5 and (hue <= 20 or hue >= 340)
        return (42, 130, 218) if uses_red else (255, 55, 55)

    def update_preview_mask_selection_outline(self, source_index: int | None = None) -> None:
        """Draw a temporary contour around the selected mask objects."""
        self.clear_preview_mask_selection_outline()
        model = self.preview_state.tiff_model
        if model is None or source_index is None:
            return
        roles = list(self.preview_state.current_layer_roles or [])
        role = str(roles[source_index] if 0 <= source_index < len(roles) else "").lower()
        data = np.asarray(model.get("data"))
        if "mask" not in role or data.ndim != 5 or not 0 <= source_index < data.shape[2]:
            return

        z_count = int(data.shape[1])
        page_index = max(0, min(int(self.preview_state.page_index), data.shape[0] * z_count - 1))
        values = np.asarray(data[page_index // z_count, page_index % z_count, source_index])
        if values.ndim != 2:
            return
        foreground = values != 0
        if not np.any(foreground):
            return

        # Grow the mask twice, then subtract the mask itself. This produces an
        # exact two-pixel exterior band without tinting any segmented pixels.
        expanded = foreground.copy()
        for _ in range(2):
            padded = np.pad(expanded, 1, mode="constant", constant_values=False)
            expanded = np.zeros_like(foreground, dtype=bool)
            for y_offset in range(3):
                for x_offset in range(3):
                    expanded |= padded[
                        y_offset : y_offset + foreground.shape[0],
                        x_offset : x_offset + foreground.shape[1],
                    ]
        outline = expanded & ~foreground
        red, green, blue = self.selected_mask_outline_color(source_index)
        height, width = values.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = red
        rgba[..., 1] = green
        rgba[..., 2] = blue
        rgba[..., 3] = np.where(outline, 255, 0).astype(np.uint8)
        image = QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()
        item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        item.setZValue(20_000)
        self.preview_scene.addItem(item)
        self._preview_mask_selection_item = item

    def on_preview_layer_selection_changed(self, row: int) -> None:
        source_index = None
        if 0 <= row < self.preview_layer_list.count():
            selected_item = self.preview_layer_list.item(row)
            if str(selected_item.data(Qt.ItemDataRole.UserRole + 2) or "") != "cell_group":
                source_index = int(selected_item.data(Qt.ItemDataRole.UserRole))
        self.preview_state.selected_layer_index = source_index
        self.update_preview_mask_selection_outline(source_index)
        if source_index is None or not hasattr(self, "mask_adjust_target_combo"):
            return
        roles = list(self.preview_state.current_layer_roles or [])
        role = str(roles[source_index] if source_index < len(roles) else "").lower()
        if role != "weka_mask":
            return
        combo_index = self.mask_adjust_target_combo.findData(f"weka_layer:{source_index}")
        if combo_index >= 0 and combo_index != self.mask_adjust_target_combo.currentIndex():
            self.mask_adjust_target_combo.setCurrentIndex(combo_index)

    def current_preview_layer_order(self) -> list[int]:
        return list(self.preview_state.layer_order)

    def on_preview_layer_order_changed(self, *_args):
        self.preview_state.layer_order = [
            int(self.preview_layer_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.preview_layer_list.count())
            if str(self.preview_layer_list.item(row).data(Qt.ItemDataRole.UserRole + 2) or "") != "cell_group"
        ]
        items = getattr(self, "_preview_layer_items", []) or []
        for z_value, source_index in enumerate(self.preview_state.layer_order):
            if 0 <= source_index < len(items):
                items[source_index].setZValue(z_value)
        population_items = dict(getattr(self, "_preview_population_items", {}) or {})
        population_z = 10_000
        for row in range(self.preview_layer_list.count()):
            list_item = self.preview_layer_list.item(row)
            if str(list_item.data(Qt.ItemDataRole.UserRole + 2) or "") != "cell_group":
                continue
            key = str(list_item.data(Qt.ItemDataRole.UserRole + 3) or "")
            self.preview_state.group_layer_display_state.setdefault(key, {})["row"] = row
            graphics_item = population_items.get(key)
            if graphics_item is not None:
                graphics_item.setZValue(population_z)
                population_z += 1

    def on_preview_composite_mode_changed(self, checked: bool):
        self.preview_state.composite_mode = bool(checked)
        self.on_preview_layer_visibility_changed()
