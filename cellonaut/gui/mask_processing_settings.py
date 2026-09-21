"""Mask processing settings table widgets."""

from __future__ import annotations

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QToolButton, QWidget

from cellonaut.config.defaults import (
    MASK_PROCESSING_STEP_DEFINITIONS,
    normalize_mask_processing_steps,
)
from cellonaut.config.relationships import image_produces_mask
from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.widgets import (
    MatrixToggleButton,
    build_processing_step_menu_button,
    make_blocked_processing_value_cell,
    processing_value_from_widget,
)


class FijiBinarySettingsCell(QWidget):
    """Use Fiji's Binary Options names for erosion, dilation, open, and close."""

    def __init__(self, value: str, changed) -> None:
        super().__init__()
        self.setProperty("processingRole", "value")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        parts = [part.strip() for part in str(value or "1,1,false").split(",")]
        if len(parts) != 3:
            parts = ["1", "1", "false"]
        self.iterations = QLineEdit(parts[0])
        self.iterations.setPlaceholderText("Iterations")
        self.iterations.setToolTip("Fiji Binary Options: Iterations (1-100).")
        self.count = QLineEdit(parts[1])
        self.count.setPlaceholderText("Count")
        self.count.setToolTip("Fiji Binary Options: Count (1-8).")
        validator = QRegularExpressionValidator(QRegularExpression(r"[0-9]*"), self)
        for edit in (self.iterations, self.count):
            edit.setValidator(validator)
            edit.textChanged.connect(changed)
            layout.addWidget(edit, 1)
        self.pad_edges = QCheckBox("Pad edges")
        self.pad_edges.setChecked(parts[2].lower() == "true")
        self.pad_edges.setToolTip("Fiji Binary Options: Pad edges when eroding.")
        self.pad_edges.toggled.connect(changed)
        layout.addWidget(self.pad_edges)

    def processing_value(self) -> str:
        return f"{self.iterations.text().strip()},{self.count.text().strip()},{str(self.pad_edges.isChecked()).lower()}"


class FijiParticleSettingsCell(QWidget):
    """Expose the mask-changing Analyze Particles options without result-window controls."""

    def __init__(self, value: str, changed) -> None:
        super().__init__()
        self.setProperty("processingRole", "value")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        parts = [part.strip() for part in str(value or "0-Infinity,0-1,false,false").split(",")]
        if len(parts) != 4:
            parts = ["0-Infinity", "0-1", "false", "false"]
        self.size_range = QLineEdit(parts[0])
        self.size_range.setPlaceholderText("Size")
        self.size_range.setToolTip("Fiji Analyze Particles: area in pixels squared, minimum-maximum (for example 20-Infinity).")
        self.circularity_range = QLineEdit(parts[1])
        self.circularity_range.setPlaceholderText("Circ.")
        self.circularity_range.setToolTip("Fiji Analyze Particles: Circularity, minimum-maximum (0-1).")
        for edit in (self.size_range, self.circularity_range):
            edit.textChanged.connect(changed)
            layout.addWidget(edit, 1)
        self.options = QToolButton()
        self.options.setText("Options")
        self.options.setToolTip("Fiji Analyze Particles options")
        self.options.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.options)
        self.exclude_edges = menu.addAction("Exclude on Edges")
        self.include_holes = menu.addAction("Include Holes")
        for action, checked in ((self.exclude_edges, parts[2]), (self.include_holes, parts[3])):
            action.setCheckable(True)
            action.setChecked(checked.lower() == "true")
            action.toggled.connect(changed)
        self.options.setMenu(menu)
        layout.addWidget(self.options)

    def processing_value(self) -> str:
        return ",".join((
            self.size_range.text().strip(), self.circularity_range.text().strip(),
            str(self.exclude_edges.isChecked()).lower(), str(self.include_holes.isChecked()).lower(),
        ))


class FijiTranslateSettingsCell(QWidget):
    """Show Fiji Translate's X and Y offsets together for the current mask."""

    def __init__(self, value: str, changed) -> None:
        super().__init__()
        self.setProperty("processingRole", "value")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        parts = [part.strip() for part in str(value or "0,0").split(",")]
        if len(parts) != 2:
            parts = ["0", "0"]
        self.x_offset = QLineEdit(parts[0])
        self.y_offset = QLineEdit(parts[1])
        validator = QRegularExpressionValidator(
            QRegularExpression(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]*)?"), self
        )
        for edit, axis in ((self.x_offset, "X"), (self.y_offset, "Y")):
            edit.setPlaceholderText(axis)
            edit.setAccessibleName(f"Translate {axis} offset in pixels")
            edit.setToolTip(f"Fiji Image > Transform > Translate: {axis} offset in pixels.")
            edit.setValidator(validator)
            edit.textChanged.connect(changed)
            layout.addWidget(edit, 1)

    def processing_value(self) -> str:
        return f"{self.x_offset.text().strip()},{self.y_offset.text().strip()}"


# Keep mask-specific step choices separate from shared table controls.
class CellonautGuiMaskProcessingSettingsMixin(GuiMixin):
    """Edit shared mask-processing recipes and per-mask values."""

    def mask_settings_step_definitions_by_type(self) -> dict[str, dict]:
        return {str(step["type"]): dict(step) for step in MASK_PROCESSING_STEP_DEFINITIONS}

    # Normalize here so every mask column is rendered from the same recipe shape.
    def normalized_mask_processing_steps_for_definition(self, image_def: dict) -> list[dict]:
        return normalize_mask_processing_steps(image_def.get("mask_processing_steps"))

    def mask_settings_step_unit_text(self, step: dict) -> str:
        units = {
            "binary_settings": "Fiji Binary Options",
            "particle_settings": "Fiji Analyze Particles",
            "translate_offsets": "X, Y px",
        }
        return units.get(str(step.get("param_key", "") or ""), "")

    def mask_settings_step_label(self, step: dict) -> str:
        label = str(step.get("label", "") or "")
        unit = self.mask_settings_step_unit_text(step)
        return f"{label} ({unit})" if unit else label

    # Mirror Fiji's menu hierarchy to make familiar operations easier to find.
    def mask_settings_step_category_path(self, step_type: str) -> str:
        categories = {
            "binary_erode": "Process > Binary",
            "binary_dilate": "Process > Binary",
            "binary_open": "Process > Binary",
            "binary_close": "Process > Binary",
            "binary_fill_holes": "Process > Binary",
            "binary_watershed": "Process > Binary",
            "binary_outline": "Process > Binary",
            "binary_skeletonize": "Process > Binary",
            "analyze_particles": "Analyze > Analyze Particles",
            "analyze_skeleton": "Plugins > Skeleton",
            "translate": "Image > Transform",
        }
        return categories.get(str(step_type or ""), "Other")

    def mask_settings_step_options(self) -> list[tuple[str, str]]:
        return [
            (str(step["type"]), self.mask_settings_step_label(dict(step)))
            for step in MASK_PROCESSING_STEP_DEFINITIONS
        ]

    def mask_settings_step_category_order(self) -> list[str]:
        return [
            "Image > Transform",
            "Process > Binary",
            "Analyze > Analyze Particles",
            "Plugins > Skeleton",
            "Other",
        ]

    def mask_settings_step_selector_width(self, widget: QWidget) -> int:
        labels = ["Choose step", *self.mask_settings_step_category_order()]
        labels.extend(label for _step_type, label in self.mask_settings_step_options())
        max_width = max(widget.fontMetrics().horizontalAdvance(label) for label in labels)
        return max(240, min(280, max_width + 48))

    def make_mask_settings_step_menu_button(self, step_type: str = "") -> QToolButton:
        definitions = self.mask_settings_step_definitions_by_type()
        return build_processing_step_menu_button(
            role="mask_step_type",
            step_type=step_type,
            definitions=definitions,
            options=self.mask_settings_step_options(),
            categories=self.mask_settings_step_category_order(),
            category_for_type=self.mask_settings_step_category_path,
            label_for_definition=self.mask_settings_step_label,
            selector_width=self.mask_settings_step_selector_width,
            activate=self.activate_processing_step_selector,
            refresh_style=self.refresh_mask_settings_step_selector_style,
        )

    # Read the stable key from the selector rather than persisting translated text.
    def mask_settings_step_type_from_widget(self, widget, fallback: str = "") -> str:
        if isinstance(widget, QToolButton):
            return str(widget.property("stepType") or fallback or "")
        if isinstance(widget, QComboBox):
            return str(widget.currentData() or fallback or "")
        return str(fallback or "")

    # Muted placeholder styling makes incomplete recipe rows visible at a glance.
    def refresh_mask_settings_step_selector_style(self, widget: QWidget):
        step_type = self.mask_settings_step_type_from_widget(widget, "")
        widget.setProperty("mutedState", "false" if step_type else "true")
        widget.setFixedWidth(self.mask_settings_step_selector_width(widget))
        if isinstance(widget, QToolButton):
            widget.setToolTip(widget.text())
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    # Group row actions in one header cell so dragging never displaces mask values.
    def make_mask_settings_step_header(self, step: dict, step_index: int, *, removable: bool) -> QWidget:
        widget = self.make_processing_header_cell()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        handle = QLabel("|||")
        handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        handle.setCursor(Qt.CursorShape.SizeAllCursor)
        handle.setProperty("processingRole", "drag_handle")
        handle.setProperty("uiRole", "tableHeaderCellLabel")
        handle.setFixedWidth(26)
        handle.setToolTip("Drag this sidebar up or down to reorder mask processing steps.")

        remove_btn = self.make_processing_remove_button(
            step_index,
            (
                "Choose an operation before removing this row."
                if not removable
                else "Remove this mask processing step."
            ),
            lambda _checked=False, sidx=step_index: self.remove_mask_processing_step(sidx),
            removable=removable,
        )

        step_button = self.make_mask_settings_step_menu_button(str(step.get("type", "") or ""))

        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(remove_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(handle, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(step_button, 1, Qt.AlignmentFlag.AlignCenter)
        return widget

    def make_mask_settings_toggle_cell(self, checked: bool, *, enabled: bool, tooltip: str) -> MatrixToggleButton:
        button = MatrixToggleButton(bool(checked))
        button.setEnabled(bool(enabled))
        button.setMinimumWidth(58)
        button.setToolTip(tooltip)
        button.setProperty("processingRole", "value")
        button.clicked.connect(self.on_processing_table_changed)
        self.set_widget_muted(button, not button.isEnabled())
        return button

    # Build controls by step type because valid values depend on both operation and mask source.
    def make_mask_settings_value_cell(
        self, step_type: str, image_def: dict, value: str | None = None, enabled: bool = True
    ):
        has_classifier = image_produces_mask(image_def)

        if not step_type:
            return make_blocked_processing_value_cell("Choose a mask processing step before setting values.")

        if step_type in {"binary_erode", "binary_dilate", "binary_open", "binary_close"}:
            return FijiBinarySettingsCell(str(value or "1,1,false"), self.on_processing_table_changed)

        if step_type in {"binary_fill_holes", "binary_watershed", "binary_outline", "binary_skeletonize"}:
            return self.make_mask_settings_toggle_cell(
                bool(enabled), enabled=has_classifier,
                tooltip="Run Fiji Process > Binary on this mask (white objects on black background).",
            )

        if step_type == "analyze_particles":
            return FijiParticleSettingsCell(
                str(value or "0-Infinity,0-1,false,false"), self.on_processing_table_changed
            )

        if step_type == "translate":
            return FijiTranslateSettingsCell(str(value or "0,0"), self.on_processing_table_changed)

        if step_type == "analyze_skeleton":
            checked = bool(enabled)
            if value is not None:
                checked = str(value).strip().lower() in {"1", "true", "yes", "on"}
            return self.make_mask_settings_toggle_cell(
                checked,
                enabled=has_classifier,
                tooltip=(
                    "Skeletonize a copy of the current mask without changing it, and export "
                    "whole-mask skeleton, branch, endpoint, and junction counts."
                ),
            )

        blocked = QLineEdit("")
        blocked.setEnabled(False)
        return blocked

    # Read recipe control values in the text format used by presets.
    def mask_settings_value_from_widget(self, widget, fallback: str = "") -> str:
        return processing_value_from_widget(widget, fallback)

    # Treat disabled controls as inactive steps without discarding their displayed value.
    def mask_settings_cell_enabled(self, widget) -> bool:
        if isinstance(widget, QPushButton) and widget.isCheckable():
            return bool(widget.isChecked())
        if isinstance(widget, QLineEdit):
            return bool(widget.isEnabled() and widget.text().strip())
        return True
