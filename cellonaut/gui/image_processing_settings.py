"""Image processing settings table widgets."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRegularExpression
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QToolButton, QWidget

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_STEP_DEFINITIONS,
    IMAGE_PROCESSING_BIT_DEPTH_OPTIONS,
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    coerce_bool,
    normalize_image_processing_steps,
)
from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.icons import colored_icon
from cellonaut.gui.widgets import (
    build_processing_step_menu_button,
    MatrixToggleButton,
    NoWheelComboBox,
    make_blocked_processing_value_cell,
    processing_value_from_widget,
)


class OutlierSettingsCell(QWidget):
    """Compact Fiji-style Radius, Threshold, and Which Outliers controls."""

    def __init__(self, value: str, changed) -> None:
        super().__init__()
        self.setProperty("processingRole", "value")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)
        self.radius = QLineEdit()
        self.radius.setPlaceholderText("R px")
        self.radius.setToolTip("Radius of the neighborhood in pixels (Fiji: Radius).")
        self.radius.setAccessibleName("Remove Outliers radius in pixels")
        self.threshold = QLineEdit()
        self.threshold.setPlaceholderText("Δ")
        self.threshold.setToolTip("Minimum difference from the neighborhood median in raw pixel values (Fiji: Threshold).")
        self.threshold.setAccessibleName("Remove Outliers pixel-value threshold")
        numeric = QRegularExpressionValidator(QRegularExpression(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]*)?"), self)
        self.radius.setValidator(numeric)
        self.threshold.setValidator(numeric)
        self.which = NoWheelComboBox()
        self.which.addItems(["Bright", "Dark"])
        self.which.setToolTip("Which Outliers: unusually bright or dark pixels.")
        self.which.setAccessibleName("Remove Outliers polarity")
        for widget in (self.radius, self.threshold):
            widget.setMinimumWidth(46)
            layout.addWidget(widget, 1)
        layout.addWidget(self.which, 1)
        if value:
            parts = [part.strip() for part in value.split(",")]
            if len(parts) == 3:
                self.radius.setText(parts[0])
                self.threshold.setText(parts[1])
                self.which.setCurrentText(parts[2])
        self.radius.textChanged.connect(changed)
        self.threshold.textChanged.connect(changed)
        self.which.currentTextChanged.connect(changed)

    def processing_value(self) -> str:
        radius = self.radius.text().strip()
        threshold = self.threshold.text().strip()
        if not radius and not threshold:
            return ""
        return f"{radius},{threshold},{self.which.currentText()}"


class BitDepthSettingsCell(QWidget):
    """Store Fiji's scale-conversion choice with the Image > Type command."""

    def __init__(self, value: str, params: dict, changed) -> None:
        super().__init__()
        self.setProperty("processingRole", "value")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        self.combo = NoWheelComboBox()
        self.combo.addItem("Inactive", "")
        for option in IMAGE_PROCESSING_BIT_DEPTH_OPTIONS:
            self.combo.addItem(option, option)
        index = self.combo.findData(value)
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        self.combo.setToolTip("Fiji Image > Type conversion. Inactive skips this channel.")
        self.scale = QCheckBox("Scale")
        self.scale.setChecked(coerce_bool(params.get("scale_when_converting", True), True))
        self.scale.setToolTip("Fiji Edit > Options > Conversions > Scale When Converting")
        self.scale.setAccessibleName("Scale when converting bit depth")
        self.setProperty("fijiOptions", {"scale_when_converting": self.scale.isChecked()})
        def update_scaling(checked: bool) -> None:
            self.setProperty("fijiOptions", {"scale_when_converting": checked})
            changed()

        self.scale.toggled.connect(update_scaling)
        self.combo.currentTextChanged.connect(changed)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.scale)

    def processing_value(self) -> str:
        return str(self.combo.currentData() or "")


# Keep image-step menus and labels separate from shared table controls.
class CellonautGuiImageProcessingSettingsMixin(GuiMixin):
    """Edit per-channel image-processing recipes and preview settings."""

    def image_processing_step_definitions_by_type(self) -> dict[str, dict]:
        return {str(step["type"]): dict(step) for step in IMAGE_PROCESSING_STEP_DEFINITIONS}

    def normalized_image_processing_steps_for_definition(self, image_def: dict) -> list[dict]:
        return normalize_image_processing_steps(
            image_def.get("image_processing_steps", []),
            {
                "bg_radii": image_def.get("bg_radii", ""),
            },
        )

    # Keep units beside recipe controls because the same numeric value can mean a
    # radius, percentage, iteration count, or intensity.
    def image_processing_step_unit_text(self, step: dict) -> str:
        units = {
            "bg_radii": "px radius",
            "contrast_saturation": "% saturated",
            "smooth_iterations": "iterations",
            "gaussian_sigma": "sigma px",
            "median_radius": "px radius",
            "bit_depth": "bit depth",
        }
        return units.get(str(step.get("param_key", "") or ""), "")

    def image_processing_step_display_label(self, step: dict) -> str:
        label = str(step.get("label", "") or "")
        unit = self.image_processing_step_unit_text(step)
        return f"{label} ({unit})" if unit else label

    def attach_fiji_step_options(self, widget: QLineEdit, params: dict, names: dict[str, str]) -> None:
        """Keep Fiji's command switches beside the numeric field without widening every row."""
        menu = QMenu(widget)
        options = {key: coerce_bool(params.get(key, False)) for key in names}
        widget.setProperty("fijiOptions", options)
        for key, label in names.items():
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(options[key])
            def update_option(checked: bool, option: str = key) -> None:
                current = dict(widget.property("fijiOptions") or {})
                current[option] = checked
                widget.setProperty("fijiOptions", current)
                self.on_processing_table_changed()

            action.toggled.connect(update_option)
        icon_factory = getattr(self, "shell_icon", None)
        icon = colored_icon("settings", "#888888", 14)
        if callable(icon_factory):
            shell_icon = icon_factory("settings", 14)
            if isinstance(shell_icon, QIcon):
                icon = shell_icon
        menu_action = QAction(icon, "Fiji command options", widget)
        menu_action.setToolTip("Fiji command options")
        menu_action.triggered.connect(
            lambda: menu.popup(widget.mapToGlobal(QPoint(widget.width(), widget.height())))
        )
        widget.addAction(menu_action, QLineEdit.ActionPosition.TrailingPosition)

    # Mirror Fiji's menu hierarchy to help users connect Cellonaut steps with the
    # commands they already know from manual ImageJ workflows.
    def image_processing_step_category_path(self, step_type: str) -> str:
        categories = {
            "smooth": "Process",
            "enhance_contrast": "Process",
            "rolling_ball_background": "Process",
            "gaussian_blur": "Process > Filters",
            "median": "Process > Filters",
            "despeckle": "Process > Noise",
            "remove_outliers": "Process > Noise",
            "apply_lut": "Image > Lookup Tables",
            "bit_depth": "Image > Type",
        }
        return categories.get(str(step_type or ""), "Other")

    def image_processing_step_options(self) -> list[tuple[str, str]]:
        return [
            (str(step["type"]), self.image_processing_step_display_label(dict(step)))
            for step in IMAGE_PROCESSING_STEP_DEFINITIONS
        ]

    # Use a fixed category order rather than alphabetical sorting to preserve the
    # familiar Fiji menu sequence.
    def image_processing_step_category_order(self) -> list[str]:
        return [
            "Process",
            "Process > Filters",
            "Process > Noise",
            "Image > Lookup Tables",
            "Image > Type",
            "Other",
        ]

    def image_processing_step_selector_width(self, widget: QWidget) -> int:
        labels = ["Choose step", *self.image_processing_step_category_order()]
        labels.extend(label for _step_type, label in self.image_processing_step_options())
        max_width = max(widget.fontMetrics().horizontalAdvance(label) for label in labels)
        return max(200, min(230, max_width + 48))

    # A hierarchical tool menu keeps the selector compact while still exposing
    # every operation without a long flat combo box.
    def make_image_processing_step_menu_button(self, step_type: str = "", scope: str | None = None) -> QToolButton:
        definitions = self.image_processing_step_definitions_by_type()
        if scope:
            definitions = {
                key: value
                for key, value in definitions.items()
                if scope in value.get("allowed_scopes", [])
            }
        options = [item for item in self.image_processing_step_options() if item[0] in definitions]
        return build_processing_step_menu_button(
            role="step_type",
            step_type=step_type,
            definitions=definitions,
            options=options,
            categories=self.image_processing_step_category_order(),
            category_for_type=self.image_processing_step_category_path,
            label_for_definition=self.image_processing_step_display_label,
            selector_width=self.image_processing_step_selector_width,
            activate=self.activate_processing_step_selector,
            refresh_style=self.refresh_image_processing_step_selector_style,
        )

    # Read the stable step identifier stored on the menu button rather than its
    # user-facing text, which may include units or later wording changes.
    def image_processing_step_type_from_widget(self, widget, fallback: str = "") -> str:
        if isinstance(widget, QToolButton):
            return str(widget.property("stepType") or fallback or "")
        return str(fallback or "")

    # Empty selector rows are intentionally muted; repolish after property changes
    # because Qt stylesheets do not refresh dynamic properties automatically.
    def refresh_image_processing_step_selector_style(self, widget: QWidget):
        step_type = self.image_processing_step_type_from_widget(widget, "")
        widget.setProperty("mutedState", "false" if step_type else "true")
        widget.setFixedWidth(self.image_processing_step_selector_width(widget))
        if isinstance(widget, QToolButton):
            widget.setToolTip(widget.text())
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    # Some display-oriented operations change pixel values. Warn only when they
    # are placed before measurement, where that choice affects quantitative output.
    def image_processing_step_warns_for_scope(self, step_type: str, scope: str) -> bool:
        definition = self.image_processing_step_definitions_by_type().get(str(step_type or ""), {})
        return (
            str(scope or "") == "Measurement image"
            and not bool(definition.get("measurement_safe", False))
            and bool(step_type)
        )

    def refresh_image_processing_warning_label(self, label: QLabel, step_type: str, scope: str):
        warns = self.image_processing_step_warns_for_scope(step_type, scope)
        label.setVisible(warns)
        label.setProperty("warningActive", "true" if warns else "false")
        label.setToolTip("This step can affect measurement values, including through later processing." if warns else "")

    # A compact symbol preserves table width while its tooltip explains the risk
    # when the selected scope affects measurements.
    def make_image_processing_warning_label(self, step_type: str, scope: str) -> QLabel:
        label = QLabel("\u26a0")
        label.setProperty("processingRole", "warning")
        self.refresh_image_processing_warning_label(label, step_type, scope)
        return label

    # Input validators prevent impossible characters without rejecting temporary
    # states such as a blank field or unfinished decimal during typing.
    def set_image_processing_value_validator(self, widget: QLineEdit, param_key: str) -> None:
        patterns = {
            "bg_radii": r"[0-9.,; \t]*",
            "contrast_saturation": r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]*)?",
            "smooth_iterations": r"[0-9]*",
            "gaussian_sigma": r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]*)?",
            "median_radius": r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]*)?",
        }
        pattern = patterns.get(str(param_key or ""))
        if not pattern:
            return
        validator = QRegularExpressionValidator(QRegularExpression(pattern), widget)
        widget.setValidator(validator)

    # Bit depth has a closed set of commands, while numeric steps need editable
    # text to support blank skips and comma-separated background radii.
    def make_image_processing_value_widget(self, param_key: str, value: str, params: dict | None = None, *, scope: str = ""):
        value = str(value or "")
        params = dict(params or {})
        if param_key == "outlier_settings":
            return OutlierSettingsCell(value, self.on_processing_table_changed)
        if param_key == "bit_depth":
            return BitDepthSettingsCell(value, params, self.on_processing_table_changed)

        widget = QLineEdit(value)
        widget.setPlaceholderText("Inactive")
        widget.setProperty("blankMeansInactive", True)
        self.set_image_processing_value_validator(widget, param_key)
        if param_key == "bg_radii":
            behavior = (
                "Before mask creation, comma-separated radii run sequentially on the same image."
                if scope == IMAGE_PROCESSING_SCOPE_SEGMENTATION
                else "Before measurement, each comma-separated radius produces an independent corrected result."
            )
            widget.setToolTip(f"Background radius in pixels (e.g., 20,50,100). {behavior} Blank skips this channel.")
            self.attach_fiji_step_options(
                widget,
                params,
                {
                    "light_background": "Light background",
                    "sliding_paraboloid": "Sliding paraboloid",
                    "disable_smoothing": "Disable smoothing",
                },
            )
        elif param_key == "contrast_saturation":
            widget.setToolTip(
                "Fiji saturated-pixel percentage. Pixel values change when Normalize or Equalize is selected, "
                "or when a later Apply LUT step is used. "
                "Inactive skips this channel."
            )
            self.attach_fiji_step_options(
                widget,
                params,
                {"normalize": "Normalize pixels", "equalize": "Equalize histogram"},
            )
        elif param_key == "smooth_iterations":
            widget.setToolTip(
                "Number of smoothing passes. "
                "More passes soften noise and edges. Inactive skips this channel."
            )
        elif param_key == "gaussian_sigma":
            widget.setToolTip("Softens noise and edges. Sigma in pixels; inactive skips this channel.")
        elif param_key == "median_radius":
            widget.setToolTip("Removes local noise. Radius in pixels; inactive skips this channel.")
        widget.textChanged.connect(self.on_processing_table_changed)
        return widget

    # Preserve values as text until strict configuration validation so the table
    # can represent blank per-channel skips without inventing numeric defaults.
    def image_processing_value_from_widget(self, widget, fallback: str = "") -> str:
        return processing_value_from_widget(widget, fallback)

    # Parameterless steps use an ON/OFF cell, parameterized steps use their value,
    # and an unchosen row stays visibly blocked until an operation is selected.
    def make_image_processing_value_cell(
        self,
        param_key: str,
        value: str,
        *,
        enabled: bool = True,
        step_type: str = "",
        params: dict | None = None,
        scope: str = "",
    ):
        if not step_type:
            return make_blocked_processing_value_cell("Choose a processing step before setting channel values.")
        if not param_key:
            toggle = MatrixToggleButton(bool(enabled))
            toggle.setProperty("processingRole", "value")
            toggle.setMinimumWidth(58)
            toggle.setToolTip("Apply this step to this channel when ON.")
            toggle.clicked.connect(self.on_processing_table_changed)
            return toggle
        return self.make_image_processing_value_widget(param_key, value, params, scope=scope)

    # For value cells, a blank means skip this channel; parameterless operations
    # carry their enabled state in the matrix toggle instead.
    def image_processing_cell_enabled(self, widget) -> bool:
        if isinstance(widget, MatrixToggleButton):
            return bool(widget.isChecked())
        value = self.image_processing_value_from_widget(widget, "")
        return bool(str(value or "").strip())
