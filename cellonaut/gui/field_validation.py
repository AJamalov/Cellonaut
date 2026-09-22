"""Widget-level validation shared by Cellonaut GUI workflows.

This mixin owns visual validation state and field/table checks.  Run
readiness, setup reporting, and user confirmation remain in ``validation`` so
widget mechanics do not obscure pipeline decisions.
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from cellonaut.config.defaults import MASK_SOURCE_MODE_COMBINED
from cellonaut.config.processing_steps import (
    IMAGE_PROCESSING_PARAM_SPECS,
    MASK_PROCESSING_STEP_SPECS,
    ProcessingStepSpec,
    parse_binary_settings,
    parse_particle_settings,
    parse_outlier_settings,
    parse_translate_offsets,
)
from cellonaut.gui.mixin import GuiMixin
from cellonaut.io.path_keys import input_path_key
from cellonaut.pipeline.roi_defs import probability_class_indices_are_valid
from cellonaut.pipeline.validation import output_is_inside_input


class CellonautGuiFieldValidationMixin(GuiMixin):
    """Provide reusable validation methods without owning run decisions."""

    # Preserve the original tooltip so repeated validation never accumulates stale error messages.
    def set_widget_validation_state(self, widget, state: str = "", message: str = ""):
        if widget is None:
            return

        if widget.property("_base_tooltip") is None:
            widget.setProperty("_base_tooltip", widget.toolTip() or "")

        base_tooltip = str(widget.property("_base_tooltip") or "")
        widget.setProperty("validationState", state)

        if state == "invalid" and message:
            if base_tooltip:
                widget.setToolTip(f"{base_tooltip}\n\nValidation: {message}")
            else:
                widget.setToolTip(f"Validation: {message}")
        else:
            widget.setToolTip(base_tooltip)

        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)
        widget.update()

    # One numeric validator keeps empty, finite, and range rules consistent across processing tables.
    def validate_number_field(
        self,
        widget,
        *,
        field_name: str,
        integer: bool = False,
        minimum: float | None = None,
        maximum: float | None = None,
        allow_blank: bool = False,
    ) -> bool:
        if widget is None:
            return True

        if hasattr(widget, "property") and widget.property("qcLimitsText") is not None:
            text = str(widget.property("qcLimitsText") or "").strip()
        elif hasattr(widget, "text"):
            text = widget.text().strip()
        else:
            return True

        if not text:
            if allow_blank:
                self.set_widget_validation_state(widget, "")
                return True
            self.set_widget_validation_state(widget, "invalid", f"{field_name} cannot be empty.")
            return False

        try:
            value = int(text) if integer else float(text)
        except (TypeError, ValueError, OverflowError):
            kind = "integer" if integer else "number"
            self.set_widget_validation_state(widget, "invalid", f"{field_name} must be a valid {kind}.")
            return False

        if not math.isfinite(float(value)):
            self.set_widget_validation_state(widget, "invalid", f"{field_name} must be a finite number.")
            return False

        if minimum is not None and value < minimum:
            self.set_widget_validation_state(widget, "invalid", f"{field_name} must be >= {minimum}.")
            return False

        if maximum is not None and value > maximum:
            self.set_widget_validation_state(widget, "invalid", f"{field_name} must be <= {maximum}.")
            return False

        self.set_widget_validation_state(widget, "")
        return True

    # Processing recipes accept comma-separated values, so validate each entry before configuration conversion.
    def validate_csv_number_field(
        self,
        widget,
        *,
        field_name: str,
        minimum: float | None = None,
        allow_blank: bool = True,
    ) -> bool:
        if widget is None:
            return True

        text = widget.text().strip()

        if not text:
            if allow_blank:
                self.set_widget_validation_state(widget, "")
                return True
            self.set_widget_validation_state(widget, "invalid", f"{field_name} cannot be empty.")
            return False

        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        if not parts:
            if allow_blank:
                self.set_widget_validation_state(widget, "")
                return True
            self.set_widget_validation_state(widget, "invalid", f"{field_name} cannot be empty.")
            return False

        for part in parts:
            try:
                value = float(part)
            except (TypeError, ValueError, OverflowError):
                self.set_widget_validation_state(
                    widget,
                    "invalid",
                    f"{field_name} must contain only numbers separated by commas.",
                )
                return False

            if not math.isfinite(value):
                self.set_widget_validation_state(
                    widget,
                    "invalid",
                    f"{field_name} must contain only finite numbers.",
                )
                return False

            if minimum is not None and value < minimum:
                self.set_widget_validation_state(
                    widget,
                    "invalid",
                    f"{field_name} values must be >= {minimum}.",
                )
                return False

        self.set_widget_validation_state(widget, "")
        return True

    # Validate Weka class lists and ranges using the pipeline's rules.
    def validate_probability_classes_field(
        self,
        widget,
        *,
        field_name: str = "Probability-map class numbers",
    ) -> bool:
        if widget is None:
            return True

        text = widget.text().strip()
        if not text:
            self.set_widget_validation_state(widget, "invalid", f"{field_name} cannot be empty.")
            return False

        if not probability_class_indices_are_valid(text):
            self.set_widget_validation_state(
                widget,
                "invalid",
                f"{field_name} must use positive whole numbers such as 1, 1,3, or 1-3.",
            )
            return False

        self.set_widget_validation_state(widget, "")
        return True

    # Validate the field using the processing step's shared rules and limits.
    def validate_processing_step_widget(self, widget, spec: ProcessingStepSpec) -> bool:
        if widget is None or spec.value_kind == "none":
            self.set_widget_validation_state(widget, "")
            return True
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            self.set_widget_validation_state(widget, "")
            return True

        if spec.value_kind == "outliers":
            value_getter = getattr(widget, "processing_value", None)
            value = str(value_getter()).strip() if callable(value_getter) else ""
            editors = widget.findChildren(QLineEdit)
            if not value and spec.allow_blank:
                for editor in editors:
                    self.set_widget_validation_state(editor, "")
                return True
            try:
                parse_outlier_settings(value)
            except ValueError as exc:
                for editor in editors:
                    self.set_widget_validation_state(editor, "invalid", str(exc))
                return False
            for editor in editors:
                self.set_widget_validation_state(editor, "")
            return True

        if spec.value_kind in {"binary", "particles", "translate"}:
            value_getter = getattr(widget, "processing_value", None)
            editors = widget.findChildren(QLineEdit)
            try:
                parser = {
                    "binary": parse_binary_settings,
                    "particles": parse_particle_settings,
                    "translate": parse_translate_offsets,
                }[spec.value_kind]
                parser(value_getter() if callable(value_getter) else "")
            except ValueError as exc:
                for editor in editors:
                    self.set_widget_validation_state(editor, "invalid", str(exc))
                return False
            for editor in editors:
                self.set_widget_validation_state(editor, "")
            return True

        value_edit = widget if isinstance(widget, QLineEdit) else widget.findChild(QLineEdit)
        if (
            isinstance(value_edit, QLineEdit)
            and bool(value_edit.property("blankMeansInactive"))
            and not value_edit.text().strip()
        ):
            self.set_widget_validation_state(value_edit, "")
            return True

        if spec.value_kind == "choice":
            if isinstance(widget, QComboBox):
                text = str(widget.currentData() or widget.currentText() or "").strip()
                target_widget = widget
            else:
                value_getter = getattr(widget, "processing_value", None)
                value_edit = widget if isinstance(widget, QLineEdit) else widget.findChild(QLineEdit)
                text = str(value_getter()).strip() if callable(value_getter) else (
                    value_edit.text().strip() if isinstance(value_edit, QLineEdit) else ""
                )
                target_widget = widget
            if (text or not spec.allow_blank) and text not in spec.choices:
                choices = ", ".join(spec.choices)
                self.set_widget_validation_state(
                    target_widget,
                    "invalid",
                    f"{spec.field_name} must be one of: {choices}.",
                )
                return False
            self.set_widget_validation_state(target_widget, "")
            return True

        if spec.value_kind == "csv_float":
            return self.validate_csv_number_field(
                value_edit,
                field_name=spec.field_name,
                minimum=spec.minimum,
                allow_blank=spec.allow_blank,
            )
        return self.validate_number_field(
            value_edit,
            field_name=spec.field_name,
            integer=spec.value_kind == "int",
            minimum=spec.minimum,
            maximum=spec.maximum,
            allow_blank=spec.allow_blank,
        )

    # Disabled recipe cells retain their values, so only active steps should participate in validation.
    def validate_mask_processing_table(self) -> bool:
        if not hasattr(self, "mask_processing_table"):
            return True

        table = self.mask_processing_table
        ok = True

        processing_tables = (
            (getattr(self, "image_processing_table", None), "_image_processing_row_refs"),
            (getattr(self, "measurement_processing_table", None), "_measurement_processing_row_refs"),
        )
        for channel_table, refs_name in processing_tables:
            if channel_table is None:
                continue
            row_refs = list(getattr(self, refs_name, []))
            for row_ref in row_refs:
                param_key = str(row_ref.get("param_key", "") or "")
                row_idx = int(row_ref.get("row_idx", -1))
                col_idx = int(row_ref.get("col_idx", -1))
                value_widget = channel_table.cellWidget(row_idx, col_idx)
                spec = IMAGE_PROCESSING_PARAM_SPECS.get(param_key)
                if spec is not None:
                    ok = self.validate_processing_step_widget(value_widget, spec) and ok

        mask_step_refs = list(getattr(self, "_mask_settings_step_refs", []))
        mask_refs = list(getattr(self, "_mask_settings_mask_refs", []))

        # Every mask column applies the same step specification to a different
        # configured mask, so only widget selection remains GUI-specific.
        def validate_mask_step_cell(row_idx: int, col_idx: int, step_type: str):
            nonlocal ok
            value_widget = table.cellWidget(row_idx, col_idx)
            spec = MASK_PROCESSING_STEP_SPECS.get(step_type)
            if spec is not None:
                ok = self.validate_processing_step_widget(value_widget, spec) and ok

        if mask_step_refs:
            for step_ref in mask_step_refs:
                row_idx = int(step_ref.get("row_idx", -1))
                header_widget = table.cellWidget(row_idx, 0)
                step_selector = self.processing_child_with_role(header_widget, "mask_step_type")
                step_type = (
                    self.mask_settings_step_type_from_widget(step_selector, str(step_ref.get("step_type", "") or ""))
                    if hasattr(self, "mask_settings_step_type_from_widget")
                    else str(step_ref.get("step_type", "") or "")
                )
                if not step_type:
                    continue
                if step_type not in MASK_PROCESSING_STEP_SPECS:
                    continue
                row_enabled = (
                    self.processing_row_enabled_from_header(header_widget, bool(step_ref.get("enabled", True)))
                    if hasattr(self, "processing_row_enabled_from_header")
                    else bool(step_ref.get("enabled", True))
                )
                if not row_enabled:
                    for mask_ref in mask_refs:
                        col_idx = int(mask_ref.get("col_idx", -1))
                        if col_idx >= 0:
                            self.set_widget_validation_state(table.cellWidget(row_idx, col_idx), "")
                    continue
                for mask_ref in mask_refs:
                    col_idx = int(mask_ref.get("col_idx", -1))
                    if col_idx >= 0:
                        validate_mask_step_cell(row_idx, col_idx, step_type)

        return ok

    # Validate every displayed row because each enabled channel owns one reusable Cellpose mask.
    def validate_cellpose_settings_table(self) -> bool:
        if not hasattr(self, "cellpose_settings_table"):
            return True

        table = self.cellpose_settings_table
        ok = True

        for row_idx in range(table.rowCount()):
            diameter_edit = table.cellWidget(row_idx, 5)
            min_size_edit = table.cellWidget(row_idx, 6)
            cellprob_edit = table.cellWidget(row_idx, 7)
            flow_edit = table.cellWidget(row_idx, 8)

            ok = (
                self.validate_number_field(
                    diameter_edit,
                    field_name="Cell diameter",
                    integer=False,
                    minimum=1,
                    allow_blank=True,
                )
                and ok
            )

            ok = (
                self.validate_number_field(
                    min_size_edit,
                    field_name="Minimum cell area",
                    integer=True,
                    minimum=0,
                    allow_blank=False,
                )
                and ok
            )

            ok = (
                self.validate_number_field(
                    cellprob_edit,
                    field_name="Cellpose probability threshold",
                    integer=False,
                    allow_blank=False,
                )
                and ok
            )

            ok = (
                self.validate_number_field(
                    flow_edit,
                    field_name="Cellpose flow threshold",
                    integer=False,
                    minimum=0,
                    allow_blank=False,
                )
                and ok
            )

        return ok

    # Check every field so all errors are shown at once.
    def validate_all_fields(self) -> bool:
        ok = True
        ok = self.validate_main_paths() and ok
        ok = self.validate_mask_processing_table() and ok
        ok = self.validate_cellpose_settings_table() and ok

        if not self.validate_classifier_paths():
            ok = False
            self.log("[WARN] One or more classifier paths do not exist.")

        return ok

    # Path rows share validation, but output paths may be valid before their final directory exists.
    def validate_path_row(
        self,
        row,
        *,
        field_name: str,
        must_exist: bool = True,
        must_be_dir: bool = True,
        allow_blank: bool = False,
        create_allowed: bool = False,
    ) -> bool:
        if row is None or not hasattr(row, "edit"):
            return True

        text = row.get().strip()

        if not text:
            if allow_blank:
                self.set_widget_validation_state(row.edit, "")
                return True

            self.set_widget_validation_state(
                row.edit, "invalid", f"{field_name} cannot be empty. Choose a folder before running or previewing."
            )
            return False

        path = Path(text)

        if must_exist and not path.exists():
            self.set_widget_validation_state(
                row.edit, "invalid", f"{field_name} does not exist. Browse to a reachable folder and try again."
            )
            return False

        if path.exists() and must_be_dir and not path.is_dir():
            self.set_widget_validation_state(row.edit, "invalid", f"{field_name} must be a folder, not a file.")
            return False

        if not must_exist and create_allowed:
            parent = path.parent
            if not parent.exists():
                self.set_widget_validation_state(
                    row.edit,
                    "invalid",
                    f"Parent folder does not exist: {parent}. Choose an output path inside an existing folder.",
                )
                return False

        self.set_widget_validation_state(row.edit, "")
        return True

    # Network paths rely on the background result to avoid freezing the GUI on an unreachable share.
    def validate_main_paths(self) -> bool:
        ok = True

        ok = (
            self.validate_path_row(
                self.fiji_app,
                field_name="Bundled Fiji runtime",
                must_exist=True,
                must_be_dir=True,
                allow_blank=False,
            )
            and ok
        )

        input_text = self.input_dir.get().strip()
        cached_scan = getattr(self, "_last_input_scan_result", {}) or {}
        if input_text.startswith("\\\\"):
            if input_path_key(str(cached_scan.get("path", "") or "")) == input_path_key(input_text):
                input_ok = bool(cached_scan.get("path_exists")) and bool(cached_scan.get("path_is_dir"))
                if input_ok:
                    self.set_widget_validation_state(self.input_dir.edit, "")
                else:
                    self.set_widget_validation_state(
                        self.input_dir.edit,
                        "invalid",
                        "Input directory could not be reached during the background scan. Reconnect the network drive or choose a local folder.",
                    )
                ok = input_ok and ok
            else:
                self.set_widget_validation_state(
                    self.input_dir.edit,
                    "invalid",
                    "Input directory is being checked in the background. Wait a moment, or use Check Setup for a full report.",
                )
                ok = False
        else:
            ok = (
                self.validate_path_row(
                    self.input_dir,
                    field_name="Input directory",
                    must_exist=True,
                    must_be_dir=True,
                    allow_blank=False,
                )
                and ok
            )

        output_ok = self.validate_path_row(
            self.output_dir,
            field_name="Output directory",
            must_exist=False,
            must_be_dir=True,
            allow_blank=False,
            create_allowed=True,
        )
        if output_ok and input_text:
            input_path = Path(input_text)
            output_path = Path(self.output_dir.get().strip())
            if output_is_inside_input(input_path, output_path):
                self.set_widget_validation_state(
                    self.output_dir.edit,
                    "invalid",
                    "Output directory must be outside the input directory so results are not scanned as input data.",
                )
                output_ok = False
        ok = output_ok and ok

        return ok

    # Missing classifiers are allowed only when existing masks can satisfy the run without regeneration.
    def validate_classifier_paths(self) -> bool:
        reuse_existing_masks = bool(
            hasattr(self, "reuse_existing_masks_checkbox") and self.reuse_existing_masks_checkbox.isChecked()
        )
        ok = True
        active_defs = self.get_active_image_definitions()

        for img in active_defs:
            if str(img.get("mask_source_mode", "") or "").strip() == MASK_SOURCE_MODE_COMBINED:
                continue
            classifier = str(img.get("classifier", "") or "").strip()
            if not classifier:
                continue

            path = Path(classifier)
            if not path.exists() or not path.is_file():
                if not reuse_existing_masks:
                    ok = False

        return ok
