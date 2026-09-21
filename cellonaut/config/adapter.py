"""Adapter converting GUI state into pipeline configuration objects."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict

from cellonaut.measurement.math import parse_radii_csv
from cellonaut.pipeline.models import Config, ImageDef, MeasurementTarget, resolve_measurement_target
from cellonaut.config.defaults import (
    CELLPOSE_MODEL_OPTIONS,
    DEFAULT_CELL_DIAMETER,
    DEFAULT_CELL_MIN_SIZE,
    DEFAULT_CELLPROB_THRESHOLD,
    DEFAULT_FLOW_THRESHOLD,
    DEFAULT_MASK_SOURCE_MODE,
    COMBINED_MASK_OPERATION_OPTIONS,
    MASK_SOURCE_MODE_COMBINED,
    MASK_SOURCE_MODE_OPTIONS,
    MASK_SOURCE_MODE_WEKA,
    DEFAULT_STACK_Z_MODE,
    DEFAULT_THRESHOLD_METHOD,
    STACK_Z_MODE_OPTIONS,
    default_mask_adjustments,
    normalize_cellpose_model_type,
    default_measurement_options,
    normalize_image_processing_steps,
    normalize_mask_processing_steps,
)
from cellonaut.config.processing_steps import (
    IMAGE_PROCESSING_STEP_SPECS,
    MASK_PROCESSING_STEP_SPECS,
    validate_processing_step_params,
)
from cellonaut.config.state import ImageGuiState, CellonautGuiState


# Parse related Cellpose values as one unit so validation cannot accept a
# partially converted set of per-channel segmentation settings.
class _CellposeNumericSettings(TypedDict):
    cell_diameter: float | None
    cell_min_size: int
    cellprob_threshold: float
    flow_threshold: float


def _decode_filter_rule_items(text: str, *, strict: bool, field_name: str) -> list[tuple[str, dict, bool]]:
    """Decode JSON or compact text into metric, bounds, and compact-format markers."""
    if text.startswith("{"):
        try:
            value = json.loads(text)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"{field_name} contains invalid JSON: {exc}") from exc
            return []
        if not isinstance(value, dict):
            if strict:
                raise ValueError(f"{field_name} JSON must be an object of metric rules.")
            return []
        items = []
        for metric, limits in value.items():
            metric_name = str(metric or "").strip()
            if metric_name and isinstance(limits, dict):
                items.append((metric_name, limits, False))
            elif strict:
                raise ValueError(f"{field_name} contains an invalid rule for metric {metric!r}.")
        return items

    items = []
    for part in (part.strip() for part in text.replace("\n", ";").split(";")):
        if not part:
            continue
        if ":" not in part:
            if strict:
                raise ValueError(f"{field_name} rule {part!r} must use the format Metric:min-max.")
            continue
        metric, limits = (value.strip() for value in part.split(":", 1))
        if not metric or not limits:
            if strict:
                raise ValueError(f"{field_name} rule {part!r} must include a metric and at least one bound.")
            continue
        low, high = limits.split("-", 1) if "-" in limits else ("", limits)
        items.append((metric, {"min": low.strip(), "max": high.strip()}, True))
    return items


def _normalize_filter_rule_items(
    items: list[tuple[str, dict, bool]], *, strict: bool, field_name: str
) -> dict[str, dict[str, float]]:
    rules: dict[str, dict[str, float]] = {}
    for metric, limits, compact in items:
        unknown = set(limits) - {"min", "max"}
        if unknown:
            if strict:
                raise ValueError(
                    f"{field_name} rule '{metric}' has unsupported bound(s): "
                    + ", ".join(sorted(str(bound) for bound in unknown))
                )
            continue

        rule: dict[str, float] = {}
        for bound, label in (("min", "minimum"), ("max", "maximum")):
            raw_value = limits.get(bound)
            if raw_value is None or raw_value == "":
                continue
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError) as exc:
                if strict:
                    bound_name = label if compact else bound
                    raise ValueError(
                        f"{field_name} rule '{metric}' {bound_name} must be a number, got: {raw_value!r}"
                    ) from exc
                continue
            if not math.isfinite(parsed):
                if strict:
                    suffix = "" if compact else f", got: {raw_value!r}"
                    raise ValueError(f"{field_name} rule '{metric}' {bound} must be finite{suffix}.")
                if compact:
                    rule = {}
                    break
                continue
            rule[bound] = parsed

        if strict and not rule:
            expected = "a valid minimum, maximum, or both" if compact else "min, max, or both"
            raise ValueError(f"{field_name} rule '{metric}' must define {expected}.")
        if strict and "min" in rule and "max" in rule and rule["min"] > rule["max"]:
            raise ValueError(f"{field_name} rule '{metric}' minimum cannot exceed its maximum.")
        if rule:
            rules[metric] = rule
    return rules


# Filter rules accept compact text for manual entry and JSON for structured
# presets. Strict mode is reserved for launching so casual editing stays tolerant.
def parse_cell_qc_limits_text(
    text: str,
    *,
    strict: bool = False,
    field_name: str = "Filter rules",
) -> dict:
    text = str(text or "").strip()
    if not text:
        return {}
    items = _decode_filter_rule_items(text, strict=strict, field_name=field_name)
    return _normalize_filter_rule_items(items, strict=strict, field_name=field_name)


# Persist one of two explicit intensity sources so renamed UI wording cannot
# alter which image is used for mask quality checks.


# Validation errors must identify the affected channel or mask because the same
# setting can appear in many dynamic rows.
def _field_label(field_name: str, image_name: str | None = None) -> str:
    if image_name:
        return f"{field_name} for channel or mask '{image_name}'"
    return field_name


# Reject invalid or non-finite numbers before processing starts.
def _required_float(
    value,
    *,
    field_name: str,
    image_name: str | None = None,
    minimum: float | None = None,
) -> float:
    label = _field_label(field_name, image_name)
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    try:
        parsed = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number, got: {value!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum:g}, got: {value!r}")
    return parsed


# Integer fields reject decimal-looking values explicitly; silently truncating
# pixel counts or stack indices would process a different setting than shown.
def _required_int(
    value,
    *,
    field_name: str,
    image_name: str | None = None,
    minimum: int | None = None,
) -> int:
    label = _field_label(field_name, image_name)
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    if re.fullmatch(r"[+-]?\d+", text) is None:
        raise ValueError(f"{label} must be an integer, got: {value!r}")
    parsed = int(text)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got: {value!r}")
    return parsed


# Preserve blank as None because an omitted correction is different from an
# explicitly configured correction value of zero.
def _optional_float(value, *, field_name: str, image_name: str | None = None) -> float | None:
    if not str(value if value is not None else "").strip():
        return None
    return _required_float(value, field_name=field_name, image_name=image_name)


# A blank stack layer means automatic channel discovery; explicit layers are
# one-based to match the numbering users see in Fiji and the GUI.
def _optional_positive_int(value, *, image_name: str | None = None) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return _required_int(
        value,
        field_name="Stack layer",
        image_name=image_name,
        minimum=1,
    )


# Validate operation-specific limits together with the recipe so users receive
# a named setting error instead of a later ImageJ command failure.
def _validate_image_processing_step_params(steps: list[dict], *, image_name: str) -> None:
    validate_processing_step_params(steps, IMAGE_PROCESSING_STEP_SPECS, image_name=image_name)


# Morphology settings are checked before mask creation because some invalid
# values would otherwise fail only after Weka has finished running.
def _validate_mask_processing_step_params(steps: list[dict], *, image_name: str) -> None:
    validate_processing_step_params(steps, MASK_PROCESSING_STEP_SPECS, image_name=image_name)


# An unknown projection mode must not silently choose a different Z plane, as
# that would make a successful run scientifically misleading.
def _stack_z_mode(value, *, image_name: str | None = None) -> str:
    text = str(value or DEFAULT_STACK_Z_MODE).strip() or DEFAULT_STACK_Z_MODE
    if text not in STACK_Z_MODE_OPTIONS:
        label = _field_label("Stack Z mode", image_name)
        raise ValueError(f"{label} must be one of: {', '.join(STACK_Z_MODE_OPTIONS)}")
    return text


# Leave blank Z indices unset; otherwise require a positive whole number.
def _optional_z_index(value, *, image_name: str | None = None) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return _required_int(
        value,
        field_name="Z slice",
        image_name=image_name,
        minimum=1,
    )


# Expand Weka class lists and ranges, preserving order and removing duplicates.
def _probability_class_indices(value, *, image_name: str) -> list[int]:
    label = _field_label("Weka class selection", image_name)
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")

    indices: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"{label} must use values like 1, 1,3, or 1-3.")
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if match is None:
            raise ValueError(f"{label} must use positive integers like 1, 1,3, or 1-3.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < 1:
            raise ValueError(f"{label} values must be >= 1.")
        if start > end:
            start, end = end, start
        indices.extend(range(start, end + 1))

    return list(dict.fromkeys(indices))


# Cellpose values feed both per-target and summary configuration. Parse them once
# so those two representations cannot disagree.
def _cell_numeric_settings(image: dict, *, image_name: str) -> _CellposeNumericSettings:
    diameter = _optional_float(
        image.get("cell_diameter", DEFAULT_CELL_DIAMETER),
        field_name="Cell diameter",
        image_name=image_name,
    )
    if diameter is not None and diameter < 1:
        raise ValueError(f"{_field_label('Cell diameter', image_name)} must be >= 1, got: {diameter!r}")
    return {
        "cell_diameter": diameter,
        "cell_min_size": _required_int(
            image.get("cell_min_size", DEFAULT_CELL_MIN_SIZE),
            field_name="Minimum cell area",
            image_name=image_name,
            minimum=0,
        ),
        "cellprob_threshold": _required_float(
            image.get("cellprob_threshold", DEFAULT_CELLPROB_THRESHOLD),
            field_name="Cellpose probability threshold",
            image_name=image_name,
        ),
        "flow_threshold": _required_float(
            image.get("flow_threshold", DEFAULT_FLOW_THRESHOLD),
            field_name="Cellpose flow threshold",
            image_name=image_name,
            minimum=0,
        ),
    }


# Require whole numbers for mask adjustments so pixel values are not rounded.
def _mask_adjustments(value, *, field_name: str, image_name: str) -> dict[str, int]:
    defaults = default_mask_adjustments()
    raw = value if isinstance(value, dict) else {}
    result: dict[str, int] = {}
    for key in defaults:
        result[key] = _required_int(
            raw.get(key, defaults[key]),
            field_name=f"{field_name} {key}",
            image_name=image_name,
        )
    if result["min_size"] < 0 or result["fill_holes_area"] < 0:
        raise ValueError(f"{field_name} minimum size and fill holes for channel or mask '{image_name}' must be >= 0.")
    return result


# Convert GUI settings into pipeline configuration, checking references and required values.
def _inherit_mask_channel_metadata(images: list[ImageGuiState]) -> list[ImageGuiState]:
    channels = {
        image.name.strip(): image
        for image in images
        if not image.is_mask_only and image.mask_source_mode.strip() != MASK_SOURCE_MODE_COMBINED
    }
    inherited: list[ImageGuiState] = []
    for image in images:
        if not image.is_mask_only or image.mask_source_mode.strip() == MASK_SOURCE_MODE_COMBINED:
            inherited.append(image)
            continue
        source_name = image.mask_source_channel.strip()
        source = channels.get(source_name)
        if source is None:
            inherited.append(image)
            continue
        inherited.append(
            replace(
                image,
                folder=source.folder or source_name,
                stack_channel_index=source.stack_channel_index,
                stack_z_mode=source.stack_z_mode or DEFAULT_STACK_Z_MODE,
                stack_z_index=source.stack_z_index,
                display_color=source.display_color,
                bg_radii=source.bg_radii,
                image_processing_steps=list(source.image_processing_steps),
            )
        )
    return inherited


@dataclass(slots=True)
class _ParsedImages:
    definitions: list[ImageDef]
    probability_classes: dict[str, list[int]]
    cell_numbers: list[_CellposeNumericSettings]


def _parse_image_definitions(images: list[ImageGuiState], state: CellonautGuiState) -> _ParsedImages:
    definitions: list[ImageDef] = []
    probability_classes: dict[str, list[int]] = {}
    cell_numbers: list[_CellposeNumericSettings] = []

    for index, image in enumerate(images):
        name = (image.name or f"Image{index + 1}").strip()
        folder = (image.folder or name).strip()
        classifier_raw = image.classifier.strip()
        mask_source_mode = (image.mask_source_mode or DEFAULT_MASK_SOURCE_MODE).strip()
        mask_slot_enabled = image.mask_slot_enabled or image.is_mask_only
        if mask_slot_enabled and mask_source_mode not in MASK_SOURCE_MODE_OPTIONS:
            raise ValueError(
                f"Mask source for channel or mask '{name}' must be one of: {', '.join(MASK_SOURCE_MODE_OPTIONS)}."
            )

        probability_text = str(image.probability_class_index or "").strip()
        if mask_source_mode != MASK_SOURCE_MODE_WEKA:
            probability_text = "1"
        probability_classes[name] = _probability_class_indices(probability_text, image_name=name)
        raw = image.to_dict()
        cell_numbers.append(_cell_numeric_settings(raw, image_name=name))
        parse_radii_csv(image.bg_radii, strict=True, field_name=_field_label("Background radii", name))
        # Validate editable group conditions at the boundary without putting
        # them into pipeline configuration or evaluating them during a run.
        for group in image.cell_populations:
            for key, label in (("cell_qc_limits", "Cell-group conditions"), ("mask_qc_limits", "Mask filters")):
                parse_cell_qc_limits_text(
                    str(group.get(key) or ""), strict=True, field_name=_field_label(label, name)
                )
        if not mask_slot_enabled or mask_source_mode == MASK_SOURCE_MODE_COMBINED:
            classifier_path = None
        elif classifier_raw:
            classifier_path = Path(classifier_raw)
            if not classifier_path.exists() and not state.reuse_existing_masks:
                raise ValueError(f"Weka classifier file does not exist for mask '{name}':\n{classifier_path}")
        else:
            classifier_path = None

        image_steps = normalize_image_processing_steps(
            image.image_processing_steps,
            {"bg_radii": image.bg_radii},
        )
        _validate_image_processing_step_params(image_steps, image_name=name)
        mask_steps = normalize_mask_processing_steps(image.mask_processing_steps)
        _validate_mask_processing_step_params(mask_steps, image_name=name)
        definitions.append(
            ImageDef(
                key=f"image{index + 1}",
                label=name,
                folder_name=folder,
                model_path=classifier_path,
                display_color=image.display_color,
                mask_source_mode=mask_source_mode,
                combined_mask_operation=str(image.combined_mask_operation or "").strip().upper(),
                stack_channel_index=_optional_positive_int(image.stack_channel_index, image_name=name),
                stack_z_mode=_stack_z_mode(image.stack_z_mode, image_name=name),
                stack_z_index=_optional_z_index(image.stack_z_index, image_name=name),
                bg_radii_csv=image.bg_radii,
                probability_class_index=probability_text,
                threshold_method=image.threshold_method or DEFAULT_THRESHOLD_METHOD,
                image_processing_steps=image_steps,
                mask_processing_steps=mask_steps,
            )
        )

    if not definitions:
        raise ValueError("At least one channel or mask is required")
    labels = [image.label for image in definitions]
    duplicate = next((label for index, label in enumerate(labels) if label in labels[:index]), None)
    if duplicate is not None:
        raise ValueError(f"Channel and mask names must be unique. Duplicate name: {duplicate}")
    return _ParsedImages(definitions, probability_classes, cell_numbers)


def _resolve_image_relationships(images: list[ImageGuiState], definitions: list[ImageDef]) -> dict[str, str]:
    label_to_key = {image.label: image.key for image in definitions}
    for definition, image in zip(definitions, images):
        if image.is_mask_only:
            definition.stack_source_image_key = label_to_key.get(image.mask_source_channel.strip(), "")

    mask_source_keys = {
        definition.key
        for definition, image in zip(definitions, images)
        if definition.model_path is not None or image.mask_source_mode.strip() == MASK_SOURCE_MODE_COMBINED
    }
    for definition, image in zip(definitions, images):
        if definition.mask_source_mode != MASK_SOURCE_MODE_COMBINED:
            continue
        source_keys = [
            label_to_key[name]
            for value in image.combined_mask_sources
            if (name := str(value).strip())
            and name in label_to_key
            and label_to_key[name] in mask_source_keys
            and name != definition.label
        ]
        if len(source_keys) < 2:
            raise ValueError(f"Combined mask '{definition.label}' needs at least two source masks.")
        if definition.combined_mask_operation not in COMBINED_MASK_OPERATION_OPTIONS:
            raise ValueError(
                f"Combined mask '{definition.label}' operation must be OR, AND, or XOR."
            )
        definition.combined_mask_source_keys = source_keys

    combined_by_key = {
        definition.key: definition
        for definition in definitions
        if definition.mask_source_mode == MASK_SOURCE_MODE_COMBINED
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def validate_dependencies(key: str) -> None:
        if key in visiting:
            raise ValueError(f"Combined mask dependency cycle involving '{combined_by_key[key].label}'.")
        if key in visited or key not in combined_by_key:
            return
        visiting.add(key)
        for source_key in combined_by_key[key].combined_mask_source_keys:
            validate_dependencies(source_key)
        visiting.remove(key)
        visited.add(key)

    for key in combined_by_key:
        validate_dependencies(key)
    return label_to_key


def _build_measurement_targets(
    images: list[ImageGuiState],
    parsed: _ParsedImages,
    *,
    label_to_key: dict[str, str],
    measurement_options: dict[str, bool],
    reuse_existing_masks: bool,
) -> list[MeasurementTarget]:
    roi_label_to_key = {
        image.label: image.key
        for image in parsed.definitions
        if image.model_path is not None or image.combined_mask_source_keys
    }
    targets: list[MeasurementTarget] = []
    for index, image in enumerate(images):
        if image.is_mask_only or image.mask_source_mode.strip() == MASK_SOURCE_MODE_COMBINED:
            continue
        source_name = image.name.strip()
        if not source_name or source_name not in label_to_key:
            continue
        source_key = label_to_key[source_name]
        selected_masks = [
            name for name, checked in image.mask_relationships.items() if checked and name in roi_label_to_key
        ]
        overlay_roi_keys = [roi_label_to_key[name] for name in selected_masks]
        requested_cell_group_mask = image.cell_group_mask_source.strip()
        cell_group_mask = (
            requested_cell_group_mask
            if requested_cell_group_mask in selected_masks
            else (selected_masks[0] if selected_masks else "")
        )
        per_cell_mask_source = roi_label_to_key[cell_group_mask] if cell_group_mask else ""
        do_cell_segmentation = image.analysis_cell_segmentation_enabled
        model_type = normalize_cellpose_model_type(image.cellpose_model_type)
        custom_model_path = image.cellpose_custom_model_path.strip()
        if do_cell_segmentation and not custom_model_path and model_type not in CELLPOSE_MODEL_OPTIONS:
            raise ValueError(
                f"Cellpose model for channel or mask '{source_name}' must be one of: "
                f"{', '.join(CELLPOSE_MODEL_OPTIONS)}."
            )
        if do_cell_segmentation and custom_model_path:
            custom_model_file = Path(custom_model_path)
            if not custom_model_file.is_file() and not reuse_existing_masks:
                raise ValueError(
                    f"Custom Cellpose model file does not exist for channel or mask '{source_name}':\n"
                    f"{custom_model_file}"
                )
        if not (overlay_roi_keys or per_cell_mask_source or do_cell_segmentation):
            continue

        numbers = parsed.cell_numbers[index]
        targets.append(
            MeasurementTarget(
                source_image_key=source_key,
                overlay_base_image_key=source_key,
                overlay_roi_keys=overlay_roi_keys,
                do_cell_segmentation=do_cell_segmentation,
                cell_segmentation_source=label_to_key.get(
                    image.analysis_cell_segmentation_source or source_name, source_key
                ),
                per_cell_mask_source=per_cell_mask_source,
                overlay_whole_cell_mask=do_cell_segmentation,
                measurement_options=dict(measurement_options),
                cell_diameter=numbers["cell_diameter"],
                cell_min_size=int(numbers["cell_min_size"]),
                cell_use_gpu=True,
                cellprob_threshold=float(numbers["cellprob_threshold"]),
                flow_threshold=float(numbers["flow_threshold"]),
                cell_remove_border=image.cell_remove_border,
                cellpose_model_type=model_type,
                cellpose_custom_model_path=custom_model_path,
                cell_mask_adjustments=_mask_adjustments(
                    image.cell_mask_adjustments,
                    field_name="Cell mask adjustment",
                    image_name=source_name,
                ),
            )
        )
    return targets


def build_pipeline_config_from_gui_state(state: CellonautGuiState) -> Config:
    """Convert a committed GUI snapshot into detached, resolved execution settings.

    Parse editable numeric strings and map display-name relationships to run-local
    image keys. Blank diameter becomes None, never another target's default.
    Cell Groups are omitted because they govern later exports. Invalid settings
    raise ValueError; filesystem/backend readiness is checked separately.
    """
    if not str(state.input_dir or "").strip():
        raise ValueError("Input directory cannot be empty.")
    if not str(state.output_dir or "").strip():
        raise ValueError("Output directory cannot be empty.")

    images = _inherit_mask_channel_metadata(list(state.image_definitions))
    parsed = _parse_image_definitions(images, state)
    label_to_key = _resolve_image_relationships(images, parsed.definitions)
    measurement_options = {
        key: bool((state.measurement_options or {}).get(key, default_value))
        for key, default_value in default_measurement_options().items()
    }
    measurement_targets = _build_measurement_targets(
        images,
        parsed,
        label_to_key=label_to_key,
        measurement_options=measurement_options,
        reuse_existing_masks=state.reuse_existing_masks,
    )
    if not measurement_targets:
        raise ValueError(
            "No analysis task is enabled.\n\n"
            "Enable cell masks for at least one channel, or assign at least one "
            "mask to a measured channel."
        )

    first_image = images[0]
    cfg = Config(
        fiji_app_path=Path(state.fiji_app_path),
        input_dir=Path(state.input_dir),
        output_dir=Path(state.output_dir),
        input_structure=state.input_structure,
        images=parsed.definitions,
        exclusion_tag=state.exclusion_tag,
        threshold_method=first_image.threshold_method or DEFAULT_THRESHOLD_METHOD,
        probability_class_index=parsed.probability_classes[parsed.definitions[0].label][0],
        measurement_options=measurement_options,
        measurement_targets=list(measurement_targets),
        reuse_existing_masks=state.reuse_existing_masks,
        mask_source_dir=Path(state.mask_source_dir or state.output_dir),
    )
    # GUI state already supplies complete per-channel settings. Run defaults
    # stay independent of channel order; only the shared measurement selection
    # is copied to targets. Blank diameter is explicitly None (original scale).
    cfg.measurement_targets = [resolve_measurement_target(cfg, target) for target in measurement_targets]
    return cfg
