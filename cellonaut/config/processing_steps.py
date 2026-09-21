"""Typed processing-step specifications shared by adapters and GUI validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_STEP_DEFINITIONS,
    MASK_PROCESSING_STEP_DEFINITIONS,
)
from cellonaut.measurement.math import parse_radii_csv


@dataclass(frozen=True, slots=True)
class ProcessingStepSpec:
    """Stable metadata and value constraints for one persisted recipe step."""

    step_type: str
    label: str
    param_key: str
    default: Any
    value_kind: str
    field_name: str
    allow_blank: bool
    minimum: float | None
    maximum: float | None
    choices: tuple[str, ...]
    default_scope: str = ""


def _spec_from_definition(definition: Mapping[str, Any]) -> ProcessingStepSpec:
    validation = dict(definition.get("validation", {}) or {})
    return ProcessingStepSpec(
        step_type=str(definition.get("type", "") or ""),
        label=str(definition.get("label", "") or ""),
        param_key=str(definition.get("param_key", "") or ""),
        default=definition.get("default", ""),
        value_kind=str(validation.get("kind", "none") or "none"),
        field_name=str(validation.get("field_name", definition.get("label", "Value")) or "Value"),
        allow_blank=bool(validation.get("allow_blank", False)),
        minimum=(float(validation["minimum"]) if validation.get("minimum") is not None else None),
        maximum=(float(validation["maximum"]) if validation.get("maximum") is not None else None),
        choices=tuple(str(value) for value in validation.get("choices", []) or []),
        default_scope=str(definition.get("default_scope", "") or ""),
    )


# Build shared lookup tables for processing-step definitions and validation rules.
IMAGE_PROCESSING_STEP_SPECS = {
    spec.step_type: spec
    for spec in (_spec_from_definition(definition) for definition in IMAGE_PROCESSING_STEP_DEFINITIONS)
}
MASK_PROCESSING_STEP_SPECS = {
    spec.step_type: spec
    for spec in (_spec_from_definition(definition) for definition in MASK_PROCESSING_STEP_DEFINITIONS)
}
IMAGE_PROCESSING_PARAM_SPECS = {spec.param_key: spec for spec in IMAGE_PROCESSING_STEP_SPECS.values() if spec.param_key}


def parse_outlier_settings(value: Any) -> tuple[float, float, str]:
    """Read Fiji Remove Outliers radius, raw-value threshold, and polarity."""
    parts = [part.strip() for part in str(value or "").split(",")]
    if len(parts) != 3 or parts[2] not in {"Bright", "Dark"}:
        raise ValueError("Remove Outliers needs a radius, threshold, and Bright or Dark selection.")
    try:
        radius, threshold = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError("Remove Outliers radius and threshold must be numbers.") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Remove Outliers radius must be greater than zero pixels.")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Remove Outliers threshold must be a non-negative pixel-value difference.")
    return radius, threshold, parts[2]


def parse_binary_settings(value: Any) -> tuple[int, int, bool]:
    """Read Fiji Binary Options iterations, count, and pad-edges settings."""
    parts = [part.strip() for part in str(value or "").split(",")]
    if len(parts) != 3 or parts[2].lower() not in {"true", "false"}:
        raise ValueError("Fiji Binary Options need iterations, count, and pad edges (true or false).")
    if not re.fullmatch(r"[0-9]+", parts[0]) or not re.fullmatch(r"[0-9]+", parts[1]):
        raise ValueError("Fiji Binary Options iterations and count must be whole numbers.")
    iterations, count = int(parts[0]), int(parts[1])
    if not 1 <= iterations <= 100 or not 1 <= count <= 8:
        raise ValueError("Fiji Binary Options require iterations 1-100 and count 1-8.")
    return iterations, count, parts[2].lower() == "true"


def parse_particle_settings(value: Any) -> tuple[float, float, float, float, bool, bool]:
    """Read Fiji Analyze Particles size, circularity, and mask options."""
    parts = [part.strip() for part in str(value or "").split(",")]
    if len(parts) != 4:
        raise ValueError("Analyze Particles needs Size, Circularity, Exclude on Edges, and Include Holes.")
    size = parts[0].split("-")
    circularity = parts[1].split("-")
    if len(size) != 2 or len(circularity) != 2:
        raise ValueError("Analyze Particles Size and Circularity need Fiji-style minimum-maximum ranges.")
    try:
        min_size = float(size[0])
        max_size = float(size[1])
        min_circularity = float(circularity[0])
        max_circularity = float(circularity[1])
    except ValueError as exc:
        raise ValueError("Analyze Particles Size and Circularity must be numbers or Infinity.") from exc
    if not (math.isfinite(min_size) and 0 <= min_size <= max_size) or math.isnan(max_size):
        raise ValueError("Analyze Particles Size must run from a non-negative minimum to a larger maximum.")
    if not (0 <= min_circularity <= max_circularity <= 1):
        raise ValueError("Analyze Particles Circularity must be within 0-1.")
    if parts[2].lower() not in {"true", "false"} or parts[3].lower() not in {"true", "false"}:
        raise ValueError("Analyze Particles edge and hole options must be true or false.")
    return min_size, max_size, min_circularity, max_circularity, parts[2].lower() == "true", parts[3].lower() == "true"


def parse_translate_offsets(value: Any) -> tuple[float, float]:
    """Read Fiji Translate's X and Y offsets in pixels."""
    parts = [part.strip() for part in str(value or "").split(",")]
    if len(parts) != 2:
        raise ValueError("Translate needs X and Y pixel offsets.")
    try:
        x_offset, y_offset = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError("Translate X and Y offsets must be numbers.") from exc
    if not math.isfinite(x_offset) or not math.isfinite(y_offset):
        raise ValueError("Translate X and Y offsets must be finite numbers.")
    return x_offset, y_offset


def _qualified_field_name(spec: ProcessingStepSpec, image_name: str) -> str:
    return f"{spec.field_name} for channel or mask '{image_name}'"


def _parse_numeric_value(raw_value: Any, spec: ProcessingStepSpec, image_name: str) -> float | int:
    label = _qualified_field_name(spec, image_name)
    text = str(raw_value if raw_value is not None else "").strip()
    if spec.value_kind == "int":
        if re.fullmatch(r"[+-]?\d+", text) is None:
            raise ValueError(f"{label} must be an integer, got: {raw_value!r}")
        value: float | int = int(text)
    else:
        try:
            value = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a number, got: {raw_value!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number, got: {raw_value!r}")

    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f"{label} must be >= {spec.minimum:g}, got: {raw_value!r}")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"{label} must be <= {spec.maximum:g}, got: {raw_value!r}")
    return value


def validate_processing_step_params(
    steps: Sequence[Mapping[str, Any]],
    specs: Mapping[str, ProcessingStepSpec],
    *,
    image_name: str,
) -> None:
    """Validate enabled recipe values against their shared step specifications."""

    for step in steps:
        if not bool(step.get("enabled", True)):
            continue
        spec = specs.get(str(step.get("type", "") or ""))
        if spec is None or spec.value_kind == "none" or not spec.param_key:
            continue

        params = dict(step.get("params", {}) or {})
        raw_value = params.get(spec.param_key, spec.default)
        text = str(raw_value if raw_value is not None else "").strip()
        label = _qualified_field_name(spec, image_name)
        if not text:
            if spec.allow_blank:
                continue
            raise ValueError(f"{label} cannot be empty.")

        if spec.value_kind == "choice":
            if text not in spec.choices:
                raise ValueError(f"{label} must be one of: {', '.join(spec.choices)}, got: {raw_value!r}")
            continue
        if spec.value_kind == "outliers":
            try:
                parse_outlier_settings(text)
            except ValueError as exc:
                raise ValueError(f"{label}: {exc}") from exc
            continue
        if spec.value_kind == "binary":
            try:
                parse_binary_settings(text)
            except ValueError as exc:
                raise ValueError(f"{label}: {exc}") from exc
            continue
        if spec.value_kind == "particles":
            try:
                parse_particle_settings(text)
            except ValueError as exc:
                raise ValueError(f"{label}: {exc}") from exc
            continue
        if spec.value_kind == "translate":
            try:
                parse_translate_offsets(text)
            except ValueError as exc:
                raise ValueError(f"{label}: {exc}") from exc
            continue
        if spec.value_kind == "csv_float":
            parse_radii_csv(text, strict=True, field_name=label)
            continue
        if spec.value_kind in {"float", "int"}:
            _parse_numeric_value(raw_value, spec, image_name)
