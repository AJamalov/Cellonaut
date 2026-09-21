"""Image and mask processing recipes shared by GUI settings and the pipeline.

Recipes are stored as ordered steps so users can tune them visually while the
pipeline can replay the same operations in Fiji/ImageJ or NumPy-backed paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_BIT_DEPTH_OPTIONS,
    IMAGE_PROCESSING_STEP_APPLY_LUT,
    IMAGE_PROCESSING_STEP_BIT_DEPTH,
    IMAGE_PROCESSING_STEP_DESPECKLE,
    IMAGE_PROCESSING_STEP_REMOVE_OUTLIERS,
    IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST,
    IMAGE_PROCESSING_STEP_GAUSSIAN_BLUR,
    IMAGE_PROCESSING_STEP_MEDIAN,
    IMAGE_PROCESSING_STEP_ROLLING_BALL,
    IMAGE_PROCESSING_STEP_SMOOTH,
    coerce_bool,
)
from cellonaut.io.image_io import imageplus_to_numpy_2d
from cellonaut.io.imagej_runtime import get_java_classes, jimport
from cellonaut.measurement.math import parse_radii_csv


# Measurement execution and its diagnostic montage must apply the same safe
# transform subset before background correction and quantitative summaries.
MEASUREMENT_TRANSFORM_STEP_TYPES = frozenset(
    {
        IMAGE_PROCESSING_STEP_APPLY_LUT,
        IMAGE_PROCESSING_STEP_BIT_DEPTH,
        IMAGE_PROCESSING_STEP_DESPECKLE,
        IMAGE_PROCESSING_STEP_REMOVE_OUTLIERS,
        IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST,
        IMAGE_PROCESSING_STEP_GAUSSIAN_BLUR,
        IMAGE_PROCESSING_STEP_MEDIAN,
        IMAGE_PROCESSING_STEP_SMOOTH,
    }
)


# Use the same enabled-step order for processing, mask reuse checks, and montages.
def _enabled_steps_for_scope(
    image_def: Any,
    scope: str,
    include_step_types: Iterable[str] | None = None,
) -> list[dict]:
    steps = list(getattr(image_def, "image_processing_steps", []) or [])
    include = {str(value) for value in include_step_types} if include_step_types is not None else None
    if steps:
        return [
            dict(step)
            for step in steps
            if isinstance(step, dict)
            and bool(step.get("enabled", True))
            and str(step.get("scope", "") or "") == scope
            and (include is None or str(step.get("type", "") or "") in include)
        ]
    return []


# Revalidate numeric text here even after GUI validation because pipeline configs
# can also be created by tests, scripts, or edited JSON.
def _non_negative_float(
    value: Any,
    *,
    field_name: str,
    maximum: float | None = None,
) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number, got: {value!r}")
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0, got: {value!r}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field_name} must be <= {maximum:g}, got: {value!r}")
    return parsed


# Reject zero and fractional iteration counts before invoking Fiji, whose command
# errors otherwise provide little context about the originating recipe row.
def _positive_int(
    value: Any,
    *,
    field_name: str,
    maximum: int | None = None,
) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got: {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1, got: {value!r}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}, got: {value!r}")
    return parsed


# Build on non-negative parsing so radius and sigma validation share finite-value
# and maximum checks while still excluding zero.
def _positive_float(
    value: Any,
    *,
    field_name: str,
    maximum: float | None = None,
) -> float | None:
    parsed = _non_negative_float(value, field_name=field_name, maximum=maximum)
    if parsed is not None and parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got: {value!r}")
    return parsed


# Pass only known Fiji command names; forwarding arbitrary text would turn a
# configuration value into an unchecked command selection.
def _bit_depth_command(value: Any) -> str | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    if text not in IMAGE_PROCESSING_BIT_DEPTH_OPTIONS:
        allowed = ", ".join(IMAGE_PROCESSING_BIT_DEPTH_OPTIONS)
        raise ValueError(f"Bit depth must be one of: {allowed}, got: {value!r}")
    return text


# Treat missing or non-dictionary parameters as empty settings.
def _step_params(step: dict) -> dict:
    params = step.get("params", {})
    return dict(params) if isinstance(params, dict) else {}


# Store a Fiji command and its options for recipe execution.
@dataclass(frozen=True)
class _ImageJRecipeAction:
    command: str
    options: str
    repetitions: int
    label: str
    scale_when_converting: bool | None = None


def fiji_background_option_suffix(params: dict) -> str:
    """Use Fiji's recorded checkbox names for an explicit background command."""
    flags = (
        ("light_background", "light"),
        ("sliding_paraboloid", "sliding"),
        ("disable_smoothing", "disable"),
    )
    return "".join(f" {macro}" for key, macro in flags if coerce_bool(params.get(key, False)))


# Compile each recipe row once into executable Fiji actions and a stable signature.
# This prevents execution, montage capture, and mask-reuse checks from drifting.
def _imagej_actions_for_step(step_type: str, params: dict) -> tuple[list[_ImageJRecipeAction], str | None]:
    actions: list[_ImageJRecipeAction] = []
    signature: str | None = None

    if step_type == IMAGE_PROCESSING_STEP_ROLLING_BALL:
        radii = [float(value) for value in parse_radii_csv(params.get("bg_radii", "")) if float(value) > 0]
        option_suffix = fiji_background_option_suffix(params)
        for radius in radii:
            actions.append(
                _ImageJRecipeAction(
                    "Subtract Background...",
                    f"rolling={radius}{option_suffix}",
                    1,
                    f"Subtract Background radius={radius:g}",
                )
            )
        if radii:
            signature = "subtract_background=" + ",".join(f"{radius:g}" for radius in radii) + option_suffix
    elif step_type == IMAGE_PROCESSING_STEP_GAUSSIAN_BLUR:
        sigma = _positive_float(params.get("gaussian_sigma", ""), field_name="Gaussian sigma")
        if sigma is not None:
            actions.append(
                _ImageJRecipeAction("Gaussian Blur...", f"sigma={sigma:g}", 1, f"Gaussian Blur sigma={sigma:g}")
            )
            signature = f"gaussian_blur={sigma:g}"
    elif step_type == IMAGE_PROCESSING_STEP_MEDIAN:
        radius = _positive_float(params.get("median_radius", ""), field_name="Median radius")
        if radius is not None:
            actions.append(_ImageJRecipeAction("Median...", f"radius={radius:g}", 1, f"Median radius={radius:g}"))
            signature = f"median={radius:g}"
    elif step_type == IMAGE_PROCESSING_STEP_DESPECKLE:
        actions.append(_ImageJRecipeAction("Despeckle", "", 1, "Despeckle"))
        signature = "despeckle"
    elif step_type == IMAGE_PROCESSING_STEP_REMOVE_OUTLIERS:
        settings = str(params.get("outlier_settings", "") or "").strip()
        if settings:
            from cellonaut.config.processing_steps import parse_outlier_settings

            radius, threshold, which = parse_outlier_settings(settings)
            actions.append(
                _ImageJRecipeAction(
                    "Remove Outliers...",
                    f"radius={radius:g} threshold={threshold:g} which={which}",
                    1,
                    f"Remove Outliers radius={radius:g}, threshold={threshold:g}, {which.lower()}",
                )
            )
            signature = f"remove_outliers={radius:g},{threshold:g},{which}"
    elif step_type == IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST:
        saturation = _non_negative_float(
            params.get("contrast_saturation", ""),
            field_name="Contrast saturation",
            maximum=100,
        )
        if saturation is not None:
            flags = "".join(
                f" {macro}"
                for key, macro in (("normalize", "normalize"), ("equalize", "equalize"))
                if coerce_bool(params.get(key, False))
            )
            actions.append(
                _ImageJRecipeAction(
                    "Enhance Contrast",
                    f"saturated={saturation:.2f}{flags}",
                    1,
                    f"Enhance contrast saturated={saturation:g}%",
                )
            )
            signature = f"enhance_contrast={saturation:g}{flags}"
    elif step_type == IMAGE_PROCESSING_STEP_APPLY_LUT:
        actions.append(_ImageJRecipeAction("Apply LUT", "", 1, "Apply LUT"))
        signature = "apply_lut"
    elif step_type == IMAGE_PROCESSING_STEP_SMOOTH:
        iterations = _positive_int(
            params.get("smooth_iterations", ""),
            field_name="Smooth iterations",
            maximum=100,
        )
        if iterations is not None:
            actions.append(_ImageJRecipeAction("Smooth", "", iterations, f"Smooth x{iterations}"))
            signature = f"smooth={iterations}"
    elif step_type == IMAGE_PROCESSING_STEP_BIT_DEPTH:
        command = _bit_depth_command(params.get("bit_depth", ""))
        if command is not None:
            scaling = (
                coerce_bool(params["scale_when_converting"], True)
                if "scale_when_converting" in params
                else None
            )
            actions.append(_ImageJRecipeAction(command, "", 1, f"Convert Bit Depth {command}", scaling))
            signature = f"bit_depth={command}" + (f",scale={scaling}" if scaling is not None else "")

    return actions, signature


# Distinguish preprocessing recipes in the in-memory Weka probability-map cache.
# This signature is not a provenance check for masks loaded from a previous run.
def imagej_processing_recipe_signature(
    image_def: Any,
    scope: str,
    include_step_types: Iterable[str] | None = None,
) -> str:
    parts: list[str] = []
    for step in _enabled_steps_for_scope(image_def, scope, include_step_types):
        step_type = str(step.get("type", "") or "")
        params = _step_params(step)
        _actions, signature = _imagej_actions_for_step(step_type, params)
        if signature is not None:
            parts.append(signature)
    return ";".join(parts)


def _execute_imagej_processing_recipe(
    img: Any,
    image_def: Any,
    scope: str,
    include_step_types: Iterable[str] | None = None,
    capture: Callable[[int, str, Any], None] | None = None,
) -> tuple[Any | None, list[str]]:
    """Run one lazy Fiji copy and optionally capture each completed action."""
    IJ = None
    Duplicator = None
    current = None
    applied: list[str] = []
    action_number = 0
    try:
        for step in _enabled_steps_for_scope(image_def, scope, include_step_types):
            step_type = str(step.get("type", "") or "")
            params = _step_params(step)
            actions, _signature = _imagej_actions_for_step(step_type, params)
            for action in actions:
                # Montage inspection also visits empty recipes. Starting Java
                # here, at the first real command, keeps Cellpose-only runs on
                # their intended native NumPy path.
                if IJ is None or Duplicator is None:
                    classes = get_java_classes()
                    IJ = classes["IJ"]
                    Duplicator = classes["Duplicator"]
                if current is None:
                    current = Duplicator().run(img)
                if action.scale_when_converting is None:
                    for _index in range(action.repetitions):
                        IJ.run(current, action.command, action.options)
                else:
                    converter = jimport("ij.process.ImageConverter")
                    prefs = jimport("ij.Prefs")
                    previous_scaling = bool(converter.getDoScaling())
                    previous_calibration = bool(prefs.calibrateConversions)
                    converter.setDoScaling(action.scale_when_converting)
                    prefs.calibrateConversions = False
                    try:
                        for _index in range(action.repetitions):
                            IJ.run(current, action.command, action.options)
                    finally:
                        prefs.calibrateConversions = previous_calibration
                        converter.setDoScaling(previous_scaling)
                action_number += 1
                applied.append(action.label)
                if capture is not None:
                    capture(action_number, action.label, current)
    except Exception:
        if current is not None:
            try:
                current.close()
            except Exception:
                pass
        raise
    return current, applied


# Copy the image before the first processing step so the original stays unchanged.
def apply_imagej_processing_recipe_copy(
    img: Any,
    image_def: Any,
    scope: str,
    include_step_types: Iterable[str] | None = None,
) -> tuple[Any, list[str]]:
    if img is None:
        return None, []

    current, applied = _execute_imagej_processing_recipe(img, image_def, scope, include_step_types)
    return (current if current is not None else img), applied


# Capture a separate image copy after each processing action.
# Return temporary ImageJ images for the caller to close.
def imagej_processing_recipe_tiles(
    img: Any,
    image_def: Any,
    scope: str,
    include_step_types: Iterable[str] | None = None,
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray | None, list[Any]]:
    if img is None:
        return [], None, []

    tiles: list[tuple[str, np.ndarray]] = []
    final_arr: np.ndarray | None = None

    # Copy the NumPy array because the ImagePlus continues changing as later steps
    # run and every montage tile must preserve its own stage.
    def capture(step_number: int, label: str, target: Any) -> None:
        nonlocal final_arr
        final_arr = np.asarray(imageplus_to_numpy_2d(target)).copy()
        tiles.append((f"{step_number:02d} {label}", final_arr))

    current, _applied = _execute_imagej_processing_recipe(img, image_def, scope, include_step_types, capture)
    return tiles, final_arr, [current] if current is not None else []
