"""Pipeline target, ROI, reusable-mask, and channel-opening planning."""

from __future__ import annotations

from pathlib import Path

from cellonaut.results.artifacts import saved_cell_labels, ArtifactMetadataError
from cellonaut.results.layout import results_root
from typing import Dict, List, Optional, Sequence

import tifffile

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_SCOPE_MEASUREMENT,
    IMAGE_PROCESSING_STEP_APPLY_LUT,
    IMAGE_PROCESSING_STEP_BIT_DEPTH,
    IMAGE_PROCESSING_STEP_DESPECKLE,
    IMAGE_PROCESSING_STEP_REMOVE_OUTLIERS,
    IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST,
    IMAGE_PROCESSING_STEP_GAUSSIAN_BLUR,
    IMAGE_PROCESSING_STEP_MEDIAN,
    IMAGE_PROCESSING_STEP_ROLLING_BALL,
    IMAGE_PROCESSING_STEP_SMOOTH,
    image_processing_step_has_scope,
    image_processing_step_param_values,
    coerce_bool,
)
from cellonaut.io.image_io import read_tiff_numpy_2d
from cellonaut.masks.roi_processing import threshold_mask_path
from cellonaut.measurement.math import base_name_no_ext, parse_radii_csv
from cellonaut.pipeline.image_processing import fiji_background_option_suffix
from cellonaut.pipeline.models import Config, ImageDef, TargetConfig
from cellonaut.pipeline.roi_defs import (
    dedupe_preserve_order,
    expand_image_def_to_class_roi_defs,
    make_class_roi_def,
    mask_reference_image_keys,
    parse_class_roi_key,
    parse_probability_class_indices,
)


def measurement_background_requests(image_def: ImageDef) -> list[tuple[float, str]]:
    """Keep each measurement radius paired with its saved Fiji options."""
    if getattr(image_def, "image_processing_steps", None):
        requests: list[tuple[float, str]] = []
        for step in image_def.image_processing_steps:
            if not isinstance(step, dict) or not coerce_bool(step.get("enabled", True), True):
                continue
            if step.get("type") != IMAGE_PROCESSING_STEP_ROLLING_BALL or step.get("scope") != IMAGE_PROCESSING_SCOPE_MEASUREMENT:
                continue
            params = dict(step.get("params", {}) or {})
            options = fiji_background_option_suffix(params)
            requests.extend((radius, options) for radius in parse_radii_csv(params.get("bg_radii", "")))
        return requests
    if not image_processing_step_has_scope(
        image_def,
        IMAGE_PROCESSING_STEP_ROLLING_BALL,
        IMAGE_PROCESSING_SCOPE_MEASUREMENT,
    ):
        return []
    return [(radius, "") for radius in parse_radii_csv(getattr(image_def, "bg_radii_csv", "") or "")]


def measurement_background_radii(image_def: ImageDef) -> list[float]:
    """Return enabled rolling-ball radii used by measurement processing."""
    return [radius for radius, _options in measurement_background_requests(image_def)]


def measurement_requires_imagej_processing(image_def: ImageDef) -> bool:
    """Return whether enabled measurement steps require an ImageJ runtime."""

    for step_type, param_key in (
        (IMAGE_PROCESSING_STEP_GAUSSIAN_BLUR, "gaussian_sigma"),
        (IMAGE_PROCESSING_STEP_MEDIAN, "median_radius"),
        (IMAGE_PROCESSING_STEP_DESPECKLE, ""),
        (IMAGE_PROCESSING_STEP_REMOVE_OUTLIERS, "outlier_settings"),
        (IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST, "contrast_saturation"),
        (IMAGE_PROCESSING_STEP_APPLY_LUT, ""),
        (IMAGE_PROCESSING_STEP_SMOOTH, "smooth_iterations"),
        (IMAGE_PROCESSING_STEP_BIT_DEPTH, "bit_depth"),
    ):
        if not param_key:
            if any(
                isinstance(step, dict)
                and bool(step.get("enabled", True))
                and str(step.get("type", "") or "") == step_type
                and str(step.get("scope", "") or "") == IMAGE_PROCESSING_SCOPE_MEASUREMENT
                for step in list(getattr(image_def, "image_processing_steps", []) or [])
            ):
                return True
            continue
        if image_processing_step_param_values(
            image_def,
            step_type,
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
            param_key,
        ):
            return True
    return False


def get_enabled_measurement_targets(cfg: Config) -> list[TargetConfig]:
    """Return measurement targets that contribute work to the current run."""

    return [target for target in (cfg.measurement_targets or []) if target.enabled]


def get_image_def(cfg: Config, key: str) -> ImageDef:
    """Resolve a physical or synthetic per-class image definition by key."""

    class_info = parse_class_roi_key(key)
    if class_info is not None:
        base_key, class_index = class_info
        return make_class_roi_def(get_image_def(cfg, base_key), class_index)

    for image_def in cfg.images:
        if image_def.key == key:
            return image_def
    raise ValueError(f"Unknown image key: {key}")


def get_roi_image_defs(cfg: Config) -> List[ImageDef]:
    """Return definitions capable of producing or supplying ROI masks."""

    return [
        image_def
        for image_def in cfg.images
        if image_def.model_path is not None or getattr(image_def, "combined_mask_source_keys", None)
    ]


def get_selected_measurement_roi_defs(cfg: Config) -> List[ImageDef]:
    """Return selected ROI definitions in user-visible order."""

    roi_def_by_key = {image_def.key: image_def for image_def in get_roi_image_defs(cfg)}
    ordered_keys: List[str] = []

    for key in list(getattr(cfg, "overlay_roi_keys", []) or []):
        if key in roi_def_by_key and key not in ordered_keys:
            ordered_keys.append(key)

    reference_key = getattr(cfg, "per_cell_mask_source", "") or ""
    if reference_key in roi_def_by_key and reference_key not in ordered_keys:
        ordered_keys.append(reference_key)

    result: List[ImageDef] = []
    for key in ordered_keys:
        result.extend(expand_image_def_to_class_roi_defs(roi_def_by_key[key]))
    return result


def get_required_roi_defs_for_targets(
    cfg: Config,
    targets: Sequence[TargetConfig],
) -> List[ImageDef]:
    """Return required ROI definitions with combined-mask dependencies first."""

    roi_def_by_key = {image_def.key: image_def for image_def in get_roi_image_defs(cfg)}
    ordered_keys: List[str] = []

    for target in targets:
        for key in list(target.overlay_roi_keys or []):
            if key in roi_def_by_key and key not in ordered_keys:
                ordered_keys.append(key)

        reference_key = str(target.per_cell_mask_source or "")
        if reference_key in roi_def_by_key and reference_key not in ordered_keys:
            ordered_keys.append(reference_key)

    expanded_keys: List[str] = []
    visiting: set[str] = set()

    # Guard here as well as validation because planning helpers are also used by Check Setup.
    def add_with_dependencies(key: str) -> None:
        image_def = roi_def_by_key.get(key)
        if image_def is None:
            return
        if key in visiting:
            raise ValueError(f"Combined mask dependency cycle involving {image_def.label}.")
        if key in expanded_keys:
            return
        visiting.add(key)
        for source_key in getattr(image_def, "combined_mask_source_keys", []) or []:
            add_with_dependencies(source_key)
        visiting.remove(key)
        expanded_keys.append(key)

    for key in ordered_keys:
        add_with_dependencies(key)
    return [roi_def_by_key[key] for key in expanded_keys]


def expand_roi_keys_to_class_keys(
    cfg: Config,
    roi_keys: Sequence[str],
) -> List[str]:
    """Expand configured Weka ROI keys into concrete per-class keys."""

    expanded: List[str] = []
    for key in roi_keys:
        if key.startswith("__") or parse_class_roi_key(key) is not None:
            expanded.append(key)
            continue
        try:
            image_def = get_image_def(cfg, key)
        except ValueError:
            expanded.append(key)
            continue
        class_defs = expand_image_def_to_class_roi_defs(image_def)
        expanded.extend(class_def.key for class_def in class_defs)
    return dedupe_preserve_order(expanded)


def mask_source_results_dir(cfg: Config) -> Path:
    """Return the Results directory from which reusable masks are loaded."""

    return results_root(Path(getattr(cfg, "mask_source_dir", None) or cfg.output_dir))


def existing_weka_threshold_mask_status(
    cfg: Config,
    roi_defs: List[ImageDef],
    result_id: str,
) -> Dict[str, bool]:
    """Report whether every configured class mask is reusable for each Weka ROI."""

    results_dir = mask_source_results_dir(cfg)
    threshold_dir = results_dir / "Masks" / "MaskImages"
    status: Dict[str, bool] = {}

    for image_def in roi_defs:
        if getattr(image_def, "combined_mask_source_keys", None):
            status[image_def.key] = True
            continue
        classifier_tag = base_name_no_ext(image_def.model_path) if image_def.model_path is not None else ""
        class_indices = parse_probability_class_indices(
            image_def.probability_class_index
            if image_def.probability_class_index is not None
            else cfg.probability_class_index
        )
        try:
            status[image_def.key] = all(
                threshold_mask_path(threshold_dir, result_id, image_def.label, classifier_tag, class_index, target_key=image_def.key).is_file()
                for class_index in class_indices
            )
        except ArtifactMetadataError:
            status[image_def.key] = False
    return status


def existing_cellpose_label_status(
    cfg: Config,
    targets: Sequence[TargetConfig],
    result_id: str,
) -> Dict[str, bool]:
    """Report reusable Cellpose labels keyed by their reusable mask owner."""

    results_dir = mask_source_results_dir(cfg)
    status: Dict[str, bool] = {}

    for target in targets:
        if not target.do_cell_segmentation:
            continue
        mask_key = target.cell_segmentation_mask_source or target.source_image_key
        if mask_key in status:
            continue
        try:
            source_def = get_image_def(cfg, mask_key)
        except ValueError:
            continue
        try:
            status[mask_key] = saved_cell_labels(results_dir, result_id, mask_key, source_def.label).is_file()
        except ArtifactMetadataError:
            status[mask_key] = False
    return status


class _ReusableMaskShapeInspector:
    """Cache source and saved-mask shapes for one setup-check pass."""

    def __init__(self, file_map: Optional[Dict[str, Optional[Path]]] = None):
        self.file_map = file_map or {}
        self._image_shapes: dict[tuple[str, Optional[int], str, Optional[int]], tuple[int, int]] = {}
        self._saved_shapes: dict[Path, tuple[int, ...]] = {}

    # Compare masks against the exact channel/Z plane that execution will measure.
    def image_shape(self, image_def: ImageDef) -> Optional[tuple[int, int]]:
        source_path = self.file_map.get(image_def.key)
        if source_path is None:
            return None
        cache_key = (
            str(source_path),
            image_def.stack_channel_index,
            getattr(image_def, "stack_z_mode", "max_projection"),
            getattr(image_def, "stack_z_index", None),
        )
        if cache_key not in self._image_shapes:
            arr = read_tiff_numpy_2d(
                source_path,
                stack_channel_index=image_def.stack_channel_index,
                stack_z_mode=cache_key[2],
                stack_z_index=cache_key[3],
            )
            self._image_shapes[cache_key] = (int(arr.shape[0]), int(arr.shape[1]))
        return self._image_shapes[cache_key]

    # Squeezing matches how saved binary masks become two-dimensional arrays during execution.
    def saved_shape(self, path: Path) -> tuple[int, ...]:
        path = Path(path)
        if path not in self._saved_shapes:
            self._saved_shapes[path] = tuple(int(value) for value in tifffile.imread(path).squeeze().shape)
        return self._saved_shapes[path]


def _reusable_roi_mask_warnings(
    cfg: Config,
    roi_defs: List[ImageDef],
    result_id: str,
    threshold_dir: Path,
    inspector: _ReusableMaskShapeInspector,
) -> List[str]:
    warnings: List[str] = []
    weka_status = existing_weka_threshold_mask_status(cfg, roi_defs, result_id)
    for image_def in roi_defs:
        if getattr(image_def, "combined_mask_source_keys", None):
            continue
        if weka_status.get(image_def.key, False):
            expected_shape = inspector.image_shape(image_def)
            if image_def.model_path is not None and expected_shape is not None:
                classifier_tag = base_name_no_ext(image_def.model_path)
                for class_index in parse_probability_class_indices(image_def.probability_class_index):
                    mask_path = threshold_mask_path(
                        threshold_dir,
                        result_id,
                        image_def.label,
                        classifier_tag,
                        class_index,
                        target_key=image_def.key,
                    )
                    try:
                        mask_shape = inspector.saved_shape(mask_path)
                        if mask_shape != expected_shape:
                            warnings.append(
                                f"Reusable Weka mask shape mismatch for "
                                f"{image_def.label} class {class_index}: "
                                f"mask {mask_shape}, image {expected_shape}. "
                                f"File: {mask_path}"
                            )
                    except Exception as exc:
                        warnings.append(f"Could not validate reusable Weka mask " f"{mask_path}: {exc}")
            continue
        if not getattr(cfg, "reuse_existing_masks", False):
            continue
        classifier_tag = base_name_no_ext(image_def.model_path) if image_def.model_path is not None else ""
        classes = parse_probability_class_indices(
            image_def.probability_class_index
            if image_def.probability_class_index is not None
            else cfg.probability_class_index
        )
        expected = ", ".join(
            f"{result_id}_{image_def.label}_class{class_index}_thr_{classifier_tag}.tif" for class_index in classes
        )
        warnings.append(
            f"Weka mask not found for {image_def.label}; it will be regenerated. "
            f"Expected: {expected}. Matching depends on result ID, channel label, "
            "classifier filename, and probability class."
        )
    return warnings


def _reusable_cellpose_warnings(
    cfg: Config,
    targets: Sequence[TargetConfig],
    result_id: str,
    results_dir: Path,
    inspector: _ReusableMaskShapeInspector,
) -> List[str]:
    warnings: List[str] = []
    cellpose_status = existing_cellpose_label_status(cfg, targets, result_id)
    checked_mask_keys: set[str] = set()
    for target in targets:
        if not target.do_cell_segmentation:
            continue
        mask_key = target.cell_segmentation_mask_source or target.source_image_key
        if mask_key in checked_mask_keys:
            continue
        checked_mask_keys.add(mask_key)
        if cellpose_status.get(mask_key, False):
            try:
                source_def = get_image_def(
                    cfg,
                    target.cell_segmentation_source or target.source_image_key,
                )
                expected_shape = inspector.image_shape(source_def)
                mask_def = get_image_def(cfg, mask_key)
                label_path = saved_cell_labels(results_dir, result_id, mask_key, mask_def.label)
                if expected_shape is not None:
                    label_shape = inspector.saved_shape(label_path)
                    if label_shape != expected_shape:
                        warnings.append(
                            f"Reusable Cellpose label shape mismatch for "
                            f"{mask_def.label}: labels {label_shape}, "
                            f"image {expected_shape}. File: {label_path}"
                        )
            except Exception as exc:
                warnings.append(f"Could not validate reusable Cellpose labels for " f"{mask_key}: {exc}")
            continue
        try:
            source_def = get_image_def(cfg, mask_key)
        except ValueError:
            continue
        expected = f"{result_id}_{source_def.label}_01_cellpose_labels.tif"
        warnings.append(
            f"Cellpose labels not found for {source_def.label}; they will be regenerated. "
            f"Expected: {expected}. Matching depends on result ID and channel label."
        )
    return warnings


def reusable_mask_warnings(
    cfg: Config,
    targets: Sequence[TargetConfig],
    result_id: str,
    file_map: Optional[Dict[str, Optional[Path]]] = None,
) -> List[str]:
    """Explain mask reuse failures that will cause regeneration before a run."""

    roi_defs = get_required_roi_defs_for_targets(cfg, targets)
    if not getattr(cfg, "reuse_existing_masks", False):
        return []

    results_dir = mask_source_results_dir(cfg)
    inspector = _ReusableMaskShapeInspector(file_map)
    warnings = _reusable_roi_mask_warnings(
        cfg,
        roi_defs,
        result_id,
        results_dir / "Masks" / "MaskImages",
        inspector,
    )
    if getattr(cfg, "reuse_existing_masks", False):
        warnings.extend(_reusable_cellpose_warnings(cfg, targets, result_id, results_dir, inspector))
    return warnings


def image_keys_needed_for_processing(
    cfg: Config,
    targets: Sequence[TargetConfig],
    required_roi_defs: List[ImageDef],
    result_id: str,
) -> set[str]:
    """Return channels needed by measurements, overlays, masks, or processing."""

    needed: set[str] = set()

    def needs_mask_reference(image_def: ImageDef) -> bool:
        return any(
            isinstance(step, dict)
            and bool(step.get("enabled", True))
            and bool(step.get("type"))
            for step in list(getattr(image_def, "mask_processing_steps", []) or [])
        )

    def include_combined_reference(image_def: ImageDef) -> None:
        source_keys = list(getattr(image_def, "combined_mask_source_keys", []) or [])
        if source_keys and needs_mask_reference(image_def):
            needed.update(mask_reference_image_keys(source_keys, cfg.images))

    for target in targets:
        needed.add(target.source_image_key)
        needed.add(target.overlay_base_image_key or target.source_image_key)

    if not getattr(cfg, "reuse_existing_masks", False):
        for image_def in required_roi_defs:
            if getattr(image_def, "combined_mask_source_keys", None):
                include_combined_reference(image_def)
                continue
            needed.add(image_def.key)
        for target in targets:
            if target.do_cell_segmentation and target.cell_segmentation_source:
                needed.add(target.cell_segmentation_source)
        return {key for key in needed if key}

    weka_status = existing_weka_threshold_mask_status(
        cfg,
        required_roi_defs,
        result_id,
    )
    for image_def in required_roi_defs:
        if getattr(image_def, "combined_mask_source_keys", None):
            include_combined_reference(image_def)
            continue
        if not weka_status.get(image_def.key, False):
            needed.add(image_def.key)
            continue

        has_background_subtraction = bool(measurement_background_radii(image_def))
        if has_background_subtraction or needs_mask_reference(image_def):
            needed.add(image_def.key)

    for target in targets:
        segmentation_key = target.cell_segmentation_source or target.source_image_key
        if target.do_cell_segmentation:
            needed.add(segmentation_key)

    return {key for key in needed if key}


def pipeline_needs_imagej(
    cfg: Config,
    targets: Optional[Sequence[TargetConfig]] = None,
) -> bool:
    """Return whether the complete run plan requires Fiji/ImageJ objects."""

    targets = targets if targets is not None else get_enabled_measurement_targets(cfg)
    if get_required_roi_defs_for_targets(cfg, targets):
        return True

    for target in targets:
        try:
            source_def = get_image_def(cfg, target.source_image_key)
        except ValueError:
            continue
        if measurement_background_radii(source_def):
            return True
        if measurement_requires_imagej_processing(source_def):
            return True
    return False
