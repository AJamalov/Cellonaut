"""Measurement execution for one sample and one measured channel.

The functions here combine image data, configured masks, optional Cellpose
cells, review overlays, and table exports into the final per-target measurement
outputs used by full runs and previews.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from cellonaut.measurement.math import rad_tag
from cellonaut.masks.overlay_exports import save_general_flat_qc_overlay, save_general_overlay_stack
from cellonaut.pipeline.roi_defs import dedupe_preserve_order
from cellonaut.masks.roi_processing import measure_roi_stats, measure_stats, subtract_background_copy
from cellonaut.pipeline.cancellation import check_cancel
from cellonaut.pipeline.models import Config
from cellonaut.pipeline.planning import expand_roi_keys_to_class_keys


def is_measurement_enabled(cfg: Config, key: str, default: bool = True) -> bool:
    """Keep distinct opt-in defaults for core and optional measurements."""
    return bool(cfg.measurement_options.get(key, default))


# Build review overlays from the masks used for the completed measurement target.
def save_sample_overlay_outputs(
    cfg: Any,
    image_map: Dict[str, Any],
    roi_map: Dict[str, Any],
    extra_overlay_masks: Dict[str, np.ndarray],
    export_dirs: Dict[str, Path],
    result_id: str,
    id_label: str,
    overlay_base_img: Any,
    overlay_base_def: Any,
    native_numpy_images: bool,
    warning_text: str,
    log_func: Callable[[str], None],
) -> None:
    overlay_roi_keys = expand_roi_keys_to_class_keys(cfg, list(cfg.overlay_roi_keys or []))

    if cfg.do_cell_segmentation and cfg.overlay_whole_cell_mask:
        overlay_roi_keys.append("__whole_cell_mask__")

    if "__flagged_cell_mask__" in extra_overlay_masks:
        overlay_roi_keys.append("__flagged_cell_mask__")

    overlay_roi_keys = dedupe_preserve_order(
        [key for key in overlay_roi_keys if (key in roi_map or key in extra_overlay_masks)]
    )

    skipped_native_roi_keys: list[str] = []
    if native_numpy_images and overlay_roi_keys:
        skipped_native_roi_keys = [key for key in overlay_roi_keys if key not in extra_overlay_masks]
        overlay_roi_keys = [key for key in overlay_roi_keys if key in extra_overlay_masks]
        if skipped_native_roi_keys:
            log_func(
                f"[{id_label}] Native TIFF mode is active; skipping ImageJ mask overlay keys: "
                f"{skipped_native_roi_keys}"
            )
    if native_numpy_images and not overlay_roi_keys and skipped_native_roi_keys:
        log_func(f"[{id_label}] No NumPy-compatible overlay masks selected; skipping overlay outputs")

    if not overlay_roi_keys:
        log_func(f"[{id_label}] No overlay mask keys selected; skipping overlay outputs")
        return

    log_func(
        f"[{id_label}] Saving overlay outputs on base image {overlay_base_def.label} "
        f"with overlay keys: {overlay_roi_keys}"
    )

    save_general_overlay_stack(
        base_img=overlay_base_img,
        roi_map=roi_map,
        roi_keys=overlay_roi_keys,
        out_path=export_dirs["mask_overlays"],
        result_id=result_id,
        base_label=overlay_base_def.label,
        cfg=cfg,
        log_func=log_func,
        extra_mask_map=extra_overlay_masks,
        binary_out_path=export_dirs["final_binary_masks"],
        image_map=image_map,
        write_overlay=True,
    )

    if native_numpy_images:
        log_func(f"[{id_label}] Native TIFF mode is active; skipping flat ImageJ review PNG overlay")
    else:
        save_general_flat_qc_overlay(
            base_img=overlay_base_img,
            roi_map=roi_map,
            roi_keys=overlay_roi_keys,
            out_path=export_dirs["qc_overlay_pngs"],
            result_id=result_id,
            warning_text=warning_text,
            base_label=overlay_base_def.label,
            cfg=cfg,
            log_func=log_func,
            extra_mask_map=extra_overlay_masks,
        )


def measure_rois_for_source(
    cfg: Any,
    roi_defs: List[Any],
    roi_map: Dict[str, Any],
    row: Dict[str, Any],
    source_def: Any,
    source_measure_img: Any,
    measurement_source_img: Any,
    has_source_background_subtraction: bool,
    corrected_suffix: str,
    id_label: str,
    log_func: Callable[[str], None],
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Add selected ROI measurements to row, borrowing the prepared source images.

    source_measure_img is before background subtraction, but may already include
    measurement preprocessing. measurement_source_img supplies the corrected
    integrated density when enabled; both use the same ROI. Missing ROI keys
    are errors, while a present None ROI means zero area/signal, not whole-image
    measurement. This function does not close the borrowed images.
    """
    for roi_def in roi_defs:
        check_cancel(should_cancel)

        if roi_def.key not in roi_map:
            raise ValueError(f"Measurement mask is unavailable: {roi_def.label}")
        roi = roi_map[roi_def.key]
        # An empty, successfully prepared mask has zero area and total signal;
        # intensity distributions and geometry remain undefined.
        stats = measure_roi_stats(source_measure_img, roi) if roi is not None else {"Area": 0.0, "RawIntDen": 0.0}
        area = stats.get("Area", np.nan)
        mean = stats.get("Mean", np.nan)
        int_den = stats.get("RawIntDen", stats.get("IntDen", np.nan))

        corrected_intden = np.nan
        if has_source_background_subtraction:
            corrected_stats = measure_roi_stats(measurement_source_img, roi) if roi is not None else {"RawIntDen": 0.0}
            corrected_intden = corrected_stats.get("RawIntDen", corrected_stats.get("IntDen", np.nan))

        col_prefix = f"{source_def.label}_in_{roi_def.label}"

        # Record every selected measurement in the original result.
        def write_measurement(suffix: str, value: Any, prefix: str = col_prefix) -> None:
            row[f"{prefix}_{suffix}"] = value

        # Group Fiji statistics that share one user-facing measurement switch.
        def write_stat_group(option_key: str, suffixes: list[str], roi_stats: dict[str, Any] = stats) -> None:
            if not is_measurement_enabled(cfg, option_key, False):
                return
            for suffix in suffixes:
                write_measurement(suffix, roi_stats.get(suffix, np.nan))

        if is_measurement_enabled(cfg, "area", True):
            write_measurement("Area", area)

        if is_measurement_enabled(cfg, "mean", True):
            write_measurement("Mean", mean)

        write_stat_group("std_dev", ["StdDev"])
        write_stat_group("mode", ["Mode"])
        if (
            is_measurement_enabled(cfg, "min_max", False)
            or is_measurement_enabled(cfg, "min", False)
            or is_measurement_enabled(cfg, "max", False)
        ):
            write_measurement("Min", stats.get("Min", np.nan))
            write_measurement("Max", stats.get("Max", np.nan))
        write_stat_group("centroid", ["CentroidX", "CentroidY"])
        write_stat_group("center_of_mass", ["CenterOfMassX", "CenterOfMassY"])
        write_stat_group("perimeter", ["Perimeter"])
        write_stat_group(
            "bounding_rect",
            ["BoundingRectX", "BoundingRectY", "BoundingRectWidth", "BoundingRectHeight"],
        )
        write_stat_group("fit_ellipse", ["EllipseMajor", "EllipseMinor", "EllipseAngle"])
        write_stat_group("feret", ["Feret", "FeretX", "FeretY", "FeretAngle", "MinFeret"])

        if is_measurement_enabled(cfg, "raw_intden", True):
            write_measurement("RawIntDen", int_den)

        write_stat_group("median", ["Median"])
        write_stat_group("skewness", ["Skewness"])
        write_stat_group("kurtosis", ["Kurtosis"])

        if has_source_background_subtraction:
            write_measurement(f"{corrected_suffix}_IntDen", corrected_intden)

        if has_source_background_subtraction:
            log_func(
                f"[{id_label}] {source_def.label} in {roi_def.label}: "
                f"area={area}, mean={mean}, intden={int_den}, {corrected_suffix.lower()}_intden={corrected_intden}"
            )
        else:
            log_func(
                f"[{id_label}] {source_def.label} in {roi_def.label}: " f"area={area}, mean={mean}, intden={int_den}"
            )


# Skip the primary radius because its corrected value was already measured in the main pass.
def run_source_background_sweep(
    roi_defs: List[Any],
    roi_map: Dict[str, Any],
    row: Dict[str, Any],
    source_def: Any,
    measurement_base_img: Any,
    source_radii: List[float],
    created_temp_images: list[Any],
    id_label: str,
    log_func: Callable[[str], None],
    should_cancel: Optional[Callable[[], bool]] = None,
    source_options: Optional[List[str]] = None,
) -> None:
    if source_radii:
        log_func(f"[{id_label}] Starting Subtract Background " f"({len(source_radii)} radii)")

    primary_radius = source_radii[0] if source_radii else None
    for index, rad in enumerate(source_radii):
        check_cancel(should_cancel)

        tag_rb = rad_tag(rad)
        if primary_radius is not None and float(rad) == float(primary_radius):
            continue
        log_func(f"[{id_label}] Subtract Background radius {rad} ({tag_rb})")

        options = source_options[index] if source_options and index < len(source_options) else ""
        sweep_img = (
            subtract_background_copy(measurement_base_img, rad, options)
            if options
            else subtract_background_copy(measurement_base_img, rad)
        )
        if sweep_img is None:
            continue

        created_temp_images.append(sweep_img)

        try:
            for roi_def in roi_defs:
                if roi_def.key not in roi_map:
                    raise ValueError(f"Measurement mask is unavailable: {roi_def.label}")
                roi = roi_map[roi_def.key]
                if roi is None:
                    row[f"{source_def.label}_in_{roi_def.label}_{tag_rb}_IntDen"] = 0.0
                    continue
                _area, _mean, int_den = measure_stats(sweep_img, roi)
                row[f"{source_def.label}_in_{roi_def.label}_{tag_rb}_IntDen"] = int_den
        finally:
            try:
                sweep_img.close()
            except Exception:
                pass
            if sweep_img in created_temp_images:
                created_temp_images.remove(sweep_img)
