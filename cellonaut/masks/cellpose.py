"""Pipeline integration for Cellpose whole-cell segmentation.

This layer prepares the selected source image, chooses the Cellpose backend,
reuses compatible segmentation outputs when possible, and returns masks/tables
in the format expected by Cellonaut measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from cellonaut.runtime import PipelineRuntime, PipelineStage
from cellonaut.cell_segmentation.core import (
    CellSegmentationConfig,
    export_cell_segmentation_diagnostics,
    export_per_cell_organelle_signal_tables,
    make_per_cell_table,
    run_cell_segmentation_on_image,
)
from cellonaut.io.writers import write_dataframe_csv, write_tiff
from cellonaut.results.artifacts import record_artifact, saved_cell_labels, ArtifactMetadataError
from cellonaut.results.layout import results_root
from cellonaut.masks.cell_qc import (
    append_cell_qc_summary_rows,
    cell_label_outline_mask,
)
from cellonaut.io.image_io import imageplus_to_numpy_2d
from cellonaut.measurement.math import summarize_per_cell_table
from cellonaut.measurement.table_cleanup import drop_derived_ratio_columns
from cellonaut.pipeline.discovery import build_common_results_export_dirs
from cellonaut.pipeline.cancellation import check_cancel
from cellonaut.pipeline.planning import get_image_def
from cellonaut.masks.roi_processing import create_mask_from_roi
from cellonaut.masks.adjustments import adjust_label_image
from cellonaut.system.gpu_detection import cellpose_acceleration_enabled


WHOLE_CELL_MEASUREMENT_COLUMNS = {
    "cell_area": (("Area", "CellArea"),),
    "cell_perimeter": (("Perimeter", "CellPerimeter"),),
    "cell_mean": (("Mean", "CellMean"),),
    "cell_min_max": (("Min", "CellMin"), ("Max", "CellMax")),
    "cell_median": (("Median", "CellMedian"),),
    "cell_raw_intden": (("RawIntDen", "CellIntDen"),),
}


def selected_whole_cell_measurements(
    cell_labels: np.ndarray,
    intensity_image: np.ndarray,
    measurement_options: Dict[str, bool],
) -> pd.DataFrame:
    """Measure the current channel inside each cell and retain selected columns."""
    measured = make_per_cell_table(cell_labels, intensity_image)
    selected = pd.DataFrame({"CellID": measured["CellID"]})
    for option_key, definitions in WHOLE_CELL_MEASUREMENT_COLUMNS.items():
        if not measurement_options.get(option_key, False):
            continue
        for source_column, output_column in definitions:
            selected[output_column] = measured[source_column]
    return selected


# Reuse only the canonical label file so settings checks and runtime loading agree.
def load_existing_cellpose_labels(
    mask_source_dir: Path,
    result_id: str,
    source_label: str,
    log_func: Optional[Callable[[str], None]] = None,
    relative_subdir: Optional[Path] = None,
    *,
    source_key: str = "",
) -> Optional[np.ndarray]:
    try:
        label_path = saved_cell_labels(results_root(mask_source_dir), result_id, source_key, source_label, relative_subdir or Path())
    except ArtifactMetadataError as exc:
        if log_func is not None:
            log_func(f"[WARN] Could not resolve reusable Cellpose labels: {exc}")
        return None

    if not label_path.exists():
        return None
    try:
        import tifffile

        labels = np.asarray(tifffile.imread(label_path))
        if labels.ndim > 2:
            labels = np.squeeze(labels)
        if labels.ndim != 2:
            if log_func is not None:
                log_func(
                    f"[WARN] Existing Cellpose labels are not 2D and cannot be reused: "
                    f"{label_path} (shape={labels.shape})"
                )
            return None
        # Reject invalid IDs before casting so reused cells are never merged or renumbered.
        valid_dtype = labels.dtype.kind in "buif"
        valid_ids = (
            valid_dtype
            and bool(np.all(np.isfinite(labels)))
            and bool(np.all(labels >= 0))
            and (labels.size == 0 or labels.max().item() <= np.iinfo(np.int32).max)
        )
        if valid_ids and labels.dtype.kind == "f":
            valid_ids = bool(np.all(labels == np.floor(labels)))
        if not valid_ids:
            if log_func is not None:
                log_func(f"[WARN] Existing Cellpose labels contain invalid cell IDs and cannot be reused: {label_path}")
            return None
        return labels.astype(np.int32, copy=False)
    except Exception as exc:
        if log_func is not None:
            log_func(f"[WARN] Could not read existing Cellpose labels {label_path}: " f"{type(exc).__name__}: {exc}")
        return None


# Build Cellpose settings from the pipeline configuration.
def build_cell_segmentation_cfg_from_pipeline(cfg: Any) -> CellSegmentationConfig:
    return CellSegmentationConfig(
        diameter=cfg.cell_diameter,
        min_size=cfg.cell_min_size,
        use_gpu=cellpose_acceleration_enabled(bool(cfg.cell_use_gpu)),
        gpu_requested=bool(cfg.cell_use_gpu),
        model_type=cfg.cellpose_model_type,
        custom_model_path=cfg.cellpose_custom_model_path,
        cellprob_threshold=cfg.cellprob_threshold,
        flow_threshold=cfg.flow_threshold,
        remove_border=cfg.cell_remove_border,
        save_rois_csv=True,
        save_qc_overlay=True,
        show_labels_in_qc=True,
    )


def _load_or_generate_cell_labels(
    *,
    cfg: Any,
    cell_source_img: Any,
    cell_source_label: str,
    source_def: Any,
    export_dirs: Dict[str, Path],
    result_id: str,
    id_label: str,
    log_func: Callable[[str], None],
    should_cancel: Optional[Callable[[], bool]],
    runtime: PipelineRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray, CellSegmentationConfig]:
    labels = None
    if getattr(cfg, "reuse_existing_masks", False):
        mask_source_dir = Path(getattr(cfg, "mask_source_dir", None) or cfg.output_dir)
        current_root = build_common_results_export_dirs(cfg.output_dir)["cell_segmentation_labels"]
        try:
            relative_subdir = export_dirs["cell_segmentation_labels"].relative_to(current_root)
        except ValueError:
            relative_subdir = Path()
        labels = load_existing_cellpose_labels(
            mask_source_dir,
            result_id,
            source_def.label,
            log_func=log_func,
            relative_subdir=relative_subdir,
            source_key=str(getattr(source_def, "key", source_def.label)),
        )
        if labels is not None:
            log_func(f"[{id_label}] Reused existing Cellpose labels for {source_def.label} from {mask_source_dir / 'Results'}")
            log_func(
                f"[{id_label}] Cellpose model, diameter, probability threshold, flow threshold, minimum cell area, "
                "and border-removal settings are not used to regenerate reused labels. "
                "Current measurement settings and filter rules are still applied."
            )
        else:
            log_func(
                f"[{id_label}] Existing Cellpose labels not found for {source_def.label}; "
                "generating new labels with current settings."
            )

    reused_labels = labels is not None
    segmentation_cfg = build_cell_segmentation_cfg_from_pipeline(cfg)
    if labels is None:
        if cell_source_img is None:
            raise ValueError("Cell mask image could not be opened.")
        if runtime is not None:
            runtime.stage(PipelineStage.SEGMENTATION, "Finding cells")
        log_func(f"[{id_label}] Building cell masks on {cell_source_label}")
        cell_source_arr = imageplus_to_numpy_2d(cell_source_img)
        labels = run_cell_segmentation_on_image(
            cell_source_arr,
            segmentation_cfg,
            log_func=log_func,
            should_cancel=should_cancel,
            runtime=runtime,
        )
    else:
        if cell_source_img is None:
            raise ValueError("Cellpose source image is required to measure reused cells.")
        cell_source_arr = imageplus_to_numpy_2d(cell_source_img)

    adjustments = dict(getattr(cfg, "cell_mask_adjustments", {}) or {})
    if reused_labels and any(int(value or 0) for value in adjustments.values()):
        log_func(f"[{id_label}] Reused final Cellpose labels retain their saved adjustments; current adjustments are not reapplied.")
    if not reused_labels and any(int(value or 0) for value in adjustments.values()):
        labels = adjust_label_image(labels, **adjustments).astype(np.int32, copy=False)
        log_func(f"[{id_label}] Applied persisted Cellpose mask adjustments: {adjustments}")

    label_path = export_dirs["cell_segmentation_labels"] / f"{result_id}_{source_def.label}_01_cellpose_labels.tif"
    outline_path = export_dirs["cell_segmentation_outlines"] / f"{result_id}_{source_def.label}_cellpose_outline.tif"
    write_tiff(label_path, np.asarray(labels).astype(np.int32, copy=False))
    write_tiff(outline_path, cell_label_outline_mask(labels))
    for path, kind in ((label_path, "cell_labels"), (outline_path, "cell_outline")):
        record_artifact(path, sample=result_id, target=str(getattr(source_def, "key", source_def.label)), label=source_def.label, kind=kind)
    return labels, cell_source_arr, segmentation_cfg


def _measure_per_cell_roi_signals(
    *,
    cfg: Any,
    roi_defs: List[Any],
    roi_map: Dict[str, Any],
    analysis_cell_mask: np.ndarray,
    cell_source_img: Any,
    measurement_source_img: Any,
    measurement_source_arr: np.ndarray,
    source_measure_arr: np.ndarray,
    source_def: Any,
    cell_source_label: str,
    has_source_background_subtraction: bool,
    export_dirs: Dict[str, Path],
    result_id: str,
    save_detailed_tables: bool,
    row: Dict[str, Any],
) -> None:
    for roi_def in roi_defs:
        if roi_def.key not in roi_map:
            raise ValueError(f"Measurement mask is unavailable: {roi_def.label}")
        roi = roi_map[roi_def.key]
        roi_mask_imp = None
        try:
            if roi is None:
                roi_mask_arr = np.zeros(analysis_cell_mask.shape, dtype=bool)
            else:
                roi_mask_imp = create_mask_from_roi(
                    cell_source_img or measurement_source_img,
                    roi,
                    f"{result_id}_{roi_def.label}_mask_for_per_cell",
                )
                roi_mask_arr = imageplus_to_numpy_2d(roi_mask_imp) > 0
            corrected_arr = measurement_source_arr if has_source_background_subtraction else None
            biology_csv = (
                export_dirs["cell_signal_tables"]
                / f"{result_id}_{source_def.label}_{roi_def.label}_per_cell_signal.csv"
            )
            biology_df, _geometry_df = export_per_cell_organelle_signal_tables(
                cell_label_img=analysis_cell_mask,
                organelle_mask=roi_mask_arr,
                intensity_img=source_measure_arr,
                out_biology_csv=biology_csv if save_detailed_tables else None,
                out_geometry_csv=None,
                organelle_prefix=roi_def.label,
                corrected_intensity_img=corrected_arr,
                measurement_options=dict(getattr(cfg, "measurement_options", {}) or {}),
            )
            if save_detailed_tables:
                record_artifact(biology_csv, sample=result_id, target=str(getattr(source_def, "key", source_def.label)), label=source_def.label, kind="cell_signal", mask=roi_def.key, mask_label=roi_def.label)
            summary = summarize_per_cell_table(
                biology_df,
                roi_def.label,
                source_def.label,
                cell_source_label,
            )
            row.update(summary)
        except Exception as exc:
            raise RuntimeError(
                f"Could not calculate or export per-cell signal measurements for {roi_def.label}: {exc}"
            ) from exc
        finally:
            try:
                if roi_mask_imp is not None:
                    roi_mask_imp.close()
            except Exception:
                pass


def run_cell_segmentation_for_sample(
    cfg: Any,
    image_map: Dict[str, Any],
    roi_map: Dict[str, Any],
    roi_defs: List[Any],
    row: Dict[str, Any],
    source_def: Any,
    source_measure_img: Any,
    measurement_source_img: Any,
    has_source_background_subtraction: bool,
    export_dirs: Dict[str, Path],
    result_id: str,
    id_label: str,
    log_func: Callable[[str], None],
    should_cancel: Optional[Callable[[], bool]] = None,
    runtime: PipelineRuntime | None = None,
) -> tuple[Dict[str, np.ndarray], str]:
    """Add original per-cell measurements to row and write segmentation artifacts.

    The configured Cellpose source defines labels and filter-intensity metrics;
    source_measure_img supplies measured-channel values before background
    subtraction, and measurement_source_img supplies corrected values when enabled.
    Borrow input images without closing them. Return overlay masks (including
    integer cell labels) and warning text, or ({}, "") when segmentation is off.
    Cell Group exclusions are applied later, never to these original tables.
    """
    extra_overlay_masks: Dict[str, np.ndarray] = {}
    warning_text = ""

    if not cfg.do_cell_segmentation:
        return extra_overlay_masks, warning_text

    check_cancel(should_cancel)

    cell_source_def = get_image_def(cfg, cfg.cell_segmentation_source)
    cell_source_img = image_map.get(cell_source_def.key)
    cell_source_label = cell_source_def.label
    measurement_source_arr = imageplus_to_numpy_2d(measurement_source_img)
    source_measure_arr = imageplus_to_numpy_2d(source_measure_img)

    whole_cell_mask, cell_source_arr, cell_segmentation_cfg = _load_or_generate_cell_labels(
        cfg=cfg,
        cell_source_img=cell_source_img,
        cell_source_label=cell_source_label,
        source_def=source_def,
        export_dirs=export_dirs,
        result_id=result_id,
        id_label=id_label,
        log_func=log_func,
        should_cancel=should_cancel,
        runtime=runtime,
    )
    extra_overlay_masks["__whole_cell_mask__"] = whole_cell_mask

    cell_table = make_per_cell_table(
        whole_cell_mask,
        cell_source_arr,
    )
    measurement_options = dict(getattr(cfg, "measurement_options", {}) or {})
    measured_cell_table = selected_whole_cell_measurements(
        whole_cell_mask,
        source_measure_arr,
        measurement_options,
    )
    if cell_source_label != source_def.label:
        log_func(f"[{id_label}] Cell groups use cell mask image: {cell_source_label}")

    # Pipeline tables are the immutable measured dataset. Cell groups and
    # exclusions are exploratory derivatives exported later from Image Preview Tools.
    total_cell_count = int(len(cell_table))
    row[f"{source_def.label}_CellCount"] = total_cell_count
    export_source_table = cell_table.merge(measured_cell_table, on="CellID", how="left")
    export_table = append_cell_qc_summary_rows(
        export_source_table,
        include_filtered_summary=False,
        all_cells_table=export_source_table,
    )
    whole_cell_summary = summarize_per_cell_table(
        measured_cell_table,
        "",
        source_def.label,
        cell_source_label,
    )
    row.update(whole_cell_summary)
    analysis_cell_mask = whole_cell_mask

    diagnostics = export_cell_segmentation_diagnostics(
        label_img=whole_cell_mask,
        base_img=cell_source_arr,
        qc_png_dir=export_dirs["cell_segmentation_qc_pngs"],
        table_dir=export_dirs["cell_segmentation_tables"],
        base_name=f"{result_id}_{source_def.label}",
        cfg=cell_segmentation_cfg,
    )

    export_table_path = (
        export_dirs["cell_segmentation_tables"] / f"{result_id}_{source_def.label}_cell_measurements.csv"
    )
    write_dataframe_csv(drop_derived_ratio_columns(export_table), export_table_path, index=False)
    record_artifact(export_table_path, sample=result_id, target=str(getattr(source_def, "key", source_def.label)), label=source_def.label, kind="cell_table")
    if diagnostics and diagnostics.get("qc_overlay"):
        record_artifact(Path(diagnostics["qc_overlay"]), sample=result_id, target=str(getattr(source_def, "key", source_def.label)), label=source_def.label, kind="cell_png")

    _measure_per_cell_roi_signals(
        cfg=cfg,
        roi_defs=roi_defs,
        roi_map=roi_map,
        analysis_cell_mask=analysis_cell_mask,
        cell_source_img=cell_source_img,
        measurement_source_img=measurement_source_img,
        measurement_source_arr=measurement_source_arr,
        source_measure_arr=source_measure_arr,
        source_def=source_def,
        cell_source_label=cell_source_label,
        has_source_background_subtraction=has_source_background_subtraction,
        export_dirs=export_dirs,
        result_id=result_id,
        save_detailed_tables=True,
        row=row,
    )

    return extra_overlay_masks, warning_text
