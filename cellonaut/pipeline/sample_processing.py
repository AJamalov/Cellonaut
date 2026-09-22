"""Execution of measurements and outputs for one configured sample target."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from cellonaut.runtime import PipelineStage
from cellonaut.config.defaults import IMAGE_PROCESSING_SCOPE_MEASUREMENT
from cellonaut.io.imagej_runtime import get_java_classes
from cellonaut.masks.cellpose import run_cell_segmentation_for_sample
from cellonaut.masks.roi_processing import subtract_background_copy
from cellonaut.measurement.execution import (
    measure_rois_for_source,
    is_measurement_enabled,
    run_source_background_sweep,
    save_sample_overlay_outputs,
)
from cellonaut.measurement.math import rad_tag
from cellonaut.pipeline.cancellation import check_cancel
from cellonaut.pipeline.context import (
    build_sample_processing_context,
    SampleProcessingContext,
)
from cellonaut.pipeline.discovery import build_result_id, build_sample_label, get_protein_name_for_sample
from cellonaut.pipeline.image_processing import MEASUREMENT_TRANSFORM_STEP_TYPES, apply_imagej_processing_recipe_copy
from cellonaut.pipeline.models import Config, TargetConfig, config_for_measurement_target
from cellonaut.pipeline.planning import (
    get_image_def,
    get_selected_measurement_roi_defs,
    measurement_background_radii,
    measurement_background_requests,
)
from cellonaut.pipeline.processing_montage import save_cellpose_montage_for_sample
from cellonaut.pipeline.summary import selected_measurements_summary


def process_sample(
    sample_folder: Path,
    cfg: Config,
    log_func: Callable[[str], None],
    should_cancel: Callable[[], bool] | None = None,
    precomputed_context: SampleProcessingContext | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Process one sample and return original measurement row and status.

    The row is None when processing returns FAILED; success returns PROCESSED.
    A supplied context remains caller-owned; only contexts created here are
    closed here. Cancellation and processing exceptions propagate to the caller.
    """
    check_cancel(should_cancel)

    id_label = build_sample_label(sample_folder, cfg.input_structure)
    result_id = build_result_id(
        sample_folder,
        cfg.input_structure,
        input_dir=cfg.input_dir,
    )

    log_func("------------------------------------------------------------")
    log_func(f"[{id_label}] Starting sample processing")
    log_func(f"[{id_label}] Result ID: {result_id}")

    own_context = False
    if precomputed_context is None:
        precomputed_context = build_sample_processing_context(
            sample_folder=sample_folder,
            cfg=cfg,
            log_func=log_func,
            should_cancel=should_cancel,
        )
        own_context = True
    if precomputed_context is None:
        return None, "FAILED"

    created_temp_images: list[Any] = []
    try:
        export_dirs = precomputed_context.export_dirs
        image_map = precomputed_context.image_map
        roi_map = precomputed_context.roi_map
        native_numpy_images = precomputed_context.native_numpy_images
        duplicator = None
        if not native_numpy_images:
            duplicator = get_java_classes()["Duplicator"]
        check_cancel(should_cancel)
        source_def = get_image_def(cfg, cfg.source_image_key)
        source_img = image_map.get(source_def.key)
        if source_img is None:
            log_func(f"[{id_label}] Measured channel " f"{source_def.label} could not be opened")
            return None, "FAILED"

        overlay_base_key = cfg.overlay_base_image_key or cfg.source_image_key
        overlay_base_def = get_image_def(cfg, overlay_base_key)
        overlay_base_img = image_map.get(overlay_base_def.key)
        if overlay_base_img is None:
            overlay_base_def = source_def
            overlay_base_img = source_img

        source_radii = measurement_background_radii(source_def)
        background_requests = measurement_background_requests(source_def)
        if native_numpy_images and source_radii:
            raise RuntimeError(
                "Background-radius measurements require Fiji/ImageJ. "
                "Remove source background radii for the Cellpose-only "
                "native TIFF path."
            )

        # Measurement transforms are separated from background correction so
        # raw and background-subtracted values remain distinct.
        if native_numpy_images:
            source_measure_img = np.asarray(source_img)
        else:
            if duplicator is None:
                raise RuntimeError("ImageJ Duplicator is unavailable.")
            processed_measure_img, applied_measurement_steps = apply_imagej_processing_recipe_copy(
                source_img,
                source_def,
                IMAGE_PROCESSING_SCOPE_MEASUREMENT,
                include_step_types=MEASUREMENT_TRANSFORM_STEP_TYPES,
            )
            if processed_measure_img is source_img:
                source_measure_img = duplicator().run(source_img)
            else:
                source_measure_img = processed_measure_img
            created_temp_images.append(source_measure_img)
            if applied_measurement_steps:
                log_func(
                    f"[{id_label}] Applying measurement processing for "
                    f"{source_def.label}: {', '.join(applied_measurement_steps)}"
                )

        source_bg_radius = source_radii[0] if source_radii else 0.0
        has_source_background_subtraction = source_bg_radius > 0
        corrected_suffix = rad_tag(source_bg_radius) if has_source_background_subtraction else "Corrected"
        if has_source_background_subtraction:
            log_func(
                f"[{id_label}] Applying source background subtraction for "
                f"{source_def.label} with radius={source_bg_radius}"
            )
            primary_options = background_requests[0][1]
            bg_corrected_source = (
                subtract_background_copy(source_measure_img, source_bg_radius, primary_options)
                if primary_options
                else subtract_background_copy(source_measure_img, source_bg_radius)
            )
            created_temp_images.append(bg_corrected_source)
            measurement_source_img = bg_corrected_source
        else:
            measurement_source_img = source_measure_img

        precomputed_context.runtime.stage(PipelineStage.MEASUREMENT, "Measuring fluorescence")
        log_func(f"[{id_label}] Measurement config: " f"{selected_measurements_summary(cfg)}")
        log_func(
            f"[{id_label}] Core measurement modes | "
            f"raw integrated density={'ON' if is_measurement_enabled(cfg, 'raw_intden', True) else 'OFF'} | "
            f"background-subtracted integrated density={'ON' if has_source_background_subtraction else 'OFF'}"
        )
        file_map = precomputed_context.file_map
        source_file = file_map.get(source_def.key)
        sample_label = build_sample_label(sample_folder, cfg.input_structure)
        try:
            sample_relative_path = sample_folder.resolve().relative_to(cfg.input_dir.resolve()).as_posix()
        except ValueError:
            sample_relative_path = result_id
        row: dict[str, Any] = {
            "Label": sample_label,
            "SampleID": result_id,
            "OriginalSamplePath": str(sample_folder),
            "SampleRelativePath": sample_relative_path,
            "SampleGroup": get_protein_name_for_sample(sample_folder, cfg.input_structure),
            "SourceImageKey": source_def.key,
            "SourceImageLabel": source_def.label,
            "SourceImageFile": str(source_file) if source_file is not None else "",
        }
        row.update(precomputed_context.skeleton_metrics)
        roi_defs = get_selected_measurement_roi_defs(cfg)

        measure_rois_for_source(
            cfg=cfg,
            roi_defs=roi_defs,
            roi_map=roi_map,
            row=row,
            source_def=source_def,
            source_measure_img=source_measure_img,
            measurement_source_img=measurement_source_img,
            has_source_background_subtraction=has_source_background_subtraction,
            corrected_suffix=corrected_suffix,
            id_label=id_label,
            log_func=log_func,
            should_cancel=should_cancel,
        )
        run_source_background_sweep(
            roi_defs=roi_defs,
            roi_map=roi_map,
            row=row,
            source_def=source_def,
            measurement_base_img=source_measure_img,
            source_radii=source_radii,
            source_options=[options for _radius, options in background_requests],
            created_temp_images=created_temp_images,
            id_label=id_label,
            log_func=log_func,
            should_cancel=should_cancel,
        )
        extra_overlay_masks, warning_text = run_cell_segmentation_for_sample(
            cfg=cfg,
            image_map=image_map,
            roi_map=roi_map,
            roi_defs=roi_defs,
            row=row,
            source_def=source_def,
            source_measure_img=source_measure_img,
            measurement_source_img=measurement_source_img,
            has_source_background_subtraction=has_source_background_subtraction,
            export_dirs=export_dirs,
            result_id=result_id,
            id_label=id_label,
            log_func=log_func,
            should_cancel=should_cancel,
            runtime=precomputed_context.runtime,
            label_cache=precomputed_context.cellpose_label_cache,
        )
        if cfg.do_cell_segmentation and "__whole_cell_mask__" in extra_overlay_masks:
            try:
                cell_source_def = get_image_def(cfg, cfg.cell_segmentation_source)
                save_cellpose_montage_for_sample(
                    image_map=image_map,
                    cell_source_def=cell_source_def,
                    source_def=source_def,
                    extra_overlay_masks=extra_overlay_masks,
                    export_dir=export_dirs["processing_montages"],
                    result_id=result_id,
                    output_variant=str(getattr(cfg, "output_variant", "") or ""),
                    cell_mask_key=str(getattr(cfg, "cell_segmentation_mask_source", "") or ""),
                    log_func=log_func,
                )
            except Exception as exc:
                log_func(f"[WARN] [{id_label}] Could not save Cellpose montage PNG: {exc}")
        save_sample_overlay_outputs(
            cfg=cfg,
            image_map=image_map,
            roi_map=roi_map,
            extra_overlay_masks=extra_overlay_masks,
            export_dirs=export_dirs,
            result_id=result_id,
            id_label=id_label,
            overlay_base_img=overlay_base_img,
            overlay_base_def=overlay_base_def,
            native_numpy_images=native_numpy_images,
            warning_text=warning_text,
            log_func=log_func,
        )

        log_func(f"[{id_label}] Sample finished successfully")
        return row, "PROCESSED"
    finally:
        for image in created_temp_images:
            try:
                close = getattr(image, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        if own_context:
            precomputed_context.close()


def process_measurement_target_for_sample(
    sample_folder: Path,
    cfg: Config,
    target: TargetConfig,
    log_func: Callable[[str], None],
    should_cancel: Callable[[], bool] | None = None,
    precomputed_context: SampleProcessingContext | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Apply resolved target settings, process the sample, and label its result."""
    target_cfg = config_for_measurement_target(cfg, target)

    row, status = process_sample(
        sample_folder=sample_folder,
        cfg=target_cfg,
        log_func=log_func,
        should_cancel=should_cancel,
        precomputed_context=precomputed_context,
    )

    if status == "PROCESSED":
        target_label = get_image_def(cfg, target.source_image_key).label
        target_suffix = str(target_label or target.source_image_key)
        output_variant = str(getattr(target, "output_variant", "") or "").strip()
        if output_variant:
            target_suffix = f"{target_suffix}__{output_variant}"

        if row is not None:
            original_label = str(row.get("Label", ""))
            if original_label:
                row["Label"] = f"{original_label}__{target_suffix}"

    return row, status
