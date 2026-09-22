"""Construction and cleanup of reusable per-sample processing state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cellonaut.runtime import PipelineRuntime, PipelineStage
from cellonaut.exceptions import PipelineCancelled
from cellonaut.io.image_io import (
    find_channel_files,
    open_channel_arrays,
    open_channel_images,
    resolve_channel_indices_for_file_map,
)
from cellonaut.io.imagej_runtime import get_java_classes, load_weka_classifier
from cellonaut.masks.roi_processing import prepare_rois_for_defs
from cellonaut.pipeline.cancellation import check_cancel
from cellonaut.pipeline.processing_montage import save_processing_montages_for_sample
from cellonaut.pipeline.discovery import (
    build_common_results_export_dirs,
    build_result_id,
    build_sample_label,
    nest_export_dirs_for_sample,
)
from cellonaut.pipeline.models import Config
from cellonaut.pipeline.planning import (
    get_enabled_measurement_targets,
    get_image_def,
    get_required_roi_defs_for_targets,
    image_keys_needed_for_processing,
    pipeline_needs_imagej,
)


# Close each image only once, even when several entries reference it.
def _close_unique_images(*image_maps: dict[str, Any]) -> None:
    closed_ids: set[int] = set()
    for image_map in image_maps:
        for image in image_map.values():
            if image is None or id(image) in closed_ids:
                continue
            closed_ids.add(id(image))
            try:
                close = getattr(image, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass


@dataclass(slots=True)
class SampleProcessingContext:
    """Own opened and prepared images shared by all targets of one sample.

    The builder transfers ownership on success and closes opened images on
    failure. Run entry points close this context after all targets; a target
    borrowing it closes only its own temporary measurement images.
    """

    id_label: str
    result_id: str
    export_dirs: dict[str, Path]
    file_map: dict[str, Path | None]
    image_map: dict[str, Any]
    roi_map: dict[str, Any]
    roi_measure_img_map: dict[str, Any]
    skeleton_metrics: dict[str, int]
    native_numpy_images: bool
    cellpose_label_cache: dict[str, Any] = field(default_factory=dict)
    runtime: PipelineRuntime = field(default_factory=PipelineRuntime)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_unique_images(self.roi_measure_img_map, self.image_map)

    def __enter__(self) -> "SampleProcessingContext":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def build_sample_processing_context(
    sample_folder: Path,
    cfg: Config,
    log_func: Callable[[str], None],
    should_cancel: Callable[[], bool] | None = None,
    runtime: PipelineRuntime | None = None,
) -> SampleProcessingContext | None:
    """Open required channels and prepare masks once for all targets in a sample.

    The caller owns the returned context and must close it after the last target.
    None means channel discovery/opening failed; preparation errors propagate
    after opened images are closed. A None ROI inside a valid context denotes
    an empty mask, not a request to measure the whole image. Preparation may
    write mask and montage artifacts before any target measurement runs.
    """
    runtime = runtime if runtime is not None else PipelineRuntime()
    runtime.stage(PipelineStage.LOADING, "Loading images")
    id_label = build_sample_label(sample_folder, cfg.input_structure)
    result_id = build_result_id(
        sample_folder,
        cfg.input_structure,
        input_dir=cfg.input_dir,
    )

    enabled_targets = get_enabled_measurement_targets(cfg)
    required_roi_defs = get_required_roi_defs_for_targets(cfg, enabled_targets)
    image_keys_to_open = image_keys_needed_for_processing(
        cfg,
        enabled_targets,
        required_roi_defs,
        result_id,
    )

    file_map = find_channel_files(
        sample_folder,
        cfg,
        id_label,
        log_func,
        required_image_keys=image_keys_to_open,
    )
    if file_map is None:
        log_func(f"[{id_label}] Skipped because channel files could not be resolved")
        return None

    export_dirs = nest_export_dirs_for_sample(
        build_common_results_export_dirs(cfg.output_dir),
        cfg.input_dir,
        sample_folder,
        cfg.input_structure,
    )

    image_defs_by_key = {image.key: image for image in cfg.images}
    channel_z_modes = {
        image.key: getattr(image, "stack_z_mode", "max_projection") or "max_projection" for image in cfg.images
    }
    channel_z_indices = {image.key: getattr(image, "stack_z_index", None) for image in cfg.images}
    for key, path in file_map.items():
        image_def = image_defs_by_key.get(key)
        layer = image_def.stack_channel_index if image_def is not None else None
        z_mode = channel_z_modes.get(key, "max_projection")
        z_index = channel_z_indices.get(key)
        parts = []
        if layer is not None:
            parts.append(f"stack layer: {layer}")
        if z_mode == "single_z":
            parts.append(f"Z slice: {z_index or 1}")
        elif z_mode == "max_projection":
            parts.append("Z: max projection")
        suffix = f" | {'; '.join(parts)}" if parts else ""
        source_key = str(getattr(image_def, "stack_source_image_key", "") or "") if image_def else ""
        if source_key and source_key in image_keys_to_open and file_map.get(source_key) == path:
            log_func(f"[{id_label}] {key} uses {source_key} source: {path}{suffix}")
        else:
            log_func(f"[{id_label}] {key} file: {path}{suffix}")

    use_native_arrays = not pipeline_needs_imagej(cfg, enabled_targets)
    if image_keys_to_open:
        log_func(
            f"[{id_label}] Opening only required image channels: {sorted(image_keys_to_open)} "
            "(shared TIFF planes are loaded once)"
        )
    filtered_file_map = {key: path for key, path in file_map.items() if key in image_keys_to_open}
    channel_indices = resolve_channel_indices_for_file_map(
        image_defs=cfg.images,
        # Automatic stack layers are defined by the complete configured channel
        # order, not by the subset needed for this measurement target.
        file_map=file_map,
        input_structure=cfg.input_structure,
        log_func=log_func,
    )

    open_channels = open_channel_arrays if use_native_arrays else open_channel_images
    if use_native_arrays:
        log_func(f"[{id_label}] No Fiji/Weka steps required; " "opening TIFFs without ImageJ.")
    image_map = open_channels(
        filtered_file_map,
        id_label,
        log_func,
        channel_indices=channel_indices,
        channel_z_modes=channel_z_modes,
        channel_z_indices=channel_z_indices,
        should_cancel=should_cancel,
        cancel_exception=PipelineCancelled,
        runtime=runtime,
    )
    if image_map is None:
        log_func(f"[{id_label}] Failed to open one or more images")
        return None

    roi_measure_img_map: dict[str, Any] = {}
    skeleton_metrics: dict[str, int] = {}
    try:
        log_func(f"[{id_label}] Channels opened successfully")
        check_cancel(should_cancel)

        segmenters: dict[str, Any] = {}
        if required_roi_defs:
            if getattr(cfg, "reuse_existing_masks", False):
                mask_source = Path(getattr(cfg, "mask_source_dir", None) or cfg.output_dir)
                log_func(f"[{id_label}] Reuse existing masks is ON; " f"reading masks from {mask_source / 'Results'}.")
                log_func(
                    f"[{id_label}] Existing Weka threshold masks keep their original "
                    "classifier and threshold result; current shifts and minimum-area "
                    "cleanup are applied during this run."
                )
            else:
                weka_segmentation = get_java_classes()["WekaSegmentation"]
                for image_def in required_roi_defs:
                    if image_def.combined_mask_source_keys:
                        log_func(f"[{id_label}] Combined mask configured for " f"{image_def.label}.")
                        continue
                    if image_def.model_path is None:
                        raise RuntimeError(f"Classifier path is missing for {image_def.label}.")
                    segmenter = weka_segmentation()
                    load_weka_classifier(segmenter, image_def.model_path)
                    segmenters[image_def.key] = segmenter
                    log_func(f"[{id_label}] Loaded classifier for " f"{image_def.label}: {image_def.model_path}")

        check_cancel(should_cancel)
        runtime.stage(PipelineStage.MASKS, "Preparing masks")
        log_func("[RUN] Preparing masks")
        roi_map, roi_measure_img_map = prepare_rois_for_defs(
            image_map=image_map,
            cfg=cfg,
            out_path=export_dirs["masks"],
            result_id=result_id,
            roi_defs=required_roi_defs,
            segs=segmenters,
            log_func=log_func,
            file_map=file_map,
            skeleton_metrics=skeleton_metrics,
            skeleton_out_path=export_dirs["mask_skeletons"],
            should_cancel=should_cancel,
        )
        try:
            save_processing_montages_for_sample(
                cfg=cfg,
                image_map=image_map,
                roi_map=roi_map,
                export_dir=export_dirs["processing_montages"],
                result_id=result_id,
                log_func=log_func,
                probability_dir=export_dirs["weka_probability_maps"],
                threshold_dir=export_dirs["weka_threshold_masks"],
            )
        except Exception as exc:
            log_func(f"[WARN] [{id_label}] Could not save processing montage PNGs: {exc}")

        for key, roi in roi_map.items():
            image_def = get_image_def(cfg, key)
            if roi is None:
                log_func(f"[{id_label}] Mask for {image_def.label} is empty (zero foreground pixels)")
            else:
                log_func(f"[{id_label}] Mask available for {image_def.label}")

        return SampleProcessingContext(
            id_label=id_label,
            result_id=result_id,
            export_dirs=export_dirs,
            file_map=file_map,
            image_map=image_map,
            roi_map=roi_map,
            roi_measure_img_map=roi_measure_img_map,
            skeleton_metrics=skeleton_metrics,
            native_numpy_images=use_native_arrays,
            runtime=runtime,
        )
    except Exception:
        _close_unique_images(roi_measure_img_map, image_map)
        raise
