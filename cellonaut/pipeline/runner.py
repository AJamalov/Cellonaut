"""Full-run and first-sample preview pipeline entry points.

This module orchestrates sample discovery, shared per-sample resources, per-target
measurement execution, summaries, manifests, and preview-specific outputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from cellonaut.exceptions import SetupError, SetupErrorCode
from cellonaut.runtime import PipelineRuntime, PipelineStage, StageUpdate
from cellonaut.exceptions import PipelineCancelled
from cellonaut.io.imagej_runtime import get_ij
from cellonaut.measurement.exports import write_measurement_summary_csvs
from cellonaut.pipeline.cancellation import check_cancel
from cellonaut.pipeline.context import (
    build_sample_processing_context,
    SampleProcessingContext,
)
from cellonaut.pipeline.discovery import build_sample_label, get_measurement_sample_paths, validate_unique_result_ids
from cellonaut.pipeline.models import Config, FullRunStats, TargetConfig, SampleTargetStatus
from cellonaut.pipeline.output_manifest import save_run_manifest
from cellonaut.results.artifacts import initialize_artifact_manifest
from cellonaut.pipeline.planning import get_enabled_measurement_targets, get_image_def, pipeline_needs_imagej
from cellonaut.pipeline.preview_artifacts import collect_preview_artifacts
from cellonaut.pipeline.run_logging import make_logger
from cellonaut.pipeline.sample_processing import process_measurement_target_for_sample
from cellonaut.pipeline.summary import (
    build_pipeline_summary_dict,
    selected_measurements_summary,
)
from cellonaut.pipeline.validation import validate_config
from cellonaut.results.layout import build_results_layout
from cellonaut.version import format_app_version


@dataclass(slots=True)
class _TargetExecutionOutcome:
    row: dict[str, Any] | None
    status: str
    reason: str = ""


@dataclass(slots=True)
class _TargetBatchResult:
    """In-memory target results retained until a sample context is closed."""

    rows: list[dict[str, Any]]
    processed_targets: list[str]
    statuses: list[SampleTargetStatus]

    @property
    def processed_any(self) -> bool:
        return bool(self.processed_targets)

    @property
    def failed_any(self) -> bool:
        return any(row.status not in {"PROCESSED", "SKIPPED"} for row in self.statuses)


def _target_run_name(target: TargetConfig) -> str:
    variant = str(getattr(target, "output_variant", "") or "").strip()
    return f"{target.source_image_key} / {variant}" if variant else target.source_image_key


def _execute_measurement_target(
    *,
    sample_folder: Path,
    cfg: Any,
    target: Any,
    context: SampleProcessingContext,
    logger: Callable[[str], None],
    error_message: str,
    should_cancel: Callable[[], bool] | None,
) -> _TargetExecutionOutcome:
    """Convert one target exception into a durable failure, except cancellation."""

    try:
        row, status = process_measurement_target_for_sample(
            sample_folder=sample_folder,
            cfg=cfg,
            target=target,
            log_func=logger,
            should_cancel=should_cancel,
            precomputed_context=context,
        )
    except Exception as exc:
        if isinstance(exc, PipelineCancelled):
            raise
        reason = f"{type(exc).__name__}: {exc}"
        logger(f"{error_message}: {reason}")
        return _TargetExecutionOutcome(None, "FAILED", reason)
    reason = "" if status == "PROCESSED" else f"Target returned {status}"
    return _TargetExecutionOutcome(row, status, reason)


def _target_status_row(
    *,
    run_type: str,
    sample_id: str,
    target_name: str,
    cell_mask: str,
    status: str,
    context: SampleProcessingContext | None,
    reason: str,
) -> SampleTargetStatus:
    """Capture the stable manifest fields while the sample context is available."""

    file_map = context.file_map if context is not None else {}
    return SampleTargetStatus(
        run_type=run_type,
        sample_id=sample_id,
        target=target_name,
        status=status,
        cell_mask=cell_mask,
        source_image_file=str(file_map.get(target_name) or ""),
        reason=reason,
    )


# Full and preview runs intentionally share target isolation and status
# recording. Entry points supply their own wording and progress callbacks while
# this helper owns the identical execution lifecycle.
def _process_targets_in_context(
    *,
    cfg: Config,
    sample_folder: Path,
    sample_label: str,
    enabled_targets: list[TargetConfig],
    context: SampleProcessingContext,
    logger: Callable[[str], None],
    run_type: str,
    target_message: Callable[[str], str],
    error_message: Callable[[str], str],
    skipped_message: Callable[[str], str] | None = None,
    unexpected_status_message: Callable[[str], str] | None = None,
    progress_func: Callable[[int], None] | None = None,
    counter_func: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> _TargetBatchResult:
    result = _TargetBatchResult(rows=[], processed_targets=[], statuses=[])
    total_targets = max(1, len(enabled_targets))

    for target_index, target in enumerate(enabled_targets, start=1):
        check_cancel(should_cancel)
        if counter_func is not None:
            counter_func(f"{target_index} / {total_targets}")
        if progress_func is not None:
            progress_func(int((target_index - 1) * 100 / total_targets))

        target_name = target.source_image_key
        cell_mask = str(getattr(target, "cell_segmentation_mask_source", "") or "")
        display_target = _target_run_name(target)
        logger(target_message(display_target))
        outcome = _execute_measurement_target(
            sample_folder=sample_folder,
            cfg=cfg,
            target=target,
            context=context,
            logger=logger,
            error_message=error_message(display_target),
            should_cancel=should_cancel,
        )
        result.statuses.append(
            _target_status_row(
                run_type=run_type,
                sample_id=sample_label,
                target_name=target_name,
                cell_mask=cell_mask,
                status=outcome.status,
                context=context,
                reason=outcome.reason,
            )
        )

        if outcome.status == "PROCESSED":
            result.processed_targets.append(display_target)
            if outcome.row is not None:
                result.rows.append(outcome.row)
        elif outcome.status == "SKIPPED" and skipped_message is not None:
            logger(skipped_message(target_name))
        elif outcome.status != "FAILED" and unexpected_status_message is not None:
            logger(unexpected_status_message(target_name))

        if progress_func is not None:
            progress_func(int(target_index * 100 / total_targets))

    return result


@dataclass(slots=True)
class _PreparedFullRun:
    """Validated resources and discovery results needed by a full run."""

    logger: Callable[[str], None]
    start_time: float
    pipeline_summary: dict[str, Any]
    samples: list[Path]
    enabled_targets: list[TargetConfig]
    runtime: PipelineRuntime


@dataclass(slots=True)
class _FullRunResults:
    """Mutable aggregation produced by the sample-execution phase."""

    measurement_rows: list[dict[str, Any]]
    sample_statuses: list[SampleTargetStatus]
    processed: int
    skipped: int
    failed: int


def _log_full_run_configuration(
    cfg: Config,
    enabled_targets: list[TargetConfig],
    logger: Callable[[str], None],
) -> None:
    """Write the reproducibility settings that apply to the entire run."""

    logger("============================================================")
    logger(f"{format_app_version()} run started")
    logger(f"Input directory: {cfg.input_dir}")
    logger(f"Output directory: {cfg.output_dir}")
    logger(f"Fiji path: {cfg.fiji_app_path}")
    logger(f"Threshold method: {cfg.threshold_method}")
    logger(f"Weka class index: {cfg.probability_class_index}")
    logger(f"Reuse existing masks: {bool(getattr(cfg, 'reuse_existing_masks', False))}")
    if getattr(cfg, "reuse_existing_masks", False):
        logger(f"Mask reuse folder: {getattr(cfg, 'mask_source_dir', None) or cfg.output_dir}")
    logger("Measurements:")
    for target in enabled_targets:
        logger(
            f"  - measured_channel={target.source_image_key}, overlay_base={target.overlay_base_image_key or '(measured channel)'}, "
            f"masks={target.overlay_roi_keys or []}, cell_masks={target.do_cell_segmentation}, "
            f"cell_mask={getattr(target, 'cell_segmentation_mask_source', '') or '(measured channel)'}, "
            f"cell_mask_source={target.cell_segmentation_source or '(measured channel)'}, "
            f"output_variant={getattr(target, 'output_variant', '') or '(none)'}, "
            f"show_cell_mask={target.overlay_whole_cell_mask}"
        )
    logger("Run defaults (per-target execution settings are recorded in the pipeline summary):")
    logger(f"Cell diameter: {cfg.cell_diameter if cfg.cell_diameter is not None else 'Original scale'}")
    logger(f"Minimum cell area: {cfg.cell_min_size}")
    logger(f"Cell GPU requested: {cfg.cell_use_gpu}")
    logger(f"Cellpose probability threshold: {cfg.cellprob_threshold}")
    logger(f"Cellpose flow threshold: {cfg.flow_threshold}")
    logger(f"Cell remove border: {cfg.cell_remove_border}")
    logger(f"Cellpose model type: {cfg.cellpose_model_type}")
    logger(f"Cellpose custom model path: {cfg.cellpose_custom_model_path or '(none)'}")
    logger(f"Per-cell selected mask: {cfg.per_cell_mask_source}")
    logger("Configured channels and masks:")
    for image in cfg.images:
        logger(
            f"  - {image.key}: label={image.label}, folder={image.folder_name}, "
            f"model={image.model_path if image.model_path is not None else '(none)'}, "
            f"bg_radii_csv={image.bg_radii_csv}, mask_steps={image.mask_processing_steps}"
        )
    logger("============================================================")


def _prepare_full_run(
    cfg: Config,
    log_func: Callable[[str], None],
    should_cancel: Callable[[], bool] | None,
    runtime: PipelineRuntime,
) -> _PreparedFullRun:
    """Validate, initialize shared runtimes, and discover the sample plan."""

    start_time = time.time()
    validate_config(cfg)
    check_cancel(should_cancel)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    layout = build_results_layout(cfg.output_dir, create_root=False)
    logfile = layout["logs"] / "run_log.txt"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    _reset_log_file(logfile)
    initialize_artifact_manifest(layout["root"])

    logger = make_logger(log_func, logfile)
    enabled_targets = get_enabled_measurement_targets(cfg)
    if not enabled_targets:
        raise ValueError("No enabled measurement targets were configured.")
    _log_full_run_configuration(cfg, enabled_targets, logger)

    if pipeline_needs_imagej(cfg, enabled_targets):
        logger("Initializing Fiji...")
        logger(f"Fiji app path: {cfg.fiji_app_path}")
        runtime.stage(PipelineStage.INITIALIZING, "Initializing Fiji runtime")
        get_ij(cfg.fiji_app_path, log_func=logger)
        runtime.stage(PipelineStage.RUNNING, "Running")
    else:
        logger("Skipping Fiji initialization: this run can use native TIFF/NumPy processing.")
    check_cancel(should_cancel)

    pipeline_summary = build_pipeline_summary_dict(cfg)
    samples = get_measurement_sample_paths(cfg)
    total_samples = len(samples)
    if total_samples == 0:
        raise SetupError(SetupErrorCode.NO_SAMPLES, "No samples were found for the selected input structure.")
    validate_unique_result_ids(samples, cfg.input_structure, input_dir=cfg.input_dir)

    return _PreparedFullRun(
        logger=logger,
        start_time=start_time,
        runtime=runtime,
        pipeline_summary=pipeline_summary,
        samples=samples,
        enabled_targets=enabled_targets,
    )


def _process_full_sample(
    *,
    cfg: Config,
    sample_folder: Path,
    sample_name: str,
    enabled_targets: list[TargetConfig],
    logger: Callable[[str], None],
    should_cancel: Callable[[], bool] | None,
    runtime: PipelineRuntime,
) -> _TargetBatchResult:
    """Build and reliably close one sample context around all its targets."""

    context = build_sample_processing_context(
        sample_folder=sample_folder,
        cfg=cfg,
        log_func=logger,
        runtime=runtime,
        should_cancel=should_cancel,
    )
    if context is None:
        raise ValueError(f"Could not build processing context for sample {sample_name}")

    with context:
        return _process_targets_in_context(
            cfg=cfg,
            sample_folder=sample_folder,
            sample_label=sample_name,
            enabled_targets=enabled_targets,
            context=context,
            logger=logger,
            run_type="full",
            target_message=lambda name: f"[TARGET] Running source target {name} for sample {sample_name}",
            error_message=lambda name: f"[TARGET ERROR] {name} for sample {sample_name}",
            skipped_message=lambda name: f"[TARGET] Skipped source target {name} for sample {sample_name}",
            unexpected_status_message=lambda name: f"[TARGET] Failed source target {name} for sample {sample_name}",
            should_cancel=should_cancel,
        )


def _execute_full_run_samples(
    *,
    cfg: Config,
    prepared: _PreparedFullRun,
    progress_func: Callable[[int], None] | None,
    sample_func: Callable[[str], None] | None,
    counter_func: Callable[[str], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> _FullRunResults:
    """Execute the discovered samples while isolating failures per sample."""

    results = _FullRunResults(measurement_rows=[], sample_statuses=[], processed=0, skipped=0, failed=0)
    total_samples = len(prepared.samples)

    for sample_index, sample_folder in enumerate(prepared.samples, start=1):
        check_cancel(should_cancel)
        sample_name = build_sample_label(sample_folder, cfg.input_structure)

        if sample_func is not None:
            sample_func(sample_name)
        if counter_func is not None:
            counter_func(f"{sample_index} / {total_samples}")
        if progress_func is not None:
            progress_func(int((sample_index - 1) * 100 / total_samples))

        prepared.logger("------------------------------------------------------------")
        prepared.logger(f"[SAMPLE {sample_index}/{total_samples}] {sample_name}")

        try:
            sample_result = _process_full_sample(
                cfg=cfg,
                sample_folder=sample_folder,
                sample_name=sample_name,
                enabled_targets=prepared.enabled_targets,
                logger=prepared.logger,
                runtime=prepared.runtime,
                should_cancel=should_cancel,
            )
            results.measurement_rows.extend(sample_result.rows)
            results.sample_statuses.extend(sample_result.statuses)

            # A partly successful sample remains processed; target-level
            # failures are still retained separately in the manifest.
            if sample_result.processed_any:
                results.processed += 1
            elif sample_result.failed_any:
                results.failed += 1
            else:
                results.skipped += 1
        except PipelineCancelled:
            raise
        except Exception as exc:
            results.failed += 1
            prepared.logger(f"[SAMPLE ERROR] {sample_name}: {exc}")
            reason = f"{type(exc).__name__}: {exc}"
            reported_targets = {
                (row.target, row.cell_mask) for row in results.sample_statuses if row.sample_id == sample_name
            }
            # A context-creation failure has no target rows of its own. Fill
            # only missing rows so already-recorded target outcomes survive.
            for target in prepared.enabled_targets:
                cell_mask = str(getattr(target, "cell_segmentation_mask_source", "") or "")
                if (target.source_image_key, cell_mask) in reported_targets:
                    continue
                results.sample_statuses.append(
                    _target_status_row(
                        run_type="full",
                        sample_id=sample_name,
                        target_name=target.source_image_key,
                        cell_mask=cell_mask,
                        status="FAILED",
                        context=None,
                        reason=reason,
                    )
                )

        if progress_func is not None:
            progress_func(int(sample_index * 100 / total_samples))

    return results


def _finalize_full_run(
    *,
    cfg: Config,
    prepared: _PreparedFullRun,
    results: _FullRunResults,
) -> dict[str, Any]:
    """Write exports and manifests, then serialize the worker-facing result."""

    logger = prepared.logger
    total_samples = len(prepared.samples)
    prepared.runtime.stage(PipelineStage.EXPORT, "Saving results")
    logger("[RUN] Saving results")
    # Build once after processing because repeatedly concatenating one row makes
    # large batches progressively slower.
    measurement_table = pd.DataFrame(results.measurement_rows)

    if not measurement_table.empty:
        write_measurement_summary_csvs(measurement_table, cfg=cfg, log_func=logger, include_averages=True)

    elapsed = time.time() - prepared.start_time
    total_targets = total_samples * len(prepared.enabled_targets)
    processed_targets = sum(1 for row in results.sample_statuses if row.status == "PROCESSED")
    skipped_targets = sum(1 for row in results.sample_statuses if row.status == "SKIPPED")
    # Missing target statuses are failures too, so this remains accurate even
    # if an entire sample fails before its context can be constructed.
    failed_targets = max(0, total_targets - processed_targets - skipped_targets)
    run_stats: FullRunStats = {
        "total_samples": total_samples,
        "processed": results.processed,
        "skipped": results.skipped,
        "failed": results.failed,
        "total_targets": total_targets,
        "processed_targets": processed_targets,
        "skipped_targets": skipped_targets,
        "failed_targets": failed_targets,
        "elapsed_seconds": round(elapsed, 2),
        "warning_count": skipped_targets + failed_targets,
    }
    serialized_statuses = [row.to_dict() for row in results.sample_statuses]
    manifest_paths = save_run_manifest(
        cfg.output_dir,
        run_type="full",
        inference_devices=prepared.runtime.cellpose_inference_devices,
        pipeline_summary=prepared.pipeline_summary,
        run_stats=dict(run_stats),
        sample_status_rows=serialized_statuses,
    )
    logger(f"Saved run summary: {manifest_paths['text']}")
    logger(f"Saved run summary JSON: {manifest_paths['json']}")
    logger("============================================================")
    logger(f"Run complete in {elapsed:.2f} s")
    logger(f"Processed: {results.processed}")
    logger(f"Skipped: {results.skipped}")
    logger(f"Failed: {results.failed}")
    logger("============================================================")
    return {
        "status": ("completed_with_errors" if results.failed > 0 or failed_targets > 0 else "completed"),
        "output_dir": str(build_results_layout(cfg.output_dir)["root"]),
        "run_stats": run_stats,
        "sample_status": serialized_statuses,
        "pipeline_summary": prepared.pipeline_summary,
        "manifest_paths": manifest_paths,
    }


# One sample context remains active across targets; only durable rows are
# aggregated after the context closes.
def run_pipeline(
    cfg: Any,
    log_func: Callable[[str], None] = print,
    progress_func: Optional[Callable[[int], None]] = None,
    sample_func: Optional[Callable[[str], None]] = None,
    counter_func: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    stage_func: Callable[[StageUpdate], None] | None = None,
) -> dict[str, Any]:
    """Execute all discovered samples into cfg.output_dir and return run metadata.

    Validation resolves targets in cfg. Callers reserve output directories;
    this function does not choose a numbered run folder. Target failures are
    recorded while other work continues; setup errors and PipelineCancelled
    propagate. The result contains status, counts, per-target outcomes and
    manifest paths. Stage/sample callbacks carry data independently of log text.
    """

    prepared = _prepare_full_run(cfg, log_func, should_cancel, PipelineRuntime(stage_func))
    results = _execute_full_run_samples(
        cfg=cfg,
        prepared=prepared,
        progress_func=progress_func,
        sample_func=sample_func,
        counter_func=counter_func,
        should_cancel=should_cancel,
    )
    check_cancel(should_cancel)
    return _finalize_full_run(cfg=cfg, prepared=prepared, results=results)


# A fresh preview or run should never append beneath messages from an earlier execution.
def _reset_log_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)


# Preview starts Fiji only when the selected masks or processing steps require Java objects.
def _initialize_preview_runtime(
    cfg: Config,
    enabled_targets: list[TargetConfig],
    logger: Callable[[str], None],
    label: str,
    runtime: PipelineRuntime,
) -> None:
    if pipeline_needs_imagej(cfg, enabled_targets):
        runtime.stage(PipelineStage.INITIALIZING, "Initializing Fiji runtime")
        get_ij(cfg.fiji_app_path, log_func=logger)
        runtime.stage(PipelineStage.RUNNING, "Running preview")
    else:
        logger(f"[{label}] Skipping Fiji initialization: native TIFF/NumPy processing is enough.")


# Reuse one prepared sample context while previewing every measured channel in the matrix.
def _process_preview_targets(
    *,
    cfg: Any,
    sample_folder: Path,
    sample_label: str,
    enabled_targets: list[Any],
    logger: Callable[[str], None],
    log_label: str,
    run_type: str,
    progress_func: Optional[Callable[[int], None]],
    counter_func: Optional[Callable[[str], None]],
    should_cancel: Optional[Callable[[], bool]],
    context_error: str,
    runtime: PipelineRuntime,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    context = build_sample_processing_context(
        sample_folder=sample_folder,
        cfg=cfg,
        log_func=logger,
        runtime=runtime,
        should_cancel=should_cancel,
    )
    if context is None:
        raise ValueError(context_error)

    with context:
        total_targets = max(1, len(enabled_targets))
        if progress_func is not None:
            progress_func(0)
        if counter_func is not None:
            counter_func(f"0 / {total_targets}")

        result = _process_targets_in_context(
            cfg=cfg,
            sample_folder=sample_folder,
            sample_label=sample_label,
            enabled_targets=enabled_targets,
            context=context,
            logger=logger,
            run_type=run_type,
            target_message=lambda name: f"[{log_label}] Trying target: {name}",
            error_message=lambda name: f"[{log_label} TARGET ERROR] {name}",
            progress_func=progress_func,
            counter_func=counter_func,
            should_cancel=should_cancel,
        )

    # Convert typed statuses only at the preview API boundary; manifests and
    # worker consumers intentionally retain their established dictionary shape.
    return (
        result.rows,
        result.processed_targets,
        [status.to_dict() for status in result.statuses],
    )


def run_preview_pipeline(
    cfg: Config,
    log_func: Callable[[str], None] = print,
    progress_func: Optional[Callable[[int], None]] = None,
    counter_func: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    stage_func: Callable[[StageUpdate], None] | None = None,
    sample_func: Callable[[str], None] | None = None,
):
    """Run every enabled target on the first discovered sample using normal outputs.

    Like run_pipeline, resolve cfg and write into the caller's output directory.
    Return original measurement rows, artifact paths and run metadata. A partial
    preview has completed_with_errors=True; no successful target raises an error.
    Cancellation propagates. This is execution, not display-only rendering or
    Cell Group filtering, and does not search later samples for a usable preview.
    """
    runtime = PipelineRuntime(stage_func)
    validate_config(cfg)
    check_cancel(should_cancel)

    layout = build_results_layout(cfg.output_dir, create_root=False)
    preview_logfile = layout["logs"] / "preview_log.txt"
    _reset_log_file(preview_logfile)
    initialize_artifact_manifest(layout["root"])

    logger = make_logger(log_func, preview_logfile)

    logger("===== PREVIEW MODE =====")
    logger(f"Measurement options: {selected_measurements_summary(cfg)}")

    pipeline_summary = build_pipeline_summary_dict(cfg)

    samples = get_measurement_sample_paths(cfg)
    if not samples:
        raise SetupError(SetupErrorCode.NO_SAMPLES, "No samples found")

    enabled_targets = get_enabled_measurement_targets(cfg)
    if not enabled_targets:
        raise ValueError("No enabled measurement targets were configured.")

    _initialize_preview_runtime(cfg, enabled_targets, logger, "PREVIEW", runtime)

    sample_folder = samples[0]
    sample_label = build_sample_label(sample_folder, cfg.input_structure)
    if sample_func is not None:
        sample_func(sample_label)
    logger(f"[PREVIEW] Trying sample: {sample_label}")

    preview_rows, preview_targets, preview_status_rows = _process_preview_targets(
        cfg=cfg,
        sample_folder=sample_folder,
        sample_label=sample_label,
        enabled_targets=enabled_targets,
        logger=logger,
        log_label="PREVIEW",
        run_type="preview",
        progress_func=progress_func,
        counter_func=counter_func,
        should_cancel=should_cancel,
        context_error="No processable samples were found for preview.",
        runtime=runtime,
    )

    if not preview_targets:
        first_reason = next(
            (str(row.get("reason") or "") for row in preview_status_rows if row.get("reason")),
            "No enabled target produced a preview.",
        )
        raise ValueError(f"Preview failed for every measured channel. First error: {first_reason}")

    completed_with_errors = any(row.get("status") != "PROCESSED" for row in preview_status_rows)

    results_dir = build_results_layout(cfg.output_dir)["root"]
    last_artifacts = collect_preview_artifacts(results_dir)
    preview_csv_paths = write_measurement_summary_csvs(
        preview_rows,
        cfg=cfg,
        log_func=logger,
        include_averages=False,
    )

    logger(f"[PREVIEW] DONE: {sample_label} / targets={preview_targets}")
    logger(f"[PREVIEW] Log saved to: {preview_logfile}")
    if last_artifacts and last_artifacts.get("preview_path"):
        logger(f"[PREVIEW] Preview file: {last_artifacts['preview_path']}")
    else:
        logger("[PREVIEW] No previewable output file was found under Results/Overlays, Results/Masks, or Results/Cells")

    run_stats = {
        "total_samples": 1,
        "processed_targets": len(preview_targets),
        "skipped_targets": sum(1 for row in preview_status_rows if row.get("status") == "SKIPPED"),
        "skipped_or_failed_targets": sum(1 for row in preview_status_rows if row.get("status") != "PROCESSED"),
        "failed_targets": sum(1 for row in preview_status_rows if row.get("status") == "FAILED"),
    }
    run_manifest_paths = save_run_manifest(
        cfg.output_dir,
        run_type="preview",
        inference_devices=runtime.cellpose_inference_devices,
        pipeline_summary=pipeline_summary,
        run_stats=run_stats,
        sample_status_rows=preview_status_rows,
    )
    logger(f"[PREVIEW] Saved run summary: {run_manifest_paths['text']}")
    logger(f"[PREVIEW] Saved run summary JSON: {run_manifest_paths['json']}")

    preview_overlay_labels: list[str] = []
    if preview_targets:
        first_processed_key = preview_targets[0]
        first_target = next(target for target in enabled_targets if _target_run_name(target) == first_processed_key)
        try:
            first_base_def = get_image_def(cfg, first_target.overlay_base_image_key or first_target.source_image_key)
            preview_overlay_labels.append(first_base_def.label)
        except (KeyError, ValueError):
            preview_overlay_labels.append("Base")

        for key in list(first_target.overlay_roi_keys or []):
            try:
                preview_overlay_labels.append(get_image_def(cfg, key).label)
            except (KeyError, ValueError):
                preview_overlay_labels.append(key)

        if first_target.do_cell_segmentation and first_target.overlay_whole_cell_mask:
            preview_overlay_labels.append("Cellpose whole-cell mask")

    return {
        "preview_log": str(preview_logfile),
        "run_stats": run_stats,
        "output_dir": str(build_results_layout(cfg.output_dir)["root"]),
        "sample_label": sample_label,
        "target_names": preview_targets,
        "status": "PROCESSED",
        "completed_with_errors": completed_with_errors,
        "pipeline_summary": pipeline_summary,
        "pipeline_summary_paths": {
            "text_path": run_manifest_paths["text"],
            "json_path": run_manifest_paths["json"],
        },
        "measurement_csv_paths": preview_csv_paths,
        "artifacts": last_artifacts or {},
        "preview_path": (last_artifacts or {}).get("preview_path"),
        "preview_overlay_labels": preview_overlay_labels,
        "rows": preview_rows,
    }
