"""Setup validation and user-facing configuration checks.

This mixin gathers filesystem checks, channel readiness checks, sample
discovery summaries, and analysis-relationship validation before a preview or
full run starts.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from cellonaut.runtime import PipelineStage, stage_update
from cellonaut.config.defaults import (
    CELLPOSE_MEASUREMENT_KEYS,
    CONFIGURED_MASK_WITHIN_CELLPOSE_KEYS,
    DEFAULT_CELL_DIAMETER,
    DEFAULT_CELL_MIN_SIZE,
    DEFAULT_CELLPROB_THRESHOLD,
    DEFAULT_FLOW_THRESHOLD,
    DEFAULT_INPUT_STRUCTURE,
    QC_FILTER_MODE_LABELS,
    MASK_INTENSITY_SOURCE_LABELS,
    normalize_qc_filter_mode,
)
from cellonaut.config.relationships import (
    cellpose_mask_source_names,
    image_produces_mask,
    image_uses_non_classifier_mask,
)
from cellonaut.gui.dialogs import ConfigurationSummaryDialog, count_label
from cellonaut.gui.field_validation import CellonautGuiFieldValidationMixin
from cellonaut.gui.setup_checks import (
    build_setup_check_items as create_setup_check_items,
    compact_repeated_warnings as compact_setup_warnings,
    format_configuration_summary_text as format_setup_summary_text,
)
from cellonaut.io.writers import write_text
from cellonaut.pipeline.readiness import analyze_sample_readiness
from cellonaut.pipeline.validation import validate_config
from cellonaut.system.disk_space import (
    GIB,
    estimate_disk_space,
    format_bytes,
    readiness_source_files,
    unique_file_bytes,
)


def _unavailable_dry_run(cfg, exc: Exception) -> dict[str, Any]:
    return {
        "available": False,
        "root": str(cfg.input_dir),
        "structure": str(getattr(cfg, "input_structure", DEFAULT_INPUT_STRUCTURE) or DEFAULT_INPUT_STRUCTURE),
        "expected_folders": [],
        "total": 0,
        "ready": 0,
        "incomplete": 0,
        "blocked": 0,
        "warning_count": 0,
        "samples": [],
        "problems": [],
        "error": str(exc),
    }


def complete_configuration_summary(
    summary: dict[str, Any],
    cfg,
    active_defs: list[dict],
    relationships: list[dict],
    warnings: list[str],
    *,
    cached_fiji_scan: dict[str, Any],
    cached_input_scan: dict[str, Any],
    nd2_detected_channel_names: list[str],
) -> dict[str, Any]:
    """Run filesystem setup checks using settings collected by the GUI."""
    try:
        dry_run = analyze_sample_readiness(cfg)
    except Exception as exc:
        dry_run = _unavailable_dry_run(cfg, exc)

    completed = dict(summary)
    completed["dry_run"] = dry_run
    completed["setup_checks"] = create_setup_check_items(
        cfg,
        active_defs,
        relationships,
        warnings,
        dry_run,
        cached_fiji_scan=cached_fiji_scan,
        cached_input_scan=cached_input_scan,
        nd2_detected_channel_names=nd2_detected_channel_names,
    )
    return completed


class SetupCheckWorker(QObject):
    """Build the filesystem-backed portion of Check Setup away from the GUI thread."""

    done_signal = Signal(dict)

    def __init__(
        self,
        summary: dict[str, Any],
        cfg,
        active_defs: list[dict],
        relationships: list[dict],
        warnings: list[str],
        *,
        cached_fiji_scan: dict[str, Any],
        cached_input_scan: dict[str, Any],
        nd2_detected_channel_names: list[str],
    ):
        super().__init__()
        self.summary = dict(summary)
        self.cfg = cfg
        self.active_defs = [dict(item) for item in active_defs]
        self.relationships = [dict(item) for item in relationships]
        self.warnings = list(warnings)
        self.cached_fiji_scan = dict(cached_fiji_scan)
        self.cached_input_scan = dict(cached_input_scan)
        self.nd2_detected_channel_names = list(nd2_detected_channel_names)
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        if self._cancel_event.is_set():
            self.done_signal.emit({"cancelled": True, "error": ""})
            return
        try:
            summary = complete_configuration_summary(
                self.summary,
                self.cfg,
                self.active_defs,
                self.relationships,
                self.warnings,
                cached_fiji_scan=self.cached_fiji_scan,
                cached_input_scan=self.cached_input_scan,
                nd2_detected_channel_names=self.nd2_detected_channel_names,
            )
            if self._cancel_event.is_set():
                self.done_signal.emit({"cancelled": True, "error": ""})
                return
            self.done_signal.emit(
                {
                    "cancelled": False,
                    "error": "",
                    "summary": summary,
                    "text": format_setup_summary_text(summary),
                }
            )
        except Exception as exc:
            self.done_signal.emit({"cancelled": False, "error": f"{type(exc).__name__}: {exc}"})


# Compose widget-level field checks with runtime readiness checks so Check Setup
# and the actual Run command enforce the same configuration rules.
class CellonautGuiValidationMixin(CellonautGuiFieldValidationMixin):
    """Apply the same configuration checks to setup and execution."""

    def confirm_disk_space_for_files(
        self,
        destination: Path,
        source_files: list[Path],
        *,
        operation: str,
        output_multiplier: float,
        reserve_bytes: int,
    ) -> bool:
        """Block critically low volumes and confirm estimates below the safe recommendation."""
        try:
            estimate = estimate_disk_space(
                destination,
                source_bytes=unique_file_bytes(source_files),
                output_multiplier=output_multiplier,
                reserve_bytes=reserve_bytes,
            )
        except OSError as exc:
            reply = QMessageBox.question(
                self,
                "Disk-space check unavailable",
                f"Cellonaut could not check free space for:\n{destination}\n\n{exc}\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return reply == QMessageBox.StandardButton.Yes

        summary = (
            f"Destination: {estimate.destination}\n"
            f"Free space: {format_bytes(estimate.free_bytes)}\n"
            f"Recommended before starting: {format_bytes(estimate.recommended_bytes)}"
        )
        self.log(f"[DISK] {operation}: {summary.replace(chr(10), '; ')}")

        if estimate.critically_low:
            QMessageBox.critical(
                self,
                "Not enough disk space",
                f"{operation} cannot start because the destination has critically low free space.\n\n{summary}\n\n"
                "Free space on the destination drive and try again.",
            )
            return False
        if estimate.below_recommended:
            reply = QMessageBox.question(
                self,
                "Low disk space",
                f"There may not be enough free space to finish {operation.lower()}.\n\n{summary}\n\n"
                "The recommendation includes temporary files and a safety reserve. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return reply == QMessageBox.StandardButton.Yes
        return True

    def confirm_pipeline_disk_space(self, cfg, *, preview: bool, readiness: dict | None = None) -> bool:
        """Use resolved input TIFFs to size a preview or full-run preflight."""
        try:
            resolved = readiness if isinstance(readiness, dict) else analyze_sample_readiness(cfg)
        except Exception as exc:
            self.log(f"[DISK][WARN] Could not size resolved pipeline inputs: {exc}")
            resolved = {}
        return self.confirm_disk_space_for_files(
            Path(cfg.output_dir),
            readiness_source_files(resolved, first_ready_only=preview),
            operation="Preview" if preview else "Full pipeline run",
            output_multiplier=2.0 if preview else 4.0,
            reserve_bytes=1 * GIB if preview else 2 * GIB,
        )

    # Validate before reserving a numbered folder so configuration mistakes do
    # not leave behind failed run or preview directories.
    def validate_config_before_output(self, cfg) -> bool:
        try:
            validate_config(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid pipeline setup", str(exc))
            return False
        return True

    # Reuse checks are kept separate from a run because missing masks may be acceptable when regeneration is possible.
    def show_mask_source_report(self):
        try:
            cfg = self.collect_config()
            if not cfg.reuse_existing_masks:
                QMessageBox.information(self, "Mask reuse report", "Enable Reuse existing masks before checking saved masks.")
                return
            readiness = analyze_sample_readiness(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Mask reuse report", str(exc))
            return

        samples = list(readiness.get("samples", []) or [])
        checked = [sample for sample in samples if sample.get("ready") and "reuse_mask_warnings" in sample]
        unchecked = [sample for sample in samples if not sample.get("ready") or "reuse_mask_warnings" not in sample]
        warnings = [
            (str(sample.get("sample", "")), str(message))
            for sample in samples
            for message in list(sample.get("reuse_mask_warnings", []) or [])
        ]
        warnings.extend(
            (str(sample.get("sample", "")), "Mask check unavailable: " + "; ".join(
                sample.get("missing_required") or ["Required input files could not be resolved."]
            ))
            for sample in unchecked
        )
        samples_with_expected_masks = sum(1 for sample in checked if not sample.get("reuse_mask_warnings"))
        lines = [
            f"Reusable-mask source: {cfg.mask_source_dir or cfg.output_dir}",
            f"Samples found: {len(samples)}",
            f"Samples checked: {len(checked)}",
            f"Samples not checked: {len(unchecked)}",
            f"Samples with all expected masks: {samples_with_expected_masks}",
            f"Warnings: {len(warnings)}",
            "",
        ]
        if warnings:
            lines.extend(f"- {sample}: {message}" for sample, message in warnings[:50])
            if len(warnings) > 50:
                lines.append(f"- ... plus {len(warnings) - 50} more")
        elif checked:
            lines.append("All configured imported/reusable masks were found and passed available shape checks.")
        else:
            lines.append("No samples could be checked. Check the input folder and sample layout.")
        QMessageBox.information(
            self,
            "Mask reuse report",
            "\n".join(lines),
        )

    # These checks are advisory because Cellpose can provide the measurement
    # boundary without a separate Weka, imported, or combined mask.
    def get_analysis_matrix_warnings(self) -> list[str]:
        active_defs = self.get_active_image_definitions()
        by_name = {
            str(img.get("name", "") or "").strip(): img for img in active_defs if str(img.get("name", "") or "").strip()
        }

        warnings = []
        measurement_options = dict(self.measurement_options or {})
        wants_whole_cell = any(
            bool(measurement_options.get(key, False))
            for key in CELLPOSE_MEASUREMENT_KEYS
        )
        wants_mask_in_cell = any(
            bool(measurement_options.get(key, False))
            for key in CONFIGURED_MASK_WITHIN_CELLPOSE_KEYS
        )

        for source_def in active_defs:
            if hasattr(self, "is_physical_channel_definition") and not self.is_physical_channel_definition(source_def):
                continue
            source_name = str(source_def.get("name", "") or "").strip()
            relationships = dict(source_def.get("mask_relationships", {}))

            selected_targets = [target_name for target_name in by_name if bool(relationships.get(target_name, False))]

            for target_name in selected_targets:
                target_def = by_name.get(target_name)
                if not target_def:
                    warnings.append(
                        f"{source_name} uses unknown mask image {target_name}. "
                        "Rebuild the mask relationship in Measurements after renaming channels or loading presets."
                    )
                    continue

                classifier = str(target_def.get("classifier", "") or "").strip()
                if not image_produces_mask(target_def):
                    warnings.append(
                        f"{source_name} x {target_name}: mask source is incomplete. "
                        "Add a classifier under an image in Masks, or choose valid combined-mask sources."
                    )
                elif not image_uses_non_classifier_mask(target_def) and not Path(classifier).exists():
                    warnings.append(
                        f"{source_name} x {target_name}: classifier path does not exist: {classifier}. "
                        "Choose the .model file again, or enable Reuse existing masks if masks already exist."
                    )

            selected_cellpose = cellpose_mask_source_names(source_def)
            if not selected_cellpose and "analysis_cellpose_mask_source" not in source_def and bool(
                source_def.get("analysis_cell_segmentation_enabled", False)
            ):
                selected_cellpose = [source_name]
            if (wants_whole_cell or wants_mask_in_cell) and not selected_cellpose:
                if wants_whole_cell and wants_mask_in_cell:
                    selected_description = (
                        "Cellpose whole-cell measurements and measurements of configured masks within "
                        "Cellpose cells are selected"
                    )
                elif wants_whole_cell:
                    selected_description = "Cellpose whole-cell measurements are selected"
                else:
                    selected_description = (
                        "Measurements of configured masks within Cellpose cells are selected"
                    )
                warnings.append(
                    f"{source_name}: {selected_description} but Cellpose masks are disabled. "
                    "Turn ON a Cellpose-mask column for this channel or turn off the Cellpose measurement options."
                )
            if wants_mask_in_cell and not selected_targets:
                warnings.append(
                    f"{source_name}: measurements of configured masks within Cellpose cells are selected but no Weka or combined "
                    "mask is assigned. Turn ON a channel/mask pair or turn off those measurement options."
                )

            populations = list(source_def.get("cell_populations", []) or [])
            has_groups = any(
                str(group.get("cell_qc_limits", "") or "").strip()
                or str(group.get("mask_qc_limits", "") or "").strip()
                or bool(group.get("exclude_from_csv", False))
                for group in populations
                if isinstance(group, dict)
            )
            if has_groups and not selected_cellpose:
                warnings.append(
                    f"{source_name}: cell groups are configured but Cellpose masks are disabled. "
                    "Select a Cellpose mask or clear the cell-group conditions for this channel."
                )

        return warnings

    # Keep the GUI method for extensions while the reusable grouping logic stays Qt-free.
    @staticmethod
    def compact_repeated_warnings(warnings: list[str]) -> list[str]:
        return compact_setup_warnings(warnings)

    # Feedback stays beside the matrix so mistakes are visible while the user is still editing it.
    def update_analysis_matrix_warning_label(self):
        if not hasattr(self, "analysis_matrix_warning_label"):
            return
        if not hasattr(self, "analysis_matrix_table"):
            return
        if self._rebuilding_image_tabs:
            return
        if self.analysis_matrix_table.rowCount() <= 0:
            self.analysis_matrix_warning_label.setText(
                "Measurement setup: no measured channels yet. Configure channels first, then assign masks."
            )
            self.set_label_message_state(self.analysis_matrix_warning_label, "warning")
            return

        try:
            warnings = self.get_analysis_matrix_warnings()
        except Exception as exc:
            self.analysis_matrix_warning_label.setText(f"WARNING: Measurement setup check unavailable: {exc}")
            self.set_label_message_state(self.analysis_matrix_warning_label, "warning")
            return

        active_defs = self.get_active_image_definitions()
        has_measurement_target = any(
            bool(
                cellpose_mask_source_names(image_def)
                or (
                    "analysis_cellpose_mask_source" not in image_def
                    and image_def.get("analysis_cell_segmentation_enabled", False)
                )
            )
            or any(bool(value) for value in dict(image_def.get("mask_relationships", {}) or {}).values())
            for image_def in active_defs
            if not hasattr(self, "is_physical_channel_definition") or self.is_physical_channel_definition(image_def)
        )
        if not has_measurement_target:
            self.analysis_matrix_warning_label.setText(
                "No measurements selected. Turn ON a configured-mask or Cellpose-mask intersection."
            )
            self.set_label_message_state(self.analysis_matrix_warning_label, "warning")
            return

        if not warnings:
            self.analysis_matrix_warning_label.setText("Measurement setup: OK. No warnings found.")
            self.set_label_message_state(self.analysis_matrix_warning_label, "ok")
            return

        compacted = self.compact_repeated_warnings(warnings)
        shown = "Warnings:\n" + "\n".join(f"- {warning}" for warning in compacted[:5])
        if len(compacted) > 5:
            shown += f"\n... plus {count_label(len(compacted) - 5, 'more warning')}."

        self.analysis_matrix_warning_label.setText(shown)
        self.set_label_message_state(self.analysis_matrix_warning_label, "warning")

    # Dynamic properties let the shared stylesheet own warning colors instead of hard-coding widget styles here.
    @staticmethod
    def set_label_message_state(label, state: str):
        label.setProperty("messageState", state)
        label.style().unpolish(label)
        label.style().polish(label)
        label.update()

    # Ask before continuing because matrix warnings can be intentional and should not always block a run.
    def validate_matrix_before_run(self) -> bool:
        warnings = self.get_analysis_matrix_warnings()
        if not warnings:
            return True

        compacted = self.compact_repeated_warnings(warnings)
        text = "Cellonaut detected possible measurement setup issues:\n\n"
        text += "\n".join(f"- {w}" for w in compacted[:12])
        if len(compacted) > 12:
            text += f"\n... plus {count_label(len(compacted) - 12, 'more warning')}."

        text += "\n\nContinue anyway?"

        reply = QMessageBox.question(
            self,
            "Measurement setup warnings",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return reply == QMessageBox.StandardButton.Yes

    # Show only enabled relationships in the setup summary.
    def get_active_analysis_relationships_for_summary(self) -> list[dict]:
        active_defs = self.get_active_image_definitions()
        names = [img["name"].strip() or f"Image{i+1}" for i, img in enumerate(active_defs)]

        active_relationships = []

        for source_def in active_defs:
            if hasattr(self, "is_physical_channel_definition") and not self.is_physical_channel_definition(source_def):
                continue
            source_name = source_def.get("name", "").strip()
            source_relationships = dict(source_def.get("mask_relationships", {}))

            for target_name in names:
                enabled = bool(source_relationships.get(target_name, False))
                if not enabled:
                    continue

                active_relationships.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "self": source_name == target_name,
                    }
                )

            for cellpose_source in cellpose_mask_source_names(source_def):
                active_relationships.append(
                    {
                        "source": source_name,
                        "target": f"{cellpose_source}_Cellpose",
                        "self": source_name == cellpose_source,
                        "kind": "cellpose",
                    }
                )

        return active_relationships

    # Capture widget-backed state on the GUI thread; filesystem discovery is
    # deliberately excluded so this snapshot remains fast and thread-safe.
    def build_configuration_summary_snapshot(self) -> tuple[Any, list[dict], list[dict], list[str], dict[str, Any]]:
        cfg = self.collect_config()
        active_defs = self.get_active_image_definitions()
        relationships = self.get_active_analysis_relationships_for_summary()
        warnings = self.get_analysis_matrix_warnings()

        summary: dict[str, Any] = {
            "folders": {
                "input": str(cfg.input_dir),
                "output": str(cfg.output_dir),
                "fiji": str(cfg.fiji_app_path),
                "layout": str(cfg.input_structure),
            },
            "images": [],
            "relationships": relationships,
            "cell_segmentation": [],
            "filters": [],
            "measurements": [],
            "warnings": warnings,
            "dry_run": {},
            "setup_checks": [],
        }

        measurement_options = dict(getattr(cfg, "measurement_options", self.measurement_options) or {})
        enabled_measurements = [
            key for key, value in measurement_options.items()
            if bool(value)
        ]
        summary["measurements"].append(
            {
                "source": "All measured channels",
                "enabled": enabled_measurements,
            }
        )

        for img in active_defs:
            classifier = str(img.get("classifier", "") or "").strip()
            is_mask = not self.is_physical_channel_definition(img)
            summary["images"].append(
                {
                    "type": "Mask" if is_mask else "Channel",
                    "name": str(img.get("name", "") or "").strip(),
                    "folder": str(img.get("folder", "") or "").strip(),
                    "mask_source": str(img.get("mask_source_mode", "") or "").strip() if is_mask else "",
                    "classifier": classifier,
                    "classifier_name": Path(classifier).name if classifier else "",
                    "probability_class": str(img.get("probability_class_index", "1") or "1") if is_mask else "-",
                    "threshold": str(img.get("threshold_method", "Default") or "Default") if is_mask else "-",
                }
            )

            if bool(img.get("analysis_cell_segmentation_enabled", False)):
                summary["cell_segmentation"].append(
                    {
                        "source": str(img.get("name", "") or "").strip(),
                        "seg_source": str(img.get("analysis_cell_segmentation_source", "") or "").strip(),
                        "diameter": str(img.get("cell_diameter", DEFAULT_CELL_DIAMETER) or "Original scale"),
                        "min_size": str(img.get("cell_min_size", DEFAULT_CELL_MIN_SIZE) or DEFAULT_CELL_MIN_SIZE),
                        "cellprob": str(
                            img.get("cellprob_threshold", DEFAULT_CELLPROB_THRESHOLD) or DEFAULT_CELLPROB_THRESHOLD
                        ),
                        "flow": str(img.get("flow_threshold", DEFAULT_FLOW_THRESHOLD) or DEFAULT_FLOW_THRESHOLD),
                        "remove_border": bool(img.get("cell_remove_border", True)),
                    }
                )

            for population in img.get("cell_populations", []) or []:
                if not isinstance(population, dict):
                    continue
                filters = str(population.get("cell_qc_limits", "") or "").strip()
                mask_filters = str(population.get("mask_qc_limits", "") or "").strip()
                exclude = bool(population.get("exclude_from_csv", False))
                summary["filters"].append(
                    {
                        "source": (
                            f"{str(img.get('name', '') or '').strip()} · "
                            f"{str(population.get('name', 'Cell group') or 'Cell group')}"
                        ),
                        "cell_limits": filters,
                        "mask_limits": mask_filters,
                        "exclude": exclude,
                        "mode": QC_FILTER_MODE_LABELS[normalize_qc_filter_mode(population.get("qc_filter_mode", ""))],
                        "mask_intensity_source": MASK_INTENSITY_SOURCE_LABELS.get(
                            population.get("mask_qc_intensity_source", "Measured image"), "Measured channel"
                        ),
                    }
                )

        return cfg, active_defs, relationships, warnings, summary

    # Prefer completed background scans because repeating jar and folder walks would make Check Setup feel frozen.
    def build_setup_check_items(
        self,
        cfg,
        active_defs: list[dict],
        relationships: list[dict],
        warnings: list[str],
        dry_run: dict,
    ) -> list[dict]:
        return create_setup_check_items(
            cfg,
            active_defs,
            relationships,
            warnings,
            dry_run,
            cached_fiji_scan=getattr(self, "_last_fiji_scan_result", {}) or {},
            cached_input_scan=getattr(self, "_last_input_scan_result", {}) or {},
            nd2_detected_channel_names=getattr(self, "nd2_detected_channel_names", []) or [],
        )

    # Retain the public formatter used by the dialog and external callers.
    @staticmethod
    def format_configuration_summary_text(summary: dict) -> str:
        return format_setup_summary_text(summary)

    def set_setup_check_busy(self, busy: bool) -> None:
        enabled = not busy and not bool(getattr(self, "_primary_task_active", False))
        for name in (
            "run_button",
            "preview_pipeline_button",
            "check_setup_button",
            "mask_source_report_button",
            "nd2_import_button",
            "nd2_convert_button",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(enabled)
        if not busy and hasattr(self, "update_nd2_channel_summary"):
            self.update_nd2_channel_summary()

    # Snapshot the widgets immediately, then run directory discovery and Fiji
    # inspection in a worker so Check Setup never freezes the main event loop.
    def show_setup_check_report(self):
        thread = getattr(self, "_setup_check_thread", None)
        if thread is not None and thread.isRunning():
            return
        try:
            cfg, active_defs, relationships, warnings, summary = self.build_configuration_summary_snapshot()
        except Exception as exc:
            QMessageBox.critical(self, "Could not check setup", str(exc))
            return

        worker = SetupCheckWorker(
            summary,
            cfg,
            active_defs,
            relationships,
            warnings,
            cached_fiji_scan=getattr(self, "_last_fiji_scan_result", {}) or {},
            cached_input_scan=getattr(self, "_last_input_scan_result", {}) or {},
            nd2_detected_channel_names=getattr(self, "nd2_detected_channel_names", []) or [],
        )
        thread, worker = self.prepare_worker_thread(
            worker,
            terminal_signal=worker.done_signal,
            result_callback=self.store_setup_check_result,
            finished_callback=self.cleanup_setup_check,
            thread_factory=QThread,
        )
        self._pending_setup_check_result = None
        self._setup_check_worker = worker
        self._setup_check_thread = thread
        self.set_setup_check_busy(True)
        self.update_worker_status(stage_update(PipelineStage.CHECKING, "Checking setup"))
        self.log("[SETUP CHECK] Checking folders, samples, and external tools in the background...")
        thread.start()

    @Slot(dict)
    def store_setup_check_result(self, result: dict) -> None:
        """Retain the worker result until its QThread has completely stopped."""
        self._pending_setup_check_result = dict(result)

    @Slot(dict)
    def apply_setup_check_result(self, result: dict) -> None:
        if result.get("cancelled"):
            self.log("[SETUP CHECK] Check cancelled.")
            return
        if result.get("error"):
            QMessageBox.critical(self, "Could not check setup", str(result["error"]))
            return

        summary = dict(result.get("summary") or {})
        text = str(result.get("text", "") or "")
        try:
            output_text = str((summary.get("folders") or {}).get("output", "") or "").strip()
            if output_text:
                out_dir = Path(output_text)
                out_dir.mkdir(parents=True, exist_ok=True)
                report_path = out_dir / "setup_check_report.txt"
                write_text(report_path, text, encoding="utf-8")
                self.log(f"[SETUP CHECK] Saved report: {report_path}")
        except Exception as exc:
            self.log(f"[WARN] Could not save setup-check report: {exc}")

        dialog = ConfigurationSummaryDialog(summary, text, self)
        dialog.exec()

    @Slot()
    def cleanup_setup_check(self) -> None:
        result = getattr(self, "_pending_setup_check_result", None)
        self._pending_setup_check_result = None
        self._setup_check_worker = None
        self._setup_check_thread = None
        self.set_setup_check_busy(False)
        if getattr(self, "_worker_status_stage", None) == PipelineStage.CHECKING:
            self.update_worker_status("Ready")
        # Deliver through the main-window queued signal only after QThread.finished
        # has returned. A modal report starts a nested event loop, so opening it
        # while Qt is still deleting the worker/thread can cause a native crash.
        if result is not None and not bool(getattr(self, "_close_retry_scheduled", False)):
            self.setup_check_result_on_gui.emit(result)

    # Only physical channels correspond to input folders; generated masks must not become file requirements.
    def get_expected_image_folders(self) -> list[str]:
        active_defs = self.get_active_image_definitions()
        return [
            str(img.get("folder", "") or "").strip()
            for img in active_defs
            if str(img.get("folder", "") or "").strip()
            and (not hasattr(self, "is_physical_channel_definition") or self.is_physical_channel_definition(img))
        ]

    # Block empty runs, but let users deliberately continue when only part of a batch is runnable.
    def validate_pipeline_readiness_for_config(self, cfg) -> bool:
        if not self.validate_config_before_output(cfg):
            return False
        try:
            readiness = analyze_sample_readiness(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Run setup check failed", str(exc))
            return False
        self._last_pipeline_readiness = readiness

        total = int(readiness.get("total", 0) or 0)
        ready = int(readiness.get("ready", 0) or 0)
        blocked = int(readiness.get("blocked", readiness.get("incomplete", 0)) or 0)

        if total <= 0:
            QMessageBox.critical(
                self,
                "No samples found",
                "No TIFF samples were detected in the input folder.\n\n"
                "Open Check Setup to inspect the detected folder layout.",
            )
            return False

        if ready <= 0:
            first_problem = ""
            problems = readiness.get("problems", []) or []
            if problems:
                first_problem = "\n\nFirst problem:\n"
                first_problem += "\n".join(problems[0].get("missing_required", []) or problems[0].get("messages", []))

            QMessageBox.critical(
                self,
                "No runnable samples",
                "Samples were detected, but none have all input files needed for this run.\n\n"
                "Fix the missing channel folders/files listed below, or adjust the measured channels and mask sources in Channels."
                f"{first_problem}",
            )
            return False

        if blocked > 0:
            problems = readiness.get("problems", []) or []
            preview = []
            for entry in problems[:8]:
                missing = ", ".join(entry.get("missing_required", []) or entry.get("messages", []) or ["unknown issue"])
                preview.append(f"- {entry.get('sample', '(unknown sample)')}: {missing}")

            text = (
                f"{ready}/{total} samples are ready, but input files are missing for "
                f"{count_label(blocked, 'sample')}.\n\n"
                "The full run can continue with the ready samples and report the blocked ones as failures/skips.\n\n"
                + "\n".join(preview)
                + "\n\nContinue anyway?"
            )

            reply = QMessageBox.question(
                self,
                "Some samples are blocked",
                text,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return reply == QMessageBox.StandardButton.Yes

        return True
