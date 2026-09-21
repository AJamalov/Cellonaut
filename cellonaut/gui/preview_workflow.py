"""Full-pipeline preview execution and worker lifecycle handling."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QMessageBox

from cellonaut.gui.mixin import GuiMixin
from cellonaut.pipeline.run_outputs import config_for_numbered_child_output
from cellonaut.workers import PreviewWorker


class CellonautGuiPreviewWorkflowMixin(GuiMixin):
    """Start preview workers and handle completion, cancellation, or failure."""

    # Release both references when the worker thread finishes.
    @Slot()
    def cleanup_preview_worker_refs(self):
        self.preview_worker = None
        self.preview_worker_thread = None

    # Accept only completed pipeline results so cancelled or failed workers cannot replace the current preview.
    @Slot(dict)
    def on_preview_pipeline_done(self, result: dict):
        self.set_pipeline_busy(False)
        self.set_primary_task_active(False)

        if not result or not isinstance(result, dict) or result.get("status") != "PROCESSED":
            return
        self._preview_was_cancelled = False

        self.progress_bar.setValue(100)
        completed_with_errors = bool(result.get("completed_with_errors"))
        self.status_label.setText("Preview ready with errors" if completed_with_errors else "Preview ready")
        self.set_status_style("Warning" if completed_with_errors else "Done")

        preview_path = result.get("preview_path")
        preview_output_dir = result.get("output_dir")
        if preview_output_dir:
            self.preview_state.artifact_results_root = Path(preview_output_dir)
        sample_label = result.get("sample_label", "Preview sample")
        rows = result.get("rows") or []
        sample_status = "Preview ready with errors" if completed_with_errors else "Preview ready"
        self.current_sample_label.setText(f"{sample_status}: {sample_label}")
        self.sample_counter_label.setText("1 / 1")

        if completed_with_errors:
            self.log("[PREVIEW][WARN] Some measured channels failed. Successful preview outputs remain available.")

        if preview_path and Path(preview_path).exists():
            self.preview_file(str(preview_path))
            self.log(f"[PREVIEW] Full pipeline preview saved to: {preview_path}")
        else:
            self.reset_preview_display(clear_title=True)

            self.preview_page_label.setText("Page: - / -")
            self.log("[PREVIEW] Preview finished, but no preview file path was returned.")

        self.update_preview_artifact_info(result)

        first_row = rows[0] if rows else None
        if first_row:
            self.log(f"[PREVIEW] Measurement row generated for {sample_label}.")
        else:
            self.log("[PREVIEW] No preview measurement row available.")

    # Handle worker outcomes on the GUI thread before updating widgets.
    @Slot(dict)
    def on_preview_worker_finished(self, outcome: dict):
        self.finish_run_progress(outcome)
        state = str(outcome.get("state", "error"))
        if state == "cancelled":
            self.on_preview_cancelled()
            return
        if state == "error":
            self.on_preview_pipeline_error(str(outcome.get("error") or "Preview failed."))
            return
        self.on_preview_pipeline_done(outcome.get("result") or {})

    # Reset busy state before opening the modal error dialog.
    @Slot(str)
    def on_preview_pipeline_error(self, msg: str):
        self.set_primary_task_active(False)
        self.status_label.setText("Preview error")
        self.set_status_style("Error")
        self.set_pipeline_busy(False)
        QMessageBox.critical(self, "Preview error", msg)

    # Run previews through the normal pipeline configuration so they faithfully represent a full run.
    def preview_pipeline_clicked(self):
        if not self.validate_all_fields():
            QMessageBox.warning(
                self,
                "Invalid settings",
                "Some fields contain invalid values.\n\n"
                "Please correct the highlighted fields before running the pipeline.",
            )
            return

        if not self.validate_matrix_before_run():
            return
        if self.has_active_processing_task():
            return

        try:
            cfg = self.collect_config()
        except Exception as e:
            QMessageBox.critical(self, "Invalid settings", str(e))
            return

        if not self.validate_config_before_output(cfg):
            return

        if not self.confirm_pipeline_disk_space(cfg, preview=True):
            return

        try:
            cfg = config_for_numbered_child_output(cfg, "preview")
            self.log(f"[PREVIEW] Output folder: {cfg.output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Preview output error", str(e))
            return

        self.stop_worker_status_heartbeat()
        self._preview_was_cancelled = False
        self.status_label.setText("Preparing preview...")
        self.set_status_style("Running")
        self.set_pipeline_busy(True)
        self.current_sample_label.setText("Preparing first-sample preview...")
        self.sample_counter_label.setText("1 / 1")
        self.progress_bar.setValue(0)

        self.set_primary_task_active(True)

        self.begin_run_progress(cfg.output_dir, preview=True)
        self.preview_worker = PreviewWorker(cfg)
        self.preview_worker.sample_signal.connect(self.run_progress_receiver.sample, Qt.ConnectionType.QueuedConnection)
        self.preview_worker.progress_signal.connect(self.run_progress_receiver.progress, Qt.ConnectionType.QueuedConnection)
        self.preview_worker.counter_signal.connect(self.run_progress_receiver.counter, Qt.ConnectionType.QueuedConnection)
        self.preview_worker.status_signal.connect(self.run_progress_receiver.stage, Qt.ConnectionType.QueuedConnection)
        self.preview_worker.progress_signal.connect(self.progress_bar.setValue)
        self.preview_worker.counter_signal.connect(self.sample_counter_label.setText)
        self.preview_worker.terminal_signal.connect(self.preview_worker_finished_on_gui.emit)
        self.preview_worker.status_signal.connect(self.preview_status_on_gui.emit)

        self.preview_worker_thread, self.preview_worker = self.run_worker(
            self.preview_worker,
            self.cleanup_preview_worker_refs,
        )
