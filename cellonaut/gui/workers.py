"""Qt-side worker wiring for pipeline, preview, and ND2 jobs.

The lower-level workers emit plain signals from background work. This mixin
connects those signals to the main window, keeps status text responsive, and
cleans up Qt threads after each job finishes.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtWidgets import QMessageBox
from cellonaut.runtime import PipelineStage, StageUpdate
from cellonaut.gui.mixin import GuiMixin
from cellonaut.pipeline.models import Config
from cellonaut.pipeline.run_outputs import config_for_numbered_child_output
from cellonaut.workers import PipelineWorker


# Isolate Qt signal wiring from scientific work so worker lifecycle and native
# subprocess cancellation stay responsive without leaking into pipeline code.
class CellonautGuiWorkersMixin(GuiMixin):
    """Coordinate Qt worker lifecycles for mutually exclusive native tasks."""

    # Worker references cover the brief startup and shutdown windows where a
    # thread may not yet report running but starting another native task would
    # still be unsafe.
    def active_processing_workers(self) -> list[tuple[Any, str, str]]:
        return self.task_state.active_workers()

    # Fiji, Cellpose, and ND2 isolation all use substantial native runtimes, so
    # allowing any two primary jobs to overlap would compete for files and memory.
    def has_active_processing_task(self) -> bool:
        return bool(self.active_processing_workers())

    # Show elapsed time instead of fake progress while native stages cannot report useful percentages.
    def update_worker_status(self, update: StageUpdate | str) -> None:
        # Plain text is accepted for presentation-only local GUI notices.
        text = update["text"] if isinstance(update, dict) else str(update or "")
        stage = update["stage"] if isinstance(update, dict) else None
        previous_stage = getattr(self, "_worker_status_stage", None)
        self._worker_status_stage = stage
        self._worker_status_base = text
        long_stage = stage in {
            PipelineStage.INITIALIZING,
            PipelineStage.STAGING,
            PipelineStage.SCANNING,
            PipelineStage.CHECKING,
        }
        if long_stage:
            if hasattr(self, "progress_bar"):
                self.progress_bar.setRange(0, 0)
            if previous_stage != stage or not getattr(self, "_worker_status_started_at", None):
                self._worker_status_started_at: float | None = time.monotonic()
            timer = getattr(self, "_worker_status_heartbeat", None)
            if timer is None:
                timer = QTimer(self.as_qobject())
                timer.setInterval(1000)
                timer.timeout.connect(self.refresh_worker_status_heartbeat)
                self._worker_status_heartbeat = timer
            if not timer.isActive():
                timer.start()
            self.refresh_worker_status_heartbeat()
            return

        self.stop_worker_status_heartbeat()
        self.status_label.setText(text)

    # Update elapsed time while the task cannot report progress.
    def refresh_worker_status_heartbeat(self) -> None:
        started_at = getattr(self, "_worker_status_started_at", None)
        base = str(getattr(self, "_worker_status_base", "") or "")
        if started_at is None or not base:
            return
        elapsed = max(0, int(time.monotonic() - started_at))
        self.status_label.setText(f"{base}... {elapsed}s")

    # Stop elapsed-time updates and restore the progress bar's 0-100 range.
    def stop_worker_status_heartbeat(self) -> None:
        timer = getattr(self, "_worker_status_heartbeat", None)
        if timer is not None:
            timer.stop()
        if hasattr(self, "progress_bar") and self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self._worker_status_started_at = None
        self._worker_status_base = ""

    # Connect the worker and cleanup handlers without starting the thread,
    # so callers can store references first.
    def prepare_worker_thread(
        self,
        worker: Any,
        *,
        terminal_signal: Any | None = None,
        result_callback: Callable[..., None] | None = None,
        finished_callback: Callable[[], None] | None = None,
        thread_factory: Callable[[Any], Any] | None = None,
    ) -> tuple[Any, Any]:
        thread = (thread_factory or QThread)(self.as_qobject())
        worker.moveToThread(thread)

        thread.started.connect(worker.run)

        if hasattr(worker, "log_signal"):
            worker.log_signal.connect(self.log, Qt.ConnectionType.QueuedConnection)

        if hasattr(worker, "status_signal"):
            worker.status_signal.connect(self.update_worker_status, Qt.ConnectionType.QueuedConnection)

        terminal = terminal_signal or getattr(worker, "terminal_signal", None)
        if terminal is not None:
            if result_callback is not None:
                terminal.connect(result_callback, Qt.ConnectionType.QueuedConnection)
            # Connect QObject slots directly. Wrapping them in lambdas makes
            # PySide execute the callbacks in the worker thread; that can stop
            # its event loop before the queued result reaches the GUI.
            terminal.connect(thread.quit)
            terminal.connect(worker.deleteLater)

        if finished_callback is not None:
            thread.finished.connect(finished_callback)
        thread.finished.connect(thread.deleteLater)
        return thread, worker

    # Connect cleanup before start because an immediate validation failure can
    # otherwise finish the thread before the caller has attached its slot.
    def run_worker(
        self,
        worker: Any,
        finished_callback: Callable[[], None] | None = None,
    ) -> tuple[Any, Any]:
        thread, worker = self.prepare_worker_thread(
            worker,
            terminal_signal=getattr(worker, "terminal_signal", None),
            finished_callback=finished_callback,
        )
        thread.start()
        return thread, worker

    # Complete all GUI validation before reserving an output folder or starting background work.
    def run_clicked(self) -> None:
        if self.has_active_processing_task():
            return
        if self.run_pipeline_from_nd2_input_if_needed():
            return

        if not self.validate_all_fields():
            QMessageBox.warning(
                self,
                "Invalid settings",
                "Some fields contain invalid values.\n\n"
                "Please correct the highlighted fields before running the pipeline.",
            )
            return
        try:
            cfg = self.collect_config()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        if not self.validate_matrix_before_run():
            return

        if not self.validate_pipeline_readiness_for_config(cfg):
            return

        if not self.confirm_pipeline_disk_space(
            cfg,
            preview=False,
            readiness=getattr(self, "_last_pipeline_readiness", None),
        ):
            return

        self.run_clicked_from_config(cfg)

    # Allocate the numbered output synchronously so two starts cannot choose the same run folder.
    def run_clicked_from_config(self, cfg: Config) -> None:
        if self.has_active_processing_task():
            return

        try:
            cfg = config_for_numbered_child_output(cfg, "run")
            self.log(f"[RUN] Output folder: {cfg.output_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Run output error", str(exc))
            return

        self.stop_worker_status_heartbeat()
        self.running = True
        self.begin_run_progress(cfg.output_dir)
        self.status_label.setText("Preparing pipeline...")
        self.set_status_style("Running")
        self.set_pipeline_busy(True)
        self.set_primary_task_active(True)
        self.progress_bar.setValue(0)
        self.current_sample_label.setText("Preparing run configuration...")
        self.sample_counter_label.setText("0 / 0")
        self.log("[RUN] Starting pipeline...")

        self.worker = PipelineWorker(cfg)
        self.worker.progress_signal.connect(self.run_progress_receiver.progress, Qt.ConnectionType.QueuedConnection)
        self.worker.sample_signal.connect(self.run_progress_receiver.sample, Qt.ConnectionType.QueuedConnection)
        self.worker.counter_signal.connect(self.run_progress_receiver.counter, Qt.ConnectionType.QueuedConnection)
        self.worker.status_signal.connect(self.run_progress_receiver.stage, Qt.ConnectionType.QueuedConnection)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.sample_signal.connect(self.current_sample_label.setText)
        self.worker.counter_signal.connect(self.sample_counter_label.setText)
        self.worker.terminal_signal.connect(self.pipeline_worker_finished_on_gui.emit)
        self.worker.status_signal.connect(self.pipeline_status_on_gui.emit)

        self.worker_thread, self.worker = self.run_worker(
            self.worker,
            self.cleanup_pipeline_worker_refs,
        )

        self.save_last_settings()

    # Restore the main controls before showing the modal error so the window is usable afterward.
    @Slot(str)
    def on_pipeline_error(self, msg: str) -> None:
        self.running = False
        self.set_primary_task_active(False)
        self.status_label.setText("Error")
        self.set_status_style("Error")
        self.set_pipeline_busy(False)
        QMessageBox.critical(self, "Pipeline error", msg)

    # Clear references only after QThread has finished so cancellation can still reach an exiting worker.
    @Slot()
    def cleanup_pipeline_worker_refs(self) -> None:
        self.stop_worker_status_heartbeat()
        self.task_state.pipeline.clear()

    # Request every live worker defensively, but present one stable cancellation state in the GUI.
    def cancel_current_task(self) -> None:
        active_tasks = self.active_processing_workers()
        if not active_tasks:
            self.log("[STATUS] No running task to cancel.")
            return

        self.stop_worker_status_heartbeat()
        for worker, scope, _label in active_tasks:
            worker.request_cancel()
            self.log(f"[{scope}] Cancellation requested. Stopping the background worker...")

        self.cancel_run_progress()
        task_label = active_tasks[0][2]
        self.status_label.setText(f"Stopping {task_label}...")
        self.current_sample_label.setText(f"Stopping {task_label}...")
        self.set_status_style("Running")
        self.cancel_button.setEnabled(False)
        self.sample_counter_label.setText("Cancelling")

    @Slot()
    def on_nd2_import_cancelled(self) -> None:
        self.log("[ND2] Import cancelled.")
        self.status_label.setText("Cancelled")
        self.set_status_style("Idle")
        self.current_sample_label.setText("ND2 conversion cancelled")
        self.set_primary_task_active(False)
        self.set_pipeline_busy(False)

    # Return the full-run controls to idle without treating a user request as an error.
    @Slot()
    def on_pipeline_cancelled(self) -> None:
        self.running = False
        self.log("[RUN] Pipeline cancelled.")
        self.status_label.setText("Cancelled")
        self.set_status_style("Idle")
        self.current_sample_label.setText("Run cancelled")
        self.sample_counter_label.setText("Cancelled")
        self.set_primary_task_active(False)
        self.set_pipeline_busy(False)

    # The existing preview remains visible because a cancelled replacement never produced a complete result.
    @Slot()
    def on_preview_cancelled(self) -> None:
        self._preview_was_cancelled = True
        self.log("[PREVIEW] Preview cancelled.")
        self.status_label.setText("Cancelled")
        self.set_status_style("Idle")
        self.current_sample_label.setText("Preview cancelled")
        self.sample_counter_label.setText("Cancelled")
        self.set_primary_task_active(False)
        self.set_pipeline_busy(False)

    # Handle success, cancellation, or failure.
    @Slot(dict)
    def on_pipeline_worker_finished(self, outcome: dict) -> None:
        self.finish_run_progress(outcome)
        state = str(outcome.get("state", "error"))
        if state == "cancelled":
            self.on_pipeline_cancelled()
            return
        if state == "error":
            self.on_pipeline_error(str(outcome.get("error") or "The pipeline failed."))
            return
        self.on_worker_done(
            outcome.get("result") or {},
            completed_with_errors=(state == "partial"),
        )

    # Update result navigation only for completed runs because failed or cancelled outputs may be partial.
    def on_worker_done(
        self,
        result: dict | None = None,
        *,
        completed_with_errors: bool = False,
    ):
        self.running = False
        self.set_primary_task_active(False)

        stats = (result or {}).get("run_stats") or {}
        total = int(stats.get("total_targets", 0) or 0)
        failed = int(stats.get("failed_targets", 0) or 0)
        output_dir = str((result or {}).get("output_dir", "") or "")
        self._latest_pipeline_output_dir = output_dir
        last_run = f"Last run completed: {total} target(s), {failed} failed"
        if output_dir:
            last_run += f", output: {output_dir}"

        if completed_with_errors:
            self.status_label.setText("Completed with errors")
            self.set_status_style("Completed with errors")
            self.current_sample_label.setText(last_run)
            self.log("[RUN][WARN] Pipeline completed with errors; " "review the run summary and exported logs.")
        else:
            self.status_label.setText("Done")
            self.set_status_style("Done")
            self.current_sample_label.setText(last_run)
            self.log(f"[RUN] {last_run}")
        self.set_pipeline_busy(False)

        self.show_latest_qc_overlay()
