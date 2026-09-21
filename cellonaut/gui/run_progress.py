"""Visible progress for analysis and preview runs."""

import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Slot
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from cellonaut.runtime import PipelineStage
from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.file_browser import open_folder_in_system_browser


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class RunProgressReceiver(QObject):
    """Receive worker updates on the GUI thread before touching widgets."""

    def __init__(self, window):
        super().__init__(window.as_qobject())
        self.window = window

    @Slot(int)
    def progress(self, value):
        self.window.update_run_progress(value)

    @Slot(str)
    def sample(self, text):
        self.window.update_run_sample(text)

    @Slot(str)
    def counter(self, text):
        self.window.update_run_counter(text)

    @Slot(dict)
    def stage(self, update):
        self.window.update_run_stage(update)


class CellonautGuiRunProgressMixin(GuiMixin):
    def build_run_progress_panel(self):
        self.run_progress_receiver = RunProgressReceiver(self)
        self.run_progress_panel = QFrame()
        self.run_progress_panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self.run_progress_panel)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(4)
        header = QGridLayout()
        header.setColumnStretch(0, 1)
        header.setColumnStretch(2, 1)
        self.run_stage_row = QHBoxLayout()
        header.addLayout(self.run_stage_row, 0, 0)
        self.run_progress_title = QLabel("Running analysis")
        self.run_elapsed_label = QLabel()
        self.run_progress_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.run_elapsed_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.run_elapsed_label, 0, 2, Qt.AlignmentFlag.AlignRight)
        heading = QHBoxLayout()
        heading.addWidget(self.run_progress_title)
        self.run_sample_label = QLabel()
        self.run_sample_label.setWordWrap(False)
        self.run_sample_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.addWidget(self.run_sample_label)
        header.addLayout(heading, 0, 1)
        layout.addLayout(header)
        activity = QVBoxLayout()
        self.run_stage_label = QLabel()
        self.run_stage_label.setWordWrap(True)
        self.run_stage_label.setTextFormat(Qt.TextFormat.PlainText)
        self.run_sample_label.setTextFormat(Qt.TextFormat.PlainText)
        self.run_stage_row.addWidget(self.run_stage_label)
        self.run_stage_row.addStretch()
        self.run_activity = QProgressBar()
        self.run_activity.setRange(0, 100)
        self.run_activity.setValue(0)
        self.run_activity.setFixedHeight(8)
        self.run_activity.setTextVisible(False)
        self.run_activity.setAccessibleName("Pipeline steps completed")
        self.run_activity.setToolTip("Advances at pipeline stage boundaries; not an estimate of time remaining.")
        activity.addWidget(self.run_activity)
        layout.addLayout(activity)
        self.run_overall_progress = QProgressBar()
        self.run_overall_progress.setRange(0, 100)
        self.run_overall_progress.setAccessibleName("Samples processed")
        self.run_overall_progress.setFormat("%p% of samples processed")
        layout.addWidget(self.run_overall_progress)
        actions = QHBoxLayout()
        self.run_open_results = QPushButton("Open results")
        self.run_open_results.clicked.connect(self.open_run_progress_results)
        actions.addWidget(self.run_open_results)
        self.run_target_counts = []
        for color in ("#36a269", "#e05252", "#c68a20"):
            label = QLabel()
            label.setProperty("countColor", color)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.hide()
            self.run_target_counts.append(label)
            actions.addWidget(label)
        actions.addStretch()
        layout.addLayout(actions)
        self.run_elapsed_timer = QTimer(self.as_qobject())
        self.run_elapsed_timer.setInterval(1000)
        self.run_elapsed_timer.timeout.connect(self.refresh_run_elapsed)
        self._run_panel_active = False
        self._run_panel_cancelling = False
        self.root_layout.addWidget(self.run_progress_panel)
        self.run_progress_panel.hide()

    def begin_run_progress(self, output_dir, *, preview=False):
        self._run_panel_preview = preview
        self._run_panel_label = "Running preview" if preview else "Running analysis"
        self._run_panel_active = True
        self._run_panel_cancelling = False
        self._run_panel_output = str(output_dir)
        self._run_panel_started = time.monotonic()
        self.run_progress_title.setText(self._run_panel_label)
        self.run_activity.setRange(0, 100)
        self.run_activity.setValue(0)
        self.run_activity.show()
        unit = "targets" if preview else "samples"
        self.run_overall_progress.setFormat(f"%p% of {unit} processed")
        self.run_overall_progress.setAccessibleName(f"{unit.capitalize()} processed")
        self.run_stage_label.setText("Preparing analysis…")
        self.run_sample_label.setText("First sample" if preview else "Waiting for sample list")
        self.run_overall_progress.setValue(0)
        self.run_open_results.hide()
        for label in self.run_target_counts:
            label.hide()
        self.run_progress_panel.show()
        self.refresh_run_elapsed()
        self.run_elapsed_timer.start()

    def refresh_run_elapsed(self):
        elapsed = max(0.0, time.monotonic() - self._run_panel_started)
        elapsed_text = format_duration(elapsed)
        progress = self.run_overall_progress.value()
        if self._run_panel_active and 0 < progress < 100:
            remaining = elapsed * (100 - progress) / progress
            self.run_elapsed_label.setText(f"{elapsed_text} elapsed · ETA {format_duration(remaining)}")
        elif self._run_panel_active:
            self.run_elapsed_label.setText(f"{elapsed_text} elapsed · ETA calculating…")
        else:
            self.run_elapsed_label.setText(f"{elapsed_text} elapsed")

    @Slot(int)
    def update_run_progress(self, value):
        if self._run_panel_active:
            self.run_overall_progress.setValue(value)
            self.refresh_run_elapsed()

    @Slot(str)
    def update_run_sample(self, text):
        if self._run_panel_active:
            if str(text) != self.run_sample_label.text():
                self.run_activity.setValue(0)
            self.run_sample_label.setText(str(text))
            self.run_sample_label.setToolTip(str(text))

    @Slot(str)
    def update_run_counter(self, text):
        if self._run_panel_active and not self._run_panel_cancelling and text != "0 / 0":
            self.run_progress_title.setText(f"{self._run_panel_label} · {'Target' if self._run_panel_preview else 'Sample'} {text}")

    @Slot(dict)
    def update_run_stage(self, update):
        stage = update["stage"]
        text = update["text"]
        if self._run_panel_active and not self._run_panel_cancelling:
            if stage not in {PipelineStage.SUCCESS, PipelineStage.ERROR, PipelineStage.CANCELLED, PipelineStage.PARTIAL}:
                self.run_stage_label.setText(text)
                # Stages can be skipped or repeated across targets; never move backwards within a sample.
                milestones = {
                    PipelineStage.PREPARING: 0,
                    PipelineStage.INITIALIZING: 5, PipelineStage.LOADING: 10,
                    PipelineStage.MASKS: 30, PipelineStage.SEGMENTATION: 75,
                    PipelineStage.MEASUREMENT: 55, PipelineStage.EXPORT: 90,
                }
                if stage in milestones:
                    self.run_activity.setValue(max(self.run_activity.value(), milestones[stage]))

    def cancel_run_progress(self):
        if not self._run_panel_active:
            return
        self._run_panel_cancelling = True
        self.run_progress_title.setText("Cancelling analysis…")
        self.run_stage_label.setText("Stopping processing. Waiting for the worker to finish…")

    def finish_run_progress(self, outcome):
        if not self._run_panel_active:
            return
        self.run_elapsed_timer.stop()
        self._run_panel_active = False
        self.refresh_run_elapsed()
        state = outcome.get("state", "error")
        self.run_progress_title.setText({
            "success": "Analysis completed", "partial": "Completed with errors",
            "cancelled": "Analysis cancelled", "error": "Analysis failed",
        }.get(state, "Analysis failed"))
        stats = (outcome.get("result") or {}).get("run_stats") or {}
        if state in {"success", "partial"}:
            total = int(stats.get("total_samples", 0))
            failed = int(stats.get("failed_targets", 0))
            skipped = int(stats.get("skipped_targets", 0))
            successful = int(stats.get("processed_targets", 0))
            for label, count, name in zip(self.run_target_counts, (successful, failed, skipped), ("successful", "failed", "skipped")):
                separator = "" if name == "successful" else "&middot; &nbsp;"
                color = label.property("countColor")
                label.setText(f'{separator}<span style="color: {color}">{count}</span> {name} targets')
                label.show()
            self.run_stage_label.setText(f"{total} samples processed")
            if self._run_panel_preview:
                self.run_progress_title.setText("Preview ready" if state == "success" else "Preview ready with errors")
                self.run_stage_label.setText("Preview finished.")
            self.run_activity.setValue(100)
            self.run_overall_progress.setValue(100)
        else:
            message = str(outcome.get("error") or "Processing stopped.")
            self.run_stage_label.setText(f"{message}\nResults may be incomplete.")
        self.run_sample_label.setText(f"Results folder: {self._run_panel_output}")
        self.run_open_results.setVisible(Path(self._run_panel_output).is_dir())

    def open_run_progress_results(self):
        open_folder_in_system_browser(Path(self._run_panel_output))
