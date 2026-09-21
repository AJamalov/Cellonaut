# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.gui import preview_workflow
from cellonaut.gui.preview_workflow import CellonautGuiPreviewWorkflowMixin


class SignalStub:
    def __init__(self):
        self.connections = []
        self.emitted = []

    def connect(self, callback, *_args):
        self.connections.append(callback)

    def emit(self, *args):
        self.emitted.append(args)


class WidgetStub:
    def __init__(self):
        self.value = None

    def setText(self, value):
        self.value = value

    def setValue(self, value):
        self.value = value


class PreviewHarness(CellonautGuiPreviewWorkflowMixin):
    def __init__(self):
        self.preview_state = PreviewState()
        self.run_progress_receiver = SimpleNamespace(progress=self.update_run_progress, counter=self.update_run_counter, stage=self.update_run_stage, sample=self.update_run_sample)
        self.calls = []
        self.progress_bar = WidgetStub()
        self.status_label = WidgetStub()
        self.current_sample_label = WidgetStub()
        self.sample_counter_label = WidgetStub()
        self.preview_page_label = WidgetStub()
        self.preview_worker_finished_on_gui = SignalStub()
        self.preview_status_on_gui = SignalStub()
        self.preview_worker = None
        self.preview_worker_thread = None

    def begin_run_progress(self, output_dir, *, preview=False):
        self.calls.append(("progress_start", output_dir, preview))

    def finish_run_progress(self, outcome):
        self.calls.append(("progress_finish", outcome))

    def update_run_progress(self, value):
        pass

    def update_run_sample(self, value):
        self.calls.append(("sample", value))

    def update_run_counter(self, value):
        pass

    def update_run_stage(self, value):
        pass

    def set_pipeline_busy(self, value):
        self.calls.append(("busy", value))

    def set_primary_task_active(self, value):
        self.calls.append(("primary", value))

    def set_status_style(self, value):
        self.calls.append(("style", value))

    def log(self, value):
        self.calls.append(("log", value))

    def preview_file(self, value):
        self.calls.append(("preview", value))

    def reset_preview_display(self, *, clear_title=False):
        self.clear_preview_canvas()
        self.clear_preview_layer_controls()

    def clear_preview_canvas(self):
        self.calls.append(("clear_canvas",))

    def clear_preview_layer_controls(self):
        self.calls.append(("clear_layers",))

    def update_preview_artifact_info(self, value):
        self.calls.append(("artifacts", value))

    def on_preview_cancelled(self):
        self.calls.append(("cancelled",))


def test_preview_done_routes_successful_result_to_visible_preview(tmp_path: Path):
    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"preview")
    gui = PreviewHarness()

    result = {
        "status": "PROCESSED",
        "preview_path": str(preview_path),
        "output_dir": str(tmp_path),
        "sample_label": "Sample A",
        "rows": [{"Area": 4}],
    }
    gui.on_preview_pipeline_done(result)

    assert gui.progress_bar.value == 100
    assert gui.status_label.value == "Preview ready"
    assert gui.current_sample_label.value == "Preview ready: Sample A"
    assert gui.preview_state.artifact_results_root == tmp_path
    assert ("preview", str(preview_path)) in gui.calls
    assert ("artifacts", result) in gui.calls


def test_preview_done_preserves_controls_for_non_processed_outcome_and_clears_missing_preview():
    gui = PreviewHarness()
    gui.on_preview_pipeline_done({"status": "FAILED"})
    assert gui.progress_bar.value is None

    gui.on_preview_pipeline_done({"status": "PROCESSED", "sample_label": "Sample B"})
    assert gui.preview_page_label.value == "Page: - / -"
    assert ("clear_canvas",) in gui.calls
    assert ("clear_layers",) in gui.calls


def test_preview_worker_terminal_states_are_routed(monkeypatch):
    gui = PreviewHarness()
    errors = []
    completed = []
    monkeypatch.setattr(gui, "on_preview_pipeline_error", errors.append)
    monkeypatch.setattr(gui, "on_preview_pipeline_done", completed.append)

    gui.on_preview_worker_finished({"state": "cancelled"})
    gui.on_preview_worker_finished({"state": "error", "error": "bad preview"})
    gui.on_preview_worker_finished({"state": "success", "result": {"status": "PROCESSED"}})

    assert ("cancelled",) in gui.calls
    assert errors == ["bad preview"]
    assert completed == [{"status": "PROCESSED"}]


def test_preview_error_restores_controls_before_showing_dialog(monkeypatch):
    gui = PreviewHarness()
    dialogs = []
    monkeypatch.setattr(preview_workflow.QMessageBox, "critical", lambda *args: dialogs.append(args))

    gui.on_preview_pipeline_error("model failed")

    assert gui.status_label.value == "Preview error"
    assert ("primary", False) in gui.calls
    assert ("busy", False) in gui.calls
    assert dialogs[0][1:] == ("Preview error", "model failed")


def test_preview_pipeline_clicked_builds_and_connects_worker(monkeypatch, tmp_path: Path):
    class WorkerStub:
        def __init__(self, cfg):
            self.cfg = cfg
            self.log_signal = SignalStub()
            self.sample_signal = SignalStub()
            self.progress_signal = SignalStub()
            self.counter_signal = SignalStub()
            self.terminal_signal = SignalStub()
            self.status_signal = SignalStub()

    cfg = SimpleNamespace(output_dir=tmp_path)
    gui = PreviewHarness()
    gui.validate_all_fields = lambda: True
    gui.validate_matrix_before_run = lambda: True
    gui.has_active_processing_task = lambda: False
    gui.collect_config = lambda: cfg
    gui.validate_config_before_output = lambda value: value is cfg
    gui.confirm_pipeline_disk_space = lambda value, *, preview: value is cfg and preview
    gui.stop_worker_status_heartbeat = lambda: gui.calls.append(("stop_heartbeat",))
    gui.run_worker = lambda worker, cleanup: ("thread", worker)
    monkeypatch.setattr(preview_workflow, "config_for_numbered_child_output", lambda value, name: value)
    monkeypatch.setattr(preview_workflow, "PreviewWorker", WorkerStub)

    gui.preview_pipeline_clicked()

    assert gui.preview_worker.sample_signal.connections == [gui.update_run_sample]
    assert gui.preview_worker.log_signal.connections == []
    gui.preview_worker.sample_signal.connections[0]("arbitrary sample name")
    assert ("sample", "arbitrary sample name") in gui.calls
    assert gui.preview_worker.cfg is cfg
    assert gui.preview_worker_thread == "thread"
    assert gui.status_label.value == "Preparing preview..."
    assert gui.progress_bar.value == 0
    assert gui.preview_worker.progress_signal.connections == [gui.update_run_progress, gui.progress_bar.setValue]
    assert gui.preview_worker.counter_signal.connections == [gui.update_run_counter, gui.sample_counter_label.setText]
    assert gui.preview_worker.terminal_signal.connections == [gui.preview_worker_finished_on_gui.emit]
    assert gui.preview_worker.status_signal.connections == [gui.update_run_stage, gui.preview_status_on_gui.emit]
