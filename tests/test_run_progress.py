import pytest
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtTest import QTest

from cellonaut.gui import run_progress as run_progress_gui
from cellonaut.gui.main_window import CellonautMainWindow
from cellonaut.runtime import PipelineStage, stage_update


pytestmark = pytest.mark.gui


class ProgressEmitter(QObject):
    progress = Signal(int)
    stage = Signal(dict)
    done = Signal()

    @Slot()
    def run(self):
        self.stage.emit(stage_update(PipelineStage.MASKS, "Preparing masks"))
        self.progress.emit(50)
        self.done.emit()


def test_progress_from_worker_is_delivered_on_gui_thread(window, tmp_path, qt_application):
    window.begin_run_progress(tmp_path, preview=True)
    observed = []
    original = window.update_run_progress

    def record(value):
        observed.append(QThread.currentThread() == qt_application.thread())
        original(value)

    window.update_run_progress = record
    thread = QThread()
    emitter = ProgressEmitter()
    emitter.moveToThread(thread)
    emitter.progress.connect(window.run_progress_receiver.progress, Qt.ConnectionType.QueuedConnection)
    emitter.stage.connect(window.run_progress_receiver.stage, Qt.ConnectionType.QueuedConnection)
    emitter.done.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    emitter.done.connect(emitter.deleteLater)
    thread.started.connect(emitter.run)
    thread.start()
    try:
        for _ in range(100):
            QTest.qWait(10)
            if observed and not thread.isRunning():
                break
        assert observed == [True]
        assert window.run_overall_progress.value() == 50
        assert window.run_activity.value() == 30
        window.finish_run_progress({"state": "success"})
        assert window.run_activity.value() == 100
    finally:
        thread.quit()
        assert thread.wait(5000)


def test_stage_progress_and_long_sample_layout(window, tmp_path, qt_application):
    window.begin_run_progress(tmp_path)
    sample = "260212_4798_Hmg2-cherry_BFP-Ubc6_Ost1-neon_e_set2_001.ome"
    window.update_run_sample(sample)
    for stage, value in [(PipelineStage.LOADING, 10), (PipelineStage.MASKS, 30),
                         (PipelineStage.MEASUREMENT, 55), (PipelineStage.SEGMENTATION, 75), (PipelineStage.EXPORT, 90)]:
        window.update_run_stage(stage_update(stage, "Entirely different display wording"))
        assert window.run_activity.value() == value
    window.show()
    qt_application.processEvents()
    assert not window.run_sample_label.wordWrap()
    assert window.run_sample_label.width() >= window.run_sample_label.fontMetrics().horizontalAdvance(sample)
    assert window.run_activity.width() == window.run_overall_progress.width()
    assert window.run_stage_label.y() < window.run_activity.y()
    window.update_run_sample("next_sample.tif")
    assert window.run_activity.value() == 0


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)
    return CellonautMainWindow()


def test_run_progress_keeps_overall_progress_during_native_stage(window, tmp_path):
    window.begin_run_progress(tmp_path)
    window.update_run_progress(25)
    window.update_run_counter("2 / 4")
    window.update_run_sample("sample_002.tif")
    window.update_run_stage(stage_update(PipelineStage.INITIALIZING, "Initializing Fiji runtime"))
    window.refresh_run_elapsed()
    assert window.run_overall_progress.value() == 25
    assert window.run_stage_label.text() == "Initializing Fiji runtime"
    assert "2 / 4" in window.run_progress_title.text()
    assert window.run_sample_label.text() == "sample_002.tif"
    assert not window.run_progress_panel.isHidden()


def test_run_progress_shows_eta_and_clears_it_when_finished(monkeypatch, window, tmp_path):
    now = [100.0]
    monkeypatch.setattr(run_progress_gui.time, "monotonic", lambda: now[0])

    window.begin_run_progress(tmp_path)
    now[0] = 110.0
    window.update_run_progress(25)
    assert window.run_elapsed_label.text() == "10s elapsed · ETA 30s"

    now[0] = 120.0
    window.finish_run_progress({"state": "cancelled"})
    assert window.run_elapsed_label.text() == "20s elapsed"


@pytest.mark.parametrize("state", ["success", "partial", "cancelled", "error"])
def test_run_progress_terminal_state_and_restart(window, tmp_path, state):
    window.begin_run_progress(tmp_path)
    window.update_run_progress(25)
    window.finish_run_progress({"state": state, "error": "Example failure", "result": {
        "run_stats": {"total_samples": 4, "processed_targets": 2, "failed_targets": 1, "skipped_targets": 0}
    }})
    if state in {"success", "partial"}:
        for label, count in zip(window.run_target_counts, (2, 1, 0)):
            assert f'>{count}</span>' in label.text()
            assert not label.styleSheet()
        assert "&middot;" not in window.run_target_counts[0].text()
        assert all("&middot;" in label.text() for label in window.run_target_counts[1:])
    assert not window.run_elapsed_timer.isActive()
    assert not window.run_open_results.isHidden()
    assert window.run_overall_progress.value() == (100 if state in {"success", "partial"} else 25)
    window.update_run_stage(stage_update(PipelineStage.LOADING, "Loading images"))
    assert window.run_stage_label.text() != "Loading images"
    window.begin_run_progress(tmp_path)
    assert window.run_overall_progress.value() == 0
    assert window.run_elapsed_timer.isActive()
    assert window.run_open_results.isHidden()


def test_cancel_message_survives_queued_updates(window, tmp_path):
    window.begin_run_progress(tmp_path)
    window.cancel_run_progress()
    window.update_run_stage(stage_update(PipelineStage.SEGMENTATION, "Finding cells"))
    window.update_run_counter("3 / 4")
    assert "Cancelling" in window.run_progress_title.text()
    assert "Waiting" in window.run_stage_label.text()


def test_busy_stage_identity_survives_wording_changes(window):
    window.update_worker_status(stage_update(PipelineStage.INITIALIZING, "Starting native libraries"))
    started = window._worker_status_started_at
    assert window.progress_bar.maximum() == 0
    window.update_worker_status(stage_update(PipelineStage.INITIALIZING, "Different words"))
    assert window._worker_status_started_at == started
    assert window.progress_bar.maximum() == 0
    # A familiar long-stage label cannot turn an ordinary stage indeterminate.
    window.update_worker_status(stage_update(PipelineStage.LOADING, "Initializing Fiji runtime"))
    assert window.progress_bar.maximum() == 100
    assert not window._worker_status_heartbeat.isActive()
    window.set_status_style(stage_update(PipelineStage.RUNNING, "Error ready done"))
    assert window.status_label.property("status") == "running"
