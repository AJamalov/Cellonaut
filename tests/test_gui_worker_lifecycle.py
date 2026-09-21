from __future__ import annotations

import time
from typing import Any, cast

import pytest
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal, Slot

from cellonaut.gui.nd2 import CellonautGuiNd2Mixin
from cellonaut.gui.state import GuiTaskState
from cellonaut.gui import workers as gui_workers
from cellonaut.gui import setup_workers
from cellonaut.gui.workers import CellonautGuiWorkersMixin


pytestmark = pytest.mark.gui


class LifecycleHarness(CellonautGuiNd2Mixin, CellonautGuiWorkersMixin):
    def __init__(self):
        self.task_state = GuiTaskState()
        self.worker: Any = None
        self.preview_worker: Any = None
        self.nd2_worker: Any = None
        self.nd2_worker_thread: Any = None


class SignalStub:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback, *_args):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class ImmediateThreadStub:
    def __init__(self, _parent=None):
        self.started = SignalStub()
        self.finished = SignalStub()
        self.cleanup_was_connected_before_start = False

    def start(self):
        self.cleanup_was_connected_before_start = len(self.finished.callbacks) >= 2

    def quit(self):
        pass

    def deleteLater(self):
        pass


class MinimalWorkerStub:
    def __init__(self):
        self.terminal_signal = SignalStub()
        self.thread = None

    def moveToThread(self, thread):
        self.thread = thread

    def run(self):
        pass

    def deleteLater(self):
        pass


class CompletingWorker(QObject):
    done_signal = Signal(dict)

    @Slot()
    def run(self):
        self.done_signal.emit({"status": "complete"})


class ResultBridge(QObject):
    result_signal = Signal(dict)


def test_primary_worker_reference_blocks_overlapping_task_start():
    harness = LifecycleHarness()

    assert harness.has_active_processing_task() is False

    harness.preview_worker = object()

    assert harness.has_active_processing_task() is True
    assert harness.active_processing_workers()[0][1:] == ("PREVIEW", "preview")


def test_worker_cleanup_is_connected_before_thread_start(monkeypatch):
    monkeypatch.setattr(gui_workers, "QThread", ImmediateThreadStub)
    harness = LifecycleHarness()
    worker = MinimalWorkerStub()
    cleanup_calls = []

    thread, returned_worker = harness.run_worker(worker, lambda: cleanup_calls.append("done"))

    assert returned_worker is worker
    assert thread.cleanup_was_connected_before_start is True
    thread.finished.emit()
    assert cleanup_calls == ["done"]


def test_explicit_terminal_signal_delivers_result_before_thread_finishes():
    app = QCoreApplication.instance() or QCoreApplication([])
    harness = LifecycleHarness()
    worker = CompletingWorker()
    bridge = ResultBridge()
    results = []
    finished = []
    events = []

    def record_result(result):
        results.append(result)
        events.append("result")

    def record_finished():
        finished.append(True)
        events.append("finished")

    bridge.result_signal.connect(record_result)

    thread, _worker = harness.prepare_worker_thread(
        worker,
        terminal_signal=worker.done_signal,
        result_callback=bridge.result_signal.emit,
        finished_callback=record_finished,
        thread_factory=lambda _parent: QThread(),
    )
    thread.start()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and (not results or not finished):
        app.processEvents()
        time.sleep(0.005)

    assert results == [{"status": "complete"}]
    assert finished == [True]
    assert events == ["result", "finished"]
    assert thread.isRunning() is False


def test_cancelled_fiji_worker_reports_no_failure(monkeypatch):
    def cancelled_scan(_path, *, cancel_requested):
        assert cancel_requested()
        raise InterruptedError("Fiji installation scan cancelled")

    monkeypatch.setattr(setup_workers, "scan_fiji_installation", cancelled_scan)
    worker = setup_workers.FijiInstallationScanWorker("Fiji.app")
    results = []
    worker.done_signal.connect(results.append)
    worker.request_cancel()

    worker.run()

    assert results == [{"path": "Fiji.app", "status": None, "scan_error": "", "scan_cancelled": True}]


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ({"state": "cancelled"}, ("cancelled", None)),
        ({"state": "error", "error": "disk full"}, ("error", "disk full")),
        ({"state": "success", "result": {"output_dir": "run-1"}}, ("done", False)),
        ({"state": "partial", "result": {"output_dir": "run-2"}}, ("done", True)),
    ],
)
def test_pipeline_terminal_outcomes_take_distinct_gui_paths(outcome, expected):
    harness = cast(Any, LifecycleHarness())
    calls = []
    harness.finish_run_progress = lambda value: calls.append(("finish", value))
    harness.on_pipeline_cancelled = lambda: calls.append(("cancelled", None))
    harness.on_pipeline_error = lambda msg: calls.append(("error", msg))
    harness.on_worker_done = lambda result=None, *, completed_with_errors=False: calls.append(
        ("done", completed_with_errors)
    )

    harness.on_pipeline_worker_finished(outcome)

    assert calls == [("finish", outcome), expected]


def test_cancel_current_task_requests_every_active_worker():
    class Worker:
        def __init__(self):
            self.cancel_requests = 0

        def request_cancel(self):
            self.cancel_requests += 1

    class TextTarget:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    class Button:
        def __init__(self):
            self.enabled = True

        def setEnabled(self, enabled):
            self.enabled = enabled

    harness = cast(Any, LifecycleHarness())
    pipeline_worker = Worker()
    preview_worker = Worker()
    harness.task_state.pipeline.worker = pipeline_worker
    harness.task_state.preview.worker = preview_worker
    harness.status_label = TextTarget()
    harness.current_sample_label = TextTarget()
    harness.sample_counter_label = TextTarget()
    harness.cancel_button = Button()
    harness.log_messages = []
    harness.log = harness.log_messages.append
    harness.stop_worker_status_heartbeat = lambda: None
    harness.cancel_run_progress = lambda: None
    harness.set_status_style = lambda _style: None

    harness.cancel_current_task()

    assert pipeline_worker.cancel_requests == 1
    assert preview_worker.cancel_requests == 1
    assert harness.status_label.text == "Stopping run..."
    assert harness.current_sample_label.text == "Stopping run..."
    assert harness.sample_counter_label.text == "Cancelling"
    assert harness.cancel_button.enabled is False
    assert len(harness.log_messages) == 2
