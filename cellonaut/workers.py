"""Qt worker and subprocess bridge for long-running Cellonaut tasks.

The GUI uses Qt threads for signal delivery, but Fiji, Weka, Cellpose, and ND2
conversion can block inside native libraries. These workers can isolate that
work in spawned child processes so Cancel remains responsive even when Python
callbacks are not reached.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import io
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any, cast

from PySide6.QtCore import QObject, Signal, Slot

from cellonaut.exceptions import SetupError, SetupErrorCode
from cellonaut.runtime import PipelineStage, StageUpdate, stage_update
from cellonaut.cell_segmentation.core import clear_cellpose_model_cache
from cellonaut.exceptions import PipelineCancelled
from cellonaut.io.nd2_import import (
    ND2ImportConfig,
    convert_nd2_folder,
)
from cellonaut.io.image_io import cleanup_staged_image_process_directory
from cellonaut.pipeline.models import Config
from cellonaut.pipeline.runner import run_pipeline, run_preview_pipeline
from cellonaut.pipeline.operation_state import finalize_existing_operation_state


# Pair application-owned error identifiers with stable user guidance so subprocess and
# in-process workers present the same recovery steps.
@dataclass(frozen=True)
class UserErrorHint:
    code: SetupErrorCode
    title: str
    summary: str
    next_steps: tuple[str, ...]

    # Presentation stays with the matched hint so every worker reports setup failures consistently.
    def render(self, technical_details: str) -> str:
        steps = "\n".join(f"- {step}" for step in self.next_steps)
        return (
            f"{self.title}\n\n"
            f"{self.summary}\n\n"
            f"Next steps:\n{steps}\n\n"
            f"Technical details:\n{technical_details}"
        )


ERROR_HINTS = [
    UserErrorHint(
        SetupErrorCode.ND2_INPUT, "No ND2 files were found.",
        "Choose an input folder that contains .nd2 files.",
        ("Nested subfolders are included when scanning for ND2 files.",),
    ),
    UserErrorHint(
        SetupErrorCode.ND2_CHANNELS, "ND2 channel settings are invalid.",
        "Check the channel mapping in Import ND2.",
        ("Empty TIFF channel names are skipped, and names must be unique.",),
    ),
    UserErrorHint(
        SetupErrorCode.ND2_OUTPUT, "The converted TIFF folder is unavailable.",
        "Choose a writable converted TIFF folder.",
        ("Check the destination folder and available permissions.",),
    ),
    UserErrorHint(
        SetupErrorCode.ND2_BACKEND, "The ND2 backend is unavailable.",
        "Install the ND2 backend in this environment.",
        ("Restart Cellonaut after restoring the ND2 backend.",),
    ),
    UserErrorHint(
        SetupErrorCode.OVERLAY_MASK,
        "Overlay setup is invalid.",
        "One of the selected overlay entries is not a valid mask.",
        (
            "Open Measurements and choose a configured mask column.",
            "Remove outdated mask selections from the measurement matrix.",
            "Run Check Setup again before starting the pipeline.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.WHOLE_CELL_OVERLAY,
        "Whole-cell overlay cannot be shown.",
        "Cell masks are disabled for this measured channel.",
        (
            "Enable Cellpose masks for the measured channel.",
            "Remove the whole-cell mask overlay from the measurement setup.",
            "Run Preview One Sample to confirm the overlay.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.CELL_SOURCE,
        "Cellpose source channel is invalid.",
        "The selected Cellpose source channel no longer exists in the current channel list.",
        (
            "Open Cellpose masks.",
            "Choose one of the configured channels as the cell mask source.",
            "Run Check Setup after changing channel names or presets.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.PER_CELL_MASK,
        "Per-cell mask source is invalid.",
        "The selected per-cell mask no longer matches a configured mask source.",
        (
            "Open Measurements and select a valid mask relationship.",
            "Check that imported, combined, or Weka mask sources are complete.",
            "Run Preview One Sample before running the full batch.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.MEASURED_CHANNEL,
        "Measured channel selection is invalid.",
        "A measurement target points to a channel that is not configured.",
        (
            "Open Channels and confirm the channel names.",
            "Rebuild the measurement matrix by selecting the Measurements section.",
            "Reload or resave the preset if the configuration is invalid.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.OVERLAY_BASE,
        "Overlay base channel is invalid.",
        "The overlay base channel no longer exists in the configured channel list.",
        (
            "Open Measurements and choose a valid overlay base channel.",
            "Confirm that auto-detected TIFF stack channel names are correct.",
            "Run Check Setup before previewing again.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.NO_SAMPLES,
        "No processable samples were found.",
        "Cellonaut could not find TIFF samples for the current input configuration.",
        (
            "Check that the input folder contains supported .tif or .tiff files.",
            "For raw ND2 files, use Import ND2 and then select the converted TIFF folder.",
            "Open Check Setup to inspect missing folders and files.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.CLASSIFIER_MISSING,
        "A classifier file could not be found.",
        "One configured Weka mask points to a missing classifier path.",
        (
            "Open Channels and reselect the missing classifier.",
            "If you intend to reuse existing masks, enable mask reuse and confirm the mask source folder.",
            "Run Check Setup to list all missing classifier paths.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.INPUT_MISSING,
        "The selected input directory does not exist.",
        "The input path is missing or not reachable from this computer.",
        (
            "Choose a valid input folder.",
            "If it is a network path, wait for the background scan or reconnect the share.",
            "Use the Files tab to confirm the folder can be opened.",
        ),
    ),
    UserErrorHint(
        SetupErrorCode.FIJI_MISSING,
        "The bundled Fiji runtime does not exist.",
        "This run needs Fiji/ImageJ, but the packaged runtime is missing.",
        (
            "Reinstall Cellonaut from the complete official offline package to restore its bundled Fiji runtime.",
            "For native TIFF/Cellpose-only runs, remove Weka mask steps if Fiji is not needed.",
            "Run Check Setup again after reinstalling Cellonaut.",
        ),
    ),
]


class SignalLogStream(io.TextIOBase):
    """Forward stdout/stderr line output from libraries into the GUI log."""

    _SUPPRESSED_LINES = {
        "Operating in headless mode - the original ImageJ will have limited functionality.",
    }

    # Native libraries often write fragments, so retain a buffer until a complete line is available.
    def __init__(self, emit_func, prefix: str = ""):
        super().__init__()
        self.emit_func = emit_func
        self.prefix = prefix
        self._buffer = ""

    # Emit complete lines immediately while preserving an unfinished final fragment for the next write.
    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line and line not in self._SUPPRESSED_LINES:
                self.emit_func(f"{self.prefix}{line}")
        return len(text)

    # Forward trailing output during context cleanup so error text without a newline is not lost.
    def flush(self) -> None:
        line = self._buffer.rstrip()
        self._buffer = ""
        if line and line not in self._SUPPRESSED_LINES:
            self.emit_func(f"{self.prefix}{line}")


# Native libraries often write to stdout/stderr instead of the logger, so
# child-process output is captured and forwarded to the GUI log.
@contextlib.contextmanager
def forward_terminal_output(log_func, prefix: str = ""):
    stdout_stream = SignalLogStream(log_func, prefix=prefix)
    stderr_stream = SignalLogStream(log_func, prefix=prefix)
    with contextlib.redirect_stdout(cast(Any, stdout_stream)), contextlib.redirect_stderr(cast(Any, stderr_stream)):
        try:
            yield
        finally:
            stdout_stream.flush()
            stderr_stream.flush()


# Raw exceptions are kept in the technical details, but the first message
# should tell non-programmers what to check next.
def format_user_error(exc: Exception) -> str:
    text = str(exc).strip()
    for hint in ERROR_HINTS:
        if isinstance(exc, SetupError) and hint.code == exc.code:
            return hint.render(text)

    if not text:
        return "An unexpected error occurred."

    return (
        "An unexpected error occurred.\n\n"
        "Next steps:\n"
        "- Review the Log tab for the last operation before the failure.\n"
        "- Run Check Setup to catch missing paths, classifiers, or samples.\n"
        "- If the problem repeats, keep the run summary and log files with the dataset.\n\n"
        f"Technical details:\n{text}"
    )


FORCE_CANCEL_GRACE_SECONDS = 1.0
TERMINAL_EXIT_GRACE_SECONDS = 2.0
QUEUE_POLL_SECONDS = 0.05


# Queue messages remain primitive so they are portable across spawned processes on every platform.
def _queue_emit(out_queue, kind: str, value: Any) -> None:
    if kind == "progress":
        value = int(value)
    elif kind not in {"terminal", "status"}:
        value = str(value)
    out_queue.put((kind, value))


# Return task outcomes with the same fields for every worker.
def _terminal_outcome(state: str, result: dict[str, Any] | None = None, error: str = "") -> dict[str, Any]:
    return {"state": state, "result": result or {}, "error": error}


# Use the same preview logic for in-process and subprocess workers.
def _execute_preview_task(
    cfg: Config,
    *,
    progress_emit: Callable[[int], None],
    sample_emit: Callable[[str], None],
    counter_emit: Callable[[str], None],
    log_emit: Callable[[str], None],
    status_emit: Callable[[StageUpdate], None],
    terminal_emit: Callable[[dict[str, Any]], object],
    should_cancel: Callable[[], bool],
) -> None:
    clear_cellpose_model_cache()
    try:
        status_emit(stage_update(PipelineStage.PREPARING, "Preparing preview"))
        progress_emit(0)
        counter_emit("0 / 0")

        with forward_terminal_output(log_emit):
            result = run_preview_pipeline(
                cfg,
                log_func=log_emit,
                stage_func=status_emit,
                progress_func=progress_emit,
                counter_func=counter_emit,
                sample_func=sample_emit,
                should_cancel=should_cancel,
            )

        completed_with_errors = bool(result.get("completed_with_errors"))
        if completed_with_errors:
            log_emit("[PREVIEW][WARN] Preview completed, but one or more measured channels failed.")
            status_emit(stage_update(PipelineStage.PARTIAL, "Preview ready with errors"))
            terminal_emit(_terminal_outcome("partial", result))
        else:
            status_emit(stage_update(PipelineStage.SUCCESS, "Preview ready"))
            terminal_emit(_terminal_outcome("success", result))
    except PipelineCancelled as exc:
        log_emit(f"[PREVIEW] {exc}")
        status_emit(stage_update(PipelineStage.CANCELLED, "Cancelled"))
        terminal_emit(_terminal_outcome("cancelled", error=str(exc)))
    except Exception as exc:
        log_emit(f"[PREVIEW][ERROR] {exc}")
        status_emit(stage_update(PipelineStage.ERROR, "Preview error"))
        terminal_emit(_terminal_outcome("error", error=format_user_error(exc)))
    finally:
        clear_cellpose_model_cache()


# Share full-run outcome handling so subprocess isolation does not create a second code path.
def _execute_pipeline_task(
    cfg: Config,
    *,
    progress_emit: Callable[[int], None],
    sample_emit: Callable[[str], None],
    counter_emit: Callable[[str], None],
    log_emit: Callable[[str], None],
    status_emit: Callable[[StageUpdate], None],
    terminal_emit: Callable[[dict[str, Any]], object],
    should_cancel: Callable[[], bool],
) -> None:
    clear_cellpose_model_cache()
    try:
        status_emit(stage_update(PipelineStage.PREPARING, "Preparing analysis"))
        progress_emit(0)
        sample_emit("Preparing run...")
        counter_emit("0 / 0")

        with forward_terminal_output(log_emit):
            result = run_pipeline(
                cfg,
                log_func=log_emit,
                stage_func=status_emit,
                progress_func=progress_emit,
                sample_func=sample_emit,
                counter_func=counter_emit,
                should_cancel=should_cancel,
            )

        state = "partial" if result.get("status") == "completed_with_errors" else "success"
        if state == "partial":
            stats = result.get("run_stats") or {}
            failed = int(stats.get("failed_targets", stats.get("failed", 0)))
            log_emit(f"[RUN][WARN] Pipeline completed with {failed} failed target(s).")
            status_emit(stage_update(PipelineStage.PARTIAL, "Completed with errors"))
        else:
            log_emit("[RUN] Pipeline finished.")
            status_emit(stage_update(PipelineStage.SUCCESS, "Done"))
        terminal_emit(_terminal_outcome(state, result))
    except PipelineCancelled as exc:
        log_emit(f"[RUN] {exc}")
        status_emit(stage_update(PipelineStage.CANCELLED, "Cancelled"))
        terminal_emit(_terminal_outcome("cancelled", error=str(exc)))
    except Exception as exc:
        log_emit(f"[RUN][ERROR] {exc}")
        status_emit(stage_update(PipelineStage.ERROR, "Error"))
        terminal_emit(_terminal_outcome("error", error=format_user_error(exc)))
    finally:
        clear_cellpose_model_cache()


# Use the same ND2 outcome handling for in-process and subprocess conversion.
def _execute_nd2_task(
    cfg: ND2ImportConfig,
    *,
    progress_emit: Callable[[int], None],
    current_file_emit: Callable[[str], None],
    log_emit: Callable[[str], None],
    status_emit: Callable[[StageUpdate], None],
    terminal_emit: Callable[[dict[str, Any]], object],
    should_cancel: Callable[[], bool],
) -> None:
    try:
        status_emit(stage_update(PipelineStage.RUNNING, "Converting ND2"))
        progress_emit(0)
        current_file_emit("Preparing ND2 conversion...")

        with forward_terminal_output(log_emit):
            summary = convert_nd2_folder(
                cfg,
                log_func=log_emit,
                progress_func=progress_emit,
                current_file_func=current_file_emit,
                should_cancel=should_cancel,
            )

        state = "partial" if int(summary.get("failed", 0)) > 0 else "success"
        status_emit(stage_update(
            PipelineStage.PARTIAL if state == "partial" else PipelineStage.SUCCESS,
            "ND2 conversion completed with errors" if state == "partial" else "ND2 conversion done",
        ))
        terminal_emit(_terminal_outcome(state, summary))
    except PipelineCancelled as exc:
        log_emit("[ND2] Conversion cancelled.")
        status_emit(stage_update(PipelineStage.CANCELLED, "Cancelled"))
        terminal_emit(_terminal_outcome("cancelled", error=str(exc)))
    except Exception as exc:
        log_emit(f"[ND2][ERROR] {exc}")
        status_emit(stage_update(PipelineStage.ERROR, "ND2 conversion error"))
        terminal_emit(_terminal_outcome("error", error=format_user_error(exc)))


# Send child-process updates through the queue and read cancellation from the shared event.
def _run_process_task(cfg, out_queue, cancel_event, task, *detail_signals: str) -> None:
    task(
        cfg,
        progress_emit=partial(_queue_emit, out_queue, "progress"),
        log_emit=partial(_queue_emit, out_queue, "log"),
        status_emit=partial(_queue_emit, out_queue, "status"),
        terminal_emit=partial(_queue_emit, out_queue, "terminal"),
        should_cancel=lambda: bool(cancel_event.is_set()),
        **{
            f"{signal}_emit": partial(_queue_emit, out_queue, signal)
            for signal in detail_signals
        },
    )


# Named process targets remain module-level so Windows spawn can pickle them.
def _run_preview_process(cfg: Config, out_queue, cancel_event) -> None:
    _run_process_task(cfg, out_queue, cancel_event, _execute_preview_task, "sample", "counter")


def _run_pipeline_process(cfg: Config, out_queue, cancel_event) -> None:
    _run_process_task(cfg, out_queue, cancel_event, _execute_pipeline_task, "sample", "counter")


def _run_nd2_process(cfg: ND2ImportConfig, out_queue, cancel_event) -> None:
    _run_process_task(cfg, out_queue, cancel_event, _execute_nd2_task, "current_file")


class ProcessBackedWorker(QObject):
    """Supervise native work from a Qt thread, normally in a spawned child process.

    The worker owns its child, queue and cancellation event; the GUI owns its
    QThread. Cancellation first requests cooperation, then force-stops a blocked
    child after a grace period. Only the first terminal outcome is delivered,
    even when completion races with cancellation. Logs do not drive task state.
    """

    def __init__(self, *, use_subprocess: bool | None = None):
        super().__init__()
        self._cancel_requested = False
        self._terminal_emitted = False
        self._cancel_event: Any = None
        self._use_subprocess = True if use_subprocess is None else bool(use_subprocess)

    # Set both flags because in-process work reads the local flag while child work reads the shared event.
    def request_cancel(self) -> None:
        self._cancel_requested = True
        cancel_event = getattr(self, "_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()

    def should_cancel(self) -> bool:
        return self._cancel_requested

    # Report only the first outcome because cancellation and completion can arrive together.
    def _emit_terminal_once(self, outcome: dict[str, Any]) -> bool:
        if self._terminal_emitted:
            return False
        self._terminal_emitted = True
        operation_type = str(getattr(self, "operation_type", "") or "")
        cfg = getattr(self, "cfg", None)
        output_dir = getattr(cfg, "output_dir", None)
        if operation_type and output_dir is not None:
            terminal_state = {
                "success": "completed",
                "partial": "completed_with_errors",
                "cancelled": "cancelled",
                "error": "failed",
            }.get(str(outcome.get("state", "")), "failed")
            try:
                finalize_existing_operation_state(
                    Path(output_dir),
                    operation_type=operation_type,
                    state=terminal_state,
                    error=str(outcome.get("error", "") or ""),
                )
            except Exception as exc:
                cast(Any, self).log_signal.emit(f"[WARN] Could not update operation state: {exc}")
        cast(Any, self).terminal_signal.emit(dict(outcome))
        return True

    # Handle queue messages consistently across worker types.
    def _emit_child_message(self, kind: str, value: Any) -> bool:
        if self._terminal_emitted:
            return kind == "terminal"
        if kind == "log":
            cast(Any, self).log_signal.emit(str(value))
        elif kind == "status":
            cast(Any, self).status_signal.emit(dict(value))
        elif kind == "progress":
            cast(Any, self).progress_signal.emit(int(value))
        elif kind == "sample" and hasattr(self, "sample_signal"):
            cast(Any, self).sample_signal.emit(str(value))
        elif kind == "counter" and hasattr(self, "counter_signal"):
            cast(Any, self).counter_signal.emit(str(value))
        elif kind == "current_file" and hasattr(self, "current_file_signal"):
            cast(Any, self).current_file_signal.emit(str(value))
        elif kind == "terminal":
            outcome = (
                value
                if isinstance(value, dict)
                else _terminal_outcome(
                    "error",
                    error=format_user_error(RuntimeError("Background worker returned an invalid terminal message.")),
                )
            )
            self._emit_terminal_once(outcome)
            return True
        return False

    # Read remaining messages after the process exits so buffered logs do not hide its outcome.
    def _drain_child_queue(self, out_queue, *, ignore_handler_errors: bool = False) -> bool:
        terminal_seen = False
        while True:
            try:
                kind, value = out_queue.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            try:
                terminal_seen = self._emit_child_message(str(kind), value) or terminal_seen
            except Exception:
                if not ignore_handler_errors:
                    raise
                break
        return terminal_seen

    # Try terminating the process, then kill it if it is still running after the wait.
    def _terminate_child_process(self, process) -> None:
        if not process.is_alive():
            process.join(timeout=0)
            return
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive():
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
                process.join(timeout=0.5)

    # Supervise the child from a Qt thread so queue polling never blocks the GUI event loop.
    def _run_in_child_process(self, target) -> None:
        ctx = mp.get_context("spawn")
        out_queue = ctx.Queue()
        cancel_event = ctx.Event()
        process = ctx.Process(target=target, args=(cast(Any, self).cfg, out_queue, cancel_event))
        self._cancel_event = cancel_event
        terminal_seen = False
        terminal_seen_at = None
        cancel_started_at = None
        process_started = False

        # Cooperative cancellation is tried first. If a native runtime is stuck,
        # the child process gives the GUI a hard stop that a Qt thread cannot.
        try:
            process.start()
            process_started = True
            while True:
                try:
                    kind, value = out_queue.get(timeout=QUEUE_POLL_SECONDS)
                    if self._emit_child_message(str(kind), value):
                        terminal_seen = True
                        terminal_seen_at = terminal_seen_at or time.monotonic()
                except queue.Empty:
                    pass

                if self._cancel_requested and not terminal_seen:
                    cancel_event.set()
                    if cancel_started_at is None:
                        cancel_started_at = time.monotonic()
                    elif time.monotonic() - cancel_started_at >= FORCE_CANCEL_GRACE_SECONDS:
                        self._terminate_child_process(process)
                        cast(Any, self).log_signal.emit(
                            "[STATUS] Background worker was force-stopped after cancellation."
                        )
                        cast(Any, self).status_signal.emit(stage_update(PipelineStage.CANCELLED, "Cancelled"))
                        self._emit_terminal_once(_terminal_outcome("cancelled", error="Cancelled by user."))
                        terminal_seen = True
                        break

                if terminal_seen:
                    process.join(timeout=0.2)
                    if not process.is_alive():
                        break
                    if (
                        terminal_seen_at is not None
                        and time.monotonic() - terminal_seen_at >= TERMINAL_EXIT_GRACE_SECONDS
                    ):
                        self._terminate_child_process(process)
                        break

                if not process.is_alive():
                    terminal_seen = self._drain_child_queue(out_queue) or terminal_seen
                    if not terminal_seen:
                        if self._cancel_requested:
                            self._emit_terminal_once(_terminal_outcome("cancelled", error="Cancelled by user."))
                        else:
                            code = process.exitcode
                            self._emit_terminal_once(
                                _terminal_outcome(
                                    "error",
                                    error=format_user_error(
                                        RuntimeError(f"Background worker exited unexpectedly with code {code}.")
                                    ),
                                )
                            )
                    break
        except Exception as exc:
            cast(Any, self).log_signal.emit(f"[WORKER][ERROR] {exc}")
            cast(Any, self).status_signal.emit(stage_update(PipelineStage.ERROR, "Error"))
            self._emit_terminal_once(_terminal_outcome("error", error=format_user_error(exc)))
        finally:
            child_pid = int(process.pid) if process_started and process.pid is not None else None
            if process_started:
                if process.is_alive():
                    self._terminate_child_process(process)
                else:
                    process.join(timeout=0)
            self._drain_child_queue(out_queue, ignore_handler_errors=True)
            try:
                out_queue.close()
            except Exception:
                pass
            if process_started:
                try:
                    process.close()
                except Exception:
                    pass
            if child_pid is not None:
                cleanup_staged_image_process_directory(
                    child_pid,
                    log_func=lambda message: cast(Any, self).log_signal.emit(message),
                )
            self._cancel_event = None


class ConfiguredProcessWorker(ProcessBackedWorker):
    """Worker that runs a task with the supplied configuration."""

    cfg: Any

    def __init__(self, cfg: Any, *, use_subprocess: bool | None = None):
        super().__init__(use_subprocess=use_subprocess)
        self.cfg = cfg

    # Keep subprocess selection and common Qt signal adapters identical for every task worker.
    def _run_configured_task(self, process_target, task, *detail_signals: str) -> None:
        if self._use_subprocess:
            self._run_in_child_process(process_target)
            return
        worker = cast(Any, self)
        task(
            self.cfg,
            progress_emit=worker.progress_signal.emit,
            log_emit=worker.log_signal.emit,
            status_emit=worker.status_signal.emit,
            terminal_emit=self._emit_terminal_once,
            should_cancel=self.should_cancel,
            **{
                f"{signal}_emit": getattr(worker, f"{signal}_signal").emit
                for signal in detail_signals
            },
        )


class PreviewWorker(ConfiguredProcessWorker):
    """Background worker that runs the first-sample pipeline preview."""

    log_signal = Signal(str)
    terminal_signal = Signal(dict)
    status_signal = Signal(dict)
    progress_signal = Signal(int)
    sample_signal = Signal(str)
    counter_signal = Signal(str)
    operation_type = "preview"

    cfg: Config

    @Slot()
    def run(self):
        self._run_configured_task(_run_preview_process, _execute_preview_task, "sample", "counter")


class PipelineWorker(ConfiguredProcessWorker):
    """Background worker that executes the main quantification pipeline."""

    log_signal = Signal(str)
    terminal_signal = Signal(dict)
    status_signal = Signal(dict)
    progress_signal = Signal(int)
    sample_signal = Signal(str)
    counter_signal = Signal(str)
    operation_type = "full"

    cfg: Config

    @Slot()
    def run(self):
        self._run_configured_task(_run_pipeline_process, _execute_pipeline_task, "sample", "counter")


class ND2ImportWorker(ConfiguredProcessWorker):
    """Background worker that converts ND2 files into the expected folder structure."""

    log_signal = Signal(str)
    terminal_signal = Signal(dict)
    status_signal = Signal(dict)
    progress_signal = Signal(int)
    current_file_signal = Signal(str)
    operation_type = "nd2_conversion"

    cfg: ND2ImportConfig

    # Keep an in-process path for tests and unsupported multiprocessing environments while matching outcomes.
    @Slot()
    def run(self):
        self._run_configured_task(_run_nd2_process, _execute_nd2_task, "current_file")
