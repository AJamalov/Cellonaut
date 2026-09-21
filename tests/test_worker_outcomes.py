from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from cellonaut import workers
from cellonaut.exceptions import PipelineCancelled, SetupFileNotFoundError, SetupErrorCode
from cellonaut.runtime import PipelineStage, stage_update
from cellonaut.io.nd2_import import ND2ImportConfig
from cellonaut.pipeline.models import Config
from cellonaut.pipeline.operation_state import save_operation_state


def make_config() -> Config:
    return Config(
        fiji_app_path=Path(),
        input_dir=Path(),
        output_dir=Path(),
        input_structure="Flat TIFF files",
        images=[],
        exclusion_tag="",
        threshold_method="Default",
        probability_class_index=1,
    )


def make_nd2_config(tmp_path: Path) -> ND2ImportConfig:
    return ND2ImportConfig(
        source_dir=tmp_path / "nd2",
        output_dir=tmp_path / "tiff",
        channel_map={"Channel 1": 0},
    )


def blocking_child_worker(_cfg, out_queue, _cancel_event) -> None:
    out_queue.put(("status", stage_update(PipelineStage.RUNNING, "blocked child running")))
    time.sleep(30)


def collect_terminal_outcomes(worker) -> list[dict]:
    outcomes: list[dict] = []
    worker.terminal_signal.connect(outcomes.append)
    worker.run()
    return outcomes


def test_pipeline_worker_emits_one_partial_terminal_outcome(monkeypatch):
    monkeypatch.setattr(
        workers,
        "run_pipeline",
        lambda *_args, **_kwargs: {
            "status": "completed_with_errors",
            "run_stats": {"failed": 2},
        },
    )
    worker = workers.PipelineWorker(make_config(), use_subprocess=False)

    outcomes = collect_terminal_outcomes(worker)

    assert outcomes == [
        {
            "state": "partial",
            "result": {
                "status": "completed_with_errors",
                "run_stats": {"failed": 2},
            },
            "error": "",
        }
    ]


def test_pipeline_worker_emits_one_cancelled_terminal_outcome(monkeypatch):
    monkeypatch.setattr(
        workers,
        "run_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PipelineCancelled("cancelled")),
    )
    worker = workers.PipelineWorker(make_config(), use_subprocess=False)

    outcomes = collect_terminal_outcomes(worker)

    assert len(outcomes) == 1
    assert outcomes[0]["state"] == "cancelled"


def test_pipeline_worker_emits_one_error_terminal_outcome(monkeypatch):
    monkeypatch.setattr(
        workers,
        "run_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    worker = workers.PipelineWorker(make_config(), use_subprocess=False)

    outcomes = collect_terminal_outcomes(worker)

    assert len(outcomes) == 1
    assert outcomes[0]["state"] == "error"
    assert "Next steps:" in outcomes[0]["error"]
    assert "Technical details:\ndisk full" in outcomes[0]["error"]


def test_format_user_error_adds_actionable_setup_guidance():
    message = workers.format_user_error(SetupFileNotFoundError(SetupErrorCode.CLASSIFIER_MISSING, "Missing model at a new location"))

    assert "A classifier file could not be found." in message
    assert "Next steps:" in message
    assert "Run Check Setup" in message
    assert "Technical details:" in message


def test_preview_worker_uses_running_preview_wording(monkeypatch):
    monkeypatch.setattr(
        workers,
        "run_preview_pipeline",
        lambda *_args, **_kwargs: {"status": "PROCESSED"},
    )
    worker = workers.PreviewWorker(make_config(), use_subprocess=False)
    statuses: list[dict] = []
    worker.status_signal.connect(statuses.append)

    outcomes = collect_terminal_outcomes(worker)

    assert outcomes[0]["state"] == "success"
    assert statuses[0] == stage_update(PipelineStage.PREPARING, "Preparing preview")
    assert all("full preview" not in status["text"].lower() for status in statuses)


def test_preview_worker_reports_partial_result_when_one_target_failed(monkeypatch):
    monkeypatch.setattr(
        workers,
        "run_preview_pipeline",
        lambda *_args, **_kwargs: {
            "status": "PROCESSED",
            "completed_with_errors": True,
        },
    )
    worker = workers.PreviewWorker(make_config(), use_subprocess=False)
    statuses: list[dict] = []
    logs: list[str] = []
    worker.status_signal.connect(statuses.append)
    worker.log_signal.connect(logs.append)

    outcomes = collect_terminal_outcomes(worker)

    assert outcomes[0]["state"] == "partial"
    assert statuses[-1] == stage_update(PipelineStage.PARTIAL, "Preview ready with errors")
    assert any("one or more measured channels failed" in message for message in logs)


def test_nd2_worker_emits_partial_terminal_outcome(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        workers,
        "convert_nd2_folder",
        lambda *_args, **_kwargs: {"total": 3, "processed": 2, "failed": 1},
    )
    worker = workers.ND2ImportWorker(make_nd2_config(tmp_path), use_subprocess=False)

    outcomes = collect_terminal_outcomes(worker)

    assert outcomes == [
        {
            "state": "partial",
            "result": {"total": 3, "processed": 2, "failed": 1},
            "error": "",
        }
    ]


def test_nd2_worker_emits_cancelled_terminal_outcome(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        workers,
        "convert_nd2_folder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PipelineCancelled("cancelled")),
    )
    worker = workers.ND2ImportWorker(make_nd2_config(tmp_path), use_subprocess=False)

    outcomes = collect_terminal_outcomes(worker)

    assert len(outcomes) == 1
    assert outcomes[0]["state"] == "cancelled"
    assert outcomes[0]["error"] == "cancelled"


def test_nd2_worker_emits_error_terminal_outcome(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        workers,
        "convert_nd2_folder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("missing Bio-Formats")),
    )
    worker = workers.ND2ImportWorker(make_nd2_config(tmp_path), use_subprocess=False)

    outcomes = collect_terminal_outcomes(worker)

    assert len(outcomes) == 1
    assert outcomes[0]["state"] == "error"
    assert "Next steps:" in outcomes[0]["error"]
    assert "Technical details:\nmissing Bio-Formats" in outcomes[0]["error"]


def test_process_worker_force_stops_blocked_child(monkeypatch):
    monkeypatch.setattr(workers, "FORCE_CANCEL_GRACE_SECONDS", 0.05)
    cleaned_processes = []
    monkeypatch.setattr(
        workers,
        "cleanup_staged_image_process_directory",
        lambda process_id, log_func=None: cleaned_processes.append(process_id),
    )
    worker = workers.PipelineWorker(make_config(), use_subprocess=True)
    outcomes: list[dict] = []
    logs: list[str] = []

    def cancel_when_child_starts(_status: str) -> None:
        worker.request_cancel()

    worker.status_signal.connect(cancel_when_child_starts)
    worker.log_signal.connect(logs.append)
    worker.terminal_signal.connect(outcomes.append)

    worker._run_in_child_process(blocking_child_worker)

    assert outcomes == [{"state": "cancelled", "result": {}, "error": "Cancelled by user."}]
    assert any("force-stopped" in log for log in logs)
    assert len(cleaned_processes) == 1
    assert cleaned_processes[0] > 0


def test_process_worker_emits_only_the_first_terminal_outcome():
    worker = workers.PipelineWorker(make_config(), use_subprocess=True)
    outcomes: list[dict] = []
    worker.terminal_signal.connect(outcomes.append)

    worker._emit_child_message("terminal", {"state": "cancelled", "result": {}, "error": "cancelled"})
    worker._emit_child_message("terminal", {"state": "success", "result": {}, "error": ""})

    assert outcomes == [{"state": "cancelled", "result": {}, "error": "cancelled"}]


@pytest.mark.parametrize(
    ("terminal_state", "saved_state"),
    [
        ("success", "completed"),
        ("partial", "completed_with_errors"),
        ("cancelled", "cancelled"),
        ("error", "failed"),
    ],
)
def test_pipeline_worker_records_terminal_state(tmp_path: Path, terminal_state: str, saved_state: str):
    cfg = make_config()
    cfg.output_dir = tmp_path / terminal_state
    save_operation_state(cfg.output_dir, operation_type="full", state="in_progress")
    worker = workers.PipelineWorker(cfg, use_subprocess=False)

    worker._emit_terminal_once(
        {"state": terminal_state, "result": {}, "error": "stopped" if terminal_state == "error" else ""}
    )

    state_path = cfg.output_dir / "Results" / "Logs" / "RunState.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["state"] == saved_state
    assert state["started_utc"] <= state["updated_utc"]
    assert state["error"] == ("stopped" if terminal_state == "error" else "")


def test_process_worker_cleans_up_child_after_bridge_error(monkeypatch):
    worker = workers.PipelineWorker(make_config(), use_subprocess=True)
    outcomes: list[dict] = []
    worker.terminal_signal.connect(outcomes.append)

    def reject_child_message(_kind: str, _value: object) -> bool:
        raise RuntimeError("queue bridge failed")

    monkeypatch.setattr(worker, "_emit_child_message", reject_child_message)

    worker._run_in_child_process(blocking_child_worker)

    assert len(outcomes) == 1
    assert outcomes[0]["state"] == "error"
    assert "queue bridge failed" in outcomes[0]["error"]
    assert worker._cancel_event is None
