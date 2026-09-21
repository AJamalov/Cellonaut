from __future__ import annotations

import json

from cellonaut.pipeline import operation_state


def test_terminal_state_preserves_operation_start_time(tmp_path, monkeypatch):
    timestamps = iter(("2026-01-01T10:00:00+00:00", "2026-01-01T10:05:00+00:00"))
    monkeypatch.setattr(operation_state, "_now_iso", lambda: next(timestamps))

    path = operation_state.save_operation_state(
        tmp_path,
        operation_type="full",
        state="in_progress",
        reset_started=True,
    )
    operation_state.finalize_existing_operation_state(
        tmp_path,
        operation_type="full",
        state="completed",
    )

    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["started_utc"] == "2026-01-01T10:00:00+00:00"
    assert state["updated_utc"] == "2026-01-01T10:05:00+00:00"


def test_new_attempt_resets_previous_start_time(tmp_path, monkeypatch):
    timestamps = iter(("2026-01-01T10:00:00+00:00", "2026-01-02T09:00:00+00:00"))
    monkeypatch.setattr(operation_state, "_now_iso", lambda: next(timestamps))

    path = operation_state.save_operation_state(
        tmp_path,
        operation_type="nd2_conversion",
        state="cancelled",
        reset_started=True,
    )
    operation_state.save_operation_state(
        tmp_path,
        operation_type="nd2_conversion",
        state="in_progress",
        reset_started=True,
    )

    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["started_utc"] == "2026-01-02T09:00:00+00:00"
