"""Durable state markers for run, preview, and ND2 output folders."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from cellonaut.io.writers import write_text
from cellonaut.results.layout import build_results_layout
from cellonaut.version import __version__


# UTC timestamps remain comparable when runs move between workstations or are
# reviewed from a machine in another timezone.
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Pipeline state lives beside its logs, while converted TIFF folders keep their
# marker at the root because they do not use the Results layout.
def operation_state_path(output_dir: Path, operation_type: str) -> Path:
    if operation_type == "nd2_conversion":
        return Path(output_dir) / "ConversionState.json"
    return build_results_layout(Path(output_dir), create_root=False)["run_state_json"]


# Ignore unreadable previous state so the current run can still save its status.
def _read_existing_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# Publish the marker atomically because it is the authority for whether a
# partially populated output folder represents a completed operation.
def save_operation_state(
    output_dir: Path,
    *,
    operation_type: str,
    state: str,
    error: str = "",
    reset_started: bool = False,
) -> Path:
    path = operation_state_path(output_dir, operation_type)
    existing = _read_existing_state(path)
    now = _now_iso()
    payload = {
        "state_type": "Cellonaut Operation State",
        "schema_version": 1,
        "app_version": __version__,
        "operation_type": operation_type,
        "state": state,
        "started_utc": now if reset_started else existing.get("started_utc") or now,
        "updated_utc": now,
        "output_directory": str(Path(output_dir)),
        "error": str(error or ""),
    }
    write_text(path, json.dumps(payload, indent=2), encoding="utf-8")
    return path


# Only update a marker that was reserved at operation start. This keeps unit
# workers and direct library calls from creating state files in unrelated paths.
def finalize_existing_operation_state(
    output_dir: Path,
    *,
    operation_type: str,
    state: str,
    error: str = "",
) -> Path | None:
    path = operation_state_path(output_dir, operation_type)
    if not path.is_file():
        return None
    return save_operation_state(
        output_dir,
        operation_type=operation_type,
        state=state,
        error=error,
    )
