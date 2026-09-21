"""Compare completed Cellonaut runs and previews without depending on Qt widgets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import ntpath
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

import pandas as pd

from cellonaut.measurement.tables import human_readable_column_name


SUMMARY_CSV_NAME = "Measurements.csv"
INTERMEDIATE_SUMMARY_CSV_NAME = "Measurements_Detailed.csv"
RUN_MANIFEST_NAME = "RunSummary.json"


# Resolve the small set of files required for comparison once so later table
# logic cannot mix paths from neighboring run or preview folders.
@dataclass(frozen=True)
class ComparableRun:
    operation_dir: Path
    results_dir: Path
    measurements_path: Path
    manifest_path: Path
    label: str
    operation_type: str
    sequence_number: int


# Raw values travel with their directional change so UI coloring and CSV export
# are derived from the same comparison result.
@dataclass(frozen=True)
class MeasurementChange:
    sample: str
    measurement: str
    measurement_key: str
    baseline: float | None
    current: float | None
    difference: float | None
    percent_change: float | None
    status: str


# Settings can be added or removed as well as replaced, so preserve text values
# and an explicit status instead of forcing them into numeric change logic.
@dataclass(frozen=True)
class SettingChange:
    setting: str
    baseline: str
    current: str
    status: str


# Store comparison results for reuse when filtering the dialog.
@dataclass(frozen=True)
class RunComparison:
    baseline_run: ComparableRun
    current_run: ComparableRun
    measurement_changes: list[MeasurementChange]
    setting_changes: list[SettingChange]


# Users may start from a run folder, Results folder, or artifact selected in the
# file tree, so walk upward only until the first complete comparison pair appears.
def resolve_comparable_run(path: Path | str) -> ComparableRun:
    candidate = Path(path)
    if candidate.is_file():
        candidate = candidate.parent

    for folder in (candidate, *candidate.parents):
        results_dir = folder if folder.name.casefold() == "results" else folder / "Results"
        measurements = results_dir / "CSV Data" / SUMMARY_CSV_NAME
        if not measurements.is_file():
            measurements = results_dir / "CSV Data" / INTERMEDIATE_SUMMARY_CSV_NAME
        manifest = results_dir / "Logs" / RUN_MANIFEST_NAME
        if measurements.is_file() and manifest.is_file():
            operation_dir = results_dir.parent
            operation_match = re.fullmatch(
                r"(run|preview)_(\d+)",
                operation_dir.name,
                flags=re.IGNORECASE,
            )
            return ComparableRun(
                operation_dir=operation_dir,
                results_dir=results_dir,
                measurements_path=measurements,
                manifest_path=manifest,
                label=operation_dir.name or str(operation_dir),
                operation_type=operation_match.group(1).casefold() if operation_match else "other",
                sequence_number=int(operation_match.group(2)) if operation_match else -1,
            )
    raise ValueError(
        "The selected folder does not contain a completed Cellonaut run or preview with "
        f"{SUMMARY_CSV_NAME} and {RUN_MANIFEST_NAME}."
    )


# Group runs and previews separately, then sort by number so run_2 comes before run_10.
def _run_sort_key(run: ComparableRun) -> tuple[int, int, int, str]:
    type_order = 0 if run.operation_type == "run" else 1
    try:
        modified = run.manifest_path.stat().st_mtime_ns
    except OSError:
        modified = 0
    return type_order, run.sequence_number, modified, run.operation_dir.name.casefold()


# Find comparisons in the selected folder and numbered run or preview folders
# that contain both required files.
def discover_comparable_runs(path: Path | str) -> list[ComparableRun]:
    selected = Path(path)
    if selected.is_file():
        selected = selected.parent

    try:
        selected_run = resolve_comparable_run(selected)
        root = selected_run.operation_dir.parent
    except ValueError:
        root = selected

    runs: list[ComparableRun] = []
    candidates = [root]
    try:
        candidates.extend(
            child
            for child in root.iterdir()
            if child.is_dir() and re.fullmatch(r"(?:run|preview)_\d+", child.name, flags=re.IGNORECASE)
        )
    except OSError:
        return []

    seen: set[str] = set()
    for candidate in candidates:
        try:
            run = resolve_comparable_run(candidate)
        except ValueError:
            continue
        key = str(run.operation_dir.resolve()).casefold()
        if key in seen or run.operation_type not in {"run", "preview"}:
            continue
        seen.add(key)
        runs.append(run)
    return sorted(runs, key=_run_sort_key)


# Fail with the run-summary path in the message because a malformed manifest
# should not be mistaken for a missing measurement export.
def _load_manifest(run: ComparableRun) -> dict[str, Any]:
    try:
        value = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read run summary: {run.manifest_path}\n{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Run summary is not a JSON object: {run.manifest_path}")
    return value


# Relative identifiers allow the same dataset to be compared after its parent
# folder moved, while retaining nested sample folders that may share a name.
def _sample_identifier(value: Any, input_directory: Any) -> str:
    text = str(value or "").strip()
    root_text = str(input_directory or "").strip()
    if not text:
        return "(unnamed sample)"
    if text.casefold() == "average" or not root_text:
        return text

    windows_style = bool(re.match(r"^[A-Za-z]:[\\/]", text)) or "\\" in text
    try:
        if windows_style:
            windows_sample_path = PureWindowsPath(text)
            windows_root_path = PureWindowsPath(root_text)
            relative = ntpath.relpath(str(windows_sample_path), str(windows_root_path))
            if relative != ".." and not relative.startswith(f"..{ntpath.sep}"):
                return str(PureWindowsPath(relative))
        else:
            posix_sample_path = PurePosixPath(text)
            posix_root_path = PurePosixPath(root_text)
            return str(posix_sample_path.relative_to(posix_root_path))
    except (ValueError, OSError):
        pass
    return text


# Wide exports contain sparse target-specific columns. Collapse only identical
# sample/measurement pairs and reject conflicting duplicates instead of averaging them.
def _measurement_values(run: ComparableRun, manifest: dict[str, Any]) -> dict[tuple[str, str], float]:
    try:
        table = pd.read_csv(run.measurements_path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read measurement table: {run.measurements_path}\n{exc}") from exc
    if table.empty:
        return {}

    sample_column = "Sample" if "Sample" in table.columns else "Label" if "Label" in table.columns else ""
    if not sample_column:
        raise ValueError(f"Measurement table has no Sample column: {run.measurements_path}")

    input_directory = manifest.get("input_directory", "")
    values: dict[tuple[str, str], float] = {}
    for column in table.columns:
        if str(column) == sample_column:
            continue
        numeric: Any = pd.to_numeric(pd.Series(table[column], index=table.index), errors="coerce")
        for index, value in numeric.items():
            if pd.isna(value):
                continue
            sample = _sample_identifier(table.at[index, sample_column], input_directory)
            key = (sample, str(column))
            number = float(value)
            previous = values.get(key)
            if previous is not None and not math.isclose(previous, number, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    "The measurement table contains conflicting duplicate values for " f"{sample} / {column}."
                )
            values[key] = number
    return values


# Preserve added and removed measurements instead of replacing absence with
# zero, which would invent a numerical change that the exports do not contain.
def _measurement_change(
    key: tuple[str, str],
    baseline_values: dict[tuple[str, str], float],
    current_values: dict[tuple[str, str], float],
) -> MeasurementChange:
    sample, measurement_key = key
    baseline = baseline_values.get(key)
    current = current_values.get(key)
    difference: float | None = None
    percent_change: float | None = None

    if baseline is None:
        status = "Added"
    elif current is None:
        status = "Removed"
    else:
        difference = current - baseline
        if baseline != 0:
            percent_change = difference / abs(baseline) * 100.0
        status = "Unchanged" if math.isclose(baseline, current, rel_tol=1e-9, abs_tol=1e-12) else "Changed"

    return MeasurementChange(
        sample=sample,
        measurement=human_readable_column_name(measurement_key),
        measurement_key=measurement_key,
        baseline=baseline,
        current=current,
        difference=difference,
        percent_change=percent_change,
        status=status,
    )


# Volatile output paths differ by design; all settings that can influence
# processing remain in the comparison, including input paths and mask sources.
def _manifest_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    config = dict(manifest.get("configuration_snapshot", {}) or {})
    config.pop("output_dir", None)
    runtime = dict(manifest.get("runtime_environment", {}) or {})
    fingerprints: dict[str, Any] = {}
    for entry in runtime.get("configured_file_fingerprints", []) or []:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role", "") or "(unnamed file)")
        fingerprints[role] = {
            "path": entry.get("path", ""),
            "sha256": entry.get("sha256", ""),
            "exists": entry.get("exists", False),
        }
    return {
        "Cellonaut version": manifest.get("app_version", ""),
        "Configuration": config,
        "Effective measurement targets": manifest.get("measurement_targets", []),
        "Dependency versions": runtime.get("dependency_versions", {}),
        "Configured files": fingerprints,
    }


# Primitive lists stay together because a relationship change is easier to read
# as one before/after value than as several shifted numeric indexes.
def _flatten_settings(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item).casefold()):
            child = f"{prefix} > {key}" if prefix else str(key)
            flattened.update(_flatten_settings(value[key], child))
        return flattened
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        flattened = {}
        for index, item in enumerate(value, start=1):
            child = f"{prefix} [{index}]"
            flattened.update(_flatten_settings(item, child))
        return flattened
    return {prefix: value}


# Distinguish missing, blank, and false values because each represents a
# different configuration state when troubleshooting changed results.
def _display_setting(value: Any) -> str:
    if value is None:
        return "(not set)"
    if value == "":
        return "(blank)"
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


# Omit equal settings so users can focus on plausible causes of result changes.
def _setting_changes(baseline_manifest: dict[str, Any], current_manifest: dict[str, Any]) -> list[SettingChange]:
    baseline = _flatten_settings(_manifest_settings(baseline_manifest))
    current = _flatten_settings(_manifest_settings(current_manifest))
    changes: list[SettingChange] = []
    for setting in sorted(set(baseline) | set(current), key=str.casefold):
        baseline_value = baseline.get(setting)
        current_value = current.get(setting)
        if baseline_value == current_value:
            continue
        if setting not in baseline:
            status = "Added"
        elif setting not in current:
            status = "Removed"
        else:
            status = "Changed"
        changes.append(
            SettingChange(
                setting=setting,
                baseline=_display_setting(baseline_value) if setting in baseline else "(missing)",
                current=_display_setting(current_value) if setting in current else "(missing)",
                status=status,
            )
        )
    return changes


# Load each export once and return comparison data without GUI objects.
def compare_runs(baseline_path: Path | str, current_path: Path | str) -> RunComparison:
    baseline_run = resolve_comparable_run(baseline_path)
    current_run = resolve_comparable_run(current_path)
    if baseline_run.operation_dir.resolve() == current_run.operation_dir.resolve():
        raise ValueError("Choose two different runs to compare.")

    baseline_manifest = _load_manifest(baseline_run)
    current_manifest = _load_manifest(current_run)
    baseline_values = _measurement_values(baseline_run, baseline_manifest)
    current_values = _measurement_values(current_run, current_manifest)
    measurement_changes = [
        _measurement_change(key, baseline_values, current_values)
        for key in sorted(set(baseline_values) | set(current_values), key=lambda item: (item[0].casefold(), item[1]))
    ]
    return RunComparison(
        baseline_run=baseline_run,
        current_run=current_run,
        measurement_changes=measurement_changes,
        setting_changes=_setting_changes(baseline_manifest, current_manifest),
    )
