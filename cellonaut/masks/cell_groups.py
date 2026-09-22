"""Qt-independent Cell Group evaluation for preview and derived exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from cellonaut.masks.cell_qc import (
    cell_label_series, cell_qc_rules_to_text, combine_qc_label_sets,
    evaluate_cell_qc_table, normalize_cell_qc_rules,
)


@dataclass
class CellGroupResult:
    index: int
    name: str
    color: str
    exclude_from_csv: bool
    labels: frozenset[int] = field(default_factory=frozenset)
    has_filters: bool = False
    missing_metrics: tuple[str, ...] = ()
    error: str = ""


@dataclass
class CellGroupsResult:
    groups: list[CellGroupResult]
    cell_labels: frozenset[int]
    excluded_labels: frozenset[int]

    @property
    def kept_labels(self) -> frozenset[int]:
        return self.cell_labels - self.excluded_labels

    def require_available(self) -> None:
        """Export requires a complete evaluation; Preview may show known groups."""
        for group in self.groups:
            if group.error:
                raise ValueError(f"could not evaluate {group.name} ({group.error})")
            if group.missing_metrics:
                raise ValueError("Cell-group metrics unavailable: " + ", ".join(group.missing_metrics))


def _selected_mask_label(image_def: dict[str, Any]) -> str:
    relationships = dict(image_def.get("mask_relationships", {}) or {})
    selected = str(image_def.get("cell_group_mask_source", "") or "").strip()
    if selected and relationships.get(selected):
        return selected
    return next((str(label) for label, selected in relationships.items() if selected), "")


def _preparation_key(settings: dict[str, Any]) -> tuple[str, str, str]:
    mask = _selected_mask_label(settings)
    source = "Cell mask image" if settings.get("mask_qc_intensity_source") == "Cell mask image" else "Measured image"
    channel = str(settings.get("name", "") or "")
    if source == "Cell mask image":
        channel = str(
            settings.get("analysis_cellpose_mask_source", "")
            or settings.get("analysis_cell_segmentation_source", "")
            or channel
        )
    return mask, source, channel


def evaluate_cell_groups(
    table: pd.DataFrame,
    image_def: dict[str, Any],
    parse_rules: Callable[[str], dict],
    *,
    prepare_table: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame] | None = None,
) -> CellGroupsResult:
    """Evaluate all groups without changing the original table or GUI state.

    A group's members satisfy its configured conditions. Unconfigured groups
    are empty, not all cells. CSV exclusions are the union of members of groups
    marked exclude_from_csv; overlapping groups do not duplicate exclusions.

    Numeric NaNs fail active conditions. Absent columns are unavailable, not
    failures or matches: the affected group has no members and lists its missing
    metrics. Preparation/parser errors are recorded separately. Preview can
    render available memberships; exporters must call require_available().

    Preparation receives the union of required mask metrics once for each
    selected mask/intensity source. It must return a new table when adding
    metrics, preserving saved measured-channel values. All reuse is local to
    this call; a subsequent call observes changed masks, pixels and settings.
    """
    all_labels = frozenset(int(value) for value in cell_label_series(table) if int(value) > 0)
    groups: list[CellGroupResult] = []
    requests: list[tuple[dict, dict, dict]] = []
    preparations: dict[tuple[str, str, str], tuple[dict, dict]] = {}
    for population in image_def.get("cell_populations", []) or []:
        if not isinstance(population, dict):
            continue
        index = len(groups)
        group = CellGroupResult(
            index, str(population.get("name", f"Cell group {index + 1}") or f"Cell group {index + 1}"),
            str(population.get("color", "#00d7ff") or "#00d7ff"), bool(population.get("exclude_from_csv", False)),
        )
        groups.append(group)
        settings = {**image_def, **population, "name": image_def.get("name", "")}
        cell_rules, mask_rules = {}, {}
        try:
            cell_rules = normalize_cell_qc_rules(parse_rules(str(settings.get("cell_qc_limits") or "")))
            mask_rules = normalize_cell_qc_rules(parse_rules(str(settings.get("mask_qc_limits") or "")))
            group.has_filters = bool(cell_rules or mask_rules)
            if mask_rules and prepare_table is not None and all_labels:
                _, required = preparations.setdefault(_preparation_key(settings), (settings, {}))
                required.update(mask_rules)
        except Exception as exc:
            group.error = f"{type(exc).__name__}: {exc}"
        requests.append((settings, cell_rules, mask_rules))

    prepared: dict[tuple[str, str, str], pd.DataFrame] = {}
    preparation_errors: dict[tuple[str, str, str], str] = {}
    for key, (settings, required) in preparations.items():
        assert prepare_table is not None
        try:
            prepared[key] = prepare_table(table.copy(), {**settings, "mask_qc_limits": cell_qc_rules_to_text(required)})
        except Exception as exc:
            preparation_errors[key] = f"{type(exc).__name__}: {exc}"

    excluded: set[int] = set()
    for group, (settings, cell_rules, mask_rules) in zip(groups, requests):
        if group.error:
            continue
        key = _preparation_key(settings)
        if mask_rules and key in preparation_errors:
            group.error = preparation_errors[key]
            continue
        current = prepared.get(key, table) if mask_rules else table
        try:
            failed, summary = _evaluate_rules(current, settings, cell_rules, mask_rules)
            group.missing_metrics = tuple(summary["cell_missing"] + summary["mask_missing"])
            if group.has_filters and not group.missing_metrics:
                group.labels = all_labels - failed
                if group.exclude_from_csv:
                    excluded.update(group.labels)
        except Exception as exc:
            group.error = f"{type(exc).__name__}: {exc}"
    return CellGroupsResult(groups, all_labels, frozenset(excluded))


def excluded_labels_from_current_rules(
    table: pd.DataFrame,
    image_def: dict[str, Any],
    parse_rules,
) -> tuple[set[int], dict[str, Any]]:
    """Evaluate legacy direct-rule exports, returning failed IDs and diagnostics.

    These are condition failures, unlike the passing members excluded by a
    current Cell Group. Keep this entry point for callers without cell_populations.
    """
    cell_rules = normalize_cell_qc_rules(parse_rules(str(image_def.get("cell_qc_limits", "") or "")))
    mask_rules = normalize_cell_qc_rules(parse_rules(str(image_def.get("mask_qc_limits", "") or "")))
    return _evaluate_rules(table, image_def, cell_rules, mask_rules)


def _evaluate_rules(table, image_def, cell_rules, mask_rules):
    mode = str(image_def.get("qc_filter_mode", "") or "Exclude if any filter fails")
    desired_mask_source = str(image_def.get("mask_qc_intensity_source", "Measured image") or "Measured image")

    cell_table, cell_labels, cell_summary = evaluate_cell_qc_table(table, cell_rules)
    mask_table, mask_labels, mask_summary = evaluate_cell_qc_table(table, mask_rules)
    # An unconfigured category cannot satisfy an either-category match.
    # Use rule presence, not failed-label counts: an active category may pass every cell.
    if mode == "Exclude only if both fail" and not (cell_rules and mask_rules):
        excluded = set(cell_labels if cell_rules else mask_labels)
    else:
        excluded = combine_qc_label_sets(cell_labels, mask_labels, mode)

    labels = cell_label_series(table)
    total_count = int((labels > 0).sum())
    return excluded, {
        "total": total_count,
        "cell_flagged": int(len(cell_labels)),
        "mask_flagged": int(len(mask_labels)),
        "excluded": int(len(excluded)),
        "cell_missing": list(cell_summary.get("missing_metrics", []) or []),
        "mask_missing": list(mask_summary.get("missing_metrics", []) or []),
        "has_filters": bool(cell_rules or mask_rules),
        "exclude_enabled": bool(image_def.get("cell_qc_exclude_flagged", False)),
        "mode": mode,
        "mask_intensity_source": desired_mask_source,
        "table_mask_intensity_source": (
            str(table["MaskQC_IntensitySource"].dropna().iloc[0])
            if "MaskQC_IntensitySource" in table.columns and not table["MaskQC_IntensitySource"].dropna().empty
            else ""
        ),
        "cell_table": cell_table,
        "mask_table": mask_table,
    }


