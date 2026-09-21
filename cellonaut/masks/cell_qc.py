"""Numeric condition evaluation and mask metrics used by Cell Groups.

Filters are stored as simple metric limits, but result tables may use Fiji,
Cellpose, or Cellonaut column names. These helpers report failed conditions;
cell_groups decides membership and export exclusion without changing measurements.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from cellonaut.config.defaults import normalize_qc_filter_mode


# Editable metric lists live beside their column mappings so the GUI cannot
# offer a filter that the evaluation engine does not understand.
CELL_FILTER_METRICS = (
    "Area",
    "Mean intensity",
    "Minimum intensity",
    "Maximum intensity",
    "Raw integrated density",
    "Integrated density",
    "Perimeter",
    "Circularity",
    "Solidity",
)
MASK_FILTER_METRICS = (
    "Mask area",
    "Mask fraction of cell area",
    "Mean intensity inside mask",
    "Mask integrated density",
    "Mask integrated density / cell area",
    "Fraction of cell intensity in mask",
)

_FILTER_COLUMN_GROUPS = {
    "area": ("Area", "CellArea"),
    "meanintensity": ("Mean", "MeanGrayValue", "CellMean"),
    "minimumintensity": ("Min", "Minimum"),
    "maximumintensity": ("Max", "Maximum"),
    "rawintegrateddensity": ("RawIntDen",),
    "integrateddensity": ("IntDen",),
    "perimeter": ("Perimeter",),
    "circularity": ("Circularity", "Circ."),
    "solidity": ("Solidity",),
    "maskarea": ("PositiveAreaInCell",),
    "maskfractionofcellarea": ("PositiveAreaFractionInCell",),
    "meanintensityinsidemask": ("MeanInPositiveArea", "MeanGrayValueInPositiveArea"),
    "maskintegrateddensity": ("RawIntDenInCell",),
    "maskintegrateddensitycellarea": ("RawIntDenPerCellArea",),
    "fractionofcellintensityinmask": ("FractionOfCellIntDen",),
}
_FILTER_METRIC_KEYS = {
    "mean": "meanintensity",
    "meangrayvalue": "meanintensity",
    "min": "minimumintensity",
    "minimum": "minimumintensity",
    "max": "maximumintensity",
    "maximum": "maximumintensity",
    "rawintden": "rawintegrateddensity",
    "intden": "integrateddensity",
    "positiveareaincell": "maskarea",
    "positiveareafractionincell": "maskfractionofcellarea",
    "meaninpositivearea": "meanintensityinsidemask",
    "meangrayvalueinpositivearea": "meanintensityinsidemask",
    "rawintdenincell": "maskintegrateddensity",
    "rawintdenpercellarea": "maskintegrateddensitycellarea",
    "fractionofcellintden": "fractionofcellintensityinmask",
}


# Normalize rules again at execution time because callers can construct Config
# objects directly instead of passing through the stricter GUI adapter.
def normalize_cell_qc_rules(raw_rules: Any) -> dict[str, dict[str, Optional[float]]]:
    if not isinstance(raw_rules, dict):
        return {}

    normalized: dict[str, dict[str, Optional[float]]] = {}
    for metric, limits in raw_rules.items():
        metric_name = str(metric or "").strip()
        if not metric_name or not isinstance(limits, dict):
            continue

        rule: dict[str, Optional[float]] = {}
        for bound in ("min", "max"):
            value = limits.get(bound, None)
            if value in (None, ""):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(parsed):
                rule[bound] = parsed

        if rule:
            normalized[metric_name] = rule

    return normalized


# Persisted filter summaries stay compact while using the same normalization as
# execution so invalid or non-finite bounds are never displayed as active rules.
def cell_qc_rules_to_text(rules: dict) -> str:
    normalized_rules = normalize_cell_qc_rules(rules)
    parts = []
    for metric, limits in normalized_rules.items():
        min_value = limits.get("min")
        max_value = limits.get("max")
        min_text = "" if min_value is None else f"{min_value:g}"
        max_text = "" if max_value is None else f"{max_value:g}"
        parts.append(f"{metric}:{min_text}-{max_text}")
    return "; ".join(parts)


# Match readable filter names to their measurement-table columns.
def find_matching_cell_qc_column(df: pd.DataFrame, metric_name: str) -> Optional[str]:
    target = re.sub(r"[^a-z0-9]+", "", str(metric_name or "").strip().lower())
    if not target:
        return None
    group_key = _FILTER_METRIC_KEYS.get(target, target)
    wanted = _FILTER_COLUMN_GROUPS.get(group_key, (metric_name,))
    normalized_columns = {
        re.sub(r"[^a-z0-9]+", "", str(column).strip().lower()): column for column in df.columns
    }

    for candidate in wanted:
        candidate_key = re.sub(r"[^a-z0-9]+", "", str(candidate).strip().lower())
        if candidate_key in normalized_columns:
            return normalized_columns[candidate_key]

    for column_key, original in normalized_columns.items():
        for candidate in wanted:
            candidate_key = re.sub(r"[^a-z0-9]+", "", str(candidate).strip().lower())
            if column_key.endswith(candidate_key):
                return original

    return None


# Prefer explicit object identifiers because result tables also contain sample
# labels; synthesize sequential labels only for plain, unlabeled tables.
def cell_label_series(cell_table: pd.DataFrame) -> pd.Series:
    saw_label_column = False
    for col in ("CellID", "Cell_ID", "cell_id", "Object", "ID", "id", "Label", "label"):
        if col in cell_table.columns:
            saw_label_column = True
            label_values = cast(
                pd.Series,
                pd.to_numeric(pd.Series(cell_table[col], index=cell_table.index), errors="coerce"),
            )
            labels = pd.Series(label_values.fillna(0).astype(int), index=cell_table.index, dtype=int)
            if (labels > 0).any():
                return labels
    if saw_label_column:
        return pd.Series(0, index=cell_table.index, dtype=int)
    return pd.Series(np.arange(1, len(cell_table) + 1), index=cell_table.index, dtype=int)


def evaluate_cell_qc_table(
    cell_table: pd.DataFrame,
    rules: dict[str, dict[str, Optional[float]]],
) -> tuple[pd.DataFrame, set[int], dict[str, Any]]:
    """Return an annotated copy, failed cell IDs, and an evaluation summary.

    Bounds are inclusive. NaN/nonnumeric values fail an active bound; absent
    columns are listed in missing_metrics instead of flagging every cell.
    The caller must handle unavailable metrics before interpreting membership.
    No rows are removed and the input table is unchanged.
    """
    table = cell_table.copy()
    rules = normalize_cell_qc_rules(rules)
    table["QC_OutOfRange"] = False
    table["QC_Reasons"] = ""
    summary: dict[str, Any] = {
        "rules_applied": rules,
        "missing_metrics": [],
        "flagged_count": 0,
        "total_count": int(len(table)),
    }

    if table.empty or not rules:
        return table, set(), summary

    reasons_by_index: dict[Any, list[str]] = {idx: [] for idx in table.index}
    for metric, limits in rules.items():
        col = find_matching_cell_qc_column(table, metric)
        if col is None:
            summary["missing_metrics"].append(metric)
            continue

        values = cast(
            pd.Series,
            pd.to_numeric(pd.Series(table[col], index=table.index), errors="coerce"),
        )
        min_value = limits.get("min")
        max_value = limits.get("max")
        # Undefined measurements cannot satisfy an active numeric condition.
        metric_flag = values.isna() if min_value is not None or max_value is not None else pd.Series(False, index=table.index)
        if min_value is not None:
            metric_flag = metric_flag | (values < float(min_value))
        if max_value is not None:
            metric_flag = metric_flag | (values > float(max_value))

        for idx in table.index[metric_flag.fillna(False)]:
            raw_value = values.loc[idx]
            parts = []
            if pd.notna(raw_value):
                value = float(raw_value)
                if min_value is not None and value < float(min_value):
                    parts.append(f"{metric}<{min_value:g}")
                if max_value is not None and value > float(max_value):
                    parts.append(f"{metric}>{max_value:g}")
            reasons_by_index[idx].extend(parts or [f"{metric}=missing" if pd.isna(raw_value) else f"{metric}=out_of_range"])

    table["QC_Reasons"] = ["; ".join(reasons_by_index[idx]) for idx in table.index]
    reason_text = pd.Series(table["QC_Reasons"], index=table.index).map(lambda value: str(value))
    table["QC_OutOfRange"] = reason_text.map(len) > 0

    labels = cell_label_series(table)
    out_of_range = pd.Series(table["QC_OutOfRange"], index=table.index).astype(bool)
    flagged_labels: set[int] = {int(label) for label in labels[out_of_range].tolist()}
    flagged_labels.discard(0)

    summary["flagged_count"] = int(table["QC_OutOfRange"].sum())
    return table, flagged_labels, summary


def combine_qc_label_sets(cell_labels: set[int], mask_labels: set[int], mode: str) -> set[int]:
    mode = normalize_qc_filter_mode(mode)
    cell_labels = {int(label) for label in (cell_labels or set()) if int(label) != 0}
    mask_labels = {int(label) for label in (mask_labels or set()) if int(label) != 0}

    if mode == "Exclude only if both fail":
        return cell_labels & mask_labels
    if mode == "Use cell filters only":
        return set(cell_labels)
    if mode == "Use mask filters only":
        return set(mask_labels)
    return cell_labels | mask_labels


# Derive mask-in-cell filter metrics directly from the label image so those
# rules remain available even when optional measurement exports are disabled.
def make_mask_qc_metric_table(
    cell_mask: Any,
    mask_filter: Any,
    intensity_img: Any,
) -> pd.DataFrame:
    """Measure mask overlap per positive cell ID in matching 2-D arrays.

    Areas count pixels; PositiveAreaFractionInCell is a fraction, not percent.
    RawIntDenInCell sums intensity only inside the mask/cell intersection.
    Empty intersections have zero sum and NaN mean; FractionOfCellIntDen is
    NaN when total cell intensity is zero. Input NaNs are otherwise preserved.
    Invalid shapes return an empty table; singleton axes may be squeezed.
    """
    labels = np.asarray(cell_mask)
    if labels.ndim > 2:
        labels = np.squeeze(labels)
    if labels.ndim != 2:
        return pd.DataFrame()

    mask = np.asarray(mask_filter)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.shape != labels.shape:
        return pd.DataFrame()
    mask = mask.astype(bool, copy=False)

    intensity = np.asarray(intensity_img)
    if intensity.ndim > 2:
        intensity = np.squeeze(intensity)
    if intensity.shape != labels.shape:
        return pd.DataFrame()

    # Group foreground pixels once rather than scan the entire image for each
    # cell. Stable sorting preserves each cell's original pixel order and NumPy
    # sum/mean dtype semantics (including float32 rounding and NaNs).
    foreground = labels.ravel() > 0
    ids, inverse, counts = np.unique(labels.ravel()[foreground], return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    values = intensity.ravel()[foreground][order]
    selected = mask.ravel()[foreground][order]
    rows = []
    offset = 0
    for label, count in zip(ids, counts):
        cell_area = int(count)
        cell_values = values[offset:offset + cell_area]
        positive_values = cell_values[selected[offset:offset + cell_area]]
        offset += cell_area
        positive_area = int(positive_values.size)
        cell_intden = float(cell_values.sum()) if cell_values.size else 0.0
        positive_intden = float(positive_values.sum()) if positive_values.size else 0.0
        positive_fraction = float(positive_area / cell_area)

        rows.append(
            {
                "CellID": int(label),
                "MaskQC_CellArea": cell_area,
                "PositiveAreaInCell": positive_area,
                "PositiveAreaFractionInCell": positive_fraction,
                "MeanInPositiveArea": float(positive_values.mean()) if positive_values.size else np.nan,
                "RawIntDenInCell": positive_intden,
                "RawIntDenPerCellArea": float(positive_intden / cell_area) if cell_area else 0.0,
                "FractionOfCellIntDen": (
                    float(positive_intden / cell_intden) if cell_intden != 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# Return a Boolean mask instead of modifying labels so callers can use the same
# selection for preview and exported overlays.
def make_mask_for_cell_labels(cell_mask: Any, labels: set[int]) -> Optional[np.ndarray]:
    if cell_mask is None or not labels:
        return None
    arr = np.asarray(cell_mask)
    if arr.size == 0:
        return None
    return np.isin(arr.astype(np.int64, copy=False), list(labels))


# Compute boundaries from neighboring labels so adjacent cells both remain
# visible instead of collapsing into one foreground outline.
def cell_label_outline_mask(label_img: Any) -> np.ndarray:
    labels = np.asarray(label_img)
    if labels.ndim > 2:
        labels = np.squeeze(labels)
    if labels.ndim != 2:
        raise ValueError(f"Cell outline export expects a 2D label image, got shape {labels.shape}")

    labels = labels.astype(np.int64, copy=False)
    foreground = labels > 0
    outline = np.zeros(labels.shape, dtype=bool)

    outline[:-1, :] |= foreground[:-1, :] & (labels[:-1, :] != labels[1:, :])
    outline[1:, :] |= foreground[1:, :] & (labels[1:, :] != labels[:-1, :])
    outline[:, :-1] |= foreground[:, :-1] & (labels[:, :-1] != labels[:, 1:])
    outline[:, 1:] |= foreground[:, 1:] & (labels[:, 1:] != labels[:, :-1])

    outline[0, :] |= foreground[0, :]
    outline[-1, :] |= foreground[-1, :]
    outline[:, 0] |= foreground[:, 0]
    outline[:, -1] |= foreground[:, -1]

    return outline.astype(np.uint8) * 255


# Append summaries after filtering so exports retain both the original totals
# and the values that downstream analysis will actually use.
def append_cell_qc_summary_rows(
    table: Optional[pd.DataFrame],
    *,
    include_filtered_summary: bool,
    all_cells_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame() if table is None else table.copy()

    work = table.copy()
    all_work = all_cells_table.copy() if all_cells_table is not None else work.copy()
    if "QC_Include" in work.columns:
        work = work.drop(columns=["QC_Include"])
    if "QC_Include" in all_work.columns:
        all_work = all_work.drop(columns=["QC_Include"])

    label_col: Any = "CellID" if "CellID" in work.columns else work.columns[0]
    numeric_cols = []
    for col in work.columns:
        if col == label_col or str(col) == "QC_OutOfRange":
            continue
        values = cast(pd.Series, pd.to_numeric(pd.Series(work[col], index=work.index), errors="coerce"))
        if values.notna().any():
            numeric_cols.append(col)

    # Keep the original CSV column order.
    def _summary_row(label: str, subset: pd.DataFrame, mode: str) -> dict[Any, Any]:
        row: dict[Any, Any] = {col: "" for col in work.columns}
        row[label_col] = label
        if "QC_OutOfRange" in row:
            row["QC_OutOfRange"] = ""
        if "QC_Reasons" in row:
            row["QC_Reasons"] = mode
        for col in numeric_cols:
            values = cast(pd.Series, pd.to_numeric(pd.Series(subset[col], index=subset.index), errors="coerce"))
            if values.notna().any():
                row[col] = float(values.sum() if mode == "SUM" else values.mean())
        return row

    summary_rows = [
        _summary_row("ALL_CELLS_SUM", all_work, "SUM"),
        _summary_row("ALL_CELLS_MEAN", all_work, "MEAN"),
    ]

    if include_filtered_summary and "QC_OutOfRange" in work.columns:
        include_mask = pd.Series(work["QC_OutOfRange"], index=work.index).fillna(False).astype(bool)
        included = cast(pd.DataFrame, work.loc[~include_mask].copy())
        summary_rows += [
            _summary_row("FILTERED_CELLS_SUM", included, "SUM"),
            _summary_row("FILTERED_CELLS_MEAN", included, "MEAN"),
        ]

    return pd.concat([work, pd.DataFrame(summary_rows)], ignore_index=True)
