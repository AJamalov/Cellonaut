"""Live filter preview lookup, mask adjustment, and filtered table exports.

These helpers reconnect overlay files to their saved cell tables and label
masks, apply current filter rules, and write per-table or batch filtered CSV
exports without rerunning Weka or Cellpose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import tifffile

from cellonaut.cell_segmentation.core import make_per_cell_table
from cellonaut.measurement.table_cleanup import drop_derived_ratio_columns
from cellonaut.masks.adjustments import adjust_label_image as adjust_label_image
from cellonaut.masks.adjustments import adjust_mask_image as adjust_mask_image
from cellonaut.masks.cell_qc import (
    cell_label_series,
    find_matching_cell_qc_column,
    make_mask_qc_metric_table,
    normalize_cell_qc_rules,
)
from cellonaut.masks.cell_groups import (
    evaluate_cell_groups,
    _selected_mask_label as _selected_mask_label,
    excluded_labels_from_current_rules as excluded_labels_from_current_rules,
)
from cellonaut.results.layout import build_results_layout
from cellonaut.results.artifacts import ArtifactResolver, ArtifactMetadataError, image_definition, current_layer_labels, load_artifact_sidecar as _load_preview_sidecar


# Live filtering resolves every required artifact as one bundle so a
# preview cannot accidentally combine tables and masks from different runs.
@dataclass
class PreviewFilterData:
    result_id: str
    source_label: str
    table_path: Path
    labels_path: Path
    table: pd.DataFrame
    label_image: np.ndarray


# Return paths and messages together because batch export may succeed for some
# tables while still needing to report skipped or unreadable inputs.
@dataclass
class BatchFilterExportResult:
    export_dir: Path
    kept_combined_path: Optional[Path]
    report_combined_path: Optional[Path]
    all_measurements_path: Optional[Path]
    simple_measurements_path: Optional[Path]
    readable_measurements_path: Optional[Path]
    signal_combined_path: Optional[Path]
    group_membership_path: Optional[Path]
    processed_count: int
    skipped_count: int
    kept_count: int
    total_count: int
    messages: list[str]


def attach_live_cell_shape_metrics(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Rebuild filter-only shape ratios from saved labels, not exported CSV columns."""
    missing = [key for key in ("Circularity", "Solidity") if key not in table.columns]
    if not missing or "CellID" not in table.columns:
        return table
    computed = make_per_cell_table(labels, np.zeros_like(labels, dtype=np.uint8))
    if computed.empty:
        return table
    return _merge_metrics_by_numeric_cell_id(table, computed.loc[:, ["CellID", *missing]], missing)


def _merge_metrics_by_numeric_cell_id(
    table: pd.DataFrame,
    metrics: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Join cell rows while retaining nonnumeric SUM/MEAN rows unchanged."""
    if "CellID" not in table.columns or "CellID" not in metrics.columns or not metric_columns:
        return table
    join_key = "__cellonaut_numeric_cell_id"
    while join_key in table.columns or join_key in metrics.columns:
        join_key = f"_{join_key}"
    left = table.copy()
    right = metrics.loc[:, ["CellID", *metric_columns]].copy()
    left[join_key] = pd.to_numeric(left["CellID"], errors="coerce")
    right[join_key] = pd.to_numeric(right.pop("CellID"), errors="coerce")
    right = right.loc[right[join_key].notna()].drop_duplicates(subset=[join_key], keep="first")
    return left.merge(right, on=join_key, how="left").drop(columns=[join_key])


def _merge_mask_signal_metrics(
    table: pd.DataFrame,
    signal_table: pd.DataFrame,
    mask_label: str,
) -> pd.DataFrame:
    """Attach the selected mask's per-cell metrics using QC's canonical names."""
    if "CellID" not in table.columns or "CellID" not in signal_table.columns:
        return table

    source_columns = {
        f"{mask_label}Area_InCell": "PositiveAreaInCell",
        f"{mask_label}AreaFraction_InCell": "PositiveAreaFractionInCell",
        f"{mask_label}Mean_InCell": "MeanInPositiveArea",
        f"{mask_label}MeanGrayValue_InCell": "MeanInPositiveArea",
        f"{mask_label}IntDen_InCell": "RawIntDenInCell",
        f"{mask_label}IntDenPerCellArea": "RawIntDenPerCellArea",
        f"{mask_label}FractionOfCellIntDen": "FractionOfCellIntDen",
        "CellIntDen": "MeasuredCellIntDen",
    }
    available: dict[str, str] = {}
    for source, target in source_columns.items():
        if source in signal_table.columns and target not in available.values():
            available[source] = target
    if not available:
        return table

    signal_rows = signal_table.loc[:, ["CellID", *available]].rename(columns=available)
    target_columns = [target for target in available.values() if target not in table.columns]
    if target_columns:
        table = _merge_metrics_by_numeric_cell_id(table, signal_rows, target_columns)

    def ratio(numerator: str, denominator: str) -> pd.Series:
        top = cast(pd.Series, pd.to_numeric(pd.Series(table[numerator], index=table.index), errors="coerce"))
        bottom = cast(pd.Series, pd.to_numeric(pd.Series(table[denominator], index=table.index), errors="coerce"))
        return top.div(bottom.where(bottom != 0))

    cell_area = "CellArea" if "CellArea" in table.columns else "Area"
    if "PositiveAreaFractionInCell" not in table.columns and "PositiveAreaInCell" in table.columns and cell_area in table.columns:
        table["PositiveAreaFractionInCell"] = ratio("PositiveAreaInCell", cell_area)
    if "RawIntDenPerCellArea" not in table.columns and "RawIntDenInCell" in table.columns and cell_area in table.columns:
        table["RawIntDenPerCellArea"] = ratio("RawIntDenInCell", cell_area)
    if "FractionOfCellIntDen" not in table.columns and "RawIntDenInCell" in table.columns and "MeasuredCellIntDen" in table.columns:
        table["FractionOfCellIntDen"] = ratio("RawIntDenInCell", "MeasuredCellIntDen")
    return table


def attach_live_mask_metrics(
    table: pd.DataFrame,
    label_image: np.ndarray,
    mask_image: np.ndarray,
    intensity_image: np.ndarray,
    *,
    replace_existing: bool = True,
) -> pd.DataFrame:
    """Calculate mask-in-cell columns from the requested source pixels."""
    metrics = make_mask_qc_metric_table(label_image, mask_image, intensity_image)
    if metrics.empty or "CellID" not in table.columns:
        return table
    metric_columns = [column for column in metrics.columns if column != "CellID"]
    if replace_existing:
        table = table.drop(columns=metric_columns, errors="ignore")
    else:
        metric_columns = [column for column in metric_columns if column not in table.columns]
    return _merge_metrics_by_numeric_cell_id(table, metrics, metric_columns)


# Anchor artifact lookup to the nearest Results ancestor so opening an older run
# cannot accidentally link against the output folder currently selected in Setup.
def _results_root_from_preview(preview_file: Path) -> Optional[Path]:
    for parent in [preview_file.parent, *preview_file.parents]:
        if parent.name.lower() == "results":
            return parent
    return None


# Strip only known export suffixes because sample names legitimately contain
# underscores and must remain intact when matching tables and masks.
def _result_id_from_name(name: str, source_label: str = "") -> str:
    cleaned = str(name or "")
    for suffix in (
        "_combined_overlay",
        "_cellpose_outline_stack",
        "_overlay",
        "_qc",
        "_qc_overlay",
    ):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break

    source_label = str(source_label or "").strip()
    if source_label and cleaned.endswith(f"_{source_label}"):
        cleaned = cleaned[: -(len(source_label) + 1)]

    return cleaned or str(name or "")


# Prefer the longest configured label so a channel named GFP_mask is not
# mistaken for a shorter GFP label embedded in the same filename.
def _label_from_preview_name(stem: str, image_defs: list[dict[str, Any]]) -> str:
    names = [str(img.get("name", "") or "").strip() for img in image_defs]
    names = [name for name in names if name]
    for name in sorted(names, key=len, reverse=True):
        if name in stem:
            return name
    return ""


# Sidecars preserve layer names when filenames alone are ambiguous after users
# rename channels between generating and reopening a preview.

def _table_base_from_path(path: Path) -> str:
    name = path.name
    suffix = "_cell_measurements.csv"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


# Test known labels before splitting on the final underscore so channel names
# containing underscores remain recoverable.
def _split_table_base(base: str, label_candidates: list[str]) -> tuple[str, str]:
    for label in sorted([v for v in label_candidates if v], key=len, reverse=True):
        suffix = f"_{label}"
        if base.endswith(suffix):
            return base[: -len(suffix)], label
    if "_" in base:
        result_id, label = base.rsplit("_", 1)
        return result_id, label
    return base, ""


# Read the canonical source label so preview filters select the same channel as pipeline exports.
def _source_label_from_table(table: pd.DataFrame) -> str:
    if "SourceImageLabel" in table.columns:
        values = cast(
            pd.Series,
            pd.Series(table["SourceImageLabel"], index=table.index).dropna().astype(str),
        )
        values = cast(pd.Series, values.loc[values.str.strip() != ""])
        if not values.empty:
            return str(values.iloc[0]).strip()
    return ""


# Remove summary rows before counting or joining cells; their textual CellID
# labels belong in reports but must never be treated as measured objects.
def _numeric_cell_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty or "CellID" not in df.columns:
        return df.iloc[:0].copy()
    work = df.copy()
    work["_CellIDNumeric"] = pd.to_numeric(work["CellID"], errors="coerce")
    numeric_ids = pd.Series(work["_CellIDNumeric"], index=work.index)
    work = cast(pd.DataFrame, work.loc[numeric_ids.notna()].copy())
    work = cast(pd.DataFrame, work.loc[pd.Series(work["_CellIDNumeric"], index=work.index) > 0].copy())
    work["CellID"] = work["_CellIDNumeric"].astype(int)
    return cast(pd.DataFrame, work.drop(columns=["_CellIDNumeric"]))


# Return the mask label with each path because filtered signal summaries need
# that relationship name when rebuilding combined measurement columns.
def _find_signal_tables(root: Path, result_id: str, source_label: str, relative_parent: Path = Path()) -> list[tuple[Path, str]]:
    prefix = f"{result_id}_{source_label}_"
    suffix = "_per_cell_signal.csv"
    matches: list[tuple[Path, str]] = []
    folder = root / "CSV Data" / "Signal" / relative_parent
    if not folder.exists():
        return matches
    for path in sorted(folder.glob(f"{prefix}*{suffix}")):
        roi_label = path.name[len(prefix) : -len(suffix)]
        if roi_label:
            matches.append((path, roi_label))
    return matches


# Rank several weak filename clues rather than trusting one substring; this is
# needed when users reopen artifacts after renaming channels in the current setup.
def _artifact_score(
    *,
    base: str,
    preview_stem: str,
    label: str,
    source_candidates: list[str],
    result_candidates: list[str],
) -> int:
    score = 0
    if base and base in preview_stem:
        score += 100
    if any(result and base.startswith(f"{result}_") for result in result_candidates):
        score += 40
    if label and label in source_candidates:
        score += 30
    if label and label in preview_stem:
        score += 20
    if base and preview_stem.startswith(base):
        score += 20
    return score


# Exact-path lookup remains deterministic and separate from the broader glob
# fallback used for artifacts produced by older runs in the same 1.0 schema.
def _find_first_existing(candidates: list[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


# Apply the same ranking and deterministic filename tie-break to tables and
# masks so fallback lookup cannot drift as the two artifact paths evolve.
def _best_scored_artifact(
    paths: list[Path],
    *,
    table: bool,
    preview_stem: str,
    label_candidates: list[str],
    source_candidates: list[str],
    result_candidates: list[str],
) -> tuple[Optional[Path], str, str]:
    ranked = []
    for path in paths:
        base = (
            _table_base_from_path(path)
            if table
            else path.name.replace("_01_cellpose_labels.tif", "").replace("_cellpose_labels.tif", "")
        )
        result, label = _split_table_base(base, label_candidates)
        score = _artifact_score(
            base=base,
            preview_stem=preview_stem,
            label=label,
            source_candidates=source_candidates,
            result_candidates=result_candidates,
        )
        # A channel match alone never establishes sample identity.
        exact_base = _result_id_from_name(preview_stem) == base
        if score > 0 and (result in result_candidates or exact_base):
            ranked.append((score, path, result, label))
    if not ranked:
        return None, "", ""
    _score, path, result, label = min(ranked, key=lambda item: (-item[0], item[1].name.lower()))
    return path, result, label


# Resolve the table and label image as one pair because displaying exclusions
# from one sample on another sample's mask would produce a plausible but false preview.
def find_preview_filter_data(
    preview_file: str | Path,
    image_defs: list[dict[str, Any]],
) -> tuple[Optional[PreviewFilterData], str]:
    """Use exact run-local relationships, or the isolated historical resolver."""
    try:
        resolver = ArtifactResolver.load(Path(preview_file))
        if resolver is None:
            return _find_legacy_preview_filter_data(preview_file, image_defs)
        record = resolver.record(Path(preview_file))
        resolver.path(record)
        table_record = resolver.related(record, "cell_table")
        labels_record = resolver.related(record, "cell_labels")
        table_path = resolver.path(table_record)
        labels_path = resolver.path(labels_record)
        table = pd.read_csv(table_path)
        labels = np.asarray(tifffile.imread(labels_path))
        if labels.ndim > 2:
            labels = np.squeeze(labels)
        if labels.ndim != 2:
            raise ValueError(f"Cell label mask is not 2D: shape {labels.shape}")
        table = attach_live_cell_shape_metrics(table, labels)
        definition = image_definition(table_record, image_defs) or {}
        source_label = str(definition.get("name") or table_record.get("label", ""))
        mask_label = _selected_mask_label(definition)
        signal = resolver.signal(record, mask_label, image_defs)
        if signal is not None:
            table = _merge_mask_signal_metrics(table, pd.read_csv(resolver.path(signal)), signal["mask_label"])
        return PreviewFilterData(record["sample"], source_label, table_path, labels_path, table, labels), ""
    except (ArtifactMetadataError, OSError, ValueError, tifffile.TiffFileError) as exc:
        return None, f"Could not resolve saved artifacts: {exc}"


def _find_legacy_preview_filter_data(
    preview_file: str | Path,
    image_defs: list[dict[str, Any]],
) -> tuple[Optional[PreviewFilterData], str]:
    preview_path = Path(preview_file)
    if not preview_path.exists():
        return None, "No preview file is loaded."

    results_root = _results_root_from_preview(preview_path)
    if results_root is None:
        return None, "Open an overlay from a Results folder to preview cell groups."

    sidecar = _load_preview_sidecar(preview_path)
    current_labels = [str(img.get("name", "") or "").strip() for img in image_defs]
    current_labels = [label for label in current_labels if label]
    sidecar_labels = [
        str(value or "").strip()
        for value in [
            sidecar.get("base_label", ""),
            *(sidecar.get("layer_labels", []) or []),
        ]
        if str(value or "").strip()
    ]
    source_candidates = []
    for candidate in [
        _label_from_preview_name(preview_path.stem, image_defs),
        *sidecar_labels,
        *current_labels,
    ]:
        candidate = str(candidate or "").strip()
        if candidate and candidate not in source_candidates:
            source_candidates.append(candidate)

    source_label = source_candidates[0] if source_candidates else ""
    if not source_label:
        # Last resort: infer from available table names below.
        source_label = ""
    result_candidates = []
    for candidate_label in source_candidates or [""]:
        result = _result_id_from_name(preview_path.stem, candidate_label)
        if result and result not in result_candidates:
            result_candidates.append(result)
    if preview_path.stem and preview_path.stem not in result_candidates:
        result_candidates.append(preview_path.stem)

    source_def = find_image_def_for_label(image_defs, source_label) or {}
    cell_source_label = str(source_def.get("analysis_cell_segmentation_source", "") or source_label).strip()
    label_candidates = [label for label in [source_label, *source_candidates] if label]
    if cell_source_label and cell_source_label not in label_candidates:
        label_candidates.append(cell_source_label)

    layout = build_results_layout(results_root, create_root=False)
    relative_parent = Path()
    for category in ("mask_overlays", "qc_overlay_pngs", "cell_segmentation_outlines", "cell_segmentation_qc_pngs"):
        try:
            relative_parent = preview_path.parent.relative_to(layout[category])
            break
        except ValueError:
            continue
    cell_table_dirs = [layout["cell_segmentation_tables"] / relative_parent]
    cell_label_dirs = [layout["cell_segmentation_labels"] / relative_parent]
    table_path = _find_first_existing(
        [
            table_dir / f"{result_id}_{label}_cell_measurements.csv"
            for table_dir in cell_table_dirs
            for result_id in result_candidates
            for label in label_candidates
        ]
    )
    if table_path is None:
        table_path, inferred_result_id, inferred_label = _best_scored_artifact(
            [path for folder in cell_table_dirs for path in sorted(folder.glob("*_cell_measurements.csv"))],
            table=True,
            preview_stem=preview_path.stem,
            label_candidates=label_candidates,
            source_candidates=source_candidates,
            result_candidates=result_candidates,
        )
        if table_path is not None:
            if inferred_result_id and inferred_result_id not in result_candidates:
                result_candidates.insert(0, inferred_result_id)
            if inferred_label and inferred_label not in label_candidates:
                label_candidates.insert(0, inferred_label)
            if inferred_label:
                source_label = inferred_label

    # The label image must belong to the exact selected table, never a
    # separately ranked sample or channel.
    table_base = _table_base_from_path(table_path) if table_path is not None else ""
    labels_path = _find_first_existing([
        folder / f"{table_base}{suffix}"
        for folder in cell_label_dirs
        for suffix in ("_01_cellpose_labels.tif", "_cellpose_labels.tif")
    ]) if table_base else None

    if table_path is None:
        searched = ", ".join(result_candidates[:3]) or preview_path.stem
        checked = ", ".join(str(path) for path in cell_table_dirs)
        return None, f"No cell measurement table found near {searched}. Checked {checked}."
    if labels_path is None:
        searched = ", ".join(result_candidates[:3]) or preview_path.stem
        checked = ", ".join(str(path) for path in cell_label_dirs)
        return None, f"No Cellpose label mask found near {searched}. Checked {checked}."

    table_base = _table_base_from_path(table_path)
    inferred_result_id, inferred_source_label = _split_table_base(table_base, label_candidates)
    result_id = inferred_result_id or (result_candidates[0] if result_candidates else preview_path.stem)
    source_label = inferred_source_label or source_label or (label_candidates[0] if label_candidates else "")

    try:
        table = pd.read_csv(table_path)
    except Exception as exc:
        return None, f"Could not read cell measurement table: {type(exc).__name__}: {exc}"

    try:
        labels = np.asarray(tifffile.imread(labels_path))
        if labels.ndim > 2:
            labels = np.squeeze(labels)
        if labels.ndim != 2:
            return None, f"Cell label mask is not 2D: shape {labels.shape}"
    except Exception as exc:
        return None, f"Could not read Cellpose label mask: {type(exc).__name__}: {exc}"

    table = attach_live_cell_shape_metrics(table, labels)

    image_def = find_image_def_for_label(image_defs, source_label)
    mask_label = _selected_mask_label(image_def or {})
    if mask_label:
        signal_dirs = [layout["cell_signal_tables"] / relative_parent]
        signal_path = _find_first_existing(
            [folder / f"{result_id}_{source_label}_{mask_label}_per_cell_signal.csv" for folder in signal_dirs]
        )
        if signal_path is None:
            signal_path = next(
                (
                    path
                    for folder in signal_dirs
                    for path in sorted(folder.glob(f"{result_id}_{source_label}_*_per_cell_signal.csv"))
                    if mask_label.lower() in path.stem.lower()
                ),
                None,
            )
        if signal_path is not None:
            try:
                table = _merge_mask_signal_metrics(table, pd.read_csv(signal_path), mask_label)
                if "MaskQC_IntensitySource" not in table.columns:
                    table["MaskQC_IntensitySource"] = "Measured image"
            except Exception:
                pass

    return PreviewFilterData(
        result_id=result_id,
        source_label=source_label,
        table_path=table_path,
        labels_path=labels_path,
        table=table,
        label_image=labels,
    ), ""


# Match display labels exactly because similarly named channels may carry
# different filter rules and mask relationships.
def find_image_def_for_label(image_defs: list[dict[str, Any]], label: str) -> Optional[dict[str, Any]]:
    for img in image_defs:
        if str(img.get("name", "") or "").strip() == label:
            return img
    return None


def filtered_table_from_current_rules(
    table: pd.DataFrame,
    image_def: dict[str, Any],
    parse_rules,
    *,
    prepare_table=None,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int], dict[str, Any]]:
    """Return kept cell rows, a full annotated report, excluded IDs, and counts.

    The report retains original rows, including summaries; the kept table drops
    summaries and excluded cells. Neither replaces the measured input. Current
    groups use shared evaluation and reject unavailable metrics. Callers without
    groups retain the legacy direct-rule path and LiveFilter report columns.
    """
    populations = [dict(value) for value in image_def.get("cell_populations", []) or [] if isinstance(value, dict)]
    if populations:
        result = evaluate_cell_groups(table, image_def, parse_rules, prepare_table=prepare_table)
        result.require_available()
        labels = cell_label_series(table)
        report = table.copy()
        report["CellGroup_CellID"] = labels
        group_columns: list[tuple[str, str]] = []
        used_group_columns: set[str] = set()
        for group in result.groups:
            name = group.name
            safe_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or f"Group_{group.index + 1}"
            column = f"CellGroup_{safe_name}"
            base_column = column
            suffix = 2
            while column.casefold() in used_group_columns:
                column = f"{base_column}_{suffix}"
                suffix += 1
            used_group_columns.add(column.casefold())
            report[column] = labels.isin(group.labels)
            group_columns.append((column, name))
        report["CellGroups"] = report.apply(
            lambda row: ";".join(group_name for column, group_name in group_columns if bool(row[column])),
            axis=1,
        )
        report["CellGroupCount"] = report[[column for column, _name in group_columns]].fillna(False).astype(bool).sum(axis=1)
        report["CellGroup_ExcludedFromCSV"] = labels.isin(result.excluded_labels)
        cell_rows = _numeric_cell_rows(report)
        kept = cast(pd.DataFrame, cell_rows.loc[cell_label_series(cell_rows).isin(result.kept_labels)].copy())
        return kept, report, set(result.excluded_labels), {
            "total": len(result.cell_labels),
            "excluded": len(result.excluded_labels),
            "group_counts": {group.name: len(group.labels) for group in result.groups},
            "has_filters": any(group.has_filters for group in result.groups),
        }

    evaluation_table = prepare_table(table, image_def) if prepare_table is not None and not table.empty else table
    excluded, summary = excluded_labels_from_current_rules(evaluation_table, image_def, parse_rules)
    missing = list(summary.get("cell_missing", [])) + list(summary.get("mask_missing", []))
    if missing:
        raise ValueError("Cell-group metrics unavailable: " + ", ".join(missing))
    labels = cell_label_series(table)

    report = table.copy()
    report["LiveFilter_CellID"] = labels
    report["LiveFilter_Excluded"] = labels.isin(excluded)

    cell_table = summary.get("cell_table")
    if isinstance(cell_table, pd.DataFrame):
        if "QC_OutOfRange" in cell_table.columns:
            report["LiveFilter_CellFilterOutOfRange"] = cell_table["QC_OutOfRange"].values
        if "QC_Reasons" in cell_table.columns:
            report["LiveFilter_CellFilterReasons"] = cell_table["QC_Reasons"].values

    mask_table = summary.get("mask_table")
    if isinstance(mask_table, pd.DataFrame):
        if "QC_OutOfRange" in mask_table.columns:
            report["LiveFilter_MaskFilterOutOfRange"] = mask_table["QC_OutOfRange"].values
        if "QC_Reasons" in mask_table.columns:
            report["LiveFilter_MaskFilterReasons"] = mask_table["QC_Reasons"].values

    cell_rows = _numeric_cell_rows(report)
    excluded_rows = pd.Series(cell_rows["LiveFilter_Excluded"], index=cell_rows.index).astype(bool)
    kept = cast(pd.DataFrame, cell_rows.loc[~excluded_rows].copy())
    return kept, report, excluded, summary


def export_all_filtered_result_tables(
    results_root: str | Path,
    image_defs: list[dict[str, Any]],
    parse_rules,
    *,
    export_dir: str | Path | None = None,
) -> BatchFilterExportResult:
    """Write derived Cell Group tables from saved measurements and current settings.

    By default reserve a numbered export folder, leaving original files intact;
    an explicit export_dir must be a separate destination. Resolve relationships
    from metadata when present, with filename lookup only for legacy results.
    Report per-table failures/skips in the result; invalid run metadata raises.
    """
    from cellonaut.cell_segmentation.core import normalize_cell_segmentation_export_table
    from cellonaut.io.writers import write_dataframe_csv
    from cellonaut.measurement.math import summarize_per_cell_table
    from cellonaut.measurement.table_formatting import (
        drop_empty_rows_and_columns,
        readable_results_table,
        readable_transposed_table,
        simple_measurements_table,
    )
    from cellonaut.measurement.tables import normalize_export_table

    root = Path(results_root)
    layout = build_results_layout(root, create_root=False)
    root = layout["root"]
    tables_dir = layout["cell_segmentation_tables"]
    if export_dir is not None:
        out_dir = Path(export_dir)
    else:
        from cellonaut.pipeline.run_outputs import next_numbered_child

        out_dir = next_numbered_child(layout["filtered_exports"], "export")

    labels = [str(img.get("name", "") or "").strip() for img in image_defs]
    labels = [label for label in labels if label]

    kept_frames: list[pd.DataFrame] = []
    report_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    all_measurement_rows: list[dict[str, Any]] = []
    messages: list[str] = []
    processed = 0
    skipped = 0
    kept_total = 0
    total = 0

    resolver = ArtifactResolver.load(root)
    if resolver is None and not tables_dir.exists():
        return BatchFilterExportResult(
            export_dir=out_dir,
            kept_combined_path=None,
            report_combined_path=None,
            all_measurements_path=None,
            simple_measurements_path=None,
            readable_measurements_path=None,
            signal_combined_path=None,
            group_membership_path=None,
            processed_count=0,
            skipped_count=0,
            kept_count=0,
            total_count=0,
            messages=[f"No table folder found: {tables_dir}"],
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    table_records = resolver.matching(kind="cell_table") if resolver is not None else []
    table_paths = (
        [root / record["path"] for record in table_records]
        if resolver is not None else sorted(tables_dir.rglob("*_cell_measurements.csv"))
    )
    for table_path in table_paths:
        relative_parent = table_path.parent.relative_to(
            tables_dir if table_path.is_relative_to(tables_dir) else root
        )
        sample_out_dir = out_dir / relative_parent
        record = resolver.record(table_path) if resolver is not None else None
        if record is not None:
            result_id, source_label = record["sample"], record.get("label", "")
        else:
            table_base = _table_base_from_path(table_path)
            result_id, source_label = _split_table_base(table_base, labels)

        try:
            if resolver is not None and record is not None:
                table_path = resolver.path(record)
            table = pd.read_csv(table_path)
        except Exception as exc:
            skipped += 1
            messages.append(f"Skipped {table_path.name}: could not read table ({type(exc).__name__}: {exc})")
            continue

        if resolver is not None and record is not None:
            try:
                labels_path = resolver.path(resolver.related(record, "cell_labels"))
            except ArtifactMetadataError as exc:
                skipped += 1
                messages.append(f"Skipped {table_path.name}: {exc}")
                continue
        else:
            labels_path = layout["cell_segmentation_labels"] / relative_parent / f"{_table_base_from_path(table_path)}_01_cellpose_labels.tif"
        # Shared only while evaluating this sample: label image, overlay planes
        # and sidecar data are reused across the requested intensity sources.
        image_cache: dict[tuple[Path, int], Any] = {}
        if labels_path.is_file():
            try:
                label_image = np.asarray(tifffile.imread(labels_path))
                if label_image.ndim > 2:
                    label_image = np.squeeze(label_image)
                if label_image.ndim != 2:
                    raise ValueError(f"Cell label mask is not 2D: shape {label_image.shape}")
                image_cache[(labels_path, 0)] = label_image
                table = attach_live_cell_shape_metrics(table, image_cache[(labels_path, 0)])
            except Exception as exc:
                messages.append(f"Cell shape filters unavailable for {table_path.name}: {type(exc).__name__}: {exc}")
                if resolver is not None:
                    skipped += 1
                    continue

        source_label = source_label or _source_label_from_table(table)
        image_def = (
            image_definition(record, image_defs) if record is not None
            else find_image_def_for_label(image_defs, source_label)
        )
        if image_def is None:
            skipped += 1
            label_text = source_label or "(unknown channel)"
            messages.append(f"Skipped {table_path.name}: no current cell-group settings found for {label_text}")
            continue

        source_label = str(image_def.get("name") or source_label)
        try:
            if resolver is not None and record is not None:
                signals = [
                    (resolver.path(item), item["mask_label"])
                    for item in resolver.matching(
                        sample=record["sample"], target=record["target"], kind="cell_signal"
                    )
                ]
            else:
                signals = _find_signal_tables(root, result_id, source_label, relative_parent)
        except ArtifactMetadataError as exc:
            skipped += 1
            messages.append(f"Skipped {table_path.name}: {exc}")
            continue
        mask_label = _selected_mask_label(image_def)
        if mask_label:
            signal_label = mask_label
            if resolver is not None and record is not None:
                signal = resolver.signal(record, mask_label, image_defs)
                signal_path = resolver.path(signal) if signal is not None else None
                signal_label = signal["mask_label"] if signal is not None else mask_label
            else:
                signal_path = next((path for path, label in signals if label == mask_label), None)
            if signal_path is not None and signal_path.is_file():
                try:
                    table = _merge_mask_signal_metrics(table, pd.read_csv(signal_path), signal_label)
                except Exception as exc:
                    messages.append(f"Mask filter metrics unavailable for {table_path.name}: {type(exc).__name__}: {exc}")

        try:
            kept, report, _excluded, summary = filtered_table_from_current_rules(
                table, image_def, parse_rules,
                prepare_table=lambda current, settings, relative_parent=relative_parent, result_id=result_id, source_label=source_label, labels_path=labels_path, image_cache=image_cache, record=record: saved_mask_filter_metrics(
                    current, settings, layout, relative_parent, result_id, source_label, labels_path, parse_rules,
                    image_cache=image_cache, resolver=resolver, record=record, image_defs=image_defs,
                ),
            )
        except Exception as exc:
            skipped += 1
            messages.append(f"Skipped {table_path.name}: filter evaluation failed ({type(exc).__name__}: {exc})")
            continue

        kept = drop_derived_ratio_columns(kept)
        report = drop_derived_ratio_columns(report)

        kept = kept.copy()
        report = report.copy()
        kept.insert(0, "ResultID", result_id)
        report.insert(0, "ResultID", result_id)
        kept.insert(1, "FilterSource", source_label)
        report.insert(1, "FilterSource", source_label)
        kept_cell_rows = _numeric_cell_rows(kept)
        kept_cell_ids = pd.Series(
            pd.to_numeric(kept_cell_rows["CellID"], errors="coerce"),
            index=kept_cell_rows.index,
        )
        kept_labels = set(kept_cell_ids.dropna().astype(int).tolist())
        total_count = int(summary.get("total", len(_numeric_cell_rows(table))))
        row: dict[str, Any] = {
            "Label": result_id,
            "SourceImageLabel": source_label,
            f"{source_label}_CellCount_TotalBeforeQC": total_count,
            f"{source_label}_CellCount": len(kept_labels),
            f"{source_label}_CellQC_OutOfRangeCount": int(summary.get("cell_flagged", 0) or 0),
            f"{source_label}_MaskQC_OutOfRangeCount": int(summary.get("mask_flagged", 0) or 0),
            f"{source_label}_QC_ExcludedCount": int(summary.get("excluded", 0) or 0),
        }

        cell_source = str(image_def.get("analysis_cell_segmentation_source", "") or source_label)
        row.update(summarize_per_cell_table(kept_cell_rows, "", source_label, cell_source))
        table_stem = table_path.stem
        kept_path = sample_out_dir / f"{table_stem}_filtered.csv"
        report_path = sample_out_dir / f"{table_stem}_filter_report.csv"
        write_dataframe_csv(normalize_cell_segmentation_export_table(kept), kept_path, index=False)
        write_dataframe_csv(normalize_cell_segmentation_export_table(report), report_path, index=False)

        for signal_path, roi_label in signals:
            try:
                signal_df = pd.read_csv(signal_path)
            except Exception as exc:
                messages.append(
                    f"Skipped signal table {signal_path.name}: could not read table ({type(exc).__name__}: {exc})"
                )
                continue
            signal_cell_rows = _numeric_cell_rows(signal_df)
            if "CellID" not in signal_cell_rows.columns:
                messages.append(f"Skipped signal table {signal_path.name}: missing CellID column")
                continue
            signal_cell_ids = pd.Series(signal_cell_rows["CellID"], index=signal_cell_rows.index)
            filtered_signal = cast(
                pd.DataFrame,
                signal_cell_rows.loc[signal_cell_ids.isin(list(kept_labels))].copy(),
            )
            filtered_signal = drop_derived_ratio_columns(filtered_signal)
            filtered_signal.insert(0, "ResultID", result_id)
            filtered_signal.insert(1, "FilterSource", source_label)
            filtered_signal.insert(2, "Mask", roi_label)
            filtered_signal_path = sample_out_dir / f"{signal_path.stem}_filtered.csv"
            write_dataframe_csv(
                normalize_cell_segmentation_export_table(filtered_signal), filtered_signal_path, index=False
            )
            signal_frames.append(filtered_signal)
            row.update(summarize_per_cell_table(filtered_signal, roi_label, source_label, cell_source))

        all_measurement_rows.append(row)
        kept_frames.append(kept)
        report_frames.append(report)
        processed += 1
        table_total = int(summary.get("total", len(_numeric_cell_rows(table))))
        total += table_total
        kept_total += len(kept)

    kept_combined_path = None
    report_combined_path = None
    all_measurements_path = None
    simple_measurements_path = None
    readable_measurements_path = None
    signal_combined_path = None
    group_membership_path = None
    if kept_frames:
        kept_combined_path = out_dir / "All_Cell_Measurements_Filtered.csv"
        kept_combined = pd.concat(kept_frames, ignore_index=True)
        write_dataframe_csv(normalize_cell_segmentation_export_table(kept_combined), kept_combined_path, index=False)
    if report_frames:
        report_combined_path = out_dir / "All_Cell_Measurements_Filter_Report.csv"
        report_combined = pd.concat(report_frames, ignore_index=True)
        write_dataframe_csv(
            normalize_cell_segmentation_export_table(report_combined), report_combined_path, index=False
        )
        membership_rows: list[dict[str, Any]] = []
        if "CellGroups" in report_combined.columns:
            for _, report_row in _numeric_cell_rows(report_combined).iterrows():
                matched = [name.strip() for name in str(report_row.get("CellGroups", "") or "").split(";") if name.strip()]
                for group_name in matched or ["Ungrouped"]:
                    membership_rows.append(
                        {
                            "ResultID": report_row.get("ResultID", ""),
                            "MeasuredChannel": report_row.get("FilterSource", ""),
                            "CellID": report_row.get("CellID", ""),
                            "CellGroup": group_name,
                            "ExcludedFromCSV": bool(report_row.get("CellGroup_ExcludedFromCSV", False)),
                        }
                    )
        if membership_rows:
            group_membership_path = out_dir / "Cell_Group_Membership.csv"
            write_dataframe_csv(pd.DataFrame(membership_rows), group_membership_path, index=False)
    if signal_frames:
        signal_combined_path = out_dir / "All_Per_Cell_Signal_Filtered.csv"
        signal_combined = pd.concat(signal_frames, ignore_index=True)
        signal_combined = drop_empty_rows_and_columns(signal_combined, protected_columns=["ResultID", "CellID"])
        write_dataframe_csv(
            normalize_cell_segmentation_export_table(signal_combined), signal_combined_path, index=False
        )
    if all_measurement_rows:
        all_df = drop_empty_rows_and_columns(pd.DataFrame(all_measurement_rows), protected_columns=["Label"])
        all_df = normalize_export_table(all_df)
        simple_df = simple_measurements_table(all_df)
        simple_measurements_path = out_dir / "Measurements.csv"
        write_dataframe_csv(readable_results_table(simple_df), simple_measurements_path, index=False)
        all_measurements_path = simple_measurements_path
        readable_measurements_path = out_dir / "Measurements_By_Metric.csv"
        write_dataframe_csv(
            readable_transposed_table(simple_df, sample_column="Sample"), readable_measurements_path, index=False
        )

    return BatchFilterExportResult(
        export_dir=out_dir,
        kept_combined_path=kept_combined_path,
        report_combined_path=report_combined_path,
        all_measurements_path=all_measurements_path,
        simple_measurements_path=simple_measurements_path,
        readable_measurements_path=readable_measurements_path,
        signal_combined_path=signal_combined_path,
        group_membership_path=group_membership_path,
        processed_count=processed,
        skipped_count=skipped,
        kept_count=kept_total,
        total_count=total,
        messages=messages,
    )


def saved_mask_filter_metrics(table, settings, layout, relative_parent, result_id, source_label, labels_path, parse_rules, *, image_cache=None, resolver=None, record=None, image_defs=None):
    """Prepare one mask/source combination, reusing this operation's loaded pixels."""
    if resolver is None:
        resolver = ArtifactResolver.load(labels_path)
        if resolver is not None:
            record = resolver.record(labels_path)
    rules = parse_rules(str(settings.get("mask_qc_limits", "") or ""))
    if not rules:
        return table
    cell_source = settings.get("mask_qc_intensity_source", "Measured image") == "Cell mask image"
    if not cell_source and all(find_matching_cell_qc_column(table, metric) is not None for metric in normalize_cell_qc_rules(rules)):
        return table
    mask_label = _selected_mask_label(settings)
    intensity_label = str(settings.get("analysis_cell_segmentation_source", "") or source_label) if cell_source else source_label
    from cellonaut.artifact_naming import portable_component
    from cellonaut.io.image_io import read_tiff_numpy_2d

    if resolver is not None and record is not None:
        overlay = resolver.path(resolver.related(record, "combined_overlay"))
    else:
        # Legacy export relationship: exact sample/channel overlay filename.
        overlay = layout["mask_overlays"] / relative_parent / f"{result_id}_{portable_component(source_label)}_combined_overlay.tif"
    cache = {} if image_cache is None else image_cache
    if (overlay, -1) not in cache:
        cache[(overlay, -1)] = resolver.sidecar(resolver.record(overlay)) if resolver is not None else _load_preview_sidecar(overlay)
    sidecar = cache[(overlay, -1)]
    names = current_layer_labels(sidecar, image_defs or []) if resolver is not None else list(sidecar.get("layer_labels", []))
    if not overlay.is_file() or not labels_path.is_file() or mask_label not in names or intensity_label not in names:
        if not cell_source:
            return table
        raise ValueError("Requested Cellpose-source mask intensity is unavailable in saved results; regenerate the run with its source channels.")
    if (labels_path, 0) not in cache:
        cache[(labels_path, 0)] = np.asarray(tifffile.imread(labels_path))
    def plane(label):
        index = names.index(label) + 1
        if (overlay, index) not in cache:
            cache[(overlay, index)] = read_tiff_numpy_2d(overlay, stack_channel_index=index)
        return cache[(overlay, index)]
    return attach_live_mask_metrics(table, cache[(labels_path, 0)], plane(mask_label), plane(intensity_label), replace_existing=cell_source)
