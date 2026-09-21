"""Measurement-table aggregation and CSV export services.

The pipeline runner supplies completed measurement rows; this module owns the
filesystem-oriented table formats derived from those rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd

from cellonaut.artifact_naming import collision_resistant_ascii_component
from cellonaut.config.defaults import INPUT_STRUCTURE_GROUPED_BY_PROTEIN
from cellonaut.io.writers import write_dataframe_csv
from cellonaut.measurement.table_cleanup import drop_derived_ratio_columns
from cellonaut.measurement.table_formatting import (
    drop_empty_rows_and_columns,
    measurement_summary_table,
    readable_transposed_table,
    readable_results_table,
    simple_measurements_table,
    tidy_measurements_table,
)
from cellonaut.measurement.tables import (
    normalize_export_table,
    sort_measurement_rows_for_export,
)
from cellonaut.results.labels import is_summary_label
from cellonaut.results.layout import build_results_layout


# Keep per-channel filenames ASCII-portable without collapsing distinct
# Unicode display labels onto the same output path.
def _channel_table_filename_stem(label: str) -> str:
    return collision_resistant_ascii_component(label, fallback="Channel")


def _with_legacy_cell_count_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Preserve historical CSV columns without carrying pipeline filter state.

    Original measurements contain all cells. Existing diagnostic columns on
    imported/filtered tables keep their own values; only absent columns are
    synthesized. Never mutate the authoritative measurement rows/table.
    """
    table = table.copy()
    for column in list(table.columns):
        if not str(column).endswith("_CellCount"):
            continue
        prefix = str(column).removesuffix("_CellCount")
        counts = table[column]
        defaults = {
            "CellCount_TotalBeforeQC": counts,
            "CellQC_OutOfRangeCount": counts.where(counts.isna(), 0),
            "MaskQC_OutOfRangeCount": counts.where(counts.isna(), 0),
            "QC_ExcludedCount": counts.where(counts.isna(), 0),
            "CellQC_ExcludedCount": counts.where(counts.isna(), 0),
        }
        for suffix, values in defaults.items():
            name = f"{prefix}_{suffix}"
            if name not in table:
                table[name] = values
    return table


# Derive every per-channel file from the same sorted source table so merged and
# channel-specific exports cannot disagree about row values.
def _write_per_channel_tables(
    table: pd.DataFrame,
    *,
    channel_dir: Path,
    log_func: Callable[[str], None],
) -> dict[str, str]:
    channel_column = "SourceImageLabel" if "SourceImageLabel" in table.columns else "SourceImageKey"
    if channel_column not in table.columns:
        return {}
    channel_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    channel_values = pd.Series(table[channel_column], index=table.index).fillna("").map(str)
    for channel_label in dict.fromkeys(value for value in channel_values if value):
        channel_rows = cast(pd.DataFrame, table.loc[channel_values == channel_label].copy())
        channel_rows = drop_empty_rows_and_columns(channel_rows, protected_columns=["Label"])
        channel_path = channel_dir / f"{_channel_table_filename_stem(channel_label)}_Measurements.csv"
        write_dataframe_csv(readable_results_table(normalize_export_table(channel_rows)), channel_path, index=False)
        written[f"channel:{channel_label}"] = str(channel_path)
        log_func(f"Saved per-channel measurements: {channel_path}")
    return written


def _extract_skeleton_table(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern = re.compile(r"^(.+)_MaskSkeleton_(SkeletonCount|BranchCount|EndpointCount|JunctionCount)$")
    skeleton_columns = [str(column) for column in table.columns if pattern.match(str(column))]
    if not skeleton_columns:
        return table, pd.DataFrame()

    records: list[dict[str, Any]] = []
    identifiers = [column for column in ("Label", "SampleID", "OriginalSamplePath") if column in table.columns]
    for _, source_row in table[[*identifiers, *skeleton_columns]].iterrows():
        records_by_mask: dict[str, dict[str, Any]] = {}
        for column in skeleton_columns:
            match = pattern.match(column)
            if match is None or bool(pd.isna(source_row[column])):
                continue
            mask_label, metric_name = match.groups()
            record = records_by_mask.setdefault(
                mask_label,
                {
                    "Sample": source_row.get("Label", ""),
                    "SampleID": source_row.get("SampleID", ""),
                    "OriginalSamplePath": source_row.get("OriginalSamplePath", ""),
                    "Mask": mask_label,
                },
            )
            record[metric_name] = source_row[column]
        records.extend(records_by_mask.values())

    columns = [
        "Sample",
        "SampleID",
        "OriginalSamplePath",
        "Mask",
        "SkeletonCount",
        "BranchCount",
        "EndpointCount",
        "JunctionCount",
    ]
    skeletons = pd.DataFrame(records).drop_duplicates().reindex(columns=columns) if records else pd.DataFrame()
    return table.drop(columns=skeleton_columns), skeletons


def _write_merged_measurement_tables(
    table: pd.DataFrame,
    *,
    summary_dir: Path,
    log_func: Callable[[str], None],
    cfg: Any,
    include_summary: bool,
) -> dict[str, str]:
    exported = normalize_export_table(drop_empty_rows_and_columns(table, protected_columns=["Label"]))
    simple = simple_measurements_table(exported)
    primary_path = summary_dir / "Measurements.csv"
    by_metric_path = summary_dir / "Measurements_By_Metric.csv"
    write_dataframe_csv(readable_results_table(simple), primary_path, index=False)
    write_dataframe_csv(readable_transposed_table(simple, sample_column="Sample"), by_metric_path, index=False)
    tidy_sample_column = "OriginalSamplePath" if "OriginalSamplePath" in table.columns else "Label"
    tidy = tidy_measurements_table(table, sample_column=tidy_sample_column)

    written = {
        "all": str(primary_path),
        "short": str(primary_path),
        "all_long": str(by_metric_path),
        "simple": str(primary_path),
        "simple_long": str(by_metric_path),
    }
    if include_summary and not tidy.empty:
        summary = measurement_summary_table(
            tidy,
            grouped_samples=cfg.input_structure == INPUT_STRUCTURE_GROUPED_BY_PROTEIN,
        )
        if not summary.empty:
            summary_path = summary_dir / "Measurement_Summary.csv"
            summary = summary.rename(columns={"Measurement": "MeasurementKey", "MeasurementLabel": "Measurement"})
            summary = cast(
                pd.DataFrame,
                summary.loc[
                    :,
                    ["Group", "Measurement", "MeasurementKey", "Unit", "N", "Mean", "SD", "Median", "Min", "Max"],
                ].copy(),
            )
            write_dataframe_csv(summary, summary_path, index=False)
            written["summary"] = str(summary_path)

    log_func(f"Saved primary measurements: {primary_path}")
    return written


def write_measurement_summary_csvs(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    cfg: Any,
    log_func: Callable[[str], None],
    include_averages: bool = True,
) -> dict[str, str]:
    """Write all configured measurement-table formats from completed rows."""

    table = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows or [])
    table = drop_derived_ratio_columns(table)
    if table.empty:
        return {}

    table = _with_legacy_cell_count_columns(table)
    table = sort_measurement_rows_for_export(table)
    layout = build_results_layout(cfg.output_dir, create_root=False)
    summary_dir = layout["summaries"]
    written = _write_per_channel_tables(table, channel_dir=layout["channel_tables"], log_func=log_func)

    summary_dir.mkdir(parents=True, exist_ok=True)
    table, skeletons = _extract_skeleton_table(table)
    if not skeletons.empty:
        skeleton_path = summary_dir / "Skeleton_Measurements.csv"
        write_dataframe_csv(skeletons, skeleton_path, index=False)
        written["skeletons"] = str(skeleton_path)
        log_func(f"Saved skeleton measurements: {skeleton_path}")
    labels = pd.Series(table["Label"], index=table.index).fillna("").map(str) if "Label" in table.columns else None
    if labels is not None:
        table = cast(pd.DataFrame, table.loc[~labels.apply(is_summary_label) & (labels.str.strip() != "")].copy())
    written.update(
        _write_merged_measurement_tables(
            table,
            summary_dir=summary_dir,
            log_func=log_func,
            cfg=cfg,
            include_summary=include_averages,
        )
    )
    return written
