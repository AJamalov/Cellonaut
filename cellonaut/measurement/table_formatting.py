"""Human-facing column selection, labels and transposition for measurement tables."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, cast

import pandas as pd

from cellonaut.measurement.schema import (
    measurement_unit,
)
from cellonaut.measurement.table_cleanup import drop_empty_rows_and_columns
from cellonaut.measurement.tables import (
    _column_sort_key,
    _readable_row_sort_key,
    human_readable_column_name,
)


# Preserve relative paths: their parent folders distinguish repeated filenames.
def compact_sample_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute():
        text = re.split(r"[\\/]", text)[-1]
    return text


# Shorten sample names and paths in the table's identifier columns.
def compact_sample_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    for col in ("Label", "Sample", "SampleID", "SourceImageFile", "OriginalSamplePath"):
        if col in out.columns:
            out[col] = out[col].map(compact_sample_label)
    return out


# Separate scientific measurements from provenance and filter diagnostics for the simple export.
def is_filter_or_metadata_column(column: str) -> bool:
    text = str(column or "")
    if text in {
        "Label",
        "SampleID",
        "SampleRelativePath",
        "SampleGroup",
        "OriginalSamplePath",
        "SourceImageKey",
        "SourceImageLabel",
        "SourceImageFile",
    }:
        return True
    markers = (
        "QC_",
        "CellQC_",
        "MaskQC_",
        "OutOfRange",
        "Excluded",
        "ExcludeReason",
        "TotalBeforeQC",
        "CellCount",
    )
    return any(marker in text for marker in markers)


# Keep the simple table focused on values most users graph or compare directly.
def simple_measurements_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    keep = []
    if "Label" in df.columns:
        keep.append("Label")
    for col in df.columns:
        if col in keep:
            continue
        if is_filter_or_metadata_column(str(col)):
            continue
        keep.append(col)
    out = df.loc[:, keep].copy()
    return drop_empty_rows_and_columns(compact_sample_column(out), protected_columns=["Label"])


def tidy_measurements_table(df: pd.DataFrame, *, sample_column: str = "Sample") -> pd.DataFrame:
    """Put one numeric observation on each row for direct statistical analysis."""

    columns = ["SampleID", "Sample", "SampleGroup", "MeasuredChannel", "Measurement", "MeasurementLabel", "Value", "Unit"]
    if df is None or df.empty or sample_column not in df.columns:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})

    records: list[dict[str, Any]] = []
    for _, source_row in df.iterrows():
        sample = source_row.get(sample_column, "")
        sample_id = source_row.get("SampleID", "") or source_row.get("Label", "") or compact_sample_label(sample)
        measured_channel = source_row.get("SourceImageLabel", "") or source_row.get("SourceImageKey", "")
        for column in df.columns:
            key = str(column)
            if key == sample_column or is_filter_or_metadata_column(key):
                continue
            converted = cast(pd.Series, pd.to_numeric(pd.Series([source_row[column]]), errors="coerce"))
            value = converted.iloc[0]
            if pd.isna(value):
                continue
            records.append(
                {
                    "SampleID": sample_id,
                    "Sample": sample,
                    "SampleGroup": source_row.get("SampleGroup", "UNKNOWN"),
                    "MeasuredChannel": measured_channel,
                    "Measurement": key,
                    "MeasurementLabel": human_readable_column_name(key),
                    "Value": value,
                    "Unit": measurement_unit(key),
                }
            )
    return pd.DataFrame.from_records(records, columns=columns)


def measurement_summary_table(tidy: pd.DataFrame, *, grouped_samples: bool = False) -> pd.DataFrame:
    """Summarize numeric sample observations without mixing summaries into raw data."""

    columns = ["Group", "Measurement", "MeasurementLabel", "Unit", "N", "Mean", "SD", "Median", "Min", "Max"]
    if tidy is None or tidy.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})

    work = tidy.copy()
    if grouped_samples:
        # Group names are dataset metadata, not a prefix of an encoded ID.
        work["Group"] = work["SampleGroup"].fillna("UNKNOWN") if "SampleGroup" in work.columns else "UNKNOWN"
    else:
        work["Group"] = "All samples"

    rows: list[dict[str, Any]] = []
    for group_key, values in work.groupby(["Group", "Measurement"], sort=False, dropna=False):
        group, measurement = cast(tuple[Any, Any], group_key)
        numeric = cast(
            pd.Series,
            pd.to_numeric(pd.Series(values["Value"], index=values.index), errors="coerce"),
        ).dropna()
        if numeric.empty:
            continue
        rows.append(
            {
                "Group": group,
                "Measurement": measurement,
                "MeasurementLabel": values["MeasurementLabel"].iloc[0],
                "Unit": values["Unit"].iloc[0],
                "N": int(numeric.count()),
                "Mean": numeric.mean(),
                "SD": numeric.std(ddof=1) if len(numeric) > 1 else float("nan"),
                "Median": numeric.median(),
                "Min": numeric.min(),
                "Max": numeric.max(),
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)


# Put measurements in rows so many samples can be compared without extremely wide spreadsheets.
def readable_transposed_table(
    df: pd.DataFrame,
    *,
    sample_column: str = "Label",
    measurement_column: str = "Measurement",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    work = drop_empty_rows_and_columns(compact_sample_column(df), protected_columns=[sample_column])
    if sample_column not in work.columns:
        return readable_results_table(work)

    original_indexes = {str(col): idx for idx, col in enumerate(work.columns)}
    ordered_columns = [sample_column] + sorted(
        [col for col in work.columns if col != sample_column],
        key=lambda col: _readable_row_sort_key(str(col), original_indexes.get(str(col), 0)),
    )
    work = work.loc[:, ordered_columns].copy()

    sample_labels = [compact_sample_label(value) for value in work[sample_column].tolist()]
    seen: dict[str, int] = {}
    unique_labels = []
    for idx, label in enumerate(sample_labels, start=1):
        label = label or f"Sample {idx}"
        seen[label] = seen.get(label, 0) + 1
        unique_labels.append(label if seen[label] == 1 else f"{label} ({seen[label]})")

    rows = []
    for col in work.columns:
        if col == sample_column:
            continue
        values = work[col].tolist()
        if not any(pd.notna(value) and str(value).strip() != "" for value in values):
            continue
        row = {measurement_column: human_readable_column_name(str(col))}
        row.update(dict(zip(unique_labels, values)))
        rows.append(row)
    if not rows:
        empty_columns = [measurement_column, *(str(label) for label in unique_labels)]
        return pd.DataFrame({column: pd.Series(dtype=object) for column in empty_columns})
    return drop_empty_rows_and_columns(pd.DataFrame(rows), protected_columns=[measurement_column])


# Apply readable column names after sorting and calculations.
def readable_results_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    cleaned = drop_empty_rows_and_columns(compact_sample_column(df))
    ordered_columns = sorted(
        list(cleaned.columns), key=lambda col: _column_sort_key(str(col), list(cleaned.columns).index(col))
    )
    work = cleaned.loc[:, ordered_columns].copy()
    counts: dict[str, int] = {}
    columns = []
    for col in work.columns:
        readable = human_readable_column_name(str(col))
        counts[readable] = counts.get(readable, 0) + 1
        if counts[readable] > 1:
            readable = f"{readable} ({counts[readable]})"
        columns.append(readable)
    out = work.copy()
    out.columns = columns
    return out
