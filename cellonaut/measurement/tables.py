"""CSV table normalization, summaries, and readable measurement exports.

The helpers here keep output column names, summary rows, long-format exports,
and human-readable measurement tables stable across pipeline and preview runs.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, cast

import pandas as pd

from cellonaut.measurement.table_cleanup import drop_empty_rows_and_columns

from cellonaut.measurement.schema import (
    METRIC_SPECS_BY_KEY,
    RELATION_METRIC_PATTERN,
    match_count_column,
    match_metric_column,
    measurement_unit,
)
from cellonaut.measurement.table_summaries import (
    add_per_folder_average_rows,
    add_summary_mean_row,
    insert_group_header_rows,
    move_summary_rows_to_front,
    row_matches_exclusion_tag,
)
from cellonaut.results.labels import SUMMARY_MEAN_LABEL, is_summary_label


__all__ = [
    "add_per_folder_average_rows",
    "add_summary_mean_row",
    "insert_group_header_rows",
    "move_summary_rows_to_front",
    "row_matches_exclusion_tag",
]


_ID_COLUMNS = {
    "SampleRelativePath",
    "SampleGroup",
    "Label",
    "Sample",
    "SampleID",
    "OriginalSamplePath",
    "SourceImageKey",
    "SourceImageLabel",
    "SourceImageFile",
    "CellposeMaskKey",
    "CellposeMaskLabel",
}

_EXPORT_DROP_COLUMNS = {
    "SampleRelativePath",
    "SampleGroup",
    "Label",
    "SampleID",
    "SourceImageKey",
    "SourceImageLabel",
    "SourceImageFile",
    "CellposeMaskKey",
}


_FIXED_COLUMN_NAMES = {
    "Label": "Sample",
    "Sample": "Sample",
    "SampleID": "Sample ID",
    "OriginalSamplePath": "Sample",
    "SourceImageKey": "Measured channel key",
    "SourceImageLabel": "Measured channel",
    "SourceImageFile": "Measured channel file",
    "CellposeMaskKey": "Cellpose mask key",
    "CellposeMaskLabel": "Cellpose mask",
    "Column": "Measurement",
    "Value": "Value",
    "SourceTarget": "Measured channel key",
    "CellID": "Cell ID",
    "LiveFilter_CellID": "Cell ID used by live filter",
    "LiveFilter_Excluded": "Live filter: excluded",
    "LiveFilter_CellFilterOutOfRange": "Live filter: cell filter failed",
    "LiveFilter_CellFilterReasons": "Live filter: cell filter reason",
    "LiveFilter_MaskFilterOutOfRange": "Live filter: mask filter failed",
    "LiveFilter_MaskFilterReasons": "Live filter: mask filter reason",
    "QC_OutOfRange": "Cell filter: out of range",
    "QC_Reasons": "Cell filter: reason",
    "QC_Excluded": "Excluded from filtered measurements",
    "QC_ExcludeReason": "Exclusion reason",
    "MaskQC_OutOfRange": "Mask filter: out of range",
    "MaskQC_Reasons": "Mask filter: reason",
    "MaskQC_IntensitySource": "Mask filter intensity source",
    "MaskQC_CellArea": "Cell area used for mask filter (px²)",
    "Area": "Cell area (px²)",
    "Mean": "Cell mean intensity",
    "Average": "Cell mean intensity",
    "Min": "Cell minimum intensity",
    "Max": "Cell maximum intensity",
    "StdDev": "Cell intensity standard deviation",
    "Mode": "Cell modal gray value",
    "CentroidX": "Cell centroid X",
    "CentroidY": "Cell centroid Y",
    "CenterOfMassX": "Cell center of mass X",
    "CenterOfMassY": "Cell center of mass Y",
    "Perimeter": "Cell perimeter",
    "BoundingRectX": "Cell bounding rectangle X",
    "BoundingRectY": "Cell bounding rectangle Y",
    "BoundingRectWidth": "Cell bounding rectangle width",
    "BoundingRectHeight": "Cell bounding rectangle height",
    "EllipseMajor": "Cell fitted ellipse major axis",
    "EllipseMinor": "Cell fitted ellipse minor axis",
    "EllipseAngle": "Cell fitted ellipse angle",
    "Circularity": "Cell circularity",
    "AspectRatio": "Cell aspect ratio",
    "Roundness": "Cell roundness",
    "Solidity": "Cell solidity",
    "Feret": "Cell Feret diameter",
    "FeretX": "Cell Feret X",
    "FeretY": "Cell Feret Y",
    "FeretAngle": "Cell Feret angle",
    "MinFeret": "Cell minimum Feret diameter",
    "Median": "Cell median intensity",
    "Skewness": "Cell intensity skewness",
    "Kurtosis": "Cell intensity kurtosis",
    "AreaFraction": "Cell area fraction",
    "RawIntDen": "Cell raw integrated density",
    "IntDen": "Cell integrated density",
    "CellArea": "Cellpose whole-cell area (px²)",
    "CellPerimeter": "Cellpose whole-cell perimeter",
    "CellMean": "Cellpose whole-cell mean intensity",
    "CellAverage": "Cellpose whole-cell mean intensity",
    "CellStdDev": "Cellpose whole-cell intensity standard deviation",
    "CellMode": "Cellpose whole-cell modal gray value",
    "CellMin": "Cellpose whole-cell minimum intensity",
    "CellMax": "Cellpose whole-cell maximum intensity",
    "CellCentroidX": "Cellpose whole-cell centroid X",
    "CellCentroidY": "Cellpose whole-cell centroid Y",
    "CellCenterOfMassX": "Cellpose whole-cell center of mass X",
    "CellCenterOfMassY": "Cellpose whole-cell center of mass Y",
    "CellBoundingRectX": "Cellpose whole-cell bounding rectangle X",
    "CellBoundingRectY": "Cellpose whole-cell bounding rectangle Y",
    "CellBoundingRectWidth": "Cellpose whole-cell bounding rectangle width",
    "CellBoundingRectHeight": "Cellpose whole-cell bounding rectangle height",
    "CellEllipseMajor": "Cellpose whole-cell fitted ellipse major axis",
    "CellEllipseMinor": "Cellpose whole-cell fitted ellipse minor axis",
    "CellEllipseAngle": "Cellpose whole-cell fitted ellipse angle",
    "CellFeret": "Cellpose whole-cell maximum Feret diameter",
    "CellCircularity": "Cellpose whole-cell circularity",
    "CellSolidity": "Cellpose whole-cell solidity",
    "CellMedian": "Cellpose whole-cell median intensity",
    "CellIntDen": "Cellpose whole-cell raw integrated density",
    "CellSkewness": "Cellpose whole-cell intensity skewness",
    "CellKurtosis": "Cellpose whole-cell intensity kurtosis",
    "CellCorrectedMean": "Cellpose whole-cell mean intensity (background subtracted)",
    "CellCorrectedAverage": "Cellpose whole-cell mean intensity (background subtracted)",
    "CellCorrectedIntDen": "Cellpose whole-cell integrated density (background subtracted)",
    "RestOfCellArea": "Rest of cell area (px²)",
    "RestOfCellAreaFraction": "Rest of cell area fraction",
    "RestOfCellMean": "Rest of cell mean intensity",
    "RestOfCellAverage": "Rest of cell mean intensity",
    "RestOfCellIntDen": "Rest of cell raw integrated density",
    "RestOfCellIntDenPerCellArea": "Rest of cell integrated density per cell area",
    "RestOfCellFractionOfCellIntDen": "Rest of cell fraction of cell integrated density",
    "RestOfCellCorrectedMean": "Rest of cell mean intensity (background subtracted)",
    "RestOfCellCorrectedAverage": "Rest of cell mean intensity (background subtracted)",
    "RestOfCellCorrectedIntDen": "Rest of cell integrated density (background subtracted)",
    "RestOfCellCorrectedIntDenPerCellArea": "Rest of cell background-subtracted integrated density per cell area",
    "RestOfCellFractionOfCellCorrectedIntDen": "Rest of cell fraction of cell background-subtracted integrated density",
    "PositiveArea": "Mask area (px²)",
    "PositiveAreaInCell": "Mask area inside cell (px²)",
    "PositiveAreaFraction": "Mask fraction of cell area",
    "PositiveAreaFractionInCell": "Mask fraction of cell area",
    "MeanInPositiveArea": "Mean intensity inside mask",
    "AverageInPositiveArea": "Mean intensity inside mask",
    "RawIntDenInCell": "Raw integrated density inside mask",
    "RawIntDenPerCellArea": "Raw integrated density inside mask per cell area",
    "FractionOfCellIntDen": "Fraction of cell integrated density inside mask",
}

_SKELETON_METRIC_NAMES = {
    "SkeletonCount": "disconnected skeleton count",
    "BranchCount": "branch count",
    "EndpointCount": "endpoint count",
    "JunctionCount": "junction count",
}


# Decode known machine names before applying generic replacements so scientific meaning is not lost.
def human_readable_column_name(column: str) -> str:
    text = str(column or "")
    readable = _human_readable_column_name_without_unit(text)
    unit = measurement_unit(text)
    return f"{readable} ({unit})" if unit and not readable.endswith(f"({unit})") else readable


def _human_readable_column_name_without_unit(text: str) -> str:
    if text in _FIXED_COLUMN_NAMES:
        return _FIXED_COLUMN_NAMES[text]

    skeleton_match = re.match(
        r"^(.+)_MaskSkeleton_(SkeletonCount|BranchCount|EndpointCount|JunctionCount)$",
        text,
    )
    if skeleton_match:
        mask, metric = skeleton_match.groups()
        return f"{mask} mask skeleton: {_SKELETON_METRIC_NAMES[metric]}"

    readable = _humanize_organelle_column(text)
    if readable:
        return readable

    readable = _humanize_rolling_ball_column(text)
    if readable:
        return readable

    readable = _humanize_relation_measurement(text)
    if readable:
        return readable

    count_match = match_count_column(text)
    if count_match is not None:
        prefix, count_spec = count_match
        return _humanize_measurement_prefix(prefix) + count_spec.readable_suffix

    metric_match = match_metric_column(text)
    if metric_match is not None:
        prefix, metric_spec = metric_match
        return _humanize_measurement_prefix(prefix) + f": {metric_spec.readable_suffix}"

    replacements = [
        ("_CorrectedIntDen_", ": integrated density after background subtraction at "),
        ("_RB", ": integrated density after background subtraction at RB"),
    ]
    for suffix, readable_suffix in replacements:
        if text.endswith(suffix):
            prefix = text[: -len(suffix)]
            return _humanize_measurement_prefix(prefix) + readable_suffix

    return _humanize_measurement_prefix(text)


_PERCELL_METRIC_NAMES = {
    "TotalCellArea": "total cell area (px²)",
    "TotalCellPerimeter": "total cell perimeter",
    "MeanOfCellMeans": "mean cell intensity",
    "AverageOfCellMeans": "mean cell intensity",
    "MeanOfCellStdDevs": "mean cell intensity standard deviation",
    "AverageOfCellStdDevs": "mean cell intensity standard deviation",
    "MeanOfCellModes": "mean cell modal gray value",
    "AverageOfCellModes": "mean cell modal gray value",
    "MinimumCellIntensity": "minimum Cellpose whole-cell intensity",
    "MaximumCellIntensity": "maximum Cellpose whole-cell intensity",
    "MeanCellCentroidX": "mean cell centroid X",
    "MeanCellCentroidY": "mean cell centroid Y",
    "MeanCellCenterOfMassX": "mean cell center of mass X",
    "MeanCellCenterOfMassY": "mean cell center of mass Y",
    "MeanCellBoundingRectX": "mean cell bounding rectangle X",
    "MeanCellBoundingRectY": "mean cell bounding rectangle Y",
    "MeanCellBoundingRectWidth": "mean cell bounding rectangle width",
    "MeanCellBoundingRectHeight": "mean cell bounding rectangle height",
    "MeanCellEllipseMajor": "mean cell fitted ellipse major axis",
    "MeanCellEllipseMinor": "mean cell fitted ellipse minor axis",
    "MeanCellEllipseAngle": "mean cell fitted ellipse angle",
    "MeanCellFeretDiameter": "mean cell maximum Feret diameter",
    "MeanCellCircularity": "mean cell circularity",
    "MeanCellSolidity": "mean cell solidity",
    "MeanOfCellMedians": "mean cell median intensity",
    "AverageOfCellMedians": "mean cell median intensity",
    "MeanCellSkewness": "mean cell intensity skewness",
    "MeanCellKurtosis": "mean cell intensity kurtosis",
    "AverageOfCellAverages": "mean cell intensity",
    "SumCellIntDen": "sum cell raw integrated density",
    "MeanOfCellCorrectedMeans": "mean cell intensity (background subtracted)",
    "AverageOfCellCorrectedAverages": "mean cell intensity (background subtracted)",
    "SumCellCorrectedIntDen": "sum cell integrated density (background subtracted)",
    "TotalMaskArea": "total mask area inside cells (px²)",
    "MeanOfMaskMeans": "mean intensity inside the mask",
    "MeanOfMaskStdDevs": "mean intensity standard deviation inside the mask",
    "AverageOfMaskStdDevs": "mean intensity standard deviation inside the mask",
    "MeanOfMaskModes": "mean modal gray value inside the mask",
    "AverageOfMaskModes": "mean modal gray value inside the mask",
    "MinimumMaskIntensity": "minimum intensity inside the mask",
    "MaximumMaskIntensity": "maximum intensity inside the mask",
    "MeanMaskCentroidX": "mean mask centroid X inside cells",
    "MeanMaskCentroidY": "mean mask centroid Y inside cells",
    "MeanMaskCenterOfMassX": "mean mask center of mass X inside cells",
    "MeanMaskCenterOfMassY": "mean mask center of mass Y inside cells",
    "TotalMaskPerimeter": "total mask perimeter inside cells",
    "MeanMaskBoundingRectX": "mean mask bounding rectangle X inside cells",
    "MeanMaskBoundingRectY": "mean mask bounding rectangle Y inside cells",
    "MeanMaskBoundingRectWidth": "mean mask bounding rectangle width inside cells",
    "MeanMaskBoundingRectHeight": "mean mask bounding rectangle height inside cells",
    "MeanMaskEllipseMajor": "mean mask fitted ellipse major axis inside cells",
    "MeanMaskEllipseMinor": "mean mask fitted ellipse minor axis inside cells",
    "MeanMaskEllipseAngle": "mean mask fitted ellipse angle inside cells",
    "MeanMaskFeretDiameter": "mean mask maximum Feret diameter inside cells",
    "MeanMaskCircularity": "mean mask circularity inside cells",
    "MeanMaskSolidity": "mean mask solidity inside cells",
    "MeanOfMaskMedians": "mean median intensity inside the mask",
    "AverageOfMaskMedians": "mean median intensity inside the mask",
    "MeanMaskSkewness": "mean intensity skewness inside the mask",
    "MeanMaskKurtosis": "mean intensity kurtosis inside the mask",
    "AverageOfMaskAverages": "mean intensity inside the mask",
    "SumIntDen": "sum raw integrated density inside the mask",
    "MeanOfCorrectedMaskMeans": "mean intensity inside the mask (background subtracted)",
    "MeanOfCorrectedMaskStdDevs": "mean intensity standard deviation inside the mask (background subtracted)",
    "MinimumCorrectedMaskIntensity": "minimum intensity inside the mask (background subtracted)",
    "MaximumCorrectedMaskIntensity": "maximum intensity inside the mask (background subtracted)",
    "MeanOfCorrectedMaskMedians": "mean median intensity inside the mask (background subtracted)",
    "AverageOfCorrectedMaskAverages": "mean intensity inside the mask (background subtracted)",
    "SumCorrectedIntDen": "sum integrated density inside the mask (background subtracted)",
    "TotalRestArea": "total non-mask cell area (px²)",
    "MeanOfRestMeans": "mean intensity in the rest of the cell",
    "AverageOfRestAverages": "mean intensity in the rest of the cell",
    "SumRestIntDen": "sum raw integrated density in the rest of the cell",
    "MeanOfRestCorrectedMeans": "mean intensity in the rest of the cell (background subtracted)",
    "AverageOfRestCorrectedAverages": "mean intensity in the rest of the cell (background subtracted)",
    "SumRestCorrectedIntDen": "sum integrated density in the rest of the cell (background subtracted)",
}


# Parse the most specific region patterns first because their prefixes overlap.
def _split_percell_column(text: str) -> dict[str, str] | None:
    match = re.match(r"^(.+)_measured_with_(.+)_cellpose_mask_minus_(.+)_mask_PerCell_(.+)$", text)
    if match:
        measured, cell_mask, mask, metric = match.groups()
        return {
            "measured": measured,
            "region": f"{cell_mask} Cellpose cells outside {mask} mask",
            "metric": metric,
        }
    match = re.match(r"^(.+)_measured_with_(.+)_cellpose_mask_PerCell_(.+)$", text)
    if match:
        measured, mask, metric = match.groups()
        return {"measured": measured, "region": f"{mask} Cellpose cells", "metric": metric}
    match = re.match(r"^(.+)_measured_with_(.+)_mask_PerCell_(.+)$", text)
    if match:
        measured, mask, metric = match.groups()
        return {"measured": measured, "region": f"{mask} mask", "metric": metric}
    match = re.match(r"^(.+)_measured_with_(.+)_PerCell_(.+)$", text)
    if match:
        return {"measured": match.group(1), "region": f"{match.group(2)} mask", "metric": match.group(3)}
    match = re.match(r"^(.+)_PerCell_(.+)$", text)
    if match:
        return {"measured": "", "region": f"{match.group(1)} mask", "metric": match.group(2)}
    return None


# Rebuild per-cell labels from parsed parts so channel and mask roles stay explicit.
def _humanize_percell_column(text: str) -> str:
    parsed = _split_percell_column(text)
    if parsed is None:
        return ""
    measured = parsed["measured"]
    region = parsed["region"]
    metric = parsed["metric"]
    metric_text = _PERCELL_METRIC_NAMES.get(metric, _humanize_measurement_prefix(metric))
    if measured:
        return f"{measured}({region.removesuffix(' mask')}) : {metric_text}"
    return f"{region} : {metric_text}"


# Keep a readable fallback for new columns that do not yet have a dedicated formatter.
def _humanize_measurement_prefix(text: str) -> str:
    text = str(text or "")
    text = text.replace("_measured_with_", " measured with ")
    text = text.replace("_in_", " measured in ")
    text = text.replace("_PerCell_", " per cell ")
    text = text.replace("_", " ")
    return " ".join(text.split())


# Reverse the filename-safe radius encoding when presenting rolling-ball results.
def _format_rb_radius(rb_tag: str) -> str:
    text = str(rb_tag or "")
    if text.startswith("RB"):
        text = text[2:]
    text = text.replace("p", ".")
    try:
        value = float(text)
    except ValueError:
        return text
    return str(int(value)) if value.is_integer() else str(value)


# Centralize metric wording so guides and readable tables use the same terminology.
def _metric_display_name(metric: str) -> str:
    spec = METRIC_SPECS_BY_KEY.get(metric)
    return spec.display_name if spec is not None else _humanize_measurement_prefix(metric)


# Keep the subtraction radius visible because each sweep column represents a different result.
def _humanize_rolling_ball_column(text: str) -> str:
    match = re.match(r"^(.+)_in_(.+)_(RB[0-9p.]+)_IntDen$", text)
    if not match:
        return ""
    measured, mask, rb_tag = match.groups()
    radius = _format_rb_radius(rb_tag)
    return f"{measured}({mask}) : integrated density ({radius}px Subtract Background)"


# Apply user-facing terminology only when exporting, leaving calculation names untouched.
def _average_column_name(column: str) -> str:
    text = str(column or "")
    replacements = [
        (SUMMARY_MEAN_LABEL, "Average"),
        ("MEAN", "AVERAGE"),
        ("CorrectedMeanRatio", "CorrectedAverageRatio"),
        ("MeanRatio", "AverageRatio"),
        ("CorrectedMean", "CorrectedAverage"),
        ("MeanOf", "AverageOf"),
        ("MeanIn", "AverageIn"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r"(^|_)Mean(?=_|$)", lambda match: f"{match.group(1)}MeanGrayValue", text)
    return text


# Remove internal routing metadata after it has served sorting and aggregation.
def normalize_export_table(df: pd.DataFrame, *, include_sample: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()

    has_relative_paths = "SampleRelativePath" in out.columns
    if has_relative_paths:
        out["Sample"] = out["SampleRelativePath"]
        out = out.drop(columns=["OriginalSamplePath"], errors="ignore")
    elif "OriginalSamplePath" in out.columns:
        out = out.rename(columns={"OriginalSamplePath": "Sample"})
        if "Label" in out.columns:
            sample_text = pd.Series(out["Sample"], index=out.index).map(lambda value: str(value).strip())
            empty_sample = out["Sample"].isna() | (sample_text == "")
            out.loc[empty_sample, "Sample"] = out.loc[empty_sample, "Label"]
    elif include_sample and "Label" in out.columns and "Sample" not in out.columns:
        out.insert(0, "Sample", pd.Series(out["Label"], index=out.index))

    drop_cols = [col for col in out.columns if str(col) in _EXPORT_DROP_COLUMNS]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    out = out.rename(columns={col: _average_column_name(str(col)) for col in out.columns})

    if "Sample" in out.columns:
        out["Sample"] = out["Sample"].map(lambda value: "Average" if str(value) == SUMMARY_MEAN_LABEL else value)
        if not has_relative_paths:
            out["Sample"] = out["Sample"].map(_sample_filename)

    return drop_empty_rows_and_columns(out, protected_columns=["Sample"] if "Sample" in out.columns else [])


def _sample_filename(value: Any) -> str:
    text = str(value or "").strip()
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute():
        return re.split(r"[\\/]", text)[-1]
    return text


# Group samples by measured channel while leaving summary rows after real observations.
def sort_measurement_rows_for_export(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    work = df.copy()
    if "Label" not in work.columns:
        return work

    labels = pd.Series(work["Label"], index=work.index).fillna("").map(str)
    summary_mask = labels.apply(is_summary_label) | (labels.map(str.strip) == "")
    sample_rows = cast(pd.DataFrame, work.loc[~summary_mask].copy())
    summary_rows = cast(pd.DataFrame, work.loc[summary_mask].copy())
    if sample_rows.empty:
        return work

    target_col = None
    for candidate in ("SourceImageLabel", "SourceImageKey"):
        if candidate in sample_rows.columns:
            target_col = candidate
            break

    if target_col is None:
        return work

    sample_rows["_cellonaut_target_sort"] = (
        pd.Series(
            sample_rows[target_col],
            index=sample_rows.index,
        )
        .fillna("")
        .map(lambda value: str(value).lower())
    )
    sample_rows["_cellonaut_sample_sort"] = (
        pd.Series(
            sample_rows["Label"],
            index=sample_rows.index,
        )
        .fillna("")
        .astype(str)
    )
    sample_rows["_cellonaut_original_order"] = range(len(sample_rows))
    sample_rows = cast(
        pd.DataFrame,
        sample_rows.sort_values(
            by=["_cellonaut_target_sort", "_cellonaut_sample_sort", "_cellonaut_original_order"],
            kind="stable",
        ).drop(columns=["_cellonaut_target_sort", "_cellonaut_sample_sort", "_cellonaut_original_order"]),
    )
    return pd.concat([sample_rows, summary_rows], ignore_index=True)


# Preserve the distinction between measured channel and defining mask in readable names.
def _split_source_mask_prefix(prefix: str) -> tuple[str, str]:
    marker = "_measured_with_"
    if marker in prefix:
        source, mask = prefix.split(marker, 1)
        return source, mask
    return "", prefix


# Decode direct channel-in-mask measurements before broader underscore replacement runs.
def _humanize_relation_measurement(text: str) -> str:
    match = re.match(
        rf"^(.+)_in_(.+)_({RELATION_METRIC_PATTERN})$",
        text,
    )
    if not match:
        return ""
    measured, mask, metric = match.groups()
    return f"{measured}({mask}) : {_metric_display_name(metric)}"


# Handle configured-mask-within-Cellpose-cell metrics as one encoded family.
_ORGANELLE_COLUMN_PATTERNS = (
    (r"^(.+)Area_InCell$", "area inside cell (px²)"),
    (r"^(.+)AreaFraction_InCell$", "area fraction inside cell"),
    (r"^(.+)Mean_InCell$", "mean intensity inside cell"),
    (r"^(.+)StdDev_InCell$", "intensity standard deviation inside cell"),
    (r"^(.+)Mode_InCell$", "modal gray value inside cell"),
    (r"^(.+)Min_InCell$", "minimum intensity inside cell"),
    (r"^(.+)Max_InCell$", "maximum intensity inside cell"),
    (r"^(.+)CentroidX_InCell$", "centroid X inside cell"),
    (r"^(.+)CentroidY_InCell$", "centroid Y inside cell"),
    (r"^(.+)CenterOfMassX_InCell$", "center of mass X inside cell"),
    (r"^(.+)CenterOfMassY_InCell$", "center of mass Y inside cell"),
    (r"^(.+)Perimeter_InCell$", "perimeter inside cell"),
    (r"^(.+)BoundingRectX_InCell$", "bounding rectangle X inside cell"),
    (r"^(.+)BoundingRectY_InCell$", "bounding rectangle Y inside cell"),
    (r"^(.+)BoundingRectWidth_InCell$", "bounding rectangle width inside cell"),
    (r"^(.+)BoundingRectHeight_InCell$", "bounding rectangle height inside cell"),
    (r"^(.+)EllipseMajor_InCell$", "fitted ellipse major axis inside cell"),
    (r"^(.+)EllipseMinor_InCell$", "fitted ellipse minor axis inside cell"),
    (r"^(.+)EllipseAngle_InCell$", "fitted ellipse angle inside cell"),
    (r"^(.+)Feret_InCell$", "maximum Feret diameter inside cell"),
    (r"^(.+)Circularity_InCell$", "circularity inside cell"),
    (r"^(.+)Solidity_InCell$", "solidity inside cell"),
    (r"^(.+)Median_InCell$", "median intensity inside cell"),
    (r"^(.+)Skewness_InCell$", "intensity skewness inside cell"),
    (r"^(.+)Kurtosis_InCell$", "intensity kurtosis inside cell"),
    (r"^(.+)MeanGrayValue_InCell$", "mean intensity inside cell"),
    (r"^(.+)Average_InCell$", "mean intensity inside cell"),
    (r"^(.+)IntDen_InCell$", "raw integrated density inside cell"),
    (r"^(.+)IntDenPerCellArea$", "integrated density inside cell per cell area"),
    (r"^(.+)FractionOfCellIntDen$", "fraction of cell integrated density"),
    (r"^(.+)CorrectedMean_InCell$", "mean intensity inside cell (background subtracted)"),
    (r"^(.+)CorrectedStdDev_InCell$", "intensity standard deviation inside cell (background subtracted)"),
    (r"^(.+)CorrectedMin_InCell$", "minimum intensity inside cell (background subtracted)"),
    (r"^(.+)CorrectedMax_InCell$", "maximum intensity inside cell (background subtracted)"),
    (r"^(.+)CorrectedMedian_InCell$", "median intensity inside cell (background subtracted)"),
    (r"^(.+)CorrectedAverage_InCell$", "mean intensity inside cell (background subtracted)"),
    (r"^(.+)CorrectedIntDen_InCell$", "integrated density inside cell (background subtracted)"),
    (r"^(.+)CorrectedIntDenPerCellArea$", "background-subtracted integrated density inside cell per cell area"),
    (r"^(.+)FractionOfCellCorrectedIntDen$", "fraction of cell background-subtracted integrated density"),
)

_ORGANELLE_RATIO_METRIC_NAMES = {
    "MeanRatio": "mean intensity ratio",
    "AverageRatio": "mean intensity ratio",
    "IntDenRatio": "integrated density ratio",
    "CorrectedMeanRatio": "background-subtracted mean intensity ratio",
    "CorrectedAverageRatio": "background-subtracted mean intensity ratio",
    "CorrectedIntDenRatio": "background-subtracted integrated density ratio",
}


def _humanize_organelle_column(text: str) -> str:
    readable = _humanize_percell_column(text)
    if readable:
        return readable

    ratio_match = re.match(
        r"^(.+)_to_RestOfCell_(MeanRatio|IntDenRatio|CorrectedMeanRatio|CorrectedIntDenRatio)$",
        text,
    )
    if ratio_match:
        label, metric = ratio_match.groups()
        source, mask = _split_source_mask_prefix(label)
        label_text = f"{source} measured with {mask} mask" if source else label
        metric_name = _ORGANELLE_RATIO_METRIC_NAMES.get(metric, metric)
        return f"{label_text}: mask compared with rest of cell: {metric_name}"

    for pattern, suffix in _ORGANELLE_COLUMN_PATTERNS:
        match = re.match(pattern, text)
        if match:
            source, mask = _split_source_mask_prefix(match.group(1))
            label_text = f"{source} measured with {mask} mask" if source else match.group(1)
            return f"{label_text}: {suffix}"
    return ""


_MEASUREMENT_METRIC_ORDER = {
    "Area": 10,
    "Mean": 20,
    "Average": 20,
    "MeanGrayValue": 20,
    "StdDev": 22,
    "Mode": 23,
    "Min": 30,
    "Max": 40,
    "CentroidX": 41,
    "CentroidY": 42,
    "CenterOfMassX": 43,
    "CenterOfMassY": 44,
    "Perimeter": 45,
    "BoundingRectX": 46,
    "BoundingRectY": 47,
    "BoundingRectWidth": 48,
    "BoundingRectHeight": 49,
    "EllipseMajor": 50,
    "EllipseMinor": 51,
    "EllipseAngle": 52,
    "Feret": 53,
    "FeretX": 54,
    "FeretY": 55,
    "FeretAngle": 56,
    "MinFeret": 57,
    "Circularity": 58,
    "Solidity": 59,
    "RawIntDen": 60,
    "IntDen": 61,
    "Median": 62,
    "Skewness": 63,
    "Kurtosis": 64,
    "AspectRatio": 65,
    "Roundness": 66,
    "AreaFraction": 67,
}

_EXACT_COLUMN_ORDER = {
    "Label": 0,
    "Sample": 0,
    "SampleID": 1,
    "OriginalSamplePath": 2,
    "SourceImageLabel": 3,
    "SourceImageKey": 4,
    "SourceImageFile": 5,
    "CellposeMaskLabel": 6,
    "CellposeMaskKey": 7,
    "CellID": 10,
    "LiveFilter_CellID": 11,
}


# Keep measurements for related channels and masks next to each other.
def _measurement_sort_parts(column: str) -> tuple[str, str, int, str]:
    text = str(column or "")
    relation = re.match(
        rf"^(.+)_in_(.+)_({RELATION_METRIC_PATTERN})$",
        text,
    )
    rb_relation = re.match(r"^(.+)_in_(.+)_(RB[0-9p.]+)_IntDen$", text)
    if relation:
        measured, mask, metric = relation.groups()
    elif rb_relation:
        measured, mask, metric = rb_relation.groups()
    else:
        parsed = _split_percell_column(text)
        if parsed is not None:
            return parsed["measured"], parsed["region"], 70, text
        return "", "", 90, text

    if metric.startswith("RB"):
        order = 61
    else:
        order = _MEASUREMENT_METRIC_ORDER.get(metric, 80)
    return measured, mask, order, text


# Place identifiers and QC fields before measurements while preserving ties by original order.
def _column_sort_key(column: str, index: int) -> tuple[Any, ...]:
    text = str(column or "")
    if text in _EXACT_COLUMN_ORDER:
        return _EXACT_COLUMN_ORDER[text], index
    if text.endswith("_CellCount_TotalBeforeQC"):
        return 12, index
    if text.endswith("_CellCount"):
        return 13, index
    if "Excluded" in text or "OutOfRange" in text or "QC_" in text or "MaskQC_" in text:
        return 20, index
    if "Count" in text:
        return 31, index
    if "Area" in text:
        return 40, index
    if "Mean" in text or "Average" in text or "_Min" in text or "_Max" in text:
        return 50, index
    if "IntDen" in text:
        return 60, index
    if "Fraction" in text or "Ratio" in text:
        return 70, index
    return 80, index


# Use finer channel-mask-metric ordering when columns become rows in readable exports.
def _readable_row_sort_key(column: str, index: int) -> tuple[Any, ...]:
    text = str(column or "")
    if text == "SourceImageLabel":
        return 0, index
    if text == "SourceImageKey":
        return 1, index
    if text.endswith("_CellCount_TotalBeforeQC"):
        return 12, index
    if text.endswith("_CellCount"):
        return 13, index
    measured, mask, metric_order, fallback = _measurement_sort_parts(text)
    if measured or mask:
        return 40, measured.lower(), mask.lower(), metric_order, fallback.lower(), index
    return 100, index
