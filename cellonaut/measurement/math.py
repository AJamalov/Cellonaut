"""Per-cell aggregation and background-radius naming used by measurements and exports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, cast

import pandas as pd


def clean_name_or_none(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s if s else None


def base_name_no_ext(path: Path) -> str:
    """Return a filename stem with unsafe characters replaced by underscores."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in path.stem)


def summarize_per_cell_table(
    df: pd.DataFrame,
    organelle_label: str,
    source_label: Optional[str] = None,
    cell_mask_label: Optional[str] = None,
) -> Dict[str, float]:
    """Aggregate cell-only rows into one target's measurement columns.

    Means are unweighted means across cells, not pooled pixel means. Areas
    and integrated densities are summed. Missing/all-NaN measurements are
    omitted; an empty table with known columns yields zero totals but no means.
    Accept internal or exported MeanGrayValue names without changing the input.
    """
    if df is None:
        return {}

    # Saved CSVs expand Mean to MeanGrayValue; restore internal names for aggregation.
    df = df.rename(columns={col: str(col).replace("MeanGrayValue", "Mean") for col in df.columns
                            if "MeanGrayValue" in str(col) and str(col).replace("MeanGrayValue", "Mean") not in df.columns})
    summary: Dict[str, float] = {}
    source = str(source_label or "").strip()
    cell_mask = str(cell_mask_label or source_label or "").strip()
    organelle = str(organelle_label or "").strip()
    if source:
        cell_prefix = f"{source}_measured_with_{cell_mask}_cellpose_mask" if cell_mask else source
        mask_prefix = f"{source}_measured_with_{organelle}_mask" if organelle else source
        rest_prefix = (
            f"{source}_measured_with_{cell_mask}_cellpose_mask_minus_{organelle}_mask"
            if cell_mask and organelle
            else source
        )
    else:
        cell_prefix = cell_mask or organelle
        mask_prefix = organelle
        rest_prefix = f"{cell_mask}_cellpose_mask_minus_{organelle}_mask" if cell_mask and organelle else organelle

    # Missing optional measurements should omit a result rather than create a misleading zero.
    def _sum_if_present(src_col: str, out_key: str):
        if src_col in df.columns:
            values = cast(pd.Series, pd.to_numeric(pd.Series(df[src_col], index=df.index), errors="coerce"))
            total = 0.0 if df.empty else float(values.sum(min_count=1))
            if not math.isnan(total):
                summary[out_key] = total

    # Use numeric coercion because CSV-derived preview tables may store numbers as text.
    def _mean_if_present(src_col: str, out_key: str):
        if src_col in df.columns:
            values = cast(pd.Series, pd.to_numeric(pd.Series(df[src_col], index=df.index), errors="coerce"))
            mean = float(values.mean())
            if not math.isnan(mean):
                summary[out_key] = mean

    def _extreme_if_present(src_col: str, out_key: str, *, maximum: bool):
        if src_col in df.columns:
            values = cast(pd.Series, pd.to_numeric(pd.Series(df[src_col], index=df.index), errors="coerce"))
            value = float(values.max() if maximum else values.min())
            if not math.isnan(value):
                summary[out_key] = value

    # Ellipse orientation is axial: 1 degree and 179 degrees are close, not opposite.
    def _axial_mean_if_present(src_col: str, out_key: str):
        if src_col not in df.columns:
            return
        values = cast(pd.Series, pd.to_numeric(pd.Series(df[src_col], index=df.index), errors="coerce"))
        values = values.dropna()
        if values.empty:
            return
        doubled_radians = [math.radians(float(value) * 2.0) for value in values]
        mean_sine = sum(math.sin(value) for value in doubled_radians) / len(doubled_radians)
        mean_cosine = sum(math.cos(value) for value in doubled_radians) / len(doubled_radians)
        if math.isclose(mean_sine, 0.0, abs_tol=1e-15) and math.isclose(
            mean_cosine, 0.0, abs_tol=1e-15
        ):
            return
        summary[out_key] = (math.degrees(math.atan2(mean_sine, mean_cosine)) / 2.0) % 180.0

    _sum_if_present("CellArea", f"{cell_prefix}_PerCell_TotalCellArea")
    _sum_if_present("CellPerimeter", f"{cell_prefix}_PerCell_TotalCellPerimeter")
    _mean_if_present("CellMean", f"{cell_prefix}_PerCell_MeanOfCellMeans")
    _mean_if_present("CellStdDev", f"{cell_prefix}_PerCell_MeanOfCellStdDevs")
    _mean_if_present("CellMode", f"{cell_prefix}_PerCell_MeanOfCellModes")
    _extreme_if_present("CellMin", f"{cell_prefix}_PerCell_MinimumCellIntensity", maximum=False)
    _extreme_if_present("CellMax", f"{cell_prefix}_PerCell_MaximumCellIntensity", maximum=True)
    _mean_if_present("CellCentroidX", f"{cell_prefix}_PerCell_MeanCellCentroidX")
    _mean_if_present("CellCentroidY", f"{cell_prefix}_PerCell_MeanCellCentroidY")
    _mean_if_present("CellCenterOfMassX", f"{cell_prefix}_PerCell_MeanCellCenterOfMassX")
    _mean_if_present("CellCenterOfMassY", f"{cell_prefix}_PerCell_MeanCellCenterOfMassY")
    _mean_if_present("CellBoundingRectX", f"{cell_prefix}_PerCell_MeanCellBoundingRectX")
    _mean_if_present("CellBoundingRectY", f"{cell_prefix}_PerCell_MeanCellBoundingRectY")
    _mean_if_present("CellBoundingRectWidth", f"{cell_prefix}_PerCell_MeanCellBoundingRectWidth")
    _mean_if_present("CellBoundingRectHeight", f"{cell_prefix}_PerCell_MeanCellBoundingRectHeight")
    _mean_if_present("CellEllipseMajor", f"{cell_prefix}_PerCell_MeanCellEllipseMajor")
    _mean_if_present("CellEllipseMinor", f"{cell_prefix}_PerCell_MeanCellEllipseMinor")
    _axial_mean_if_present("CellEllipseAngle", f"{cell_prefix}_PerCell_MeanCellEllipseAngle")
    _mean_if_present("CellFeret", f"{cell_prefix}_PerCell_MeanCellFeretDiameter")
    _mean_if_present("CellCircularity", f"{cell_prefix}_PerCell_MeanCellCircularity")
    _mean_if_present("CellSolidity", f"{cell_prefix}_PerCell_MeanCellSolidity")
    _mean_if_present("CellMedian", f"{cell_prefix}_PerCell_MeanOfCellMedians")
    _sum_if_present("CellIntDen", f"{cell_prefix}_PerCell_SumCellIntDen")
    _mean_if_present("CellSkewness", f"{cell_prefix}_PerCell_MeanCellSkewness")
    _mean_if_present("CellKurtosis", f"{cell_prefix}_PerCell_MeanCellKurtosis")
    _mean_if_present("CellCorrectedMean", f"{cell_prefix}_PerCell_MeanOfCellCorrectedMeans")
    _sum_if_present("CellCorrectedIntDen", f"{cell_prefix}_PerCell_SumCellCorrectedIntDen")

    _sum_if_present(f"{organelle_label}Area_InCell", f"{mask_prefix}_PerCell_TotalMaskArea")
    _mean_if_present(f"{organelle_label}Mean_InCell", f"{mask_prefix}_PerCell_MeanOfMaskMeans")
    _mean_if_present(f"{organelle_label}StdDev_InCell", f"{mask_prefix}_PerCell_MeanOfMaskStdDevs")
    _mean_if_present(f"{organelle_label}Mode_InCell", f"{mask_prefix}_PerCell_MeanOfMaskModes")
    _extreme_if_present(
        f"{organelle_label}Min_InCell",
        f"{mask_prefix}_PerCell_MinimumMaskIntensity",
        maximum=False,
    )
    _extreme_if_present(
        f"{organelle_label}Max_InCell",
        f"{mask_prefix}_PerCell_MaximumMaskIntensity",
        maximum=True,
    )
    _mean_if_present(f"{organelle_label}Median_InCell", f"{mask_prefix}_PerCell_MeanOfMaskMedians")
    _mean_if_present(f"{organelle_label}CentroidX_InCell", f"{mask_prefix}_PerCell_MeanMaskCentroidX")
    _mean_if_present(f"{organelle_label}CentroidY_InCell", f"{mask_prefix}_PerCell_MeanMaskCentroidY")
    _mean_if_present(
        f"{organelle_label}CenterOfMassX_InCell",
        f"{mask_prefix}_PerCell_MeanMaskCenterOfMassX",
    )
    _mean_if_present(
        f"{organelle_label}CenterOfMassY_InCell",
        f"{mask_prefix}_PerCell_MeanMaskCenterOfMassY",
    )
    _sum_if_present(f"{organelle_label}Perimeter_InCell", f"{mask_prefix}_PerCell_TotalMaskPerimeter")
    _mean_if_present(
        f"{organelle_label}BoundingRectX_InCell",
        f"{mask_prefix}_PerCell_MeanMaskBoundingRectX",
    )
    _mean_if_present(
        f"{organelle_label}BoundingRectY_InCell",
        f"{mask_prefix}_PerCell_MeanMaskBoundingRectY",
    )
    _mean_if_present(
        f"{organelle_label}BoundingRectWidth_InCell",
        f"{mask_prefix}_PerCell_MeanMaskBoundingRectWidth",
    )
    _mean_if_present(
        f"{organelle_label}BoundingRectHeight_InCell",
        f"{mask_prefix}_PerCell_MeanMaskBoundingRectHeight",
    )
    _mean_if_present(
        f"{organelle_label}EllipseMajor_InCell",
        f"{mask_prefix}_PerCell_MeanMaskEllipseMajor",
    )
    _mean_if_present(
        f"{organelle_label}EllipseMinor_InCell",
        f"{mask_prefix}_PerCell_MeanMaskEllipseMinor",
    )
    _axial_mean_if_present(
        f"{organelle_label}EllipseAngle_InCell",
        f"{mask_prefix}_PerCell_MeanMaskEllipseAngle",
    )
    _mean_if_present(f"{organelle_label}Feret_InCell", f"{mask_prefix}_PerCell_MeanMaskFeretDiameter")
    _mean_if_present(
        f"{organelle_label}Circularity_InCell",
        f"{mask_prefix}_PerCell_MeanMaskCircularity",
    )
    _mean_if_present(f"{organelle_label}Solidity_InCell", f"{mask_prefix}_PerCell_MeanMaskSolidity")
    _sum_if_present(f"{organelle_label}IntDen_InCell", f"{mask_prefix}_PerCell_SumIntDen")
    _mean_if_present(f"{organelle_label}Skewness_InCell", f"{mask_prefix}_PerCell_MeanMaskSkewness")
    _mean_if_present(f"{organelle_label}Kurtosis_InCell", f"{mask_prefix}_PerCell_MeanMaskKurtosis")
    _mean_if_present(
        f"{organelle_label}CorrectedMean_InCell",
        f"{mask_prefix}_PerCell_MeanOfCorrectedMaskMeans",
    )
    _mean_if_present(
        f"{organelle_label}CorrectedStdDev_InCell",
        f"{mask_prefix}_PerCell_MeanOfCorrectedMaskStdDevs",
    )
    _extreme_if_present(
        f"{organelle_label}CorrectedMin_InCell",
        f"{mask_prefix}_PerCell_MinimumCorrectedMaskIntensity",
        maximum=False,
    )
    _extreme_if_present(
        f"{organelle_label}CorrectedMax_InCell",
        f"{mask_prefix}_PerCell_MaximumCorrectedMaskIntensity",
        maximum=True,
    )
    _mean_if_present(
        f"{organelle_label}CorrectedMedian_InCell",
        f"{mask_prefix}_PerCell_MeanOfCorrectedMaskMedians",
    )
    _sum_if_present(
        f"{organelle_label}CorrectedIntDen_InCell",
        f"{mask_prefix}_PerCell_SumCorrectedIntDen",
    )

    _sum_if_present("RestOfCellArea", f"{rest_prefix}_PerCell_TotalRestArea")
    _mean_if_present("RestOfCellMean", f"{rest_prefix}_PerCell_MeanOfRestMeans")
    _sum_if_present("RestOfCellIntDen", f"{rest_prefix}_PerCell_SumRestIntDen")
    _mean_if_present("RestOfCellCorrectedMean", f"{rest_prefix}_PerCell_MeanOfRestCorrectedMeans")
    _sum_if_present("RestOfCellCorrectedIntDen", f"{rest_prefix}_PerCell_SumRestCorrectedIntDen")

    return summary


# Accept common separators in saved settings while strict validation still reports malformed input.
def parse_radii_csv(csv: str, *, strict: bool = False, field_name: str = "Background radii") -> List[float]:
    if not csv:
        return []
    out = []
    normalized = csv.replace(";", ",").replace("\t", ",").replace(" ", ",")
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
            if not math.isfinite(v):
                raise ValueError(f"{field_name} must contain only finite numbers, got: {part!r}")
            if v < 0:
                raise ValueError(f"{field_name} values must be >= 0, got: {part!r}")
            out.append(v)
        except ValueError:
            if strict:
                raise ValueError(
                    f"{field_name} must contain non-negative numbers separated by commas, got: {part!r}"
                ) from None
    return out


# Encode decimal radii without dots so generated measurement columns remain easy to parse.
def rad_tag(r: float) -> str:
    s = str(r).replace(".", "p")
    s = "".join(ch for ch in s if ch.isdigit() or ch == "p")
    if not s:
        s = "0"
    return f"RB{s}"
