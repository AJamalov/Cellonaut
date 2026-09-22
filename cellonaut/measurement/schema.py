"""Shared semantics for generated measurement-table columns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CountColumnSpec:
    suffix: str
    readable_suffix: str
    region: str
    description: str
    guidance: str


@dataclass(frozen=True, slots=True)
class MetricSpec:
    key: str
    readable_suffix: str
    display_name: str
    description: str

    @property
    def suffix(self) -> str:
        return f"_{self.key}"


COUNT_COLUMN_SPECS = (
    CountColumnSpec(
        "_CellCount_TotalBeforeQC",
        ": total detected cell count",
        "Cellpose mask",
        "Total detected cells before filters are applied.",
        "This is the starting number of detected cells for this measured channel.",
    ),
    CountColumnSpec(
        "_CellQC_OutOfRangeCount",
        ": cells outside the saved cell-filter limits",
        "Cellpose mask",
        "Cells outside the saved cell-filter limits; not a cell-group membership count.",
        "Interpret this diagnostic with the saved filter settings; group membership alone does not imply exclusion.",
    ),
    CountColumnSpec(
        "_MaskQC_OutOfRangeCount",
        ": cells outside the saved mask-filter limits",
        "Configured mask within Cellpose cells",
        "Cells outside the saved mask-filter limits; not a cell-group membership count.",
        "Interpret this diagnostic with the saved filter settings; group membership alone does not imply exclusion.",
    ),
    CountColumnSpec(
        "_QC_ExcludedCount",
        ": cells excluded from filtered measurements",
        "Cellpose mask and assigned configured mask",
        "Cells excluded from this export by its saved exclusion rules.",
        "Interpret this diagnostic with the saved filter settings; group membership alone does not imply exclusion.",
    ),
    CountColumnSpec(
        "_CellQC_ExcludedCount",
        ": cells excluded from filtered measurements",
        "Cellpose mask and assigned configured mask",
        "Cells excluded from filtered per-cell measurements.",
        "Interpret this diagnostic with the saved filter settings; group membership alone does not imply exclusion.",
    ),
    CountColumnSpec(
        "_CellCount",
        ": retained cell count",
        "Cellpose mask",
        "Cells retained in this table after any explicit CSV exclusions.",
        "Original pipeline tables retain all detected cells; filtered exports retain cells not selected for exclusion.",
    ),
)


METRIC_SPECS = (
    MetricSpec(
        "BoundingRectWidth", "bounding rectangle width", "Bounding rectangle width", "Bounding rectangle width."
    ),
    MetricSpec(
        "BoundingRectHeight", "bounding rectangle height", "Bounding rectangle height", "Bounding rectangle height."
    ),
    MetricSpec("BoundingRectX", "bounding rectangle X", "Bounding rectangle X", "Bounding rectangle X position."),
    MetricSpec("BoundingRectY", "bounding rectangle Y", "Bounding rectangle Y", "Bounding rectangle Y position."),
    MetricSpec("CenterOfMassX", "center of mass X", "Center of mass X", "Center of mass X position."),
    MetricSpec("CenterOfMassY", "center of mass Y", "Center of mass Y", "Center of mass Y position."),
    MetricSpec("EllipseMajor", "fitted ellipse major axis", "Fitted ellipse major axis", "Fitted ellipse major axis."),
    MetricSpec("EllipseMinor", "fitted ellipse minor axis", "Fitted ellipse minor axis", "Fitted ellipse minor axis."),
    MetricSpec("EllipseAngle", "fitted ellipse angle", "Fitted ellipse angle", "Fitted ellipse angle."),
    MetricSpec("AspectRatio", "aspect ratio", "Aspect ratio", "Aspect ratio."),
    MetricSpec("AreaFraction", "area fraction", "Area fraction", "Area fraction."),
    MetricSpec("MeanGrayValue", "mean intensity", "Mean intensity", "Mean intensity."),
    MetricSpec(
        "Corrected_IntDen",
        "integrated density (background subtracted)",
        "Integrated density after background subtraction",
        "Integrated density after background subtraction.",
    ),
    MetricSpec(
        "CorrectedIntDen",
        "integrated density (background subtracted)",
        "Integrated density after background subtraction",
        "Integrated density after background subtraction.",
    ),
    MetricSpec("RawIntDen", "raw integrated density", "Raw integrated density", "Raw integrated density."),
    MetricSpec("FeretAngle", "Feret angle", "Feret angle", "Feret angle."),
    MetricSpec("MinFeret", "minimum Feret diameter", "Minimum Feret diameter", "Minimum Feret diameter."),
    MetricSpec("CentroidX", "centroid X", "Centroid X", "Centroid X position."),
    MetricSpec("CentroidY", "centroid Y", "Centroid Y", "Centroid Y position."),
    MetricSpec("Circularity", "circularity", "Circularity", "Circularity."),
    MetricSpec("Perimeter", "perimeter", "Perimeter", "Perimeter."),
    MetricSpec("Roundness", "roundness", "Roundness", "Roundness."),
    MetricSpec("Skewness", "intensity skewness", "Intensity skewness", "Intensity skewness."),
    MetricSpec("Kurtosis", "intensity kurtosis", "Intensity kurtosis", "Intensity kurtosis."),
    MetricSpec("Solidity", "solidity", "Solidity", "Solidity."),
    MetricSpec(
        "StdDev", "intensity standard deviation", "Intensity standard deviation", "Intensity standard deviation."
    ),
    MetricSpec("FeretX", "Feret X", "Feret X", "Feret X position."),
    MetricSpec("FeretY", "Feret Y", "Feret Y", "Feret Y position."),
    MetricSpec("Feret", "Feret diameter", "Feret diameter", "Feret diameter."),
    MetricSpec("Median", "median intensity", "Median intensity", "Median intensity."),
    MetricSpec("Mode", "modal gray value", "Modal gray value", "Modal gray value."),
    MetricSpec("IntDen", "integrated density", "Integrated density", "Integrated density."),
    MetricSpec("Area", "area (px²)", "Area (px²)", "Area in pixels squared."),
    MetricSpec("Average", "mean intensity", "Mean intensity", "Mean intensity."),
    MetricSpec("Mean", "mean intensity", "Mean intensity", "Mean intensity."),
    MetricSpec("Min", "minimum intensity", "Minimum", "Minimum intensity."),
    MetricSpec("Max", "maximum intensity", "Maximum", "Maximum intensity."),
)


METRIC_SPECS_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}
RELATION_METRIC_PATTERN = "|".join(
    spec.key for spec in METRIC_SPECS if spec.key not in {"Corrected_IntDen", "CorrectedIntDen"}
)


def match_count_column(column: str) -> tuple[str, CountColumnSpec] | None:
    for spec in COUNT_COLUMN_SPECS:
        if column.endswith(spec.suffix):
            return column[: -len(spec.suffix)], spec
    return None


def match_metric_column(column: str) -> tuple[str, MetricSpec] | None:
    for spec in METRIC_SPECS:
        if column.endswith(spec.suffix):
            return column[: -len(spec.suffix)], spec
    return None


def measurement_unit(column: str) -> str:
    """Return the analysis unit for an exported measurement column."""

    text = str(column or "")
    lower = text.casefold()
    if "cellcount" in lower or lower.endswith(("count", "_count")):
        return "count"
    if "angle" in lower:
        return "degrees"
    if any(
        marker in lower
        for marker in (
            "area_fraction",
            "areafraction",
            "fractionof",
            "fraction_of",
            "intdenratio",
            "meanratio",
            "averageratio",
        )
    ):
        return "ratio"
    if "intdenpercellarea" in lower:
        return "a.u."
    if any(marker in lower for marker in ("totalcellarea", "totalmaskarea", "totalrestarea", "area_incell")):
        return "px²"
    if lower.endswith("_area") or lower.endswith("cellarea"):
        return "px²"
    if any(
        marker in lower
        for marker in (
            "perimeter",
            "feret",
            "boundingrectwidth",
            "boundingrectheight",
            "boundingrectx",
            "boundingrecty",
            "centroidx",
            "centroidy",
            "centerofmassx",
            "centerofmassy",
            "ellipsemajor",
            "ellipseminor",
        )
    ):
        return "px"
    if any(marker in lower for marker in ("circularity", "solidity", "roundness", "aspectratio", "skewness", "kurtosis")):
        return "unitless"
    if "intden" in lower:
        return "a.u. × px²"
    if any(
        marker in lower
        for marker in (
            "meangrayvalue", "meanof", "averageof", "correctedmean", "correctedaverage",
            "median", "mode", "minimum", "maximum", "stddev", "min_incell", "max_incell",
        )
    ):
        return "a.u."
    if lower.endswith(("_mean", "_average", "_min", "_max", "cellmean", "cellmin", "cellmax", "cellmedian")):
        return "a.u."
    return ""
