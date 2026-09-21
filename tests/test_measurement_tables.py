from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

if getattr(pd, "__cellonaut_stub__", False):
    pytest.skip("Real pandas is not installed in this test environment.", allow_module_level=True)

# Import application modules only after confirming the real dataframe dependency is available.
from cellonaut.config.defaults import INPUT_STRUCTURE_GROUPED_BY_PROTEIN  # noqa: E402
from cellonaut.measurement.tables import (  # noqa: E402
    add_per_folder_average_rows,
    add_summary_mean_row,
    human_readable_column_name,
    normalize_export_table,
    row_matches_exclusion_tag,
)
from cellonaut.measurement.table_formatting import (  # noqa: E402
    measurement_summary_table,
    readable_transposed_table,
    simple_measurements_table,
    tidy_measurements_table,
)
from cellonaut.measurement.schema import measurement_unit  # noqa: E402
from cellonaut.measurement.table_cleanup import drop_derived_ratio_columns  # noqa: E402
from cellonaut.results.labels import is_group_average_label, is_summary_label  # noqa: E402


def test_summary_label_markers_do_not_match_ordinary_samples():
    assert is_summary_label("SUMMARY_MEAN") is True
    assert is_group_average_label("GroupA_AVERAGE") is True
    assert is_summary_label("GroupA_AVERAGE") is True
    assert is_summary_label("Sample_A") is False
    assert is_summary_label(None) is False


def test_blank_exclusion_tag_does_not_exclude_every_summary_input():
    df = pd.DataFrame(
        {
            "Label": ["Sample_A", "Sample_B"],
            "Signal_Mean": [2.0, 6.0],
        }
    )

    result = add_summary_mean_row(df, "SUMMARY_MEAN", exclude_matching_exclusion_tag=True, exclusion_tag="")

    summary = result[result["Label"] == "SUMMARY_MEAN"].iloc[0]
    assert summary["Signal_Mean"] == 4.0
    assert row_matches_exclusion_tag("Sample_A", "") is False


def test_blank_exclusion_tag_keeps_grouped_folder_averages():
    df = pd.DataFrame(
        {
            "Label": ["GroupA_cell1", "GroupA_cell2"],
            "Signal_Mean": [2.0, 6.0],
        }
    )

    result = add_per_folder_average_rows(df, INPUT_STRUCTURE_GROUPED_BY_PROTEIN, exclusion_tag="")

    avg = result[result["Label"] == "GroupA_AVERAGE"].iloc[0]
    assert avg["Signal_Mean"] == 4.0


def test_simple_measurements_table_drops_filter_and_path_columns():
    df = pd.DataFrame(
        {
            "Label": [r"C:\data\Sample_001.ome.tif"],
            "SourceImageFile": [r"C:\data\Sample_001.ome.tif"],
            "Signal_Mean": [42.0],
            "Signal_QC_ExcludedCount": [3],
            "Signal_CellCount": [12],
            "Empty": [""],
        }
    )

    result = simple_measurements_table(df)

    assert result.columns.tolist() == ["Label", "Signal_Mean"]
    assert result["Label"].tolist() == ["Sample_001.ome.tif"]


def test_tidy_measurements_are_analysis_ready_with_explicit_units():
    table = pd.DataFrame(
        {
            "Sample": ["sample_a.tif", "sample_b.tif"],
            "GFP_in_Nucleus_Area": [10.0, 20.0],
            "GFP_in_Nucleus_MeanGrayValue": [4.0, 8.0],
            "GFP_QC_ExcludedCount": [1, 0],
        }
    )

    tidy = tidy_measurements_table(table)

    assert tidy.columns.tolist() == [
        "SampleID",
        "Sample",
        "SampleGroup",
        "MeasuredChannel",
        "Measurement",
        "MeasurementLabel",
        "Value",
        "Unit",
    ]
    assert tidy["SampleID"].tolist() == ["sample_a.tif", "sample_a.tif", "sample_b.tif", "sample_b.tif"]
    assert tidy["Sample"].tolist() == ["sample_a.tif", "sample_a.tif", "sample_b.tif", "sample_b.tif"]
    assert tidy["Unit"].tolist() == ["px²", "a.u.", "px²", "a.u."]
    assert "GFP_QC_ExcludedCount" not in tidy["Measurement"].tolist()


def test_measurement_summary_keeps_statistics_separate_from_samples():
    tidy = pd.DataFrame(
        {
            "SampleID": ["Control_1", "Control_2"],
            "SampleGroup": ["Control", "Control"],
            "Sample": ["Control_1", "Control_2"],
            "Measurement": ["GFP_MeanGrayValue", "GFP_MeanGrayValue"],
            "MeasurementLabel": ["GFP: mean intensity", "GFP: mean intensity"],
            "Value": [2.0, 6.0],
            "Unit": ["a.u.", "a.u."],
        }
    )

    summary = measurement_summary_table(tidy, grouped_samples=True)

    assert summary.loc[0, "Group"] == "Control"
    assert summary.loc[0, "N"] == 2
    assert summary.loc[0, "Mean"] == 4.0
    assert summary.loc[0, "Median"] == 4.0
    assert summary.loc[0, "Min"] == 2.0
    assert summary.loc[0, "Max"] == 6.0


def test_measurement_units_cover_geometry_intensity_and_ratios():
    assert measurement_unit("Mask_Area") == "px²"
    assert measurement_unit("Mask_Perimeter") == "px"
    assert measurement_unit("Mask_EllipseAngle") == "degrees"
    assert measurement_unit("Mask_MeanGrayValue") == "a.u."
    assert measurement_unit("Mask_RawIntDen") == "a.u. × px²"
    assert measurement_unit("Mask_IntDenPerCellArea") == "a.u."
    assert measurement_unit("Mask_AreaFraction") == "ratio"
    assert measurement_unit("Mask_Skewness") == "unitless"
    assert measurement_unit("Mask_Kurtosis") == "unitless"


def test_derived_ratios_are_omitted_from_measurement_exports():
    table = pd.DataFrame({
        "CellID": [1],
        "MaskArea_InCell": [5],
        "MaskAreaFraction_InCell": [0.5],
        "Circularity": [0.8],
        "MaskIntDenPerCellArea": [10],
        "Mask_to_RestOfCell_MeanRatio": [2],
    })

    assert list(drop_derived_ratio_columns(table).columns) == ["CellID", "MaskArea_InCell"]


def test_human_readable_skeleton_junction_column():
    from cellonaut.measurement.tables import human_readable_column_name

    assert human_readable_column_name("Tubules_MaskSkeleton_JunctionCount") == "Tubules mask skeleton: junction count (count)"


def test_human_readable_names_cover_fiji_measurements():
    assert human_readable_column_name("GFP_in_Mask_StdDev") == "GFP(Mask) : Intensity standard deviation (a.u.)"
    assert human_readable_column_name("GFP_in_Mask_BoundingRectWidth") == "GFP(Mask) : Bounding rectangle width (px)"
    assert human_readable_column_name("GFP_in_Mask_FeretAngle") == "GFP(Mask) : Feret angle (degrees)"
    assert human_readable_column_name("GFP_in_Mask_AreaFraction") == "GFP(Mask) : Area fraction (ratio)"


@pytest.mark.parametrize(
    ("column", "readable", "description"),
    [
        (
            "GFP_CellCount_TotalBeforeQC",
            "GFP: total detected cell count (count)",
            "Total detected cells before filters are applied.",
        ),
        ("GFP_StdDev", "GFP: intensity standard deviation (a.u.)", "Intensity standard deviation."),
        ("GFP_BoundingRectWidth", "GFP: bounding rectangle width (px)", "Bounding rectangle width."),
        ("GFP_RawIntDen", "GFP: raw integrated density (a.u. × px²)", "Raw integrated density."),
    ],
)
def test_measurement_schema_drives_readable_names_and_descriptions(column, readable, description):
    assert human_readable_column_name(column) == readable
    from cellonaut.measurement.schema import match_count_column, match_metric_column

    match = match_count_column(column) or match_metric_column(column)
    assert match is not None
    assert match[1].description == description


def test_normalize_export_table_uses_sample_path_and_removes_redundant_metadata():
    df = pd.DataFrame(
        {
            "Label": ["Sample_001", "SUMMARY_MEAN"],
            "SampleID": ["Sample_001", ""],
            "OriginalSamplePath": [r"C:\data\Sample_001.ome.tif", ""],
            "SourceImageKey": ["image1", ""],
            "SourceImageLabel": ["GFP", ""],
            "SourceImageFile": [r"C:\data\Sample_001.ome.tif", ""],
            "Signal_Mean": [10.0, 10.0],
        }
    )

    result = normalize_export_table(df)

    assert result.columns.tolist() == ["Sample", "Signal_MeanGrayValue"]
    assert result["Sample"].tolist() == ["Sample_001.ome.tif", "Average"]


def test_readable_transposed_table_puts_measurements_in_rows():
    df = pd.DataFrame(
        {
            "Label": ["Sample_001.ome.tif", "Sample_002.ome.tif"],
            "Signal_Mean": [10, 20],
            "Signal_QC_ExcludedCount": [1, 2],
        }
    )

    result = readable_transposed_table(df)

    assert result.columns.tolist() == ["Measurement", "Sample_001.ome.tif", "Sample_002.ome.tif"]
    assert result["Measurement"].tolist() == [
        "Signal: mean intensity (a.u.)",
        "Signal: cells excluded from filtered measurements (count)",
    ]


def test_long_format_groups_by_measured_image_mask_and_rolling_ball_radius():
    df = pd.DataFrame(
        {
            "Sample": ["Sample_001", "Sample_002"],
            "GFP_in_Mask2_Area": [1, 2],
            "GFP_in_Mask1_RB40p0_IntDen": [3, 4],
            "GFP_in_Mask1_MeanGrayValue": [5, 6],
            "GFP_in_Mask1_Area": [7, 8],
            "mCherry_in_Mask1_Area": [9, 10],
        }
    )

    result = readable_transposed_table(df, sample_column="Sample")

    assert result["Measurement"].tolist() == [
        "GFP(Mask1) : Area (px²)",
        "GFP(Mask1) : Mean intensity (a.u.)",
        "GFP(Mask1) : integrated density (40px Subtract Background) (a.u. × px²)",
        "GFP(Mask2) : Area (px²)",
        "mCherry(Mask1) : Area (px²)",
    ]


def test_cell_totals_readable_names_are_short():
    df = pd.DataFrame(
        {
            "Label": ["Sample_001"],
            "Signal_CellCount": [12],
            "Signal_CellCount_TotalBeforeQC": [15],
        }
    )

    result = readable_transposed_table(df)

    assert result["Measurement"].tolist() == [
        "Signal: total detected cell count (count)",
        "Signal: retained cell count (count)",
    ]


def test_readable_transposed_table_puts_measured_channel_first():
    df = pd.DataFrame(
        {
            "Label": ["Sample_001"],
            "Signal_Mean": [10],
            "SourceImageLabel": ["GFP"],
        }
    )

    result = readable_transposed_table(df)

    assert result["Measurement"].tolist() == [
        "Measured channel",
        "Signal: mean intensity (a.u.)",
    ]
