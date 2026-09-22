from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cellonaut.measurement.math import (
    base_name_no_ext,
    clean_name_or_none,
    parse_radii_csv,
    rad_tag,
    summarize_per_cell_table,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), (" \t ", None), (" GFP ", "GFP")],
)
def test_clean_name_or_none_normalizes_optional_names(value, expected):
    assert clean_name_or_none(value) == expected


def test_measurement_math_helpers_keep_output_names_stable():
    assert base_name_no_ext(Path("model with spaces!.model")) == "model_with_spaces_"
    assert rad_tag(12.5) == "RB12p5"


@pytest.mark.parametrize(
    ("radius", "expected"),
    [(0, "RB0"), (-1.5, "RB1p5"), (float("inf"), "RB0"), (1e-05, "RB105")],
)
def test_rad_tag_produces_filename_safe_tags(radius: float, expected: str):
    assert rad_tag(radius) == expected


def test_parse_radii_csv_accepts_common_separators():
    assert parse_radii_csv("10, 20;30\t40 bad -1") == [10.0, 20.0, 30.0, 40.0]


@pytest.mark.parametrize("value", ["", ", ; \t", "bad,-2,nan,inf"])
def test_parse_radii_csv_non_strict_mode_omits_empty_or_invalid_values(value: str):
    assert parse_radii_csv(value) == []


def test_parse_radii_csv_preserves_zero_decimals_and_scientific_notation():
    assert parse_radii_csv("0, 2.5, 1e2", strict=True) == [0.0, 2.5, 100.0]


@pytest.mark.parametrize("value", ["10,bad", "-1", "NaN", "inf"])
def test_parse_radii_csv_strict_mode_rejects_invalid_values(value: str):
    with pytest.raises(ValueError, match="Background radii"):
        parse_radii_csv(value, strict=True)


def test_parse_radii_csv_strict_error_uses_configured_field_name():
    with pytest.raises(ValueError, match="Annulus sizes"):
        parse_radii_csv("invalid", strict=True, field_name="Annulus sizes")


@pytest.mark.parametrize("table", [None, pd.DataFrame()])
def test_summarize_per_cell_table_returns_empty_result_for_missing_rows(table):
    assert summarize_per_cell_table(table, "Mask", "Signal") == {}


def test_summarize_per_cell_table_totals():
    if getattr(pd, "__cellonaut_stub__", False):
        pytest.skip("pandas is stubbed in this lightweight test environment")

    df = pd.DataFrame(
        {
            "CellArea": [10, 20],
            "CellMean": [2, 4],
            "CellIntDen": [20, 80],
            "mCherryArea_InCell": [3, 5],
        }
    )

    summary = summarize_per_cell_table(df, "mCherry")

    assert summary["mCherry_PerCell_TotalCellArea"] == 30
    assert summary["mCherry_PerCell_MeanOfCellMeans"] == 3

    measured_summary = summarize_per_cell_table(df, "mCherry", "GFP")
    assert measured_summary["GFP_measured_with_GFP_cellpose_mask_PerCell_TotalCellArea"] == 30


def test_summarize_per_cell_table_includes_new_base_measurements():
    df = pd.DataFrame(
        {
            "CellArea": [2, 3],
            "CellPerimeter": [4, 6],
            "CellMin": [2, 1],
            "CellMax": [8, 12],
            "CellMedian": [5, 7],
            "MaskStdDev_InCell": [1, 3],
            "MaskMin_InCell": [4, 2],
            "MaskMax_InCell": [8, 10],
            "MaskMedian_InCell": [6, 8],
        }
    )

    summary = summarize_per_cell_table(df, "Mask", "GFP", "DIA")

    cell_prefix = "GFP_measured_with_DIA_cellpose_mask_PerCell"
    mask_prefix = "GFP_measured_with_Mask_mask_PerCell"
    assert summary[f"{cell_prefix}_TotalCellPerimeter"] == 10
    assert summary[f"{cell_prefix}_MinimumCellIntensity"] == 1
    assert summary[f"{cell_prefix}_MaximumCellIntensity"] == 12
    assert summary[f"{cell_prefix}_MeanOfCellMedians"] == 6
    assert summary[f"{mask_prefix}_MeanOfMaskStdDevs"] == 2
    assert summary[f"{mask_prefix}_MinimumMaskIntensity"] == 2
    assert summary[f"{mask_prefix}_MaximumMaskIntensity"] == 10
    assert summary[f"{mask_prefix}_MeanOfMaskMedians"] == 7


def test_summarize_per_cell_table_includes_extended_cellpose_measurements():
    df = pd.DataFrame(
        {
            "CellStdDev": [1, 3],
            "CellMode": [2, 4],
            "CellCentroidX": [10, 20],
            "CellCentroidY": [30, 40],
            "CellCenterOfMassX": [11, 21],
            "CellCenterOfMassY": [31, 41],
            "CellBoundingRectX": [8, 18],
            "CellBoundingRectY": [28, 38],
            "CellBoundingRectWidth": [4, 6],
            "CellBoundingRectHeight": [5, 7],
            "CellEllipseMajor": [6, 8],
            "CellEllipseMinor": [3, 5],
            "CellEllipseAngle": [20, 40],
            "CellFeret": [7, 9],
            "CellCircularity": [0.7, 0.9],
            "CellSolidity": [0.8, 1.0],
            "CellSkewness": [-1, 1],
            "CellKurtosis": [2, 4],
        }
    )

    summary = summarize_per_cell_table(df, "", "GFP", "DIA")
    prefix = "GFP_measured_with_DIA_cellpose_mask_PerCell"

    assert summary == pytest.approx({
        f"{prefix}_MeanOfCellStdDevs": 2.0,
        f"{prefix}_MeanOfCellModes": 3.0,
        f"{prefix}_MeanCellCentroidX": 15.0,
        f"{prefix}_MeanCellCentroidY": 35.0,
        f"{prefix}_MeanCellCenterOfMassX": 16.0,
        f"{prefix}_MeanCellCenterOfMassY": 36.0,
        f"{prefix}_MeanCellBoundingRectX": 13.0,
        f"{prefix}_MeanCellBoundingRectY": 33.0,
        f"{prefix}_MeanCellBoundingRectWidth": 5.0,
        f"{prefix}_MeanCellBoundingRectHeight": 6.0,
        f"{prefix}_MeanCellEllipseMajor": 7.0,
        f"{prefix}_MeanCellEllipseMinor": 4.0,
        f"{prefix}_MeanCellEllipseAngle": 30.0,
        f"{prefix}_MeanCellFeretDiameter": 8.0,
        f"{prefix}_MeanCellCircularity": 0.8,
        f"{prefix}_MeanCellSolidity": 0.9,
        f"{prefix}_MeanCellSkewness": 0.0,
        f"{prefix}_MeanCellKurtosis": 3.0,
    })


def test_summarize_per_cell_table_uses_axial_mean_for_ellipse_angles():
    df = pd.DataFrame({"CellEllipseAngle": [1.0, 179.0]})

    summary = summarize_per_cell_table(df, "", "GFP", "DIA")

    key = "GFP_measured_with_DIA_cellpose_mask_PerCell_MeanCellEllipseAngle"
    assert summary[key] == pytest.approx(0.0, abs=1e-12)


def test_summarize_per_cell_table_omits_all_missing_numeric_results():
    df = pd.DataFrame(
        {
            "CellArea": [None, "not measured"],
            "CellMean": [None, "not measured"],
            "CellIntDen": [5, None],
        }
    )

    summary = summarize_per_cell_table(df, "Mask", "Signal")

    assert "Signal_measured_with_Signal_cellpose_mask_PerCell_TotalCellArea" not in summary
    assert "Signal_measured_with_Signal_cellpose_mask_PerCell_MeanOfCellMeans" not in summary
    assert summary["Signal_measured_with_Signal_cellpose_mask_PerCell_SumCellIntDen"] == 5


def test_summarize_per_cell_table_emits_complete_exact_measurement_contract():
    df = pd.DataFrame(
        {
            "CellArea": [10, "20"],
            "CellPerimeter": [4, 6],
            "CellMean": [2, "4"],
            "CellMin": [2, 1],
            "CellMax": [8, 12],
            "CellMedian": [5, 7],
            "CellIntDen": [20, 80],
            "CellCorrectedMean": [1.5, 3.5],
            "CellCorrectedIntDen": [15, 70],
            "PunctaArea_InCell": [3, 5],
            "PunctaMean_InCell": [4, 6],
            "PunctaStdDev_InCell": [1, 3],
            "PunctaMin_InCell": [4, 2],
            "PunctaMax_InCell": [8, 10],
            "PunctaMedian_InCell": [6, 8],
            "PunctaIntDen_InCell": [12, 30],
            "PunctaCorrectedMean_InCell": [3, 5],
            "PunctaCorrectedStdDev_InCell": [0.5, 1.5],
            "PunctaCorrectedMin_InCell": [3, 1],
            "PunctaCorrectedMax_InCell": [7, 9],
            "PunctaCorrectedMedian_InCell": [5, 7],
            "PunctaCorrectedIntDen_InCell": [9, 25],
            "RestOfCellArea": [7, 15],
            "RestOfCellMean": [1, 3],
            "RestOfCellIntDen": [8, 50],
            "RestOfCellCorrectedMean": [0.5, 2.5],
            "RestOfCellCorrectedIntDen": [5, 45],
        }
    )

    assert summarize_per_cell_table(df, "Puncta", "GFP", "DIA") == {
        "GFP_measured_with_DIA_cellpose_mask_PerCell_TotalCellArea": 30.0,
        "GFP_measured_with_DIA_cellpose_mask_PerCell_TotalCellPerimeter": 10.0,
        "GFP_measured_with_DIA_cellpose_mask_PerCell_MeanOfCellMeans": 3.0,
        "GFP_measured_with_DIA_cellpose_mask_PerCell_MinimumCellIntensity": 1.0,
        "GFP_measured_with_DIA_cellpose_mask_PerCell_MaximumCellIntensity": 12.0,
        "GFP_measured_with_DIA_cellpose_mask_PerCell_MeanOfCellMedians": 6.0,
        "GFP_measured_with_DIA_cellpose_mask_PerCell_SumCellIntDen": 100.0,
        "GFP_measured_with_DIA_cellpose_mask_PerCell_MeanOfCellCorrectedMeans": 2.5,
        "GFP_measured_with_DIA_cellpose_mask_PerCell_SumCellCorrectedIntDen": 85.0,
        "GFP_measured_with_Puncta_mask_PerCell_TotalMaskArea": 8.0,
        "GFP_measured_with_Puncta_mask_PerCell_MeanOfMaskMeans": 5.0,
        "GFP_measured_with_Puncta_mask_PerCell_MeanOfMaskStdDevs": 2.0,
        "GFP_measured_with_Puncta_mask_PerCell_MinimumMaskIntensity": 2.0,
        "GFP_measured_with_Puncta_mask_PerCell_MaximumMaskIntensity": 10.0,
        "GFP_measured_with_Puncta_mask_PerCell_MeanOfMaskMedians": 7.0,
        "GFP_measured_with_Puncta_mask_PerCell_SumIntDen": 42.0,
        "GFP_measured_with_Puncta_mask_PerCell_MeanOfCorrectedMaskMeans": 4.0,
        "GFP_measured_with_Puncta_mask_PerCell_MeanOfCorrectedMaskStdDevs": 1.0,
        "GFP_measured_with_Puncta_mask_PerCell_MinimumCorrectedMaskIntensity": 1.0,
        "GFP_measured_with_Puncta_mask_PerCell_MaximumCorrectedMaskIntensity": 9.0,
        "GFP_measured_with_Puncta_mask_PerCell_MeanOfCorrectedMaskMedians": 6.0,
        "GFP_measured_with_Puncta_mask_PerCell_SumCorrectedIntDen": 34.0,
        "GFP_measured_with_DIA_cellpose_mask_minus_Puncta_mask_PerCell_TotalRestArea": 22.0,
        "GFP_measured_with_DIA_cellpose_mask_minus_Puncta_mask_PerCell_MeanOfRestMeans": 2.0,
        "GFP_measured_with_DIA_cellpose_mask_minus_Puncta_mask_PerCell_SumRestIntDen": 58.0,
        "GFP_measured_with_DIA_cellpose_mask_minus_Puncta_mask_PerCell_MeanOfRestCorrectedMeans": 1.5,
        "GFP_measured_with_DIA_cellpose_mask_minus_Puncta_mask_PerCell_SumRestCorrectedIntDen": 50.0,
    }


def test_summarize_per_cell_table_accepts_saved_mean_gray_value_columns():
    saved_table = pd.DataFrame(
        {
            "CellMeanGrayValue": [2, 4],
            "PunctaMeanGrayValue_InCell": [3, 7],
        }
    )

    assert summarize_per_cell_table(saved_table, "Puncta", "GFP", "DIA") == {
        "GFP_measured_with_DIA_cellpose_mask_PerCell_MeanOfCellMeans": 3.0,
        "GFP_measured_with_Puncta_mask_PerCell_MeanOfMaskMeans": 5.0,
    }


def test_summarize_per_cell_table_without_source_uses_explicit_cell_mask_prefix():
    df = pd.DataFrame({"CellArea": [2], "PunctaArea_InCell": [1], "RestOfCellArea": [1]})

    assert summarize_per_cell_table(df, "Puncta", cell_mask_label="DIA") == {
        "DIA_PerCell_TotalCellArea": 2.0,
        "Puncta_PerCell_TotalMaskArea": 1.0,
        "DIA_cellpose_mask_minus_Puncta_mask_PerCell_TotalRestArea": 1.0,
    }
