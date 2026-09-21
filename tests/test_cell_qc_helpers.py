from __future__ import annotations

from typing import Any, cast

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

if getattr(pd, "__cellonaut_stub__", False) or getattr(np, "__cellonaut_stub__", False):
    pytest.skip("Real numpy/pandas are not installed in this test environment.", allow_module_level=True)

# Import application modules only after confirming real scientific dependencies are available.
from cellonaut.masks.preview_filter_overlay import attach_live_mask_metrics  # noqa: E402
from cellonaut.masks.cell_qc import (  # noqa: E402
    append_cell_qc_summary_rows,
    cell_label_series,
    cell_label_outline_mask,
    cell_qc_rules_to_text,
    combine_qc_label_sets,
    evaluate_cell_qc_table,
    find_matching_cell_qc_column,
    make_mask_qc_metric_table,
    make_mask_for_cell_labels,
    normalize_cell_qc_rules,
)
from cellonaut.cell_segmentation.core import make_per_cell_table  # noqa: E402


def test_evaluate_cell_qc_table_flags_bounds_and_missing_metrics():
    cells = pd.DataFrame(
        {
            "CellID": [1, 2, 3],
            "Area": [10, 20, 35],
            "Mean": [2.0, 5.0, 12.0],
        }
    )

    qc_table, flagged_labels, summary = evaluate_cell_qc_table(
        cells,
        {
            "Area": {"min": 12, "max": 30},
            "MissingMetric": {"max": 1},
        },
    )

    assert flagged_labels == {1, 3}
    assert summary["flagged_count"] == 2
    assert summary["missing_metrics"] == ["MissingMetric"]
    assert qc_table["QC_OutOfRange"].tolist() == [True, False, True]
    assert qc_table["QC_Reasons"].tolist() == ["Area<12", "", "Area>30"]
    assert "QC_Include" not in qc_table.columns


def test_evaluate_cell_qc_table_treats_thresholds_as_inclusive_and_rejects_missing_values():
    cells = pd.DataFrame(
        {
            "CellID": [10, 20, 30, 40],
            "Area": [10, 20, 9.99, np.nan],
            "Mean": [5, 15, 10, "not measured"],
        }
    )

    qc_table, flagged_labels, summary = evaluate_cell_qc_table(
        cells,
        {"Area": {"min": 10, "max": 20}, "Mean intensity": {"min": 5, "max": 15}},
    )

    assert flagged_labels == {30, 40}
    assert qc_table["QC_OutOfRange"].tolist() == [False, False, True, True]
    assert qc_table["QC_Reasons"].tolist() == ["", "", "Area<10", "Area=missing; Mean intensity=missing"]
    assert summary == {
        "rules_applied": {
            "Area": {"min": 10.0, "max": 20.0},
            "Mean intensity": {"min": 5.0, "max": 15.0},
        },
        "missing_metrics": [],
        "flagged_count": 2,
        "total_count": 4,
    }


def test_evaluate_cell_qc_table_does_not_modify_its_input():
    cells = pd.DataFrame({"CellID": [1], "Area": [5]})
    original = cells.copy(deep=True)

    evaluate_cell_qc_table(cells, {"Area": {"min": 10}})

    pd.testing.assert_frame_equal(cells, original)


def test_evaluate_cell_qc_table_empty_and_no_rule_contracts():
    empty = pd.DataFrame(columns=["CellID", "Area"])
    empty_result, labels, summary = evaluate_cell_qc_table(empty, {"Area": {"min": 1}})
    assert empty_result.columns.tolist() == ["CellID", "Area", "QC_OutOfRange", "QC_Reasons"]
    assert empty_result.empty
    assert labels == set()
    assert summary["total_count"] == 0
    assert summary["flagged_count"] == 0

    cells = pd.DataFrame({"CellID": [1], "Area": [5]})
    result, labels, summary = evaluate_cell_qc_table(cells, {})
    assert result["QC_OutOfRange"].tolist() == [False]
    assert result["QC_Reasons"].tolist() == [""]
    assert labels == set()
    assert summary["rules_applied"] == {}


def test_integrated_density_filter_names_match_raw_intden_columns():
    cells = pd.DataFrame(
        {
            "CellID": [1, 2],
            "RawIntDen": [100.0, 250.0],
            "IntDen": [10.0, 25.0],
            "RawIntDenInCell": [50.0, 125.0],
        }
    )

    qc_table, flagged_labels, summary = evaluate_cell_qc_table(
        cells,
        {
            "Raw integrated density": {"max": 200},
            "Integrated density": {"max": 20},
            "Mask integrated density": {"max": 100},
        },
    )

    assert flagged_labels == {2}
    assert summary["missing_metrics"] == []
    assert qc_table["QC_Reasons"].tolist() == [
        "",
        "Raw integrated density>200; Integrated density>20; Mask integrated density>100",
    ]


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("Mean intensity", "Signal MeanGrayValue"),
        ("mean", "Signal MeanGrayValue"),
        ("Mask integrated density / cell area", "RawIntDenPerCellArea"),
        ("unknown", None),
        ("", None),
    ],
)
def test_find_matching_cell_qc_column_supports_aliases_suffixes_and_missing_metrics(metric: str, expected):
    table = pd.DataFrame(columns=["Signal MeanGrayValue", "RawIntDenPerCellArea"])
    assert find_matching_cell_qc_column(table, metric) == expected


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cell_geometry_values_and_filter_thresholds(pixel_geometry_cells):
    labels, expected = pixel_geometry_cells
    table = make_per_cell_table(labels, np.ones(labels.shape, dtype=np.uint8))
    pd.testing.assert_frame_equal(table[expected.columns], expected, check_dtype=False, rtol=1e-12, atol=0)
    # Each rule independently selects a different shape; one working metric
    # cannot conceal another missing/broken metric as in a combined OR filter.
    cases: list[tuple[dict[str, dict[str, float | None]], set[int]]] = [
        ({"Perimeter": {"max": 10}}, {17}),
        ({"Solidity": {"min": 0.95}}, {41}),
        ({"Circularity": {"max": 1.7}}, {3}),
    ]
    for rules, expected_ids in cases:
        _qc_table, flagged_labels, summary = evaluate_cell_qc_table(table, rules)
        assert summary["missing_metrics"] == []
        assert flagged_labels == expected_ids


def test_cell_label_series_prefers_cell_id_over_sample_label():
    cells = pd.DataFrame(
        {
            "Label": ["Sample_001", "Sample_001", "Sample_001"],
            "CellID": [1, 2, 3],
            "Area": [10, 20, 35],
        }
    )

    labels = cell_label_series(cells)
    _qc_table, flagged_labels, _summary = evaluate_cell_qc_table(
        cells,
        {"Area": {"max": 30}},
    )

    assert labels.tolist() == [1, 2, 3]
    assert flagged_labels == {3}


def test_cell_label_series_fallback_and_invalid_explicit_id_contracts():
    assert cell_label_series(pd.DataFrame({"Area": [1, 2, 3]})).tolist() == [1, 2, 3]
    assert cell_label_series(pd.DataFrame({"CellID": ["bad", None], "Area": [1, 2]})).tolist() == [0, 0]


def test_cell_label_series_uses_later_valid_identifier_alias():
    table = pd.DataFrame({"CellID": [None, None], "Object": [7, 9]})
    assert cell_label_series(table).tolist() == [7, 9]


def test_append_cell_qc_summary_rows_keeps_filtered_rows_and_both_summary_sets():
    cells = pd.DataFrame(
        {
            "CellID": [1, 2, 3],
            "Area": [10, 20, 35],
            "Mean": [2.0, 5.0, 12.0],
        }
    )
    qc_table, _flagged_labels, _summary = evaluate_cell_qc_table(cells, {"Area": {"max": 30}})
    filtered = cast(Any, qc_table[~qc_table["QC_OutOfRange"]].copy())

    export_table = append_cell_qc_summary_rows(
        filtered,
        include_filtered_summary=True,
        all_cells_table=qc_table,
    )

    assert export_table["CellID"].tolist() == [
        1,
        2,
        "ALL_CELLS_SUM",
        "ALL_CELLS_MEAN",
        "FILTERED_CELLS_SUM",
        "FILTERED_CELLS_MEAN",
    ]
    assert export_table.loc[export_table["CellID"] == "ALL_CELLS_SUM", "Area"].iloc[0] == 65.0
    assert export_table.loc[export_table["CellID"] == "FILTERED_CELLS_SUM", "Area"].iloc[0] == 30.0
    assert "QC_Include" not in export_table.columns


def test_append_cell_qc_summary_rows_has_exact_schema_values_and_reason_markers():
    filtered = pd.DataFrame(
        {
            "CellID": [1],
            "Area": [10],
            "Mean": [2.0],
            "Comment": ["kept"],
            "QC_OutOfRange": [False],
            "QC_Reasons": [""],
            "QC_Include": [True],
        }
    )
    all_cells = pd.concat(
        [
            filtered,
            pd.DataFrame(
                [{"CellID": 2, "Area": 30, "Mean": 6.0, "Comment": "removed", "QC_OutOfRange": True, "QC_Reasons": "Area>20", "QC_Include": False}]
            ),
        ],
        ignore_index=True,
    )

    result = append_cell_qc_summary_rows(
        filtered,
        include_filtered_summary=True,
        all_cells_table=all_cells,
    )

    assert result.columns.tolist() == ["CellID", "Area", "Mean", "Comment", "QC_OutOfRange", "QC_Reasons"]
    summaries = result.iloc[1:].reset_index(drop=True)
    assert summaries["CellID"].tolist() == [
        "ALL_CELLS_SUM",
        "ALL_CELLS_MEAN",
        "FILTERED_CELLS_SUM",
        "FILTERED_CELLS_MEAN",
    ]
    assert summaries["Area"].tolist() == [40.0, 20.0, 10.0, 10.0]
    assert summaries["Mean"].tolist() == [8.0, 4.0, 2.0, 2.0]
    assert summaries["Comment"].tolist() == ["", "", "", ""]
    assert summaries["QC_OutOfRange"].tolist() == ["", "", "", ""]
    assert summaries["QC_Reasons"].tolist() == ["SUM", "MEAN", "SUM", "MEAN"]


def test_append_cell_qc_summary_rows_handles_none_empty_and_disabled_filtered_rows():
    assert append_cell_qc_summary_rows(None, include_filtered_summary=True).empty
    empty = pd.DataFrame(columns=["CellID", "Area"])
    pd.testing.assert_frame_equal(
        append_cell_qc_summary_rows(empty, include_filtered_summary=True),
        empty,
    )
    one = pd.DataFrame({"CellID": [1], "Area": [2]})
    result = append_cell_qc_summary_rows(one, include_filtered_summary=False)
    assert result["CellID"].tolist() == [1, "ALL_CELLS_SUM", "ALL_CELLS_MEAN"]


def test_cell_outline_and_flagged_label_mask_are_binary_and_shape_stable():
    labels = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.int32,
    )

    outline = cell_label_outline_mask(labels)
    flagged = make_mask_for_cell_labels(labels, {2})
    assert flagged is not None
    flagged_outline = cell_label_outline_mask(np.where(flagged, labels, 0))

    assert outline.shape == labels.shape
    assert flagged.shape == labels.shape
    assert flagged_outline.shape == labels.shape
    assert set(np.unique(outline)).issubset({0, 255})
    assert set(np.unique(flagged_outline)).issubset({0, 255})
    assert np.count_nonzero(flagged_outline) < np.count_nonzero(outline)


def test_cell_label_outline_mask_matches_exact_single_cell_boundary():
    labels = np.zeros((5, 5), dtype=np.int16)
    labels[1:4, 1:4] = 7

    assert np.array_equal(
        cell_label_outline_mask(labels),
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 0, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_cell_label_outline_mask_preserves_both_sides_of_adjacent_labels():
    labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    assert np.array_equal(cell_label_outline_mask(labels), np.full((2, 4), 255, dtype=np.uint8))


@pytest.mark.parametrize("shape", [(2,), (2, 2, 2)])
def test_cell_label_outline_mask_rejects_non_2d_unsqueezable_images(shape):
    with pytest.raises(ValueError, match="expects a 2D label image"):
        cell_label_outline_mask(np.zeros(shape, dtype=np.int32))


def test_label_selection_has_exact_arrays_and_does_not_mutate_input():
    labels = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int16)
    original = labels.copy()

    selected = make_mask_for_cell_labels(labels, {1, 3})

    assert selected is not None
    assert np.array_equal(selected, np.array([[False, True, False], [True, False, True]], dtype=None))
    assert np.array_equal(labels, original)


def test_label_selection_empty_inputs_return_documented_sentinels():
    assert make_mask_for_cell_labels(None, {1}) is None
    assert make_mask_for_cell_labels(np.array([], dtype=int), {1}) is None
    assert make_mask_for_cell_labels(np.array([[1]], dtype=None), set()) is None


def test_mask_metrics_and_shared_rules_flags_cells_by_positive_area_fraction():
    cells = pd.DataFrame({"CellID": [1, 2], "Area": [4, 4]})
    cell_mask = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=None,
    )
    weka_mask = np.array(
        [
            [1, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    intensity = np.ones(cell_mask.shape, dtype=float)

    qc_table, flagged_labels, summary = evaluate_cell_qc_table(
        attach_live_mask_metrics(cells, cell_mask, weka_mask, intensity),
        {"Mask fraction of cell area": {"max": 0.75}},
    )

    assert flagged_labels == {2}
    assert summary["flagged_count"] == 1
    assert summary == {
        "rules_applied": {"Mask fraction of cell area": {"max": 0.75}},
        "missing_metrics": [],
        "flagged_count": 1,
        "total_count": 2,
    }
    assert qc_table["QC_OutOfRange"].tolist() == [False, True]
    assert qc_table["PositiveAreaFractionInCell"].tolist() == [0.25, 1.0]


def test_make_mask_qc_metric_table_has_exact_per_label_numeric_contract():
    cell_mask = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    positive_mask = np.array([[1, 0, 1, 1], [0, 0, 0, 1]], dtype=bool)
    intensity = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)

    metrics = make_mask_qc_metric_table(cell_mask, positive_mask, intensity)

    assert metrics["CellID"].tolist() == [1, 2]
    assert metrics["MaskQC_CellArea"].tolist() == [4, 4]
    assert metrics["PositiveAreaInCell"].tolist() == [1, 3]
    assert metrics["PositiveAreaFractionInCell"].tolist() == [0.25, 0.75]
    assert metrics["MeanInPositiveArea"].tolist() == [1.0, 5.0]
    assert metrics["RawIntDenInCell"].tolist() == [1.0, 15.0]
    assert metrics["RawIntDenPerCellArea"].tolist() == [0.25, 3.75]
    assert metrics["FractionOfCellIntDen"].tolist() == pytest.approx([1 / 14, 15 / 22])


def test_make_mask_qc_metric_table_rejects_invalid_or_mismatched_shapes():
    labels = np.ones((2, 2), dtype=np.int32)
    assert make_mask_qc_metric_table(np.ones((2, 2, 2)), labels, labels).empty
    assert make_mask_qc_metric_table(labels, np.ones((3, 3)), labels).empty
    assert make_mask_qc_metric_table(labels, labels, np.ones((3, 3))).empty


def test_mask_metrics_and_shared_rules_maps_metrics_by_cell_id_not_row_position():
    cells = pd.DataFrame({"CellID": [2, 1], "Area": [4, 4]})
    cell_mask = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=None)
    positive_mask = np.array([[1, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
    intensity = np.ones((2, 4), dtype=float)

    table, flagged, summary = evaluate_cell_qc_table(
        attach_live_mask_metrics(cells, cell_mask, positive_mask, intensity),
        {"Mask fraction of cell area": {"max": 0.75}},
    )

    assert table["CellID"].tolist() == [2, 1]
    assert table["PositiveAreaFractionInCell"].tolist() == [1.0, 0.25]
    assert table["QC_OutOfRange"].tolist() == [True, False]
    assert table["QC_Reasons"].tolist() == ["Mask fraction of cell area>0.75", ""]
    assert flagged == {2}
    assert summary["flagged_count"] == 1


def test_mask_qc_metrics_do_not_report_undefined_intensity_values_as_zero():
    cell_mask = np.ones((2, 2), dtype=np.int32)
    empty_mask = np.zeros((2, 2), dtype=bool)
    zero_intensity = np.zeros((2, 2), dtype=float)

    metrics = make_mask_qc_metric_table(cell_mask, empty_mask, zero_intensity)

    assert metrics.loc[0, "RawIntDenInCell"] == 0
    assert np.isnan(metrics.loc[0, "MeanInPositiveArea"])
    assert np.isnan(metrics.loc[0, "FractionOfCellIntDen"])


def test_combine_qc_label_sets_respects_filter_mode():
    cell_labels = {1, 2}
    mask_labels = {2, 3}

    assert combine_qc_label_sets(cell_labels, mask_labels, "Exclude if any filter fails") == {1, 2, 3}
    assert combine_qc_label_sets(cell_labels, mask_labels, "Exclude only if both fail") == {2}
    assert combine_qc_label_sets(cell_labels, mask_labels, "Use cell filters only") == {1, 2}
    assert combine_qc_label_sets(cell_labels, mask_labels, "Use mask filters only") == {2, 3}


def test_combine_qc_label_sets_normalizes_unknown_mode_and_removes_background():
    assert combine_qc_label_sets({0, 1}, {0, 2}, "unknown persisted value") == {1, 2}
    assert combine_qc_label_sets(set(), set(), "Exclude only if both fail") == set()


def test_normalize_cell_qc_rules_drops_empty_and_invalid_values():
    rules = normalize_cell_qc_rules(
        {
            "Area": {"min": "10", "max": ""},
            "Mean": {"min": None, "max": "bad"},
            "Perimeter": {"min": float("nan")},
            "Circularity": {"max": float("inf")},
            "": {"min": 1},
            "Solidity": {"max": 0.95},
        }
    )

    assert rules == {
        "Area": {"min": 10.0},
        "Solidity": {"max": 0.95},
    }


@pytest.mark.parametrize("rules", [None, [], "Area", 1])
def test_normalize_cell_qc_rules_rejects_non_mapping_inputs(rules):
    assert normalize_cell_qc_rules(rules) == {}


def test_cell_qc_rules_text_uses_only_normalized_finite_limits():
    assert cell_qc_rules_to_text(
        {
            "Area": {"min": "10", "max": 20},
            "Mean": {"min": "bad", "max": 5.5},
            "Ignored": {"min": float("nan")},
        }
    ) == "Area:10-20; Mean:-5.5"
