"""Release regressions for result linking and Cell Groups exports."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.config.adapter import parse_cell_qc_limits_text
from cellonaut.masks.preview_filter_overlay import (
    export_all_filtered_result_tables,
    find_preview_filter_data,
    filtered_table_from_current_rules,
)
from cellonaut.results.layout import build_results_layout


def fixture(root, nested=False):
    layout = build_results_layout(root)
    relative = Path("batch") if nested else Path()
    for key in ("cell_segmentation_tables", "cell_segmentation_labels", "cell_signal_tables", "mask_overlays"):
        (layout[key] / relative).mkdir(parents=True)
    pd.DataFrame({"CellID": [1, 2], "Area": [2, 2]}).to_csv(
        layout["cell_segmentation_tables"] / relative / "sample_GFP_cell_measurements.csv", index=False
    )
    tifffile.imwrite(
        layout["cell_segmentation_labels"] / relative / "sample_GFP_01_cellpose_labels.tif",
        np.array([[1, 1], [2, 2]], dtype=np.int32),
    )
    overlay = layout["mask_overlays"] / relative / "sample_GFP_combined_overlay.tif"
    tifffile.imwrite(overlay, np.ones((2, 2), dtype=np.uint16))
    return layout, relative, overlay


@pytest.mark.parametrize("nested", [False, True])
def test_target_mask_and_exported_intensity_names_work_in_preview_and_batch(tmp_path, nested):
    layout, relative, overlay = fixture(tmp_path, nested)
    for mask, values in (("A", [10, 100]), ("B", [100, 10])):
        pd.DataFrame({"CellID": [1, 2], f"{mask}MeanGrayValue_InCell": values}).to_csv(
            layout["cell_signal_tables"] / relative / f"sample_GFP_{mask}_per_cell_signal.csv", index=False
        )
    settings = {
        "name": "GFP",
        "mask_relationships": {"A": True, "B": True},
        "cell_group_mask_source": "B",
        "cell_populations": [
            {"name": "Bright", "mask_qc_limits": "Mean intensity inside mask:50-", "exclude_from_csv": True}
        ],
    }
    data, message = find_preview_filter_data(overlay, [settings])
    assert not message and data is not None
    assert data.table["MeanInPositiveArea"].tolist() == [100, 10]
    result = export_all_filtered_result_tables(layout["root"], [settings], parse_cell_qc_limits_text)
    assert result.processed_count == 1 and result.skipped_count == 0
    assert result.kept_combined_path is not None
    assert pd.read_csv(result.kept_combined_path)["CellID"].tolist() == [2]
    assert (result.export_dir / relative / "sample_GFP_cell_measurements_filtered.csv").is_file()
    assert (result.export_dir / relative / "sample_GFP_B_per_cell_signal_filtered.csv").is_file()


@pytest.mark.parametrize("missing", ["table", "labels"])
def test_preview_never_borrows_another_samples_results(tmp_path, missing):
    layout, _, overlay = fixture(tmp_path)
    key, suffix = (
        ("cell_segmentation_tables", "_cell_measurements.csv")
        if missing == "table"
        else ("cell_segmentation_labels", "_01_cellpose_labels.tif")
    )
    original = layout[key] / f"sample_GFP{suffix}"
    original.rename(layout[key] / f"other_GFP{suffix}")
    data, message = find_preview_filter_data(overlay, [{"name": "GFP"}])
    assert data is None and message


def test_missing_required_metric_blocks_export_and_reports_skipped_table(tmp_path):
    layout, _, _ = fixture(tmp_path)
    settings = {
        "name": "GFP",
        "cell_populations": [
            {"name": "Bright", "mask_qc_limits": "Mean intensity inside mask:50-", "exclude_from_csv": True}
        ],
    }
    with pytest.raises(ValueError, match="metrics unavailable"):
        filtered_table_from_current_rules(pd.DataFrame({"CellID": [1, 2]}), settings, parse_cell_qc_limits_text)
    result = export_all_filtered_result_tables(layout["root"], [settings], parse_cell_qc_limits_text)
    assert result.processed_count == 0 and result.skipped_count == 1
    assert result.kept_combined_path is None
    assert any("metrics unavailable" in message for message in result.messages)

@pytest.mark.parametrize("nested", [False, True])
def test_group_source_switch_replaces_saved_metrics_and_preserves_summaries(tmp_path, nested):
    import json

    layout, relative, overlay = fixture(tmp_path, nested)
    pd.DataFrame({"CellID": [1, 2], "Mean": [100, 10], "CellMeanGrayValue": [10, 100],
                  "CellArea": [2, 2]}).to_csv(
        layout["cell_segmentation_tables"] / relative / "sample_GFP_cell_measurements.csv", index=False)
    pd.DataFrame({"CellID": [1, 2], "AMeanGrayValue_InCell": [10, 100]}).to_csv(
        layout["cell_signal_tables"] / relative / "sample_GFP_A_per_cell_signal.csv", index=False)
    tifffile.imwrite(overlay, np.array([[[10, 10], [100, 100]], [[100, 100], [10, 10]],
                                      [[1, 1], [1, 1]]], dtype=np.uint16), metadata={"axes": "CYX"}, photometric="minisblack")
    metadata = layout["mask_overlay_metadata"] / relative
    metadata.mkdir(parents=True)
    (metadata / overlay.with_suffix(".json").name).write_text(json.dumps({"layer_labels": ["GFP", "BF", "A"]}))
    settings = {"name": "GFP", "analysis_cell_segmentation_source": "BF", "cell_group_mask_source": "A", "mask_relationships": {"A": True},
                "cell_populations": [{"name": "Bright", "mask_qc_limits": "Mean intensity inside mask:50-",
                                      "mask_qc_intensity_source": "Cell mask image", "exclude_from_csv": True}]}
    result = export_all_filtered_result_tables(layout["root"], [settings], parse_cell_qc_limits_text)
    assert result.processed_count == 1, result.messages
    assert result.kept_combined_path is not None
    assert result.all_measurements_path is not None
    assert pd.read_csv(result.kept_combined_path)["CellID"].tolist() == [2]
    summary = pd.read_csv(result.all_measurements_path)
    assert len(summary.columns) > 1
    assert 100 in summary.select_dtypes(include="number").iloc[0].values


def test_live_source_replaces_canonical_mask_intensity():
    from cellonaut.masks.preview_filter_overlay import attach_live_mask_metrics

    table = pd.DataFrame({"CellID": [1, 2], "MeanInPositiveArea": [100, 10]})
    updated = attach_live_mask_metrics(table, np.array([[1, 1], [2, 2]]), np.ones((2, 2)),
                                      np.array([[10, 10], [100, 100]]))
    assert updated["MeanInPositiveArea"].tolist() == [10, 100]
    assert table["MeanInPositiveArea"].tolist() == [100, 10]


def test_summary_reads_saved_means_without_using_segmentation_source_mean():
    from cellonaut.measurement.math import summarize_per_cell_table

    values = summarize_per_cell_table(pd.DataFrame({"MeanGrayValue": [999], "CellMeanGrayValue": [10],
                                                    "AMeanGrayValue_InCell": [20]}), "A", "GFP", "BF")
    assert values["GFP_measured_with_BF_cellpose_mask_PerCell_MeanOfCellMeans"] == 10
    assert values["GFP_measured_with_A_mask_PerCell_MeanOfMaskMeans"] == 20
    assert 999 not in values.values()



def test_each_population_resolves_its_own_source():
    from cellonaut.masks.preview_filter_overlay import attach_live_mask_metrics

    labels = np.array([[1, 1], [2, 2]])
    table = pd.DataFrame({"CellID": [1, 2], "MeanInPositiveArea": [100, 10]})
    settings = {"name": "GFP", "cell_populations": [
        {"name": "Measured bright", "mask_qc_limits": "Mean intensity inside mask:50-"},
        {"name": "Source bright", "mask_qc_limits": "Mean intensity inside mask:50-",
         "mask_qc_intensity_source": "Cell mask image", "exclude_from_csv": True}]}

    def prepare(current, group):
        assert group["name"] == "GFP"
        if group.get("mask_qc_intensity_source") == "Cell mask image":
            return attach_live_mask_metrics(current, labels, np.ones((2, 2)), np.array([[10, 10], [100, 100]]))
        return current

    kept, report, _, _ = filtered_table_from_current_rules(table, settings, parse_cell_qc_limits_text,
                                                          prepare_table=prepare)
    assert kept["CellID"].tolist() == [1]
    assert report["CellGroups"].tolist() == ["Measured bright", "Source bright"]


def test_unavailable_requested_source_skips_export(tmp_path):
    layout, _, _ = fixture(tmp_path)
    settings = {"name": "GFP", "cell_group_mask_source": "A", "mask_relationships": {"A": True},
                "mask_qc_limits": "Mean intensity inside mask:50-", "mask_qc_intensity_source": "Cell mask image"}
    result = export_all_filtered_result_tables(layout["root"], [settings], parse_cell_qc_limits_text)
    assert result.processed_count == 0 and result.skipped_count == 1
    assert any("unavailable" in message for message in result.messages)

@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('saved_means', [False, True])
def test_batch_measured_source_rebuilds_only_missing_metrics(tmp_path, nested, saved_means):
    import json

    layout, relative, overlay = fixture(tmp_path, nested)
    pixels = np.array([[100, 100], [10, 10]], dtype=np.uint16)
    tifffile.imwrite(overlay, np.stack([pixels, np.ones((2, 2), dtype=np.uint16)]),
                     metadata={'axes': 'CYX'}, photometric='minisblack')
    metadata = layout['mask_overlay_metadata'] / relative
    metadata.mkdir(parents=True)
    (metadata / overlay.with_suffix('.json').name).write_text(json.dumps({'layer_labels': ['GFP', 'A']}))
    if saved_means:
        # Saved measurements can include transforms; raw overlay values must not replace them.
        pd.DataFrame({'CellID': [1, 2], 'AMeanGrayValue_InCell': [10, 100]}).to_csv(
            layout['cell_signal_tables'] / relative / 'sample_GFP_A_per_cell_signal.csv', index=False)
    settings = {'name': 'GFP', 'mask_relationships': {'A': True}, 'cell_group_mask_source': 'A',
                'cell_populations': [{'name': 'Bright', 'mask_qc_limits': 'Mean intensity inside mask:50-',
                                      'exclude_from_csv': True}]}
    result = export_all_filtered_result_tables(layout['root'], [settings], parse_cell_qc_limits_text)
    assert result.processed_count == 1 and result.skipped_count == 0, result.messages
    assert result.kept_combined_path is not None
    assert pd.read_csv(result.kept_combined_path)['CellID'].tolist() == ([1] if saved_means else [2])

@pytest.mark.parametrize('selected', ['A', 'A mask'])
def test_preview_uses_exact_selected_mask_name(selected):
    from typing import Any
    from cellonaut.gui.preview_filter import CellonautGuiPreviewFilterMixin

    gui: Any = CellonautGuiPreviewFilterMixin()
    gui.preview_state = PreviewState()
    gui.parse_cell_qc_limits_text = parse_cell_qc_limits_text
    gui.preview_state.tiff_model = {'data': np.array([
        [[100, 100], [100, 100]], [[1, 1], [0, 0]], [[0, 0], [1, 1]]
    ])[None, None]}
    gui.preview_state.current_labels = ['GFP', 'A', 'A mask']
    gui.preview_state.current_layer_roles = ['image', 'weka_mask', 'weka_mask']
    settings = {'name': 'GFP', 'mask_relationships': {'A': True, 'A mask': True},
                'cell_group_mask_source': selected, 'mask_qc_limits': 'Mean intensity inside mask:50-'}
    table = gui.attach_preview_mask_filter_metrics(
        pd.DataFrame({'CellID': [1, 2]}), np.array([[1, 1], [2, 2]]), settings)
    assert table['PositiveAreaInCell'].tolist() == ([2, 0] if selected == 'A' else [0, 2])


@pytest.mark.parametrize('empty', [False, True])
def test_cell_export_names_are_stable_across_repeated_normalization(empty):
    from cellonaut.cell_segmentation.core import normalize_cell_segmentation_export_table

    original = pd.DataFrame({'CellID': [1], 'Mean': [5], 'AMean_InCell': [10],
                             'CellCorrectedMean': [3], 'BMeanGrayValue_InCell': [20]})
    if empty:
        original = original.iloc[:0]
    exported = normalize_cell_segmentation_export_table(original)
    assert exported.columns.tolist() == ['CellID', 'MeanGrayValue', 'AMeanGrayValue_InCell',
                                         'CellCorrectedMeanGrayValue', 'BMeanGrayValue_InCell']
    pd.testing.assert_frame_equal(normalize_cell_segmentation_export_table(exported), exported)
    assert 'Mean' in original.columns


def test_filtered_signal_csv_retains_exported_mean_header(tmp_path):
    layout, _, _ = fixture(tmp_path)
    pd.DataFrame({'CellID': [1, 2], 'AMeanGrayValue_InCell': [10, 100]}).to_csv(
        layout['cell_signal_tables'] / 'sample_GFP_A_per_cell_signal.csv', index=False)
    result = export_all_filtered_result_tables(layout['root'], [{'name': 'GFP'}], parse_cell_qc_limits_text)
    assert result.processed_count == 1
    assert result.signal_combined_path is not None
    for path in [result.export_dir / 'sample_GFP_A_per_cell_signal_filtered.csv', result.signal_combined_path]:
        saved = pd.read_csv(path)
        assert saved['AMeanGrayValue_InCell'].tolist() == [10, 100]
        assert not any('GrayValueGrayValue' in column for column in saved.columns)

@pytest.mark.parametrize('initially_empty', [False, True])
@pytest.mark.parametrize('groups', [False, True])
def test_zero_kept_cells_export_empty_tables_and_zero_totals(tmp_path, initially_empty, groups):
    layout, _, _ = fixture(tmp_path)
    ids = [] if initially_empty else [1, 2]
    pd.DataFrame({'CellID': ids, 'Area': [2] * len(ids), 'CellArea': [2] * len(ids),
                  'CellMeanGrayValue': [10] * len(ids)}).to_csv(
        layout['cell_segmentation_tables'] / 'sample_GFP_cell_measurements.csv', index=False)
    tifffile.imwrite(layout['cell_segmentation_labels'] / 'sample_GFP_01_cellpose_labels.tif',
                     np.zeros((2, 2), dtype=np.int32) if initially_empty else np.array([[1, 1], [2, 2]], dtype=np.int32))
    pd.DataFrame({'CellID': ids, 'AArea_InCell': [1] * len(ids),
                  'AMeanGrayValue_InCell': [10] * len(ids)}).to_csv(
        layout['cell_signal_tables'] / 'sample_GFP_A_per_cell_signal.csv', index=False)
    from typing import Any
    settings: dict[str, Any] = {'name': 'GFP'}
    if groups:
        settings['cell_populations'] = [{'name': 'All', 'cell_qc_limits': 'Area:1-', 'exclude_from_csv': True}]
    else:
        settings['cell_qc_limits'] = 'Area:100-'
    result = export_all_filtered_result_tables(layout['root'], [settings], parse_cell_qc_limits_text)
    assert result.processed_count == 1 and result.skipped_count == 0, result.messages
    assert result.kept_count == 0
    assert result.kept_combined_path is not None
    assert result.signal_combined_path is not None
    assert result.all_measurements_path is not None
    for path in [result.kept_combined_path, result.signal_combined_path,
                 result.export_dir / 'sample_GFP_A_per_cell_signal_filtered.csv']:
        saved = pd.read_csv(path)
        assert saved.empty and 'CellID' in saved.columns
    summary = pd.read_csv(result.all_measurements_path)
    area_columns = [column for column in summary.columns if 'area' in column.lower()]
    assert len(area_columns) >= 2
    assert all(summary[column].iloc[0] == 0 for column in area_columns)
    assert not any('mean intensity' in column.lower() for column in summary.columns)

@pytest.mark.parametrize('groups', [False, True])
def test_empty_filter_table_does_not_require_source_pixels(groups):
    rules = {'mask_qc_limits': 'Mean intensity inside mask:50-', 'mask_qc_intensity_source': 'Cell mask image'}
    settings = {'name': 'GFP', 'cell_populations': [rules]} if groups else rules

    def unavailable_pixels(*_args):
        raise AssertionError('No pixels are needed when there are no cells')

    kept, _, excluded, _ = filtered_table_from_current_rules(
        pd.DataFrame({'CellID': pd.Series(dtype=int), 'CellArea': pd.Series(dtype=float)}),
        settings, parse_cell_qc_limits_text, prepare_table=unavailable_pixels)
    assert kept.empty and 'CellID' in kept.columns and not excluded
