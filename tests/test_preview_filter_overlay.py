from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.masks.cell_groups import excluded_labels_from_current_rules
from cellonaut.gui import preview_filter as preview_filter_gui
from cellonaut.gui.preview_filter import CellonautGuiPreviewFilterMixin
from cellonaut.masks.adjustments import adjust_label_image, adjust_mask_image
from cellonaut.measurement.table_formatting import readable_results_table
from cellonaut.masks.preview_filter_overlay import (
    _merge_mask_signal_metrics,
    attach_live_cell_shape_metrics,
    attach_live_mask_metrics,
    export_all_filtered_result_tables,
    filtered_table_from_current_rules,
    find_preview_filter_data,
)


def test_mask_signal_metrics_are_available_to_live_filter_rules():
    cell_table = pd.DataFrame({"CellID": [1, 2], "Area": [10, 20]})
    signal_table = pd.DataFrame(
        {
            "CellID": ["SUM", 1, 2],
            "NucleusArea_InCell": [7, 2, 5],
            "NucleusAreaFraction_InCell": [0.7, 0.2, 0.5],
        }
    )

    merged = _merge_mask_signal_metrics(cell_table, signal_table, "Nucleus")
    _kept, _report, excluded, summary = filtered_table_from_current_rules(
        merged,
        {
            "cell_qc_limits": "",
            "mask_qc_limits": "Mask fraction of cell area:-0.4",
            "qc_filter_mode": "Exclude if any filter fails",
        },
        lambda text: {"Mask fraction of cell area": {"max": 0.4}} if text else {},
    )

    assert excluded == {2}
    assert summary["mask_flagged"] == 1


def test_mask_filter_can_derive_fraction_from_raw_measurements():
    cell_table = pd.DataFrame({"CellID": [1, 2], "Area": [10, 10]})
    signal_table = pd.DataFrame({"CellID": [1, 2], "NucleusArea_InCell": [2, 8]})

    merged = _merge_mask_signal_metrics(cell_table, signal_table, "Nucleus")

    assert merged["PositiveAreaFractionInCell"].tolist() == [0.2, 0.8]


def test_cell_shape_filters_can_rebuild_ratios_from_labels(pixel_geometry_cells):
    labels, expected = pixel_geometry_cells
    expected = expected.set_index("CellID").loc[[41, 3, 17]].reset_index()
    table = expected[["CellID", "Area"]].copy()
    before = table.copy(deep=True)

    enriched = attach_live_cell_shape_metrics(table, labels)

    columns = ["CellID", "Area", "Circularity", "Solidity"]
    pd.testing.assert_frame_equal(enriched[columns], expected[columns], check_dtype=False, rtol=1e-12, atol=0)
    pd.testing.assert_frame_equal(table, before)


def test_live_mask_metrics_are_calculated_from_preview_pixels():
    labels = np.array([[1, 1], [2, 2]], dtype=np.uint16)
    mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    intensity = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    table = attach_live_mask_metrics(pd.DataFrame({"CellID": [1, 2]}), labels, mask, intensity)

    assert table["PositiveAreaInCell"].tolist() == [1, 2]
    assert table["PositiveAreaFractionInCell"].tolist() == [0.5, 1.0]
    assert table["MeanInPositiveArea"].tolist() == [10.0, 35.0]


def test_live_mask_metrics_preserve_summary_rows_with_text_cell_ids():
    labels = np.array([[1, 1], [2, 2]], dtype=np.uint16)
    mask = np.ones((2, 2), dtype=np.uint8)
    intensity = np.ones((2, 2), dtype=np.uint16)
    source = pd.DataFrame({"CellID": [1, 2, "ALL_CELLS_SUM", "ALL_CELLS_MEAN"]})

    table = attach_live_mask_metrics(source, labels, mask, intensity)

    assert table["CellID"].tolist() == [1, 2, "ALL_CELLS_SUM", "ALL_CELLS_MEAN"]
    assert table["PositiveAreaInCell"].iloc[:2].tolist() == [2.0, 2.0]
    assert table["PositiveAreaInCell"].iloc[2:].isna().all()


class PreviewFilterSyncHarness(CellonautGuiPreviewFilterMixin):
    def __init__(self, preview_file):
        self.preview_state = PreviewState()
        self.preview_state.file_path = str(preview_file)
        self.unavailable = None
        self.messages = []

    def commit_gui_edits(self) -> None:
        raise RuntimeError("settings sync failed")

    def get_active_image_definitions(self) -> list[dict]:
        return [{"name": "GFP", "cell_populations": [{"name": "Group A"}]}]

    def parse_cell_qc_limits_text(self, _text: str) -> dict:
        return {}

    def set_preview_filter_unavailable(self, message: str, *, failed: bool = False):
        self.unavailable = (message, failed)

    def log(self, message: str):
        self.messages.append(message)


def test_preview_filter_reports_settings_sync_failure(tmp_path):
    preview_file = tmp_path / "preview.tif"
    preview_file.write_bytes(b"preview placeholder")
    preview = PreviewFilterSyncHarness(preview_file)

    preview.refresh_preview_filter_overlay()

    assert preview.unavailable == (
        "could not read current cell-group settings (RuntimeError: settings sync failed)",
        True,
    )
    assert preview.messages == [
        "[PREVIEW][WARN] could not read current cell-group settings (RuntimeError: settings sync failed)"
    ]


def test_preview_filter_reports_group_evaluation_failure(monkeypatch, tmp_path):
    preview_file = tmp_path / "preview.tif"
    preview_file.write_bytes(b"preview placeholder")
    preview = PreviewFilterSyncHarness(preview_file)
    preview.commit_gui_edits = lambda: None
    monkeypatch.setattr(
        preview_filter_gui,
        "find_preview_filter_data",
        lambda *_args: (
            type("Data", (), {
                "source_label": "GFP",
                "table": pd.DataFrame({"CellID": [1]}),
                "label_image": np.array([[1]], dtype=np.uint16),
            })(),
            "",
        ),
    )
    monkeypatch.setattr(preview_filter_gui, "find_image_def_for_label", lambda defs, _label: defs[0])
    monkeypatch.setattr(
        preview,
        "parse_cell_qc_limits_text",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid rule")),
    )

    preview.refresh_preview_filter_overlay()

    assert preview.unavailable == ("could not evaluate Group A (ValueError: invalid rule)", True)
    assert preview.messages == [
        "[PREVIEW][WARN] could not evaluate Group A (ValueError: invalid rule)"
    ]


def test_preview_filter_data_matches_full_result_id_with_underscores(tmp_path):
    results = tmp_path / "preview_1" / "Results"
    overlay = results / "Overlays" / "Sample_A_GFP_combined_overlay.tif"
    table = results / "CSV Data" / "Cell Measurements" / "Sample_A_GFP_cell_measurements.csv"
    labels = results / "Cells" / "TIFF Labels" / "Sample_A_GFP_01_cellpose_labels.tif"

    overlay.parent.mkdir(parents=True)
    tifffile.imwrite(overlay, np.zeros((2, 4, 4), dtype=np.uint8))
    table.parent.mkdir(parents=True)
    pd.DataFrame({"CellID": [1], "Area": [20]}).to_csv(table, index=False)
    labels.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(labels, np.ones((4, 4), dtype=np.uint16))

    data, message = find_preview_filter_data(
        overlay,
        [{"name": "GFP", "analysis_cell_segmentation_source": "mCherry"}],
    )

    assert message == ""
    assert data is not None
    assert data.result_id == "Sample_A"
    assert data.table_path == table
    assert data.labels_path == labels


def test_preview_filter_data_loads_selected_mask_signal_metrics(tmp_path):
    results = tmp_path / "preview_1" / "Results"
    overlay = results / "Overlays" / "Sample_A_GFP_combined_overlay.tif"
    table = results / "CSV Data" / "Cell Measurements" / "Sample_A_GFP_cell_measurements.csv"
    signal = results / "CSV Data" / "Signal" / "Sample_A_GFP_Nucleus_per_cell_signal.csv"
    labels = results / "Cells" / "TIFF Labels" / "Sample_A_GFP_01_cellpose_labels.tif"
    overlay.parent.mkdir(parents=True)
    tifffile.imwrite(overlay, np.zeros((3, 4, 4), dtype=np.uint8))
    table.parent.mkdir(parents=True)
    pd.DataFrame({"CellID": [1, 2], "Area": [10, 20]}).to_csv(table, index=False)
    signal.parent.mkdir(parents=True)
    pd.DataFrame(
        {"CellID": [1, 2], "NucleusArea_InCell": [2, 8], "NucleusAreaFraction_InCell": [0.2, 0.8]}
    ).to_csv(signal, index=False)
    labels.parent.mkdir(parents=True)
    tifffile.imwrite(labels, np.ones((4, 4), dtype=np.uint16))

    data, message = find_preview_filter_data(
        overlay,
        [{"name": "GFP", "mask_relationships": {"Nucleus": True}}],
    )

    assert message == ""
    assert data is not None
    assert data.table["PositiveAreaInCell"].tolist() == [2, 8]
    assert data.table["PositiveAreaFractionInCell"].tolist() == [0.2, 0.8]


def test_preview_filter_data_uses_overlay_sidecar_when_channels_were_renamed(tmp_path):
    results = tmp_path / "preview_1" / "Results"
    overlay = results / "Overlays" / "Sample_001_GFP_combined_overlay.tif"
    sidecar = overlay.with_suffix(".json")
    table = results / "CSV Data" / "Cell Measurements" / "Sample_001_GFP_cell_measurements.csv"
    labels = results / "Cells" / "TIFF Labels" / "Sample_001_GFP_01_cellpose_labels.tif"

    overlay.parent.mkdir(parents=True)
    tifffile.imwrite(overlay, np.zeros((2, 4, 4), dtype=np.uint8))
    sidecar.write_text(
        '{"base_label":"GFP","layer_labels":["Brightfield","mCherry","GFP"]}',
        encoding="utf-8",
    )
    table.parent.mkdir(parents=True)
    pd.DataFrame({"CellID": [1], "Area": [20]}).to_csv(table, index=False)
    labels.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(labels, np.ones((4, 4), dtype=np.uint16))

    data, message = find_preview_filter_data(
        overlay,
        [{"name": "Channel 3", "analysis_cell_segmentation_source": "Channel 2"}],
    )

    assert message == ""
    assert data is not None
    assert data.source_label == "GFP"
    assert data.table_path == table
    assert data.labels_path == labels


def test_preview_filter_data_scores_table_when_current_name_no_longer_matches(tmp_path):
    results = tmp_path / "preview_1" / "Results"
    overlay = results / "Overlays" / "Sample_001_GFP_combined_overlay.tif"
    table = results / "CSV Data" / "Cell Measurements" / "Sample_001_GFP_cell_measurements.csv"
    labels = results / "Cells" / "TIFF Labels" / "Sample_001_GFP_01_cellpose_labels.tif"

    overlay.parent.mkdir(parents=True)
    tifffile.imwrite(overlay, np.zeros((2, 4, 4), dtype=np.uint8))
    table.parent.mkdir(parents=True)
    pd.DataFrame({"CellID": [1], "Area": [20]}).to_csv(table, index=False)
    labels.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(labels, np.ones((4, 4), dtype=np.uint16))

    data, message = find_preview_filter_data(
        overlay,
        [{"name": "Channel 3", "analysis_cell_segmentation_source": "Channel 2"}],
    )

    assert message == ""
    assert data is not None
    assert data.result_id == "Sample_001"
    assert data.source_label == "GFP"


def test_preview_filter_data_keeps_scored_table_and_mask_on_same_sample(tmp_path):
    results = tmp_path / "preview_1" / "Results"
    overlay = results / "Overlays" / "Sample_01_GFP_combined_overlay.tif"
    table_dir = results / "CSV Data" / "Cell Measurements"
    label_dir = results / "Cells" / "TIFF Labels"
    expected_table = table_dir / "Sample_01_GFP_cell_measurements.csv"
    expected_labels = label_dir / "Sample_01_GFP_01_cellpose_labels.tif"

    overlay.parent.mkdir(parents=True)
    tifffile.imwrite(overlay, np.zeros((2, 4, 4), dtype=np.uint8))
    table_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    pd.DataFrame({"CellID": [1], "Area": [10]}).to_csv(expected_table, index=False)
    pd.DataFrame({"CellID": [2], "Area": [99]}).to_csv(
        table_dir / "Sample_010_GFP_cell_measurements.csv", index=False
    )
    tifffile.imwrite(expected_labels, np.ones((4, 4), dtype=np.uint16))
    tifffile.imwrite(label_dir / "Sample_010_GFP_01_cellpose_labels.tif", np.full((4, 4), 2, dtype=np.uint16))

    data, message = find_preview_filter_data(
        overlay,
        [{"name": "Renamed channel", "analysis_cell_segmentation_source": "Renamed source"}],
    )

    assert message == ""
    assert data is not None
    assert data.table_path == expected_table
    assert data.labels_path == expected_labels
    assert data.table["Area"].tolist() == [10]
    assert np.all(data.label_image == 1)


def test_filtered_table_from_current_rules_exports_kept_and_report_rows():
    table = pd.DataFrame(
        {
            "CellID": [1, 2, 3],
            "Area": [10, 20, 30],
        }
    )

    kept, report, excluded, summary = filtered_table_from_current_rules(
        table,
        {"cell_qc_limits": "Area:-20", "mask_qc_limits": "", "qc_filter_mode": "Exclude if any filter fails"},
        lambda text: {"Area": {"max": 20}} if text else {},
    )

    assert excluded == {3}
    assert summary["excluded"] == 1
    assert kept["CellID"].tolist() == [1, 2]
    assert report["LiveFilter_Excluded"].tolist() == [False, False, True]


def test_filtered_table_does_not_count_or_export_summary_rows_as_cells():
    table = pd.DataFrame(
        {
            "CellID": [1, 2, "ALL_CELLS_SUM", "ALL_CELLS_MEAN"],
            "Area": [10, 30, 40, 20],
        }
    )

    kept, report, excluded, summary = filtered_table_from_current_rules(
        table,
        {"cell_qc_limits": "Area:-20", "mask_qc_limits": "", "qc_filter_mode": "Exclude if any filter fails"},
        lambda text: {"Area": {"max": 20}} if text else {},
    )

    assert excluded == {2}
    assert summary["total"] == 2
    assert kept["CellID"].tolist() == [1]
    assert report["CellID"].tolist() == [1, 2, "ALL_CELLS_SUM", "ALL_CELLS_MEAN"]


def test_cell_groups_export_boolean_and_compact_membership_columns():
    table = pd.DataFrame({"CellID": [1, 2, 3], "Area": [10, 20, 30]})
    image_def = {
        "cell_populations": [
            {"name": "Small cells", "cell_qc_limits": "Area:-15", "exclude_from_csv": False},
            {"name": "Large cells", "cell_qc_limits": "Area:25-", "exclude_from_csv": True},
        ]
    }

    kept, report, excluded, _summary = filtered_table_from_current_rules(
        table,
        image_def,
        lambda text: (
            {"Area": {"max": 15}}
            if text == "Area:-15"
            else {"Area": {"min": 25}}
            if text == "Area:25-"
            else {}
        ),
    )

    assert report["CellGroup_Small_cells"].tolist() == [True, False, False]
    assert report["CellGroup_Large_cells"].tolist() == [False, False, True]
    assert report["CellGroups"].tolist() == ["Small cells", "", "Large cells"]
    assert report["CellGroupCount"].tolist() == [1, 0, 1]
    assert excluded == {3}
    assert kept["CellID"].tolist() == [1, 2]


def test_adjust_label_image_shifts_without_mutating_source():
    labels = np.zeros((4, 5), dtype=np.uint16)
    labels[1, 1] = 7

    shifted = adjust_label_image(labels, dx=2, dy=1)

    assert labels[1, 1] == 7
    assert shifted[2, 3] == 7
    assert shifted[1, 1] == 0


def test_adjust_mask_image_removes_small_objects():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0, 0] = 255
    mask[2:4, 2:4] = 255

    adjusted = adjust_mask_image(mask, min_size=2)

    assert adjusted[0, 0] == 0
    assert adjusted[2, 2] == 255


def test_minimum_mask_area_treats_diagonally_touching_pixels_as_one_particle():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255
    mask[1, 1] = 255

    adjusted = adjust_mask_image(mask, min_size=2)

    assert adjusted[0, 0] != 0
    assert adjusted[1, 1] != 0


def test_adjust_mask_image_fills_only_holes_within_area_limit():
    mask = np.full((12, 12), 255, dtype=np.uint8)
    mask[2, 2] = 0
    mask[6:9, 6:9] = 0

    adjusted = adjust_mask_image(mask, fill_holes_area=4)

    assert adjusted[2, 2] == 255
    assert np.all(adjusted[6:9, 6:9] == 0)


def test_adjust_label_image_applies_hole_limit_per_cell_label():
    labels = np.zeros((12, 24), dtype=np.uint16)
    labels[1:11, 1:11] = 3
    labels[3, 3] = 0
    labels[1:11, 13:23] = 7
    labels[5:8, 16:19] = 0

    adjusted = adjust_label_image(labels, fill_holes_area=4)

    assert adjusted[3, 3] == 3
    assert np.all(adjusted[5:8, 16:19] == 0)


def test_readable_results_table_names_and_groups_columns():
    table = pd.DataFrame(
        {
            "GFP_in_mCherry_Mean": [12.5],
            "Label": ["Sample_001"],
            "GFP_QC_ExcludedCount": [3],
            "CellID": [1],
            "PositiveAreaFractionInCell": [0.4],
        }
    )

    readable = readable_results_table(table)

    assert readable.columns.tolist() == [
        "Sample",
        "Cell ID",
        "GFP: cells excluded from filtered measurements (count)",
        "Mask fraction of cell area (ratio)",
        "GFP(mCherry) : Mean intensity (a.u.)",
    ]


def test_export_all_filtered_result_tables_writes_per_table_and_combined_outputs(tmp_path):
    results = tmp_path / "run_1" / "Results"
    tables = results / "CSV Data" / "Cell Measurements"
    tables.mkdir(parents=True)

    pd.DataFrame(
        {
            "CellID": [1, 2, 3],
            "Area": [10, 25, 40],
            "SourceImageLabel": ["GFP", "GFP", "GFP"],
        }
    ).to_csv(tables / "Sample_001_GFP_cell_measurements.csv", index=False)
    pd.DataFrame(
        {
            "CellID": [1, 2],
            "Area": [5, 35],
            "SourceImageLabel": ["GFP", "GFP"],
        }
    ).to_csv(tables / "Sample_002_GFP_cell_measurements.csv", index=False)
    signal_dir = results / "CSV Data" / "Signal"
    signal_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "CellID": ["SUM", "MEAN", 1, 2, 3],
            "CellArea": ["", "", 10, 20, 30],
            "mCherryArea_InCell": ["", "", 2, 4, 6],
            "mCherryMean_InCell": ["", "", 100, 200, 300],
            "Centroid_Y": ["", "", 1, 2, 3],
        }
    ).to_csv(signal_dir / "Sample_001_GFP_mCherry_per_cell_signal.csv", index=False)

    result = export_all_filtered_result_tables(
        results,
        [{"name": "GFP", "cell_qc_limits": "Area:-30", "qc_filter_mode": "Exclude if any filter fails"}],
        lambda text: {"Area": {"max": 30}} if text else {},
    )

    assert result.processed_count == 2
    assert result.skipped_count == 0
    assert result.kept_count == 3
    assert result.total_count == 5
    assert result.kept_combined_path is not None
    assert result.kept_combined_path.exists()
    assert result.export_dir.name == "export_1"
    assert (result.export_dir / "Sample_001_GFP_cell_measurements_filtered.csv").exists()
    assert (result.export_dir / "Sample_001_GFP_cell_measurements_filter_report.csv").exists()
    assert not (result.export_dir / "All_Cell_Measurements_Filtered_Readable.csv").exists()
    assert result.all_measurements_path is not None
    assert result.all_measurements_path.exists()
    assert result.simple_measurements_path is not None
    assert result.simple_measurements_path.exists()
    assert result.readable_measurements_path is not None
    assert result.readable_measurements_path.exists()
    assert not (result.export_dir / "Measurements_Detailed.csv").exists()
    assert not (result.export_dir / "Measurements_Tidy.csv").exists()
    assert result.signal_combined_path is not None
    assert result.signal_combined_path.exists()

    combined = pd.read_csv(result.kept_combined_path)
    assert combined["CellID"].tolist() == [1, 2, 1]
    summary = pd.read_csv(result.all_measurements_path)
    sample_1 = summary[summary["Sample"] == "Sample_001"].iloc[0]
    assert sample_1["GFP(mCherry) : total mask area inside cells (px²)"] == 6
    assert result.group_membership_path is None
    signal = pd.read_csv(result.signal_combined_path)
    assert signal["CellID"].tolist() == [1, 2]


def test_empty_cell_group_does_not_match_or_exclude_cells():
    table = pd.DataFrame({"CellID": [1, 2], "Area": [10, 30]})
    kept, report, excluded, summary = filtered_table_from_current_rules(
        table,
        {"cell_populations": [{"name": "Empty", "cell_qc_limits": "",
                               "mask_qc_limits": "", "exclude_from_csv": True}]},
        lambda _text: {},
    )

    assert kept["CellID"].tolist() == [1, 2]
    assert excluded == set()
    assert report["CellGroup_Empty"].tolist() == [False, False]
    assert report["CellGroups"].tolist() == ["", ""]
    assert summary["group_counts"] == {"Empty": 0}


def test_empty_cell_group_has_no_preview_members(monkeypatch, tmp_path):
    preview_file = tmp_path / "preview.tif"
    preview_file.write_bytes(b"preview placeholder")
    preview = PreviewFilterSyncHarness(preview_file)
    preview.commit_gui_edits = lambda: None
    captured = []
    monkeypatch.setattr(
        preview_filter_gui, "find_preview_filter_data",
        lambda *_args: (
            type("Data", (), {
                "source_label": "GFP",
                "table": pd.DataFrame({"CellID": [1, 2]}),
                "label_image": np.array([[1, 2]], dtype=np.uint16),
            })(), "",
        ),
    )
    monkeypatch.setattr(preview_filter_gui, "find_image_def_for_label", lambda defs, _label: defs[0])
    monkeypatch.setattr(preview, "rebuild_preview_population_layers", captured.extend)

    preview.refresh_preview_filter_overlay()

    assert preview.unavailable is None
    assert len(captured) == 1
    assert captured[0]["labels"] == set()


def test_group_exclusion_does_not_select_cells_with_undefined_intensity():
    table = pd.DataFrame({"CellID": [1, 2, 3], "Mean": [5.0, np.nan, 1.0]})
    definition = {"cell_populations": [{
        "name": "Bright", "cell_qc_limits": "Mean intensity:4-",
        "mask_qc_limits": "", "exclude_from_csv": True,
    }]}
    kept, report, excluded, _ = filtered_table_from_current_rules(
        table, definition, lambda text: {"Mean intensity": {"min": 4}} if text else {},
    )
    assert report["CellGroup_Bright"].tolist() == [True, False, False]
    assert excluded == {1}
    assert kept["CellID"].tolist() == [2, 3]


def test_either_category_ignores_unconfigured_category():
    table = pd.DataFrame({"CellID": [1, 2], "Area": [10, 30]})
    for active_key in ("cell_qc_limits", "mask_qc_limits"):
        group = {"name": "Small", "cell_qc_limits": "", "mask_qc_limits": "",
                 "qc_filter_mode": "Exclude only if both fail", "exclude_from_csv": True}
        group[active_key] = "Area:-15"
        definition = {"cell_populations": [group]}
        def parse(text):
            return {"Area": {"max": 15}} if text else {}
        failed, _ = excluded_labels_from_current_rules(table, group, parse)
        kept, report, excluded, _ = filtered_table_from_current_rules(table, definition, parse)
        assert failed == {2}
        assert report["CellGroup_Small"].tolist() == [True, False]
        assert excluded == {1}
        assert kept["CellID"].tolist() == [2]


def test_either_category_keeps_active_category_that_passes_every_cell():
    table = pd.DataFrame({"CellID": [1, 2], "Area": [10, 30]})
    failed, _ = excluded_labels_from_current_rules(
        table,
        {"cell_qc_limits": "strict", "mask_qc_limits": "lenient",
         "qc_filter_mode": "Exclude only if both fail"},
        lambda text: {"Area": {"max": 15 if text == "strict" else 40}},
    )
    assert failed == set()
