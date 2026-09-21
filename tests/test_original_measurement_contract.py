"""Original measurements, preset compatibility, and derived Cell Group exports."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

from cellonaut.config.adapter import build_pipeline_config_from_gui_state, parse_cell_qc_limits_text
from cellonaut.config.defaults import INPUT_STRUCTURE_FLAT_TIFFS, default_image_definition
from cellonaut.config.preset_store import read_json_object, write_json_object
from cellonaut.config.state import CellonautGuiState, ImageGuiState
from cellonaut.masks.preview_filter_overlay import (
    export_all_filtered_result_tables,
    filtered_table_from_current_rules,
    find_preview_filter_data,
)
from cellonaut.measurement.exports import _with_legacy_cell_count_columns, write_measurement_summary_csvs
from cellonaut.pipeline import runner
from cellonaut.pipeline.models import Config, ImageDef
from cellonaut.pipeline.summary import build_pipeline_summary_dict
from cellonaut.results.layout import build_results_layout


LEGACY_SETTINGS = {
    "cell_qc_limits", "mask_qc_limits", "cell_qc_exclude_flagged",
    "qc_filter_mode", "mask_qc_intensity_source",
}


@pytest.mark.parametrize("enabled", [True, "true", False, "false"])
def test_legacy_preset_migrates_once_and_current_presets_save_only_groups(tmp_path, enabled):
    legacy = {
        "name": "GFP", "cell_qc_limits": "Area:2-10", "mask_qc_limits": "",
        "qc_filter_mode": "Use cell filters only", "mask_qc_intensity_source": "Cell mask image",
        "cell_qc_exclude_flagged": enabled,
    }
    old_path = tmp_path / "old.json"
    write_json_object(old_path, {"image_definitions": [legacy]})
    loaded = ImageGuiState.from_dict(read_json_object(old_path)["image_definitions"][0], 0)
    current = loaded.to_dict()
    assert not LEGACY_SETTINGS.intersection(current)
    assert not LEGACY_SETTINGS.intersection(default_image_definition(0))
    group = current["cell_populations"][0]
    assert group["cell_qc_limits"] == "Area:2-10"
    assert group["qc_filter_mode"] == "Use cell filters only"
    assert group["mask_qc_intensity_source"] == "Cell mask image"
    assert group["exclude_from_csv"] is (enabled in (True, "true"))
    current_path = tmp_path / "current.json"
    write_json_object(current_path, {"image_definitions": [current]})
    restored = ImageGuiState.from_dict(read_json_object(current_path)["image_definitions"][0], 0)
    assert restored.to_dict() == current
    table = pd.DataFrame({"CellID": [1, 2, 3], "Area": [1, 5, 20]})
    kept, _, excluded, _ = filtered_table_from_current_rules(table, current, parse_cell_qc_limits_text)
    assert excluded == ({2} if group["exclude_from_csv"] else set())
    assert kept["CellID"].tolist() == ([1, 3] if group["exclude_from_csv"] else [1, 2, 3])


def test_explicit_groups_override_stale_mirrors_and_partial_groups_keep_old_defaults():
    legacy = {"cell_qc_limits": "Area:2-10", "mask_qc_intensity_source": "Cell mask image",
              "cell_qc_exclude_flagged": True,
              "cell_populations": [{"name": "No conditions", "cell_qc_limits": "", "exclude_from_csv": False},
                                   {"name": "Partial"}]}
    original = deepcopy(legacy)
    current = ImageGuiState.from_dict(legacy, 0).to_dict()
    assert legacy == original
    assert not LEGACY_SETTINGS.intersection(current)
    empty, partial = current["cell_populations"]
    assert empty["cell_qc_limits"] == ""
    assert empty["exclude_from_csv"] is False
    assert partial["cell_qc_limits"] == "Area:2-10"
    assert partial["mask_qc_intensity_source"] == "Cell mask image"
    assert partial["exclude_from_csv"] is False


def test_group_choices_do_not_enter_execution_or_change_pipeline_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("cellonaut.pipeline.summary.build_runtime_metadata", lambda _cfg: {})
    base = {"name": "GFP", "analysis_cell_segmentation_enabled": True, "cell_diameter": "30"}
    first = CellonautGuiState.from_widget_values(
        fiji_app_path="", input_dir=str(tmp_path / "in"), output_dir=str(tmp_path / "out"),
        input_structure=INPUT_STRUCTURE_FLAT_TIFFS, image_definitions=[base],
    )
    second = deepcopy(first)
    second.image_definitions[0].cell_populations = [
        {"name": "Small", "cell_qc_limits": "Area:-20", "exclude_from_csv": True}
    ]
    original_cfg = build_pipeline_config_from_gui_state(first)
    grouped_cfg = build_pipeline_config_from_gui_state(second)
    assert asdict(original_cfg) == asdict(grouped_cfg)
    assert build_pipeline_summary_dict(original_cfg) == build_pipeline_summary_dict(grouped_cfg)
    for obj in (original_cfg, *original_cfg.measurement_targets):
        for key in ("cell_qc_rules", "mask_qc_rules", "cell_qc_exclude_flagged", "qc_filter_mode",
                    "mask_qc_intensity_source", "cell_populations"):
            assert not hasattr(obj, key)


def test_original_export_matches_historical_csv_schema_without_mutating_measurements(tmp_path):
    row = {"Label": "sample__GFP", "SourceImageLabel": "GFP", "GFP_in_Mask_Area": 5.0,
           "GFP_CellCount": 2, "GFP_measured_with_GFP_cellpose_mask_PerCell_TotalCellArea": 5.0}
    historical = {"Label": "sample__GFP", "SourceImageLabel": "GFP", "GFP_in_Mask_Area": 5.0,
                  "GFP_CellCount_TotalBeforeQC": 2, "GFP_CellCount": 2,
                  "GFP_CellQC_OutOfRangeCount": 0, "GFP_MaskQC_OutOfRangeCount": 0,
                  "GFP_QC_ExcludedCount": 0, "GFP_CellQC_ExcludedCount": 0,
                  "GFP_measured_with_GFP_cellpose_mask_PerCell_TotalCellArea": 5.0}
    original = deepcopy(row)
    paths = []
    for name, value in (("new", row), ("old", historical)):
        paths.append(write_measurement_summary_csvs(
            [value], cfg=SimpleNamespace(output_dir=tmp_path / name, input_structure=INPUT_STRUCTURE_FLAT_TIFFS),
            log_func=lambda _message: None,
        ))
    assert paths[0].keys() == paths[1].keys()
    for key in paths[0]:
        pd.testing.assert_frame_equal(pd.read_csv(paths[0][key]), pd.read_csv(paths[1][key]))
    assert row == original


def test_serialization_keeps_historical_exclusions_and_missing_channel_values():
    table = pd.DataFrame([{"Label": "old", "GFP_CellCount": 2, "GFP_QC_ExcludedCount": 3},
                          {"Label": "other", "DAPI_CellCount": 7}])
    original = table.copy(deep=True)
    exported = _with_legacy_cell_count_columns(table)
    pd.testing.assert_frame_equal(table, original)
    assert exported.loc[0, "GFP_QC_ExcludedCount"] == 3
    assert exported.loc[1, "DAPI_CellCount_TotalBeforeQC"] == 7
    assert exported.loc[1, "DAPI_CellQC_ExcludedCount"] == 0
    assert pd.isna(exported.loc[1, "GFP_CellQC_OutOfRangeCount"])


def test_group_filter_and_export_leave_original_rows_and_old_result_files_unchanged(tmp_path):
    layout = build_results_layout(tmp_path)
    for key in ("cell_segmentation_tables", "cell_segmentation_labels", "mask_overlays"):
        layout[key].mkdir(parents=True, exist_ok=True)
    table_path = layout["cell_segmentation_tables"] / "sample_GFP_cell_measurements.csv"
    table = pd.DataFrame({"CellID": [1, 2], "Area": [1, 4], "MeanGrayValue": [10, 20],
                          "QC_Excluded": [False, False]})
    table.to_csv(table_path, index=False)
    labels_path = layout["cell_segmentation_labels"] / "sample_GFP_01_cellpose_labels.tif"
    tifffile.imwrite(labels_path, np.array([[1, 0, 0], [0, 2, 2], [0, 2, 2]], dtype=np.int32))
    overlay = layout["mask_overlays"] / "sample_GFP_combined_overlay.tif"
    tifffile.imwrite(overlay, np.ones((3, 3), dtype=np.uint16))
    files_before = {p: p.read_bytes() for p in layout["root"].rglob("*") if p.is_file()}
    original = table.copy(deep=True)
    settings = {"name": "GFP", "cell_populations": [
        {"name": "Small", "cell_qc_limits": "Area:-2", "exclude_from_csv": True}]}
    kept, report, excluded, _ = filtered_table_from_current_rules(table, settings, parse_cell_qc_limits_text)
    assert kept["CellID"].tolist() == [2]
    assert excluded == {1}
    assert report["CellGroup_ExcludedFromCSV"].tolist() == [True, False]
    pd.testing.assert_frame_equal(table, original)
    data, message = find_preview_filter_data(overlay, [settings])
    assert data is not None and not message
    assert data.table["CellID"].tolist() == [1, 2]
    result = export_all_filtered_result_tables(layout["root"], [settings], parse_cell_qc_limits_text)
    assert result.processed_count == 1 and result.kept_combined_path is not None
    assert pd.read_csv(result.kept_combined_path)["CellID"].tolist() == [2]
    assert all(p.read_bytes() == content for p, content in files_before.items())


def test_preview_returns_only_authoritative_rows(make_sample_context, tmp_path, monkeypatch):
    from cellonaut.pipeline.models import SampleTargetStatus

    cfg = Config(
        fiji_app_path=tmp_path / "Fiji", input_dir=tmp_path / "input", output_dir=tmp_path,
        input_structure=INPUT_STRUCTURE_FLAT_TIFFS, images=[ImageDef("gfp", "GFP", "GFP", None)],
        exclusion_tag="", threshold_method="Otsu", probability_class_index=1,
    )
    target = SimpleNamespace(source_image_key="gfp", overlay_base_image_key="gfp", overlay_roi_keys=[],
                             do_cell_segmentation=False, overlay_whole_cell_mask=False)
    row = {"Label": "sample__GFP", "GFP_in_Mask_Area": 5.0}
    monkeypatch.setattr(runner, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(runner, "build_pipeline_summary_dict", lambda _cfg: {})
    monkeypatch.setattr(runner, "get_measurement_sample_paths", lambda _cfg: [tmp_path / "sample.tif"])
    monkeypatch.setattr(runner, "get_enabled_measurement_targets", lambda _cfg: [target])
    monkeypatch.setattr(runner, "_initialize_preview_runtime", lambda *_args: None)
    monkeypatch.setattr(runner, "build_sample_label", lambda *_args: "sample")
    monkeypatch.setattr(runner, "build_sample_processing_context", lambda **_kwargs: make_sample_context())
    monkeypatch.setattr(runner, "_process_targets_in_context", lambda **_kwargs: runner._TargetBatchResult(
        [row], ["gfp"], [SampleTargetStatus("preview", "sample", "gfp", "PROCESSED")]))
    monkeypatch.setattr(runner, "collect_preview_artifacts", lambda _root: {})
    monkeypatch.setattr(runner, "save_run_manifest", lambda *_args, **_kwargs: {"text": "", "json": ""})
    monkeypatch.setattr(runner, "write_measurement_summary_csvs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "get_image_def", lambda *_args: SimpleNamespace(label="GFP"))
    result = runner.run_preview_pipeline(cfg, log_func=lambda _message: None)
    assert result["rows"] == [row]
    assert "qc_rows" not in result


def test_duplicate_qc_row_flow_has_no_production_references():
    package = Path(__file__).resolve().parents[1] / "cellonaut"
    assert not [path for path in package.rglob("*.py") if "qc_row" in path.read_text(encoding="utf-8")]
