from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from cellonaut.cell_segmentation import core
from cellonaut.cell_segmentation.core import (
    CellSegmentationConfig,
    export_cell_segmentation_diagnostics,
    export_per_cell_organelle_signal_tables,
    make_per_cell_organelle_signal_tables,
    run_cell_segmentation_on_image,
    run_cellpose_segmentation,
)
from cellonaut.config.defaults import CONFIGURED_MASK_WITHIN_CELLPOSE_KEYS
from cellonaut.masks import cellpose as cellpose_pipeline
from cellonaut.masks.cellpose import (
    build_cell_segmentation_cfg_from_pipeline,
    selected_whole_cell_measurements,
)
from cellonaut.measurement.math import summarize_per_cell_table


def test_cellpose_diagnostic_outputs_are_always_enabled(monkeypatch):
    monkeypatch.setattr(cellpose_pipeline, "cellpose_acceleration_enabled", lambda requested: requested)
    pipeline_cfg = SimpleNamespace(
        cell_diameter=70,
        cell_min_size=1250,
        cell_use_gpu=True,
        cellpose_model_type="cpsam",
        cellpose_custom_model_path="",
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        cell_remove_border=True,
    )

    cellpose_cfg = build_cell_segmentation_cfg_from_pipeline(pipeline_cfg)

    assert cellpose_cfg.save_rois_csv is True
    assert cellpose_cfg.save_qc_overlay is True


def test_selected_whole_cell_measurements_use_the_measured_channel():
    labels = np.array([[1, 1, 0], [2, 2, 2]], dtype=np.int32)
    intensity = np.array([[2, 6, 99], [1, 5, 9]], dtype=float)

    measured = selected_whole_cell_measurements(
        labels,
        intensity,
        {
            "cell_area": True,
            "cell_perimeter": True,
            "cell_mean": False,
            "cell_min_max": True,
            "cell_median": True,
            "cell_raw_intden": False,
        },
    )

    assert measured.columns.tolist() == [
        "CellID",
        "CellArea",
        "CellPerimeter",
        "CellMin",
        "CellMax",
        "CellMedian",
    ]
    first = measured.loc[measured["CellID"] == 1].iloc[0]
    second = measured.loc[measured["CellID"] == 2].iloc[0]
    assert first["CellArea"] == 2
    assert first["CellMin"] == 2
    assert first["CellMax"] == 6
    assert first["CellMedian"] == 4
    assert second["CellMedian"] == 5


def test_selected_whole_cell_measurements_expose_extended_native_statistics():
    labels = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]],
        dtype=np.int32,
    )
    intensity = np.array(
        [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 2, 2], [0, 0, 4, 4]],
        dtype=float,
    )
    extended_keys = {
        "cell_std_dev", "cell_mode", "cell_centroid", "cell_center_of_mass",
        "cell_bounding_rect", "cell_fit_ellipse", "cell_feret", "cell_circularity",
        "cell_solidity", "cell_skewness", "cell_kurtosis",
    }

    measured = selected_whole_cell_measurements(
        labels,
        intensity,
        {key: True for key in extended_keys},
    )

    assert measured.columns.tolist() == [
        "CellID", "CellStdDev", "CellMode", "CellCentroidX", "CellCentroidY",
        "CellCenterOfMassX", "CellCenterOfMassY", "CellBoundingRectX", "CellBoundingRectY",
        "CellBoundingRectWidth", "CellBoundingRectHeight", "CellEllipseMajor", "CellEllipseMinor",
        "CellEllipseAngle", "CellFeret", "CellCircularity", "CellSolidity", "CellSkewness",
        "CellKurtosis",
    ]
    first = measured.loc[measured["CellID"] == 1].iloc[0]
    assert first["CellStdDev"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert first["CellMode"] == 1
    assert first["CellCentroidX"] == 0.5
    assert first["CellCentroidY"] == 0.5
    assert first["CellCenterOfMassX"] == pytest.approx(0.6)
    assert first["CellCenterOfMassY"] == pytest.approx(0.7)
    assert first[["CellBoundingRectX", "CellBoundingRectY"]].tolist() == [0.0, 0.0]
    assert first[["CellBoundingRectWidth", "CellBoundingRectHeight"]].tolist() == [2.0, 2.0]
    assert first["CellEllipseMajor"] == pytest.approx(2.0)
    assert first["CellEllipseMinor"] == pytest.approx(2.0)
    assert first["CellFeret"] == pytest.approx(np.sqrt(5))
    assert first["CellCircularity"] == pytest.approx(np.pi)
    assert first["CellSolidity"] == 1
    assert first["CellSkewness"] == pytest.approx(0.0)
    assert first["CellKurtosis"] == pytest.approx(-1.2)


def test_whole_cell_center_of_mass_is_blank_for_zero_intensity_cells():
    measured = selected_whole_cell_measurements(
        np.ones((2, 2), dtype=np.int32),
        np.zeros((2, 2), dtype=float),
        {"cell_center_of_mass": True},
    )

    assert np.isnan(cast(float, measured.loc[0, "CellCenterOfMassX"]))
    assert np.isnan(cast(float, measured.loc[0, "CellCenterOfMassY"]))


def test_pipeline_cellpose_settings_obey_installed_cpu_choice(monkeypatch):
    monkeypatch.setattr(cellpose_pipeline, "cellpose_acceleration_enabled", lambda _requested: False)
    pipeline_cfg = SimpleNamespace(
        cell_diameter=None,
        cell_min_size=15,
        cell_use_gpu=True,
        cellpose_model_type="cpsam",
        cellpose_custom_model_path="",
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        cell_remove_border=False,
        measurement_options={},
    )

    cellpose_cfg = build_cell_segmentation_cfg_from_pipeline(pipeline_cfg)

    assert cellpose_cfg.use_gpu is False
    assert cellpose_cfg.gpu_requested is True


def test_cellpose_pipeline_can_be_disabled_without_inputs():
    overlays, warning = cellpose_pipeline.run_cell_segmentation_for_sample(
        SimpleNamespace(do_cell_segmentation=False),
        {},
        {},
        [],
        {},
        None,
        None,
        None,
        False,
        {},
        "sample",
        "sample",
        lambda _message: None,
    )

    assert overlays == {}
    assert warning == ""


@pytest.mark.parametrize("legacy_filter_settings", [False, True])
def test_cellpose_pipeline_keeps_original_measurements_unfiltered(monkeypatch, tmp_path: Path, legacy_filter_settings):
    labels = np.array([[1, 0, 0], [0, 2, 2], [0, 2, 2]], dtype=np.int32)
    monkeypatch.setattr(
        cellpose_pipeline,
        "run_cell_segmentation_on_image",
        lambda *_args, **_kwargs: labels,
    )
    cfg = SimpleNamespace(
        do_cell_segmentation=True,
        cell_segmentation_source="cell",
        cell_diameter=30,
        cell_min_size=1,
        cell_use_gpu=False,
        cellpose_model_type="cpsam",
        cellpose_custom_model_path="",
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        cell_remove_border=False,
        measurement_options={"cell_area": True, "cell_mean": True, "cell_raw_intden": True},
        reuse_existing_masks=False,
        cell_mask_adjustments={},
    )
    if legacy_filter_settings:
        # Even a loose caller carrying old settings cannot alter originals.
        cfg.cell_qc_rules = {"Area": {"max": 2}}
        cfg.cell_qc_exclude_flagged = True
    source_def = SimpleNamespace(key="cell", label="Cell")
    cfg.images = [source_def]
    source_image = np.arange(9, dtype=np.uint16).reshape(3, 3)
    export_dirs = {
        "cell_segmentation_labels": tmp_path / "labels",
        "cell_segmentation_outlines": tmp_path / "outlines",
        "cell_segmentation_qc_pngs": tmp_path / "qc",
        "cell_segmentation_tables": tmp_path / "tables",
        "cell_signal_tables": tmp_path / "signals",
    }
    row: dict[str, object] = {}

    overlays, warning = cellpose_pipeline.run_cell_segmentation_for_sample(
        cfg,
        {"cell": source_image},
        {},
        [],
        row,
        source_def,
        source_image,
        source_image,
        False,
        export_dirs,
        "sample",
        "sample",
            lambda _message: None,
        )

    assert np.array_equal(overlays["__whole_cell_mask__"], labels)
    assert "__flagged_cell_mask__" not in overlays
    assert row["Cell_CellCount"] == 2
    assert not any("QC" in key for key in row)
    assert row == {
        "CellposeMaskKey": "cell",
        "CellposeMaskLabel": "Cell",
        "Cell_CellCount": 2,
        "Cell_measured_with_Cell_cellpose_mask_PerCell_TotalCellArea": 5.0,
        "Cell_measured_with_Cell_cellpose_mask_PerCell_MeanOfCellMeans": 3.0,
        "Cell_measured_with_Cell_cellpose_mask_PerCell_SumCellIntDen": 24.0,
    }
    assert warning == ""
    assert (export_dirs["cell_segmentation_labels"] / "sample_Cell_01_cellpose_labels.tif").exists()
    assert (export_dirs["cell_segmentation_outlines"] / "sample_Cell_cellpose_outline.tif").exists()
    exported = pd.read_csv(export_dirs["cell_segmentation_tables"] / "sample_Cell_cell_measurements.csv")
    exported_ids = set(exported["CellID"].dropna().astype(str))
    assert "1" in exported_ids
    assert "2" in exported_ids


def test_cellpose_v4_eval_receives_only_supported_arguments(monkeypatch):
    captured = {}

    class FakeModel:
        def eval(self, image, **kwargs):
            captured["shape"] = image.shape
            captured["kwargs"] = kwargs
            return np.ones(image.shape, dtype=np.int32), None, None

    monkeypatch.setattr(core, "get_cellpose_model", lambda _cfg: FakeModel())

    labels = run_cellpose_segmentation(
        np.ones((4, 5), dtype=np.uint16),
        CellSegmentationConfig(diameter=42, cellprob_threshold=-1.0, flow_threshold=0.7),
    )

    assert captured["shape"] == (4, 5)
    assert captured["kwargs"] == {
        "diameter": 42,
        "min_size": 15,
        "cellprob_threshold": -1.0,
        "flow_threshold": 0.7,
    }
    assert labels.dtype == np.int32


def test_cellpose_run_passes_requested_and_effective_gpu_choice_to_log(monkeypatch):
    backend_args = {}
    logs = []

    class FakeModel:
        def eval(self, image, **_kwargs):
            return np.zeros(image.shape, dtype=np.int32), None, None

    monkeypatch.setattr(core, "get_cellpose_model", lambda _cfg: FakeModel())
    monkeypatch.setattr(
        core,
        "describe_cellpose_backend",
        lambda **kwargs: backend_args.update(kwargs) or SimpleNamespace(label="Cellpose backend", detail="CPU fallback"),
    )

    run_cellpose_segmentation(
        np.ones((4, 5), dtype=np.uint16),
        CellSegmentationConfig(use_gpu=False, gpu_requested=True),
        log_func=logs.append,
    )

    assert backend_args["acceleration_allowed"] is False
    assert backend_args["acceleration_requested"] is True
    assert logs == ["Cellpose backend: CPU fallback"]


def test_cellpose_run_logs_actual_inference_device_after_eval(monkeypatch):
    logs = []

    class FakeModel:
        device = SimpleNamespace(type="cpu")

        def eval(self, image, **_kwargs):
            return np.zeros(image.shape, dtype=np.int32), None, None

    monkeypatch.setattr(core, "get_cellpose_model", lambda _cfg: FakeModel())
    monkeypatch.setattr(
        core,
        "describe_cellpose_backend",
        lambda **_kwargs: SimpleNamespace(label="Cellpose backend", detail="CPU fallback"),
    )

    run_cellpose_segmentation(
        np.ones((4, 5), dtype=np.uint16),
        CellSegmentationConfig(use_gpu=False, gpu_requested=True),
        log_func=logs.append,
    )

    assert logs[-1] == "Cellpose inference device: CPU"

def test_cellpose_builtin_model_name_is_passed_explicitly(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    cellpose_module = ModuleType("cellpose")
    cellpose_module.models = SimpleNamespace(CellposeModel=FakeModel)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cellpose", cellpose_module)
    try:
        for model_type in ("cpsam", "cpsam_v2"):
            core.clear_cellpose_model_cache()
            model = cast(
                FakeModel,
                core.get_cellpose_model(CellSegmentationConfig(model_type=model_type, use_gpu=False)),
            )
            assert model.kwargs == {
                "gpu": False,
                "pretrained_model": model_type,
            }
        assert len(created) == 2
    finally:
        core.clear_cellpose_model_cache()



def test_cellpose_models_are_reused_only_for_matching_configs(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    cellpose_module = ModuleType("cellpose")
    cellpose_module.models = SimpleNamespace(CellposeModel=FakeModel)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cellpose", cellpose_module)
    core.clear_cellpose_model_cache()
    try:
        cpu_cfg = CellSegmentationConfig(model_type="cpsam", use_gpu=False)
        first = cast(FakeModel, core.get_cellpose_model(cpu_cfg))
        second = core.get_cellpose_model(cpu_cfg)
        gpu = cast(
            FakeModel,
            core.get_cellpose_model(CellSegmentationConfig(model_type="cpsam", use_gpu=True)),
        )

        assert first is second
        assert gpu is not first
        assert len(created) == 2
        assert first.kwargs == {"gpu": False, "pretrained_model": "cpsam"}
        assert gpu.kwargs == {"gpu": True, "pretrained_model": "cpsam"}
    finally:
        core.clear_cellpose_model_cache()


def test_cellpose_model_cache_is_bounded_and_can_be_cleared(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, **kwargs):
            created.append(self)

    cellpose_module = ModuleType("cellpose")
    cellpose_module.models = SimpleNamespace(CellposeModel=FakeModel)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cellpose", cellpose_module)
    core.clear_cellpose_model_cache()
    try:
        first_cfg = CellSegmentationConfig(model_type="model-1", use_gpu=False)
        core.get_cellpose_model(first_cfg)
        core.get_cellpose_model(CellSegmentationConfig(model_type="model-2", use_gpu=False))
        core.get_cellpose_model(CellSegmentationConfig(model_type="model-3", use_gpu=False))
        assert len(core._CELLPOSE_MODEL_CACHE) == core._CELLPOSE_MODEL_CACHE_LIMIT

        core.get_cellpose_model(first_cfg)
        assert len(created) == 4

        core.clear_cellpose_model_cache()
        assert not core._CELLPOSE_MODEL_CACHE
    finally:
        core.clear_cellpose_model_cache()


@pytest.mark.parametrize("remove_border", [False, True])
def test_cell_segmentation_cleanup_removes_small_and_border_cells(monkeypatch, remove_border):
    raw_labels = np.zeros((10, 12), dtype=np.int32)
    raw_labels[2, 2:4] = 4       # Interior, area 2: below threshold.
    raw_labels[2, 6:9] = 8      # Interior, area 3: exactly at threshold.
    raw_labels[5:7, 2:5] = 12   # Interior, area 6: above threshold.
    raw_labels[0, 6:10] = 20    # Border, area 4: size filter must retain it.
    original = raw_labels.copy()
    monkeypatch.setattr(core, "run_cellpose_segmentation", lambda *_args, **_kwargs: raw_labels)

    labels = run_cell_segmentation_on_image(
        np.ones(raw_labels.shape, dtype=np.uint8),
        CellSegmentationConfig(min_size=3, remove_border=remove_border),
    )

    expected = np.zeros_like(raw_labels)
    expected[2, 6:9] = 1
    expected[5:7, 2:5] = 2
    if not remove_border:
        expected[0, 6:10] = 3
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_array_equal(raw_labels, original)


def test_per_cell_organelle_table_uses_only_mask_pixels_inside_each_cell():
    cell_labels = np.array([[1, 1, 0], [1, 1, 2], [0, 2, 2]], dtype=np.int32)
    organelle_mask = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    intensity = np.array([[10, 20, 0], [30, 40, 50], [0, 60, 70]], dtype=np.float64)

    biology, geometry = make_per_cell_organelle_signal_tables(
        cell_labels,
        organelle_mask,
        intensity,
        organelle_prefix="Mask",
    )

    first = biology.loc[biology["CellID"] == 1].iloc[0]
    second = biology.loc[biology["CellID"] == 2].iloc[0]
    assert first["MaskArea_InCell"] == 2
    assert first["MaskIntDen_InCell"] == 50
    assert second["MaskArea_InCell"] == 1
    assert second["MaskIntDen_InCell"] == 50
    assert set(geometry["CellID"]) == {1, 2}


def test_per_cell_organelle_table_keeps_empty_means_and_omits_ratios():
    cell_labels = np.ones((2, 2), dtype=np.int32)
    empty_mask = np.zeros((2, 2), dtype=np.uint8)
    zero_intensity = np.zeros((2, 2), dtype=np.float64)

    biology, _geometry = make_per_cell_organelle_signal_tables(
        cell_labels,
        empty_mask,
        zero_intensity,
        organelle_prefix="Mask",
        corrected_intensity_img=zero_intensity,
    )

    row = biology.iloc[0]
    assert row["MaskIntDen_InCell"] == 0
    assert np.isnan(row["MaskMean_InCell"])
    assert not any("Fraction" in column or "Ratio" in column for column in biology.columns)


def test_per_cell_export_honors_metric_choices_in_table_and_summary(tmp_path: Path):
    output = tmp_path / "per_cell.csv"
    biology, _geometry = export_per_cell_organelle_signal_tables(
        cell_label_img=np.ones((2, 2), dtype=np.int32),
        organelle_mask=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        intensity_img=np.array([[2, 4], [6, 8]], dtype=float),
        out_biology_csv=output,
        organelle_prefix="Mask",
        measurement_options={
            "positive_area_in_cell": True,
            "mean_in_positive_area": False,
            "std_dev_in_positive_area": True,
            "min_max_in_positive_area": True,
            "median_in_positive_area": True,
            "raw_intden_in_cell": False,
        },
    )

    saved = pd.read_csv(output)
    assert biology.loc[0, "MaskArea_InCell"] == 2
    assert "MaskArea_InCell" in saved.columns
    assert "MaskMean_InCell" not in biology.columns
    assert "MaskIntDen_InCell" not in biology.columns
    assert biology.loc[0, "MaskStdDev_InCell"] == pytest.approx(np.sqrt(18.0))
    assert biology.loc[0, "MaskMin_InCell"] == 2
    assert biology.loc[0, "MaskMax_InCell"] == 8
    assert biology.loc[0, "MaskMedian_InCell"] == 5
    assert "MaskStdDev_InCell" in saved.columns
    assert "MaskMin_InCell" in saved.columns
    assert "MaskMax_InCell" in saved.columns
    assert "MaskMedian_InCell" in saved.columns
    assert not any("Fraction" in column or "Ratio" in column for column in saved.columns)


def test_configured_mask_within_cells_supports_the_complete_measurement_set(tmp_path: Path):
    labels = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]],
        dtype=np.int32,
    )
    intensity = np.array(
        [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 2, 2], [0, 0, 4, 4]],
        dtype=float,
    )
    output = tmp_path / "all_mask_within_cell_measurements.csv"

    biology, _geometry = export_per_cell_organelle_signal_tables(
        cell_label_img=labels,
        organelle_mask=labels > 0,
        intensity_img=intensity,
        out_biology_csv=output,
        organelle_prefix="Mask",
        measurement_options={key: True for key in CONFIGURED_MASK_WITHIN_CELLPOSE_KEYS},
    )

    expected_columns = {
        "MaskArea_InCell", "MaskMean_InCell", "MaskStdDev_InCell", "MaskMode_InCell",
        "MaskMin_InCell", "MaskMax_InCell", "MaskCentroidX_InCell", "MaskCentroidY_InCell",
        "MaskCenterOfMassX_InCell", "MaskCenterOfMassY_InCell", "MaskPerimeter_InCell",
        "MaskBoundingRectX_InCell", "MaskBoundingRectY_InCell", "MaskBoundingRectWidth_InCell",
        "MaskBoundingRectHeight_InCell", "MaskEllipseMajor_InCell", "MaskEllipseMinor_InCell",
        "MaskEllipseAngle_InCell", "MaskFeret_InCell", "MaskCircularity_InCell",
        "MaskSolidity_InCell", "MaskIntDen_InCell", "MaskMedian_InCell",
        "MaskSkewness_InCell", "MaskKurtosis_InCell",
    }
    assert expected_columns.issubset(biology.columns)
    expected_saved_columns = (expected_columns - {"MaskMean_InCell"}) | {
        "MaskMeanGrayValue_InCell"
    }
    assert expected_saved_columns.issubset(pd.read_csv(output).columns)

    first = biology.loc[biology["CellID"] == 1].iloc[0]
    assert first["MaskArea_InCell"] == 4
    assert first["MaskMean_InCell"] == 2.5
    assert first["MaskMode_InCell"] == 1
    assert first["MaskCenterOfMassX_InCell"] == pytest.approx(0.6)
    assert first["MaskCenterOfMassY_InCell"] == pytest.approx(0.7)
    assert first["MaskFeret_InCell"] == pytest.approx(np.sqrt(5))
    assert first["MaskCircularity_InCell"] == pytest.approx(np.pi)
    assert first["MaskSolidity_InCell"] == 1
    assert first["MaskSkewness_InCell"] == pytest.approx(0.0)
    assert first["MaskKurtosis_InCell"] == pytest.approx(-1.2)

    summary = summarize_per_cell_table(biology, "Mask", "GFP", "DIA")
    prefix = "GFP_measured_with_Mask_mask_PerCell"
    assert summary[f"{prefix}_TotalMaskArea"] == 8
    assert summary[f"{prefix}_TotalMaskPerimeter"] == pytest.approx(8)
    assert summary[f"{prefix}_MeanOfMaskModes"] == pytest.approx(1.5)
    assert summary[f"{prefix}_MeanMaskCircularity"] == pytest.approx(np.pi)
    assert summary[f"{prefix}_MeanMaskSolidity"] == 1


def test_export_cell_segmentation_diagnostics_writes_roi_outline_csv_when_enabled(tmp_path: Path):
    cfg = CellSegmentationConfig(
        save_rois_csv=True,
        save_qc_overlay=False,
    )
    labels = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.int32,
    )
    qc_png_dir = tmp_path / "Cells" / "PNG" / "Batch1"
    table_dir = tmp_path / "CSV Data" / "Cell Measurements" / "Batch1"

    results = export_cell_segmentation_diagnostics(
        labels,
        labels,
        qc_png_dir,
        table_dir,
        "sample",
        cfg,
    )

    roi_path = table_dir / "sample_ROI_outlines.csv"
    assert results["roi_outlines_csv"] == str(roi_path)
    assert roi_path.exists()
    assert set(pd.read_csv(roi_path)["CellID"]) == {1}
