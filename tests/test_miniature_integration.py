"""Small real-file integration checks for the native TIFF pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tifffile

from cellonaut.io.image_io import get_tiff_stack_info, read_tiff_numpy_2d
from cellonaut.masks.preview_filter_overlay import attach_live_mask_metrics
from cellonaut.masks.cell_qc import evaluate_cell_qc_table, make_mask_qc_metric_table
from cellonaut.pipeline.discovery import STRUCTURE_FLAT_TIFFS
from cellonaut.pipeline.models import Config, ImageDef, MeasurementTarget
from cellonaut.pipeline.runner import run_pipeline


def _miniature_config(dataset: dict[str, Any]) -> Config:
    return Config(
        fiji_app_path=Path(dataset["input_dir"]) / "missing-fiji",
        input_dir=Path(dataset["input_dir"]),
        output_dir=Path(dataset["output_dir"]),
        input_structure=STRUCTURE_FLAT_TIFFS,
        images=[
            ImageDef("signal", "Signal", "Signal", None, stack_channel_index=1),
            ImageDef("reference", "Reference", "Reference", None, stack_channel_index=2),
        ],
        exclusion_tag="_excluded_",
        threshold_method="Default",
        probability_class_index=1,
        measurement_options={
            "area": True,
            "mean": True,
        },
        measurement_targets=[MeasurementTarget(source_image_key="signal")],
    )


def test_miniature_ome_fixture_round_trips_channel_metadata_and_pixels(miniature_tiff_dataset):
    dataset = miniature_tiff_dataset

    info = get_tiff_stack_info(dataset["stack_path"])
    signal = read_tiff_numpy_2d(dataset["stack_path"], stack_channel_index=1)
    reference = read_tiff_numpy_2d(dataset["stack_path"], stack_channel_index=2)

    assert info["channel_count"] == 2
    assert info["channel_names"] == ["Signal", "Reference"]
    assert info["axes"] == "CYX"
    assert np.array_equal(signal, dataset["signal"])
    assert np.array_equal(reference, dataset["reference"])


def test_miniature_native_pipeline_writes_real_tables_status_and_manifest(miniature_tiff_dataset):
    dataset = miniature_tiff_dataset
    cfg = _miniature_config(dataset)
    messages: list[str] = []

    result = run_pipeline(cfg, log_func=messages.append)

    assert result["status"] == "completed", "\n".join(messages)
    assert result["run_stats"]["total_samples"] == 1
    assert result["run_stats"]["processed"] == 1
    assert result["run_stats"]["failed"] == 0
    assert [row["status"] for row in result["sample_status"]] == ["PROCESSED"]
    assert all(row["target"] == "signal" for row in result["sample_status"])
    assert any("opening TIFFs without ImageJ" in message for message in messages)
    assert not cfg.fiji_app_path.exists()

    summary_path = cfg.output_dir / "Results" / "CSV Data" / "Measurements.csv"
    assert summary_path.is_file()
    table = pd.read_csv(summary_path)
    assert table.columns.tolist() == ["Sample"]
    assert table["Sample"].tolist() == [dataset["stack_path"].name]
    assert not (cfg.output_dir / "Results" / "CSV Data" / "Measurements_Detailed.csv").exists()
    assert not (cfg.output_dir / "Results" / "CSV Data" / "Measurements_Tidy.csv").exists()
    sample_values = [str(value) for value in table["Sample"].tolist()]
    assert any("miniature_sample.ome.tif" in value for value in sample_values)

    manifest_path = Path(result["manifest_paths"]["json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_type"] == "full"
    assert manifest["run_stats"]["processed"] == 1
    assert len(manifest["sample_status"]) == 1
    assert manifest["input_structure"] == STRUCTURE_FLAT_TIFFS
    assert manifest["measurement_targets"][0]["source_image_key"] == "signal"


def test_miniature_pipeline_exports_numeric_cell_measurements(miniature_tiff_dataset, monkeypatch):
    dataset = miniature_tiff_dataset
    cfg = _miniature_config(dataset)
    cfg.cell_use_gpu = False
    cfg.cell_min_size = 0
    cfg.cell_remove_border = False
    target = cfg.measurement_targets[0]
    target.do_cell_segmentation = True
    target.cell_segmentation_source = "signal"

    def segment_fixture(image, _settings, **_kwargs):
        np.testing.assert_array_equal(image, dataset["signal"])
        return tifffile.imread(dataset["cell_labels_path"])

    monkeypatch.setattr("cellonaut.masks.cellpose.run_cell_segmentation_on_image", segment_fixture)
    result = run_pipeline(cfg)

    assert result["status"] == "completed"
    table = pd.read_csv(
        cfg.output_dir / "Results" / "CSV Data" / "Cell Measurements"
        / "miniature_sample.ome_Signal_cell_measurements.csv"
    )
    cells = table[table["CellID"].isin(["1", "2"])]
    assert cells["CellID"].tolist() == ["1", "2"]
    assert cells["Area"].tolist() == [9.0, 6.0]
    assert cells["RawIntDen"].tolist() == [22.0, 52.0]
    assert cells["Mean"].tolist() == pytest.approx([22 / 9, 52 / 6])


def test_miniature_mask_files_produce_exact_qc_metrics_and_flags(miniature_tiff_dataset):
    dataset = miniature_tiff_dataset
    labels = tifffile.imread(dataset["cell_labels_path"])
    positive_mask = tifffile.imread(dataset["positive_mask_path"])
    signal = read_tiff_numpy_2d(dataset["stack_path"], stack_channel_index=1)

    metrics = make_mask_qc_metric_table(labels, positive_mask, signal)

    assert metrics["CellID"].tolist() == [1, 2]
    assert metrics["MaskQC_CellArea"].tolist() == [9, 6]
    assert metrics["PositiveAreaInCell"].tolist() == [4, 3]
    assert metrics["PositiveAreaFractionInCell"].tolist() == [4 / 9, 0.5]
    assert metrics["MeanInPositiveArea"].tolist() == [3.0, 8.0]
    assert metrics["RawIntDenInCell"].tolist() == [12.0, 24.0]
    assert metrics["RawIntDenPerCellArea"].tolist() == [12 / 9, 4.0]
    assert metrics["FractionOfCellIntDen"].tolist() == [12 / 22, 24 / 52]

    cell_table = pd.DataFrame({"CellID": [2, 1], "Area": [6, 9]})
    qc_table, flagged, summary = evaluate_cell_qc_table(
        attach_live_mask_metrics(cell_table, labels, positive_mask, signal),
        {"Mask fraction of cell area": {"max": 0.48}},
    )

    assert flagged == {2}
    assert qc_table["CellID"].tolist() == [2, 1]
    assert qc_table["QC_OutOfRange"].tolist() == [True, False]
    assert qc_table["QC_Reasons"].tolist() == ["Mask fraction of cell area>0.48", ""]
    assert summary["flagged_count"] == 1
