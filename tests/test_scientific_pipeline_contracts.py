"""Regression cases for plane identity, empty masks, and exported sample identity."""

from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast

import numpy as np
import pandas as pd
import pytest

from cellonaut.masks import roi_processing as rois
from cellonaut.measurement import execution
from cellonaut.measurement.exports import write_measurement_summary_csvs
from cellonaut.measurement.table_formatting import measurement_summary_table, tidy_measurements_table
from cellonaut.measurement.tables import normalize_export_table
from cellonaut.pipeline.discovery import STRUCTURE_GROUPED_BY_PROTEIN
from cellonaut.pipeline.models import ImageDef
from cellonaut.results.comparison import ComparableRun, _measurement_values


def test_bundled_tutorial_measurements_match_source_pixels_and_final_masks():
    import tifffile

    assets = Path(__file__).resolve().parents[1] / "cellonaut" / "data" / "assets"
    source = np.asarray(tifffile.imread(assets / "Cellonaut_32bit_input" / "Cellonaut_32bits.tif"))
    results = assets / "Cellonaut_32bit_output" / "preview_1" / "Results"
    exported = pd.read_csv(results / "CSV Data" / "Measurements.csv").iloc[0]
    mask_dir = results / "Masks" / "BinaryMasks"

    for class_index in (1, 2, 3):
        mask_path = mask_dir / f"Cellonaut_32bits_Cellonaut_mask_Class{class_index}_binary.tif"
        foreground = np.asarray(tifffile.imread(mask_path)) > 0
        pixels = np.asarray(source[foreground], dtype=np.float64)
        prefix = f"Cellonaut(Cellonaut mask_Class{class_index})"

        assert exported[f"{prefix} : Area (px²)"] == int(foreground.sum())
        assert exported[f"{prefix} : Mean intensity (a.u.)"] == pytest.approx(float(pixels.mean()))
        assert exported[f"{prefix} : Raw integrated density (a.u. × px²)"] == pytest.approx(
            float(pixels.sum(dtype=np.float64))
        )


@pytest.mark.parametrize("same_plane", [False, True])
def test_weka_cache_uses_opened_plane_even_when_channel_was_inferred(monkeypatch, tmp_path, same_plane):
    model = tmp_path / "classifier.model"
    model.write_bytes(b"model")
    images = [ImageDef(key, key, key, model) for key in ("first", "second")]
    first_plane = object()
    second_plane = first_plane if same_plane else object()
    classified = []

    def classify(image, *_args):
        classified.append(image)
        return NS(source=image, close=lambda: None)

    monkeypatch.setattr(rois, "get_java_classes", lambda: {"FileSaver": object})
    monkeypatch.setattr(rois, "save_imagej_tiff", lambda *_args: True)
    monkeypatch.setattr(rois, "build_roi", lambda result, *_args, **_kwargs: {1: result.source})
    cfg = NS(output_dir=tmp_path, mask_source_dir=None, reuse_existing_masks=False, probability_class_index=1)
    roi_map, _ = rois.prepare_rois_for_defs(
        image_map={"first": first_plane, "second": second_plane},
        cfg=cfg,
        out_path=tmp_path / "Masks",
        result_id="sample",
        roi_defs=images,
        segs={key: NS(applyClassifier=classify) for key in ("first", "second")},
        file_map={key: tmp_path / "same.ome.tif" for key in ("first", "second")},
        log_func=lambda _message: None,
    )
    assert roi_map == {"first": first_plane, "second": second_plane}
    assert len(classified) == (1 if same_plane else 2)


def test_empty_mask_measurements_contribute_zero_to_summary(monkeypatch):
    monkeypatch.setattr(execution, "measure_roi_stats", lambda *_args: {"Area": 100.0, "RawIntDen": 500.0, "Mean": 5.0})
    rows = []
    for name, roi in (("empty", None), ("positive", object())):
        row = {"Label": name}
        execution.measure_rois_for_source(
            cfg=NS(measurement_options={}),
            roi_defs=[NS(key="mask", label="Mask")],
            roi_map={"mask": roi},
            row=row,
            source_def=NS(label="GFP"),
            source_measure_img=object(),
            measurement_source_img=object(),
            has_source_background_subtraction=True,
            corrected_suffix="RB40p0",
            id_label=name,
            log_func=lambda _message: None,

        )
        rows.append(row)
    assert np.isnan(rows[0]["GFP_in_Mask_Mean"])
    summary = measurement_summary_table(tidy_measurements_table(pd.DataFrame(rows), sample_column="Label"))
    summary = summary.set_index("Measurement")
    assert list(cast(pd.Series, summary.loc["GFP_in_Mask_Area", ["N", "Mean"]])) == [2, 50.0]
    assert list(cast(pd.Series, summary.loc["GFP_in_Mask_RawIntDen", ["N", "Mean"]])) == [2, 250.0]
    assert list(cast(pd.Series, summary.loc["GFP_in_Mask_RB40p0_IntDen", ["N", "Mean"]])) == [2, 250.0]
    assert summary.loc["GFP_in_Mask_Mean", "N"] == 1


def test_missing_mask_is_not_reported_as_zero():
    with pytest.raises(ValueError, match="unavailable"):
        execution.measure_rois_for_source(
            cfg=NS(measurement_options={}),
            roi_defs=[NS(key="mask", label="Mask")],
            roi_map={},
            row={},
            source_def=NS(label="GFP"),
            source_measure_img=object(),
            measurement_source_img=object(),
            has_source_background_subtraction=False,
            corrected_suffix="",
            id_label="sample",
            log_func=lambda _message: None,

        )


def test_empty_reused_mask_does_not_require_regeneration(monkeypatch, tmp_path):
    mask_path = rois.threshold_mask_path(tmp_path, "sample", "Mask", "classifier", 1)
    mask_path.write_bytes(b"mask")
    image = NS(getProcessor=lambda: NS(getHistogram=lambda: [12, 0, 0]), getRoi=lambda: None, close=lambda: None)
    monkeypatch.setattr(
        rois,
        "get_java_classes",
        lambda: {
            "IJ": NS(openImage=lambda _path: image, setThreshold=lambda *_args: None, run=lambda *_args: None),
            "ShapeRoi": object,
        },
    )
    assert rois.load_existing_class_rois(tmp_path, "sample", "Mask", "classifier", 1, lambda _msg: None) == {1: None}


def test_empty_background_sweep_has_zero_total_signal(monkeypatch):
    monkeypatch.setattr(execution, "subtract_background_copy", lambda *_args: NS(close=lambda: None))
    row = {}
    execution.run_source_background_sweep(
        roi_defs=[NS(key="mask", label="Mask")],
        roi_map={"mask": None},
        row=row,
        source_def=NS(label="GFP"),
        measurement_base_img=object(),
        source_radii=[40.0, 70.0],
        created_temp_images=[],
        id_label="sample",
        log_func=lambda _msg: None,

    )
    assert row["GFP_in_Mask_RB70p0_IntDen"] == 0.0


def test_exports_preserve_group_names_and_distinct_samples(tmp_path):
    rows = []
    for group, value in (("Drug_A", 10.0), ("Drug_B", 100.0)):
        rows.append(
            {
                "Label": f"{group}_sample__GFP",
                "SampleID": f"{group}_sample",
                "SampleRelativePath": f"{group}/sample",
                "SampleGroup": group,
                "OriginalSamplePath": str(tmp_path / "input" / group / "sample"),
                "SourceImageKey": "gfp",
                "SourceImageLabel": "GFP",
                "GFP_in_Mask_Area": value,
            }
        )
    written = write_measurement_summary_csvs(
        rows,
        cfg=NS(output_dir=tmp_path / "output", input_structure=STRUCTURE_GROUPED_BY_PROTEIN),
        log_func=lambda _msg: None,
    )
    summary = pd.read_csv(written["summary"])
    assert summary["Group"].tolist() == ["Drug_A", "Drug_B"]
    assert summary["Mean"].tolist() == [10.0, 100.0]
    assert summary["N"].tolist() == [1, 1]
    for key in ("all", "channel:GFP"):
        table = pd.read_csv(written[key])
        assert table["Sample"].tolist() == ["Drug_A/sample", "Drug_B/sample"]
        assert normalize_export_table(table)["Sample"].tolist() == ["Drug_A/sample", "Drug_B/sample"]
    transposed = pd.read_csv(written["all_long"])
    assert "Drug_A/sample" in transposed.columns and "Drug_B/sample" in transposed.columns
    run = ComparableRun(
        operation_dir=tmp_path / "output",
        results_dir=tmp_path / "output" / "Results",
        measurements_path=Path(written["all"]),
        manifest_path=tmp_path / "manifest.json",
        label="run_1",
        operation_type="run",
        sequence_number=1,
    )
    values = _measurement_values(run, {"input_directory": str(tmp_path / "input")})
    assert {sample for sample, _metric in values} == {"Drug_A/sample", "Drug_B/sample"}
    assert sorted(values.values()) == [10.0, 100.0]


def test_per_cell_empty_mask_keeps_cells_and_zero_signal(tmp_path):
    from cellonaut.masks.cellpose import _measure_per_cell_roi_signals

    row = {}
    labels = np.array([[1, 1], [2, 2]], dtype=np.int32)
    signal = np.full((2, 2), 10.0)
    _measure_per_cell_roi_signals(
        cfg=NS(
            measurement_options={
                "positive_area_in_cell": True,
                "raw_intden_in_cell": True,
                "mean_in_positive_area": True,
            }
        ),
        roi_defs=[NS(key="mask", label="Mask")],
        roi_map={"mask": None},
        analysis_cell_mask=labels,
        cell_source_img=None,
        measurement_source_img=None,
        measurement_source_arr=signal,
        source_measure_arr=signal,
        source_def=NS(label="GFP"),
        cell_source_label="GFP",
        has_source_background_subtraction=True,
        export_dirs={"cell_signal_tables": tmp_path},
        result_id="sample",
        save_detailed_tables=True,
        row=row,
    )
    cells = pd.read_csv(tmp_path / "sample_GFP_Mask_per_cell_signal.csv")
    cells = cast(pd.DataFrame, cells.loc[cells["CellID"].isin(["1", "2"])].copy())
    assert cells["CellID"].tolist() == ["1", "2"]
    assert cells["MaskArea_InCell"].tolist() == [0, 0]
    assert cells["MaskIntDen_InCell"].tolist() == [0, 0]
    assert cells["MaskCorrectedIntDen_InCell"].tolist() == [0, 0]
    assert bool(cells["MaskMeanGrayValue_InCell"].isna().all())
    assert row["GFP_measured_with_Mask_mask_PerCell_SumIntDen"] == 0.0


def test_missing_source_cannot_become_an_empty_mask(tmp_path):
    image = ImageDef("mask", "Mask", "GFP", tmp_path / "classifier.model")
    cfg = NS(output_dir=tmp_path, mask_source_dir=None, reuse_existing_masks=False, probability_class_index=1)
    with pytest.raises(ValueError, match="Missing image"):
        rois.prepare_rois_for_defs(
            image_map={},
            cfg=cfg,
            out_path=tmp_path / "Masks",
            result_id="sample",
            roi_defs=[image],
            segs={},
            log_func=lambda _msg: None,
        )


@pytest.mark.parametrize("grouped", [False, True])
def test_pipeline_preserves_sample_metadata_from_real_input_folders(tmp_path, monkeypatch, grouped):
    import tifffile

    from cellonaut.pipeline.discovery import STRUCTURE_FLAT_TIFFS
    from cellonaut.pipeline.models import Config, MeasurementTarget
    from cellonaut.pipeline.runner import run_pipeline

    input_dir = tmp_path / "input"
    for group, value in (("Drug_A", 10), ("Drug_B", 100)):
        path = input_dir / group / "sample" / "GFP" / "image.tif" if grouped else input_dir / group / "sample.tif"
        path.parent.mkdir(parents=True)
        tifffile.imwrite(path, np.full((8, 8), value, dtype=np.uint16))
    monkeypatch.setattr(
        "cellonaut.masks.cellpose.run_cell_segmentation_on_image",
        lambda image, *_args, **_kwargs: np.ones(image.shape, dtype=np.int32),
    )
    target = MeasurementTarget("gfp")
    target.do_cell_segmentation = True
    target.cell_segmentation_source = "gfp"
    target.measurement_options = {"cell_mean": True}
    cfg = Config(
        fiji_app_path=tmp_path / "no-fiji",
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        input_structure=STRUCTURE_GROUPED_BY_PROTEIN if grouped else STRUCTURE_FLAT_TIFFS,
        images=[ImageDef("gfp", "GFP", "GFP", None)],
        exclusion_tag="",
        threshold_method="Default",
        probability_class_index=1,
        cell_use_gpu=False,
        measurement_targets=[target],
    )
    result = run_pipeline(cfg)
    assert result["status"] == "completed"
    assert result["run_stats"]["processed"] == 2
    tables = cfg.output_dir / "Results" / "CSV Data"
    measurements = pd.read_csv(tables / "Measurements.csv")
    suffix = "sample" if grouped else "sample.tif"
    assert measurements["Sample"].tolist() == [f"Drug_A/{suffix}", f"Drug_B/{suffix}"]
    summary = pd.read_csv(tables / "Measurement_Summary.csv")
    if grouped:
        assert summary["Group"].tolist() == ["Drug_A", "Drug_B"]
        assert summary["Mean"].tolist() == [10.0, 100.0]
    else:
        assert summary["N"].tolist() == [2]
        assert summary["Mean"].tolist() == [55.0]
