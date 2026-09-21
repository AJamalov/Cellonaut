from __future__ import annotations

from cellonaut.runtime import PipelineRuntime

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cellonaut.measurement import exports as measurement_exports
from cellonaut.pipeline import runner


def test_reset_log_file_propagates_locked_file_error(monkeypatch, tmp_path: Path):
    log_path = tmp_path / "run_log.txt"

    def fail_unlink(_path, *, missing_ok: bool = False):
        del missing_ok
        raise PermissionError("log is locked")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(PermissionError, match="log is locked"):
        runner._reset_log_file(log_path)


def test_run_pipeline_reports_mixed_sample_results(make_sample_context, monkeypatch, tmp_path: Path):
    samples = [tmp_path / "sample_a", tmp_path / "sample_b"]
    for sample in samples:
        sample.mkdir()

    cfg = SimpleNamespace(
        output_dir=tmp_path / "output",
        input_dir=tmp_path,
        fiji_app_path=tmp_path / "Fiji",
        threshold_method="Default",
        probability_class_index=1,
        reuse_existing_masks=False,
        images=[],
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=[],
        do_cell_segmentation=False,
        cell_segmentation_source="image1",
        cell_diameter=70,
        cell_min_size=0,
        cell_use_gpu=False,
        cellprob_threshold=0,
        flow_threshold=0.4,
        cell_remove_border=True,
        cellpose_model_type="cpsam_v2",
        cellpose_custom_model_path=None,
        per_cell_mask_source="",
        input_structure="Flat TIFF files",
        exclusion_tag="",
        measurement_options={},
    )
    target = SimpleNamespace(
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=[],
        do_cell_segmentation=False,
        cell_segmentation_source="image1",
        overlay_whole_cell_mask=False,
    )
    contexts = {sample: make_sample_context(file_map={"image1": sample / "image.tif"}) for sample in samples}
    process_results = iter(
        [
            ({"Label": "sample_a"}, "PROCESSED"),
            (None, "FAILED"),
        ]
    )

    monkeypatch.setattr(runner, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(runner, "get_enabled_measurement_targets", lambda _cfg: [target])
    monkeypatch.setattr(runner, "pipeline_needs_imagej", lambda *_args: False)
    monkeypatch.setattr(runner, "get_measurement_sample_paths", lambda _cfg: samples)
    monkeypatch.setattr(
        runner,
        "build_sample_processing_context",
        lambda sample_folder, **_kwargs: contexts[sample_folder],
    )
    monkeypatch.setattr(
        runner,
        "process_measurement_target_for_sample",
        lambda **_kwargs: next(process_results),
    )
    monkeypatch.setattr(runner, "build_pipeline_summary_dict", lambda _cfg: {})
    monkeypatch.setattr(runner, "build_sample_label", lambda path, _structure: path.name)
    monkeypatch.setattr(runner, "write_measurement_summary_csvs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "save_run_manifest",
        lambda *_args, **_kwargs: {"text": "summary.txt", "json": "summary.json"},
    )

    result = runner.run_pipeline(cfg, log_func=lambda _message: None)

    assert result["status"] == "completed_with_errors"
    assert result["run_stats"]["processed"] == 1
    assert result["run_stats"]["failed"] == 1
    assert result["run_stats"]["total_targets"] == 2
    assert result["run_stats"]["processed_targets"] == 1
    assert result["run_stats"]["failed_targets"] == 1
    assert [row["status"] for row in result["sample_status"]] == [
        "PROCESSED",
        "FAILED",
    ]


def test_run_pipeline_continues_with_later_target_after_target_exception(make_sample_context, monkeypatch, tmp_path: Path):
    sample = tmp_path / "sample_a"
    sample.mkdir()
    cfg = SimpleNamespace(
        output_dir=tmp_path / "output",
        input_dir=tmp_path,
        fiji_app_path=tmp_path / "Fiji",
        threshold_method="Default",
        probability_class_index=1,
        reuse_existing_masks=False,
        images=[],
        cell_diameter=None,
        cell_min_size=15,
        cell_use_gpu=False,
        cellprob_threshold=0,
        flow_threshold=0.4,
        cell_remove_border=True,
        cellpose_model_type="cpsam_v2",
        cellpose_custom_model_path="",
        per_cell_mask_source="",
        input_structure="Flat TIFF files",
        exclusion_tag="",
        measurement_options={},
    )
    targets = [
        SimpleNamespace(
            source_image_key=key,
            overlay_base_image_key=key,
            overlay_roi_keys=[],
            do_cell_segmentation=False,
            cell_segmentation_source=key,
            overlay_whole_cell_mask=False,
        )
        for key in ("image1", "image2")
    ]
    context = make_sample_context(**{"file_map": {"image1": sample / "one.tif", "image2": sample / "two.tif"}})
    attempted = []

    def process_target(**kwargs):
        key = kwargs["target"].source_image_key
        attempted.append(key)
        if key == "image1":
            raise RuntimeError("target-specific failure")
        return {"Label": "sample_a", "SourceImageKey": key}, "PROCESSED"

    monkeypatch.setattr(runner, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(runner, "get_enabled_measurement_targets", lambda _cfg: targets)
    monkeypatch.setattr(runner, "pipeline_needs_imagej", lambda *_args: False)
    monkeypatch.setattr(runner, "get_measurement_sample_paths", lambda _cfg: [sample])
    monkeypatch.setattr(runner, "build_sample_processing_context", lambda **_kwargs: context)
    monkeypatch.setattr(runner, "process_measurement_target_for_sample", process_target)
    monkeypatch.setattr(runner, "build_pipeline_summary_dict", lambda _cfg: {})
    monkeypatch.setattr(runner, "build_sample_label", lambda path, _structure: path.name)
    monkeypatch.setattr(runner, "write_measurement_summary_csvs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "save_run_manifest",
        lambda *_args, **_kwargs: {"text": "summary.txt", "json": "summary.json"},
    )

    result = runner.run_pipeline(cfg, log_func=lambda _message: None)

    assert attempted == ["image1", "image2"]
    assert [row["status"] for row in result["sample_status"]] == ["FAILED", "PROCESSED"]
    assert result["run_stats"]["processed_targets"] == 1
    assert result["run_stats"]["failed_targets"] == 1
    assert result["status"] == "completed_with_errors"


def test_preview_targets_continue_after_one_target_raises(make_sample_context, monkeypatch, tmp_path: Path):
    sample = tmp_path / "sample_a"
    sample.mkdir()
    targets = [SimpleNamespace(source_image_key=key) for key in ("image1", "image2")]
    context = make_sample_context(**{"file_map": {"image1": sample / "one.tif", "image2": sample / "two.tif"}})
    attempted: list[str] = []

    def process_target(**kwargs):
        key = kwargs["target"].source_image_key
        attempted.append(key)
        if key == "image1":
            raise RuntimeError("bad first target")
        return {"SourceImageKey": key}, "PROCESSED"

    monkeypatch.setattr(runner, "build_sample_processing_context", lambda **_kwargs: context)
    monkeypatch.setattr(runner, "process_measurement_target_for_sample", process_target)

    rows, processed, statuses = runner._process_preview_targets(
        runtime=PipelineRuntime(),
        cfg=SimpleNamespace(),
        sample_folder=sample,
        sample_label="sample_a",
        enabled_targets=targets,
        logger=lambda _message: None,
        log_label="PREVIEW",
        run_type="preview",
        progress_func=None,
        counter_func=None,
        should_cancel=None,
        context_error="missing context",
    )

    assert attempted == ["image1", "image2"]
    assert processed == ["image2"]
    assert rows == [{"SourceImageKey": "image2"}]
    assert [row["status"] for row in statuses] == ["FAILED", "PROCESSED"]
    assert "bad first target" in statuses[0]["reason"]
    assert context._closed


def test_write_measurement_summary_failure_propagates(monkeypatch, tmp_path: Path):
    cfg = SimpleNamespace(
        output_dir=tmp_path,
        measurement_options={},
        input_structure="Flat TIFF files",
        exclusion_tag="",
    )
    monkeypatch.setattr(
        measurement_exports,
        "write_dataframe_csv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    rows = pd.DataFrame([{"Label": "sample_a", "Value": 1}])

    try:
        runner.write_measurement_summary_csvs(
            rows,
            cfg=cfg,
            log_func=lambda _message: None,
        )
    except OSError as exc:
        assert "disk full" in str(exc)
    else:
        raise AssertionError("Expected output failure to propagate")


def test_measurement_summary_always_writes_required_result_tables(tmp_path: Path):
    cfg = SimpleNamespace(
        output_dir=tmp_path,
        input_structure="Unsorted TIFF images in input folder",
        exclusion_tag="",
    )
    rows = pd.DataFrame(
        [
            {"Label": "sample_a__GFP", "SourceImageLabel": "GFP", "GFP_Mean": 1.0},
            {"Label": "sample_a__DAPI", "SourceImageLabel": "DAPI", "DAPI_Mean": 2.0},
        ]
    )

    written = runner.write_measurement_summary_csvs(rows, cfg=cfg, log_func=lambda _message: None)

    assert set(written) == {
        "channel:GFP",
        "channel:DAPI",
        "all",
        "short",
        "all_long",
        "simple",
        "simple_long",
        "summary",
    }
    assert (tmp_path / "Results" / "CSV Data" / "Per-Channel Tables" / "GFP_Measurements.csv").is_file()
    assert (tmp_path / "Results" / "CSV Data" / "Measurements.csv").is_file()


def test_combined_measurement_exports_are_clear_and_analysis_ready(tmp_path: Path):
    cfg = SimpleNamespace(
        output_dir=tmp_path,
        input_structure="Unsorted TIFF images in input folder",
        exclusion_tag="",
    )
    rows = pd.DataFrame(
        [
            {
                "Label": "sample_a__GFP",
                "SampleID": "sample_a",
                "OriginalSamplePath": "C:/data/sample_a.tif",
                "SourceImageLabel": "GFP",
                "GFP_in_Nucleus_Area": 10.0,
                "GFP_in_Nucleus_MeanGrayValue": 2.0,
            },
            {
                "Label": "sample_b__GFP",
                "SampleID": "sample_b",
                "OriginalSamplePath": "C:/data/sample_b.tif",
                "SourceImageLabel": "GFP",
                "GFP_in_Nucleus_Area": 20.0,
                "GFP_in_Nucleus_MeanGrayValue": 6.0,
            },
        ]
    )

    written = runner.write_measurement_summary_csvs(rows, cfg=cfg, log_func=lambda _message: None)
    csv_dir = tmp_path / "Results" / "CSV Data"

    assert set(written) == {
        "channel:GFP",
        "all",
        "short",
        "all_long",
        "simple",
        "simple_long",
        "summary",
    }
    assert (csv_dir / "Measurements.csv").is_file()
    assert (csv_dir / "Measurements_By_Metric.csv").is_file()
    assert (csv_dir / "Measurement_Summary.csv").is_file()
    assert not (csv_dir / "Measurements_Detailed.csv").exists()
    assert not (csv_dir / "Measurements_Tidy.csv").exists()
    assert not (csv_dir / "All_Measurements_Row_Guide.csv").exists()

    measurements = pd.read_csv(csv_dir / "Measurements.csv")
    assert measurements["Sample"].tolist() == ["sample_a.tif", "sample_b.tif"]
    assert "Average" not in measurements["Sample"].tolist()
    assert measurements.columns.tolist() == [
        "Sample",
        "GFP(Nucleus) : Area (px²)",
        "GFP(Nucleus) : Mean intensity (a.u.)",
    ]
    summary = pd.read_csv(csv_dir / "Measurement_Summary.csv")
    assert summary["N"].tolist() == [2, 2]
    assert summary["Measurement"].tolist() == [
        "GFP(Nucleus) : Area (px²)",
        "GFP(Nucleus) : Mean intensity (a.u.)",
    ]


def test_per_channel_tables_keep_unicode_labels_on_distinct_paths(tmp_path: Path):
    cfg = SimpleNamespace(
        output_dir=tmp_path,
        input_structure="Unsorted TIFF images in input folder",
        exclusion_tag="",
    )
    rows = pd.DataFrame(
        [
            {"Label": "sample_a__alpha", "SourceImageLabel": "α", "Value": 1.0},
            {"Label": "sample_a__beta", "SourceImageLabel": "β", "Value": 2.0},
        ]
    )

    written = runner.write_measurement_summary_csvs(rows, cfg=cfg, log_func=lambda _message: None)
    channel_dir = tmp_path / "Results" / "CSV Data" / "Per-Channel Tables"
    exported = sorted(channel_dir.glob("*_Measurements.csv"))

    assert set(written) == {
        "channel:α",
        "channel:β",
        "all",
        "short",
        "all_long",
        "simple",
        "simple_long",
        "summary",
    }
    assert len(exported) == 2
    assert exported[0].name != exported[1].name
