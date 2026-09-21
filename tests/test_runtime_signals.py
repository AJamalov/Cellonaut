"""Runtime facts and control updates are independent of presentation wording."""

import json
import pickle
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from cellonaut import workers
from cellonaut.cell_segmentation import core
from cellonaut.cell_segmentation.core import CellSegmentationConfig
from cellonaut.exceptions import SetupError, SetupErrorCode, SetupFileNotFoundError
from cellonaut.io import image_io
from cellonaut.io.imagej_runtime import format_weka_runtime_error
from cellonaut.pipeline import runner
from cellonaut.pipeline import readiness
from cellonaut.pipeline.models import Config, ImageDef, MeasurementTarget
from cellonaut.pipeline.output_manifest import save_run_manifest
from cellonaut.results.layout import build_results_layout
from cellonaut.runtime import PipelineRuntime, PipelineStage, stage_update


@pytest.mark.parametrize("preview", [False, True])
def test_native_run_emits_stages_and_sample_without_reading_logs(tmp_path, monkeypatch, preview):
    source = tmp_path / "input"
    source.mkdir()
    tifffile.imwrite(source / "sample.tif", np.ones((8, 8), dtype=np.uint16))
    cfg = Config(
        input_dir=source, output_dir=tmp_path / "output", fiji_app_path=tmp_path / "unused",
        input_structure="Unsorted TIFF images in input folder", exclusion_tag="",
        threshold_method="Otsu", probability_class_index=1,
        images=[ImageDef("a", "Signal", "a", None)],
        measurement_targets=[MeasurementTarget(source_image_key="a", overlay_roi_keys=[],
                                               do_cell_segmentation=True, cell_segmentation_source="a",
                                               cell_min_size=0, cell_remove_border=False)],
    )
    class Model:
        device = SimpleNamespace(type="cpu")

        def eval(self, image, **_kwargs):
            labels = np.zeros(image.shape, dtype=np.int32)
            labels[2:5, 2:5] = 1
            return labels, None, None

    monkeypatch.setattr(core, "get_cellpose_model", lambda _cfg: Model())
    monkeypatch.setattr(core, "describe_cellpose_backend", lambda **_kwargs: SimpleNamespace(label="Backend", detail="CPU"))
    stages, samples, logs = [], [], []
    # Discard all original wording at the log boundary, including Preview prefixes.
    original_make_logger = runner.make_logger
    monkeypatch.setattr(runner, "make_logger", lambda sink, path: original_make_logger(
        lambda _message: sink("changed human-readable log"), path,
    ))
    run = runner.run_preview_pipeline if preview else runner.run_pipeline
    result = run(cfg, stage_func=stages.append, sample_func=samples.append, log_func=logs.append)
    assert result["status"] == ("PROCESSED" if preview else "completed")
    assert len(samples) == 1 and "sample" in samples[0]
    assert logs and set(logs) == {"changed human-readable log"}
    ids = [event["stage"] for event in stages]
    assert PipelineStage.LOADING in ids
    assert PipelineStage.MASKS in ids
    assert PipelineStage.MEASUREMENT in ids
    assert PipelineStage.SEGMENTATION in ids
    manifest = json.loads(build_results_layout(cfg.output_dir)["run_manifest_json"].read_text())
    assert manifest["cellpose_inference_devices"] == ["CPU"]
    log_name = "preview_log.txt" if preview else "run_log.txt"
    persisted_log = (build_results_layout(cfg.output_dir)["logs"] / log_name).read_text(encoding="utf-8")
    assert "Cellpose inference device: CPU" in persisted_log
    assert "Measurement config:" in persisted_log
    if not preview:
        assert ids[-1] == PipelineStage.EXPORT


@pytest.mark.parametrize("fails", [False, True])
def test_device_fact_is_recorded_only_after_success_even_without_logging(monkeypatch, tmp_path, fails):
    class Model:
        device = SimpleNamespace(type="cuda")

        def eval(self, image, **_kwargs):
            if fails:
                raise RuntimeError("inference failed")
            return np.zeros(image.shape, dtype=np.int32), None, None

    monkeypatch.setattr(core, "get_cellpose_model", lambda _cfg: Model())
    runtime = PipelineRuntime()
    if fails:
        with pytest.raises(RuntimeError, match="inference failed"):
            core.run_cellpose_segmentation(np.ones((3, 3)), CellSegmentationConfig(), runtime=runtime)
    else:
        core.run_cellpose_segmentation(np.ones((3, 3)), CellSegmentationConfig(), runtime=runtime)
    paths = save_run_manifest(tmp_path, run_type="preview", pipeline_summary={"runtime_environment": {"python": {"version": "test"}}},
                              inference_devices=runtime.cellpose_inference_devices)
    manifest = json.loads(Path(paths["json"]).read_text())
    assert manifest["cellpose_inference_devices"] == ([] if fails else ["CUDA"])
    assert PipelineRuntime().cellpose_inference_devices == set()


def test_worker_queue_preserves_stage_identity_and_logs_independently():
    messages = queue.Queue()
    update = stage_update(PipelineStage.MASKS, "A newly worded message")
    workers._queue_emit(messages, "status", update)
    workers._queue_emit(messages, "log", "[PREVIEW] Trying sample: not a sample event")
    cfg = Config(fiji_app_path=Path(), input_dir=Path(), output_dir=Path(),
                 input_structure="Unsorted TIFF images in input folder", images=[],
                 exclusion_tag="", threshold_method="Otsu", probability_class_index=1)
    worker = workers.PreviewWorker(cfg, use_subprocess=False)
    stages, logs, samples = [], [], []
    worker.status_signal.connect(stages.append)
    worker.log_signal.connect(logs.append)
    worker.sample_signal.connect(samples.append)
    while not messages.empty():
        worker._emit_child_message(*pickle.loads(pickle.dumps(messages.get())))
    assert stages == [update]
    assert logs == ["[PREVIEW] Trying sample: not a sample event"]
    assert samples == []
    worker._emit_child_message("sample", "explicit sample")
    assert samples == ["explicit sample"]


def test_setup_guidance_uses_code_and_preserves_exception_contract():
    exc = SetupFileNotFoundError(SetupErrorCode.CLASSIFIER_MISSING, "Entirely new wording")
    restored = pickle.loads(pickle.dumps(exc))
    assert isinstance(restored, FileNotFoundError)
    assert "reselect the missing classifier" in workers.format_user_error(restored)
    assert "Entirely new wording" in workers.format_user_error(restored)
    unrelated = ValueError("Classifier file does not exist")
    assert "reselect the missing classifier" not in workers.format_user_error(unrelated)
    assert "No processable samples" in workers.format_user_error(SetupError(SetupErrorCode.NO_SAMPLES, "Changed"))
    assert "No processable samples" in workers.format_user_error(SetupError(SetupErrorCode.NO_SAMPLES, ""))


def test_third_party_java_fallback_remains_separate():
    result = format_weka_runtime_error(RuntimeError("java.lang.OutOfMemoryError: Java heap space"), mask_label="Mask")
    assert "increase CELLONAUT_JAVA_HEAP" in str(result)
    unknown = format_weka_runtime_error(RuntimeError("unrecognized library failure"), mask_label="Mask")
    assert "unrecognized library failure" in str(unknown)


def test_network_copy_signals_stage_without_interpreting_logs(tmp_path, monkeypatch):
    source = tmp_path / "sample.tif"
    source.write_bytes(b"image data")
    monkeypatch.setattr(image_io, "staged_image_process_directory", lambda: tmp_path / "staging")
    stages, logs = [], []
    copied = image_io.stage_image_to_local_temp(source, logs.append, runtime=PipelineRuntime(stages.append))
    assert copied.read_bytes() == source.read_bytes()
    assert [event["stage"] for event in stages] == [PipelineStage.STAGING]
    assert any("Staging network image locally" in message for message in logs)


def test_readiness_uses_explicit_ambiguity_not_warning_text(tmp_path, monkeypatch):
    sample = tmp_path / "sample.tif"
    sample.touch()
    cfg = Config(fiji_app_path=tmp_path, input_dir=tmp_path, output_dir=tmp_path / "output",
                 input_structure="Unsorted TIFF images in input folder", exclusion_tag="",
                 threshold_method="Otsu", probability_class_index=1,
                 images=[ImageDef("a", "A", "a", None)],
                 measurement_targets=[MeasurementTarget(source_image_key="a")])

    def discover(_folder, _cfg, _label, log_func, *, required_image_keys, ambiguity_func):
        log_func("Unrelated ordinary message")
        ambiguity_func("Two equally suitable source images")
        return {"a": sample}

    monkeypatch.setattr(readiness, "find_channel_files", discover)
    report = readiness.analyze_sample_readiness(cfg)
    assert report["samples"][0]["status"] == "READY_WITH_WARNINGS"
    assert report["samples"][0]["ambiguous"] == ["Two equally suitable source images"]
