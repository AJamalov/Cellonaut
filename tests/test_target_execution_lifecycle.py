"""Exercise real target orchestration using explicit, owned sample resources."""

from copy import deepcopy
from cellonaut.runtime import PipelineRuntime

import numpy as np
import pytest
import tifffile

from cellonaut.exceptions import PipelineCancelled
from cellonaut.pipeline import context as context_module, runner, sample_processing
from cellonaut.pipeline.models import Config, ImageDef, MeasurementTarget


class ImageResource:
    def __init__(self):
        self.pixels = np.arange(4).reshape(2, 2)
        self.reads = 0
        self.closes = 0

    def __array__(self, dtype=None, copy=None):
        assert not self.closes
        self.reads += 1
        return np.asarray(self.pixels, dtype=dtype)

    def close(self):
        self.closes += 1


@pytest.fixture
def sample_resources(tmp_path, make_sample_context):
    images = [ImageDef(key, key.upper(), key, None) for key in ("a", "b")]
    images.append(ImageDef("mask", "Mask", "mask", tmp_path / "classifier.model"))
    cfg = Config(
        fiji_app_path=tmp_path / "Fiji", input_dir=tmp_path, output_dir=tmp_path / "output",
        input_structure="Unsorted TIFF images in input folder", images=images,
        exclusion_tag="", threshold_method="Otsu", probability_class_index=1,
        source_image_key="a", overlay_base_image_key="a", cell_diameter=70,
        measurement_targets=[
            MeasurementTarget(source_image_key=key, overlay_roi_keys=["mask"],
                              measurement_options={"area": True, "mean": True, "raw_intden": True},
                              cell_diameter=diameter)
            for key, diameter in (("a", 30), ("b", None))
        ],
    )
    image = ImageResource()
    tifffile.imwrite(tmp_path / "sample.tif", image.pixels.astype(np.uint16))
    context = make_sample_context(
        image_map={"a": image, "b": image}, roi_map={"mask": None},
        roi_measure_img_map={"mask": image}, skeleton_metrics={"Mask_Branches": 3},
        file_map={"a": tmp_path / "sample.tif", "b": tmp_path / "sample.tif"},
    )
    return cfg, context, image, tmp_path / "sample.tif"


def test_real_targets_reuse_context_preserve_rows_and_report_progress(sample_resources, monkeypatch):
    cfg, context, image, sample = sample_resources
    before = deepcopy(cfg)
    builds, progress, counters = [], [], []

    def build(**_kwargs):
        builds.append(1)
        return context

    monkeypatch.setattr(runner, "build_sample_processing_context", build)
    rows, processed, statuses = runner._process_preview_targets(
        runtime=PipelineRuntime(),
        cfg=cfg, sample_folder=sample, sample_label="sample", enabled_targets=cfg.measurement_targets,
        logger=lambda _message: None, log_label="PREVIEW", run_type="preview",
        progress_func=progress.append, counter_func=counters.append, should_cancel=None, context_error="missing",
    )
    assert builds == [1]
    assert image.reads == 2 and image.closes == 1
    assert processed == ["a", "b"]
    assert [row["Label"] for row in rows] == ["sample__A", "sample__B"]
    for row, label in zip(rows, ("A", "B")):
        assert row[f"{label}_in_Mask_Area"] == 0
        assert row[f"{label}_in_Mask_RawIntDen"] == 0
        assert np.isnan(row[f"{label}_in_Mask_Mean"])
        assert row["Mask_Branches"] == 3
        assert row["SourceImageFile"] == str(sample)
    assert [status["status"] for status in statuses] == ["PROCESSED", "PROCESSED"]
    assert progress == [0, 0, 50, 50, 100]
    assert counters == ["0 / 2", "1 / 2", "2 / 2"]
    assert cfg == before


@pytest.mark.parametrize("outcome", ["normal", "exception", "cancel"])
def test_owned_context_closes_for_every_target_outcome(sample_resources, monkeypatch, outcome):
    cfg, context, image, sample = sample_resources
    monkeypatch.setattr(sample_processing, "build_sample_processing_context", lambda **_kwargs: context)
    cancel = False

    def log(message):
        nonlocal cancel
        if "Core measurement modes" in message:
            if outcome == "exception":
                raise RuntimeError("status consumer failed")
            cancel = outcome == "cancel"

    def run():
        return sample_processing.process_measurement_target_for_sample(
            sample, cfg, cfg.measurement_targets[0], log, should_cancel=lambda: cancel,
        )

    if outcome == "normal":
        row, status = run()
        assert status == "PROCESSED" and row is not None and row["A_in_Mask_Area"] == 0
    else:
        with pytest.raises(PipelineCancelled if outcome == "cancel" else RuntimeError):
            run()
    assert image.closes == 1


def test_borrowed_context_survives_target_failure(sample_resources):
    cfg, context, image, sample = sample_resources
    cfg.measurement_targets[0].source_image_key = "missing"
    with context:
        with pytest.raises(ValueError, match="Unknown image key"):
            sample_processing.process_measurement_target_for_sample(
                sample, cfg, cfg.measurement_targets[0], lambda _message: None, precomputed_context=context,
            )
        assert image.closes == 0
        row, status = sample_processing.process_measurement_target_for_sample(
            sample, cfg, cfg.measurement_targets[1], lambda _message: None, precomputed_context=context,
        )
        assert row is not None and row["B_in_Mask_Area"] == 0 and status == "PROCESSED"
    assert image.closes == 1


def test_imagej_lookup_failure_closes_owned_context(sample_resources, monkeypatch):
    cfg, context, image, sample = sample_resources
    context.native_numpy_images = False
    monkeypatch.setattr(sample_processing, "build_sample_processing_context", lambda **_kwargs: context)

    def unavailable():
        raise RuntimeError("ImageJ unavailable")

    monkeypatch.setattr(sample_processing, "get_java_classes", unavailable)
    with pytest.raises(RuntimeError, match="ImageJ unavailable"):
        sample_processing.process_sample(sample, cfg, lambda _message: None)
    assert image.closes == 1


@pytest.mark.parametrize("outcome", ["normal", "exception", "cancel"])
def test_temporary_image_cleanup_keeps_borrowed_source_open(sample_resources, monkeypatch, outcome):
    from types import SimpleNamespace

    cfg, context, source, sample = sample_resources
    cfg.per_cell_mask_source = "mask"
    context.native_numpy_images = False
    duplicate = ImageResource()
    monkeypatch.setattr(sample_processing, "get_java_classes", lambda: {
        "Duplicator": lambda: SimpleNamespace(run=lambda _image: duplicate),
    })
    cancel = False

    def log(message):
        nonlocal cancel
        if "Core measurement modes" in message:
            if outcome == "exception":
                raise RuntimeError("status consumer failed")
            cancel = outcome == "cancel"

    def run():
        return sample_processing.process_sample(
            sample, cfg, log, should_cancel=lambda: cancel, precomputed_context=context,
        )

    # An empty prepared ROI requires no backend statistics, but still visits
    # the cancellation checkpoint after the measurement image is duplicated.
    if outcome == "normal":
        assert run()[1] == "PROCESSED"
    else:
        with pytest.raises(PipelineCancelled if outcome == "cancel" else RuntimeError):
            run()
    assert duplicate.closes == 1 and source.closes == 0
    context.close()
    assert source.closes == 1


@pytest.mark.parametrize("control", ["progress", "counter", "cancel"])
def test_preview_control_failure_closes_context(sample_resources, monkeypatch, control):
    cfg, context, image, sample = sample_resources
    monkeypatch.setattr(runner, "build_sample_processing_context", lambda **_kwargs: context)

    def fail(_value):
        raise RuntimeError("consumer failed")

    with pytest.raises(PipelineCancelled if control == "cancel" else RuntimeError):
        runner._process_preview_targets(
            runtime=PipelineRuntime(),
            cfg=cfg, sample_folder=sample, sample_label="sample", enabled_targets=cfg.measurement_targets,
            logger=lambda _message: None, log_label="PREVIEW", run_type="preview",
            progress_func=fail if control == "progress" else None,
            counter_func=fail if control == "counter" else None,
            should_cancel=lambda: control == "cancel", context_error="missing",
        )
    assert image.closes == 1


def test_cancellation_after_opening_closes_partial_context(sample_resources, monkeypatch):
    cfg, _context, image, sample = sample_resources
    cfg.images = cfg.images[:2]
    cfg.measurement_targets = [MeasurementTarget(source_image_key="a")]
    cancelled = False

    def opened(*_args, **_kwargs):
        nonlocal cancelled
        cancelled = True
        return {"a": image, "b": image}

    monkeypatch.setattr(context_module, "find_channel_files", lambda *_args, **_kwargs: {"a": sample})
    monkeypatch.setattr(context_module, "open_channel_arrays", opened)
    with pytest.raises(PipelineCancelled):
        context_module.build_sample_processing_context(sample, cfg, lambda _message: None, lambda: cancelled)
    assert image.closes == 1
