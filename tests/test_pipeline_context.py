from __future__ import annotations

from pathlib import Path

import pytest

from cellonaut.pipeline import context as pipeline_context
from cellonaut.pipeline.models import MeasurementTarget, Config, ImageDef
from cellonaut.pipeline.discovery import STRUCTURE_FLAT_TIFFS


class CloseTracker:
    def __init__(self, *, fail: bool = False):
        self.close_calls = 0
        self.fail = fail

    def close(self):
        self.close_calls += 1
        if self.fail:
            raise RuntimeError("close failed")


def make_config(tmp_path: Path) -> Config:
    return Config(
        fiji_app_path=tmp_path / "Fiji",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        input_structure="Flat TIFF files",
        images=[ImageDef("image1", "Signal", "Signal", None)],
        exclusion_tag="_ut_",
        threshold_method="Default",
        probability_class_index=1,
        measurement_targets=[MeasurementTarget(source_image_key="image1")],
    )


def test_close_context_closes_each_unique_image_once(make_sample_context):
    shared = CloseTracker()
    derived = CloseTracker()

    make_sample_context(
        image_map={"source": shared},
        roi_measure_img_map={"source": shared, "derived": derived},
    ).close()

    assert shared.close_calls == 1
    assert derived.close_calls == 1


def test_close_context_continues_when_an_image_close_fails(make_sample_context):
    failing = CloseTracker(fail=True)
    healthy = CloseTracker()

    make_sample_context(image_map={"failing": failing, "healthy": healthy}).close()

    assert failing.close_calls == 1
    assert healthy.close_calls == 1


def test_typed_context_manager_closes_resources_once(tmp_path: Path):
    image = CloseTracker()
    context = pipeline_context.SampleProcessingContext(
        id_label="sample",
        result_id="sample",
        export_dirs={"masks": tmp_path},
        file_map={},
        image_map={"source": image},
        roi_map={},
        roi_measure_img_map={"source": image},
        skeleton_metrics={},
        native_numpy_images=False,
    )

    with context as active:
        assert active.result_id == "sample"
        assert active.image_map["source"] is image

    context.close()
    assert image.close_calls == 1


def test_context_build_closes_open_images_when_roi_preparation_fails(
    monkeypatch,
    tmp_path: Path,
):
    cfg = make_config(tmp_path)
    opened = CloseTracker()
    sample = tmp_path / "sample.tif"

    monkeypatch.setattr(
        pipeline_context,
        "find_channel_files",
        lambda *_args, **_kwargs: {"image1": sample},
    )
    monkeypatch.setattr(
        pipeline_context,
        "resolve_channel_indices_for_file_map",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        pipeline_context,
        "open_channel_arrays",
        lambda *_args, **_kwargs: {"image1": opened},
    )
    monkeypatch.setattr(
        pipeline_context,
        "prepare_rois_for_defs",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("ROI setup failed")),
    )

    with pytest.raises(RuntimeError, match="ROI setup failed"):
        pipeline_context.build_sample_processing_context(
            sample,
            cfg,
            lambda _message: None,
        )

    assert opened.close_calls == 1


def test_context_resolves_stack_layers_from_complete_configured_file_map(monkeypatch, tmp_path: Path):
    cfg = make_config(tmp_path)
    cfg.input_structure = STRUCTURE_FLAT_TIFFS
    cfg.images.append(ImageDef("image2", "Second", "Second", None))
    cfg.measurement_targets = [MeasurementTarget(source_image_key="image2")]
    sample = tmp_path / "sample.tif"
    opened = CloseTracker()
    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        pipeline_context,
        "find_channel_files",
        lambda *_args, **_kwargs: {"image1": sample, "image2": sample},
    )
    monkeypatch.setattr(
        pipeline_context,
        "image_keys_needed_for_processing",
        lambda *_args, **_kwargs: {"image2"},
    )

    def capture_channel_indices(**kwargs):
        captured["resolved"] = dict(kwargs["file_map"])
        return {"image1": 1, "image2": 2}

    def capture_opened_files(file_map, *_args, **_kwargs):
        captured["opened"] = dict(file_map)
        return {"image2": opened}

    monkeypatch.setattr(pipeline_context, "resolve_channel_indices_for_file_map", capture_channel_indices)
    monkeypatch.setattr(pipeline_context, "open_channel_arrays", capture_opened_files)
    monkeypatch.setattr(pipeline_context, "prepare_rois_for_defs", lambda **_kwargs: ({}, {}))
    monkeypatch.setattr(pipeline_context, "save_processing_montages_for_sample", lambda **_kwargs: [])

    context = pipeline_context.build_sample_processing_context(sample, cfg, lambda _message: None)

    try:
        assert context is not None
        assert set(captured["resolved"]) == {"image1", "image2"}
        assert set(captured["opened"]) == {"image2"}
    finally:
        if context is not None:
            context.close()


def test_context_build_saves_processing_montages_after_roi_preparation(
    monkeypatch,
    tmp_path: Path,
):
    cfg = make_config(tmp_path)
    opened = CloseTracker()
    roi_measure_image = CloseTracker()
    sample = tmp_path / "sample.tif"
    roi = object()
    calls = []

    monkeypatch.setattr(
        pipeline_context,
        "find_channel_files",
        lambda *_args, **_kwargs: {"image1": sample},
    )
    monkeypatch.setattr(
        pipeline_context,
        "resolve_channel_indices_for_file_map",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        pipeline_context,
        "open_channel_arrays",
        lambda *_args, **_kwargs: {"image1": opened},
    )

    def fake_prepare_rois_for_defs(**kwargs):
        calls.append(("prepare_rois", kwargs))
        return {"image1": roi}, {"image1": roi_measure_image}

    def fake_save_processing_montages_for_sample(**kwargs):
        calls.append(("save_montages", kwargs))
        return [kwargs["export_dir"] / "sample_montage.png"]

    monkeypatch.setattr(
        pipeline_context,
        "prepare_rois_for_defs",
        fake_prepare_rois_for_defs,
    )
    monkeypatch.setattr(
        pipeline_context,
        "save_processing_montages_for_sample",
        fake_save_processing_montages_for_sample,
    )

    context = pipeline_context.build_sample_processing_context(
        sample,
        cfg,
        lambda _message: None,
    )

    try:
        assert context is not None
        assert [name for name, _kwargs in calls] == [
            "prepare_rois",
            "save_montages",
        ]

        prepare_kwargs = calls[0][1]
        montage_kwargs = calls[1][1]
        assert montage_kwargs["cfg"] is cfg
        assert montage_kwargs["image_map"] is prepare_kwargs["image_map"]
        assert montage_kwargs["roi_map"] == {"image1": roi}
        assert montage_kwargs["result_id"] == context.result_id
        assert montage_kwargs["export_dir"] == context.export_dirs["processing_montages"]
        assert montage_kwargs["probability_dir"] == context.export_dirs["weka_probability_maps"]
        assert montage_kwargs["threshold_dir"] == context.export_dirs["weka_threshold_masks"]
    finally:
        if context is not None:
            context.close()
