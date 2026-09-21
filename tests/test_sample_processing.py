from __future__ import annotations

from pathlib import Path

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_SCOPE_MEASUREMENT,
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
)
from cellonaut.pipeline import sample_processing
from cellonaut.pipeline.models import Config, ImageDef


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
        source_image_key="image1",
        overlay_base_image_key="image1",
    )


def test_process_sample_closes_owned_context_when_source_image_is_missing(
    make_sample_context,
    monkeypatch,
    tmp_path: Path,
):
    cfg = make_config(tmp_path)
    context = make_sample_context(**{
        "export_dirs": {},
        "image_map": {},
        "roi_map": {},
        "file_map": {},
        "native_numpy_images": True,
    })
    monkeypatch.setattr(
        sample_processing,
        "build_sample_processing_context",
        lambda **_kwargs: context,
    )

    row, status = sample_processing.process_sample(
        tmp_path / "sample.tif",
        cfg,
        lambda _message: None,
    )

    assert (row, status) == (None, "FAILED")
    assert context._closed


def test_measurement_background_radii_respect_processing_scope():
    image_def = ImageDef(
        "image1",
        "Signal",
        "Signal",
        None,
        bg_radii_csv="50",
        image_processing_steps=[
            {
                "type": "rolling_ball_background",
                "enabled": True,
                "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                "params": {"bg_radii": "50"},
            }
        ],
    )

    assert sample_processing.measurement_background_radii(image_def) == []

    image_def.image_processing_steps[0]["scope"] = IMAGE_PROCESSING_SCOPE_MEASUREMENT

    assert sample_processing.measurement_background_radii(image_def) == [50.0]


def test_measurement_background_requests_keep_each_radius_options():
    image_def = ImageDef(
        "image1",
        "Signal",
        "Signal",
        None,
        image_processing_steps=[
            {"type": "rolling_ball_background", "enabled": True, "scope": "Measurement image",
             "params": {"bg_radii": "20,40", "light_background": True, "sliding_paraboloid": True}},
            {"type": "rolling_ball_background", "enabled": True, "scope": "Measurement image",
             "params": {"bg_radii": "80", "disable_smoothing": True}},
        ],
    )

    assert sample_processing.measurement_background_requests(image_def) == [
        (20.0, " light sliding"), (40.0, " light sliding"), (80.0, " disable"),
    ]


def test_process_sample_selects_measurement_recipe_and_sweeps_processed_source(
    make_sample_context, imagej_measurement_backend, tmp_path: Path,
):
    backend = imagej_measurement_backend
    cfg = make_config(tmp_path)
    cfg.images[0].image_processing_steps = [
        {"type": "smooth", "enabled": True, "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
         "params": {"smooth_iterations": "1"}},
        {"type": "enhance_contrast", "enabled": True, "scope": IMAGE_PROCESSING_SCOPE_MEASUREMENT,
         "params": {"contrast_saturation": "15"}},
        {"type": "rolling_ball_background", "enabled": True, "scope": IMAGE_PROCESSING_SCOPE_MEASUREMENT,
         "params": {"bg_radii": "40,70"}},
    ]
    other = ImageDef("other", "Other", "Other", None, image_processing_steps=[
        {"type": "enhance_contrast", "enabled": True, "scope": IMAGE_PROCESSING_SCOPE_MEASUREMENT,
         "params": {"contrast_saturation": "90"}},
    ])
    mask = ImageDef("mask", "Mask", "Mask", tmp_path / "mask.model")
    # Put the unrelated definition first, and use it as the overlay base, so
    # index-based or overlay-based recipe/source selection cannot pass.
    cfg.images = [other, cfg.images[0], mask]
    cfg.overlay_base_image_key = "other"
    cfg.per_cell_mask_source = "mask"
    cfg.overlay_roi_keys = []
    cfg.do_cell_segmentation = False
    source = backend.Image([[1, 2], [3, 4]])
    unrelated = backend.Image([[100, 200], [300, 400]])
    context = make_sample_context(
        image_map={"image1": source, "other": unrelated}, roi_map={"mask": backend.roi},
        file_map={"image1": tmp_path / "sample.tif"}, native_numpy_images=False,
    )

    row, status = sample_processing.process_sample(
        tmp_path / "sample.tif", cfg, lambda _message: None, precomputed_context=context,
    )

    assert status == "PROCESSED" and row is not None
    assert row["SourceImageKey"] == "image1"
    assert row["Signal_in_Mask_Area"] == 4
    assert row["Signal_in_Mask_RawIntDen"] == 10
    assert row["Signal_in_Mask_RB40p0_IntDen"] == 10
    assert row["Signal_in_Mask_RB70p0_IntDen"] == 10
    # Command history, not fabricated transformed pixel values, is the oracle
    # for selection/order. Actual Fiji transform numerics require backend tests.
    contrast = ("Enhance Contrast", "saturated=15.00")
    primary = ("Subtract Background...", "rolling=40.0")
    secondary = ("Subtract Background...", "rolling=70.0")
    assert [(command, options) for _image, command, options in backend.commands] == [
        contrast, primary, secondary,
    ]
    assert [history for _image, history in backend.measured] == [
        (contrast,), (contrast, primary), (contrast, secondary),
    ]
    processed = backend.copies[0][1]
    assert [parent for parent, _copy in backend.copies] == [source, processed, processed]
    assert all(image.closes == 1 for _parent, image in backend.copies)
    assert source.closes == unrelated.closes == 0
    assert source.history == unrelated.history == []
