from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_SCOPE_MEASUREMENT,
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    IMAGE_PROCESSING_STEP_ROLLING_BALL,
)
from cellonaut.pipeline import processing_montage
from cellonaut.pipeline.processing_montage import (
    PROCESSING_MONTAGE_METADATA_KEY,
    _artifact_arrays,
    _overlay_labels,
    _overlay_mask,
    _select_probability_layer,
    _tile_display_limits,
    _to_uint8,
    save_montage_png,
    save_cellpose_montage_for_sample,
    save_processing_montages_for_sample,
)


def test_save_montage_png_writes_labeled_preview_only_png(tmp_path):
    out_file = tmp_path / "montage.png"
    saved = save_montage_png(
        [
            ("Raw", np.arange(100, dtype=np.uint16).reshape(10, 10)),
            ("Step 1", np.flipud(np.arange(100, dtype=np.uint16).reshape(10, 10))),
        ],
        out_file,
        max_tile_side=32,
    )

    assert saved == out_file
    assert out_file.exists()
    with Image.open(out_file) as image:
        assert image.format == "PNG"
        assert image.width > 0
        assert image.height > 0
        metadata = json.loads(image.info[PROCESSING_MONTAGE_METADATA_KEY])

    assert [tile["label"] for tile in metadata["tiles"]] == ["Raw", "Step 1"]
    assert all(tile["width"] == 10 and tile["height"] == 10 for tile in metadata["tiles"])


def test_save_montage_png_returns_none_without_tiles(tmp_path):
    out_file = tmp_path / "empty.png"

    assert save_montage_png([], out_file) is None
    assert not out_file.exists()


def test_raw_tiles_use_dtype_range_without_percentile_blowout():
    raw = np.asarray([[0, 32768]], dtype=np.uint16)

    rendered = _to_uint8(raw, limits=_tile_display_limits("Raw", raw, None))

    assert 120 <= int(rendered[0, 1]) <= 135


def test_probability_tiles_use_local_contrast_when_values_are_low():
    probability = np.asarray([[0, 10]], dtype=np.uint16)

    rendered = _to_uint8(
        probability,
        limits=_tile_display_limits("Weka probability map", probability, None),
    )

    assert int(rendered[0, 1]) == 255


def test_unreadable_optional_montage_artifact_is_logged(tmp_path, monkeypatch):
    probability_dir = tmp_path / "probability"
    probability_dir.mkdir()
    artifact = probability_dir / "sample_GFP_prob_1.tif"
    artifact.write_bytes(b"invalid tiff")
    messages = []

    def fail_read(_path):
        raise ValueError("broken artifact")

    monkeypatch.setattr("cellonaut.pipeline.processing_montage.tifffile.imread", fail_read)

    tiles = _artifact_arrays(
        probability_dir=probability_dir,
        threshold_dir=None,
        result_id="sample",
        label="GFP",
        log_func=messages.append,
    )

    assert tiles == []
    assert len(messages) == 1
    assert "Could not read montage probability map" in messages[0]
    assert artifact.name in messages[0]
    assert "ValueError: broken artifact" in messages[0]


def test_rolling_ball_tiles_close_duplicate_when_fiji_fails(monkeypatch):
    closed = []

    class FakeDuplicate:
        def close(self):
            closed.append(True)

    class FakeDuplicator:
        def run(self, _image):
            return FakeDuplicate()

    class FakeIJ:
        @staticmethod
        def run(_image, _command, _options):
            raise RuntimeError("rolling ball failed")

    monkeypatch.setattr(
        processing_montage,
        "get_java_classes",
        lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator},
    )

    with pytest.raises(RuntimeError, match="rolling ball failed"):
        processing_montage._rolling_ball_tiles(object(), [8], sequential=True)

    assert closed == [True]


def test_rolling_ball_tiles_keep_each_fiji_option_with_its_radius(monkeypatch):
    calls = []
    class FakeImage:
        def close(self):
            pass

    class FakeDuplicator:
        def run(self, _image):
            return FakeImage()

    class FakeIJ:
        @staticmethod
        def run(_image, command, options):
            calls.append((command, options))

    monkeypatch.setattr(processing_montage, "get_java_classes", lambda: {"IJ": FakeIJ, "Duplicator": FakeDuplicator})
    monkeypatch.setattr(processing_montage, "imageplus_to_numpy_2d", lambda _image: np.zeros((2, 2)))
    _tiles, _array, temps = processing_montage._rolling_ball_tiles(
        object(), [20, 40], sequential=False, options=[" light", " sliding disable"]
    )

    assert calls == [
        ("Subtract Background...", "rolling=20.0 light"),
        ("Subtract Background...", "rolling=40.0 sliding disable"),
    ]
    assert len(temps) == 2


def test_probability_layer_selection_uses_requested_class_and_preserves_rgb():
    stack = np.stack(
        [
            np.full((3, 5), 10, dtype=np.uint8),
            np.full((3, 5), 20, dtype=np.uint8),
        ]
    )
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)

    assert np.array_equal(_select_probability_layer(stack, 2), stack[1])
    assert np.array_equal(_select_probability_layer(stack, 5), stack[0])
    assert np.array_equal(_select_probability_layer(rgb, 2), rgb)


def test_mask_and_label_overlays_use_distinct_boundary_colors():
    base = np.arange(25, dtype=np.uint8).reshape(5, 5)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:3] = 1
    labels[1:4, 3:4] = 2

    mask_overlay = _overlay_mask(base, mask)
    label_overlay = _overlay_labels(base, labels)

    assert np.any(np.all(mask_overlay == [255, 0, 0], axis=2))
    assert np.any(np.all(label_overlay == [0, 220, 255], axis=2))
    assert not np.array_equal(mask_overlay, label_overlay)


def test_cellpose_montage_contains_flagged_views_and_logs_saved_file(monkeypatch, tmp_path):
    captured = {}

    def save(tiles, out_file):
        captured["tiles"] = tiles
        captured["out_file"] = out_file
        return out_file

    monkeypatch.setattr(processing_montage, "save_montage_png", save)
    monkeypatch.setattr(
        processing_montage,
        "imageplus_to_numpy_2d",
        lambda _image: np.arange(16, dtype=np.uint8).reshape(4, 4),
    )
    messages = []
    labels = np.array(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]],
        dtype=np.int32,
    )

    saved = save_cellpose_montage_for_sample(
        image_map={"cell": object()},
        cell_source_def=SimpleNamespace(key="cell"),
        source_def=SimpleNamespace(label="GFP"),
        extra_overlay_masks={
            "__whole_cell_mask__": labels,
            "__flagged_cell_mask__": labels == 2,
        },
        export_dir=tmp_path,
        result_id="Sample 1",
        log_func=messages.append,
    )

    assert saved == tmp_path / "Sample_1" / "Sample_1_GFP_cellpose_montage.png"
    assert [label for label, _array in captured["tiles"]] == [
        "Cellpose input",
        "Cellpose labels",
        "Cellpose outlines",
        "Final overlay",
        "CSV-excluded cell groups",
        "Flagged overlay",
    ]
    assert "Saved Cellpose montage" in messages[0]


def test_cellpose_montage_skips_missing_labels_or_source_image(tmp_path):
    definition = SimpleNamespace(key="cell", label="GFP")

    assert (
        save_cellpose_montage_for_sample(
            image_map={"cell": object()},
            cell_source_def=definition,
            source_def=definition,
            extra_overlay_masks={},
            export_dir=tmp_path,
            result_id="sample",
            log_func=lambda _message: None,
        )
        is None
    )
    assert (
        save_cellpose_montage_for_sample(
            image_map={},
            cell_source_def=definition,
            source_def=definition,
            extra_overlay_masks={"__whole_cell_mask__": np.ones((2, 2), dtype=np.int32)},
            export_dir=tmp_path,
            result_id="sample",
            log_func=lambda _message: None,
        )
        is None
    )


def test_native_empty_processing_montage_does_not_start_fiji(monkeypatch, tmp_path):
    def fail_if_java_is_loaded():
        raise AssertionError("Cellpose-only montage preparation must not start Fiji")

    monkeypatch.setattr(
        "cellonaut.pipeline.image_processing.get_java_classes",
        fail_if_java_is_loaded,
    )
    image_def = SimpleNamespace(key="gfp", label="GFP", image_processing_steps=[])

    saved = save_processing_montages_for_sample(
        cfg=SimpleNamespace(images=[image_def]),
        image_map={"gfp": np.zeros((8, 8), dtype=np.uint16)},
        roi_map={},
        export_dir=tmp_path,
        result_id="sample",
        log_func=lambda _message: None,
    )

    assert saved == []


def test_processing_montages_run_segmentation_and_measurement_recipes_and_close_images(monkeypatch, tmp_path):
    class TemporaryImage:
        def __init__(self, name):
            self.name = name
            self.closed = 0

        def close(self):
            self.closed += 1

    segmentation_temp = TemporaryImage("segmentation")
    transform_temp = TemporaryImage("transform")
    transformed_base = TemporaryImage("transformed base")
    rolling_temp = TemporaryImage("rolling")
    source_image = object()
    base = np.arange(16, dtype=np.uint16).reshape(4, 4)
    image_def = SimpleNamespace(
        key="gfp",
        label="GFP",
        image_processing_steps=[
            {
                "type": IMAGE_PROCESSING_STEP_ROLLING_BALL,
                "scope": IMAGE_PROCESSING_SCOPE_MEASUREMENT,
                "params": {"bg_radii": "8"},
            }
        ],
    )
    cfg = SimpleNamespace(images=[image_def])
    saved_tiles = []
    messages = []

    def recipe(_image, _definition, scope, **_kwargs):
        if scope == IMAGE_PROCESSING_SCOPE_SEGMENTATION:
            return [("Segmentation step", base + 1)], base + 1, [segmentation_temp]
        return [("Measurement transform", base + 2)], base + 2, [transform_temp]

    def save(tiles, out_file):
        saved_tiles.append((tiles, out_file))
        return out_file

    monkeypatch.setattr(processing_montage, "imageplus_to_numpy_2d", lambda _image: base)
    monkeypatch.setattr(processing_montage, "imagej_processing_recipe_tiles", recipe)
    monkeypatch.setattr(
        processing_montage,
        "apply_imagej_processing_recipe_copy",
        lambda *_args, **_kwargs: (transformed_base, []),
    )
    monkeypatch.setattr(
        processing_montage,
        "_rolling_ball_tiles",
        lambda *_args, **_kwargs: ([("Measurement sweep radius=8", base + 3)], base + 3, [rolling_temp]),
    )
    monkeypatch.setattr(processing_montage, "_mask_array_from_roi", lambda *_args: base > 5)
    monkeypatch.setattr(
        processing_montage,
        "_artifact_arrays",
        lambda **_kwargs: [("Weka probability map", base)],
    )
    monkeypatch.setattr(processing_montage, "save_montage_png", save)

    saved = save_processing_montages_for_sample(
        cfg=cfg,
        image_map={"gfp": source_image},
        roi_map={"gfp": object(), "__whole_cell_mask__": object()},
        export_dir=tmp_path,
        result_id="Sample 1",
        log_func=messages.append,
    )

    assert len(saved) == 2
    assert [label for label, _array in saved_tiles[0][0]] == [
        "Raw",
        "Segmentation step",
        "Weka probability map",
        "Mask",
        "Final overlay",
    ]
    assert [label for label, _array in saved_tiles[1][0]] == [
        "Raw",
        "Measurement transform",
        "Measurement sweep radius=8",
    ]
    assert all(image.closed == 1 for image in [segmentation_temp, transform_temp, transformed_base, rolling_temp])
    assert len(messages) == 2


def test_processing_montage_closes_recipe_images_when_saving_fails(monkeypatch, tmp_path):
    class TemporaryImage:
        closed = 0

        def close(self):
            self.closed += 1

    temporary = TemporaryImage()
    image_def = SimpleNamespace(key="gfp", label="GFP", image_processing_steps=[])
    monkeypatch.setattr(
        processing_montage,
        "imageplus_to_numpy_2d",
        lambda _image: np.ones((2, 2), dtype=np.uint8),
    )
    monkeypatch.setattr(
        processing_montage,
        "imagej_processing_recipe_tiles",
        lambda *_args, **_kwargs: ([], None, [temporary]),
    )
    monkeypatch.setattr(processing_montage, "_mask_array_from_roi", lambda *_args: None)
    monkeypatch.setattr(
        processing_montage,
        "save_montage_png",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        save_processing_montages_for_sample(
            cfg=SimpleNamespace(images=[image_def]),
            image_map={"gfp": object()},
            roi_map={"gfp": object()},
            export_dir=tmp_path,
            result_id="sample",
            log_func=lambda _message: None,
        )

    assert temporary.closed == 1
