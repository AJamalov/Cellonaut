from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import cellonaut.masks.overlay_exports as overlay_exports
import cellonaut.masks.roi_rasterization as roi_rasterization
from cellonaut.masks.overlay_exports import save_general_overlay_stack


def test_overlay_stack_skips_absent_base_image_without_creating_output(tmp_path):
    messages = []

    save_general_overlay_stack(
        base_img=None,
        roi_map={},
        roi_keys=["mask"],
        out_path=tmp_path / "overlays",
        result_id="Sample_001",
        base_label="Signal",
        cfg=SimpleNamespace(images=[]),
        log_func=messages.append,
    )

    assert messages == ["[Sample_001] Overlay stack skipped: base image is None"]
    assert not (tmp_path / "overlays").exists()


def test_overlay_stack_rejects_mask_with_different_dimensions(tmp_path):
    image = np.ones((4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"Mask shape \(3, 5\) does not match base image shape \(4, 5\)"):
        save_general_overlay_stack(
            base_img=image,
            roi_map={},
            roi_keys=["__whole_cell_mask__"],
            out_path=tmp_path / "overlays",
            result_id="Sample_001",
            base_label="Signal",
            cfg=SimpleNamespace(images=[]),
            log_func=lambda _message: None,
            extra_mask_map={"__whole_cell_mask__": np.ones((3, 5), dtype=np.uint8)},
        )


def test_create_mask_from_unmasked_area_roi_fills_interior(monkeypatch):
    class FakeByteProcessor:
        def __init__(self, width, height):
            self.pixels = np.zeros((height, width), dtype=np.uint8)

        def set(self, x, y, value):
            self.pixels[y, x] = value

    class FakeImagePlus:
        def __init__(self, title, processor):
            self.title = title
            self.processor = processor

    class FakeRoi:
        def getMask(self):
            return None

        def getBounds(self):
            return SimpleNamespace(x=2, y=1, width=3, height=2)

        def contains(self, x, y):
            return 2 <= x < 5 and 1 <= y < 3

    ref_img = SimpleNamespace(getWidth=lambda: 7, getHeight=lambda: 5)
    monkeypatch.setattr(
        roi_rasterization,
        "get_java_classes",
        lambda: {"ImagePlus": FakeImagePlus, "ByteProcessor": FakeByteProcessor},
    )

    result = roi_rasterization.imagej_roi_to_mask_image(ref_img, FakeRoi(), "mask")

    assert np.count_nonzero(result.processor.pixels) == 6
    assert np.all(result.processor.pixels[1:3, 2:5] == 255)


def test_overlay_stack_keeps_distinct_source_channels_in_config_order(tmp_path):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    image_map = {
        "image1": np.full((4, 5), 11, dtype=np.uint16),
        "image2": np.full((4, 5), 22, dtype=np.uint16),
        "image3": np.full((4, 5), 33, dtype=np.uint16),
    }
    cfg = SimpleNamespace(
        images=[
            SimpleNamespace(key="image1", label="Channel 1"),
            SimpleNamespace(key="image2", label="Channel 2"),
            SimpleNamespace(key="image3", label="Channel 3"),
        ]
    )

    save_general_overlay_stack(
        base_img=image_map["image1"],
        roi_map={},
        roi_keys=["__whole_cell_mask__"],
        out_path=tmp_path,
        result_id="Sample_001",
        base_label="Channel 1",
        cfg=cfg,
        log_func=lambda _msg: None,
        extra_mask_map={"__whole_cell_mask__": np.ones((4, 5), dtype=np.uint8)},
        image_map=image_map,
    )

    stack = tifffile.imread(tmp_path / "Sample_001_Channel_1_combined_overlay.tif")

    assert stack.shape[0] == 4
    assert np.all(stack[0] == 11)
    assert np.all(stack[1] == 22)
    assert np.all(stack[2] == 33)


def test_overlay_stack_preserves_16bit_channels_when_base_is_8bit(tmp_path):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    image_map = {
        "image1": np.full((4, 5), 7, dtype=np.uint8),
        "image2": np.full((4, 5), 4000, dtype=np.uint16),
    }
    cfg = SimpleNamespace(
        images=[
            SimpleNamespace(key="image1", label="Brightfield"),
            SimpleNamespace(key="image2", label="Signal"),
        ]
    )

    save_general_overlay_stack(
        base_img=image_map["image1"],
        roi_map={},
        roi_keys=["__whole_cell_mask__"],
        out_path=tmp_path,
        result_id="Sample_001",
        base_label="Brightfield",
        cfg=cfg,
        log_func=lambda _msg: None,
        extra_mask_map={"__whole_cell_mask__": np.ones((4, 5), dtype=np.uint8)},
        image_map=image_map,
    )

    stack = tifffile.imread(tmp_path / "Sample_001_Brightfield_combined_overlay.tif")

    assert stack.dtype == np.uint16
    assert np.all(stack[0] == 7)
    assert np.all(stack[1] == 4000)


def test_disabling_review_overlay_still_writes_final_binary_mask(tmp_path):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    overlay_dir = tmp_path / "overlays"
    binary_dir = tmp_path / "binary"
    image = np.full((4, 5), 7, dtype=np.uint8)
    cfg = SimpleNamespace(images=[SimpleNamespace(key="image1", label="Channel 1")])

    save_general_overlay_stack(
        base_img=image,
        roi_map={},
        roi_keys=["__whole_cell_mask__"],
        out_path=overlay_dir,
        result_id="Sample_001",
        base_label="Channel 1",
        cfg=cfg,
        log_func=lambda _msg: None,
        extra_mask_map={"__whole_cell_mask__": np.ones((4, 5), dtype=np.uint8)},
        binary_out_path=binary_dir,
        write_overlay=False,
    )

    binary_mask = tifffile.imread(binary_dir / "Sample_001_WholeCellMask_binary.tif")
    assert np.all(binary_mask == 255)
    assert not (overlay_dir / "Sample_001_Channel_1_combined_overlay.tif").exists()
    assert not (overlay_dir / "Sample_001_Channel_1_combined_overlay.json").exists()


def test_binary_mask_write_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        overlay_exports,
        "write_tiff",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    cfg = SimpleNamespace(images=[SimpleNamespace(key="image1", label="Channel 1")])

    with pytest.raises(OSError, match="disk full"):
        save_general_overlay_stack(
            base_img=np.ones((4, 5), dtype=np.uint8),
            roi_map={},
            roi_keys=["__whole_cell_mask__"],
            out_path=tmp_path / "overlays",
            result_id="Sample_001",
            base_label="Channel 1",
            cfg=cfg,
            log_func=lambda _msg: None,
            extra_mask_map={"__whole_cell_mask__": np.ones((4, 5), dtype=np.uint8)},
            binary_out_path=tmp_path / "binary",
        )


def test_overlay_sidecar_write_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        overlay_exports,
        "write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("sidecar denied")),
    )
    cfg = SimpleNamespace(images=[SimpleNamespace(key="image1", label="Channel 1")])

    with pytest.raises(OSError, match="sidecar denied"):
        save_general_overlay_stack(
            base_img=np.ones((4, 5), dtype=np.uint8),
            roi_map={},
            roi_keys=["__whole_cell_mask__"],
            out_path=tmp_path / "overlays",
            result_id="Sample_001",
            base_label="Channel 1",
            cfg=cfg,
            log_func=lambda _msg: None,
            extra_mask_map={"__whole_cell_mask__": np.ones((4, 5), dtype=np.uint8)},
            binary_out_path=tmp_path / "binary",
        )


def test_overlay_stack_sidecar_marks_weka_mask_layers(tmp_path, monkeypatch):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    image_map = {
        "image1": np.full((4, 5), 7, dtype=np.uint8),
        "image2": np.full((4, 5), 11, dtype=np.uint8),
    }
    cfg = SimpleNamespace(
        images=[
            SimpleNamespace(key="image1", label="Brightfield"),
            SimpleNamespace(key="image2", label="mCherry"),
        ]
    )

    monkeypatch.setattr(
        overlay_exports,
        "_mask_array_from_roi_or_extra",
        lambda _base_img, _roi, _extra_mask, label_shape: np.ones(label_shape, dtype=bool),
    )

    save_general_overlay_stack(
        base_img=image_map["image1"],
        roi_map={"image2": object()},
        roi_keys=["image2", "__whole_cell_mask__"],
        out_path=tmp_path,
        result_id="Sample_001",
        base_label="Brightfield",
        cfg=cfg,
        log_func=lambda _msg: None,
        extra_mask_map={"__whole_cell_mask__": np.ones((4, 5), dtype=np.uint8)},
        image_map=image_map,
    )

    sidecar = json.loads((tmp_path / "Sample_001_Brightfield_combined_overlay.json").read_text(encoding="utf-8"))

    assert sidecar["layer_labels"] == ["Brightfield", "mCherry", "mCherry", "WholeCellMask"]
    assert sidecar["layer_roles"] == ["image", "image", "weka_mask", "cellpose_mask"]
    assert sidecar["layer_keys"] == ["image1", "image2", "image2", "__whole_cell_mask__"]

def test_overlay_stack_omits_duplicate_mask_source_image(tmp_path, monkeypatch):
    source = np.full((4, 5), 7, dtype=np.uint8)
    image_map = {"image1": source, "image2": source}
    cfg = SimpleNamespace(
        images=[
            SimpleNamespace(key="image1", label="GFP", folder_name="GFP"),
            SimpleNamespace(key="image2", label="GFP mask", folder_name="GFP", stack_source_image_key="image1"),
        ]
    )
    monkeypatch.setattr(
        overlay_exports, "_mask_array_from_roi_or_extra",
        lambda _base_img, _roi, _extra_mask, label_shape: np.ones(label_shape, dtype=bool),
    )

    save_general_overlay_stack(
        base_img=source,
        roi_map={"image2": object()},
        roi_keys=["image2"],
        out_path=tmp_path,
        result_id="Sample_001",
        base_label="GFP",
        cfg=cfg,
        log_func=lambda _message: None,
        image_map=image_map,
    )

    sidecar = json.loads((tmp_path / "Sample_001_GFP_combined_overlay.json").read_text(encoding="utf-8"))
    assert sidecar["layer_labels"] == ["GFP", "GFP mask"]
    assert sidecar["layer_roles"] == ["image", "weka_mask"]
    assert sidecar["layer_keys"] == ["image1", "image2"]


def test_overlay_stack_labels_mask_layers_with_configured_mask_name(tmp_path, monkeypatch):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    image_map = {
        "image1": np.full((4, 5), 7, dtype=np.uint8),
        "image2": np.full((4, 5), 11, dtype=np.uint8),
    }
    cfg = SimpleNamespace(
        images=[
            SimpleNamespace(key="image1", label="GFP", folder_name="GFP", stack_channel_index=None),
            SimpleNamespace(key="image2", label="Hmg2 mask", folder_name="GFP", stack_channel_index=None),
        ]
    )

    monkeypatch.setattr(
        overlay_exports,
        "_mask_array_from_roi_or_extra",
        lambda _base_img, _roi, _extra_mask, label_shape: np.ones(label_shape, dtype=bool),
    )

    save_general_overlay_stack(
        base_img=image_map["image1"],
        roi_map={"image2": object()},
        roi_keys=["image2"],
        out_path=tmp_path,
        result_id="Sample_001",
        base_label="GFP",
        cfg=cfg,
        log_func=lambda _msg: None,
        image_map=image_map,
    )

    sidecar = json.loads((tmp_path / "Sample_001_GFP_combined_overlay.json").read_text(encoding="utf-8"))

    assert sidecar["layer_labels"] == ["GFP", "GFP", "Hmg2 mask"]
    assert sidecar["layer_roles"] == ["image", "image", "weka_mask"]
    assert sidecar["layer_keys"] == ["image1", "image2", "image2"]

def test_overlay_stack_infers_source_channel_colors_when_config_colors_are_blank(tmp_path, monkeypatch):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    image_map = {
        "image1": np.full((4, 5), 7, dtype=np.uint8),
        "image2": np.full((4, 5), 11, dtype=np.uint8),
        "image3": np.full((4, 5), 13, dtype=np.uint8),
    }
    cfg = SimpleNamespace(
        images=[
            SimpleNamespace(key="image1", label="GFP", folder_name="GFP", display_color="", stack_channel_index=None),
            SimpleNamespace(key="image2", label="TxRed", folder_name="TxRed", display_color="", stack_channel_index=None),
            SimpleNamespace(key="image3", label="DAPI", folder_name="DAPI", display_color="", stack_channel_index=None),
        ]
    )

    monkeypatch.setattr(
        overlay_exports,
        "_mask_array_from_roi_or_extra",
        lambda _base_img, _roi, _extra_mask, label_shape: np.ones(label_shape, dtype=bool),
    )

    save_general_overlay_stack(
        base_img=image_map["image1"],
        roi_map={"image2": object()},
        roi_keys=["image2"],
        out_path=tmp_path,
        result_id="Sample_001",
        base_label="GFP",
        cfg=cfg,
        log_func=lambda _msg: None,
        image_map=image_map,
    )

    sidecar = json.loads((tmp_path / "Sample_001_GFP_combined_overlay.json").read_text(encoding="utf-8"))

    assert sidecar["layer_labels"] == ["GFP", "TxRed", "DAPI", "TxRed"]
    assert sidecar["layer_colors"][:3] == ["#00FF00", "#FF0000", "#0000FF"]
    assert sidecar["layer_colors"][3] not in set(sidecar["layer_colors"][:3])

def test_overlay_stack_assigns_mask_colors_distinct_from_image_layers(tmp_path, monkeypatch):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    image_map = {
        "image1": np.full((4, 5), 7, dtype=np.uint8),
        "image2": np.full((4, 5), 11, dtype=np.uint8),
        "image3": np.full((4, 5), 13, dtype=np.uint8),
    }
    cfg = SimpleNamespace(
        images=[
            SimpleNamespace(key="image1", label="GFP", folder_name="GFP", display_color="#00FF00", stack_channel_index=None),
            SimpleNamespace(key="image2", label="TxRed", folder_name="TxRed", display_color="#FF00FF", stack_channel_index=None),
            SimpleNamespace(key="image3", label="DAPI", folder_name="DAPI", display_color="#00FFFF", stack_channel_index=None),
        ]
    )

    monkeypatch.setattr(
        overlay_exports,
        "_mask_array_from_roi_or_extra",
        lambda _base_img, _roi, _extra_mask, label_shape: np.ones(label_shape, dtype=bool),
    )

    save_general_overlay_stack(
        base_img=image_map["image1"],
        roi_map={"image2": object(), "image3": object()},
        roi_keys=["image2", "image3", "__whole_cell_mask__"],
        out_path=tmp_path,
        result_id="Sample_001",
        base_label="GFP",
        cfg=cfg,
        log_func=lambda _msg: None,
        extra_mask_map={"__whole_cell_mask__": np.ones((4, 5), dtype=np.uint8)},
        image_map=image_map,
    )

    sidecar = json.loads((tmp_path / "Sample_001_GFP_combined_overlay.json").read_text(encoding="utf-8"))
    image_colors = {
        color
        for color, role in zip(sidecar["layer_colors"], sidecar["layer_roles"])
        if role == "image"
    }
    mask_colors = [
        color
        for color, role in zip(sidecar["layer_colors"], sidecar["layer_roles"])
        if role != "image"
    ]

    assert image_colors == {"#00FF00", "#FF00FF", "#00FFFF"}
    assert mask_colors
    assert all(color not in image_colors for color in mask_colors)
    assert len(mask_colors) == len(set(mask_colors))
