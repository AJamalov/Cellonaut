from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellonaut.io import image_io
from cellonaut.io.image_io import (
    _optional_float,
    _unsigned_pixels_for_bit_depth,
    get_tiff_stack_info,
    normalize_numpy_image,
    read_tiff_numpy_2d,
    resolve_channel_indices_for_file_map,
    select_tiff_stack_channel,
)
from cellonaut.pipeline.discovery import STRUCTURE_FLAT_TIFFS, STRUCTURE_SAMPLES_DIRECTLY


def test_normalize_numpy_image_keeps_2d_arrays():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.array([[1, 2], [3, 4]], dtype=None)

    assert np.array_equal(normalize_numpy_image(arr), arr)


def test_normalize_numpy_image_uses_first_plane_by_default():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.array([[[1, 2], [3, 4]], [[9, 8], [7, 6]]], dtype=None)

    assert np.array_equal(normalize_numpy_image(arr), arr[0])


def test_normalize_numpy_image_can_use_max_projection_for_fallback_loader():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.array([[[1, 9], [3, 4]], [[9, 2], [7, 6]]], dtype=None)

    assert np.array_equal(normalize_numpy_image(arr, stack_mode="max"), np.array([[9, 9], [7, 6]], dtype=None))


def test_normalize_numpy_image_applies_single_index_once_to_flattened_pages():
    arr = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    selected = normalize_numpy_image(arr, stack_mode="single_z", stack_index=4)

    assert np.array_equal(selected, arr.reshape(6, 4, 5)[3])


def test_normalize_numpy_image_takes_first_rgb_channel():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[..., 0] = 5
    arr[..., 1] = 9

    assert np.array_equal(normalize_numpy_image(arr), np.full((2, 2), 5, dtype=np.uint8))


def test_unsigned_pixels_for_bit_depth_restores_signed_uint8_pixels():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    pixels = np.array([-1, 0, 127, -128], dtype=np.int8)

    assert np.array_equal(
        _unsigned_pixels_for_bit_depth(pixels, 8),
        np.array([255, 0, 127, 128], dtype=np.uint8),
    )


def test_unsigned_pixels_for_bit_depth_restores_signed_uint16_pixels():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    pixels = np.array([-1, 0, 32767, -32768], dtype=np.int16)

    assert np.array_equal(
        _unsigned_pixels_for_bit_depth(pixels, 16),
        np.array([65535, 0, 32767, 32768], dtype=np.uint16),
    )


def test_select_tiff_stack_channel_uses_one_based_layer():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.array(
        [
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
            [[3, 3], [3, 3]],
        ],
        dtype=None,
    )

    assert np.array_equal(select_tiff_stack_channel(arr, 2), np.full((2, 2), 2))


def test_select_tiff_stack_channel_max_projects_selected_channel_z_stack():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.zeros((2, 3, 2, 2), dtype=np.uint16)
    arr[1, 0] = np.array([[1, 10], [3, 4]], dtype=np.uint16)
    arr[1, 1] = np.array([[7, 2], [8, 1]], dtype=np.uint16)
    arr[1, 2] = np.array([[5, 6], [2, 9]], dtype=np.uint16)

    selected = select_tiff_stack_channel(arr, 2, axes="CZYX")

    assert np.array_equal(selected, np.array([[7, 10], [8, 9]], dtype=np.uint16))


def test_select_tiff_stack_channel_can_select_one_based_z_slice():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.zeros((1, 3, 2, 2), dtype=np.uint16)
    arr[0, 0] = 10
    arr[0, 1] = 42
    arr[0, 2] = 99

    selected = select_tiff_stack_channel(
        arr,
        1,
        axes="CZYX",
        stack_z_mode="single_z",
        stack_z_index=2,
    )

    assert np.array_equal(selected, np.full((2, 2), 42, dtype=np.uint16))


def test_select_tiff_stack_channel_z_index_does_not_apply_to_time_axis():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.zeros((1, 1, 3, 2, 2), dtype=np.uint16)
    arr[0, 0, 1] = 42

    selected = select_tiff_stack_channel(
        arr,
        1,
        axes="TCZYX",
        stack_z_mode="single_z",
        stack_z_index=2,
    )

    assert np.array_equal(selected, np.full((2, 2), 42, dtype=np.uint16))


def test_select_tiff_stack_channel_uses_first_timepoint_after_z_projection():
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    arr = np.zeros((2, 1, 2, 2, 2), dtype=np.uint16)
    arr[0, 0, 0] = 10
    arr[0, 0, 1] = 20
    arr[1, 0, 0] = 100
    arr[1, 0, 1] = 200

    selected = select_tiff_stack_channel(
        arr,
        1,
        axes="TCZYX",
        stack_z_mode="max_projection",
    )

    assert np.array_equal(selected, np.full((2, 2), 20, dtype=np.uint16))


def test_select_tiff_stack_channel_does_not_treat_explicit_z_as_channels():
    arr = np.zeros((3, 2, 2), dtype=np.uint16)
    arr[0] = 10
    arr[1] = 42
    arr[2] = 20

    selected = select_tiff_stack_channel(arr, axes="ZYX", stack_z_mode="max_projection")

    assert np.array_equal(selected, np.full((2, 2), 42, dtype=np.uint16))


def test_first_requested_layer_still_respects_explicit_z_metadata():
    arr = np.zeros((3, 2, 2), dtype=np.uint16)
    arr[0] = 10
    arr[1] = 42
    arr[2] = 20

    selected = select_tiff_stack_channel(
        arr,
        stack_channel_index=1,
        axes="ZYX",
        stack_z_mode="max_projection",
    )

    assert np.array_equal(selected, np.full((2, 2), 42, dtype=np.uint16))


def test_select_tiff_stack_channel_defaults_to_first_declared_channel():
    arr = np.zeros((2, 2, 2), dtype=np.uint16)
    arr[0] = 10
    arr[1] = 90

    selected = select_tiff_stack_channel(arr, axes="CYX")

    assert np.array_equal(selected, np.full((2, 2), 10, dtype=np.uint16))


def test_read_tiff_numpy_2d_reads_selected_stack_layer(tmp_path):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    path = tmp_path / "stack.tif"
    arr = np.array(
        [
            [[10, 10], [10, 10]],
            [[42, 42], [42, 42]],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(path, arr, photometric="minisblack")

    info = get_tiff_stack_info(path)
    selected = read_tiff_numpy_2d(path, stack_channel_index=2)

    assert info["channel_count"] == 2
    assert np.array_equal(selected, np.full((2, 2), 42, dtype=np.uint16))


def test_get_tiff_stack_info_reads_ome_channel_metadata(tmp_path):
    path = tmp_path / "channels.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((2, 4, 5), dtype=np.uint16),
        ome=True,
        metadata={
            "axes": "CYX",
            "Channel": {
                "Name": ["GFP", "DAPI"],
                "Color": [0x00FF00, 0x0000FF],
            },
            "PhysicalSizeX": 0.25,
            "PhysicalSizeY": 0.3,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
    )

    info = get_tiff_stack_info(path)

    assert info["channel_names"] == ["GFP", "DAPI"]
    assert info["channel_colors"] == ["#00FF00", "#0000FF"]
    assert info["physical_size_x"] == 0.25
    assert info["physical_size_y"] == 0.3


def test_get_tiff_stack_info_keeps_explicit_z_stack_as_one_channel(tmp_path):
    path = tmp_path / "z_stack.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((3, 4, 5), dtype=np.uint16),
        ome=True,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )

    info = get_tiff_stack_info(path)

    assert info["axes"] == "ZYX"
    assert info["channel_count"] == 1


def test_optional_physical_size_ignores_malformed_metadata():
    assert _optional_float("not-a-number") is None


class _ImageDef:
    def __init__(self, key: str, label: str, stack_channel_index=None, stack_source_image_key: str = ""):
        self.key = key
        self.label = label
        self.stack_channel_index = stack_channel_index
        self.stack_source_image_key = stack_source_image_key


def test_resolve_channel_indices_infers_flat_tiff_stack_layers_by_row_order(tmp_path):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, np.zeros((3, 4, 5), dtype=np.uint16), photometric="minisblack")
    logs = []

    result = resolve_channel_indices_for_file_map(
        image_defs=[
            _ImageDef("image1", "Channel 1"),
            _ImageDef("image2", "Channel 2"),
            _ImageDef("image3", "Channel 3"),
        ],
        file_map={"image1": path, "image2": path, "image3": path},
        input_structure=STRUCTURE_FLAT_TIFFS,
        log_func=logs.append,
    )

    assert result == {"image1": 1, "image2": 2, "image3": 3}
    assert "Channel 1=layer 1" in logs[0]


def test_resolve_channel_indices_keeps_explicit_and_non_flat_values(tmp_path):
    path = tmp_path / "sample.tif"
    path.write_bytes(b"not read")

    result = resolve_channel_indices_for_file_map(
        image_defs=[
            _ImageDef("image1", "Channel 1", 2),
            _ImageDef("image2", "Channel 2", None),
        ],
        file_map={"image1": path, "image2": path},
        input_structure=STRUCTURE_SAMPLES_DIRECTLY,
        log_func=lambda _msg: None,
    )

    assert result == {"image1": 2, "image2": None}


def test_resolve_channel_indices_avoids_explicit_layers_for_automatic_channels(tmp_path):
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, np.zeros((3, 4, 5), dtype=np.uint16), photometric="minisblack")

    result = resolve_channel_indices_for_file_map(
        image_defs=[
            _ImageDef("image1", "Channel 1", 2),
            _ImageDef("image2", "Channel 2"),
            _ImageDef("image3", "Channel 3"),
        ],
        file_map={"image1": path, "image2": path, "image3": path},
        input_structure=STRUCTURE_FLAT_TIFFS,
        log_func=lambda _msg: None,
    )

    assert result == {"image1": 2, "image2": 1, "image3": 3}


def test_resolve_channel_indices_does_not_count_mask_only_rows(tmp_path):
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, np.zeros((2, 4, 5), dtype=np.uint16), photometric="minisblack")

    result = resolve_channel_indices_for_file_map(
        image_defs=[
            _ImageDef("image1", "Channel 1"),
            _ImageDef("image2", "Channel 2"),
            _ImageDef("mask1", "Channel 1 mask", stack_source_image_key="image1"),
        ],
        file_map={"image1": path, "image2": path, "mask1": path},
        input_structure=STRUCTURE_FLAT_TIFFS,
        log_func=lambda _msg: None,
    )

    assert result == {"image1": 1, "image2": 2, "mask1": 1}


def test_stage_image_removes_partial_copy_when_cancelled(tmp_path, monkeypatch):
    source = tmp_path / "source.tif"
    source.write_bytes(b"x" * (2 * 1024 * 1024))
    staging_root = tmp_path / "temp"
    staging_root.mkdir()
    monkeypatch.setattr(image_io.tempfile, "gettempdir", lambda: str(staging_root))
    checks = iter([False, True])

    with pytest.raises(RuntimeError, match="cancelled"):
        image_io.stage_image_to_local_temp(
            source,
            lambda _msg: None,
            should_cancel=lambda: next(checks),
        )

    process_root = staging_root / "cellonaut_staged_images"
    assert not process_root.exists() or list(process_root.iterdir()) == []


def test_stage_image_preserves_cancellation_when_partial_cleanup_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.tif"
    source.write_bytes(b"x" * (2 * 1024 * 1024))
    staging_root = tmp_path / "temp"
    staging_root.mkdir()
    monkeypatch.setattr(image_io.tempfile, "gettempdir", lambda: str(staging_root))
    monkeypatch.setattr(Path, "unlink", lambda _path, **_kwargs: (_ for _ in ()).throw(PermissionError("locked")))
    checks = iter([False, True])

    with pytest.raises(RuntimeError, match="cancelled"):
        image_io.stage_image_to_local_temp(
            source,
            lambda _msg: None,
            should_cancel=lambda: next(checks),
        )


def test_cleanup_staged_image_process_directory_removes_only_requested_process(tmp_path, monkeypatch):
    monkeypatch.setattr(image_io.tempfile, "gettempdir", lambda: str(tmp_path))
    first = image_io.staged_image_process_directory(101)
    second = image_io.staged_image_process_directory(202)
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "partial.tif").write_bytes(b"partial")
    (second / "active.tif").write_bytes(b"active")

    image_io.cleanup_staged_image_process_directory(101)

    assert not first.exists()
    assert (second / "active.tif").is_file()


def test_open_channel_images_reuses_identical_tiff_plane(tmp_path, monkeypatch):
    path = tmp_path / "sample.tif"
    opened = []
    logs = []

    def fake_open(*_args, **_kwargs):
        image = object()
        opened.append(image)
        return image

    monkeypatch.setattr(image_io, "open_image_with_tifffile", fake_open)

    images = image_io.open_channel_images(
        {"image1": path, "image2": path}, "sample", logs.append,
        channel_indices={"image1": 1, "image2": 1},
    )

    assert images is not None
    assert len(opened) == 1
    assert images["image1"] is not None
    assert images["image2"] is not None
    assert images["image1"] is images["image2"]
    assert any("Reusing opened TIFF plane for image2" in message for message in logs)


def test_open_channel_images_keeps_distinct_stack_planes(tmp_path, monkeypatch):
    path = tmp_path / "stack.tif"
    opened_indices = []

    def fake_open(_path, *, stack_channel_index, **_kwargs):
        opened_indices.append(stack_channel_index)
        return object()

    monkeypatch.setattr(image_io, "open_image_with_tifffile", fake_open)

    images = image_io.open_channel_images(
        {"image1": path, "image2": path}, "sample", lambda _message: None,
        channel_indices={"image1": 1, "image2": 2},
    )

    assert images is not None
    assert opened_indices == [1, 2]
    assert images["image1"] is not None
    assert images["image2"] is not None
    assert images["image1"] is not images["image2"]


def test_open_channel_images_removes_staged_unc_copy(tmp_path, monkeypatch):
    staged_path = tmp_path / "staged.tif"
    opened_image = object()
    logs = []

    def fake_stage(*_args, **_kwargs):
        staged_path.write_bytes(b"staged")
        return staged_path

    monkeypatch.setattr(image_io, "stage_image_to_local_temp", fake_stage)
    monkeypatch.setattr(
        image_io,
        "open_image_with_tifffile",
        lambda path, *_args, **_kwargs: opened_image if path == staged_path else None,
    )

    result = image_io.open_channel_images(
        {"image1": Path(r"\\server\share\sample.tif")},
        "sample",
        logs.append,
    )

    assert result == {"image1": opened_image}
    assert not staged_path.exists()
    assert not any("Removed staged image" in message for message in logs)


def test_open_channel_images_removes_staged_unc_copy_when_open_fails(tmp_path, monkeypatch):
    staged_path = tmp_path / "staged.tif"

    def fake_stage(*_args, **_kwargs):
        staged_path.write_bytes(b"staged")
        return staged_path

    def fail_staged_open(path, *_args, **_kwargs):
        if path == staged_path:
            raise RuntimeError("open failed")
        return None

    monkeypatch.setattr(image_io, "stage_image_to_local_temp", fake_stage)
    monkeypatch.setattr(image_io, "open_image_with_tifffile", fail_staged_open)

    with pytest.raises(RuntimeError, match="open failed"):
        image_io.open_channel_images(
            {"image1": Path(r"\\server\share\sample.tif")},
            "sample",
            lambda _msg: None,
        )

    assert not staged_path.exists()

@pytest.mark.parametrize('count', [1, 2])
def test_automatic_channel_mapping_rejects_missing_physical_channels(tmp_path, count):
    path = tmp_path / 'incomplete.tif'
    tifffile.imwrite(path, np.full((count, 8, 8), 10, dtype=np.uint16),
                     metadata={'axes': 'CYX'}, photometric='minisblack')
    with pytest.raises(ValueError, match='physical channels are configured'):
        resolve_channel_indices_for_file_map(
            image_defs=[_ImageDef(key, key) for key in ('GFP', 'RFP', 'BF')],
            file_map={key: path for key in ('GFP', 'RFP', 'BF')},
            input_structure=STRUCTURE_FLAT_TIFFS, log_func=lambda _: None)


def test_automatic_channel_mapping_rejects_unreadable_metadata(tmp_path):
    path = tmp_path / 'broken.tif'
    path.write_bytes(b'broken')
    with pytest.raises(ValueError, match='Cannot automatically map'):
        resolve_channel_indices_for_file_map(
            image_defs=[_ImageDef('GFP', 'GFP')], file_map={'GFP': path},
            input_structure=STRUCTURE_FLAT_TIFFS, log_func=lambda _: None)
