from __future__ import annotations

import numpy as np
import pytest
import tifffile

from cellonaut.io.fiji_tiff import (
    hex_to_lut,
    hex_to_ome_color,
    write_fiji_channel_stack,
    write_ome_channel_stack,
)


def test_invalid_hex_color_falls_back_to_white():
    expected_white = np.tile(np.arange(256, dtype=np.uint8), (3, 1))
    np.testing.assert_array_equal(hex_to_lut("#nothex"), expected_white)


def test_ome_colors_are_signed_opaque_rgba_values():
    assert hex_to_ome_color("#FFFFFF") == -1
    assert hex_to_ome_color("#FF0000") == -16776961
    assert hex_to_ome_color("#00FF00") == 16711935
    assert hex_to_ome_color("#0000FF") == 65535


def test_write_fiji_channel_stack_has_composite_metadata(tmp_path):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    path = tmp_path / "stack.tif"
    data = np.zeros((2, 4, 5), dtype=np.uint16)

    write_fiji_channel_stack(path, data, ["Base", "Mask"], ["#FFFFFF", "#00FFFF"])

    with tifffile.TiffFile(path) as tif:
        metadata = tif.imagej_metadata

    assert metadata is not None
    assert metadata["channels"] == 2
    assert metadata["mode"] == "composite"
    assert metadata["Labels"] == ["Base", "Mask"]
    assert len(metadata["LUTs"]) == 2


@pytest.mark.parametrize(
    ("labels", "colors", "message"),
    [
        (["Only one"], ["#FFFFFF", "#00FFFF"], "channel labels"),
        (["Base", "Mask"], ["#FFFFFF"], "channel colors"),
    ],
)
def test_fiji_channel_stack_rejects_metadata_count_mismatch(tmp_path, labels, colors, message):
    path = tmp_path / "stack.tif"
    data = np.zeros((2, 4, 5), dtype=np.uint16)

    with pytest.raises(ValueError, match=message):
        write_fiji_channel_stack(path, data, labels, colors)

    assert not path.exists()


def test_fiji_channel_stack_rejects_non_channel_stack_shape(tmp_path):
    with pytest.raises(ValueError, match="CYX dimensions"):
        write_fiji_channel_stack(
            tmp_path / "stack.tif",
            np.zeros((4, 5), dtype=np.uint16),
            ["Base"],
            ["#FFFFFF"],
        )


def test_fiji_channel_stack_rejects_unknown_display_mode(tmp_path):
    with pytest.raises(ValueError, match="Unsupported Fiji display mode"):
        write_fiji_channel_stack(
            tmp_path / "stack.tif",
            np.zeros((1, 4, 5), dtype=np.uint16),
            ["Base"],
            ["#FFFFFF"],
            mode="unknown",
        )


def test_write_ome_channel_stack_records_rgba_colors_and_pixel_size(tmp_path):
    path = tmp_path / "stack.ome.tif"
    data = np.zeros((2, 4, 5), dtype=np.uint16)

    write_ome_channel_stack(
        path,
        data,
        ["Red", "Green"],
        ["#FF0000", "#00FF00"],
        "Sample",
        physical_size_x=0.25,
        physical_size_y=0.5,
    )

    with tifffile.TiffFile(path) as tif:
        ome_xml = tif.ome_metadata or ""

    assert 'Name="Red" Color="-16776961"' in ome_xml
    assert 'Name="Green" Color="16711935"' in ome_xml
    assert 'PhysicalSizeX="0.25" PhysicalSizeXUnit="um"' in ome_xml
    assert 'PhysicalSizeY="0.5" PhysicalSizeYUnit="um"' in ome_xml
