from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cellonaut.io.nd2_import import (
    _select_channel_plane,
    auto_build_channel_map_from_names,
    get_nd2_channel_colors,
    get_nd2_channel_names,
    metadata_color_to_hex,
    read_first_nd2_position,
)


def test_metadata_color_to_hex_reads_nd2_color_object():
    color = SimpleNamespace(r=12, g=34, b=56)

    assert metadata_color_to_hex(color) == "#0C2238"


def test_get_nd2_channel_colors_prefers_metadata_colors():
    nd2_file = SimpleNamespace(
        metadata=SimpleNamespace(
            channels=[
                SimpleNamespace(channel=SimpleNamespace(color=SimpleNamespace(r=0, g=255, b=0))),
                SimpleNamespace(channel=SimpleNamespace(color=SimpleNamespace(r=255, g=0, b=0))),
                SimpleNamespace(channel=SimpleNamespace(color=SimpleNamespace(r=0, g=0, b=255))),
            ]
        ),
        sizes={"C": 3},
    )

    colors = get_nd2_channel_colors(nd2_file, ["GFP", "TxRed", "DAPI"])

    assert colors == ["#00FF00", "#FF0000", "#0000FF"]


def test_get_nd2_channel_colors_falls_back_to_channel_names():
    nd2_file = SimpleNamespace(metadata=SimpleNamespace(channels=[]), sizes={"C": 2})

    colors = get_nd2_channel_colors(nd2_file, ["GFP", "DAPI"])

    assert colors == ["#00FF00", "#0000FF"]


def test_auto_channel_map_defaults_to_detected_channel_names():
    assert auto_build_channel_map_from_names(["GFP", "DAPI"]) == {"GFP": "GFP", "DAPI": "DAPI"}


def test_channel_names_replace_blanks_and_make_duplicates_unique():
    nd2_file = SimpleNamespace(
        metadata=SimpleNamespace(
            channels=[
                SimpleNamespace(channel=SimpleNamespace(name="GFP")),
                SimpleNamespace(channel=SimpleNamespace(name="gfp")),
                SimpleNamespace(channel=SimpleNamespace(name="")),
            ]
        ),
        sizes={"C": 3},
    )

    assert get_nd2_channel_names(nd2_file) == ["GFP", "gfp 2", "Channel 3"]


def test_channel_plane_uses_first_xy_position_before_z_projection():
    array = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.uint16).reshape(2, 3, 2, 4, 5)
    sizes = {"P": 2, "Z": 3, "C": 2, "Y": 4, "X": 5}

    selected = _select_channel_plane(array, sizes, channel_index=1, z_mode="max_projection", z_index=0)

    assert np.array_equal(selected, np.max(array[0, :, 1], axis=0))


def test_nd2_reader_requests_only_the_first_xy_position():
    expected = np.arange(12, dtype=np.uint16).reshape(3, 4)

    class FakeNd2File:
        def __init__(self):
            self.requested_position = None

        def asarray(self, *, position=None):
            self.requested_position = position
            return expected

    nd2_file = FakeNd2File()

    result = read_first_nd2_position(nd2_file)

    assert nd2_file.requested_position == 0
    assert np.shares_memory(result, expected)


def test_malformed_rgb_metadata_is_ignored():
    assert metadata_color_to_hex(["red", 0, 0]) == ""
    assert metadata_color_to_hex([np.nan, 0, 0]) == ""
