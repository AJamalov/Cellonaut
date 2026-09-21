from __future__ import annotations

import numpy as np
import pytest
import tifffile

from cellonaut.gui.preview_normalization import CellonautGuiPreviewNormalizationMixin
from cellonaut.masks.cellpose import load_existing_cellpose_labels


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bad_value", [-1.0, 1.5, float("nan"), float("inf"), float(2**31)])
def test_reused_cell_labels_reject_invalid_ids(tmp_path, bad_value, dtype):
    path = tmp_path / "Results" / "Cells" / "TIFF Labels" / "sample_cell_01_cellpose_labels.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(path, np.array([[0, 1], [2, bad_value]], dtype=dtype))
    messages = []

    assert load_existing_cellpose_labels(tmp_path, "sample", "cell", messages.append) is None
    assert any("invalid cell IDs" in message and str(path) in message for message in messages)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float64])
def test_reused_cell_labels_preserve_valid_nonconsecutive_ids(tmp_path, dtype):
    path = tmp_path / "Results" / "Cells" / "TIFF Labels" / "sample_cell_01_cellpose_labels.tif"
    path.parent.mkdir(parents=True)
    original = np.array([[0, 7], [42, 42]], dtype=dtype)
    tifffile.imwrite(path, original)

    loaded = load_existing_cellpose_labels(tmp_path, "sample", "cell")

    assert loaded is not None
    np.testing.assert_array_equal(loaded, original)
    assert loaded.dtype == np.int32


@pytest.mark.parametrize("shape", [(1, 8), (8, 1), (1, 1)])
def test_preview_without_axis_metadata_preserves_single_pixel_dimensions(shape):
    original = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    preview = CellonautGuiPreviewNormalizationMixin()

    model = preview.normalize_tiff_to_tzcyx(original)

    assert model["data"].shape == (1, 1, 1, *shape)
    np.testing.assert_array_equal(model["data"][0, 0, 0], original)


@pytest.mark.parametrize("axes", ["TZCYX", "CTZYX", "YXTZC"])
def test_preview_axis_reordering_preserves_all_pixels(axes):
    target_axes = "TZCYX"
    expected = np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    source = expected.transpose([target_axes.index(axis) for axis in axes]).copy()
    before = source.copy()

    model = CellonautGuiPreviewNormalizationMixin().normalize_tiff_to_tzcyx(source, {"axes": axes})

    np.testing.assert_array_equal(model["data"], expected)
    np.testing.assert_array_equal(source, before)
