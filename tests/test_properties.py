"""Property-based invariants for mutation-sensitive numerical helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cellonaut.masks.cell_qc import (
    cell_label_outline_mask,
    combine_qc_label_sets,
    make_mask_for_cell_labels,
)
from cellonaut.measurement.math import (
    base_name_no_ext,
    parse_radii_csv,
    summarize_per_cell_table,
)

hypothesis = pytest.importorskip("hypothesis")
st = importlib.import_module("hypothesis.strategies")
given = hypothesis.given
settings = hypothesis.settings


PROPERTY_SETTINGS = settings(max_examples=40, deadline=None, derandomize=True, database=None)


@st.composite
def label_images(draw: Any) -> np.ndarray:
    height = draw(st.integers(min_value=1, max_value=8))
    width = draw(st.integers(min_value=1, max_value=8))
    pixels = draw(
        st.lists(
            st.integers(min_value=0, max_value=6),
            min_size=height * width,
            max_size=height * width,
        )
    )
    return np.asarray(pixels, dtype=np.int16).reshape(height, width)


@PROPERTY_SETTINGS
@given(
    st.text(
        alphabet=st.characters(
            blacklist_characters="/\\\x00",
            blacklist_categories=("Cs",),
        ),
        min_size=1,
        max_size=60,
    )
)
def test_model_stem_sanitization_is_deterministic_and_filename_safe(name: str):
    result = base_name_no_ext(Path(f"{name}.model"))

    assert result == base_name_no_ext(Path(f"{name}.model"))
    assert all(character.isalnum() or character in "._-" for character in result)
    assert len(result) == len(Path(f"{name}.model").stem)


@PROPERTY_SETTINGS
@given(
    st.lists(
        st.floats(
            min_value=0,
            max_value=1_000_000,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        max_size=20,
    )
)
def test_radius_parser_round_trips_every_finite_nonnegative_list(values: list[float]):
    encoded = ",".join(repr(value) for value in values)
    assert parse_radii_csv(encoded, strict=True) == pytest.approx(values)


@PROPERTY_SETTINGS
@given(
    st.floats(
        min_value=-1_000_000,
        max_value=-np.finfo(np.float32).tiny,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
)
def test_radius_parser_strictly_rejects_every_negative_finite_value(value: float):
    with pytest.raises(ValueError, match="must contain non-negative numbers"):
        parse_radii_csv(repr(value), strict=True)


@PROPERTY_SETTINGS
@given(label_images(), st.sets(st.integers(min_value=1, max_value=6), max_size=6))
def test_cell_label_selection_matches_membership_without_mutation(
    labels: np.ndarray,
    excluded_labels: set[int],
):
    original = labels.copy()
    selected = make_mask_for_cell_labels(labels, excluded_labels)

    if not excluded_labels:
        assert selected is None
    else:
        expected_selection = np.isin(labels, sorted(excluded_labels))
        assert selected is not None
        assert np.array_equal(selected, expected_selection)
    assert np.array_equal(labels, original)


@PROPERTY_SETTINGS
@given(label_images())
def test_cell_outlines_are_binary_shape_stable_and_never_mark_background(labels: np.ndarray):
    outline = cell_label_outline_mask(labels)

    assert outline.shape == labels.shape
    assert outline.dtype == np.uint8
    assert set(np.unique(outline)).issubset({0, 255})
    assert not np.any(outline[labels == 0])
    if labels.shape[0] and labels.shape[1]:
        border_foreground = np.zeros(labels.shape, dtype=bool)
        border_foreground[0, :] = labels[0, :] > 0
        border_foreground[-1, :] = labels[-1, :] > 0
        border_foreground[:, 0] |= labels[:, 0] > 0
        border_foreground[:, -1] |= labels[:, -1] > 0
        assert np.all(outline[border_foreground] == 255)


@PROPERTY_SETTINGS
@given(
    st.sets(st.integers(min_value=0, max_value=30), max_size=20),
    st.sets(st.integers(min_value=0, max_value=30), max_size=20),
)
def test_qc_filter_modes_follow_exact_set_algebra(cell_labels: set[int], mask_labels: set[int]):
    cells = cell_labels - {0}
    masks = mask_labels - {0}

    assert combine_qc_label_sets(cell_labels, mask_labels, "Exclude if any filter fails") == cells | masks
    assert combine_qc_label_sets(cell_labels, mask_labels, "Exclude only if both fail") == cells & masks
    assert combine_qc_label_sets(cell_labels, mask_labels, "Use cell filters only") == cells
    assert combine_qc_label_sets(cell_labels, mask_labels, "Use mask filters only") == masks


@PROPERTY_SETTINGS
@given(
    st.lists(
        st.floats(
            min_value=-1_000_000,
            max_value=1_000_000,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=1,
        max_size=30,
    )
)
def test_per_cell_summary_preserves_sum_mean_and_input_table(values: list[float]):
    table = pd.DataFrame({"CellArea": values, "CellMean": values})
    original = table.copy(deep=True)

    result = summarize_per_cell_table(table, "Mask", "Signal")

    assert result["Signal_measured_with_Signal_cellpose_mask_PerCell_TotalCellArea"] == pytest.approx(
        sum(values)
    )
    assert result["Signal_measured_with_Signal_cellpose_mask_PerCell_MeanOfCellMeans"] == pytest.approx(
        sum(values) / len(values)
    )
    pd.testing.assert_frame_equal(table, original)
