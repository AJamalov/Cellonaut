"""Shared membership semantics and operation-local metric preparation."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.config.adapter import parse_cell_qc_limits_text
from cellonaut.gui import preview_filter as gui_module
from cellonaut.masks import cell_qc, preview_filter_overlay as exports
from cellonaut.masks.cell_groups import evaluate_cell_groups


class PreviewHarness(gui_module.CellonautGuiPreviewFilterMixin):
    def __init__(self, preview_file, definition, prepare):
        self.preview_state = PreviewState()
        self.preview_state.file_path = str(preview_file)
        self.definition = definition
        self.prepare = prepare
        self.populations = []
        self.unavailable = None

    def get_active_image_definitions(self):
        return [self.definition]

    def parse_cell_qc_limits_text(self, text):
        return parse_cell_qc_limits_text(text)

    def attach_preview_mask_filter_metrics(self, table, label_image, image_def):
        return self.prepare(table, image_def)

    def rebuild_preview_population_layers(self, populations):
        self.populations = populations

    def set_preview_filter_unavailable(self, message, *, failed=False):
        self.unavailable = (message, failed)


def render_preview(tmp_path, monkeypatch, table, definition, prepare):
    path = tmp_path / "preview.tif"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(gui_module, "find_preview_filter_data", lambda *_args: (
        SimpleNamespace(source_label="GFP", table=table, label_image=np.array([[1]]), table_path=path), ""
    ))
    preview = PreviewHarness(path, definition, prepare)
    preview._refresh_preview_filter_overlay()
    return preview


def test_preview_and_export_share_memberships_overlaps_nan_empty_groups_and_exclusions(tmp_path, monkeypatch):
    table = pd.DataFrame({"CellID": [1, 2, 3, 4, 5, "ALL_CELLS_MEAN"],
                          "Area": [2, 5, 8, 10, 20, 9], "Mean": [1, 10, np.nan, 8, 20, 9]})
    before = table.copy(deep=True)
    definition = {"name": "GFP", "mask_relationships": {"Mask": True}, "cell_populations": [
        {"name": "Medium bright", "cell_qc_limits": "Area:5-15; Mean intensity:8-15", "exclude_from_csv": True},
        {"name": "Mask bright", "mask_qc_limits": "Mask area:1-; Mean intensity inside mask:6-"},
        {"name": "Very bright", "cell_qc_limits": "Mean intensity:15-", "exclude_from_csv": True},
        {"name": "No matches", "cell_qc_limits": "Area:100-", "exclude_from_csv": True},
        {"name": "Unconfigured", "exclude_from_csv": True},
    ]}
    labels = np.repeat(np.arange(1, 6), 2).reshape(2, 5)
    pixels = np.repeat([1, 10, np.nan, 8, 20], 2).reshape(2, 5)
    calls = []
    def prepare(current, settings):
        calls.append(settings)
        return exports.attach_live_mask_metrics(current, labels, np.ones_like(labels), pixels)

    preview = render_preview(tmp_path, monkeypatch, table, definition, prepare)
    result = evaluate_cell_groups(table, definition, parse_cell_qc_limits_text, prepare_table=prepare)
    kept, report, excluded, summary = exports.filtered_table_from_current_rules(
        table, definition, parse_cell_qc_limits_text, prepare_table=prepare)
    expected = [{2, 4}, {2, 4, 5}, {5}, set(), set()]
    assert preview.unavailable is None
    assert [group["labels"] for group in preview.populations] == expected
    assert [group.labels for group in result.groups] == expected
    for name, members in zip(["Medium_bright", "Mask_bright", "Very_bright", "No_matches", "Unconfigured"], expected):
        assert set(report.loc[report[f"CellGroup_{name}"], "CellID"]) == members
    assert report["CellGroupCount"].tolist() == [0, 2, 0, 2, 2, 0]
    assert excluded == result.excluded_labels == {2, 4, 5}
    assert kept["CellID"].tolist() == [1, 3]
    assert result.kept_labels == {1, 3}
    assert summary["group_counts"] == {"Medium bright": 2, "Mask bright": 3, "Very bright": 1,
                                       "No matches": 0, "Unconfigured": 0}
    assert len(calls) == 3  # One preparation in each independent operation.
    pd.testing.assert_frame_equal(table, before)


def test_unavailable_metrics_are_explicit_preview_empty_and_export_refuses(tmp_path, monkeypatch):
    table = pd.DataFrame({"CellID": [1, 2], "Area": [1, 4]})
    definition = {"name": "GFP", "cell_populations": [
        {"name": "Known", "cell_qc_limits": "Area:2-"},
        {"name": "Unavailable", "cell_qc_limits": "Solidity:0.5-", "exclude_from_csv": True},
    ]}
    def prepare(current, _settings):
        return current
    result = evaluate_cell_groups(table, definition, parse_cell_qc_limits_text)
    assert result.groups[1].missing_metrics == ("Solidity",)
    assert result.groups[1].labels == set()
    assert not result.excluded_labels
    preview = render_preview(tmp_path, monkeypatch, table, definition, prepare)
    assert preview.unavailable is None
    assert [group["labels"] for group in preview.populations] == [{2}, set()]
    with pytest.raises(ValueError, match="metrics unavailable: Solidity"):
        exports.filtered_table_from_current_rules(table, definition, parse_cell_qc_limits_text)


def test_preparation_errors_are_shared_by_affected_groups_and_do_not_mutate_table():
    table = pd.DataFrame({"CellID": [1, 2], "Area": [1, 4]})
    original = table.copy()
    definition = {"cell_populations": [
        {"name": "One", "mask_qc_limits": "Mask area:2-"},
        {"name": "Two", "mask_qc_limits": "Mean intensity inside mask:2-"},
        {"name": "Known", "cell_qc_limits": "Area:2-"},
    ]}
    calls = []
    def prepare(current, _settings):
        calls.append(1)
        current["Area"] = 0
        raise ValueError("source pixels unavailable")
    result = evaluate_cell_groups(table, definition, parse_cell_qc_limits_text, prepare_table=prepare)
    assert len(calls) == 1
    assert result.groups[0].error == result.groups[1].error == "ValueError: source pixels unavailable"
    assert result.groups[2].labels == {2}
    with pytest.raises(ValueError, match="source pixels unavailable"):
        result.require_available()
    pd.testing.assert_frame_equal(table, original)


def test_all_mask_metrics_prepared_once_per_source_and_recomputed_next_operation(monkeypatch):
    labels = np.array([[1, 1], [2, 2]])
    table = pd.DataFrame({"CellID": [1, 2]})
    definition = {"name": "GFP", "analysis_cell_segmentation_source": "BF", "mask_relationships": {"A": True},
                  "cell_populations": [
        {"name": "Area", "mask_qc_limits": "Mask area:1-"},
        {"name": "Measured", "mask_qc_limits": "Mean intensity inside mask:50-"},
        {"name": "Source", "mask_qc_limits": "Mean intensity inside mask:50-",
         "mask_qc_intensity_source": "Cell mask image", "exclude_from_csv": True},
        {"name": "Source again", "mask_qc_limits": "Mask fraction of cell area:0.5-",
         "mask_qc_intensity_source": "Cell mask image"},
    ]}
    gui: Any = gui_module.CellonautGuiPreviewFilterMixin()
    gui.preview_state = PreviewState()
    gui.parse_cell_qc_limits_text = parse_cell_qc_limits_text
    gui.preview_state.tiff_model = {"data": np.array([[[100, 100], [10, 10]], [[10, 10], [100, 100]],
                                                [[1, 1], [1, 1]]])[None, None]}
    gui.preview_state.current_labels = ["GFP", "BF", "A"]
    gui.preview_state.current_layer_roles = ["image", "image", "weka_mask"]
    calls = []
    original = exports.make_mask_qc_metric_table
    def metrics(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(exports, "make_mask_qc_metric_table", metrics)
    def prepare(current, settings):
        return gui.attach_preview_mask_filter_metrics(current, labels, settings)
    result = evaluate_cell_groups(table, definition, parse_cell_qc_limits_text, prepare_table=prepare)
    result.require_available()
    assert [group.labels for group in result.groups] == [{1, 2}, {1}, {2}, {1, 2}]
    assert result.excluded_labels == {2}
    assert len(calls) == 2
    gui.preview_state.tiff_model["data"][0, 0, 0] = 0
    refreshed = evaluate_cell_groups(table, definition, parse_cell_qc_limits_text, prepare_table=prepare)
    assert refreshed.groups[1].labels == set()
    assert len(calls) == 4


@pytest.mark.parametrize("mode, expected", [
    ("Exclude if any filter fails", {2}), ("Exclude only if both fail", {1, 2, 3}),
    ("Use cell filters only", {1, 2}), ("Use mask filters only", {2, 3}),
])
def test_category_combinations_keep_existing_semantics(mode, expected):
    table = pd.DataFrame({"CellID": [1, 2, 3], "Area": [1, 2, 3], "PositiveAreaInCell": [1, 2, 3]})
    definition = {"cell_populations": [{"cell_qc_limits": "Area:-2", "mask_qc_limits": "Mask area:2-",
                                       "qc_filter_mode": mode}]}
    assert evaluate_cell_groups(table, definition, parse_cell_qc_limits_text).groups[0].labels == expected


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_grouped_mask_metrics_match_original_per_cell_scans_exactly(dtype):
    rng = np.random.default_rng(74)
    labels = rng.choice([0, 1, 9001, 2_000_000_000], (37, 43)).astype(np.int64)
    mask = rng.integers(0, 2, labels.shape).astype(bool)
    pixels = rng.integers(0, 1000, labels.shape).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        pixels /= 13
        pixels[0, 0] = np.nan
    originals = (labels.copy(), mask.copy(), pixels.copy())
    expected = []
    # Reference algorithm used before this refactor: full-image scan per cell.
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        cell = labels == label
        positive = cell & mask
        area = int(cell.sum())
        positive_area = int(positive.sum())
        total = float(pixels[cell].sum())
        signal = float(pixels[positive].sum())
        expected.append({"CellID": label, "MaskQC_CellArea": area, "PositiveAreaInCell": positive_area,
                         "PositiveAreaFractionInCell": positive_area / area,
                         "MeanInPositiveArea": float(pixels[positive].mean()) if positive_area else np.nan,
                         "RawIntDenInCell": signal, "RawIntDenPerCellArea": signal / area,
                         "FractionOfCellIntDen": signal / total if total != 0 else np.nan})
    actual = cell_qc.make_mask_qc_metric_table(labels, mask, pixels)
    pd.testing.assert_frame_equal(actual, pd.DataFrame(expected), check_exact=True)
    for original, current in zip(originals, (labels, mask, pixels)):
        np.testing.assert_array_equal(original, current)


def test_saved_batch_reuses_label_image_sidecar_and_overlay_planes(tmp_path, monkeypatch):
    import json
    import tifffile
    from cellonaut.io import image_io
    from cellonaut.results.layout import build_results_layout

    layout = build_results_layout(tmp_path)
    for key in ("cell_segmentation_tables", "cell_segmentation_labels", "mask_overlays", "mask_overlay_metadata"):
        layout[key].mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"CellID": [1, 2], "Area": [2, 2]}).to_csv(
        layout["cell_segmentation_tables"] / "sample_GFP_cell_measurements.csv", index=False)
    labels_path = layout["cell_segmentation_labels"] / "sample_GFP_01_cellpose_labels.tif"
    tifffile.imwrite(labels_path, np.array([[1, 1], [2, 2]], dtype=np.int32))
    overlay = layout["mask_overlays"] / "sample_GFP_combined_overlay.tif"
    tifffile.imwrite(overlay, np.array([[[100, 100], [10, 10]], [[10, 10], [100, 100]], [[1, 1], [1, 1]]], dtype=np.uint16),
                     metadata={"axes": "CYX"}, photometric="minisblack")
    (layout["mask_overlay_metadata"] / overlay.with_suffix(".json").name).write_text(
        json.dumps({"layer_labels": ["GFP", "BF", "A"]}))
    definition = {"name": "GFP", "analysis_cell_segmentation_source": "BF", "mask_relationships": {"A": True},
                  "cell_populations": [
        {"name": "Measured", "mask_qc_limits": "Mean intensity inside mask:50-"},
        {"name": "Measured again", "mask_qc_limits": "Mask area:1-"},
        {"name": "Source", "mask_qc_limits": "Mean intensity inside mask:50-",
         "mask_qc_intensity_source": "Cell mask image", "exclude_from_csv": True},
        {"name": "Source again", "mask_qc_limits": "Mask area:1-", "mask_qc_intensity_source": "Cell mask image"},
    ]}
    reads, planes, sidecars, calculations = [], [], [], []
    original_read, original_plane = tifffile.imread, image_io.read_tiff_numpy_2d
    original_sidecar, original_metrics = exports._load_preview_sidecar, exports.make_mask_qc_metric_table
    def read(path, *args, **kwargs):
        reads.append(Path(path))
        return original_read(path, *args, **kwargs)
    def plane(path, **kwargs):
        planes.append(kwargs["stack_channel_index"])
        return original_plane(path, **kwargs)
    def sidecar(path):
        sidecars.append(path)
        return original_sidecar(path)
    def metrics(*args):
        calculations.append(1)
        return original_metrics(*args)
    monkeypatch.setattr(tifffile, "imread", read)
    monkeypatch.setattr(image_io, "read_tiff_numpy_2d", plane)
    monkeypatch.setattr(exports, "_load_preview_sidecar", sidecar)
    monkeypatch.setattr(exports, "make_mask_qc_metric_table", metrics)
    before = deepcopy(definition)
    result = exports.export_all_filtered_result_tables(layout["root"], [definition], parse_cell_qc_limits_text)
    assert result.processed_count == 1, result.messages
    assert reads.count(labels_path) == 1
    assert sorted(planes) == [1, 2, 3]
    assert sidecars == [overlay]
    assert len(calculations) == 2
    assert definition == before
    assert result.kept_combined_path is not None
    assert pd.read_csv(result.kept_combined_path)["CellID"].tolist() == [1]
