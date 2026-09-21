# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QModelIndex, QSize, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.gui import config as config_gui  # noqa: E402
from cellonaut.gui.main_window import CellonautMainWindow  # noqa: E402
from cellonaut.gui.preview import CellonautGuiPreviewMixin  # noqa: E402
from cellonaut.gui.preview_layer_list import PreviewLayerList  # noqa: E402
from cellonaut.gui.preview_layers import preview_layer_default_rank  # noqa: E402


pytestmark = pytest.mark.gui


class PreviewHarness(CellonautGuiPreviewMixin):
    pass

    def __init__(self):
        self.preview_state = PreviewState()


class CheckedStub:
    def __init__(self, checked):
        self.checked = checked

    def isChecked(self):
        return self.checked


def install_mask_adjustment_controls(preview):
    preview.mask_adjust_target_combo = QComboBox()
    preview.mask_adjust_x_spin = QSpinBox()
    preview.mask_adjust_x_spin.setRange(-500, 500)
    preview.mask_adjust_y_spin = QSpinBox()
    preview.mask_adjust_y_spin.setRange(-500, 500)
    preview.mask_adjust_grow_spin = QSpinBox()
    preview.mask_adjust_grow_spin.setRange(-50, 50)
    preview.mask_adjust_min_size_spin = QSpinBox()
    preview.mask_adjust_min_size_spin.setRange(0, 10_000_000)
    preview.mask_adjust_fill_holes_spin = QSpinBox()
    preview.mask_adjust_fill_holes_spin.setRange(0, 1000)


def test_constant_positive_mask_normalizes_to_visible_binary_layer():
    preview = PreviewHarness()
    mask = np.zeros((8, 9), dtype=np.uint16)
    mask[2:6, 3:7] = 4095

    shown = preview.normalize_for_preview(mask)

    assert shown.dtype == np.uint8
    assert shown[0, 0] == 0
    assert shown[3, 4] == 255
    assert np.count_nonzero(shown) == np.count_nonzero(mask)


def test_float_logo_uses_imagej_min_max_display_range():
    preview = PreviewHarness()
    logo_values = np.array([[0.0, 153.0, 171.33333, 255.0]], dtype=np.float32)

    shown = preview.normalize_for_preview(logo_values)

    np.testing.assert_array_equal(shown, [[0, 153, 171, 255]])


def test_float_probability_map_keeps_linear_imagej_display_range():
    preview = PreviewHarness()
    probabilities = np.array([[0.0, 0.25, 0.5, 1.0]], dtype=np.float32)

    shown = preview.normalize_for_preview(probabilities)

    np.testing.assert_array_equal(shown, [[0, 64, 128, 255]])


def test_tiff_preview_pages_are_created_as_lazy_placeholders():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = PreviewHarness()
    preview.preview_state.tiff_model = {"data": np.zeros((2, 3, 1, 8, 9), dtype=np.uint8)}
    preview.render_tiff_preview_page = lambda _index: (_ for _ in ()).throw(AssertionError("rendered eagerly"))

    pages, labels = preview.build_tiff_preview_pages()

    assert len(pages) == 6
    assert all(isinstance(page, QPixmap) and page.isNull() for page in pages)
    assert labels == [
        "T 1 / 2 | Z 1 / 3",
        "T 1 / 2 | Z 2 / 3",
        "T 1 / 2 | Z 3 / 3",
        "T 2 / 2 | Z 1 / 3",
        "T 2 / 2 | Z 2 / 3",
        "T 2 / 2 | Z 3 / 3",
    ]


def test_overlay_stack_uses_layer_count_as_channel_axis_for_masks():
    preview = PreviewHarness()
    stack = np.zeros((4, 8, 9), dtype=np.uint16)
    stack[0, :, :] = np.arange(72, dtype=np.uint16).reshape(8, 9)
    stack[1, 1:4, 1:4] = 1000
    stack[2, 2:6, 3:7] = 1000
    stack[3, 4:7, 5:8] = 1000

    model = preview.normalize_tiff_to_tzcyx(
        stack,
        sidecar={
            "type": "cellonaut_overlay_preview",
            "layer_labels": ["Base", "ROI", "WholeCellMask", "FilteredCells"],
        },
    )

    assert model["data"].shape == (1, 1, 4, 8, 9)
    assert np.count_nonzero(model["data"][0, 0, 2]) == np.count_nonzero(stack[2])


def test_small_grayscale_stack_does_not_treat_image_height_as_channels():
    preview = PreviewHarness()
    stack = np.arange(20 * 8 * 9, dtype=np.uint16).reshape(20, 8, 9)

    model = preview.normalize_tiff_to_tzcyx(stack)

    assert model["data"].shape == (1, 20, 1, 8, 9)
    assert np.array_equal(model["data"][0, 4, 0], stack[4])


def test_four_dimensional_stack_without_metadata_flattens_non_spatial_pages():
    preview = PreviewHarness()
    stack = np.arange(17 * 18 * 4 * 5, dtype=np.uint16).reshape(17, 18, 4, 5)

    model = preview.normalize_tiff_to_tzcyx(stack)

    assert model["data"].shape == (1, 17 * 18, 1, 4, 5)
    assert np.array_equal(model["data"][0, 18, 0], stack[1, 0])


def test_tifffile_q_axis_is_kept_as_preview_pages():
    preview = PreviewHarness()
    stack = np.arange(3 * 5 * 6, dtype=np.uint16).reshape(3, 5, 6)

    model = preview.normalize_tiff_to_tzcyx(stack, sidecar={"axes": "QYX"})

    assert model["data"].shape == (1, 3, 1, 5, 6)
    assert np.array_equal(model["data"][0, 2, 0], stack[2])


def test_mask_adjustment_does_not_guess_targets_without_layer_roles():
    preview = PreviewHarness()
    preview.preview_state.current_labels = [
        "GFP",
        "mCherry",
        "WholeCellMask",
        "Weka mask: DAPI",
        "FilteredCells",
    ]

    assert preview.preview_weka_mask_layer_indices() == []


def test_mask_adjustment_prefers_overlay_layer_roles_for_weka_targets():
    preview = PreviewHarness()
    preview.preview_state.current_labels = [
        "GFP",
        "mCherry",
        "mCherry",
        "WholeCellMask",
        "FlaggedCells",
    ]
    preview.preview_state.current_layer_roles = [
        "image",
        "image",
        "weka_mask",
        "cellpose_mask",
        "filtered_cells",
    ]

    assert preview.preview_weka_mask_layer_indices() == [2]


def test_mask_adjustment_loads_recipe_for_linked_layer():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = PreviewHarness()
    install_mask_adjustment_controls(preview)
    preview.preview_state.current_labels = ["GFP", "Hmg2"]
    preview.preview_state.current_layer_roles = ["image", "weka_mask"]
    preview.preview_state.current_layer_keys = ["image1", "image2"]
    active_defs = [
        {"name": "GFP"},
        {
            "name": "Hmg2 mask",
            "mask_adjustments": {
                "dx": 3,
                "dy": -1,
                "grow_px": 2,
                "min_size": 12,
                "fill_holes_area": 5,
            },
        },
    ]
    preview.get_active_image_definitions = lambda: active_defs
    preview.update_mask_adjust_target_options()
    preview.load_persisted_mask_adjustments()

    assert preview.mask_adjust_target_key() == "weka_layer:1"
    assert preview.mask_adjust_x_spin.value() == 3
    assert preview.mask_adjust_fill_holes_spin.value() == 5


def test_mask_adjustment_keeps_unsaved_values_when_tab_is_reopened():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = PreviewHarness()
    install_mask_adjustment_controls(preview)
    preview.preview_state.current_labels = ["GFP", "Mask"]
    preview.preview_state.current_layer_roles = ["image", "weka_mask"]
    preview.preview_state.current_layer_keys = ["image1", "image2"]
    preview.get_active_image_definitions = lambda: [
        {"name": "GFP"}, {"name": "Mask", "mask_adjustments": {"dx": 3}}
    ]
    preview.update_mask_adjust_target_options()
    preview.load_persisted_mask_adjustments()
    preview.mask_adjust_x_spin.setValue(7)
    preview.save_current_mask_adjustment_values()

    preview.load_persisted_mask_adjustments()

    assert preview.mask_adjust_x_spin.value() == 7


def test_mask_adjustment_reset_stays_available_for_adjusted_layer_with_zero_controls():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = PreviewHarness()
    install_mask_adjustment_controls(preview)
    preview.mask_adjust_target_combo.addItem("Mask", "weka_layer:1")
    preview.preview_state.mask_adjust_target = "weka_layer:1"
    preview.preview_state.current_labels = ["Image", "Mask"]
    preview.preview_state.current_layer_keys = ["image1", "mask1", "weka_layer:1__adjusted_preview"]
    preview.mask_adjust_status_label = QLabel()
    preview.mask_adjust_preview_button = QPushButton()
    preview.mask_adjust_reset_button = QPushButton()

    preview.update_mask_adjustment_status()

    assert preview.mask_adjust_reset_button.isEnabled()


def test_new_preview_clears_unsaved_mask_adjustments():
    preview = PreviewHarness()
    preview.preview_state.mask_adjustments_by_target = {"weka_layer:1": {"dx": 7}}
    preview.preview_state.mask_adjustment_last_preview_values = {"weka_layer:1": {"dx": 7}}
    preview.clear_preview_tiff_state = lambda: None
    preview.clear_preview_canvas = lambda: None
    preview.clear_preview_layer_controls = lambda: None
    preview.clear_preview_filter_overlay = lambda: None
    preview.update_preview_snapshot_controls = lambda: None

    preview.reset_preview_display()

    assert preview.preview_state.mask_adjustments_by_target == {}
    assert preview.preview_state.mask_adjustment_last_preview_values == {}


def test_cell_group_display_settings_survive_refresh():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = CellonautMainWindow()
    preview.preview_state.filter_data = SimpleNamespace(label_image=np.array([[1, 2]], dtype=np.uint16))
    groups = [
        {"key": "0", "name": "A", "color": "#00FFFF", "labels": {1}},
        {"key": "1", "name": "B", "color": "#FF0000", "labels": {2}},
    ]
    preview.rebuild_preview_population_layers(groups)
    first = preview.preview_layer_list.item(0)
    first_row = preview.preview_layer_list.itemWidget(first)
    first_row.findChild(QCheckBox).setChecked(False)
    first_row.findChild(QSlider).setValue(35)
    first_row.findChild(QPushButton).menu().actions()[5].trigger()

    preview.rebuild_preview_population_layers(groups)

    assert preview._preview_population_items["0"].isVisible() is False
    assert preview._preview_population_items["0"].opacity() == pytest.approx(0.35)
    assert preview.preview_state.group_layer_display_state["0"]["color"] == "#00FF00"
    assert preview.preview_layer_list.itemWidget(preview.preview_layer_list.item(0)).findChild(QCheckBox).isChecked() is False


def test_cell_group_link_failure_clears_old_layers_and_export():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = CellonautMainWindow()
    preview.preview_state.filter_data = SimpleNamespace(label_image=np.array([[1]], dtype=np.uint16))
    preview.rebuild_preview_population_layers(
        [{"key": "0", "name": "A", "color": "#00FFFF", "labels": {1}}]
    )
    preview.analysis_export_filtered_csv_button.setEnabled(True)

    preview.set_preview_filter_unavailable("linked table missing")

    assert preview.preview_state.filter_data is None
    assert preview._preview_population_items == {}
    assert preview.preview_layer_list.count() == 0
    assert not preview.analysis_export_filtered_csv_button.isEnabled()


def test_cellpose_outline_metadata_does_not_offer_source_channels_as_weka_targets():
    preview = PreviewHarness()
    preview.preview_state.current_labels = [
        "GFP",
        "mCherry",
        "CellposeOutline",
        "FilteredCellsOutline",
    ]
    preview.preview_state.current_layer_roles = [
        "image",
        "image",
        "cellpose_outline",
        "filtered_cells",
    ]

    assert preview.preview_weka_mask_layer_indices() == []


def test_overlay_layer_labels_rely_on_category_for_image_and_mask_identity():
    preview = PreviewHarness()
    preview.preview_state.current_layer_roles = [
        "image",
        "weka_mask",
        "cellpose_mask",
    ]

    assert preview.preview_layer_display_label(0, "GFP") == "GFP"
    assert preview.preview_layer_display_label(1, "Hmg2") == "Hmg2"
    assert preview.preview_layer_display_label(2, "Cells") == "Cells"


def test_non_composite_mode_keeps_all_enabled_layers_visible():
    preview = PreviewHarness()
    preview.preview_state.layer_visibility = [True, True, False]
    preview.preview_state.composite_mode = False
    preview.selected_preview_layer_index = lambda: 1

    assert preview.get_overlay_layer_visibility() == [True, True, False]


def test_composite_mode_uses_additive_renderer_instead_of_stacked_items():
    preview = PreviewHarness()
    preview.preview_state.tiff_model = {
        "data": np.zeros((1, 1, 2, 3, 4), dtype=np.uint8),
        "is_overlay": True,
    }
    preview.preview_state.composite_mode = True

    assert preview.should_use_preview_layer_items() is False

    preview.preview_state.composite_mode = False

    assert preview.should_use_preview_layer_items() is True


def test_preview_layer_colors_replace_duplicate_sidecar_colors():
    preview = PreviewHarness()
    model = {"data": np.zeros((1, 1, 4, 3, 4), dtype=np.uint8)}

    colors = preview.get_tiff_channel_colors(
        model,
        {"layer_colors": ["#FFFFFF", "#FFFFFF", "#00FF00", "#00FF00"]},
    )

    assert len(colors) == 4
    assert len(set(colors)) == 4
    assert colors[0] == "#FFFFFF"
    assert colors[1] == "#FF00FF"
    assert colors[2] == "#00FF00"


def test_generated_overlay_sidecar_preserves_duplicate_channel_colors():
    preview = PreviewHarness()
    model = {"data": np.zeros((1, 1, 5, 3, 4), dtype=np.uint8)}
    sidecar = {
        "type": "cellonaut_overlay_preview",
        "layer_colors": ["#00FF00", "#FF0000", "#0000FF", "#FF0000", "#0000FF"],
    }

    colors = preview.get_tiff_channel_colors(model, sidecar)

    assert colors == ["#00FF00", "#FF0000", "#0000FF", "#FF0000", "#0000FF"]


def test_overlay_preview_controls_keep_duplicate_channel_colors():
    preview = PreviewHarness()
    preview.preview_state.preserve_layer_colors = True
    preview.preview_state.overlay_colors = ["#FF0000", "#FF0000", "#0000FF", "#0000FF"]

    preview.ensure_preview_overlay_colors(4)

    assert preview.preview_state.overlay_colors == ["#FF0000", "#FF0000", "#0000FF", "#0000FF"]


def test_preview_layer_color_defaults_remain_unique_beyond_palette_length():
    preview = PreviewHarness()

    colors = preview.unique_preview_layer_colors(14, [])

    assert len(colors) == 14
    assert len(set(colors)) == 14


def test_manual_preview_color_can_match_other_layers():
    preview = PreviewHarness()
    preview.preview_state.overlay_colors = ["#FFFFFF", "#FF00FF", "#00FF00"]

    preview.set_preview_channel_color(2, "#FF00FF")

    assert preview.preview_state.overlay_colors[2] == "#FF00FF"
    assert preview.preview_state.overlay_colors == ["#FFFFFF", "#FF00FF", "#FF00FF"]
    preview.ensure_preview_overlay_colors(3)
    assert preview.preview_state.overlay_colors == ["#FFFFFF", "#FF00FF", "#FF00FF"]


def test_mask_adjustment_preview_adds_adjusted_weka_layer():
    preview = PreviewHarness()
    data = np.zeros((1, 1, 2, 6, 6), dtype=np.uint16)
    data[0, 0, 1, 2:4, 2:4] = 4095
    preview.preview_state.tiff_model = {"data": data}
    preview.preview_state.current_labels = ["Base", "Mask"]
    preview.preview_state.current_layer_roles = ["image", "weka_mask"]
    preview.preview_state.current_layer_keys = ["base", "mask"]
    preview.preview_state.overlay_colors = ["#FFFFFF", "#FF00FF"]
    preview.preview_state.overlay_opacities = [1.0, 1.0]
    preview.preview_state.normalized_page_cache = {}
    preview.preview_state.render_revision = 0
    preview.mask_adjust_target_key = lambda: "weka_layer:1"
    preview.save_current_mask_adjustment_values = lambda: None
    preview.mask_adjustment_values = lambda: {
        "target": "weka_layer:1",
        "dx": 1,
        "dy": 0,
        "grow_px": 0,
        "min_size": 0,
        "fill_holes_area": 0,
    }
    captured_states = []

    def set_layer_controls(labels, preserved_state=None):
        preview.preview_state.current_labels = list(labels)
        captured_states.append(dict(preserved_state or {}))

    preview.capture_preview_layer_control_state = lambda: {
        "visibility": {"base": True, "mask": False},
        "colors": {"base": "#FFFFFF", "mask": "#FF00FF"},
        "opacities": {"base": 1.0, "mask": 0.35},
        "order": ["mask", "base"],
    }
    preview.set_preview_layer_controls = set_layer_controls
    preview.rebuild_tiff_preview_pages = lambda: None
    preview.update_mask_adjustment_status = lambda: None

    preview.preview_current_mask_adjustments()

    updated = preview.preview_state.tiff_model["data"]
    assert updated.shape == (1, 1, 3, 6, 6)
    assert preview.preview_state.current_labels[-1] == "Mask"
    assert preview.preview_state.current_layer_roles[-1] == "adjusted_weka_mask"
    assert preview.preview_state.current_layer_keys[-1] == "weka_layer:1__adjusted_preview"
    assert preview.preview_state.overlay_colors[-1] == "#FF0000"
    assert captured_states[-1]["visibility"]["mask"] is False
    assert captured_states[-1]["order"] == ["mask", "base"]
    assert updated.dtype == np.uint16
    assert updated[0, 0, 1, 2, 2] == 4095
    assert updated[0, 0, 2, 2, 3] == 4095
    assert updated[0, 0, 2, 2, 2] == 0


def test_adjusted_mask_preview_color_prefers_unused_red_then_purple():
    preview = PreviewHarness()

    assert preview.adjusted_preview_layer_color(["#00FF00", "#0000FF"]) == "#FF0000"
    assert preview.adjusted_preview_layer_color(["#00FF00", "#FF0000"]) == "#8000FF"


def test_mask_adjustment_preview_clamps_stale_layer_metadata():
    preview = PreviewHarness()
    data = np.zeros((1, 1, 2, 6, 6), dtype=np.uint8)
    data[0, 0, 1, 2:4, 2:4] = 255
    preview.preview_state.tiff_model = {"data": data}
    preview.preview_state.current_labels = [
        "Base",
        "Mask",
        "Duplicate Base",
        "Duplicate Mask",
    ]
    preview.preview_state.current_layer_roles = [
        "image",
        "weka_mask",
        "image",
        "weka_mask",
    ]
    preview.preview_state.current_layer_keys = [
        "base",
        "mask",
        "duplicate_base",
        "duplicate_mask",
    ]
    preview.preview_state.overlay_colors = ["#FFFFFF", "#FF00FF", "#00FF00", "#FFFF00"]
    preview.preview_state.overlay_opacities = [1.0, 0.7, 1.0, 0.5]
    preview.preview_state.normalized_page_cache = {}
    preview.preview_state.render_revision = 0
    preview.mask_adjust_target_key = lambda: "weka_layer:1"
    preview.save_current_mask_adjustment_values = lambda: None
    preview.mask_adjustment_values = lambda: {
        "target": "weka_layer:1",
        "dx": 0,
        "dy": 0,
        "grow_px": 1,
        "min_size": 0,
        "fill_holes_area": 0,
    }
    preview.capture_preview_layer_control_state = lambda: {
        "visibility": {"base": True, "mask": True},
        "colors": {"base": "#FFFFFF", "mask": "#FF00FF"},
        "opacities": {"base": 1.0, "mask": 0.7},
        "order": ["base", "mask"],
    }
    preview.set_preview_layer_controls = lambda labels, preserved_state=None: setattr(preview.preview_state, "current_labels", list(labels))
    preview.rebuild_tiff_preview_pages = lambda: None
    preview.update_mask_adjustment_status = lambda: None

    preview.preview_current_mask_adjustments()

    assert preview.preview_state.tiff_model["data"].shape == (1, 1, 3, 6, 6)
    assert preview.preview_state.current_labels == ["Base", "Mask", "Mask"]
    assert preview.preview_state.current_layer_roles == [
        "image",
        "weka_mask",
        "adjusted_weka_mask",
    ]
    assert preview.preview_state.current_layer_keys == [
        "base",
        "mask",
        "weka_layer:1__adjusted_preview",
    ]


def test_preview_layer_controls_build_directly_in_preserved_order():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = PreviewHarness()
    preview.preview_layer_bar = QLabel()
    preview.preview_layer_list = PreviewLayerList()
    preview.preview_state.composite_mode = False
    preview.preview_state.current_labels = []
    preview.preview_state.current_layer_roles = ["image", "weka_mask", "weka_mask"]
    preview.preview_state.current_layer_keys = ["base", "mask_a", "mask_b"]
    preview.preview_state.overlay_colors = ["#FFFFFF", "#FF00FF", "#00FF00"]
    preview.preview_state.overlay_opacities = [1.0, 0.4, 0.8]
    preview._preview_layer_items = []
    preview.clear_preview_layer_items = lambda: None
    preview.update_mask_adjust_target_options = lambda: None
    preview.preview_color_palette = lambda: []
    preview.set_preview_color_button_style = lambda *_args, **_kwargs: None
    preview.set_preview_channel_opacity = lambda *_args, **_kwargs: None
    preview.on_preview_layer_visibility_changed = lambda *_args, **_kwargs: None
    preview.on_preview_layer_selection_changed = lambda *_args, **_kwargs: None

    preview.set_preview_layer_controls(
        ["Base", "Mask A", "Mask B"],
        preserved_state={
            "visibility": {
                "base::image::0": True,
                "mask_a::weka_mask::0": False,
                "mask_b::weka_mask::0": True,
            },
            "colors": {
                "base::image::0": "#FFFFFF",
                "mask_a::weka_mask::0": "#FF00FF",
                "mask_b::weka_mask::0": "#00FF00",
            },
            "opacities": {
                "base::image::0": 1.0,
                "mask_a::weka_mask::0": 0.4,
                "mask_b::weka_mask::0": 0.8,
            },
            "order": [
                "mask_b::weka_mask::0",
                "base::image::0",
                "mask_a::weka_mask::0",
            ],
        },
    )

    order = [
        int(preview.preview_layer_list.item(row).data(Qt.ItemDataRole.UserRole))
        for row in range(preview.preview_layer_list.count())
    ]
    assert order == [0, 2, 1]
    assert preview.preview_state.layer_order == [0, 2, 1]
    assert len(preview._preview_overlay_checks) == 3
    assert isinstance(preview._preview_overlay_checks[1], QCheckBox)
    assert preview._preview_overlay_checks[1].isChecked() is False
    group_labels = preview.preview_layer_list.findChildren(QLabel)
    group_texts = [label.text() for label in group_labels if label.property("uiRole") == "previewLayerGroup"]
    assert group_texts == ["IMAGES", "MASKS"]
    assert all(slider.minimumWidth() == slider.maximumWidth() == 84 for slider in preview._preview_opacity_sliders)
    assert all(
        slider.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
        for slider in preview._preview_opacity_sliders
    )
    assert not any(label.property("uiRole") == "previewLayerOpacityLabel" for label in group_labels)
    preview._preview_opacity_sliders[1].setValue(65)
    assert preview._preview_opacity_sliders[1].toolTip() == "Opacity: 65%"


def test_preview_layer_drag_is_limited_to_its_category():
    assert PreviewLayerList.drop_categories_match("Masks", "Masks") is True
    assert PreviewLayerList.drop_categories_match("Images", "Masks") is False
    assert PreviewLayerList.drop_categories_match("Masks", "Mask adjustments") is False


def test_cellpose_mask_defaults_below_other_masks_without_overriding_saved_order():
    roles = ["cellpose_mask", "weka_mask", "mask"]

    assert sorted(roles, key=preview_layer_default_rank) == ["weka_mask", "mask", "cellpose_mask"]
    assert sorted(
        roles,
        key=lambda role: preview_layer_default_rank(role, has_saved_order=True),
    ) == roles


def test_preview_layer_category_header_follows_first_layer_after_reorder():
    layer_list = PreviewLayerList()
    for index in range(2):
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole + 1, "Masks")
        item.setSizeHint(QSize(0, 76 if index == 0 else 60))
        row_widget = QWidget()
        row_layout = QVBoxLayout(row_widget)
        if index == 0:
            heading = QLabel("MASKS")
            heading.setProperty("uiRole", "previewLayerGroup")
            row_layout.addWidget(heading)
        row_layout.addWidget(QLabel(f"Layer {index}"))
        layer_list.addItem(item)
        layer_list.setItemWidget(item, row_widget)

    assert layer_list.model().moveRow(QModelIndex(), 0, QModelIndex(), 2) is True
    layer_list.refresh_category_headers()

    header_rows = []
    for row in range(layer_list.count()):
        widget = layer_list.itemWidget(layer_list.item(row))
        if any(label.property("uiRole") == "previewLayerGroup" for label in widget.findChildren(QLabel)):
            header_rows.append(row)
    assert header_rows == [0]


def test_clicking_selected_preview_layer_again_deselects_it():
    layer_list = PreviewLayerList()
    layer_list.resize(240, 80)
    layer_list.addItem("Mask")
    layer_list.setCurrentRow(0)
    layer_list.show()

    QTest.mouseClick(
        layer_list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=layer_list.visualItemRect(layer_list.item(0)).center(),
    )

    assert layer_list.currentRow() == -1
    assert layer_list.selectedItems() == []


def test_mask_preview_keeps_duplicate_label_layers_distinct():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = CellonautMainWindow()
    preview.preview_composite_checkbox.setChecked(False)
    labels = ["POI", "Hmg2", "BFP", "BFP", "Hmg2"]
    roles = ["image", "image", "image", "weka_mask", "weka_mask"]
    keys = ["image1", "image2", "image3", "image3", "image2"]
    data = np.zeros((1, 1, 5, 8, 8), dtype=np.uint8)
    data[0, 0, 3, 1:3, 1:3] = 255
    data[0, 0, 4, 4:6, 4:6] = 255
    preview.preview_state.tiff_model = {"data": data, "is_overlay": True}
    preview.preview_state.current_layer_roles = list(roles)
    preview.preview_state.current_layer_keys = list(keys)
    preview.preview_state.overlay_colors = ["#FFFFFF", "#FF00FF", "#00FFFF", "#00FFFF", "#FF00FF"]
    preview.preview_state.overlay_opacities = [1.0] * 5
    preview.preview_state.force_additive_composite = True
    preview.set_preview_layer_controls(labels)
    mask_row = next(
        row
        for row in range(preview.preview_layer_list.count())
        if int(preview.preview_layer_list.item(row).data(Qt.ItemDataRole.UserRole)) == 4
    )
    preview.preview_layer_list.setCurrentRow(mask_row)
    assert preview.mask_adjust_target_key() == "weka_layer:4"
    preview.mask_adjust_target_combo.setCurrentIndex(preview.mask_adjust_target_combo.findData("weka_layer:3"))
    assert int(preview.preview_layer_list.currentItem().data(Qt.ItemDataRole.UserRole)) == 3
    preview.mask_adjust_target_combo.setCurrentIndex(preview.mask_adjust_target_combo.findData("weka_layer:4"))

    preview.mask_adjust_min_size_spin.setValue(1)
    preview.preview_current_mask_adjustments()
    preview.preview_current_mask_adjustments()

    assert preview.mask_adjust_target_key() == "weka_layer:4"
    assert preview.preview_state.tiff_model["data"].shape == (1, 1, 6, 8, 8)
    assert preview.preview_state.current_labels == [
        "POI",
        "Hmg2",
        "BFP",
        "BFP",
        "Hmg2",
        "Hmg2",
    ]
    assert preview.preview_state.current_layer_roles == [
        "image",
        "image",
        "image",
        "weka_mask",
        "weka_mask",
        "adjusted_weka_mask",
    ]
    group_labels = preview.preview_layer_list.findChildren(QLabel)
    group_texts = [label.text() for label in group_labels if label.property("uiRole") == "previewLayerGroup"]
    assert group_texts[-1] == "MASK ADJUSTMENTS"
    adjusted_row = preview.preview_layer_list.count() - 1
    preview.preview_layer_list.setCurrentRow(adjusted_row)
    assert preview._preview_mask_selection_item is not None
    outline_image = preview._preview_mask_selection_item.pixmap().toImage()
    assert outline_image.pixelColor(4, 4).alpha() == 0
    assert outline_image.pixelColor(3, 4).alpha() == 255
    assert outline_image.pixelColor(2, 4).alpha() == 255
    assert outline_image.pixelColor(1, 4).alpha() == 0
    assert any(
        outline_image.pixelColor(x, y).red() > outline_image.pixelColor(x, y).blue()
        for y in range(outline_image.height())
        for x in range(outline_image.width())
        if outline_image.pixelColor(x, y).alpha() > 0
    )
    preview.preview_state.overlay_colors[-1] = "#ff0000"
    preview.on_preview_layer_selection_changed(adjusted_row)
    assert preview._preview_mask_selection_item is not None
    outline_image = preview._preview_mask_selection_item.pixmap().toImage()
    assert any(
        outline_image.pixelColor(x, y).blue() > outline_image.pixelColor(x, y).red()
        for y in range(outline_image.height())
        for x in range(outline_image.width())
        if outline_image.pixelColor(x, y).alpha() > 0
    )
    row_indices = [
        int(preview.preview_layer_list.item(row).data(Qt.ItemDataRole.UserRole))
        for row in range(preview.preview_layer_list.count())
    ]
    assert sorted(row_indices) == [0, 1, 2, 3, 4, 5]
    preview.mask_adjust_min_size_spin.setValue(0)
    assert preview.mask_adjust_reset_button.isEnabled()
    preview.preview_current_mask_adjustments()
    assert preview.preview_state.tiff_model["data"].shape == (1, 1, 5, 8, 8)
    assert not preview.mask_adjust_reset_button.isEnabled()


def test_standard_image_preview_clears_prior_tiff_overlay_state(tmp_path):
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = CellonautMainWindow()
    preview.preview_state.tiff_model = {
        "data": np.ones((1, 1, 1, 4, 4), dtype=np.uint8),
        "is_overlay": True,
    }
    preview.preview_state.force_additive_composite = True
    preview.preview_state.page_revisions = [-1]

    png_path = tmp_path / "sample_montage.png"
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(0x00FF00)
    assert image.save(str(png_path))

    preview.preview_file(str(png_path))

    assert preview.preview_state.tiff_model is None
    assert preview.preview_state.force_additive_composite is False
    assert preview.preview_state.file_path == str(png_path)
    assert preview.preview_file_title_label.text() == png_path.name
    assert "Type: standard image" in preview.preview_info_label.text()
    assert preview.preview_empty_state.isHidden() is True

    preview.reset_preview_display(clear_title=True)
    assert preview.preview_empty_state.isHidden() is False


def test_recent_images_menu_tracks_successful_previews(tmp_path):
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = CellonautMainWindow()
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(0x00FF00)
    assert image.save(str(first_path))
    assert image.save(str(second_path))

    preview.preview_file(str(first_path))
    preview.preview_file(str(second_path))
    preview.preview_file(str(first_path))
    preview.rebuild_recent_preview_images_menu()

    assert preview._recent_preview_images == [str(first_path.resolve()), str(second_path.resolve())]
    assert preview.preview_recent_images_button.isEnabled() is True
    assert [action.text() for action in preview.preview_recent_images_menu.actions()] == ["first.png", "second.png"]
    assert preview.preview_recent_images_menu.actions()[0].toolTip() == str(first_path.resolve())


def test_recent_images_persist_across_startup_state_and_are_limited(tmp_path):
    app = QApplication.instance() or QApplication([])
    _ = app
    source = CellonautMainWindow()
    paths = []
    for index in range(12):
        path = tmp_path / f"image-{index}.png"
        path.touch()
        paths.append(str(path.resolve()))
        source.record_recent_preview_image(str(path))

    saved = source.get_startup_state_dict()
    assert saved["recent_preview_images"] == list(reversed(paths))[:10]

    restored = CellonautMainWindow()
    restored.apply_startup_state_dict(
        {
            "recent_preview_images": [
                saved["recent_preview_images"][0],
                saved["recent_preview_images"][0],
                str(tmp_path / "missing.png"),
                *saved["recent_preview_images"][1:],
                123,
            ]
        }
    )

    assert restored._recent_preview_images == saved["recent_preview_images"]
    assert restored.preview_recent_images_button.isEnabled() is True


def test_recent_images_round_trip_through_last_settings_file(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    _ = app
    settings_path = tmp_path / "last_settings.json"
    image_path = tmp_path / "remembered.png"
    image_path.touch()
    monkeypatch.setattr(config_gui, "SETTINGS_FILE", settings_path)

    source = CellonautMainWindow()
    source.record_recent_preview_image(str(image_path))
    source.save_last_settings()

    restored = CellonautMainWindow()
    restored._recent_preview_images = []
    restored.preview_recent_images_button.setEnabled(False)
    restored.load_last_settings_if_available()

    assert restored._recent_preview_images == [str(image_path.resolve())]
    assert restored.preview_recent_images_button.isEnabled() is True


def test_tiff_preview_releases_previous_state_before_reading(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = CellonautMainWindow()
    preview.preview_state.tiff_model = {"data": np.ones((1, 1, 1, 4, 4), dtype=np.uint8)}
    preview.preview_state.pages = [QPixmap(4, 4)]

    class FailingTiffFile:
        def __init__(self, _path):
            assert preview.preview_state.tiff_model is None
            assert preview.preview_state.pages == []
            raise OSError("read failed")

    monkeypatch.setattr("cellonaut.gui.preview.tifffile.TiffFile", FailingTiffFile)

    with pytest.raises(ValueError, match="read failed"):
        preview.preview_tiff_image(str(tmp_path / "sample.tif"))


def test_tiff_axes_metadata_preserves_cyx_channel_order():
    preview = PreviewHarness()
    stack = np.zeros((4, 5, 6), dtype=np.uint16)
    for channel_index in range(stack.shape[0]):
        stack[channel_index] = channel_index + 10

    model = preview.normalize_tiff_to_tzcyx(
        stack,
        sidecar={"axes": "CYX", "layer_labels": ["DIA", "GFP", "TxRed", "DAPI"]},
    )

    assert model["data"].shape == (1, 1, 4, 5, 6)
    assert model["metadata_channel_axis"] is True
    assert np.array_equal(model["data"][0, 0, 0], stack[0])
    assert np.array_equal(model["data"][0, 0, 2], stack[2])


def test_tiff_axes_metadata_preserves_yxc_channel_order():
    preview = PreviewHarness()
    stack = np.zeros((5, 6, 4), dtype=np.uint16)
    for channel_index in range(stack.shape[-1]):
        stack[..., channel_index] = channel_index + 20

    model = preview.normalize_tiff_to_tzcyx(
        stack,
        sidecar={"axes": "YXC", "layer_labels": ["DIA", "GFP", "TxRed", "DAPI"]},
    )

    assert model["data"].shape == (1, 1, 4, 5, 6)
    assert np.array_equal(model["data"][0, 0, 0], stack[..., 0])
    assert np.array_equal(model["data"][0, 0, 3], stack[..., 3])


def test_ome_tiff_channel_stack_uses_layer_rendering_not_rgb(tmp_path):
    app = QApplication.instance() or QApplication([])
    _ = app
    path = tmp_path / "converted.ome.tif"
    stack = np.zeros((4, 5, 6), dtype=np.uint16)
    stack[0] = 100
    stack[1] = 200
    stack[2] = 300
    stack[3] = 400
    tifffile.imwrite(
        path,
        stack,
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "Channel": {
                "Name": ["DIA", "GFP", "TxRed", "DAPI"],
                "Color": [0xFFFFFF, 0x00FF00, 0xFF0000, 0x0000FF],
            },
        },
    )

    preview = CellonautMainWindow()
    preview.preview_tiff_image(str(path))

    assert preview.preview_state.force_additive_composite is True
    assert preview.preview_state.current_labels == ["DIA", "GFP", "TxRed", "DAPI"]
    assert preview.preview_state.tiff_model is not None
    assert np.array_equal(preview.preview_state.tiff_model["data"][0, 0, 0], stack[0])
    assert np.array_equal(preview.preview_state.tiff_model["data"][0, 0, 2], stack[2])


def test_nd2_channel_name_colors_match_exported_tiff_colors():
    preview = PreviewHarness()

    colors = preview.inferred_channel_colors_from_labels(["DIA", "GFP", "TxRed", "DAPI"])

    assert colors == ["#FFFFFF", "#00FF00", "#FF0000", "#0000FF"]


def test_first_nonwhite_overlay_layer_renders_in_channel_color():
    app = QApplication.instance() or QApplication([])
    _ = app
    preview = PreviewHarness()
    page = np.full((1, 3, 3), 255, dtype=np.uint8)

    pixmap = preview.make_tiff_page_pixmap(
        page,
        labels=["GFP"],
        colors=["#00FF00"],
        visibility=[True],
        opacities=[1.0],
        already_normalized=True,
    )
    pixel = pixmap.toImage().pixelColor(1, 1)

    assert pixel.green() > pixel.red()
    assert pixel.green() > pixel.blue()
