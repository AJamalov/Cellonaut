"""Preview ownership through the real window, loaders, controls and navigation."""

import json

import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QSlider

from cellonaut.gui.main_window import CellonautMainWindow
from cellonaut.gui.state import PreviewState
from cellonaut.masks.preview_filter_overlay import PreviewFilterData


pytestmark = pytest.mark.gui


@pytest.fixture
def window(monkeypatch, qt_application):
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "save_last_settings", lambda self: None)
    build_ui = CellonautMainWindow.build_ui
    states_at_build = []

    def build(self):
        assert isinstance(self.preview_state, PreviewState)
        states_at_build.append(self.preview_state)
        build_ui(self)

    monkeypatch.setattr(CellonautMainWindow, "build_ui", build)
    widget = CellonautMainWindow()
    assert states_at_build == [widget.preview_state]
    yield widget
    widget.close()
    widget.deleteLater()
    qt_application.processEvents()


def write_overlay(path, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((2, 2, 8, 8), value, dtype=np.uint16)
    pixels[:, 1] = 0
    pixels[:, 1, 2:6, 2:6] = 1
    tifffile.imwrite(path, pixels, metadata={"axes": "ZCYX"}, photometric="minisblack")
    path.with_suffix(".json").write_text(json.dumps({
        "axes": "ZCYX", "layer_labels": ["GFP", "Mask"], "layer_roles": ["image", "weka_mask"],
        "layer_keys": ["image1", "image2"], "layer_colors": ["#ffffff", "#ff0000"],
    }), encoding="utf-8")
    return path


def test_preview_state_exists_before_widget_construction(window):
    # Construction connects signals that immediately query these fields.
    state = window.preview_state
    assert isinstance(state, PreviewState)
    assert state.file_path is None and state.tiff_model is None
    assert state.pages == [] and state.page_index == 0
    assert state.layer_visibility == [] and state.cell_group_result is None
    assert state.artifact_index == state.artifact_file_index == -1
    assert not hasattr(CellonautMainWindow, "_get_preview_state")
    for alias in ("_preview_file_path", "_preview_pages", "_preview_tiff_model",
                  "_preview_current_labels", "_preview_layer_order", "_preview_group_layer_display_state"):
        assert not hasattr(window, alias)
    other = PreviewState()
    state.layer_visibility.append(False)
    assert other.layer_visibility == []


def test_loading_and_controls_use_one_model(window, tmp_path):
    path = write_overlay(tmp_path / "sample.tif")
    state = window.preview_state
    assert window.preview_file(str(path)) is True
    assert window.preview_state is state
    assert state.file_path == str(path) and state.current_labels == ["GFP", "Mask"]
    assert state.tiff_model["data"].shape == (1, 2, 2, 8, 8)
    assert state.layer_visibility == [True, True]
    window._preview_overlay_checks[1].setChecked(False)
    window._preview_opacity_sliders[1].setValue(35)
    assert state.layer_visibility == [True, False]
    assert state.overlay_opacities == [1.0, 0.35]
    assert window.capture_preview_layer_control_state()["visibility"][window.preview_layer_identity(1)] is False
    window.preview_composite_checkbox.setChecked(True)
    assert state.composite_mode is True
    window.show_preview_page(1)
    assert state.page_index == 1
    window.preview_layer_list.setCurrentRow(1)
    assert state.selected_layer_index == 1
    assert state.mask_adjust_target == "weka_layer:1"
    # Rendering reads model values even if a widget is temporarily stale during rebuilding.
    window._preview_overlay_checks[1].blockSignals(True)
    window._preview_overlay_checks[1].setChecked(True)
    assert window.get_overlay_layer_visibility() == [True, False]
    window._preview_overlay_checks[1].blockSignals(False)


def test_sample_channel_navigation_and_unrelated_dataset_clear_selection(window, tmp_path):
    first = write_overlay(tmp_path / "old" / "a_GFP.tif")
    second = write_overlay(tmp_path / "old" / "a_RFP.tif", 200)
    third = write_overlay(tmp_path / "old" / "b_GFP.tif", 300)
    unrelated = write_overlay(tmp_path / "new" / "other.tif", 400)
    state = window.preview_state
    state.artifact_mode = "overlay"
    state.artifact_results_root = first.parent
    state.artifact_entries = [{"sample": "a", "files": [first, second]}, {"sample": "b", "files": [third]}]
    window.open_preview_artifact_entry(0)
    window.step_preview_artifact_file(1)
    assert state.file_path == str(second) and state.artifact_index == 0 and state.artifact_file_index == 1
    window.step_preview_artifact_sample(1)
    assert state.file_path == str(third) and state.artifact_index == 1 and state.artifact_file_index == 0
    # Opening from Files also updates the navigation model and its counters.
    assert window.preview_file(str(second)) is True
    assert state.artifact_index == 0 and state.artifact_file_index == 1
    assert window.preview_sample_nav_label.text() == "Sample: 1 / 2"
    assert window.preview_artifact_nav_label.text() == "Overlay: 2 / 2"
    state.mask_adjustments_by_target = {"weka_layer:1": {"dx": 10}}
    assert window.preview_file(str(unrelated)) is True
    assert state.file_path == str(unrelated)
    assert state.artifact_entries == [] and state.artifact_results_root is None
    assert state.artifact_index == state.artifact_file_index == -1
    assert state.mask_adjustments_by_target["weka_layer:1"]["dx"] == 0
    assert not window.preview_next_sample_button.isEnabled()
    assert window.preview_artifact_selector.currentIndex() == -1


def test_group_membership_and_display_state_survive_rebuild(window, tmp_path, monkeypatch):
    path = write_overlay(tmp_path / "sample.tif")
    labels = np.ones((8, 8), dtype=np.int32)
    labels[4:] = 2
    definition = {"name": "GFP", "cell_populations": [
        {"name": "Large", "cell_qc_limits": "Area:10-", "exclude_from_csv": True},
    ]}
    monkeypatch.setattr(window, "get_active_image_definitions", lambda: [definition])
    monkeypatch.setattr(window, "commit_gui_edits", lambda: None)
    monkeypatch.setattr("cellonaut.gui.preview_filter.find_preview_filter_data", lambda *_args: (
        PreviewFilterData(result_id="sample", source_label="GFP", table_path=path.with_suffix(".csv"),
                          labels_path=path, table=pd.DataFrame({"CellID": [1, 2], "Area": [8, 20]}),
                          label_image=labels), ""
    ))
    assert window.preview_file(str(path)) is True
    state = window.preview_state
    assert state.cell_group_result.groups[0].labels == {2}
    assert state.cell_group_result.excluded_labels == {2}
    row = next(window.preview_layer_list.item(i) for i in range(window.preview_layer_list.count())
               if window.preview_layer_list.item(i).data(Qt.ItemDataRole.UserRole + 2) == "cell_group")
    control = window.preview_layer_list.itemWidget(row)
    control.findChild(QCheckBox).setChecked(False)
    control.findChild(QSlider).setValue(25)
    assert state.group_layer_display_state["0"]["visible"] is False
    assert state.group_layer_display_state["0"]["opacity"] == 25
    window.refresh_preview_filter_overlay()
    assert state.cell_group_result.groups[0].labels == {2}
    assert not window._preview_population_items["0"].isVisible()
    assert window._preview_population_items["0"].opacity() == 0.25


@pytest.mark.parametrize("action", ["reset", "failed_load", "close"])
def test_clearing_preview_releases_loaded_state(window, tmp_path, action):
    path = write_overlay(tmp_path / "sample.tif")
    assert window.preview_file(str(path)) is True
    state = window.preview_state
    state.artifact_entries = [{"sample": "a", "files": [path]}]
    state.artifact_results_root = path.parent
    window._preview_overlay_checks[1].setChecked(False)
    window.preview_composite_checkbox.setChecked(True)  # Schedule a deferred render.
    if action == "reset":
        window.reset_preview_display(clear_title=True)
    elif action == "failed_load":
        broken = tmp_path / "broken.tif"
        broken.write_bytes(b"not a TIFF")
        assert window.preview_file(str(broken)) is False
    else:
        window.close()
    assert state.file_path is None and state.tiff_model is None and state.filter_data is None
    assert state.cell_group_result is None and state.pages == [] and state.page_index == 0
    assert state.current_labels == state.current_layer_keys == state.current_layer_roles == []
    assert state.layer_visibility == state.overlay_opacities == state.overlay_colors == []
    assert state.artifact_entries == [] and state.artifact_results_root is None
    assert state.artifact_index == state.artifact_file_index == -1
    assert window.preview_artifact_selector.currentIndex() == -1
    assert state.mask_adjustments_by_target == {} and state.group_layer_display_state == {}
    assert window.preview_scene.items() == []
    assert not window._preview_render_timer.isActive()
