"""Configuration ownership exercised through the real composed GUI."""

from copy import deepcopy
import json

import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QPushButton

from cellonaut.config.adapter import build_pipeline_config_from_gui_state
from cellonaut.config.state import EditableGuiConfiguration
from cellonaut.gui.main_window import CellonautMainWindow


pytestmark = pytest.mark.gui


@pytest.fixture
def window(monkeypatch, qt_application, tmp_path):
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "schedule_input_path_scan", lambda *args, **kwargs: None)
    monkeypatch.setattr(CellonautMainWindow, "save_last_settings", lambda self: None)
    widget = CellonautMainWindow()
    model = tmp_path / "mask.model"
    model.write_text("fixture", encoding="utf-8")
    widget.apply_preset_dict({
        "input_dir": str(tmp_path / "input"),
        "output_dir": str(tmp_path / "output"),
        "image_definitions": [
            {"name": "A", "mask_slot_enabled": False,
             "analysis_cell_segmentation_enabled": True,
             "analysis_cell_segmentation_source": "B", "mask_relationships": {"Mask": True}},
            {"name": "B", "mask_slot_enabled": False,
             "analysis_cell_segmentation_enabled": True},
            {"name": "Mask", "is_mask_only": True, "mask_source_channel": "A", "classifier": str(model)},
        ],
    }, schedule_scan=False)
    yield widget
    widget.close()
    widget.deleteLater()
    qt_application.processEvents()


def pending_text(widget, text):
    # Simulate an edit that has not emitted its commit signal yet.
    with QSignalBlocker(widget):
        widget.setText(text)


def test_commit_captures_each_editor_once_and_collection_is_read_only(window, monkeypatch):
    state = window.configuration_state
    assert isinstance(state, EditableGuiConfiguration)
    calls = []
    methods = (
        "_read_image_row_edits", "_read_processing_table_edits", "_read_cellpose_table_edits",
        "_read_analysis_matrix_edits", "_read_inline_filter_edits", "_read_measurement_edits",
        "normalize_image_definitions",
    )
    for name in methods:
        original = getattr(window, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(window, name, record)
    pending_text(window.cellpose_settings_table.cellWidget(0, 5), "30")
    assert state.image_definitions[0]["cell_diameter"] == ""
    assert window.collect_gui_state().image_definitions[0].cell_diameter == ""
    window.commit_gui_edits()
    assert state.image_definitions[0]["cell_diameter"] == "30"
    assert sorted(calls) == sorted(methods)
    before = deepcopy(state)
    calls.clear()
    first = window.collect_gui_state()
    for _ in range(3):
        assert window.collect_gui_state() == first
        assert window.get_active_image_definitions() == state.image_definitions
        assert window.get_preset_dict()["image_definitions"] == state.image_definitions
    assert state == before and calls == []
    first.image_definitions[0].mask_relationships.clear()
    assert state.image_definitions[0]["mask_relationships"]["Mask"] is True


def test_pending_editors_are_captured_by_immediate_preset_save(window, tmp_path):
    window.add_image_processing_step("rolling_ball_background")
    pending_text(window.image_processing_table.cellWidget(0, 2), "17")
    pending_text(window.image_processing_table.cellWidget(0, 3), "23")
    pending_text(window.cellpose_settings_table.cellWidget(0, 5), "30")
    pending_text(window.cellpose_settings_table.cellWidget(1, 5), "")
    checks = {str(cb.property("measurementKey")): cb for cb in window._inline_measurement_checks}
    with QSignalBlocker(checks["cell_area"]):
        checks["cell_area"].setChecked(True)
    with QSignalBlocker(window.analysis_matrix_table.cellWidget(1, 0)):
        window.analysis_matrix_table.cellWidget(1, 0).setChecked(True)
    pending_text(window.output_dir.edit, str(tmp_path / "latest-output"))

    path = tmp_path / "saved.json"
    window._save_preset_to_path(path, "Saved", created=True, notify=False, refresh=False)
    saved = json.loads(path.read_text(encoding="utf-8"))
    a, b, _mask = saved["image_definitions"]
    assert a["cell_diameter"] == "30" and b["cell_diameter"] == ""
    assert a["image_processing_steps"][0]["params"]["bg_radii"] == "17"
    assert b["image_processing_steps"][0]["params"]["bg_radii"] == "23"
    assert b["mask_relationships"]["Mask"] is True
    assert saved["measurement_options"]["cell_area"] is True
    assert saved["output_dir"] == str(tmp_path / "latest-output")


def test_cellpose_only_channel_is_visible_and_runnable_from_measurement_matrix(window):
    window.apply_preset_dict(
        {
            "image_definitions": [
                {
                    "name": "Cells",
                    "folder": "Cells",
                    "mask_slot_enabled": False,
                    "analysis_cell_segmentation_enabled": True,
                }
            ]
        },
        schedule_scan=False,
    )

    matrix = window.analysis_matrix_table
    assert matrix.rowCount() == 1
    assert matrix.columnCount() == 1
    assert matrix.horizontalHeaderItem(0).text() == "Cells_Cellpose"
    cellpose_toggle = matrix.cellWidget(0, 0)
    assert cellpose_toggle.isChecked() is True

    config = build_pipeline_config_from_gui_state(window.collect_gui_state())
    assert len(config.measurement_targets) == 1
    assert config.measurement_targets[0].do_cell_segmentation is True
    assert config.measurement_targets[0].overlay_roi_keys == []

    cellpose_toggle.click()
    settings_enable = window.cellpose_settings_table.cellWidget(0, 0).findChild(QPushButton)
    assert settings_enable.isChecked() is True
    assert window.get_active_image_definitions()[0]["analysis_cellpose_mask_source"] == ""

    cellpose_toggle.click()
    assert settings_enable.isChecked() is True
    assert window.get_active_image_definitions()[0]["analysis_cellpose_mask_source"] == "Cells"

    settings_enable.click()
    assert window.analysis_matrix_table.columnCount() == 0
    assert window.get_active_image_definitions()[0]["analysis_cell_segmentation_enabled"] is False


def test_one_cellpose_mask_can_measure_multiple_channel_rows(window):
    window.apply_preset_dict(
        {
            "image_definitions": [
                {
                    "name": name,
                    "folder": name,
                    "mask_slot_enabled": False,
                    "analysis_cell_segmentation_enabled": index == 0,
                    "cell_diameter": "41" if index == 0 else "20",
                }
                for index, name in enumerate(("Channel1", "Channel2", "Channel3", "Channel4"))
            ]
        },
        schedule_scan=False,
    )

    matrix = window.analysis_matrix_table
    assert matrix.rowCount() == 4
    assert matrix.columnCount() == 1
    assert matrix.horizontalHeaderItem(0).text() == "Channel1_Cellpose"
    assert matrix.cellWidget(0, 0).isChecked() is True

    for row in range(1, 4):
        matrix.cellWidget(row, 0).click()

    definitions = window.get_active_image_definitions()
    assert [item["analysis_cellpose_mask_source"] for item in definitions] == ["Channel1"] * 4
    config = build_pipeline_config_from_gui_state(window.collect_gui_state())
    assert len(config.measurement_targets) == 4
    assert {target.cell_segmentation_mask_source for target in config.measurement_targets} == {"image1"}
    assert {target.cell_segmentation_source for target in config.measurement_targets} == {"image1"}
    assert {target.cell_diameter for target in config.measurement_targets} == {41.0}


def test_each_enabled_cellpose_mask_can_be_selected_for_the_same_measured_channel(window):
    window.apply_preset_dict(
        {
            "image_definitions": [
                {
                    "name": name,
                    "folder": name,
                    "mask_slot_enabled": False,
                    "analysis_cell_segmentation_enabled": index < 2,
                }
                for index, name in enumerate(("Channel1", "Channel2", "Signal"))
            ]
        },
        schedule_scan=False,
    )

    matrix = window.analysis_matrix_table
    assert [matrix.horizontalHeaderItem(column).text() for column in range(2)] == [
        "Channel1_Cellpose",
        "Channel2_Cellpose",
    ]
    matrix.cellWidget(2, 0).click()
    assert matrix.cellWidget(2, 0).isChecked() is True
    matrix.cellWidget(2, 1).click()
    assert matrix.cellWidget(2, 0).isChecked() is True
    assert matrix.cellWidget(2, 1).isChecked() is True
    signal_definition = window.get_active_image_definitions()[2]
    assert signal_definition["analysis_cellpose_mask_sources"] == ["Channel1", "Channel2"]
    config = build_pipeline_config_from_gui_state(window.collect_gui_state())
    signal_targets = [target for target in config.measurement_targets if target.source_image_key == "image3"]
    assert [target.cell_segmentation_mask_source for target in signal_targets] == ["image1", "image2"]
    assert [target.output_variant for target in signal_targets] == [
        "Channel1_Cellpose",
        "Channel2_Cellpose",
    ]


@pytest.mark.parametrize("index,new_name", [(0, "Renamed A"), (1, "Renamed B"), (2, "Renamed mask")])
def test_rename_commits_and_remaps_even_before_editing_finished(window, index, new_name, tmp_path):
    # The pending name may be committed by Save instead of editingFinished.
    pending_text(window.image_rows[index].name.edit, new_name)
    window._save_preset_to_path(tmp_path / "renamed.json", "Renamed", created=True, notify=False, refresh=False)
    saved = window.get_preset_dict()
    definitions = saved["image_definitions"]
    assert definitions[index]["name"] == new_name
    assert definitions[2]["mask_source_channel"] == (new_name if index == 0 else "A")
    assert definitions[0]["analysis_cell_segmentation_source"] == (new_name if index == 1 else "B")
    assert definitions[0]["analysis_cellpose_mask_source"] == (new_name if index == 0 else "A")
    target = new_name if index == 2 else "Mask"
    assert definitions[0]["mask_relationships"][target] is True
    assert definitions[0]["cell_group_mask_source"] == target
    # Selectors have also been rendered from the committed data: repeated saves
    # cannot write their pre-rename text back into relationships.
    window.commit_gui_edits()
    assert window.get_preset_dict() == saved


def test_preset_load_renders_owned_state_without_scraping_old_widgets(window, monkeypatch):
    pending_text(window.cellpose_settings_table.cellWidget(0, 5), "99")
    saved = {
        "input_dir": "another-dataset", "measurement_options": {"cell_area": True},
        "image_definitions": [{"name": "Loaded", "mask_slot_enabled": False, "cell_diameter": "12"}],
    }
    calls = []
    monkeypatch.setattr(window, "_read_cellpose_table_edits", lambda *_: calls.append("read"))
    window.apply_preset_dict(saved, schedule_scan=False)
    assert calls == []
    assert window.configuration_state.image_definitions[0]["cell_diameter"] == "12"
    assert window.cellpose_settings_table.cellWidget(0, 5).text() == "12"
    assert window.input_dir.get() == window.configuration_state.input_dir == "another-dataset"
    assert window.configuration_state.measurement_options["cell_area"] is True
    assert window.image_rows[0].name.get() == "Loaded"
    checks = {str(cb.property("measurementKey")): cb for cb in window._inline_measurement_checks}
    assert checks["cell_area"].isChecked() is True
    assert window.get_preset_dict()["measurement_options"]["cell_area"] is True


def test_legacy_preset_migration_is_at_load_boundary(window):
    window.apply_preset_dict({"image_definitions": [{
        "name": "Legacy", "classifier": "legacy.model", "cell_qc_limits": "Area:5-12",
        "cell_qc_exclude_flagged": True, "mask_relationships": {"Legacy": True},
    }]}, schedule_scan=False)
    state = window.configuration_state
    assert len(state.image_definitions) == 2  # Old channel mask slot becomes a mask.
    assert state.image_definitions[0]["cell_populations"][0]["cell_qc_limits"] == "Area:5-12"
    assert state.image_definitions[0]["mask_relationships"]["Legacy mask"] is True
    saved = window.get_preset_dict()
    assert "cell_qc_limits" not in saved["image_definitions"][0]
    window.apply_preset_dict(saved, schedule_scan=False)
    assert window.get_preset_dict() == saved


def test_execution_uses_detached_committed_snapshot(window):
    pending_text(window.cellpose_settings_table.cellWidget(0, 5), "30")
    window.commit_gui_edits()
    snapshot = window.collect_gui_state()
    pending_text(window.cellpose_settings_table.cellWidget(0, 5), "99")
    config = build_pipeline_config_from_gui_state(snapshot)
    assert config.measurement_targets[0].cell_diameter == 30
    assert config.measurement_targets[1].cell_diameter is None
    # The execution entry point commits pending widgets once itself.
    latest = window.collect_config()
    assert latest.measurement_targets[0].cell_diameter == 99
    assert config.measurement_targets[0].cell_diameter == 30


def test_cell_group_edits_and_invalid_partial_input_preserve_consistent_state(window):
    window.show_analysis_settings_panel(0, "filters")
    edit = window._inline_cell_filter_rows["Area"]["min_edit"]
    pending_text(edit, "5")
    window.commit_gui_edits()
    saved = window.get_preset_dict()
    assert saved["image_definitions"][0]["cell_populations"][0]["cell_qc_limits"] == "Area:5-"
    pending_text(edit, "-")
    window.commit_gui_edits()
    assert window.get_active_image_definitions()[0]["cell_populations"] == saved["image_definitions"][0]["cell_populations"]
    assert "Area" in window.analysis_filters_hint.text()
    pending_text(edit, "")
    window.commit_gui_edits()
    assert window.get_active_image_definitions()[0]["cell_populations"][0]["cell_qc_limits"] == ""


def test_configuration_lifecycle_keeps_preview_ownership_separate(window):
    preview = window.preview_state
    preview.mask_adjustments_by_target["preview-only"] = {"dx": 3}
    window.add_image_row()
    assert len(window.configuration_state.image_definitions) == 4
    window.remove_image_row_at(3)
    assert len(window.configuration_state.image_definitions) == 3
    window.apply_preset_dict({}, schedule_scan=False)
    window.commit_gui_edits()
    assert window.preview_state is preview
    assert preview.mask_adjustments_by_target["preview-only"] == {"dx": 3}
    assert not hasattr(preview, "image_definitions")
    assert not hasattr(window.configuration_state, "preview_state")


def test_preset_replacement_refreshes_open_cell_group_editor(window):
    window.show_analysis_settings_panel(0, "filters")
    pending_text(window._inline_cell_filter_rows["Area"]["min_edit"], "99")
    preset = window.get_preset_dict()
    preset["image_definitions"][0]["cell_populations"][0]["cell_qc_limits"] = "Area:7-"
    window.apply_preset_dict(preset, schedule_scan=False)
    assert window._inline_cell_filter_rows["Area"]["min_edit"].text() == "7"
    window.commit_gui_edits()
    assert window.get_preset_dict()["image_definitions"][0]["cell_populations"][0]["cell_qc_limits"] == "Area:7-"


def test_commit_during_table_construction_preserves_configuration(window):
    window.add_image_processing_step("rolling_ball_background")
    window.add_mask_processing_step("binary_fill_holes")
    window.image_processing_table.cellWidget(0, 2).setText("17")
    window.commit_gui_edits()
    before = window.collect_gui_state()
    window._rebuilding_image_tabs = True
    try:
        window.cellpose_settings_table.setRowCount(0)
        window.analysis_matrix_table.setRowCount(0)
        window.image_processing_table.setRowCount(0)
        window.mask_processing_table.setRowCount(0)
        window.commit_gui_edits()
        assert window.collect_gui_state() == before
    finally:
        window._rebuilding_image_tabs = False
    # Empty editors also preserve values until they have actually been built.
    window.commit_gui_edits()
    assert window.collect_gui_state() == before
    window.rebuild_image_rows(sync_from_ui=False)
    assert window.image_processing_table.cellWidget(0, 2).text() == "17"


def test_editing_and_saving_are_allowed_during_scan_but_execution_waits(window, tmp_path):
    window._pending_input_scan_path = window.input_dir.get()
    window._last_input_scan_result = {}
    pending_text(window.cellpose_settings_table.cellWidget(0, 5), "30")
    window._save_preset_to_path(tmp_path / "pending.json", "Pending", created=True, notify=False, refresh=False)
    assert window.configuration_state.image_definitions[0]["cell_diameter"] == "30"
    with pytest.raises(ValueError, match="still being scanned"):
        window.collect_config()


def test_dirty_refresh_does_not_replace_name_editor_during_typing(window, qt_application):
    window.show()
    window.show_pipeline_section(window.pipeline_sections.index(window.project_section))
    edit = window.image_rows[0].name.edit
    edit.setFocus()
    qt_application.processEvents()
    assert edit.hasFocus()
    pending_text(edit, "New name")
    window.refresh_preset_dirty_indicator()
    assert window.image_rows[0].name.edit is edit
    assert edit.hasFocus()
    assert window._preset_dirty is True
    edit.editingFinished.emit()
    assert window.configuration_state.image_definitions[0]["name"] == "New name"


def test_removing_selected_group_renders_survivor_before_next_commit(window):
    window.show_analysis_settings_panel(0, "filters")
    window._inline_cell_filter_rows["Area"]["min_edit"].setText("5")
    original = deepcopy(window.image_definitions[0]["cell_populations"][0])
    window.add_analysis_population()
    window._inline_cell_filter_rows["Area"]["min_edit"].setText("99")
    window.remove_analysis_population()
    window.commit_gui_edits()
    assert window.image_definitions[0]["cell_populations"] == [original]
    assert window._inline_cell_filter_rows["Area"]["min_edit"].text() == "5"


def test_switching_group_commits_pending_edits_to_previous_owner(window):
    window.show_analysis_settings_panel(0, "filters")
    window.add_analysis_population()
    pending_text(window._inline_cell_filter_rows["Area"]["min_edit"], "99")
    pending_text(window.analysis_population_name_edit, "Second group")
    window.analysis_population_combo.setCurrentIndex(0)
    window.commit_gui_edits()
    groups = window.image_definitions[0]["cell_populations"]
    assert groups[0]["cell_qc_limits"] == ""
    assert groups[1]["cell_qc_limits"] == "Area:99-"
    assert groups[1]["name"] == "Second group"
