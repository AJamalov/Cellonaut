# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QMenu,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QToolButton,
)

from cellonaut.config.defaults import (
    MASK_SOURCE_MODE_COMBINED,
    default_measurement_options,
)
from cellonaut.gui.dataset import CellonautGuiDatasetMixin
from cellonaut.gui.image_processing_settings import BitDepthSettingsCell, OutlierSettingsCell
from cellonaut.gui.mask_processing_settings import FijiBinarySettingsCell, FijiParticleSettingsCell, FijiTranslateSettingsCell
from cellonaut.gui.widgets import MatrixToggleButton
from cellonaut.gui.validation import CellonautGuiValidationMixin


pytestmark = pytest.mark.gui


class DummyRoiSettingsGui(CellonautGuiDatasetMixin, CellonautGuiValidationMixin):
    def __init__(self):
        self._rebuilding_image_tabs = False
        self.measurement_options = default_measurement_options()
        self.image_definitions = []
        self.image_rows = []
        self.image_processing_table = QTableWidget()
        self.measurement_processing_table = QTableWidget()
        self.mask_processing_table = QTableWidget()


class MatrixChangeHarness(CellonautGuiDatasetMixin):
    def __init__(self):
        self._rebuilding_image_tabs = False
        self.calls = []

    def commit_gui_edits(self, *, require_ready_input=False):
        self.calls.append("save")

    def refresh_analysis_matrix_row_states(self):
        self.calls.append("refresh")

    def update_analysis_matrix_warning_label(self):
        self.calls.append("warnings")

    def build_analysis_matrix_for_current_source(self):
        self.calls.append("rebuild")


def test_export_all_filtered_csvs_reports_settings_sync_failure(monkeypatch):
    app = QApplication.instance() or QApplication([])
    _ = app
    gui = DummyRoiSettingsGui()
    messages = []

    def fail_sync():
        raise RuntimeError("settings sync failed")

    monkeypatch.setattr(gui, "commit_gui_edits", fail_sync)
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )

    gui.export_all_filter_preview_csvs()

    assert messages == [
        (
            "Export filtered CSV for all samples in the pipeline",
            "Could not prepare the current cell-group settings:\nRuntimeError: settings sync failed",
        )
    ]


def test_normalization_repairs_mask_assignment_and_source_metadata():
    app = QApplication.instance() or QApplication([])
    _ = app
    gui = DummyRoiSettingsGui()

    definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "signal_folder",
                "display_color": "#00ff00",
                "stack_channel_index": "3",
                "mask_slot_enabled": False,
            },
            {
                "name": "Signal mask",
                "folder": "stale_folder",
                "is_mask_only": True,
                "mask_source_channel": "Missing channel",
                "mask_slot_enabled": True,
            },
        ]
    )

    mask_definition = next(image_def for image_def in definitions if image_def["name"] == "Signal mask")
    assert mask_definition["mask_source_channel"] == "Signal"
    assert mask_definition["folder"] == "signal_folder"
    assert mask_definition["display_color"] == "#00ff00"
    assert mask_definition["stack_channel_index"] == "3"


def test_removing_channel_prunes_references_and_reassigns_its_masks():
    app = QApplication.instance() or QApplication([])
    _ = app
    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "signal_folder",
                "mask_slot_enabled": False,
            },
            {
                "name": "Reference",
                "folder": "reference_folder",
                "stack_channel_index": "2",
                "mask_slot_enabled": False,
                "mask_relationships": {"Signal": True, "Signal mask": True},
            },
            {
                "name": "Signal mask",
                "folder": "signal_folder",
                "is_mask_only": True,
                "mask_source_channel": "Signal",
                "mask_slot_enabled": True,
            },
        ]
    )

    gui.remove_image_row_at(0)

    assert [image_def["name"] for image_def in gui.image_definitions] == ["Reference", "Signal mask"]
    mask_definition = gui.image_definitions[1]
    assert mask_definition["mask_source_channel"] == "Reference"
    assert mask_definition["folder"] == "reference_folder"
    assert mask_definition["stack_channel_index"] == "2"
    for image_def in gui.image_definitions:
        assert "Signal" not in image_def["mask_relationships"]


def processing_scope_button(widget, scope):
    return next(
        button
        for button in widget.findChildren(QPushButton)
        if button.property("processingRole") == "scope" and button.property("scopeValue") == scope
    )


def processing_scope_button_texts(widget):
    return [
        button.text()
        for button in widget.findChildren(QPushButton)
        if button.property("processingRole") == "scope" and button.isEnabled()
    ]


def processing_step_combo(widget):
    return next(
        button for button in widget.findChildren(QToolButton) if button.property("processingRole") == "step_type"
    )


def menu_action_by_data(menu, data):
    for action in menu.actions():
        if str(action.data() or "") == data:
            return action
        submenu = action.menu()
        if submenu is not None:
            found = menu_action_by_data(submenu, data)
            if found is not None:
                return found
    return None


def menu_has_path(menu, labels):
    if not labels:
        return True
    head, *tail = labels
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None and action.text() == head:
            return menu_has_path(submenu, tail)
    return False


def trigger_processing_step(button, step_type):
    action = menu_action_by_data(button.menu(), step_type)
    assert action is not None
    action.trigger()


def processing_enabled_checkbox(widget):
    button = next(
        (button for button in widget.findChildren(QPushButton) if button.property("processingRole") == "enabled"),
        None,
    )
    if button is not None:
        return button
    return next(
        checkbox for checkbox in widget.findChildren(QCheckBox) if checkbox.property("processingRole") == "enabled"
    )


def processing_warning_label(widget):
    return next(label for label in widget.findChildren(QLabel) if label.property("processingRole") == "warning")


def processing_step_cell(table, step_index):
    return table.cellWidget(step_index, 0)


def processing_value_cell(table, step_index, channel_index=0):
    return table.cellWidget(step_index, channel_index + 2)


def processing_toggle_button(widget):
    assert isinstance(widget, MatrixToggleButton)
    return widget


def mask_step_cell(table, step_index):
    return table.cellWidget(step_index, 0)


def mask_step_button(table, step_index):
    return next(
        button
        for button in mask_step_cell(table, step_index).findChildren(QToolButton)
        if button.property("processingRole") == "mask_step_type"
    )


def mask_enabled_checkbox(table, step_index):
    enable_cell = table.cellWidget(step_index, 1)
    button = next(
        (
            button
            for button in enable_cell.findChildren(QPushButton)
            if button.property("processingRole") == "enabled"
        ),
        None,
    )
    if button is not None:
        return button
    return next(
        checkbox
        for checkbox in enable_cell.findChildren(QCheckBox)
        if checkbox.property("processingRole") == "enabled"
    )


def mask_column_by_name(table, name):
    for col_idx in range(1, table.columnCount()):
        header = table.horizontalHeaderItem(col_idx)
        if header is not None and header.text() == name:
            return col_idx
    raise AssertionError(f"Mask column not found: {name}")


def mask_value_cell(table, step_index, mask_name):
    return table.cellWidget(step_index, mask_column_by_name(table, mask_name))


def add_mask_steps(gui, *step_types):
    for step_type in step_types:
        gui.add_mask_processing_step(step_type)
    return gui.mask_processing_table


def test_cellpose_settings_disable_inputs_but_keep_enable_button_active():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.cellpose_settings_table = QTableWidget()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
                "analysis_cell_segmentation_enabled": False,
            },
        ]
    )

    gui.rebuild_cellpose_settings_table()
    table = gui.cellpose_settings_table
    enable_host = table.cellWidget(0, 0)
    enable_button = enable_host.findChild(QPushButton)
    remove_border_button = table.cellWidget(0, 1)
    source_combo = table.cellWidget(0, 2)
    diameter_edit = table.cellWidget(0, 5)

    assert enable_button.isEnabled() is True
    assert enable_button.text() == ""
    assert isinstance(remove_border_button, MatrixToggleButton)
    assert remove_border_button.text() == "ON"
    assert source_combo.isEnabled() is False
    assert source_combo.property("processingRowState") == "inactive"
    assert diameter_edit.isEnabled() is False
    assert diameter_edit.property("processingRowState") == "inactive"

    enable_button.click()

    assert source_combo.isEnabled() is True
    assert source_combo.property("processingRowState") == "active"
    assert diameter_edit.isEnabled() is True
    assert diameter_edit.property("processingRowState") == "active"

    remove_border_button.click()
    gui.commit_gui_edits()
    assert gui.image_definitions[0]["cell_remove_border"] is False


def test_weka_threshold_lives_with_mask_source_controls():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_tabs = QTabWidget()
    gui.mask_tabs = QTabWidget()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
            {
                "name": "Mask",
                "folder": "Mask",
                "classifier": "mask.model",
                "is_mask_only": True,
                "mask_source_channel": "Signal",
                "threshold_method": "Otsu",
            },
        ]
    )

    gui.rebuild_image_rows(sync_from_ui=False)
    mask_row = next(row for row in gui.image_rows if row and row.threshold_method is not None)
    threshold_combo = mask_row.threshold_method
    threshold_host = mask_row.threshold_method_host

    assert threshold_combo is not None
    assert threshold_host is not None

    assert threshold_combo.currentText() == "Otsu"
    assert threshold_host.isHidden() is False

    threshold_combo.setCurrentText("Triangle")
    active_defs = gui.get_active_image_definitions()
    mask_def = next(image_def for image_def in active_defs if image_def.get("name") == "Mask")
    assert mask_def["threshold_method"] == "Triangle"


def test_mask_source_controls_use_fiji_names_and_preserve_weka_classes():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_tabs = QTabWidget()
    gui.mask_source_tabs = QTabWidget()
    gui.mask_tabs = QTabWidget()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {"name": "Signal", "folder": "Signal", "mask_slot_enabled": False},
            {
                "name": "Mask",
                "folder": "Signal",
                "classifier": "mask.model",
                "is_mask_only": True,
                "mask_source_channel": "Signal",
                "probability_class_index": "1,3",
            },
        ]
    )

    gui.rebuild_image_rows(sync_from_ui=False)
    row = next(row for row in gui.image_rows if row and row.threshold_method is not None)
    assert row.threshold_method.itemText(0) == "Default (ImageJ IsoData)"
    assert row.threshold_method.itemData(0) == "Default"
    assert "MinError (I)" in [row.threshold_method.itemText(index) for index in range(row.threshold_method.count())]

    active_mask = next(item for item in gui.get_active_image_definitions() if item["name"] == "Mask")
    assert active_mask["probability_class_index"] == "1,3"


def test_mask_form_labels_fit_and_selected_model_shows_class_information(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    _ = app
    model_path = tmp_path / "mask.model"
    model_path.write_bytes(b"test")
    monkeypatch.setattr(
        "cellonaut.gui.dataset_images.inspect_weka_classifier",
        lambda _path, _fiji_path: ["background", "cell", "bud"],
    )

    gui = DummyRoiSettingsGui()
    gui.image_tabs = QTabWidget()
    gui.mask_source_tabs = QTabWidget()
    gui.mask_tabs = QTabWidget()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {"name": "Signal", "folder": "Signal", "mask_slot_enabled": False},
            {
                "name": "Mask",
                "folder": "Signal",
                "classifier": str(model_path),
                "is_mask_only": True,
                "mask_source_channel": "Signal",
            },
        ]
    )

    gui.rebuild_image_rows(sync_from_ui=False)
    row = next(row for row in gui.image_rows if row and row.classifier is not None)
    assert row.threshold_method_host.label.minimumWidth() == 250
    assert row.probability_class_index.label.minimumWidth() == 250

    row.classifier.edit.editingFinished.emit()
    assert "3 classes" in row.roi_status.text()
    assert "1: background; 2: cell; 3: bud" in row.roi_status.text()


def test_mask_source_tabs_assign_image_and_keep_combined_masks_global():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_tabs = QTabWidget()
    gui.mask_source_tabs = QTabWidget()
    gui.mask_tabs = QTabWidget()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {"name": "Signal", "folder": "Signal", "mask_slot_enabled": False},
            {"name": "Reference", "folder": "Reference", "mask_slot_enabled": False},
        ]
    )

    gui.rebuild_image_rows(sync_from_ui=False)
    assert [gui.mask_source_tabs.tabText(index) for index in range(gui.mask_source_tabs.count())] == [
        "Signal",
        "Reference",
        "Combined masks",
    ]
    assert gui.mask_source_tabs.tabBar().tabTextColor(0) == QColor("#e8757e")
    assert gui.mask_source_tabs.tabBar().tabTextColor(1) == QColor("#e8757e")

    source_pages = [gui.mask_source_tabs.widget(index) for index in range(gui.mask_source_tabs.count())]
    channel_pages = [gui.image_tabs.widget(index) for index in range(gui.image_tabs.count())]
    gui.on_mask_source_tab_changed(1)
    assert [gui.mask_source_tabs.widget(index) for index in range(gui.mask_source_tabs.count())] == source_pages
    assert [gui.image_tabs.widget(index) for index in range(gui.image_tabs.count())] == channel_pages
    gui.add_mask_row()
    assigned = gui.image_definitions[-1]
    assert assigned["mask_source_channel"] == "Reference"
    assert assigned["mask_source_mode"] != MASK_SOURCE_MODE_COMBINED
    assert next(row for row in gui.image_rows if row and row.classifier is not None).mask_source_channel is None
    assert gui.mask_source_tabs.tabBar().tabTextColor(1) == QColor("#d9a441")
    assert "incomplete" in gui.mask_source_tabs.tabToolTip(1).lower()

    gui.on_mask_source_tab_changed(2)
    gui.add_mask_row()
    combined = gui.image_definitions[-1]
    assert combined["mask_source_mode"] == MASK_SOURCE_MODE_COMBINED
    assert combined["combined_mask_sources"] == []
    assert combined["combined_mask_operation"] == "OR"
    assert gui.mask_source_tabs.tabText(gui.mask_source_tabs.currentIndex()) == "Combined masks"
    combined_row = next(row for row in gui.image_rows if row and row.combined_mask_operation is not None)
    assert combined_row.combined_mask_operation is not None
    assert [combined_row.combined_mask_operation.itemText(index) for index in range(3)] == [
        "OR (Combine)",
        "AND",
        "XOR",
    ]
    combined_row.combined_mask_operation.setCurrentText("XOR")
    gui.commit_gui_edits()
    assert gui.image_definitions[-1]["combined_mask_operation"] == "XOR"


def test_mask_tab_rebuild_hides_removed_pages_before_deferred_deletion():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_tabs = QTabWidget()
    gui.mask_source_tabs = QTabWidget()
    gui.mask_tabs = QTabWidget()
    gui.image_definitions = gui.normalize_image_definitions(
        [{"name": "Signal", "folder": "Signal", "mask_slot_enabled": False}]
    )
    gui.rebuild_image_rows(sync_from_ui=False)

    old_source_page = gui.mask_source_tabs.widget(0)
    old_add_mask_page = gui.mask_tabs.widget(0)
    gui.on_mask_tab_clicked(0)

    assert old_source_page.isHidden() is True
    assert old_add_mask_page.isHidden() is True


def test_mask_processing_keeps_background_controls_enabled_without_classifier():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
                "classifier": "",
                "bg_radii": "50",
            },
            {
                "name": "Mask",
                "folder": "Mask",
                "classifier": "mask.model",
                "is_mask_only": True,
                "mask_source_channel": "Signal",
            },
        ]
    )

    gui.rebuild_mask_processing_table()

    channel_table = gui.measurement_processing_table
    table = gui.mask_processing_table

    assert channel_table.horizontalHeader().isHidden() is False
    assert channel_table.verticalHeader().isVisible() is False
    assert channel_table.horizontalHeaderItem(0).text() == "Before measurement"
    assert channel_table.horizontalHeaderItem(1).text() == "Enable"
    assert channel_table.horizontalHeaderItem(2).text() == "Signal"
    assert processing_step_combo(processing_step_cell(channel_table, 0)).text() == "Subtract Background (px radius)"
    assert processing_scope_button_texts(processing_step_cell(channel_table, 0)) == []
    image_remove = next(
        button
        for button in processing_step_cell(channel_table, 0).findChildren(QPushButton)
        if button.property("processingRole") == "remove"
    )
    assert image_remove.property("cellonautIconName") == "trash-2"
    assert image_remove.minimumSize() == image_remove.maximumSize()
    assert image_remove.minimumSize().width() == image_remove.minimumSize().height()
    assert image_remove.minimumSize().width() <= 24
    assert channel_table.cellWidget(1, 0).findChild(QPushButton).text() == "Add step"
    assert channel_table.rowCount() == 2
    assert channel_table.horizontalHeaderItem(2).text() == "Signal"
    assert processing_value_cell(channel_table, 0).text() == "50"
    assert processing_value_cell(channel_table, 0).isEnabled() is True

    assert table.horizontalHeader().isHidden() is False
    assert table.verticalHeader().isVisible() is False
    assert table.rowCount() == 2
    assert table.columnCount() >= 2
    assert table.horizontalHeaderItem(0).text() == "Mask processing steps"
    assert mask_column_by_name(table, "Mask") >= 1
    assert table.cellWidget(1, 0).findChild(QPushButton).text() == "Add step"
    mask_step_header = mask_step_cell(table, 0)
    mask_step_selector = mask_step_button(table, 0)
    mask_enabled = mask_enabled_checkbox(table, 0)
    assert mask_step_selector.width() >= 240
    mask_step_layout = mask_step_header.layout()
    assert mask_step_layout.indexOf(mask_enabled) == -1
    assert mask_step_layout.stretch(mask_step_layout.indexOf(mask_step_selector)) == 1
    assert table.columnWidth(0) == gui.PROCESSING_STEP_COLUMN_WIDTH

    gui.add_mask_processing_step("analyze_particles")
    table = gui.mask_processing_table
    assert mask_step_button(table, 0).text() == "Analyze Particles (Fiji Analyze Particles)"
    assert mask_enabled_checkbox(table, 0).text() == ""
    assert mask_step_cell(table, 0).property("processingRowState") == "active"
    assert mask_value_cell(table, 0, "Mask").property("processingRowState") == "active"
    assert mask_value_cell(table, 0, "Mask").isEnabled() is True
    assert isinstance(mask_value_cell(table, 0, "Mask"), FijiParticleSettingsCell)
    assert menu_has_path(mask_step_button(table, 0).menu(), ["Analyze", "Analyze Particles"])
    assert menu_action_by_data(mask_step_button(table, 0).menu(), "analyze_particles") is not None
    assert menu_has_path(mask_step_button(table, 0).menu(), ["Process", "Binary"])
    assert not menu_has_path(mask_step_button(table, 0).menu(), ["Image", "Adjust", "Threshold"])
    assert table.cellWidget(1, 0).findChild(QPushButton).text() == "Add step"


def test_mask_processing_skeleton_toggle_matches_other_step_toggles():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Mask",
                "folder": "Mask",
                "classifier": "mask.model",
                "is_mask_only": True,
                "mask_source_channel": "Signal",
            },
        ]
    )

    table = add_mask_steps(gui, "analyze_skeleton")
    toggle = mask_value_cell(table, 0, "Mask")

    assert isinstance(toggle, QPushButton)
    assert toggle.text() == "ON"
    assert toggle.property("processingRole") == "value"
    assert toggle.minimumWidth() == 58


def test_fiji_mask_settings_reject_invalid_values_in_the_table():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [{"name": "Mask", "folder": "Mask", "classifier": "mask.model", "is_mask_only": True}]
    )
    table = add_mask_steps(gui, "binary_erode", "analyze_particles")
    binary = mask_value_cell(table, 0, "Mask")
    particles = mask_value_cell(table, 1, "Mask")
    assert isinstance(binary, FijiBinarySettingsCell)
    assert isinstance(particles, FijiParticleSettingsCell)

    binary.iterations.setText("0")
    particles.circularity_range.setText("0.5-1.2")
    assert gui.validate_mask_processing_table() is False
    assert binary.iterations.property("validationState") == "invalid"
    assert particles.circularity_range.property("validationState") == "invalid"

    binary.iterations.setText("2")
    particles.circularity_range.setText("0.5-1")
    assert gui.validate_mask_processing_table() is True


def validator_state(widget, text):
    pos = 0
    state, _text, _pos = widget.validator().validate(text, pos)
    return state


def test_mask_processing_steps_can_be_reordered_and_added():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Mask",
                "folder": "Mask",
                "classifier": "mask.model",
                "is_mask_only": True,
                "mask_source_channel": "Signal",
            },
        ]
    )

    gui.rebuild_mask_processing_table()
    table = add_mask_steps(gui, "analyze_particles", "binary_erode", "binary_dilate")
    assert mask_step_button(table, 1).text() == "Erode (Fiji Binary Options)"

    gui.reorder_mask_processing_recipe(1, 0)
    table = gui.mask_processing_table
    assert mask_step_button(table, 0).text() == "Erode (Fiji Binary Options)"

    gui.commit_gui_edits()
    mask_def = next(image_def for image_def in gui.image_definitions if image_def.get("name") == "Mask")
    assert [step["type"] for step in mask_def["mask_processing_steps"]][0] == "binary_erode"

    gui.add_mask_processing_step()
    table = gui.mask_processing_table
    added_step = mask_step_button(table, 3)
    assert added_step.text() == "Choose step"
    assert added_step.property("mutedState") == "true"
    assert mask_value_cell(table, 3, "Mask").isEnabled() is False

    trigger_processing_step(added_step, "binary_fill_holes")
    table = gui.mask_processing_table
    assert mask_step_button(table, 3).text() == "Fill Holes"


def test_combined_mask_source_selection_is_restored_from_preset_state():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    widget = QListWidget()
    widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
    gui._set_listwidget_items_with_selection(
        widget,
        ["Hmg2", "BFP", "Other"],
        selected={"Hmg2", "BFP"},
    )

    assert {item.text() for item in widget.selectedItems()} == {"Hmg2", "BFP"}


def test_image_processing_starts_with_a_protected_choose_step_row():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.rebuild_mask_processing_table()
    channel_table = gui.image_processing_table
    assert channel_table.rowCount() == 2
    assert channel_table.columnCount() == 3
    assert channel_table.horizontalHeaderItem(0).text() == "Before mask creation"
    assert channel_table.horizontalHeaderItem(1).text() == "Enable"
    assert channel_table.horizontalHeaderItem(2).text() == "Signal"
    assert channel_table.cellWidget(1, 0).findChild(QPushButton).text() == "Add step"
    assert processing_step_combo(processing_step_cell(channel_table, 0)).text() == "Choose step"
    assert processing_enabled_checkbox(channel_table.cellWidget(0, 1)).text() == ""
    assert processing_step_cell(channel_table, 0).property("processingRowState") == "inactive"
    assert processing_value_cell(channel_table, 0).property("processingRowState") == "inactive"
    assert processing_step_combo(processing_step_cell(channel_table, 0)).property("mutedState") == "true"
    assert processing_scope_button_texts(processing_step_cell(channel_table, 0)) == []
    assert isinstance(processing_value_cell(channel_table, 0), QLineEdit)
    assert processing_value_cell(channel_table, 0).isEnabled() is False
    assert processing_step_cell(channel_table, 0).findChild(QPushButton).property("cellonautIconName") == "trash-2"
    assert processing_step_cell(channel_table, 0).findChild(QPushButton).isEnabled() is False

    header_combo = processing_step_combo(processing_step_cell(channel_table, 0))
    trigger_processing_step(header_combo, "rolling_ball_background")

    channel_table = gui.image_processing_table
    selected_combo = processing_step_combo(processing_step_cell(channel_table, 0))
    assert selected_combo.text() == "Subtract Background (px radius)"
    assert processing_enabled_checkbox(channel_table.cellWidget(0, 1)).text() == ""
    assert processing_step_cell(channel_table, 0).property("processingRowState") == "active"
    assert processing_value_cell(channel_table, 0).property("processingRowState") == "active"
    assert processing_value_cell(channel_table, 0).isEnabled() is True
    assert processing_value_cell(channel_table, 0).isReadOnly() is False
    assert selected_combo.property("mutedState") == "false"
    assert selected_combo.property("stepType") != ""
    assert processing_scope_button_texts(processing_step_cell(channel_table, 0)) == []
    assert processing_value_cell(channel_table, 0).text() == ""


def test_image_processing_stages_replace_scope_buttons():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.add_image_processing_step("enhance_contrast")
    gui.add_measurement_processing_step("smooth")

    assert processing_step_combo(processing_step_cell(gui.image_processing_table, 0)).text() == "Enhance contrast (% saturated)"
    assert processing_step_combo(processing_step_cell(gui.measurement_processing_table, 0)).text() == "Smooth (iterations)"
    assert processing_scope_button_texts(processing_step_cell(gui.image_processing_table, 0)) == []
    assert processing_scope_button_texts(processing_step_cell(gui.measurement_processing_table, 0)) == []
    assert menu_has_path(
        processing_step_combo(processing_step_cell(gui.image_processing_table, 0)).menu(), ["Analyze"]
    ) is False
    assert menu_has_path(
        processing_step_combo(processing_step_cell(gui.measurement_processing_table, 0)).menu(), ["Analyze"]
    ) is False
    assert processing_warning_label(processing_step_cell(gui.image_processing_table, 0)).isVisible() is False
    assert processing_warning_label(processing_step_cell(gui.measurement_processing_table, 0)).property(
        "warningActive"
    ) == "true"
    assert [step["scope"] for step in gui.image_definitions[0]["image_processing_steps"]] == [
        "Segmentation input",
        "Measurement image",
    ]


def test_common_fiji_preprocessing_steps_are_available_and_saved():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.add_image_processing_step("gaussian_blur")
    gui.add_image_processing_step("median")
    gui.add_image_processing_step("despeckle")
    gui.add_image_processing_step("bit_depth")
    channel_table = gui.image_processing_table

    gaussian_combo = processing_step_combo(processing_step_cell(channel_table, 0))
    assert menu_has_path(gaussian_combo.menu(), ["Process", "Filters"])
    assert menu_has_path(gaussian_combo.menu(), ["Image", "Type"])
    assert menu_action_by_data(gaussian_combo.menu(), "gaussian_blur") is not None
    combo_widths = [
        processing_step_combo(processing_step_cell(channel_table, step_index)).width() for step_index in range(4)
    ]
    assert len(set(combo_widths)) == 1
    assert 200 <= combo_widths[0] <= 230

    assert gaussian_combo.text() == "Gaussian Blur (sigma px)"
    assert processing_step_combo(processing_step_cell(channel_table, 1)).text() == "Median (px radius)"
    assert processing_step_combo(processing_step_cell(channel_table, 2)).text() == "Despeckle"
    assert processing_step_combo(processing_step_cell(channel_table, 3)).text() == "Convert Bit Depth (bit depth)"
    assert isinstance(processing_value_cell(channel_table, 0), QLineEdit)
    assert processing_value_cell(channel_table, 0).text() == ""
    assert processing_value_cell(channel_table, 0).placeholderText() == "Inactive"
    assert processing_value_cell(channel_table, 0).placeholderText() == "Inactive"
    assert isinstance(processing_value_cell(channel_table, 1), QLineEdit)
    assert processing_value_cell(channel_table, 1).text() == ""
    assert isinstance(processing_value_cell(channel_table, 2), MatrixToggleButton)
    assert processing_toggle_button(processing_value_cell(channel_table, 2)).isChecked() is True
    bit_depth_cell = processing_value_cell(channel_table, 3)
    assert isinstance(bit_depth_cell, BitDepthSettingsCell)
    assert bit_depth_cell.combo.currentText() == "Inactive"
    assert bit_depth_cell.processing_value() == ""
    assert bit_depth_cell.scale.isChecked() is True

    gui.commit_gui_edits()

    steps = gui.image_definitions[0]["image_processing_steps"]
    assert [step["type"] for step in steps] == [
        "gaussian_blur",
        "median",
        "despeckle",
        "bit_depth",
    ]
    assert [step["enabled"] for step in steps] == [False, False, True, False]
    assert steps[0]["params"] == {"gaussian_sigma": ""}
    assert steps[1]["params"] == {"median_radius": ""}
    assert steps[2]["params"] == {}
    assert steps[3]["params"] == {"bit_depth": "", "scale_when_converting": True}


def test_fiji_command_options_persist_per_image_step():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions([{"name": "Signal", "folder": "Signal"}])
    gui.add_image_processing_step("rolling_ball_background")
    radius = processing_value_cell(gui.image_processing_table, 0)
    radius.setText("40")
    background_menu = radius.findChild(QMenu)
    assert background_menu is not None
    background_menu.actions()[0].setChecked(True)
    background_menu.actions()[1].setChecked(True)

    gui.add_image_processing_step("enhance_contrast")
    saturation = processing_value_cell(gui.image_processing_table, 1)
    saturation.setText("0.35")
    contrast_menu = saturation.findChild(QMenu)
    assert contrast_menu is not None
    contrast_menu.actions()[0].setChecked(True)

    gui.add_image_processing_step("bit_depth")
    bit_depth = processing_value_cell(gui.image_processing_table, 2)
    assert isinstance(bit_depth, BitDepthSettingsCell)
    bit_depth.combo.setCurrentText("8-bit")
    bit_depth.scale.setChecked(False)
    gui.commit_gui_edits()

    params = [step["params"] for step in gui.image_definitions[0]["image_processing_steps"]]
    assert params == [
        {"bg_radii": "40", "light_background": True, "sliding_paraboloid": True, "disable_smoothing": False},
        {"contrast_saturation": "0.35", "normalize": True, "equalize": False},
        {"bit_depth": "8-bit", "scale_when_converting": False},
    ]


def test_fiji_binary_controls_save_settings_and_show_supported_choices():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [{"name": "Mask", "folder": "Mask", "classifier": "mask.model", "is_mask_only": True}]
    )
    gui.add_mask_processing_step("binary_erode")
    table = gui.mask_processing_table
    selector = mask_step_button(table, 0)
    assert menu_action_by_data(selector.menu(), "binary_erode") is not None
    assert menu_action_by_data(selector.menu(), "binary_outline") is not None
    assert menu_action_by_data(selector.menu(), "binary_skeletonize") is not None
    for removed_step in ("min_roi_area", "shift_x", "shift_y", "grow_px", "fill_holes_area"):
        assert menu_action_by_data(selector.menu(), removed_step) is None

    settings = mask_value_cell(table, 0, "Mask")
    assert isinstance(settings, FijiBinarySettingsCell)
    settings.iterations.setText("4")
    settings.count.setText("2")
    settings.pad_edges.setChecked(True)
    gui.commit_gui_edits()

    assert gui.image_definitions[0]["mask_processing_steps"][0]["params"] == {"binary_settings": "4,2,true"}


def test_fiji_analyze_particles_controls_save_mask_options():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [{"name": "Mask", "folder": "Mask", "classifier": "mask.model", "is_mask_only": True}]
    )
    gui.add_mask_processing_step("analyze_particles")
    cell = mask_value_cell(gui.mask_processing_table, 0, "Mask")
    assert isinstance(cell, FijiParticleSettingsCell)
    cell.size_range.setText("20-Infinity")
    cell.circularity_range.setText("0.2-1")
    cell.exclude_edges.setChecked(True)
    cell.include_holes.setChecked(True)
    gui.commit_gui_edits()

    assert gui.image_definitions[0]["mask_processing_steps"][0]["params"] == {
        "particle_settings": "20-Infinity,0.2-1,true,true"
    }


def test_fiji_translate_controls_save_both_offsets():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [{"name": "Mask", "folder": "Mask", "classifier": "mask.model", "is_mask_only": True}]
    )
    gui.add_mask_processing_step("translate")
    table = gui.mask_processing_table
    assert menu_has_path(mask_step_button(table, 0).menu(), ["Image", "Transform"])
    cell = mask_value_cell(table, 0, "Mask")
    assert isinstance(cell, FijiTranslateSettingsCell)
    cell.x_offset.setText("-2.5")
    cell.y_offset.setText("3")
    gui.commit_gui_edits()

    assert gui.image_definitions[0]["mask_processing_steps"][0]["params"] == {
        "translate_offsets": "-2.5,3"
    }


def test_apply_lut_step_without_parameter_is_saved():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.add_image_processing_step("apply_lut")
    channel_table = gui.image_processing_table
    toggle = processing_toggle_button(processing_value_cell(channel_table, 0))
    assert toggle.isChecked() is True

    gui.commit_gui_edits()
    assert gui.image_definitions[0]["image_processing_steps"] == [
        {
                "type": "apply_lut",
                "enabled": True,
                "row_enabled": True,
                "scope": "Segmentation input",
            "params": {},
        }
    ]


def test_image_processing_allows_duplicate_step_types():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.add_image_processing_step("rolling_ball_background")
    gui.add_image_processing_step("rolling_ball_background")

    channel_table = gui.image_processing_table
    assert channel_table.rowCount() == 3
    assert processing_step_combo(processing_step_cell(channel_table, 0)).text() == "Subtract Background (px radius)"
    assert processing_step_combo(processing_step_cell(channel_table, 1)).text() == "Subtract Background (px radius)"

    processing_value_cell(channel_table, 0).setText("25")
    processing_value_cell(channel_table, 1).setText("75")
    gui.commit_gui_edits()

    steps = gui.image_definitions[0]["image_processing_steps"]
    assert [step["type"] for step in steps] == [
        "rolling_ball_background",
        "rolling_ball_background",
    ]
    assert [step["params"]["bg_radii"] for step in steps] == ["25", "75"]


def test_image_processing_steps_can_be_reordered_and_saved():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
                "bg_radii": "50",
            },
        ]
    )

    gui.rebuild_mask_processing_table()
    gui.add_measurement_processing_step("smooth")
    channel_table = gui.measurement_processing_table
    assert channel_table.horizontalHeaderItem(1).text() == "Enable"
    assert channel_table.horizontalHeaderItem(2).text() == "Signal"
    assert processing_step_combo(processing_step_cell(channel_table, 0)).text() == "Smooth (iterations)"
    assert processing_step_combo(processing_step_cell(channel_table, 1)).text() == "Subtract Background (px radius)"
    lock_indicators = [
        label
        for label in channel_table.cellWidget(1, 0).findChildren(QLabel)
        if label.property("processingRole") == "locked_step"
    ]
    assert len(lock_indicators) == 1
    assert "independent background-corrected measurement" in lock_indicators[0].toolTip()
    assert not any(
        label.property("processingRole") == "locked_step"
        for label in channel_table.cellWidget(0, 0).findChildren(QLabel)
    )

    gui.reorder_measurement_processing_recipe(1, 0)

    channel_table = gui.measurement_processing_table
    assert channel_table.horizontalHeaderItem(2).text() == "Signal"
    assert processing_step_combo(processing_step_cell(channel_table, 0)).text() == "Smooth (iterations)"
    assert processing_step_combo(processing_step_cell(channel_table, 1)).text() == "Subtract Background (px radius)"

    processing_value_cell(channel_table, 0).setText("9")
    processing_value_cell(channel_table, 1).setText("75")
    gui.commit_gui_edits()

    image_def = gui.image_definitions[0]
    assert image_def["bg_radii"] == "75"
    assert [step["type"] for step in image_def["image_processing_steps"]] == [
        "smooth",
        "rolling_ball_background",
    ]
    assert [step["scope"] for step in image_def["image_processing_steps"]] == [
        "Measurement image",
        "Measurement image",
    ]


def test_remove_outliers_editor_persists_fiji_settings():
    app = QApplication.instance() or QApplication([])
    _ = app
    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions([{"name": "Signal", "folder": "Signal"}])
    gui.rebuild_mask_processing_table()
    gui.add_image_processing_step("remove_outliers")

    cell = gui.image_processing_table.cellWidget(0, 2)
    assert isinstance(cell, OutlierSettingsCell)
    assert menu_has_path(processing_step_combo(processing_step_cell(gui.image_processing_table, 0)).menu(), ["Process", "Noise"])
    cell.radius.setText("2")
    assert gui.validate_mask_processing_table() is False
    assert cell.threshold.property("validationState") == "invalid"
    cell.threshold.setText("50")
    cell.which.setCurrentText("Dark")
    assert gui.validate_mask_processing_table() is True
    gui.commit_gui_edits()

    step = gui.image_definitions[0]["image_processing_steps"][0]
    assert step["type"] == "remove_outliers"
    assert step["enabled"] is True
    assert step["params"] == {"outlier_settings": "2,50,Dark"}


def test_image_processing_value_can_be_disabled_for_one_channel():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "GFP",
                "folder": "GFP",
            },
            {
                "name": "DAPI",
                "folder": "DAPI",
            },
        ]
    )

    gui.add_image_processing_step("smooth")
    channel_table = gui.image_processing_table
    assert channel_table.columnCount() == 4
    assert channel_table.horizontalHeaderItem(1).text() == "Enable"
    assert channel_table.horizontalHeaderItem(2).text() == "GFP"
    assert channel_table.horizontalHeaderItem(3).text() == "DAPI"

    processing_value_cell(channel_table, 0, 0).setText("5")
    assert processing_value_cell(channel_table, 0, 1).text() == ""

    gui.commit_gui_edits()

    gfp_step = gui.image_definitions[0]["image_processing_steps"][0]
    dapi_step = gui.image_definitions[1]["image_processing_steps"][0]
    assert gfp_step == {
        "type": "smooth",
        "enabled": True,
        "row_enabled": True,
        "scope": "Segmentation input",
        "params": {"smooth_iterations": "5"},
    }
    assert dapi_step == {
        "type": "smooth",
        "enabled": False,
        "row_enabled": True,
        "scope": "Segmentation input",
        "params": {"smooth_iterations": ""},
    }


def test_image_processing_value_fields_reject_letters_at_input_level():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.add_image_processing_step("rolling_ball_background")
    gui.add_image_processing_step("enhance_contrast")
    gui.add_image_processing_step("smooth")
    gui.add_image_processing_step("gaussian_blur")
    channel_table = gui.image_processing_table

    bg_cell = processing_value_cell(channel_table, 0)
    contrast_cell = processing_value_cell(channel_table, 1)
    smooth_cell = processing_value_cell(channel_table, 2)
    gaussian_cell = processing_value_cell(channel_table, 3)

    assert bg_cell.validator().validate("25,abc", 0)[0] == QValidator.State.Invalid
    assert bg_cell.validator().validate("25,50", 0)[0] == QValidator.State.Acceptable
    assert contrast_cell.validator().validate("15x", 0)[0] == QValidator.State.Invalid
    assert contrast_cell.validator().validate("15.5", 0)[0] == QValidator.State.Acceptable
    assert smooth_cell.validator().validate("5a", 0)[0] == QValidator.State.Invalid
    assert smooth_cell.validator().validate("5", 0)[0] == QValidator.State.Acceptable
    assert gaussian_cell.validator().validate("nan", 0)[0] == QValidator.State.Invalid
    assert gaussian_cell.validator().validate("1.5", 0)[0] == QValidator.State.Acceptable


def test_image_processing_invalid_values_are_highlighted_and_block_validation():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.add_image_processing_step("rolling_ball_background")
    gui.add_image_processing_step("enhance_contrast")
    gui.add_image_processing_step("smooth")
    gui.add_image_processing_step("gaussian_blur")
    channel_table = gui.image_processing_table

    bg_cell = processing_value_cell(channel_table, 0)
    contrast_cell = processing_value_cell(channel_table, 1)
    smooth_cell = processing_value_cell(channel_table, 2)
    gaussian_cell = processing_value_cell(channel_table, 3)

    bg_cell.setText("25,abc")
    contrast_cell.setText("101")
    smooth_cell.setText("0")
    gaussian_cell.setText("nan")

    assert gui.validate_mask_processing_table() is False
    assert bg_cell.property("validationState") == "invalid"
    assert contrast_cell.property("validationState") == "invalid"
    assert smooth_cell.property("validationState") == "invalid"
    assert gaussian_cell.property("validationState") == "invalid"
    assert "Validation:" in bg_cell.toolTip()

    bg_cell.setText("25,50")
    contrast_cell.setText("15")
    smooth_cell.setText("5")
    gaussian_cell.setText("1.5")

    assert gui.validate_mask_processing_table() is True
    assert bg_cell.property("validationState") == ""
    assert contrast_cell.property("validationState") == ""
    assert smooth_cell.property("validationState") == ""
    assert gaussian_cell.property("validationState") == ""


def test_image_processing_blank_values_remain_valid_skips():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.add_image_processing_step("smooth")
    channel_table = gui.image_processing_table
    smooth_cell = processing_value_cell(channel_table, 0)

    assert smooth_cell.text() == ""
    assert gui.validate_mask_processing_table() is True
    assert smooth_cell.property("validationState") == ""


def test_first_configured_image_processing_step_can_be_removed():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {
                "name": "Signal",
                "folder": "Signal",
            },
        ]
    )

    gui.add_image_processing_step("smooth")
    channel_table = gui.image_processing_table
    assert channel_table.rowCount() == 2

    remove_button = processing_step_cell(channel_table, 0).findChild(QPushButton)
    assert remove_button.property("cellonautIconName") == "trash-2"
    assert remove_button.isEnabled() is True
    remove_button.click()

    assert not any(step["type"] == "smooth" for step in gui.image_definitions[0]["image_processing_steps"])
    placeholder_remove = processing_step_cell(gui.image_processing_table, 0).findChild(QPushButton)
    assert placeholder_remove.isEnabled() is False


def test_add_and_remove_buttons_work_in_both_image_processing_stages():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions([{"name": "Signal", "folder": "Signal"}])
    gui.rebuild_mask_processing_table()

    for table, step_type in (
        (gui.image_processing_table, "smooth"),
        (gui.measurement_processing_table, "smooth"),
    ):
        table.cellWidget(table.rowCount() - 1, 0).findChild(QPushButton).click()
        table.cellWidget(table.rowCount() - 1, 0).findChild(QPushButton).click()
        assert table.rowCount() == 4
        blank_remove = next(
            button
            for button in processing_step_cell(table, 2).findChildren(QPushButton)
            if button.property("processingRole") == "remove"
        )
        assert blank_remove.isEnabled() is True
        blank_remove.click()
        assert table.rowCount() == 3
        step_index = table.rowCount() - 2
        selector = processing_step_combo(processing_step_cell(table, step_index))
        trigger_processing_step(selector, step_type)
        table = gui.image_processing_table if table is gui.image_processing_table else gui.measurement_processing_table
        assert processing_value_cell(table, step_index).isReadOnly() is False
        remove_button = next(
            button
            for button in processing_step_cell(table, step_index).findChildren(QPushButton)
            if button.property("processingRole") == "remove"
        )
        assert remove_button.isEnabled() is True
        remove_button.click()


def test_new_blank_mask_processing_step_can_be_removed():
    app = QApplication.instance() or QApplication([])
    _ = app

    gui = DummyRoiSettingsGui()
    gui.image_definitions = gui.normalize_image_definitions(
        [
            {"name": "Signal", "folder": "Signal"},
            {
                "name": "Mask",
                "folder": "Mask",
                "is_mask_only": True,
                "mask_source_channel": "Signal",
            },
        ]
    )
    gui.rebuild_mask_processing_table()
    gui.add_mask_processing_step()
    table = gui.mask_processing_table
    assert table.rowCount() == 3

    blank_remove = next(
        button
        for button in mask_step_cell(table, 1).findChildren(QPushButton)
        if button.property("processingRole") == "remove"
    )
    assert blank_remove.isEnabled() is True
    blank_remove.click()
    assert table.rowCount() == 2


def test_matrix_toggle_updates_in_place_without_rebuilding_view():
    gui = MatrixChangeHarness()

    gui.on_analysis_matrix_changed()

    assert gui.calls == ["save", "refresh", "warnings"]
