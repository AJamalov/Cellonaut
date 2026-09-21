from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeySequence, QStandardItemModel
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QWidget,
    QWidgetAction,
)

from cellonaut.gui.main_window import CellonautMainWindow
from cellonaut.gui import build as build_gui
from cellonaut.gui import config as config_gui
from cellonaut.gui import presets as presets_gui
from cellonaut.gui import dataset as dataset_gui
from cellonaut.gui import results as results_gui
from cellonaut.gui import validation as validation_gui
from cellonaut.io.fiji_installation import FijiComponentStatus, FijiInstallationStatus


pytestmark = pytest.mark.gui


class UnexpectedTopLevelShowFilter(QObject):
    """Record child controls accidentally shown before they receive a parent."""

    def __init__(self):
        super().__init__()
        self.widgets = []

    def eventFilter(self, watched, event):
        if (
            event.type() == QEvent.Type.Show
            and isinstance(watched, QWidget)
            and watched.isWindow()
            and not isinstance(watched, CellonautMainWindow)
        ):
            self.widgets.append(watched.metaObject().className())
        return False


@pytest.fixture(autouse=True)
def _disable_real_fiji_scans(monkeypatch):
    """GUI construction tests supply scan results explicitly; never scan the bundled Fiji tree."""
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)


class CloseEventStub:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


class RunningThreadStub:
    def isRunning(self):
        return True


class WorkerStub:
    def __init__(self):
        self.cancelled = False

    def request_cancel(self):
        self.cancelled = True


# GUI tests must never overwrite the developer's real startup preferences when windows close.
@pytest.fixture(autouse=True)
def prevent_user_settings_writes(monkeypatch):
    monkeypatch.setattr(CellonautMainWindow, "save_last_settings", lambda self: None)


def test_startup_and_mask_tabs_do_not_show_temporary_top_level_controls(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: True)
    show_filter = UnexpectedTopLevelShowFilter()
    app.installEventFilter(show_filter)

    window = CellonautMainWindow()
    try:
        window.finish_startup_initialization()

        source_tabs = window.mask_source_tabs
        source_index = 1 if source_tabs.count() > 1 else 0
        window.on_mask_source_tab_changed(source_index)

        mask_tabs = window.mask_tabs
        add_index = mask_tabs.count() - 1
        if mask_tabs.count() == 1:
            window.on_mask_tab_clicked(add_index)
        else:
            window.on_mask_tab_changed(add_index)

        assert show_filter.widgets == []
    finally:
        app.removeEventFilter(show_filter)
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_help_topics_and_context_links():
    app = QApplication.instance() or QApplication([])
    window = CellonautMainWindow()
    try:
        window.open_help_topic("Cellpose")
        cellpose_index = window.help_stack.currentIndex()
        assert window.left_stack.currentWidget() is window.help_tab
        assert window.help_nav_button.isChecked()

        assert window.help_topic_selector.count() == len(window.help_pages()) - 1
        assert window.help_topic_selector.currentText() == "Cellpose"
        assert window.help_stack.currentIndex() == cellpose_index
        assert not hasattr(window, "help_search")

        # The title-area link follows the active panel and clears stale searches.
        link = window.pipeline_editor_help_link
        assert link.parentWidget() is window.pipeline_editor_header
        for index, topic in enumerate(window.pipeline_section_help_topics):
            window.show_pipeline_section(index)
            assert link.accessibleName() == f"Help: {topic}"
            assert not any(
                label.accessibleName().startswith("Help:")
                for label in window.pipeline_sections[index].content.findChildren(QLabel)
            )
            link.linkActivated.emit("help")
            assert window.help_topic_selector.currentText() == topic
            assert window.left_stack.currentWidget() is window.help_tab
            help_page = cast(QTextBrowser, window.help_stack.currentWidget())
            assert help_page.toPlainText().startswith(topic)
        window.open_help_topic("Trainable Weka Segmentation")
        assert window.help_topic_selector.currentText() == "Trainable Weka Segmentation"
    finally:
        window.close()
        app.processEvents()


def test_main_window_starts_and_closes_offscreen(monkeypatch):
    app = QApplication.instance() or QApplication([])
    startup_calls = []

    # Keep this startup smoke test independent of user AppData and path validation.
    monkeypatch.setattr(
        CellonautMainWindow,
        "load_last_settings_if_available",
        lambda self: startup_calls.append("settings"),
    )
    monkeypatch.setattr(
        CellonautMainWindow,
        "load_default_preset_on_startup",
        lambda self: startup_calls.append("preset"),
    )
    monkeypatch.setattr(
        CellonautMainWindow,
        "validate_all_fields",
        lambda self: startup_calls.append("validation"),
    )
    window = CellonautMainWindow()
    try:
        assert startup_calls == []
        window.finish_startup_initialization()

        assert startup_calls == ["settings", "preset", "validation"]
        assert window._startup_initialization_complete is True
        assert window.centralWidget() is window.central
        assert window.as_qobject() is window
        assert window.left_stack.count() == 6
        assert [button.text() for button in window.primary_navigation_buttons] == [
            "Pipeline",
            "Files",
            "Log",
            "Settings",
            "Help",
            "About",
        ]
        assert window.left_stack.indexOf(window.log_tab) == 2
        assert window.left_stack.indexOf(window.help_tab) == 4
        assert window.left_stack.indexOf(window.about_tab) == 5
        assert window.left_stack.currentWidget() is window.pipeline_tab
        assert window.pipeline_nav_button.isChecked() is True
        window.files_nav_button.click()
        assert window.left_stack.currentWidget() is window.files_left_tab
        assert window.files_nav_button.isChecked() is True
        assert window.pipeline_nav_button.isChecked() is False
        window.pipeline_nav_button.click()
        assert window.left_stack.currentWidget() is window.pipeline_tab
        assert all(not button.icon().isNull() for button in window.primary_navigation_buttons)
        assert window.help_topic_selector.count() == len(window.help_pages()) - 1
        assert window.help_stack.count() == len(window.help_pages()) - 1
        assert all(
            window.help_topic_selector.itemText(index) != "About"
            for index in range(window.help_topic_selector.count())
        )
        window.help_topic_selector.setCurrentIndex(2)
        assert window.help_stack.currentIndex() == 2
        assert "Cellonaut" in window.about_browser.toPlainText()
        window.about_nav_button.click()
        assert window.left_stack.currentWidget() is window.about_tab
        assert window.about_nav_button.isChecked() is True
        window.pipeline_nav_button.click()
        assert window.left_stack.currentWidget() is window.pipeline_tab
        assert window.primary_navigation_rail.width() == 120
        root_margins = window.root_layout.contentsMargins()
        assert (root_margins.left(), root_margins.top(), root_margins.right(), root_margins.bottom()) == (0, 0, 0, 0)
        left_container_layout = window.left_container.layout()
        status_layout = window.bottom_status_frame.layout()
        assert left_container_layout is not None
        assert status_layout is not None
        assert left_container_layout.spacing() == 0
        assert all(button.minimumHeight() == 42 for button in window.primary_navigation_buttons)
        settings_position = window.primary_navigation_layout.indexOf(window.settings_nav_button)
        assert window.primary_navigation_layout.indexOf(window.guided_tutorial_button) == settings_position + 1
        assert window.guided_tutorial_button.property("cellonautIconName") == "book-open"
        assert window.guided_tutorial_button.icon().isNull() is False
        assert window.primary_navigation_layout.indexOf(window.help_nav_button) == settings_position + 2
        assert window.primary_navigation_layout.indexOf(window.about_nav_button) == settings_position + 3
        assert all(section.toggle_button.height() == 72 for section in window.pipeline_sections)
        assert all(section.minimumHeight() == 72 for section in window.pipeline_sections)
        assert all(section.maximumHeight() == 72 for section in window.pipeline_sections)
        assert all(section.header_text_layout.count() == 2 for section in window.pipeline_sections)
        assert all(section.step_badge.parentWidget() is section.toggle_button for section in window.pipeline_sections)
        assert all(section.step_badge.isWindow() is False for section in window.pipeline_sections)
        assert window.app_header_frame.minimumHeight() == 60
        assert window.bottom_status_frame.height() == 28
        assert window.pipeline_spinner_label.text() == "●"
        assert window.pipeline_spinner_label.property("status") == "idle"
        assert window.current_sample_label.text() == "No task running"
        assert window.progress_bar.isHidden() is True
        assert window.progress_bar.width() == 240
        window.set_status_style("Running")
        window.set_pipeline_busy(True)
        assert window.pipeline_spinner_label.property("status") == "running"
        assert window.progress_bar.isHidden() is False
        window.set_pipeline_busy(False)
        window.set_status_style("Idle")
        assert window.progress_bar.isHidden() is True
        status_margins = status_layout.contentsMargins()
        assert (status_margins.left(), status_margins.top(), status_margins.right(), status_margins.bottom()) == (
            12,
            2,
            12,
            2,
        )
        assert window.run_button.icon().isNull() is False
        assert window.preview_pipeline_button.icon().isNull() is False
        assert window.check_setup_button.icon().isNull() is False
        assert window.run_button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
        assert window.preview_pipeline_button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
        assert window.check_setup_button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
        assert window.run_button.minimumWidth() == 132
        assert window.preview_pipeline_button.minimumWidth() == 164
        assert window.check_setup_button.minimumWidth() == 118
        assert window.brand_version_label.isHidden() is True
        assert window.right_layout.indexOf(window.preview_tab) >= 0
        assert not hasattr(window, "preview_tabs")
        assert not hasattr(window, "main_splitter")
        assert window.main_workspace_layout.indexOf(window.left_container) == 0
        assert window.main_workspace_layout.indexOf(window.right_container) == 1
        assert window.main_workspace_layout.stretch(0) == 1
        assert window.main_workspace_layout.stretch(1) == 1
        window.show()
        app.processEvents()
        assert abs(window.right_container.width() - window.left_container.width()) <= 1
        assert abs(
            (window.left_container.width() / (window.left_container.width() + window.right_container.width())) - 0.5
        ) <= 0.03
        assert window.left_container.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
        assert window.left_container.minimumWidth() == 470
        assert window.right_container.minimumWidth() == 390
        assert [
            window.project_section.step_badge.text(),
            window.images_section.step_badge.text(),
            window.cellpose_settings_section.step_badge.text(),
            window.image_processing_section.step_badge.text(),
            window.analysis_section.step_badge.text(),
        ] == ["1", "2", "3", "4", "5"]
        assert "channel(s)" in window.project_section.summary_label.text()
        assert "mask source(s)" in window.images_section.summary_label.text()
        assert not hasattr(window, "analysis_measurements_button")
        assert window.analysis_measurements_panel.isHidden() is False
        assert window.analysis_whole_cell_measurement_checks_host.title() == "Whole-cell measurements"
        assert window.analysis_cell_measurement_checks_host.title() == "Mask inside each cell"
        assert not hasattr(window, "analysis_output_measurement_checks_host")
        assert window.images_group.isHidden() is False
        assert not hasattr(window, "channels_waiting_label")
        assert "measurement option(s) enabled" in window.analysis_section.summary_label.text()
        scientific_tables = [
            window.image_processing_table,
            window.mask_processing_table,
            window.measurement_processing_table,
            window.cellpose_settings_table,
            window.analysis_matrix_table,
        ]
        for table in scientific_tables:
            assert table.property("scientificTable") == "true"
            assert table.showGrid() is False
            assert table.alternatingRowColors() is True
            assert table.wordWrap() is False
            assert table.verticalHeader().defaultSectionSize() == 32
            assert table.verticalHeader().minimumSectionSize() == 28
            assert table.horizontalHeader().height() == 30
        for table in (
            window.image_processing_table,
            window.mask_processing_table,
            window.measurement_processing_table,
            window.cellpose_settings_table,
        ):
            assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Fixed
            assert table.horizontalHeader().sectionsMovable() is False
        cellpose_header_items = [
            window.cellpose_settings_table.horizontalHeaderItem(column)
            for column in range(window.cellpose_settings_table.columnCount())
        ]
        assert all(item is not None for item in cellpose_header_items)
        cellpose_headers = [item.text() for item in cellpose_header_items if item is not None]
        cellpose_header_metrics = window.cellpose_settings_table.horizontalHeader().fontMetrics()
        assert all(
            window.cellpose_settings_table.columnWidth(column)
            >= cellpose_header_metrics.horizontalAdvance(header) + 28
            for column, header in enumerate(cellpose_headers)
        )
        assert cellpose_headers[-2:] == ["Probability threshold", "Flow threshold"]
        assert window.cellpose_settings_table.columnWidth(4) < 180
        assert all(not window.cellpose_settings_table.isColumnHidden(column) for column in range(9))
        assert not hasattr(window, "cellpose_advanced_row")
        assert window.analysis_matrix_table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert window.analysis_matrix_table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert window.pipeline_editor_header.isHidden() is True
        assert window.pipeline_overview_header.isHidden() is False
        assert window.pipeline_overview_title.text() == "Pipeline"
        assert window._active_pipeline_section_index is None

        window.images_section.toggle_button.click()
        assert window._active_pipeline_section_index == 1
        assert window.pipeline_overview_header.isHidden() is True
        assert window.pipeline_editor_title.text() == "2. Masks"
        assert window.images_section.content.isVisible() is True
        assert window.images_section.content.property("editorVisible") == "true"
        assert window.images_section.maximumHeight() == 16_777_215
        assert window.project_section.isHidden() is True
        assert window.pipeline_previous_section_button.text() == "← Setup"
        assert window.pipeline_next_section_button.text() == "Cellpose →"

        window.pipeline_next_section_button.click()
        assert window._active_pipeline_section_index == 2
        assert window.cellpose_settings_section.content.isVisible() is True
        assert window.images_section.isHidden() is True

        window.pipeline_back_button.click()
        assert window._active_pipeline_section_index is None
        assert window.pipeline_overview_header.isHidden() is False
        assert all(not section.isHidden() for section in window.pipeline_sections)
        assert all(section.content.isHidden() for section in window.pipeline_sections)
        assert all(section.content.property("editorVisible") == "false" for section in window.pipeline_sections)
        assert window.pipeline_overview_step_count.text() == "5 steps"
        assert window.preview_tools_section.step_badge.isHidden() is True
        assert "Open a TIFF overlay" in window.preview_tools_context_label.text()

        window.preview_tools_section.toggle_button.click()
        assert window._active_pipeline_section_index == window.pipeline_sections.index(window.preview_tools_section)
        assert window.pipeline_editor_title.text() == "Image Preview Tools"
        assert window.preview_tools_tabs.currentWidget() is window.analysis_mask_adjust_panel
        assert window.mask_adjust_target_combo.isEnabled() is False
        assert all(
            widget.isEnabled() is False
            for widget in (
                window.mask_adjust_x_spin,
                window.mask_adjust_y_spin,
                window.mask_adjust_grow_spin,
                window.mask_adjust_min_size_spin,
                window.mask_adjust_fill_holes_spin,
            )
        )
        window.preview_tools_tabs.setCurrentWidget(window.analysis_filters_panel)
        assert window._inline_analysis_settings_panel == "filters"
        assert window.preview_tools_tabs.currentWidget() is window.analysis_filters_panel
        window.pipeline_back_button.click()
        assert window.image_tabs.count() >= 2
        assert window.preview_scene is not None
        assert window.preview_toolbar.property("uiRole") == "toolbar"
        assert window.preview_artifact_selector.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
        snapshot_index = window.preview_artifact_selector.findData("snapshot")
        assert snapshot_index >= 0
        selector_model = cast(QStandardItemModel, window.preview_artifact_selector.model())
        assert selector_model.item(snapshot_index).isEnabled() is False
        assert window.preview_workspace_title.text() == "Image Preview"
        assert window.preview_tab_layout.indexOf(window.preview_workspace_title) < window.preview_tab_layout.indexOf(
            window.preview_toolbar
        )
        preview_toolbar_layout = window.preview_toolbar.layout()
        assert preview_toolbar_layout is not None
        assert preview_toolbar_layout.count() == 4
        assert preview_toolbar_layout.indexOf(window.preview_sample_navigation) >= 0
        assert preview_toolbar_layout.indexOf(window.preview_image_actions) >= 0
        sample_navigation_layout = window.preview_sample_navigation.layout()
        image_actions_layout = window.preview_image_actions.layout()
        assert sample_navigation_layout is not None
        assert image_actions_layout is not None
        assert sample_navigation_layout.indexOf(window.preview_prev_sample_button) >= 0
        assert sample_navigation_layout.indexOf(window.preview_next_sample_button) >= 0
        assert sample_navigation_layout.indexOf(window.preview_sample_nav_label) >= 0
        assert image_actions_layout.indexOf(window.preview_inspector_toggle_button) >= 0
        assert image_actions_layout.indexOf(window.preview_recent_images_button) >= 0
        assert window.preview_recent_images_button.isEnabled() is False
        assert image_actions_layout.indexOf(window.preview_snapshot_toggle_button) == -1
        assert window.preview_inspector.minimumWidth() == 190
        assert window.preview_inspector.maximumWidth() == 220
        preview_workspace_layout = window.preview_workspace.layout()
        assert preview_workspace_layout is not None
        assert preview_workspace_layout.indexOf(window.preview_tool_strip) == 0
        assert window.preview_tool_strip.width() == 64
        assert window.preview_bottom_bar.height() == 42
        assert preview_workspace_layout.indexOf(window.preview_view) == 1
        assert window.preview_tab_layout.indexOf(window.preview_bottom_bar) > window.preview_tab_layout.indexOf(
            window.preview_workspace
        )
        preview_tool_strip_layout = window.preview_tool_strip.layout()
        preview_bottom_layout = window.preview_bottom_bar.layout()
        assert preview_tool_strip_layout is not None
        assert preview_bottom_layout is not None
        assert not hasattr(window, "preview_pan_button")
        assert not hasattr(window, "preview_fit_button")
        tool_categories = [
            label.text()
            for label in window.preview_tool_strip.findChildren(QLabel)
            if label.property("uiRole") == "previewToolCategory"
        ]
        assert tool_categories == ["View", "Zoom", "Snapshot", "Export"]
        assert not window.preview_export_image_button.icon().isNull()
        assert window.preview_export_image_button.accessibleName() == "Save preview image"
        assert window.preview_snapshot_toggle_button.toolTip() == "Show or hide the square snapshot selection on the preview image."
        assert window.preview_metadata_button.parentWidget() is window.preview_inspector
        assert not hasattr(window, "preview_inspector_close_button")
        assert preview_bottom_layout.indexOf(window.preview_prev_artifact_button) >= 0
        assert preview_bottom_layout.indexOf(window.preview_next_artifact_button) >= 0
        assert preview_bottom_layout.indexOf(window.preview_zoom_label) >= 0
        assert preview_bottom_layout.indexOf(window.preview_reset_view_button) == -1
        assert window.preview_inspector.isHidden() is True
        assert window.preview_layer_bar.property("uiRole") == "flatPanel"
        assert not hasattr(window, "preview_filter_layer_group")
        busy_states = []
        monkeypatch.setattr(
            window,
            "_refresh_preview_filter_overlay",
            lambda: busy_states.append(window.preview_work_busy_indicator.isVisible()),
        )
        window.refresh_preview_filter_overlay()
        assert busy_states == [True]
        assert window.preview_work_busy_indicator.isHidden() is True
        assert window.preview_work_busy_icon.pixmap().isNull() is False
        assert window.preview_work_busy_icon.property("cellonautIconName") == "hourglass"
        assert not hasattr(window, "preview_work_busy_timer")
        assert window.preview_info_panel.property("uiRole") == "flatPanel"
        window.preview_inspector_toggle_button.click()
        assert window.preview_inspector.isHidden() is False
        window.preview_inspector_toggle_button.click()
        assert window.preview_inspector.isHidden() is True
        window.preview_metadata_button.setChecked(True)
        assert window.preview_inspector.isHidden() is False
        assert window.preview_info_panel.isHidden() is False
        window.preview_metadata_button.setChecked(False)
        assert window.preview_inspector.isHidden() is True
        window.preview_inspector_toggle_button.setChecked(True)
        window.update_preview_responsive_layout(600)
        assert window.preview_inspector_toggle_button.isChecked() is False
        assert window.preview_inspector.isHidden() is True
        assert window._preview_inspector_auto_collapsed is True
        window.update_preview_responsive_layout(900)
        assert window.preview_inspector_toggle_button.isChecked() is True
        assert window.preview_inspector.isHidden() is False
        assert window._preview_inspector_auto_collapsed is False
        window.preview_inspector_toggle_button.setChecked(False)
        assert window.preview_inspector_toggle_button.text() == "Layers"
        assert window.preview_inspector_title.text() == "Layers and metadata"

        window.preview_fullscreen_button.setChecked(True)
        assert window._preview_focus_mode is True
        assert window.app_header_frame.isHidden() is True
        assert window.left_container.isHidden() is True
        assert window.bottom_status_frame.isHidden() is True
        assert window.preview_tab.isVisible() is True
        window.set_preview_focus_mode(False)
        assert window._preview_focus_mode is False
        assert window.app_header_frame.isHidden() is False
        assert window.left_container.isHidden() is False
        assert window.bottom_status_frame.isHidden() is False
        assert window.log_flush_timer.isActive()
        assert window.preview_composite_checkbox.isChecked() is False
        assert window.browser_back_button.text() == ""
        assert window.browser_forward_button.text() == ""
        assert window.browser_up_button.text() == ""
        assert window.browser_back_button.icon().isNull() is False
        assert window.browser_forward_button.icon().isNull() is False
        assert window.browser_up_button.icon().isNull() is False
        assert not hasattr(window, "browser_path_label")
        assert not hasattr(window, "browser_open_file_button")
        assert window.browser_computer_button.text() == "Home"
        assert window.browser_open_location_button.text() == "Open location"
        assert not hasattr(window, "browser_nd2_button")
        assert window.browser_input_button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
        assert window.browser_output_button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed

        files_toolbar_item = window.files_left_tab_layout.itemAt(0)
        assert files_toolbar_item is not None
        files_toolbar_widget = files_toolbar_item.widget()
        assert files_toolbar_widget is not None
        assert files_toolbar_widget is window.files_browser_toolbar
        files_toolbar_layout = files_toolbar_widget.layout()
        assert files_toolbar_layout is not None
        files_toolbar_item = files_toolbar_layout.itemAt(0)
        address_row_item = files_toolbar_layout.itemAt(1)
        assert files_toolbar_item is not None
        assert address_row_item is not None
        files_toolbar = files_toolbar_item.layout()
        address_row = address_row_item.layout()
        assert files_toolbar is not None
        assert address_row is not None
        assert files_toolbar.indexOf(window.browser_compare_runs_button) >= 0
        assert files_toolbar.indexOf(window.files_help_link) == files_toolbar.indexOf(window.browser_compare_runs_button) + 1
        assert files_toolbar.indexOf(window.browser_open_location_button) == files_toolbar.count() - 1
        assert files_toolbar.indexOf(window.browser_computer_button) < files_toolbar.indexOf(window.browser_input_button)
        assert address_row.count() == 3
        assert address_row.indexOf(window.browser_path_entry) >= 0
        assert address_row.indexOf(window.browser_go_button) >= 0
        assert address_row.indexOf(window.browser_browse_button) > address_row.indexOf(window.browser_go_button)
        assert [action.text() for action in window.browser_browse_button.menu().actions()] == [
            "Open image...",
            "Open folder...",
        ]
        assert window.preview_empty_text_label.text() == "Drag and drop image here"
        assert window.preview_empty_state.isHidden() is False

        assert window.root_layout.indexOf(window.app_header_frame) < window.root_layout.indexOf(window.main_workspace)
        assert not window.bottom_status_frame.isHidden()
        assert window.root_layout.indexOf(window.bottom_status_frame) > window.root_layout.indexOf(window.main_workspace)
        preset_row = window.preset_commands_widget.layout()
        assert preset_row is not None
        assert preset_row.indexOf(window.save_to_preset_button) < preset_row.indexOf(window.save_preset_as_button)
        assert preset_row.indexOf(window.save_preset_as_button) < preset_row.indexOf(window.revert_preset_button)
        assert preset_row.indexOf(window.revert_preset_button) < preset_row.indexOf(window.preset_overflow_button)
        assert window.preset_combo.maximumWidth() == 220
        assert window.preset_overflow_menu.actions()
        command_layout = window.pipeline_commands_widget.layout()
        assert command_layout is not None
        assert command_layout.indexOf(window.run_button) >= 0
        assert command_layout.indexOf(window.preview_pipeline_button) >= 0
        assert command_layout.indexOf(window.check_setup_button) == command_layout.indexOf(window.preview_pipeline_button) + 1
        import_action = cast(QWidgetAction, window.preset_overflow_menu.actions()[0])
        assert import_action.defaultWidget() is window.import_preset_button
        delete_action = cast(QWidgetAction, window.preset_overflow_menu.actions()[-1])
        assert delete_action.defaultWidget() is window.delete_preset_button
        assert window.delete_preset_button.text() == "Delete preset"
        assert window.delete_preset_button.property("cellonautIconName") == "trash-2"
        assert not window.delete_preset_button.icon().isNull()
        assert window.brand_name_label.text() == "Cellonaut"
        assert window.brand_icon_label.isHidden() is False
        assert window.brand_icon_label.pixmap().isNull() is False
        assert window.brand_icon_label.width() == 36
        assert window.brand_icon_label.height() == 36
        assert window.brand_version_label.text().startswith("Version ")
        assert not hasattr(window, "load_preset_button")
        assert window.cancel_button.isHidden() is True

        monkeypatch.setattr(window, "show_latest_qc_overlay", lambda: None)
        window.on_worker_done(
            {
                "run_stats": {
                    "failed": 1,
                    "total_targets": 4,
                    "failed_targets": 1,
                }
            },
            completed_with_errors=True,
        )

        assert window.status_label.text() == "Completed with errors"
        assert window.current_sample_label.text() == "Last run completed: 4 target(s), 1 failed"
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_check_setup_runs_report_completion_off_gui_thread(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)
    opened_dialogs = []

    class DialogStub:
        def __init__(self, summary, text, parent):
            assert parent._setup_check_thread is None
            opened_dialogs.append((summary, text, parent))

        def exec(self):
            return 0

    window = CellonautMainWindow()
    summary = {
        "folders": {"input": "C:/images", "output": "", "fiji": "C:/Fiji", "layout": "flat"},
        "images": [],
        "relationships": [],
        "cell_segmentation": [],
        "filters": [],
        "measurements": [],
        "warnings": [],
        "dry_run": {},
        "setup_checks": [],
    }
    monkeypatch.setattr(
        window,
        "build_configuration_summary_snapshot",
        lambda: (object(), [], [], [], summary),
    )
    monkeypatch.setattr(
        validation_gui,
        "complete_configuration_summary",
        lambda snapshot, *_args, **_kwargs: {
            **snapshot,
            "dry_run": {"available": True, "total": 0, "ready": 0, "blocked": 0},
            "setup_checks": [{"status": "OK", "check": "Worker", "detail": "Completed"}],
        },
    )
    monkeypatch.setattr(validation_gui, "ConfigurationSummaryDialog", DialogStub)
    monkeypatch.setattr(validation_gui, "write_text", lambda *_args, **_kwargs: None)
    try:
        started = time.monotonic()
        window.show_setup_check_report()

        assert time.monotonic() - started < 0.2
        assert window._setup_check_thread is not None
        assert window.check_setup_button.isEnabled() is False

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and (window._setup_check_thread is not None or not opened_dialogs):
            app.processEvents()
            time.sleep(0.005)

        assert window._setup_check_thread is None
        assert opened_dialogs and opened_dialogs[0][0]["setup_checks"][0]["check"] == "Worker"
        assert window.status_label.text() == "Ready"
        assert window.check_setup_button.isEnabled() is True
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_loading_preset_separates_machine_settings_and_schedules_one_scan(monkeypatch):
    internal_fiji = Path(r"C:\Bundled\Fiji.app")
    monkeypatch.setattr(config_gui, "get_bundled_fiji_path", lambda: internal_fiji)
    monkeypatch.setattr(config_gui, "get_internal_fiji_path", lambda: internal_fiji)
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    try:
        scan_calls = []
        monkeypatch.setattr(
            window,
            "schedule_input_path_scan",
            lambda path, immediate=False: scan_calls.append((path, immediate)),
        )
        window.apply_startup_state_dict(
            {
                "appearance_mode": "dark_teal",
                "fiji_app_path": r"C:\Fiji.app",
            }
        )
        window.apply_preset_dict(
            {
                "appearance_mode": "light_blue",
                "ui_scale": "120%",
                "compact_mode": False,
                "input_dir": r"C:\Data\Ost1_TIFF",
                "output_dir": r"C:\Data\Ost1_output",
                "fiji_app_path": r"C:\WrongFiji.app",
                "threshold_method": "Otsu",
                "nd2_detected_channel_names": ["Stale channel"],
            },
        )

        assert scan_calls == [(r"C:\Data\Ost1_TIFF", True)]
        assert window._suppress_input_path_scan is False
        preset = window.get_preset_dict()
        bundled_default_path = Path(__file__).parents[1] / "cellonaut" / "data" / "presets" / "Default.json"
        bundled_default = json.loads(bundled_default_path.read_text(encoding="utf-8"))
        assert set(preset) == set(bundled_default)
        assert set(window.get_startup_state_dict()) == {"appearance_mode", "recent_preview_images"}
        for excluded_key in (
            "appearance_mode",
            "fiji_app_path",
            "window_width",
            "window_height",
            "threshold_method",
            "nd2_detected_channel_names",
            "nd2_detected_sizes",
            "nd2_detected_file_count",
            "nd2_detected_first_display",
            "ui_scale",
            "compact_mode",
            "image_processing_state",
        ):
            assert excluded_key not in preset
        assert not hasattr(window, "ui_scale")
        assert not hasattr(window, "compact_mode")
        assert window.appearance_mode.get() == "dark_teal"
        assert window.fiji_app.get() == str(internal_fiji)
        assert window.nd2_detected_channel_names == []
        assert [window.appearance_mode.combo.itemText(index) for index in range(4)] == [
            "Light Blue",
            "Dark Blue",
            "Dark Teal",
            "Dark Purple",
        ]
        assert [window.appearance_mode.combo.itemData(index) for index in range(4)] == [
            "light_blue",
            "dark_blue",
            "dark_teal",
            "dark_purple",
        ]
        assert window.settings_nav_button.text() == "Settings"
        assert not hasattr(window, "options_tab")
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_main_window_can_start_maximized(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    try:
        window.showMaximized()
        app.processEvents()

        assert window.isMaximized()
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_preset_round_trip_excludes_application_level_nd2_and_fiji_values(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)

    window = CellonautMainWindow()
    try:
        monkeypatch.setattr(window, "schedule_input_path_scan", lambda *_args, **_kwargs: None)
        window.nd2_z_mode.combo.blockSignals(True)
        window.nd2_z_mode.set("Single Z slice")
        window.nd2_z_mode.combo.blockSignals(False)
        window.nd2_z_index_spin.setMaximum(5)
        window.nd2_z_index_spin.setValue(3)
        window.nd2_channel_folder_state = {"GFP": "Green"}
        window.nd2_output_dir.set("C:/current-conversion")
        window.apply_preset_dict(
            {
                "nd2_z_index": "invalid",
                "nd2_z_mode": "Max projection",
                "nd2_use_converted_input": "false",
                "nd2_channel_folder_state": ["invalid"],
                "nd2_output_dir": "C:/other-conversion",
                "fiji_app_path": "C:/other-fiji",
                "reuse_existing_masks": "false",
                "measurement_options": {"area": "false", "mean": "true"},
                "image_definitions": [{
                    "name": "GFP", "cell_qc_limits": "Area:2-10", "mask_qc_limits": "",
                    "cell_qc_exclude_flagged": True, "qc_filter_mode": "Use cell filters only",
                    "mask_qc_intensity_source": "Cell mask image",
                }],
            }
        )

        assert window.nd2_z_mode.get() == "Single Z slice"
        assert window.nd2_z_index_spin.value() == 3
        assert window.nd2_channel_folder_state == {"GFP": "Green"}
        assert window.nd2_output_dir.get() == "C:/current-conversion"
        assert window.reuse_existing_masks_checkbox.isChecked() is False
        assert window.measurement_options["area"] is False
        assert window.measurement_options["mean"] is True

        saved = window.get_preset_dict()
        for removed_key in (
            "fiji_app_path",
            "nd2_output_dir",
            "nd2_z_mode",
            "nd2_z_index",
            "nd2_channel_folder_state",
            "nd2_use_converted_input",
        ):
            assert removed_key not in saved
        assert {image["cellpose_model_type"] for image in saved["image_definitions"]} == {"cpsam"}
        for image in saved["image_definitions"]:
            assert not {"cell_qc_limits", "mask_qc_limits", "cell_qc_exclude_flagged",
                        "qc_filter_mode", "mask_qc_intensity_source"}.intersection(image)
        group = saved["image_definitions"][0]["cell_populations"][0]
        assert group["cell_qc_limits"] == "Area:2-10"
        assert group["exclude_from_csv"] is True
        assert group["mask_qc_intensity_source"] == "Cell mask image"
        window.apply_preset_dict(saved)
        assert window.get_preset_dict() == saved
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_settings_opens_cross_platform_data_folder(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    app_data_dir = tmp_path / "Cellonaut"
    opened_urls = []
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)
    monkeypatch.setattr(build_gui, "USER_APP_DIR", app_data_dir)
    monkeypatch.setattr(
        build_gui,
        "ensure_user_app_dirs",
        lambda: (app_data_dir / "presets").mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        build_gui.QDesktopServices,
        "openUrl",
        lambda url: opened_urls.append(url) or True,
    )

    window = CellonautMainWindow()
    try:
        help_text = window.data_folder_help_label.text()
        assert "crash logs, saved settings, and preset files" in help_text
        assert "macOS" not in help_text
        assert "Linux" not in help_text
        assert "AppData" not in help_text
        assert window.import_preset_button.text() == "Import Preset..."
        assert window.export_preset_button.text() == "Export Selected Preset..."
        assert window.import_preset_button.accessibleName()
        assert window.export_preset_button.accessibleName()

        assert window.open_data_folder_button.text() == "Open Cellonaut Data Folder"
        window.open_data_folder_button.click()

        assert (app_data_dir / "presets").is_dir()
        assert [Path(url.toLocalFile()) for url in opened_urls] == [app_data_dir]

        text_file = tmp_path / "RunSummary.txt"
        text_file.write_text("Pipeline summary", encoding="utf-8")
        window.show_left_page(window.pipeline_tab)
        window.open_text_in_log_tab(text_file)

        assert window.left_stack.currentWidget() is window.log_tab
        assert window.log_nav_button.isChecked() is True
        assert window.log_box.toPlainText() == "Pipeline summary"
        assert window.right_layout.indexOf(window.preview_tab) >= 0
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_preset_selection_loads_immediately_without_unsaved_changes(monkeypatch):
    app = QApplication.instance() or QApplication([])
    load_calls = []
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)
    monkeypatch.setattr(
        CellonautMainWindow,
        "load_selected_preset",
        lambda self: load_calls.append(self.preset_combo.currentText()) or True,
    )
    monkeypatch.setattr(CellonautMainWindow, "preset_has_unsaved_changes", lambda self: False)

    window = CellonautMainWindow()
    try:
        window.preset_combo.blockSignals(True)
        window.preset_combo.clear()
        window.preset_combo.addItems(["Preset A", "Preset B"])
        window.preset_combo.setCurrentIndex(0)
        window.preset_combo.blockSignals(False)
        window._active_preset_name = "Preset A"

        window.preset_combo.setCurrentIndex(1)
        app.processEvents()
        assert load_calls == ["Preset B"]
        assert not hasattr(window, "load_preset_button")
        assert window.save_to_preset_button.text() == "Save"
        assert window.save_to_preset_button.icon().isNull() is False
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.parametrize(
    ("reply", "expected_save_calls", "expected_load_calls", "expected_selection"),
    [
        (QMessageBox.StandardButton.Save, 1, ["Preset B"], "Preset B"),
        (QMessageBox.StandardButton.Discard, 0, ["Preset B"], "Preset B"),
        (QMessageBox.StandardButton.Cancel, 0, [], "Preset A"),
    ],
)
def test_preset_selection_resolves_unsaved_changes_before_loading(
    monkeypatch,
    reply,
    expected_save_calls,
    expected_load_calls,
    expected_selection,
):
    app = QApplication.instance() or QApplication([])
    load_calls = []
    save_calls = []
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "preset_has_unsaved_changes", lambda self: True)
    monkeypatch.setattr(
        CellonautMainWindow,
        "save_active_preset_changes",
        lambda self: save_calls.append(self._active_preset_name) or True,
    )
    monkeypatch.setattr(
        CellonautMainWindow,
        "load_selected_preset",
        lambda self: load_calls.append(self.preset_combo.currentText()) or True,
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: reply)

    window = CellonautMainWindow()
    try:
        window.preset_combo.blockSignals(True)
        window.preset_combo.clear()
        window.preset_combo.addItems(["Preset A", "Preset B"])
        window.preset_combo.setCurrentIndex(0)
        window.preset_combo.blockSignals(False)
        window._active_preset_name = "Preset A"

        window.preset_combo.setCurrentIndex(1)
        app.processEvents()

        assert len(save_calls) == expected_save_calls
        assert load_calls == expected_load_calls
        assert window.preset_combo.currentText() == expected_selection
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("save_succeeds", [True, False])
def test_switching_from_edited_default_preserves_destination_after_save_as(monkeypatch, save_succeeds):
    app = QApplication.instance() or QApplication([])
    loaded = []
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "preset_has_unsaved_changes", lambda self: True)
    monkeypatch.setattr(
        CellonautMainWindow,
        "load_selected_preset",
        lambda self: loaded.append(self.preset_combo.currentText()) or True,
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Save)

    window = CellonautMainWindow()
    try:
        window.preset_combo.blockSignals(True)
        window.preset_combo.clear()
        window.preset_combo.addItems(["Default", "Preset B"])
        window.preset_combo.setCurrentIndex(0)
        window.preset_combo.blockSignals(False)
        window._active_preset_name = "Default"

        def save_as():
            if not save_succeeds:
                return False
            window.preset_combo.blockSignals(True)
            window.preset_combo.addItem("New preset")
            window.preset_combo.setCurrentIndex(window.preset_combo.findText("New preset"))
            window.preset_combo.blockSignals(False)
            window._active_preset_name = "New preset"
            return True

        monkeypatch.setattr(window, "save_preset_as", save_as)
        window.preset_combo.setCurrentIndex(1)

        assert loaded == (["Preset B"] if save_succeeds else [])
        assert window.preset_combo.currentText() == ("Preset B" if save_succeeds else "Default")
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_pipeline_mask_controls_live_in_setup_and_cancel_stays_in_header(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    try:
        command_layout = window.pipeline_commands_widget.layout()
        assert command_layout is not None
        assert command_layout.indexOf(window.run_button) >= 0
        assert command_layout.indexOf(window.cancel_button) >= 0
        assert command_layout.indexOf(window.reuse_existing_masks_checkbox) == -1
        assert window.reuse_existing_masks_checkbox.parentWidget() is window.io_group
        assert window.mask_source_report_button.parentWidget() is window.io_group

        window.set_cancel_button_active(True)
        assert window.cancel_button.isVisibleTo(window.top_status_frame) is True
        assert window.cancel_button.isEnabled() is True
        window.set_cancel_button_active(False)
        assert window.cancel_button.isHidden() is True

        repeated_titles = [
            label.text()
            for label in window.mask_processing_group.findChildren(QLabel)
            if label.text() == "Mask processing"
        ]
        assert repeated_titles == []
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_preset_edit_marker_and_direct_save_feedback(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    preset_path = tmp_path / "Preset A.json"
    try:
        window.preset_combo.blockSignals(True)
        window.preset_combo.clear()
        window.preset_combo.addItem("Preset A")
        window.preset_combo.setCurrentIndex(0)
        window.preset_combo.blockSignals(False)
        window.input_dir.set("")
        window.output_dir.set("")
        window.remember_preset_state("Preset A", window.get_preset_dict())
        initial_setup_summary = window.project_section.summary_label.text()

        window.show()
        window.raise_()
        window.activateWindow()
        app.processEvents()
        window.input_dir.edit.setFocus()
        QTest.keyClicks(window.input_dir.edit, "x")
        assert window.input_dir.get() == "x"
        assert window.images_group.isHidden() is False
        assert window.input_dir.edit.property("cellonautPresetEditWatcher") is True
        window.refresh_preset_dirty_indicator()

        assert window.preset_dirty_label.isVisible() is True
        assert window.save_to_preset_button.isEnabled() is True
        assert window.revert_preset_button.isEnabled() is True
        assert initial_setup_summary.startswith("Choose input and output folders")
        assert window.project_section.summary_label.text() != initial_setup_summary

        preset_path.write_text(json.dumps({"_preset_removed_field": "discard me"}), encoding="utf-8")
        monkeypatch.setattr(window, "preset_path_from_name", lambda _name: preset_path)
        monkeypatch.setattr(window, "refresh_presets_list", lambda **_kwargs: None)
        modal_calls = []
        monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: modal_calls.append(True))

        window.save_to_selected_preset()

        saved = json.loads(preset_path.read_text(encoding="utf-8"))
        assert "_preset_removed_field" not in saved
        assert saved["panel_notes"] == window.get_preset_dict()["panel_notes"]
        assert modal_calls == []
        assert window.status_label.text() == "Preset saved: Preset A"
        assert window.preset_dirty_label.isHidden() is True
        assert window.save_to_preset_button.isEnabled() is False
        assert window.revert_preset_button.isEnabled() is False
    finally:
        feedback_timer = getattr(window, "_preset_feedback_timer", None)
        if feedback_timer is not None:
            feedback_timer.stop()
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_default_preset_shows_lock_and_uses_save_as(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    (presets_dir / "Default.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(presets_gui, "PRESETS_DIR", presets_dir)

    window = CellonautMainWindow()
    try:
        monkeypatch.setattr(window, "ensure_default_preset", lambda: None)
        window.refresh_presets_list(select_name="Default")
        default_index = window.preset_combo.findText("Default")

        assert default_index >= 0
        assert not window.preset_combo.itemIcon(default_index).isNull()
        assert "locked" in window.preset_combo.itemData(default_index, Qt.ItemDataRole.ToolTipRole).lower()

        window._preset_dirty = True
        window.refresh_preset_control_states()
        assert window.save_to_preset_button.isEnabled() is True
        assert "new preset" in window.save_to_preset_button.toolTip().lower()
        assert window.delete_preset_button.isEnabled() is False

        save_as_calls = []
        monkeypatch.setattr(window, "save_preset_as", lambda: save_as_calls.append(True) or True)
        window.save_to_selected_preset()

        assert save_as_calls == [True]
        assert json.loads((presets_dir / "Default.json").read_text(encoding="utf-8")) == {}
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_primary_task_state_locks_competing_controls(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    try:
        window.preset_combo.blockSignals(True)
        window.preset_combo.clear()
        window.preset_combo.addItem("Preset A")
        window.preset_combo.setCurrentIndex(0)
        window.preset_combo.blockSignals(False)
        window._preset_dirty = True
        window.refresh_preset_control_states()

        window.set_primary_task_active(True)
        assert window.run_button.isEnabled() is False
        assert window.preview_pipeline_button.isEnabled() is False
        assert window.check_setup_button.isEnabled() is False
        assert window.reuse_existing_masks_checkbox.isEnabled() is False
        assert window.mask_source_report_button.isEnabled() is False
        assert window.preset_combo.isEnabled() is False
        assert window.save_to_preset_button.isEnabled() is False
        assert window.import_preset_button.isEnabled() is False
        assert window.export_preset_button.isEnabled() is False
        assert window.nd2_convert_button.isEnabled() is False
        assert window.cancel_button.isHidden() is False
        assert window.cancel_button.isEnabled() is True

        window.cancel_button.setEnabled(False)
        assert window.cancel_button.isHidden() is False

        window.set_primary_task_active(False)
        assert window.run_button.isEnabled() is True
        assert window.preview_pipeline_button.isEnabled() is True
        assert window.check_setup_button.isEnabled() is True
        assert window.preset_combo.isEnabled() is True
        assert window.save_to_preset_button.isEnabled() is True
        assert window.import_preset_button.isEnabled() is True
        assert window.export_preset_button.isEnabled() is True
        assert window.nd2_convert_button.isEnabled() is False
        assert window.cancel_button.isHidden() is True
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_pipeline_commands_reflow_without_moving_file_buttons(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    try:
        window.update_pipeline_command_layout(2000)
        preset_index = window.app_header_layout.indexOf(window.preset_commands_widget)
        preset_position = cast(tuple[int, int, int, int], window.app_header_layout.getItemPosition(preset_index))
        assert preset_position[0] == 0

        window.update_pipeline_command_layout(420)
        preset_index = window.app_header_layout.indexOf(window.preset_commands_widget)
        preset_position = cast(tuple[int, int, int, int], window.app_header_layout.getItemPosition(preset_index))
        assert preset_position[0] == 1

        toolbar_item = window.files_left_tab_layout.itemAt(0)
        assert toolbar_item is not None
        toolbar_widget = toolbar_item.widget()
        assert toolbar_widget is not None
        assert toolbar_widget is window.files_browser_toolbar
        toolbar_layout = toolbar_widget.layout()
        assert toolbar_layout is not None
        toolbar_row_item = toolbar_layout.itemAt(0)
        assert toolbar_row_item is not None
        toolbar = toolbar_row_item.layout()
        assert toolbar is not None
        expected_order = [
            window.browser_back_button,
            window.browser_forward_button,
            window.browser_up_button,
            window.browser_computer_button,
            window.browser_input_button,
            window.browser_output_button,
            window.browser_compare_runs_button,
        ]
        positions = [toolbar.indexOf(widget) for widget in expected_order]
        assert positions == sorted(positions)
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_files_browse_menu_opens_images_and_folders(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    image_path = tmp_path / "chosen.tif"
    image_path.touch()
    folder_path = tmp_path / "chosen-folder"
    folder_path.mkdir()
    window = CellonautMainWindow()
    try:
        roots = []
        previews = []
        monkeypatch.setattr(window, "set_browser_root", roots.append)
        monkeypatch.setattr(window, "open_file_in_right_panel", previews.append)
        monkeypatch.setattr(
            results_gui.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(image_path), "Images"),
        )
        monkeypatch.setattr(
            results_gui.QFileDialog,
            "getExistingDirectory",
            lambda *_args, **_kwargs: str(folder_path),
        )

        window.browse_for_preview_image()
        assert roots == [str(tmp_path)]
        assert previews == [str(image_path)]
        window.browse_for_folder()
        assert roots[-1] == str(folder_path)
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_cellpose_custom_model_can_be_browsed_and_cleared(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    model_path = tmp_path / "custom-cellpose-model"
    model_path.touch()
    monkeypatch.setattr(
        dataset_gui.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(model_path), "All files"),
    )
    window = CellonautMainWindow()
    try:
        enable_cell = window.cellpose_settings_table.cellWidget(0, 0)
        enable = enable_cell.findChild(QPushButton)
        assert enable is not None
        enable.click()
        custom_cell = window.cellpose_settings_table.cellWidget(0, 4)
        edit = custom_cell.findChild(QLineEdit, "CellposeCustomModelPath")
        browse = custom_cell.findChild(QPushButton, "CellposeCustomModelBrowse")
        clear = custom_cell.findChild(QPushButton, "CellposeCustomModelClear")
        assert edit is not None
        assert browse is not None and not browse.icon().isNull()
        assert clear is not None and not clear.icon().isNull()
        assert clear.isEnabled() is False

        browse.click()
        assert window.image_definitions[0]["cellpose_custom_model_path"] == str(model_path)

        custom_cell = window.cellpose_settings_table.cellWidget(0, 4)
        clear = custom_cell.findChild(QPushButton, "CellposeCustomModelClear")
        assert clear is not None and clear.isEnabled() is True
        clear.click()
        assert window.image_definitions[0]["cellpose_custom_model_path"] == ""
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_files_open_location_opens_the_displayed_path(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    window = CellonautMainWindow()
    try:
        opened = []
        monkeypatch.setattr(results_gui, "open_folder_in_system_browser", opened.append)

        assert window.browser_open_location_button.isEnabled() is False
        window.browser_path_entry.setText(str(tmp_path))
        assert window.browser_open_location_button.isEnabled() is True
        window.browser_open_location_button.click()

        assert opened == [tmp_path]
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_nd2_import_uses_dedicated_always_available_dialog(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    try:
        assert window.io_group.layout().indexOf(window.nd2_group) == -1
        assert window.nd2_import_button.isEnabled() is True
        assert window.nd2_convert_button.isEnabled() is False
        assert not hasattr(window, "nd2_use_converted_input")
        assert not hasattr(window, "nd2_apply_channels_button")
        assert not hasattr(window, "nd2_convert_and_run_button")
        assert not hasattr(window, "nd2_inspect_button")
        assert window.nd2_detected_channels_label.text() == "Waiting for files..."
        assert window.nd2_convert_button.isHidden() is True
        assert window.nd2_channel_rows_container.isHidden() is True
        assert window.nd2_mapping_help.isHidden() is True
        assert window.nd2_progress_panel.isHidden() is True
        assert window.nd2_progress_bar.value() == 0

        window.nd2_source_dir.set(str(tmp_path))
        assert window.nd2_source_scan_timer.isActive() is True
        window.nd2_source_scan_timer.stop()

        window.nd2_import_button.click()
        app.processEvents()

        assert window.nd2_dialog.isHidden() is False
        assert window.nd2_group.isHidden() is False
        assert window.nd2_source_dir.isHidden() is False
        assert window.nd2_output_dir.isHidden() is False
        assert not hasattr(window, "nd2_scroll")
        assert not hasattr(window, "nd2_scroll_content")
        nd2_dialog_layout = window.nd2_dialog.layout()
        assert nd2_dialog_layout is not None
        assert nd2_dialog_layout.indexOf(window.nd2_group) >= 0
    finally:
        window.nd2_dialog.close()
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_revert_preset_discards_unsaved_changes_with_feedback(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    load_calls = []
    try:
        window.preset_combo.blockSignals(True)
        window.preset_combo.clear()
        window.preset_combo.addItem("Preset A")
        window.preset_combo.setCurrentIndex(0)
        window.preset_combo.blockSignals(False)
        window._active_preset_name = "Preset A"
        monkeypatch.setattr(window, "preset_has_unsaved_changes", lambda: True)
        monkeypatch.setattr(window, "load_selected_preset", lambda: load_calls.append(True) or True)
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
        )

        window.revert_selected_preset()

        assert load_calls == [True]
        assert window.status_label.text() == "Preset reverted: Preset A"
    finally:
        feedback_timer = getattr(window, "_preset_feedback_timer", None)
        if feedback_timer is not None:
            feedback_timer.stop()
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_toolbar_shortcuts_focus_order_and_availability(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    try:
        assert window.save_to_preset_button.shortcut() in QKeySequence.keyBindings(QKeySequence.StandardKey.Save)
        assert window.save_preset_as_button.shortcut() in QKeySequence.keyBindings(QKeySequence.StandardKey.SaveAs)
        assert window.cancel_button.shortcut().toString() == "Esc"
        assert window.run_button.nextInFocusChain() is window.preview_pipeline_button
        assert window.preview_pipeline_button.nextInFocusChain() is window.check_setup_button
        for widget in (
            window.run_button,
            window.preview_pipeline_button,
            window.check_setup_button,
            window.cancel_button,
            window.browser_back_button,
            window.browser_go_button,
            window.browser_browse_button,
        ):
            assert widget.accessibleName()
            assert widget.toolTip()

        window.input_dir.set("")
        window.output_dir.set("")
        window.nd2_output_dir.set("")
        window.browser_path_entry.clear()
        window.update_browser_navigation_buttons()
        assert window.browser_input_button.isEnabled() is False
        assert window.browser_output_button.isEnabled() is False
        assert window.browser_go_button.isEnabled() is False
        assert window.browser_browse_button.isEnabled() is True

        window.input_dir.set(str(tmp_path))
        window.output_dir.set(str(tmp_path))
        window.nd2_output_dir.set(str(tmp_path))
        window.browser_path_entry.setText(str(tmp_path))
        window.set_browser_root(str(tmp_path))
        assert window.browser_input_button.isEnabled() is True
        assert window.browser_output_button.isEnabled() is True
        assert window.browser_go_button.isEnabled() is True
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_close_requests_cancellation_and_waits_for_active_worker(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    worker = WorkerStub()
    window.worker = worker
    window.worker_thread = RunningThreadStub()
    window._close_retry_scheduled = True
    event = CloseEventStub()
    try:
        window.closeEvent(event)

        assert worker.cancelled is True
        assert event.ignored is True
        assert event.accepted is False
    finally:
        window.worker = None
        window.worker_thread = None
        window.log_flush_timer.stop()
        window.deleteLater()
        app.processEvents()


def test_close_cancels_setup_check_worker(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    worker = WorkerStub()
    window._setup_check_worker = worker
    window._setup_check_thread = RunningThreadStub()
    event = CloseEventStub()
    try:
        window.closeEvent(event)

        assert worker.cancelled is True
        assert event.ignored is True
        assert event.accepted is False
    finally:
        window._setup_check_worker = None
        window._setup_check_thread = None
        window.log_flush_timer.stop()
        window.deleteLater()
        app.processEvents()


def test_close_cancels_fiji_scan_worker(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    worker = WorkerStub()
    window._fiji_scan_worker = worker
    window._fiji_scan_thread = RunningThreadStub()
    event = CloseEventStub()
    try:
        window.closeEvent(event)

        assert worker.cancelled is True
        assert event.ignored is True
        assert event.accepted is False
    finally:
        window._fiji_scan_worker = None
        window._fiji_scan_thread = None
        window.log_flush_timer.stop()
        window.deleteLater()
        app.processEvents()


def test_close_discards_pending_scans_and_blocks_late_rescans(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    event = CloseEventStub()
    try:
        window._pending_input_scan_path = "late-input"
        window._pending_fiji_scan_path = "late-fiji"
        window._input_scan_debounce.start(0)
        window._fiji_scan_debounce.start(0)

        window.closeEvent(event)
        app.processEvents()

        assert event.accepted is True
        assert window._closing_requested is True
        assert window._pending_input_scan_path == ""
        assert window._pending_fiji_scan_path == ""
        assert not window._input_scan_debounce.isActive()
        assert not window._fiji_scan_debounce.isActive()
        window.schedule_input_path_scan("late-input", immediate=True)
        window._pending_input_scan_path = "late-input"
        window.start_pending_input_path_scan()
        assert window._input_scan_thread is None
        assert not window._input_scan_debounce.isActive()
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_cancel_current_task_requests_active_worker_and_updates_status(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)

    window = CellonautMainWindow()
    worker = WorkerStub()
    window.worker = worker
    try:
        window.cancel_current_task()

        assert worker.cancelled is True
        assert window.status_label.text() == "Stopping run..."
        assert window.current_sample_label.text() == "Stopping run..."
        assert window.sample_counter_label.text() == "Cancelling"
        assert window.cancel_button.isEnabled() is False
    finally:
        window.worker = None
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_fiji_component_checklist_shows_versions_without_found_rows(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)

    window = CellonautMainWindow()
    try:
        fiji_path = tmp_path / "Fiji.app"
        status = FijiInstallationStatus(
            path=fiji_path,
            path_exists=True,
            path_is_dir=True,
            looks_like_fiji=True,
            launcher=fiji_path / "ImageJ-win64.exe",
            components=(
                FijiComponentStatus(
                    "launcher",
                    "Fiji launcher",
                    True,
                    "Found launcher: C:/Fiji.app/ImageJ-win64.exe",
                ),
                FijiComponentStatus("java", "Java", True, "Found Java.", version="21.0.7"),
                FijiComponentStatus("bioformats", "Bio-Formats / ND2", True, "Found.", version="8.5.0"),
                FijiComponentStatus("weka", "Trainable Weka Segmentation", True, "Found.", version="4.0.0"),
                FijiComponentStatus("imagescience", "ImageScience", False, "ImageScience was not found."),
            ),
            version="2.18.1-SNAPSHOT",
        )

        html = window.fiji_component_checklist_html(str(fiji_path), status=status)

        assert "Fiji 2.18.1-SNAPSHOT" in html
        assert "Java 21.0.7" in html
        assert "Bio-Formats / ND2 8.5.0" in html
        assert "Trainable Weka Segmentation 4.0.0" in html
        assert "ImageScience is not installed" in html
        assert "font-size:15px" in html
        assert "Derivatives, Laplacian, and Structure" in html
        assert "How to install ImageScience:" in html
        assert "Open Fiji Folder" in html
        assert "Run the Fiji launcher" in html
        assert "Manage update sites" in html
        assert "Apply changes" in html
        assert "found" not in html.lower()
        assert "ImageScience was not found." not in html
        assert str(fiji_path) not in html

        opened_urls = []
        monkeypatch.setattr(
            "cellonaut.gui.build.QDesktopServices.openUrl",
            lambda url: opened_urls.append(url) or True,
        )
        window.fiji_app.set(str(fiji_path))
        window.apply_fiji_component_scan_result(
            {"path": str(fiji_path), "status": status, "scan_error": ""}
        )
        assert window.open_fiji_folder_button.isHidden() is False
        settings_icon_key = window.settings_nav_button.icon().cacheKey()
        assert not window.settings_nav_button.icon().isNull()
        assert window.settings_warning_icon_label.isHidden() is False
        assert not window.settings_warning_icon_label.pixmap().isNull()
        warning_image = window.settings_warning_icon_label.pixmap().toImage()
        assert any(
            (color := warning_image.pixelColor(x, y)).alpha() > 0
            and color.red() > 240
            and color.green() > 180
            and color.blue() < 100
            for y in range(warning_image.height())
            for x in range(warning_image.width())
        )
        assert "ImageScience is not installed" in window.settings_nav_button.toolTip()
        window.open_fiji_folder_button.click()
        assert [Path(url.toLocalFile()) for url in opened_urls] == [fiji_path]

        installed_status = FijiInstallationStatus(
            path=fiji_path,
            path_exists=True,
            path_is_dir=True,
            looks_like_fiji=True,
            launcher=fiji_path / "ImageJ-win64.exe",
            components=tuple(
                FijiComponentStatus(
                    component.key,
                    component.label,
                    True,
                    "ImageScience found.",
                    version=component.version,
                )
                if component.key == "imagescience"
                else component
                for component in status.components
            ),
            version=status.version,
        )
        window.apply_fiji_component_scan_result(
            {"path": str(fiji_path), "status": installed_status, "scan_error": ""}
        )
        assert not window.settings_nav_button.icon().isNull()
        assert window.settings_nav_button.icon().cacheKey() == window.settings_tab_default_icon.cacheKey()
        assert window.settings_nav_button.icon().cacheKey() == settings_icon_key
        assert window.settings_warning_icon_label.isHidden() is True
        assert window.settings_nav_button.toolTip() == "Settings"
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_runtime_status_summarizes_application_versions(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)

    window = CellonautMainWindow()
    try:
        html = window.runtime_installation_summary_html(str(tmp_path / "Fiji.app"))
        assert "Cellonaut 1.0.0" in html
        assert "Cellpose:</b>" in html
        assert "PyTorch:</b>" in html
        assert "Installation:" not in html
        assert "Default model:" not in html
        assert "Built-in Cellpose models:" not in html
    finally:
        window.log_flush_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()
