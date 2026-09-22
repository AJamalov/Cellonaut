from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QRect, QSettings
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QDialog, QMessageBox, QPushButton, QWidget

import cellonaut.gui.guided_tutorial as guided_tutorial_module
from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.gui.guided_tutorial import GuidedTutorialMixin, TUTORIAL_STEPS
from cellonaut.gui.main_window import CellonautMainWindow


pytestmark = pytest.mark.gui


class TutorialHarness(GuidedTutorialMixin, QWidget):
    input_dir: QWidget

    def __init__(self, settings):
        self.preview_state = PreviewState()
        super().__init__()
        self.guided_tutorial_button = QPushButton("Tutorial", self)
        self.files_left_tab = QWidget(self)
        self.guided_tutorial_button.clicked.connect(self.toggle_guided_tutorial)
        self.initialize_guided_tutorial()
        self._tutorial_settings = settings
        self.applied_steps = []
        self.restored_snapshot = None
        self._update_tutorial_button()

    def commit_gui_edits(self):
        pass

    def get_preset_dict(self):
        return {"original": True}

    def _prepare_tutorial_workspace(self):
        pass

    def _show_tutorial_section(self, index):
        self.section = index

    def show_left_page(self, page):
        self.left_page = page

    def _apply_tutorial_step(self, step_index):
        self.applied_steps.append(step_index)
        if TUTORIAL_STEPS[step_index].action == "run":
            self._tutorial_run_state = "complete"
        return True

    def apply_preset_dict(self, snapshot, **_kwargs):
        self.restored_snapshot = snapshot

    def has_active_processing_task(self):
        return False


def test_tutorial_uses_one_final_processing_disable_step_and_current_names():
    actions = [step.action for step in TUTORIAL_STEPS]
    preview_tools_step = next(step for step in TUTORIAL_STEPS if step.target == "preview_tools_tabs")
    diameter_step = next(step for step in TUTORIAL_STEPS if step.action == "cellpose_diameter")

    assert [action for action in actions if action.startswith("processing_disable_")] == [
        "processing_disable_examples"
    ]
    assert preview_tools_step.title == "Image Preview Tools"
    assert preview_tools_step.body.startswith("Image Preview Tools")
    assert "Blank keeps the original scale" in diameter_step.body


def test_guided_tutorial_advances_and_remembers_completion(qt_application, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)

    assert gui.guided_tutorial_button.property("attention") is True
    gui.guided_tutorial_button.click()
    assert gui.guided_tutorial_button.text() == "Skip tutorial"
    assert gui._tutorial_step == 0
    assert gui._tutorial_previous_button.isEnabled() is False

    for step in range(len(TUTORIAL_STEPS)):
        assert gui._tutorial_step == step
        assert gui._tutorial_phase == 0
        if TUTORIAL_STEPS[step].action:
            gui._tutorial_next_button.click()
            if TUTORIAL_STEPS[step].single_click:
                assert gui._tutorial_step == step + 1
                continue
            assert gui._tutorial_phase == 1
            assert gui._tutorial_highlighted_widget is not None or gui._tutorial_target(TUTORIAL_STEPS[step].target) is None
        gui._tutorial_next_button.click()

    assert gui._tutorial_dialog is None
    assert gui.applied_steps == [
        index for index, step in enumerate(TUTORIAL_STEPS) if step.action
    ]
    assert gui.restored_snapshot == {"original": True}
    assert gui.guided_tutorial_button.text() == "Tutorial"
    assert gui.guided_tutorial_button.property("attention") is False
    assert settings.value("guided_tutorial_seen", False, type=bool) is True


def test_previous_button_rebuilds_state_before_selected_step(qt_application, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    gui.guided_tutorial_button.click()
    for _ in range(2):
        gui._tutorial_next_button.click()

    gui.applied_steps.clear()
    gui._tutorial_previous_button.click()

    assert gui._tutorial_step == 0
    assert gui._tutorial_phase == 1
    assert gui.applied_steps == []
    assert gui._tutorial_previous_button.isEnabled() is True


def test_completion_waits_for_finish_and_centers_dialog(qt_application, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)
    gui.resize(900, 650)
    gui.show()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    gui.guided_tutorial_button.click()

    gui._tutorial_step = len(TUTORIAL_STEPS) - 1
    gui._prepare_tutorial_step()
    qt_application.processEvents()

    assert gui._tutorial_dialog is not None
    assert gui._tutorial_heading.text() == "Tutorial complete"
    assert gui._tutorial_next_button.text() == "Finish"
    assert (gui._tutorial_dialog.frameGeometry().center() - gui.frameGeometry().center()).manhattanLength() <= 2
    assert gui.restored_snapshot is None
    gui._tutorial_next_button.click()
    assert gui.restored_snapshot == {"original": True}


def test_tutorial_places_cursor_on_next_after_dialog_moves(qt_application, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)
    gui.resize(900, 650)
    gui.show()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    positions = []
    monkeypatch.setattr(QCursor, "setPos", lambda point: positions.append(point))
    gui.guided_tutorial_button.click()
    qt_application.processEvents()

    assert positions[-1] == gui._tutorial_next_button.mapToGlobal(gui._tutorial_next_button.rect().center())
    gui._tutorial_next_button.click()
    qt_application.processEvents()
    assert positions[-1] == gui._tutorial_next_button.mapToGlobal(gui._tutorial_next_button.rect().center())
    gui._end_guided_tutorial()


def test_guided_tutorial_skip_and_decline(qt_application, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    gui.guided_tutorial_button.click()
    assert gui._tutorial_dialog is None
    assert gui.guided_tutorial_button.property("attention") is True

    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    gui.guided_tutorial_button.click()
    gui.guided_tutorial_button.click()
    assert gui._tutorial_dialog is None
    assert gui.guided_tutorial_button.property("attention") is False
    assert settings.value("guided_tutorial_seen", False, type=bool) is True


def test_tutorial_workspace_contains_browsable_premade_weka_results(tmp_path):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)

    GuidedTutorialMixin._prepare_tutorial_workspace(gui)

    assert gui._tutorial_input_dir is not None
    assert gui._tutorial_output_dir is not None
    assert gui._tutorial_model is not None
    assert gui._tutorial_preview_result is not None
    results = gui._tutorial_output_dir / "preview_1" / "Results"
    assert (gui._tutorial_input_dir / "Cellonaut_32bits.tif").is_file()
    assert gui._tutorial_model.parent.name == "Cellonaut_32bit_classifier"
    assert (gui._tutorial_model.parent / "Cellonaut_classifier_data.arff").is_file()
    assert (results / "CSV Data" / "Measurements.csv").is_file()
    assert gui._tutorial_preview_result.is_file()
    assert (results / "Overlays" / "TIFF Overlays" / "Cellonaut_32bits_Cellonaut_combined_overlay.tif").is_file()


def test_closing_tutorial_dialog_counts_as_skip(qt_application, tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)

    gui.guided_tutorial_button.click()
    assert gui._tutorial_dialog is not None
    gui._tutorial_dialog.close()

    assert gui._tutorial_dialog is None
    assert gui.guided_tutorial_button.text() == "Tutorial"
    assert settings.value("guided_tutorial_seen", False, type=bool) is True


def test_tutorial_builds_real_logo_pipeline_settings(qt_application, tmp_path, monkeypatch):
    actions = {step.action: index for index, step in enumerate(TUTORIAL_STEPS) if step.action}
    monkeypatch.setattr(CellonautMainWindow, "schedule_input_path_scan", lambda *args, **kwargs: None)
    gui = CellonautMainWindow()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    model = tmp_path / "classifier.model"
    model.write_bytes(b"model")
    gui._tutorial_input_dir = input_dir
    gui._tutorial_output_dir = output_dir
    gui._tutorial_model = model

    assert gui._apply_tutorial_step(0) is True
    assert gui.input_dir.get() == str(input_dir)

    assert gui._apply_tutorial_step(1) is True
    assert gui.output_dir.get() == str(output_dir)

    assert gui._apply_tutorial_step(2) is True
    definitions = gui.get_active_image_definitions()
    assert [item["name"] for item in definitions] == ["Cellonaut"]

    assert gui._apply_tutorial_step(3) is True
    definitions = gui.get_active_image_definitions()
    assert [item["name"] for item in definitions] == ["Cellonaut", "Cellonaut mask"]
    assert definitions[1]["classifier"] == ""

    assert gui._apply_tutorial_step(4) is True
    definitions = gui.get_active_image_definitions()
    assert definitions[1]["classifier"] == str(model)

    assert gui._apply_tutorial_step(5) is True
    definitions = gui.get_active_image_definitions()
    assert definitions[1]["probability_class_index"] == "1,2,3"

    assert gui._apply_tutorial_step(actions["cellpose_on"]) is True
    assert gui.get_active_image_definitions()[0]["analysis_cell_segmentation_enabled"] is True
    assert gui._apply_tutorial_step(actions["cellpose_diameter"]) is True
    assert str(gui.get_active_image_definitions()[0]["cell_diameter"]) == "30"
    assert gui._apply_tutorial_step(actions["cellpose_off"]) is True
    assert gui.get_active_image_definitions()[0]["analysis_cell_segmentation_enabled"] is False

    for add_action, key, step_type in (
        ("processing_add_before_mask", "image_processing_steps", "gaussian_blur"),
        ("processing_add_after_mask", "mask_processing_steps", "binary_fill_holes"),
        ("processing_add_before_measurement", "image_processing_steps", "rolling_ball_background"),
    ):
        assert gui._apply_tutorial_step(actions[add_action]) is True
        definition = gui.get_active_image_definitions()[1 if key == "mask_processing_steps" else 0]
        example = next(step for step in definition[key] if step["type"] == step_type)
        assert example["row_enabled"] is True
        if step_type == "gaussian_blur":
            assert example["params"]["gaussian_sigma"] == "1"
        elif step_type == "rolling_ball_background":
            assert example["params"]["bg_radii"] == "40"

    assert gui._apply_tutorial_step(actions["processing_disable_examples"]) is True
    definitions = gui.get_active_image_definitions()
    for key, step_type, definition_index in (
        ("image_processing_steps", "gaussian_blur", 0),
        ("mask_processing_steps", "binary_fill_holes", 1),
        ("image_processing_steps", "rolling_ball_background", 0),
    ):
        example = next(step for step in definitions[definition_index][key] if step["type"] == step_type)
        assert example["row_enabled"] is False

    assert gui._apply_tutorial_step(actions["relationship"]) is True
    definitions = gui.get_active_image_definitions()
    assert definitions[0]["mask_relationships"]["Cellonaut mask"] is True

    assert gui._apply_tutorial_step(actions["measurements"]) is True
    assert gui.measurement_options["area"] is True
    assert gui.measurement_options["mean"] is True
    assert gui.measurement_options["raw_intden"] is True
    assert gui.measurement_options["positive_area_in_cell"] is False
    assert "Measurements of configured masks within Cellpose cells are selected" not in gui.analysis_matrix_warning_label.text()

    gui._tutorial_preview_state = "idle"
    assert gui._apply_tutorial_step(actions["check"]) is True
    assert gui._tutorial_check_state == "simulated"
    assert gui._apply_tutorial_step(actions["preview"]) is True
    assert gui._tutorial_preview_state == "complete"
    assert gui._apply_tutorial_step(actions["run"]) is True
    assert gui._tutorial_run_state == "complete"


def test_measurement_warning_reads_committed_choices(qt_application, monkeypatch):
    actions = {step.action: index for index, step in enumerate(TUTORIAL_STEPS) if step.action}
    monkeypatch.setattr(CellonautMainWindow, "schedule_input_path_scan", lambda *args, **kwargs: None)
    gui = CellonautMainWindow()
    gui._tutorial_input_dir = guided_tutorial_module.get_resource_path(
        "cellonaut/data/assets/Cellonaut_32bit_input"
    )
    gui._tutorial_output_dir = guided_tutorial_module.get_resource_path(
        "cellonaut/data/assets/Cellonaut_32bit_output"
    )
    gui._tutorial_model = guided_tutorial_module.get_resource_path(
        "cellonaut/data/assets/Cellonaut_32bit_classifier/Cellonaut_classifier.model"
    )
    for step in (0, 1, 2, 3, 4, actions["relationship"], actions["measurements"]):
        gui._apply_tutorial_step(step)
    gui.measurement_options["positive_area_in_cell"] = True

    gui.update_analysis_matrix_warning_label()

    assert "Measurements of configured masks within Cellpose cells are selected" in gui.analysis_matrix_warning_label.text()
    assert gui.measurement_options["positive_area_in_cell"] is True
    gui.commit_gui_edits()
    gui.update_analysis_matrix_warning_label()
    assert "Measurements of configured masks within Cellpose cells are selected" not in gui.analysis_matrix_warning_label.text()
    assert gui.measurement_options["positive_area_in_cell"] is False


def test_tutorial_processing_previous_restores_prior_row_state(qt_application, monkeypatch):
    monkeypatch.setattr(CellonautMainWindow, "schedule_input_path_scan", lambda *args, **kwargs: None)
    gui = CellonautMainWindow()
    gui._tutorial_model = guided_tutorial_module.get_resource_path(
        "cellonaut/data/assets/Cellonaut_32bit_classifier/Cellonaut_classifier.model"
    )
    actions = {step.action: index for index, step in enumerate(TUTORIAL_STEPS) if step.action}
    for action in (
        "channel",
        "mask",
        "classifier",
        "processing_add_before_mask",
        "processing_add_after_mask",
        "processing_add_before_measurement",
        "processing_disable_examples",
    ):
        gui._apply_tutorial_step(actions[action])
    gui._undo_tutorial_step(actions["processing_disable_examples"])
    definitions = gui.get_active_image_definitions()
    assert all(
        next(step for step in definitions[index][key] if step["type"] == step_type)["row_enabled"]
        for key, step_type, index in (
            ("image_processing_steps", "gaussian_blur", 0),
            ("mask_processing_steps", "binary_fill_holes", 1),
            ("image_processing_steps", "rolling_ball_background", 0),
        )
    )
    gui._undo_tutorial_step(actions["processing_add_before_mask"])
    assert not any(
        step["type"] == "gaussian_blur"
        for step in gui.get_active_image_definitions()[0]["image_processing_steps"]
    )


def test_tutorial_expands_result_tree_and_opens_overlay(qt_application, monkeypatch):
    monkeypatch.setattr(CellonautMainWindow, "schedule_input_path_scan", lambda *args, **kwargs: None)
    gui = CellonautMainWindow()
    gui._prepare_tutorial_workspace()
    actions = {step.action: index for index, step in enumerate(TUTORIAL_STEPS) if step.action}

    gui._apply_tutorial_step(actions["show_results"])
    qt_application.processEvents()
    assert gui._browser_root_path == str(gui._tutorial_output_dir)
    gui._apply_tutorial_step(actions["open_reference_preview"])
    gui._apply_tutorial_step(actions["open_results"])
    qt_application.processEvents()
    assert gui.file_tree.isExpanded(gui.fs_model.index(str(gui._tutorial_reference_dir)))
    assert gui.file_tree.isExpanded(gui.fs_model.index(str(gui._tutorial_results_dir)))
    assert gui._browser_root_path == str(gui._tutorial_output_dir)

    gui._apply_tutorial_step(actions["open_overlay"])
    qt_application.processEvents()
    assert gui.preview_state.file_path == str(gui._tutorial_overlay_path)
    assert gui.file_tree.currentIndex() == gui.fs_model.index(str(gui._tutorial_overlay_path))

    gui._apply_tutorial_step(actions["show_layers"])
    assert gui.preview_inspector.isVisible() or not gui.preview_inspector.isHidden()
    indices = gui._tutorial_mask_layer_indices()
    original_colors = {index: gui.preview_state.overlay_colors[index] for index in indices.values()}
    gui._apply_tutorial_step(actions["change_layer_color"])
    assert {number: gui.preview_state.overlay_colors[index] for number, index in indices.items()} == {
        1: "#FFFFFF", 2: "#B0B0B0", 3: "#8000FF",
    }
    gui._undo_tutorial_step(actions["change_layer_color"])
    assert {index: gui.preview_state.overlay_colors[index] for index in indices.values()} == original_colors
    gui._apply_tutorial_step(actions["change_layer_color"])

    gui._apply_tutorial_step(actions["show_snapshot"])
    assert gui.preview_snapshot_toggle_button.isChecked()
    gui._apply_tutorial_step(actions["capture_snapshot"])
    assert gui._tutorial_saved_snapshot is not None
    assert gui._tutorial_saved_snapshot.is_file()
    gui._apply_tutorial_step(actions["inspect_snapshot"])
    assert gui.preview_state.file_path == str(gui._tutorial_saved_snapshot)
    assert gui._tutorial_temporary_files is not None
    gui._tutorial_temporary_files.cleanup()


def test_tutorial_highlight_moves_between_targets(qt_application, tmp_path):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)
    gui.resize(300, 180)
    first = QPushButton(gui)
    second = QPushButton(gui)
    first.setGeometry(20, 20, 100, 30)
    second.setGeometry(20, 60, 100, 30)
    gui.show()
    qt_application.processEvents()

    gui._highlight_tutorial_target(first)
    assert gui._tutorial_highlight_frame is not None
    assert gui._tutorial_highlight_frame.isVisible() is True
    assert gui._tutorial_highlight_frame.width() == first.width() + 8
    assert gui._tutorial_highlight_frame.height() == first.height() + 8
    gui._highlight_tutorial_target(second)
    assert gui._tutorial_highlighted_widget is second


def test_real_tutorial_targets_use_exact_visible_global_geometry(qt_application, tmp_path, monkeypatch):
    monkeypatch.setattr(CellonautMainWindow, "schedule_input_path_scan", lambda *args, **kwargs: None)
    gui = CellonautMainWindow()
    gui.resize(1400, 900)
    gui.show()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    model = tmp_path / "classifier.model"
    model.write_bytes(b"model")
    gui._tutorial_input_dir = input_dir
    gui._tutorial_output_dir = output_dir
    gui._tutorial_model = model
    for step in range(5):
        gui._apply_tutorial_step(step)

    classifier_row = gui.image_rows[1]
    assert classifier_row is not None
    classifier = classifier_row.classifier
    assert classifier is not None
    checks = (
        (1, "mask_classifier", classifier.edit),
        (3, "image_processing_group", gui.image_processing_group),
        (3, "mask_processing_group", gui.mask_processing_group),
        (4, "analysis_measurements_panel", gui.analysis_measurements_panel),
    )
    for section, name, expected_target in checks:
        gui._show_tutorial_section(section)
        qt_application.processEvents()
        target = gui._tutorial_target(name)
        assert target is not None
        assert target is expected_target
        gui.pipeline_scroll.ensureWidgetVisible(target, 24, 24)
        qt_application.processEvents()
        gui._highlight_tutorial_target(target)
        assert gui._tutorial_highlight_frame is not None
        expected_global = gui._tutorial_visible_global_rect(target).adjusted(-4, -4, 4, 4)
        actual_global = QRect(
            gui._tutorial_highlight_frame.mapToGlobal(QPoint(0, 0)),
            gui._tutorial_highlight_frame.size(),
        )
        assert actual_global == expected_global


def test_highlight_reacquires_a_rebuilt_step_target(qt_application, tmp_path):
    settings = QSettings(str(tmp_path / "tutorial.ini"), QSettings.Format.IniFormat)
    gui = TutorialHarness(settings)
    gui.resize(320, 180)
    first = QPushButton(gui)
    replacement = QPushButton(gui)
    first.setGeometry(20, 20, 100, 30)
    replacement.setGeometry(40, 80, 140, 30)
    gui.input_dir = first
    gui._tutorial_dialog = QDialog(gui)
    gui.show()
    qt_application.processEvents()

    gui._highlight_tutorial_target(first)
    gui.input_dir = replacement
    gui._refresh_tutorial_highlight_geometry()

    assert gui._tutorial_highlighted_widget is replacement
    assert gui._tutorial_highlight_frame is not None
    expected_global = gui._tutorial_visible_global_rect(replacement).adjusted(-4, -4, 4, 4)
    actual_global = QRect(
        gui._tutorial_highlight_frame.mapToGlobal(QPoint(0, 0)),
        gui._tutorial_highlight_frame.size(),
    )
    assert actual_global == expected_global


def test_previous_after_classifier_rebuilds_without_stale_qt_targets(qt_application, tmp_path, monkeypatch):
    monkeypatch.setattr(CellonautMainWindow, "schedule_input_path_scan", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    gui = CellonautMainWindow()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    model = tmp_path / "classifier.model"
    model.write_bytes(b"model")

    def prepare_workspace():
        gui._tutorial_input_dir = input_dir
        gui._tutorial_output_dir = output_dir
        gui._tutorial_model = model

    gui._prepare_tutorial_workspace = prepare_workspace
    gui.toggle_guided_tutorial()
    for _step in range(4):
        gui._tutorial_next_button.click()
        gui._tutorial_next_button.click()
    assert gui._tutorial_step == 4
    gui._tutorial_next_button.click()
    qt_application.processEvents()
    assert gui.get_active_image_definitions()[1]["classifier"] == str(model)

    gui._tutorial_previous_button.click()
    qt_application.processEvents()

    assert gui._tutorial_step == 4
    assert gui._tutorial_phase == 0
    assert gui.get_active_image_definitions()[1]["classifier"] == ""
    target = gui._tutorial_target("mask_classifier")
    assert target is not None
    assert gui._tutorial_highlighted_widget is target
