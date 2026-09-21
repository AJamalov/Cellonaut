# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from cellonaut.gui.state import PreviewState  # noqa: E402
from cellonaut.config.state import ImageGuiState
from cellonaut.gui import dataset_analysis
from cellonaut.gui.dataset_analysis import CellonautGuiDatasetAnalysisMixin


class TextEditStub:
    def __init__(self, value: str):
        self.value = value
        self.focused = False

    def text(self) -> str:
        return self.value

    def setFocus(self):
        self.focused = True


class ControlStub:
    def __init__(self, *, visible=False, text="", checked=False):
        self.visible = visible
        self.value = text
        self.checked = checked
        self.enabled = True
        self.focused = False
        self.items = []
        self.index = 0

    def isVisible(self):
        return self.visible

    def setVisible(self, value):
        self.visible = bool(value)

    def setText(self, value):
        self.value = value

    def text(self):
        return self.value

    def setCurrentText(self, value):
        self.value = value
        matching_index = next(
            (index for index, item in enumerate(self.items) if value in item),
            None,
        )
        if matching_index is not None:
            self.index = matching_index

    def currentText(self):
        return self.value

    def setChecked(self, value):
        self.checked = bool(value)

    def setEnabled(self, value):
        self.enabled = bool(value)

    def isChecked(self):
        return self.checked

    def currentIndex(self):
        return self.index

    def setCurrentIndex(self, value):
        self.index = int(value)
        if 0 <= self.index < len(self.items):
            self.value = self.items[self.index][1]

    def blockSignals(self, _blocked):
        pass

    def clear(self):
        self.items = []
        self.index = 0

    def addItem(self, text, data=None):
        self.items.append((text, text if data is None else data))

    def count(self):
        return len(self.items)

    def currentData(self):
        return self.items[self.index][1] if self.items else self.value

    def findData(self, value):
        return next((index for index, item in enumerate(self.items) if item[1] == value), -1)

    def findChildren(self, _widget_type):
        return []

    def setFocus(self):
        self.focused = True

    def property(self, name):
        return getattr(self, name, None)


class AnalysisPanelHarness(CellonautGuiDatasetAnalysisMixin):
    def __init__(self):
        self.preview_state = PreviewState()
        self.image_definitions = [
            {
                "name": "GFP",
                "analysis_cell_segmentation_enabled": True,
                "analysis_cell_segmentation_source": "Brightfield",
                "mask_relationships": {"Mask": True},
                "cell_qc_limits": "Area:2-10",
                "mask_qc_limits": "MaskArea:1-8",
                "cell_qc_exclude_flagged": True,
                "qc_filter_mode": "Exclude only if both fail",
                "mask_qc_intensity_source": "Cell mask image",
            }
        ]
        self.image_definitions = [ImageGuiState.from_dict(item, i).to_dict() for i, item in enumerate(self.image_definitions)]
        self.measurement_options = {"area": True}
        self.analysis_measurements_panel = ControlStub()
        self.analysis_filters_panel = ControlStub()
        self.analysis_mask_adjust_panel = ControlStub()
        self.analysis_measurements_title = ControlStub()
        self.analysis_measurements_hint = ControlStub()
        self.analysis_filters_title = ControlStub()
        self.analysis_filters_hint = ControlStub()
        self.analysis_filter_source_label = ControlStub()
        self.analysis_mask_adjust_title = ControlStub()
        self.analysis_mask_adjust_hint = ControlStub()
        self.analysis_exclude_filtered_checkbox = ControlStub()
        self.analysis_filter_mode_combo = ControlStub()
        self.analysis_filter_target_mask_combo = ControlStub()
        self.analysis_mask_intensity_source_combo = ControlStub()
        self.analysis_population_combo = ControlStub()
        self.analysis_population_name_edit = ControlStub()
        self.analysis_remove_population_button = ControlStub()
        self.analysis_cell_advanced_filter_group = ControlStub()
        self.analysis_mask_advanced_filter_group = ControlStub()
        self._inline_analysis_settings_updating = False
        self._inline_analysis_settings_row = None
        self._inline_analysis_settings_panel = ""
        self._inline_measurement_checks = []
        self.calls = []

    def commit_gui_edits(self):
        self._read_measurement_edits()
        if getattr(self, "_inline_cell_filter_rows", None):
            self._read_inline_filter_edits(self.image_definitions)

    def get_active_image_definitions(self):
        return self.image_definitions

    def add_inline_measurement_group(self, layout_name, keys, options):
        self.calls.append(("measurement_group", layout_name, tuple(keys), dict(options)))

    def parse_cell_qc_limits_text(self, text):
        rules = {}
        for part in str(text).split(";"):
            if not part.strip():
                continue
            metric, bounds = part.split(":", 1)
            minimum, maximum = bounds.split("-", 1)
            rules[metric] = {"min": float(minimum), "max": float(maximum)}
        return rules

    def build_inline_filter_rows(self, layout_name, metrics, rules):
        return {
            metric: {
                "min_edit": TextEditStub(str(bounds.get("min", ""))),
                "max_edit": TextEditStub(str(bounds.get("max", ""))),
            }
            for metric, bounds in rules.items()
        }

    def display_filter_mode(self, mode):
        return mode

    def update_analysis_matrix_warning_label(self):
        self.calls.append("warnings")

    def normalize_image_definitions(self, definitions):
        return definitions

    def refresh_preview_filter_overlay(self):
        self.calls.append("refresh")


def test_measurement_toggle_saves_while_filter_editor_is_active():
    gui = AnalysisPanelHarness()
    gui._inline_analysis_settings_panel = "filters"
    gui._inline_analysis_settings_row = 0
    gui.measurement_options = {"area": True, "positive_area_in_cell": True}
    checkbox = ControlStub(checked=False)
    checkbox.measurementKey = "positive_area_in_cell"
    gui._inline_measurement_checks = [checkbox]

    gui.on_inline_measurement_changed(False)

    assert gui.measurement_options == {"area": True, "positive_area_in_cell": False}
    assert gui.calls == ["warnings"]


def test_inline_filter_rows_convert_bounds_and_reject_inverted_ranges():
    gui = CellonautGuiDatasetAnalysisMixin()
    gui.preview_state = PreviewState()
    rows = {
        "Area": {"min_edit": TextEditStub("2.5"), "max_edit": TextEditStub("10")},
        "Mean": {"min_edit": TextEditStub(""), "max_edit": TextEditStub("8")},
        "Circularity": {"min_edit": TextEditStub(""), "max_edit": TextEditStub("")},
    }

    assert gui.rules_from_inline_filter_rows(rows) == {
        "Area": {"min": 2.5, "max": 10.0},
        "Mean": {"max": 8.0},
    }

    rows["Area"] = {"min_edit": TextEditStub("11"), "max_edit": TextEditStub("10")}
    with pytest.raises(ValueError, match="Area: minimum cannot be greater than maximum"):
        gui.rules_from_inline_filter_rows(rows)


def test_inline_fraction_filter_accepts_percentage_notation():
    gui = CellonautGuiDatasetAnalysisMixin()
    gui.preview_state = PreviewState()
    rows = {
        "Mask fraction of cell area": {
            "min_edit": TextEditStub("50%"),
            "max_edit": TextEditStub("75%"),
        }
    }

    assert gui.rules_from_inline_filter_rows(rows) == {"Mask fraction of cell area": {"min": 0.5, "max": 0.75}}


def test_inline_filter_edit_saves_without_refreshing_preview():
    gui = CellonautGuiDatasetAnalysisMixin()
    gui.preview_state = PreviewState()
    refreshed: list[bool] = []
    gui.refresh_preview_filter_overlay = lambda: refreshed.append(True)
    saved: list[bool] = []
    gui.on_inline_analysis_settings_changed = lambda **kwargs: saved.append(kwargs["refresh_preview"])

    gui.on_inline_filter_text_changed()
    assert saved == [False]
    assert refreshed == []


def test_export_current_filter_preview_writes_filtered_and_report_tables(monkeypatch, tmp_path):
    table_path = tmp_path / "Results" / "CSV Data" / "Cell Measurements" / "sample_cells.csv"
    table_path.parent.mkdir(parents=True)
    table = pd.DataFrame({"CellID": [1, 2], "Area": [4, 12]})
    gui = CellonautGuiDatasetAnalysisMixin()
    gui.preview_state = PreviewState()
    gui.preview_state.filter_data = SimpleNamespace(table=table, table_path=table_path)
    gui._inline_analysis_settings_row = 0
    gui.get_active_image_definitions = lambda: [{"name": "GFP"}]
    gui.commit_gui_edits = lambda: None
    gui.parse_cell_qc_limits_text = lambda _text: {}
    notices = []
    monkeypatch.setattr(
        dataset_analysis,
        "filtered_table_from_current_rules",
        lambda *_args, **_kwargs: (table.iloc[:1], table.assign(Excluded=[False, True]), {2}, {"total": 2}),
    )
    monkeypatch.setattr(dataset_analysis.QMessageBox, "information", lambda *args: notices.append(args))

    gui.export_current_filter_preview_csv()

    export_dir = tmp_path / "Results" / "Image Preview Tools" / "Cell Groups" / "export_1"
    assert (export_dir / "sample_cells_filtered_current.csv").exists()
    assert (export_dir / "sample_cells_filter_report_current.csv").exists()
    assert "Kept 1 / 2 cells. Excluded 1." in notices[0][2]


def test_export_current_filter_preview_uses_latest_inline_settings(monkeypatch, tmp_path):
    table_path = tmp_path / "Results" / "sample.csv"
    table_path.parent.mkdir()
    gui = CellonautGuiDatasetAnalysisMixin()
    gui.preview_state = PreviewState()
    gui.preview_state.filter_data = SimpleNamespace(table=pd.DataFrame({"CellID": [1]}), table_path=table_path)
    gui._inline_analysis_settings_row = 0
    definitions = [{"name": "GFP", "cell_qc_limits": "old"}]
    gui.get_active_image_definitions = lambda: definitions
    gui.commit_gui_edits = lambda: definitions.__setitem__(
        0, {"name": "GFP", "cell_qc_limits": "new"}
    )
    gui.parse_cell_qc_limits_text = lambda _text: {}
    captured = []
    monkeypatch.setattr(
        dataset_analysis,
        "filtered_table_from_current_rules",
        lambda table, image_def, _parser, **_kwargs: (
            captured.append(image_def) or table, table, set(), {"total": 1}
        ),
    )
    monkeypatch.setattr(dataset_analysis.QMessageBox, "information", lambda *_args: None)

    gui.export_current_filter_preview_csv()

    assert captured[0]["cell_qc_limits"] == "new"


def test_export_current_filter_preview_rejects_invalid_inline_rule(monkeypatch, tmp_path):
    gui = CellonautGuiDatasetAnalysisMixin()
    gui.preview_state = PreviewState()
    gui.preview_state.filter_data = SimpleNamespace(table=pd.DataFrame({"CellID": [1]}), table_path=tmp_path / "cells.csv")
    gui._inline_analysis_settings_row = 0
    gui._inline_cell_filter_rows = {"Area": object()}
    gui.rules_from_inline_filter_rows = lambda rows: (_ for _ in ()).throw(ValueError("invalid Area")) if rows else {}
    gui.commit_gui_edits = lambda: (_ for _ in ()).throw(AssertionError("must not save"))
    notices = []
    monkeypatch.setattr(dataset_analysis.QMessageBox, "critical", lambda *args: notices.append(args))

    gui.export_current_filter_preview_csv()

    assert "invalid Area" in notices[0][2]


def test_export_all_filter_previews_rejects_blank_destination(monkeypatch):
    gui = CellonautGuiDatasetAnalysisMixin()
    gui.preview_state = PreviewState()
    gui.preview_state.file_path = ""
    gui.output_dir = SimpleNamespace(get=lambda: "")
    gui.get_active_image_definitions = lambda: [{"name": "GFP"}]
    gui.commit_gui_edits = lambda: None
    notices = []
    monkeypatch.setattr(dataset_analysis.QMessageBox, "information", lambda *args: notices.append(args))
    monkeypatch.setattr(
        dataset_analysis,
        "export_all_filtered_result_tables",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not export to cwd")),
    )

    gui.export_all_filter_preview_csvs()

    assert "Select an output folder" in notices[0][2]


def test_export_all_filter_previews_uses_open_results_tree(monkeypatch, tmp_path):
    preview_file = tmp_path / "Results" / "Processing Montages" / "sample.png"
    preview_file.parent.mkdir(parents=True)
    preview_file.write_bytes(b"preview")
    gui = CellonautGuiDatasetAnalysisMixin()
    gui.preview_state = PreviewState()
    gui.preview_state.file_path = str(preview_file)
    gui.get_active_image_definitions = lambda: [{"name": "GFP"}]
    gui.commit_gui_edits = lambda: None
    gui.parse_cell_qc_limits_text = lambda _text: {}
    gui.output_dir = SimpleNamespace(get=lambda: "")
    captured = {}
    notices = []
    result = SimpleNamespace(
        export_dir=tmp_path / "export",
        processed_count=1,
        skipped_count=0,
        kept_count=3,
        total_count=4,
        kept_combined_path=tmp_path / "kept.csv",
        report_combined_path=tmp_path / "report.csv",
        all_measurements_path=None,
        simple_measurements_path=None,
        readable_measurements_path=None,
        group_membership_path=None,
        messages=["one table had no summary rows"],
    )

    def export(root, active_defs, _parser):
        captured["root"] = root
        captured["defs"] = active_defs
        return result

    monkeypatch.setattr(dataset_analysis, "export_all_filtered_result_tables", export)
    monkeypatch.setattr(dataset_analysis.QMessageBox, "information", lambda *args: notices.append(args))

    gui.export_all_filter_preview_csvs()

    assert captured["root"] == tmp_path / "Results"
    assert captured["defs"] == [{"name": "GFP"}]
    assert "Processed tables: 1" in notices[0][2]
    assert "one table had no summary rows" in notices[0][2]


def test_analysis_filter_panel_loads_channel_context_and_existing_rules():
    gui = AnalysisPanelHarness()

    gui.show_analysis_settings_panel(0, "filters")

    assert gui.analysis_filters_panel.visible is True
    assert gui.analysis_measurements_panel.visible is True
    assert gui.analysis_filters_title.value == "GFP cell groups"
    assert gui.analysis_filter_source_label.value == "Cellpose source: Brightfield · Cell-group target: Mask"
    assert gui.analysis_filter_target_mask_combo.currentData() == "Mask"
    assert gui.analysis_exclude_filtered_checkbox.checked is True
    assert gui.analysis_filter_mode_combo.value == "Exclude only if both fail"
    assert gui.analysis_mask_intensity_source_combo.value == "Cellpose source channel"
    assert gui._inline_cell_filter_rows["Area"]["min_edit"].text() == "2.0"
    assert gui._inline_mask_filter_rows["MaskArea"]["max_edit"].text() == "8.0"
    assert len([call for call in gui.calls if isinstance(call, tuple) and call[0] == "measurement_group"]) == 3


def test_analysis_filter_panel_uses_explicit_target_when_multiple_masks_are_enabled():
    gui = AnalysisPanelHarness()
    gui.image_definitions[0]["mask_relationships"] = {"Mask A": True, "Mask B": True}
    gui.image_definitions[0]["cell_group_mask_source"] = "Mask B"

    gui.show_analysis_settings_panel(0, "filters")

    assert gui.analysis_filter_target_mask_combo.currentData() == "Mask B"
    gui.analysis_filter_target_mask_combo.setCurrentText("Mask A")
    gui.on_inline_analysis_settings_changed()
    assert gui.image_definitions[0]["cell_group_mask_source"] == "Mask A"


def test_inline_filter_changes_update_definition_without_refreshing_preview():
    gui = AnalysisPanelHarness()
    gui.show_analysis_settings_panel(0, "filters")
    gui._inline_cell_filter_rows = {"Area": {"min_edit": TextEditStub("3"), "max_edit": TextEditStub("12")}}
    gui._inline_mask_filter_rows = {"MaskArea": {"min_edit": TextEditStub("2"), "max_edit": TextEditStub("9")}}
    gui.analysis_exclude_filtered_checkbox.setChecked(False)
    gui.analysis_filter_mode_combo.setCurrentText("Exclude if any filter fails")
    gui.analysis_filter_target_mask_combo.clear()
    gui.analysis_filter_target_mask_combo.addItem("Mask", "Mask")
    gui.analysis_mask_intensity_source_combo.setCurrentText("Measured channel")

    gui.on_inline_analysis_settings_changed()

    image_def = gui.image_definitions[0]
    group = image_def["cell_populations"][0]
    assert group["cell_qc_limits"] == "Area:3-12"
    assert group["mask_qc_limits"] == "MaskArea:2-9"
    assert group["exclude_from_csv"] is False
    assert group["qc_filter_mode"] == "Exclude if any filter fails"
    assert group["mask_qc_intensity_source"] == "Measured image"
    assert image_def["cell_group_mask_source"] == "Mask"
    assert "refresh" not in gui.calls


def test_cell_groups_empty_state_hides_controls_and_restores_them():
    panel = AnalysisPanelHarness()
    names = (
        "analysis_population_controls", "analysis_filter_controls_group",
        "analysis_filter_export_group", "analysis_active_filter_count_label",
        "analysis_cell_filter_host", "analysis_mask_filter_host",
        "analysis_filter_empty_label",
    )
    for name in names:
        setattr(panel, name, ControlStub())
    controls = names[:-1] + ("analysis_filters_title", "analysis_filter_source_label")
    panel.preview_state.current_layer_roles = ["cellpose_mask"]
    panel.update_inline_filter_category_visibility(True, True)
    assert all(getattr(panel, name).visible for name in controls)
    assert not panel.analysis_filter_empty_label.visible

    panel.preview_state.current_layer_roles = []
    panel.update_inline_filter_category_visibility(True, True)
    assert all(not getattr(panel, name).visible for name in controls)
    assert not panel.analysis_filters_hint.visible
    assert panel.analysis_filter_empty_label.visible
    assert "No cell masks are available" in panel.analysis_filter_empty_label.value

    panel.preview_state.current_layer_roles = ["cellpose_mask"]
    panel.update_inline_filter_category_visibility(True, True)
    assert all(getattr(panel, name).visible for name in controls)
    assert not panel.analysis_filter_empty_label.visible
