from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cellonaut.gui import presets as presets_gui
from cellonaut.gui.config import CellonautGuiConfigMixin


def test_read_json_file_rejects_non_object_presets(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a JSON object"):
        CellonautGuiConfigMixin()._read_json_file(path)


def test_read_json_file_preserves_json_decode_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        CellonautGuiConfigMixin()._read_json_file(path)


def test_read_json_file_reads_current_settings_keys(tmp_path):
    path = tmp_path / "preset.json"
    path.write_text(json.dumps({"exclusion_tag": "_ctrl_"}), encoding="utf-8")

    settings = CellonautGuiConfigMixin()._read_json_file(path)

    assert settings["exclusion_tag"] == "_ctrl_"


def test_preset_dirty_state_compares_workflow_content(monkeypatch):
    gui = CellonautGuiConfigMixin()
    current = {"input_dir": "C:/Data", "measurement_options": {"area": True}}
    monkeypatch.setattr(gui, "get_preset_dict", lambda: current)
    monkeypatch.setattr(gui, "log", lambda _message: None, raising=False)
    gui.remember_preset_state("Analysis", dict(current))

    assert gui.preset_has_unsaved_changes() is False

    current["measurement_options"]["area"] = False

    assert gui.preset_has_unsaved_changes() is True


def test_switch_save_replaces_removed_preset_metadata(tmp_path, monkeypatch):
    path = tmp_path / "Analysis.json"
    path.write_text(
        json.dumps({"input_dir": "C:/Old", "_preset_removed_field": "discard me"}),
        encoding="utf-8",
    )
    gui = CellonautGuiConfigMixin()
    monkeypatch.setattr(gui, "commit_gui_edits", lambda: None, raising=False)
    gui._active_preset_name = "Analysis"
    monkeypatch.setattr(gui, "preset_path_from_name", lambda _name: path)
    monkeypatch.setattr(gui, "get_preset_dict", lambda: {"input_dir": "C:/New"})
    monkeypatch.setattr(gui, "log", lambda _message: None, raising=False)

    assert gui.save_active_preset_changes() is True

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"input_dir": "C:/New"}
    assert gui.preset_has_unsaved_changes() is False


def test_save_selected_preset_updates_complete_workflow(tmp_path, monkeypatch):
    path = tmp_path / "Analysis.json"
    path.write_text(
        json.dumps({"input_dir": "C:/Old", "_preset_removed_field": "discard me"}),
        encoding="utf-8",
    )
    gui = CellonautGuiConfigMixin()
    monkeypatch.setattr(gui, "commit_gui_edits", lambda: None, raising=False)
    gui_with_widgets: Any = gui
    gui_with_widgets.preset_combo = type("PresetCombo", (), {"currentText": lambda self: "Analysis"})()
    monkeypatch.setattr(gui, "ensure_presets_dir", lambda: None)
    monkeypatch.setattr(gui, "preset_path_from_name", lambda _name: path)
    monkeypatch.setattr(gui, "get_preset_dict", lambda: {"input_dir": "C:/New"})
    monkeypatch.setattr(gui, "refresh_presets_list", lambda **_kwargs: None)
    monkeypatch.setattr(gui, "log", lambda _message: None, raising=False)
    monkeypatch.setattr("cellonaut.gui.presets.QMessageBox.information", lambda *_args, **_kwargs: None)

    gui.save_to_selected_preset()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"input_dir": "C:/New"}
    assert gui.preset_has_unsaved_changes() is False


def test_default_preset_save_routes_to_save_as(monkeypatch):
    gui = CellonautGuiConfigMixin()
    gui_with_widgets: Any = gui
    gui_with_widgets.preset_combo = type("PresetCombo", (), {"currentText": lambda self: "Default"})()
    gui._active_preset_name = "Default"
    save_as_calls = []
    monkeypatch.setattr(gui, "ensure_presets_dir", lambda: None)
    monkeypatch.setattr(gui, "save_preset_as", lambda: save_as_calls.append(True) or True)

    gui.save_to_selected_preset()
    assert gui.save_active_preset_changes() is True

    assert save_as_calls == [True, True]


def test_low_level_preset_save_rejects_default_name(tmp_path, monkeypatch):
    gui = CellonautGuiConfigMixin()
    monkeypatch.setattr(gui, "get_preset_dict", lambda: {"input_dir": "changed"})

    with pytest.raises(ValueError, match="Default preset is locked"):
        gui._save_preset_to_path(tmp_path / "Default.json", "Renamed", created=False)

    assert not (tmp_path / "Default.json").exists()


def test_save_as_rejects_reserved_default_name(monkeypatch):
    class DefaultNameDialog:
        def __init__(self, parent=None):
            self.parent = parent

        def exec(self):
            return presets_gui.QDialog.DialogCode.Accepted

        def get_name(self):
            return "default"

    gui = CellonautGuiConfigMixin()
    messages = []
    monkeypatch.setattr(gui, "ensure_presets_dir", lambda: None)
    monkeypatch.setattr(presets_gui, "PresetInfoDialog", DefaultNameDialog)
    monkeypatch.setattr(presets_gui.QMessageBox, "information", lambda _parent, _title, text: messages.append(text))

    assert gui.save_preset_as() is False
    assert messages and "reserved" in messages[0]


def test_saving_default_as_new_preset_preserves_default_file(tmp_path, monkeypatch):
    class NewNameDialog:
        def __init__(self, parent=None):
            pass

        def exec(self):
            return presets_gui.QDialog.DialogCode.Accepted

        def get_name(self):
            return "My analysis"

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    default_path = presets_dir / "Default.json"
    default_path.write_text('{"input_dir": "original"}', encoding="utf-8")
    gui = CellonautGuiConfigMixin()
    monkeypatch.setattr(gui, "commit_gui_edits", lambda: None, raising=False)
    gui_with_widgets: Any = gui
    gui_with_widgets.preset_combo = type("PresetCombo", (), {"currentText": lambda self: "Default"})()
    monkeypatch.setattr(presets_gui, "PRESETS_DIR", presets_dir)
    monkeypatch.setattr(presets_gui, "PresetInfoDialog", NewNameDialog)
    monkeypatch.setattr(gui, "ensure_presets_dir", lambda: None)
    monkeypatch.setattr(gui, "get_preset_dict", lambda: {"input_dir": "changed"})
    monkeypatch.setattr(gui, "refresh_presets_list", lambda **_kwargs: None)
    monkeypatch.setattr(gui, "log", lambda _message: None, raising=False)

    gui.save_to_selected_preset()

    assert json.loads(default_path.read_text(encoding="utf-8")) == {"input_dir": "original"}
    assert json.loads((presets_dir / "My analysis.json").read_text(encoding="utf-8")) == {
        "input_dir": "changed",
    }


def test_automatic_detection_updates_only_a_clean_preset_baseline(monkeypatch):
    gui = CellonautGuiConfigMixin()
    current = {"image_definitions": [{"name": "Channel 1"}]}
    monkeypatch.setattr(gui, "get_preset_dict", lambda: current)
    gui.remember_preset_state("Analysis", dict(current))

    current = {"image_definitions": [{"name": "GFP"}]}
    monkeypatch.setattr(gui, "get_preset_dict", lambda: current)
    gui.remember_automatic_preset_state_if_clean(True)
    assert gui.preset_has_unsaved_changes() is False

    current["image_definitions"][0]["name"] = "Edited by user"
    gui.remember_automatic_preset_state_if_clean(False)
    assert gui.preset_has_unsaved_changes() is True


def test_atomic_preset_write_preserves_existing_json_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "Preset.json"
    path.write_text('{"original": true}', encoding="utf-8")
    gui = CellonautGuiConfigMixin()

    temporary_paths = []

    def fail_publish(source, destination):
        temporary = Path(source)
        assert temporary != path and Path(destination) == path
        assert json.loads(temporary.read_text(encoding="utf-8")) == {"replacement": True}
        temporary_paths.append(temporary)
        raise OSError("publish failed")

    monkeypatch.setattr("cellonaut.io.writers.os.replace", fail_publish)

    with pytest.raises(OSError, match="publish failed"):
        gui._write_json_file(path, {"replacement": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"original": True}
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_default_preset_load_uses_workflow_application_path(tmp_path, monkeypatch):
    path = tmp_path / "Default.json"
    path.write_text(
        json.dumps(
            {
                "input_dir": "C:/Data",
                "output_dir": "C:/Results",
            }
        ),
        encoding="utf-8",
    )
    applied = []
    gui = CellonautGuiConfigMixin()
    monkeypatch.setattr(gui, "ensure_default_preset", lambda: None)
    monkeypatch.setattr(gui, "preset_path_from_name", lambda _name: path)
    monkeypatch.setattr(gui, "apply_preset_dict", applied.append)
    monkeypatch.setattr(gui, "refresh_presets_list", lambda **_kwargs: None)
    monkeypatch.setattr(gui, "log", lambda _message: None, raising=False)

    gui.load_default_preset_on_startup()

    assert applied == [{"input_dir": "C:/Data", "output_dir": "C:/Results"}]


def test_ensure_default_preset_recreates_empty_default(tmp_path, monkeypatch):
    path = tmp_path / "Default.json"
    path.write_text("", encoding="utf-8")
    bundled_path = tmp_path / "BundledDefault.json"
    bundled_default = {"input_dir": "", "measurement_options": {"area": True}}
    bundled_path.write_text(json.dumps(bundled_default), encoding="utf-8")
    gui = CellonautGuiConfigMixin()
    messages = []

    monkeypatch.setattr(gui, "ensure_presets_dir", lambda: None)
    monkeypatch.setattr(gui, "preset_path_from_name", lambda _name: path)
    monkeypatch.setattr("cellonaut.gui.presets.get_resource_path", lambda _path: bundled_path)
    monkeypatch.setattr(gui, "log", messages.append, raising=False)

    gui.ensure_default_preset()

    assert json.loads(path.read_text(encoding="utf-8")) == bundled_default
    assert any("Default preset was unreadable" in message for message in messages)
