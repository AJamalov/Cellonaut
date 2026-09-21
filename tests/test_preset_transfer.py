from __future__ import annotations

import json
from pathlib import Path

from cellonaut.gui import presets as presets_gui
from cellonaut.gui.config import CellonautGuiConfigMixin, private_path_tail, privacy_safe_preset


class FakeCombo:
    def __init__(self, text: str):
        self.text = text

    def currentText(self) -> str:
        return self.text


class PresetTransferHarness(CellonautGuiConfigMixin):
    def __init__(self, presets_dir: Path):
        self.presets_dir = presets_dir
        self.preset_combo = FakeCombo("Shared")
        self.loaded = []
        self.messages = []

    def ensure_presets_dir(self):
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def commit_gui_edits(self):
        pass

    def get_preset_dict(self):
        return {
            "input_dir": r"C:\Users\Researcher\Private\TIFF",
            "output_dir": r"D:\Study\Results",
            "image_definitions": [{"folder_name": "GFP", "classifier": r"D:\Models\Golgi.model"}],
            "panel_notes": {"setup": "Use the GFP channel for this workflow."},
        }

    def refresh_presets_list(self, select_name=None):
        if select_name:
            self.preset_combo.text = select_name

    def load_selected_preset(self):
        self.loaded.append(self.preset_combo.currentText())
        return True

    def log(self, message: str):
        self.messages.append(message)


def test_private_path_tail_handles_windows_posix_and_relative_paths():
    assert private_path_tail(r"C:\Users\Researcher\Experiments\Dataset 01") == "Dataset 01"
    assert private_path_tail("/home/researcher/models/classifier.model") == "classifier.model"
    assert private_path_tail("relative-folder") == "relative-folder"
    assert private_path_tail("") == ""


def test_privacy_safe_preset_removes_parent_paths_recursively_without_mutating_source():
    original = {
        "input_dir": r"C:\Users\Researcher\Private Study\TIFF",
        "output_dir": "/mnt/secret/results",
        "mask_source_dir": r"\\server\private\Masks",
        "image_definitions": [
            {
                "folder_name": "GFP",
                "classifier": r"D:\Models\Golgi.model",
                "cellpose_custom_model_path": r"C:\Models\custom.pt",
            }
        ],
    }

    exported = privacy_safe_preset(original)

    assert exported["input_dir"] == "TIFF"
    assert exported["output_dir"] == "results"
    assert exported["mask_source_dir"] == "Masks"
    image = exported["image_definitions"][0]
    assert image["folder_name"] == "GFP"
    assert image["classifier"] == "Golgi.model"
    assert image["cellpose_custom_model_path"] == "custom.pt"
    assert "Parent folders were removed" in exported["_preset_privacy"]
    assert original["input_dir"].startswith("C:")


def test_export_and_import_actions_round_trip_sanitized_preset(monkeypatch, tmp_path: Path):
    presets_dir = tmp_path / "app-data" / "presets"
    export_path = tmp_path / "Shared preset.json"
    harness = PresetTransferHarness(presets_dir)
    monkeypatch.setattr(presets_gui, "PRESETS_DIR", presets_dir)
    monkeypatch.setattr(presets_gui.QMessageBox, "information", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        presets_gui.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(export_path), "Cellonaut presets (*.json)"),
    )

    harness.export_selected_preset()

    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["input_dir"] == "TIFF"
    assert exported["output_dir"] == "Results"
    assert exported["image_definitions"][0]["classifier"] == "Golgi.model"
    assert exported["panel_notes"] == {"setup": "Use the GFP channel for this workflow."}

    monkeypatch.setattr(
        presets_gui.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(export_path), "Cellonaut presets (*.json)"),
    )
    harness.import_preset()

    imported_path = presets_dir / "Shared.json"
    assert imported_path.is_file()
    imported = json.loads(imported_path.read_text(encoding="utf-8"))
    assert imported["input_dir"] == "TIFF"
    assert imported["panel_notes"] == {"setup": "Use the GFP channel for this workflow."}
    assert not any(key.startswith("_preset_") for key in imported)
    assert harness.loaded == ["Shared"]
