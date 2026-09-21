from __future__ import annotations

from pathlib import Path

import cellonaut.gui.nd2 as nd2_gui
from cellonaut.gui.nd2 import CellonautGuiNd2Mixin
from cellonaut.gui.state import GuiTaskState
from cellonaut.gui.validation import CellonautGuiValidationMixin


class _PathValue:
    def __init__(self, value: str):
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str):
        self.value = value


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, text: str):
        self.text = text


class DummyNd2Gui(CellonautGuiNd2Mixin):
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = _PathValue(str(input_dir))
        self.nd2_output_dir = _PathValue(str(output_dir))
        self.nd2_detected_channel_names = []
        self.nd2_channel_folder_state = {}
        self.nd2_detected_channels_label = _Label()
        self.nd2_z_mode = _ComboValue("Max projection")
        self.nd2_z_index_spin = _SpinValue(1)
        self.nd2_z_index_row = _Button()

    def update_nd2_visibility(self):
        pass

    def rebuild_nd2_channel_rows(self):
        pass

    def log(self, _message: str):
        pass


class DummySetupGui(CellonautGuiValidationMixin):
    def __init__(self, input_dir: Path):
        self.nd2_detected_channel_names = []
        self._input_dir = input_dir


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real nd2, only a path fixture")


def test_collect_nd2_config_rescans_nested_nd2_channels_when_empty(tmp_path: Path, monkeypatch):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    touch(source / "ExperimentA" / "sample_001.nd2")

    monkeypatch.setattr(
        nd2_gui.ndi,
        "inspect_nd2_file",
        lambda _path: {"name": "sample_001.nd2", "channel_names": ["GFP", "DAPI"]},
    )

    gui = DummyNd2Gui(source, output)
    cfg = gui.collect_nd2_config()

    assert gui.nd2_detected_channel_names == ["GFP", "DAPI"]
    assert cfg.channel_map == {"GFP": 0, "DAPI": 1}


def test_nd2_detection_error_reports_nested_files_when_channels_are_missing(tmp_path: Path):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    touch(source / "ExperimentA" / "sample_001.nd2")
    gui = DummyNd2Gui(source, output)

    message = str(gui.nd2_channel_detection_error(source))

    assert "Found 1 ND2 file(s)" in message
    assert "ExperimentA" in message


def test_nd2_scan_populates_the_conversion_mapping(tmp_path: Path):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    first_file = source / "ExperimentA" / "sample_001.nd2"
    touch(first_file)
    gui = DummyNd2Gui(source, output)

    gui.apply_nd2_scan_result(
        {
            "path": str(source),
            "nd2_count": 1,
            "nd2_first": str(first_file),
            "nd2_channels": ["GFP", "DAPI"],
            "nd2_channel_colors": ["#00FF00", "#0000FF"],
            "nd2_sizes": {"C": 2, "Z": 1},
        }
    )

    assert gui.nd2_detected_channel_names == ["GFP", "DAPI"]
    assert gui.nd2_channel_folder_state == {"GFP": "GFP", "DAPI": "DAPI"}


def test_setup_check_reports_nested_nd2_files(tmp_path: Path):
    source = tmp_path / "input"
    touch(source / "ExperimentA" / "sample_001.nd2")
    gui = DummySetupGui(source)

    checks = gui.build_setup_check_items(
        cfg=type(
            "Cfg",
            (),
            {
                "input_dir": source,
                "output_dir": tmp_path / "out",
                "fiji_app_path": tmp_path,
            },
        )(),
        active_defs=[{"name": "GFP", "classifier": ""}],
        relationships=[],
        warnings=[],
        dry_run={"available": True, "total": 0, "ready": 0, "blocked": 0},
    )

    nd2_check = next(item for item in checks if item["check"] == "ND2 files")
    assert nd2_check["status"] == "WARNING"
    assert "1 ND2 file detected" in nd2_check["detail"]


class _ComboValue:
    def __init__(self, value: str):
        self.value = value
        self.enabled = True
        self.visible = True
        self.tooltip = ""

    def get(self) -> str:
        return self.value

    def set(self, value: str):
        self.value = str(value)

    def setEnabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def setVisible(self, visible: bool):
        self.visible = bool(visible)

    def setToolTip(self, text: str):
        self.tooltip = str(text)


class _SpinValue:
    def __init__(self, value: int = 1):
        self._value = int(value)
        self.enabled = True
        self.range = (1, 10000)
        self.tooltip = ""

    def value(self) -> int:
        return self._value

    def setValue(self, value: int):
        self._value = int(value)

    def setRange(self, low: int, high: int):
        self.range = (int(low), int(high))
        self._value = max(int(low), min(self._value, int(high)))

    def setEnabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def setToolTip(self, text: str):
        self.tooltip = text


class _Button:
    def __init__(self):
        self.enabled = True
        self.visible = True

    def setEnabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def setVisible(self, visible: bool):
        self.visible = bool(visible)


class _Progress:
    def __init__(self):
        self.value = 0

    def setValue(self, value: int):
        self.value = int(value)


def test_collect_nd2_config_uses_selected_single_z_slice(tmp_path: Path, monkeypatch):
    source = tmp_path / "input"
    output = tmp_path / "converted"
    touch(source / "ExperimentA" / "sample_001.nd2")

    monkeypatch.setattr(
        nd2_gui.ndi,
        "inspect_nd2_file",
        lambda _path: {
            "name": "sample_001.nd2",
            "channel_names": ["GFP", "DAPI"],
            "sizes": {"Z": 5, "C": 2, "Y": 32, "X": 32},
        },
    )

    gui = DummyNd2Gui(source, output)
    gui.nd2_z_mode = _ComboValue("Single Z slice")
    gui.nd2_z_index_spin = _SpinValue(3)
    gui.nd2_z_index_row = _Button()

    cfg = gui.collect_nd2_config()

    assert cfg.z_mode == "single_z"
    assert cfg.z_index == 2
    assert gui.nd2_z_index_spin.range == (1, 5)


def test_nd2_status_summary_reports_z_time_position_and_rgb_counts(tmp_path: Path):
    gui = DummyNd2Gui(tmp_path, tmp_path / "converted")
    gui.nd2_z_mode = _ComboValue("Max projection")
    gui.nd2_z_index_spin = _SpinValue(1)
    gui.nd2_z_index_row = _Button()
    gui.nd2_detected_channel_names = ["GFP", "DAPI"]
    gui.remember_nd2_scan_summary(
        nd2_count=4,
        first_display="ExperimentA/sample_001.nd2",
        sizes={"P": 2, "S": 3, "T": 3, "Z": 7, "C": 2, "Y": 64, "X": 64},
        z_summary={
            "inspected_count": 4,
            "stack_count": 2,
            "single_plane_count": 2,
            "min_stack_depth": 5,
            "max_stack_depth": 7,
            "error_count": 0,
            "complete": True,
        },
    )
    gui.refresh_nd2_status_summary()

    assert "ND2 files detected: 4" in gui.nd2_detected_channels_label.text
    assert "Channels: 2" in gui.nd2_detected_channels_label.text
    assert "Z stacks: 2 file(s), 5-7 slices; 2 single-plane file(s); exporting max projection" in (
        gui.nd2_detected_channels_label.text
    )
    assert "Timepoints: 3; exporting first" in gui.nd2_detected_channels_label.text
    assert "Positions: 2; exporting first" in gui.nd2_detected_channels_label.text
    assert "RGB components: 3; exporting first component" in gui.nd2_detected_channels_label.text
    assert "Selected for export" not in gui.nd2_detected_channels_label.text
    assert "Status:" not in gui.nd2_detected_channels_label.text
    assert "Pipeline channels" not in gui.nd2_detected_channels_label.text
    assert "After conversion" not in gui.nd2_detected_channels_label.text


def test_z_controls_are_hidden_when_every_nd2_file_is_single_plane(tmp_path: Path):
    gui = DummyNd2Gui(tmp_path, tmp_path / "converted")
    gui.remember_nd2_scan_summary(
        nd2_count=3,
        first_display="sample.nd2",
        sizes={"C": 2, "Y": 64, "X": 64},
        z_summary={
            "inspected_count": 3,
            "stack_count": 0,
            "single_plane_count": 3,
            "min_stack_depth": 0,
            "max_stack_depth": 0,
            "error_count": 0,
            "complete": True,
        },
    )

    assert gui.nd2_z_mode.visible is False
    assert gui.nd2_z_index_row.visible is False


def test_single_z_slice_is_limited_to_the_shallowest_stack(tmp_path: Path):
    gui = DummyNd2Gui(tmp_path, tmp_path / "converted")
    gui.nd2_z_mode.set("Single Z slice")
    gui.remember_nd2_scan_summary(
        nd2_count=3,
        first_display="sample.nd2",
        sizes={"Z": 7, "C": 2, "Y": 64, "X": 64},
        z_summary={
            "inspected_count": 3,
            "stack_count": 2,
            "single_plane_count": 1,
            "min_stack_depth": 4,
            "max_stack_depth": 7,
            "error_count": 0,
            "complete": True,
        },
    )

    assert gui.nd2_z_mode.visible is True
    assert gui.nd2_z_index_row.visible is True
    assert gui.nd2_z_index_spin.range == (1, 4)


class DummyNd2DoneGui(CellonautGuiNd2Mixin):
    def __init__(self, input_dir: Path, output_dir: Path):
        self.task_state = GuiTaskState()
        self.input_dir = _PathValue(str(input_dir))
        self.nd2_output_dir = _PathValue(str(output_dir))
        self.status_label = _Label()
        self.current_sample_label = _Label()
        self.progress_bar = _Progress()
        self.run_button = _Button()
        self.preview_pipeline_button = _Button()
        self.cancel_button = _Button()
        self.nd2_progress_panel = _Button()
        self.nd2_progress_label = _Label()
        self.nd2_progress_bar = _Progress()
        self.nd2_cancel_button = _Button()
        self.nd2_open_output_button = _Button()
        self.logs = []
        self.browser_root = ""

    def set_status_style(self, _status: str):
        pass

    def log(self, message: str):
        self.logs.append(message)

    def set_browser_root(self, value: str):
        self.browser_root = value

    def set_pipeline_busy(self, _busy: bool):
        pass

    def set_cancel_button_active(self, active: bool):
        self.cancel_button.setVisible(active)
        self.cancel_button.setEnabled(active)

    def set_primary_task_active(self, active: bool):
        self.set_cancel_button_active(active)


def test_nd2_conversion_completion_keeps_pipeline_input_unchanged(tmp_path: Path):
    source = tmp_path / "raw_nd2"
    output = tmp_path / "converted"
    gui = DummyNd2DoneGui(source, output)

    gui.on_nd2_import_done({"processed": 1, "total": 1, "failed": 0})

    assert gui.input_dir.get() == str(source)
    assert gui.browser_root == str(output)
    assert gui.status_label.text == "ND2 conversion done"
    assert gui.nd2_progress_label.text == (
        "Conversion complete.\n"
        "Converted: 1 | Skipped: 0 | Failed: 0\n"
        f"Saved to: {output}\n"
        "Next: select this converted TIFF folder as the pipeline Input directory."
    )
    assert gui.nd2_progress_bar.value == 100
    assert gui.nd2_cancel_button.visible is False
    assert gui.nd2_open_output_button.visible is True


def test_partial_nd2_conversion_does_not_switch_pipeline_input(tmp_path: Path):
    source = tmp_path / "raw_nd2"
    output = tmp_path / "converted"
    gui = DummyNd2DoneGui(source, output)

    gui.on_nd2_import_done(
        {"processed": 1, "total": 2, "failed": 1},
        completed_with_errors=True,
    )

    assert gui.input_dir.get() == str(source)
    assert gui.browser_root == str(output)
    assert any("failed to convert" in message for message in gui.logs)


def test_input_dir_contains_nd2_files_uses_shallow_fallback(tmp_path: Path, monkeypatch):
    nested = tmp_path / "nested"
    touch(nested / "sample_001.nd2")
    gui = DummyNd2Gui(tmp_path, tmp_path / "converted")

    def fail_recursive_scan(_root):
        raise AssertionError("recursive scan should stay in the background")

    monkeypatch.setattr(nd2_gui.ndi, "list_nd2_files", fail_recursive_scan)

    assert gui.input_dir_contains_nd2_files() is False

    touch(tmp_path / "direct.nd2")
    assert gui.input_dir_contains_nd2_files() is True


def test_setup_check_warns_when_weka_uses_fiji_without_imagescience(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    fiji_path = tmp_path / "Fiji.app"
    (fiji_path / "jars").mkdir(parents=True)
    (fiji_path / "plugins").mkdir()
    (fiji_path / "macros").mkdir()
    (fiji_path / "fiji").write_text("", encoding="utf-8")
    classifier = tmp_path / "classifier.model"
    classifier.write_text("model fixture", encoding="utf-8")
    gui = DummySetupGui(source)

    checks = gui.build_setup_check_items(
        cfg=type(
            "Cfg",
            (),
            {
                "input_dir": source,
                "output_dir": tmp_path / "out",
                "fiji_app_path": fiji_path,
            },
        )(),
        active_defs=[
            {
                "name": "Hmg2 mask",
                "classifier": str(classifier),
                "mask_source_mode": "Weka classifier",
            }
        ],
        relationships=[],
        warnings=[],
        dry_run={"available": True, "total": 0, "ready": 0, "blocked": 0},
    )

    image_science_check = next(item for item in checks if item["check"] == "Fiji ImageScience")
    assert image_science_check["status"] == "WARNING"
    assert "must be installed separately" in image_science_check["detail"]
    assert "2D Weka classifier" in image_science_check["detail"]
    assert "Derivatives, Laplacian, and Structure features" in image_science_check["detail"]
