# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import tifffile

import cellonaut.gui.file_browser as file_browser_gui
from cellonaut.pipeline import discovery
from cellonaut.config.defaults import default_image_definition
from cellonaut.gui.file_browser import (
    InputPathScanWorker,
    CellonautGuiFileBrowserMixin,
    _limited_background_input_detection,
    input_path_key,
    open_folder_in_system_browser,
    scan_input_path,
)
from cellonaut.pipeline.discovery import (
    contains_supported_tiff,
    safe_iterdir,
    scan_child_dirs,
    scan_matching_files,
)
from cellonaut.gui.main_window import CellonautMainWindow
from cellonaut.runtime import PipelineStage


pytestmark = pytest.mark.gui


class DummyFileBrowser(CellonautGuiFileBrowserMixin):
    def __init__(self):
        self.image_definitions = []
        self.logs = []
        self.status_updates = []

    def get_expected_image_folders(self) -> list[str]:
        return []

    def get_active_image_definitions(self):
        return list(self.image_definitions)

    def create_default_image_definition(self, index: int):
        return default_image_definition(index)

    def is_physical_channel_definition(self, image_def: dict) -> bool:
        return not bool(image_def.get("is_mask_only", False))

    def normalize_image_definitions(self, definitions):
        return list(definitions)

    def rebuild_image_rows(self, sync_from_ui: bool = True):
        self.rebuilt = True

    def refresh_channel_name_dependent_ui(self):
        pass

    def build_analysis_matrix_for_current_source(self):
        pass

    def update_analysis_matrix_warning_label(self):
        pass

    def remap_analysis_references_for_detected_folders(self, image_defs, old_names, new_names):
        pass

    def log(self, message: str):
        self.logs.append(message)

    def update_worker_status(self, message: str):
        self.status_updates.append(message)


class PermissionDeniedPath:
    def exists(self) -> bool:
        return True

    def is_dir(self) -> bool:
        return True

    def iterdir(self):
        raise PermissionError("blocked by test")


class FakeThread:
    def __init__(self, running: bool):
        self._running = running

    def isRunning(self) -> bool:
        return self._running


class FakeTimer:
    def __init__(self):
        self.started_with = []
        self.stop_count = 0

    def start(self, interval: int):
        self.started_with.append(interval)

    def stop(self):
        self.stop_count += 1


class FakePathValue:
    def __init__(self, value: str):
        self.value = value

    def get(self) -> str:
        return self.value


class FakeBrowserIndex:
    def isValid(self) -> bool:
        return True


class FakeFileSystemModel:
    def index(self, _path: str) -> FakeBrowserIndex:
        return FakeBrowserIndex()


class FakeTreeView:
    def __init__(self):
        self.root = None

    def setRootIndex(self, index):
        self.root = index


class FakeLineEdit:
    def __init__(self):
        self.value = ""

    def setText(self, value: str):
        self.value = value

    def clear(self):
        self.value = ""


class FakeButton:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled: bool):
        self.enabled = enabled


def navigation_browser() -> DummyFileBrowser:
    browser = DummyFileBrowser()
    browser.fs_model = FakeFileSystemModel()
    browser.file_tree = FakeTreeView()
    browser.browser_path_entry = FakeLineEdit()
    browser.browser_back_button = FakeButton()
    browser.browser_forward_button = FakeButton()
    browser.browser_up_button = FakeButton()
    browser._browser_history = []
    browser._browser_forward_history = []
    browser._browser_root_path = None
    return browser


@pytest.mark.parametrize(
    "structure",
    [
        "Samples directly in input folder",
        "Protein folders containing sample folders",
        "Unsorted TIFF images in input folder",
        "Image folders containing unsorted TIFF images",
    ],
)
def test_current_input_structure_uses_matching_scan_result(structure: str):
    browser = DummyFileBrowser()
    browser.input_dir = FakePathValue(r"C:\Data\Current")
    browser._last_input_scan_result = {
        "path": "C:/Data/Current",
        "detection": {"structure": structure},
    }

    assert browser.get_current_input_structure() == structure


# Exercise full explorer-style history because Back and Forward must move the
# current root between opposite stacks without creating duplicate entries.
def test_file_browser_back_and_forward_history(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    browser = navigation_browser()

    browser.set_browser_root(str(first))
    browser.set_browser_root(str(second))
    browser.browser_go_back()

    assert browser._browser_root_path == str(first)
    assert browser._browser_forward_history == [str(second)]
    assert browser.browser_forward_button.enabled is True

    browser.browser_go_forward()

    assert browser._browser_root_path == str(second)
    assert browser._browser_forward_history == []
    assert browser.browser_forward_button.enabled is False


# A fresh destination abandons the old forward branch, matching operating-system file browsers.
def test_file_browser_new_navigation_clears_forward_history(tmp_path: Path):
    folders = [tmp_path / name for name in ("first", "second", "third")]
    for folder in folders:
        folder.mkdir()
    browser = navigation_browser()

    browser.set_browser_root(str(folders[0]))
    browser.set_browser_root(str(folders[1]))
    browser.browser_go_back()
    browser.set_browser_root(str(folders[2]))

    assert browser._browser_root_path == str(folders[2])
    assert browser._browser_forward_history == []


# Computer is a real history location rather than a one-way reset, allowing the
# user to return immediately to the folder they were reviewing.
def test_file_browser_can_return_from_computer_view(tmp_path: Path):
    folder = tmp_path / "results"
    folder.mkdir()
    browser = navigation_browser()

    browser.set_browser_root(str(folder))
    browser.browser_go_to_computer()
    assert browser._browser_root_path is None

    browser.browser_go_back()

    assert browser._browser_root_path == str(folder)


@pytest.mark.parametrize(
    "scan_result",
    [
        {"path": r"C:\Data\Previous", "detection": {"structure": "Samples directly in input folder"}},
        {"path": r"C:\Data\Current", "detection": {"structure": ""}},
        {"path": r"C:\Data\Current", "detection": {"structure": "Unknown layout"}},
    ],
)
def test_current_input_structure_rejects_stale_or_unknown_scan_results(scan_result: dict):
    browser = DummyFileBrowser()
    browser.input_dir = FakePathValue(r"C:\Data\Current")
    browser._last_input_scan_result = scan_result

    assert browser.get_current_input_structure() == "Unsorted TIFF images in input folder"


@pytest.mark.parametrize("state_field", ["_pending_input_scan_path", "_active_input_scan_path"])
def test_current_input_structure_does_not_guess_while_current_path_is_being_scanned(state_field: str):
    browser = DummyFileBrowser()
    browser.input_dir = FakePathValue(r"C:\Data\Current")
    browser._last_input_scan_result = {
        "path": r"C:\Data\Previous",
        "detection": {"structure": "Samples directly in input folder"},
    }
    setattr(browser, state_field, "C:/Data/Current")

    with pytest.raises(ValueError, match="still being scanned"):
        browser.get_current_input_structure()


def test_safe_iterdir_returns_empty_for_permission_errors():
    assert safe_iterdir(cast(Path, PermissionDeniedPath())) == []


def test_scan_child_dirs_skips_tooling_and_cache_folders(tmp_path: Path):
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".RUFF_CACHE").mkdir()
    (tmp_path / "dist").mkdir()
    (tmp_path / "Sample_01").mkdir()

    child_dirs = scan_child_dirs(tmp_path)

    assert [path.name for path in child_dirs] == ["Sample_01"]


def test_tiff_presence_treats_inaccessible_folders_as_empty(monkeypatch, tmp_path):
    def denied(_path):
        raise PermissionError("folder access denied")

    monkeypatch.setattr(Path, "iterdir", denied)
    assert not contains_supported_tiff(tmp_path)


def test_scan_matching_files_limit_stops_after_first_match(tmp_path: Path):
    paths = [tmp_path / f"image_{index}.tif" for index in range(3)]
    for path in paths:
        path.write_bytes(b"fixture")
    visited = []

    result = scan_matching_files(
        tmp_path,
        recursive=False,
        predicate=lambda path: visited.append(path) is None,
        iterdir_func=lambda _folder: paths,
        limit=1,
    )

    assert result == [paths[0]]
    assert visited == [paths[0]]


def test_tiff_presence_cache_is_scoped_by_folder_and_recursion(monkeypatch, tmp_path: Path):
    calls = []

    def fake_scan(folder, **kwargs):
        calls.append((folder, kwargs["recursive"], kwargs["limit"]))
        return [folder / "image.tif"]

    monkeypatch.setattr(discovery, "scan_matching_files", fake_scan)
    cache = {}

    assert contains_supported_tiff(tmp_path, cache=cache) is True
    assert contains_supported_tiff(tmp_path, cache=cache) is True
    assert contains_supported_tiff(tmp_path, recursive=True, cache=cache) is True
    assert calls == [
        (tmp_path, False, 1),
        (tmp_path, True, 1),
    ]


def test_analysis_scanning_ignores_non_tiff_images(tmp_path: Path):
    (tmp_path / "preview.png").write_bytes(b"not an analysis input")
    (tmp_path / "sample.tif").write_bytes(b"tiff")
    assert discovery.iter_supported_tiff_files(tmp_path) == [tmp_path / "sample.tif"]


def test_grouped_layout_count_excludes_empty_sample_folders(tmp_path: Path):
    empty_sample = tmp_path / "Protein_A" / "Empty"
    empty_sample.mkdir(parents=True)
    channel = tmp_path / "Protein_A" / "Sample_01" / "GFP"
    channel.mkdir(parents=True)
    tifffile.imwrite(channel / "sample.tif", np.zeros((2, 2), dtype=np.uint8))

    detection, _truncated = _limited_background_input_detection(
        tmp_path,
        expected_folders=[],
        active_image_defs=[],
    )

    assert detection["structure"] == "Protein folders containing sample folders"
    assert detection["sample_count"] == 1


def test_folder_layout_detection_falls_back_from_generic_expected_names(tmp_path: Path):
    for folder_name in ("GFP", "DAPI"):
        channel_dir = tmp_path / folder_name
        channel_dir.mkdir()
        tifffile.imwrite(channel_dir / "sample_01.tif", np.zeros((2, 2), dtype=np.uint8))

    detection, _truncated = _limited_background_input_detection(
        tmp_path,
        expected_folders=["Channel 1", "Channel 2", "Channel 3"],
        active_image_defs=[],
    )

    assert detection["structure"] == "Image folders containing unsorted TIFF images"
    assert set(detection["folders"]) == {"GFP", "DAPI"}


def test_folder_layout_applies_detected_names_to_generic_channel_rows(tmp_path: Path):
    for folder_name in ("GFP", "DAPI"):
        channel_dir = tmp_path / folder_name
        channel_dir.mkdir()
        tifffile.imwrite(channel_dir / "sample_01.tif", np.zeros((2, 2), dtype=np.uint8))
    browser = DummyFileBrowser()
    browser.image_definitions = [default_image_definition(index) for index in range(3)]

    browser.apply_detected_image_folders_from_input(str(tmp_path))

    assert {image_def["name"] for image_def in browser.image_definitions} == {"GFP", "DAPI"}
    assert {image_def["folder"] for image_def in browser.image_definitions} == {"GFP", "DAPI"}


def test_folder_layout_remaps_default_mask_relationships(tmp_path: Path):
    channel_dir = tmp_path / "GFP"
    channel_dir.mkdir()
    tifffile.imwrite(channel_dir / "sample_01.tif", np.zeros((2, 2), dtype=np.uint8))
    browser = DummyFileBrowser()
    channel = default_image_definition(0)
    channel["mask_slot_enabled"] = False
    channel["mask_relationships"] = {"Channel 1 mask": True}
    mask = default_image_definition(1)
    mask.update(
        {
            "name": "Channel 1 mask",
            "folder": "Channel 1",
            "is_mask_only": True,
            "mask_source_channel": "Channel 1",
        }
    )
    browser.image_definitions = [channel, mask]

    browser.apply_detected_image_folders_from_input(str(tmp_path))

    assert browser.image_definitions[0]["name"] == "GFP"
    assert browser.image_definitions[0]["mask_relationships"] == {"GFP mask": True}
    assert browser.image_definitions[1]["name"] == "GFP mask"
    assert browser.image_definitions[1]["mask_source_channel"] == "GFP"


def test_apply_detected_image_folders_from_flat_tiff_stack_builds_channel_rows(tmp_path: Path):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    tifffile.imwrite(
        tmp_path / "sample_001.tif",
        np.zeros((3, 4, 5), dtype=np.uint16),
        photometric="minisblack",
    )
    browser = DummyFileBrowser()

    browser.apply_detected_image_folders_from_input(str(tmp_path))

    assert [img["stack_channel_index"] for img in browser.image_definitions] == ["1", "2", "3"]
    assert [img["name"] for img in browser.image_definitions] == ["Channel 1", "Channel 2", "Channel 3"]


def test_scan_input_path_collects_tiff_metadata_off_gui_thread(tmp_path: Path):
    if getattr(np, "__cellonaut_stub__", False):
        pytest.skip("numpy is stubbed in this lightweight test environment")

    tifffile.imwrite(
        tmp_path / "sample_001.tif",
        np.zeros((4, 5, 6), dtype=np.uint16),
        photometric="minisblack",
    )

    result = scan_input_path(
        str(tmp_path),
        expected_folders=[],
        active_image_defs=[],
    )

    assert result["path_exists"] is True
    assert result["path_is_dir"] is True
    assert result["detection"]["structure"] == "Unsorted TIFF images in input folder"
    assert result["tiff_info"]["channel_count"] == 4


def test_input_path_key_treats_windows_slash_styles_as_same_path():
    assert input_path_key("C:/Data/Ost1_TIFF") == input_path_key(r"C:\Data\Ost1_TIFF")


def test_main_window_exposes_queued_input_scan_bridge_signals():
    assert hasattr(CellonautMainWindow, "input_scan_result_on_gui")
    assert hasattr(CellonautMainWindow, "input_scan_finished_on_gui")


def test_scan_input_path_collects_nd2_metadata(tmp_path: Path, monkeypatch):
    nd2_path = tmp_path / "sample.nd2"
    nd2_path.write_bytes(b"fixture")
    monkeypatch.setattr(
        file_browser_gui.ndi,
        "inspect_nd2_file",
        lambda _path: {
            "channel_names": ["GFP", "DAPI"],
            "channel_colors": ["#00FF00", "#0000FF"],
            "sizes": {"T": 2, "Z": 7, "C": 2, "Y": 64, "X": 64},
        },
    )

    result = scan_input_path(
        str(tmp_path),
        expected_folders=[],
        active_image_defs=[],
    )

    assert result["nd2_count"] == 1
    assert result["nd2_first"] == str(nd2_path)
    assert result["nd2_channels"] == ["GFP", "DAPI"]
    assert result["nd2_channel_colors"] == ["#00FF00", "#0000FF"]
    assert result["nd2_sizes"] == {"T": 2, "Z": 7, "C": 2, "Y": 64, "X": 64}


def test_scan_input_path_summarizes_z_depths_for_the_nd2_importer(tmp_path: Path, monkeypatch):
    paths = [tmp_path / "plane.nd2", tmp_path / "stack_4.nd2", tmp_path / "stack_7.nd2"]
    for path in paths:
        path.write_bytes(b"fixture")
    depths = {"plane.nd2": 1, "stack_4.nd2": 4, "stack_7.nd2": 7}

    def inspect(path: Path):
        depth = depths[path.name]
        sizes = {"C": 2, "Y": 64, "X": 64}
        if depth > 1:
            sizes["Z"] = depth
        return {"channel_names": ["GFP", "DAPI"], "sizes": sizes}

    monkeypatch.setattr(file_browser_gui.ndi, "inspect_nd2_file", inspect)

    result = scan_input_path(
        str(tmp_path),
        expected_folders=[],
        active_image_defs=[],
        inspect_all_nd2_z=True,
    )

    assert result["nd2_z_summary"] == {
        "inspected_count": 3,
        "stack_count": 2,
        "single_plane_count": 1,
        "min_stack_depth": 4,
        "max_stack_depth": 7,
        "error_count": 0,
        "complete": True,
    }


def test_nd2_dataset_scan_blocks_inconsistent_channel_order(tmp_path: Path, monkeypatch):
    first = tmp_path / "a.nd2"
    second = tmp_path / "b.nd2"
    first.write_bytes(b"fixture")
    second.write_bytes(b"fixture")

    def inspect(path: Path):
        channels = ["GFP", "DAPI"] if path.name == "a.nd2" else ["DAPI", "GFP"]
        return {"channel_names": channels, "sizes": {"C": 2, "Y": 64, "X": 64}}

    monkeypatch.setattr(file_browser_gui.ndi, "inspect_nd2_file", inspect)

    result = scan_input_path(
        str(tmp_path),
        expected_folders=[],
        active_image_defs=[],
        inspect_all_nd2_z=True,
    )

    assert result["nd2_dataset_summary"]["compatible"] is False
    assert result["nd2_dataset_summary"]["issue_count"] == 1
    assert "b.nd2: channels ['DAPI', 'GFP'] do not match ['GFP', 'DAPI']" in result["nd2_dataset_summary"][
        "issues"
    ][0]


def test_schedule_input_scan_coalesces_same_path_while_scan_is_active():
    browser = DummyFileBrowser()
    browser._active_input_scan_path = r"\\server\share\data"
    browser._pending_input_scan_path = r"\\server\share\data"
    timer = FakeTimer()
    browser.__dict__["_input_scan_debounce"] = timer

    browser.schedule_input_path_scan(r"\\server\share\data")

    assert browser._pending_input_scan_path == ""
    assert timer.started_with == []
    assert timer.stop_count == 1


def test_schedule_input_scan_keeps_new_path_requested_during_active_scan():
    browser = DummyFileBrowser()
    browser._active_input_scan_path = r"\\server\share\old"
    browser._pending_input_scan_path = ""
    timer = FakeTimer()
    browser.__dict__["_input_scan_debounce"] = timer

    browser.schedule_input_path_scan(r"\\server\share\new")

    assert browser._pending_input_scan_path == r"\\server\share\new"
    assert timer.started_with == [600]


def test_schedule_input_scan_cancels_stale_worker_for_new_path():
    browser = DummyFileBrowser()
    browser._active_input_scan_path = r"\\server\share\old"
    browser._pending_input_scan_path = ""
    timer = FakeTimer()
    browser.__dict__["_input_scan_debounce"] = timer
    cancelled = []
    browser._input_scan_worker = type("Worker", (), {"request_cancel": lambda self: cancelled.append(True)})()

    browser.schedule_input_path_scan(r"\\server\share\new")

    assert cancelled == [True]
    assert browser._pending_input_scan_path == r"\\server\share\new"


def test_schedule_input_scan_skips_path_that_already_completed():
    browser = DummyFileBrowser()
    browser._active_input_scan_path = ""
    browser._last_input_scan_result = {"path": r"C:\Data\Ost1_TIFF"}
    browser._pending_input_scan_path = "C:/Data/Ost1_TIFF"
    timer = FakeTimer()
    browser.__dict__["_input_scan_debounce"] = timer

    browser.schedule_input_path_scan("C:/Data/Ost1_TIFF")

    assert browser._pending_input_scan_path == ""
    assert timer.started_with == []
    assert timer.stop_count == 1


def test_scan_input_path_limits_nd2_walk_for_large_trees(tmp_path: Path):
    for index in range(3):
        (tmp_path / f"nested_{index}").mkdir()

    result = scan_input_path(
        str(tmp_path),
        expected_folders=[],
        active_image_defs=[],
        max_nd2_dirs=1,
    )

    assert result["nd2_scan_truncated"] is True


def test_background_nd2_scan_does_not_revisit_directory_aliases(monkeypatch):
    class ScanPath:
        def __init__(self, name: str, kind: str, key: str):
            self.name = name
            self.kind = kind
            self.key = key
            self.suffix = Path(name).suffix

        def is_dir(self):
            return self.kind == "dir"

        def is_file(self):
            return self.kind == "file"

        def __str__(self):
            return self.name

    root = ScanPath("root", "dir", "root")
    child = ScanPath("child", "dir", "child")
    root_alias = ScanPath("root-alias", "dir", "root")
    nd2_path = ScanPath("sample.nd2", "file", "file")
    entries = {
        root: [child],
        child: [nd2_path, root_alias],
        root_alias: [child],
    }
    visited = []

    def fake_entries(folder, *, max_entries, cancel_requested=None):
        visited.append(folder.key)
        return entries[folder][:max_entries], False

    monkeypatch.setattr(file_browser_gui, "directory_scan_key", lambda folder: folder.key)
    monkeypatch.setattr(file_browser_gui, "_limited_dir_entries", fake_entries)

    files, truncated = file_browser_gui._limited_nd2_files(cast(Path, root), max_dirs=10, max_entries=20)

    assert files == [nd2_path]
    assert truncated is False
    assert visited == ["root", "child"]


def test_background_layout_uses_one_global_directory_budget(monkeypatch):
    class ScanPath:
        def __init__(self, name: str, key: str, children=()):
            self.name = name
            self.key = key
            self.children = list(children)
            self.suffix = ""

        def is_dir(self):
            return True

        def is_file(self):
            return False

        def stat(self):
            return type("Stat", (), {"st_ctime_ns": 0})()

    children = [ScanPath(f"sample_{index}", f"sample_{index}") for index in range(5)]
    root = ScanPath("root", "root", children)
    visited = []

    def fake_entries(folder, *, max_entries, cancel_requested=None):
        visited.append(folder.key)
        return folder.children[:max_entries], False

    monkeypatch.setattr(file_browser_gui, "directory_scan_key", lambda folder: folder.key)
    monkeypatch.setattr(file_browser_gui, "_limited_dir_entries", fake_entries)

    _detection, truncated = _limited_background_input_detection(
        cast(Path, root),
        expected_folders=[],
        active_image_defs=[],
        max_dirs=2,
        max_entries=100,
    )

    assert truncated is True
    assert visited == ["root", "sample_0"]


def test_background_layout_detection_has_scan_budget(tmp_path: Path):
    for index in range(3):
        sample = tmp_path / f"sample_{index}"
        sample.mkdir()
        channel = sample / "GFP"
        channel.mkdir()
        tifffile.imwrite(channel / f"sample_{index}.tif", np.zeros((2, 2), dtype=np.uint8))

    detection, truncated = _limited_background_input_detection(
        tmp_path,
        expected_folders=[],
        active_image_defs=[],
        max_dirs=1,
        max_entries=2,
    )

    assert truncated is True
    assert detection["sample_count"] <= 1


def test_input_scan_timeout_stops_stale_scan_heartbeat_during_pipeline():
    browser = DummyFileBrowser()
    browser.running = True
    browser._input_scan_thread = FakeThread(running=True)
    browser._active_input_scan_path = r"\\server\share\slow"
    browser._worker_status_base = "Scanning input folder"
    browser._worker_status_stage = PipelineStage.SCANNING.value
    browser.stopped_heartbeat = False

    def stop_heartbeat():
        browser.stopped_heartbeat = True
        browser._worker_status_base = ""

    browser.stop_worker_status_heartbeat = stop_heartbeat

    browser.on_input_scan_status_timeout()

    assert browser.stopped_heartbeat is True
    assert browser.status_updates == []


def test_input_scan_worker_reports_scan_exceptions(monkeypatch):
    def fail_scan(*_args, **_kwargs):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(file_browser_gui, "scan_input_path", fail_scan)
    worker = InputPathScanWorker("C:/bad", [], [])
    emitted = []
    worker.done_signal.connect(emitted.append)

    worker.run()

    assert emitted
    assert emitted[0]["path"] == "C:/bad"
    assert emitted[0]["scan_error"] == "RuntimeError: scan failed"


def test_input_scan_worker_honors_cancellation_before_scanning():
    worker = InputPathScanWorker("C:/unused", [], [])
    emitted = []
    worker.done_signal.connect(emitted.append)

    worker.request_cancel()
    worker.run()

    assert emitted[0]["scan_cancelled"] is True


def test_input_scan_timeout_clears_status_without_finishing_thread():
    browser = DummyFileBrowser()
    browser._worker_status_stage = PipelineStage.SCANNING.value
    browser._input_scan_thread = FakeThread(running=True)
    browser._active_input_scan_path = r"\\server\share\slow"
    cancelled = []
    browser._input_scan_worker = type("Worker", (), {"request_cancel": lambda self: cancelled.append(True)})()

    browser.on_input_scan_status_timeout()

    assert browser.status_updates == ["Ready"]
    assert cancelled == [True]
    assert "exceeded the time limit" in browser.logs[-1]


def test_input_scan_cleanup_clears_heartbeat_even_without_result_callback():
    browser = DummyFileBrowser()
    browser.input_dir = FakePathValue(r"C:\Data")
    browser._input_scan_status_timeout = FakeTimer()
    browser._input_scan_worker = object()
    browser._input_scan_thread = FakeThread(running=False)
    browser._active_input_scan_path = r"C:\Data"
    browser._pending_input_scan_path = ""
    browser._last_input_scan_result = {}
    browser._worker_status_base = "Scanning input folder"
    browser._worker_status_stage = PipelineStage.SCANNING.value
    stopped = []

    def stop_heartbeat():
        stopped.append(True)
        browser._worker_status_base = ""

    browser.stop_worker_status_heartbeat = stop_heartbeat

    browser.cleanup_input_path_scan()

    assert stopped == [True]
    assert browser.status_updates == ["Ready"]
    assert browser._input_scan_worker is None
    assert browser._input_scan_thread is None
    assert browser._active_input_scan_path == ""


@pytest.mark.parametrize("action", ["timeout", "stale_result", "finished"])
@pytest.mark.parametrize("wording", ["Scanning input folder", "Discovering input images"])
@pytest.mark.parametrize("stage", [PipelineStage.SCANNING.value, PipelineStage.CHECKING.value,
                                   PipelineStage.INITIALIZING.value, None])
def test_scan_cleanup_uses_stage_identity_and_preserves_other_status(action, wording, stage):
    browser = DummyFileBrowser()
    browser.input_dir = FakePathValue(r"C:\Current")
    browser._input_scan_thread = FakeThread(running=True)
    browser._active_input_scan_path = r"C:\Old"
    browser._pending_input_scan_path = ""
    browser._worker_status_stage = stage
    browser._worker_status_base = wording
    stopped = []
    browser.stop_worker_status_heartbeat = lambda: stopped.append(True)

    if action == "timeout":
        browser.on_input_scan_status_timeout()
    elif action == "stale_result":
        browser.apply_input_path_scan_result({"path": r"C:\Old"})
    else:
        browser.cleanup_input_path_scan()

    owns_status = stage == PipelineStage.SCANNING
    assert stopped == ([True] if owns_status else [])
    assert browser.status_updates == (["Ready"] if owns_status else [])
    if not owns_status:
        assert browser._worker_status_stage == stage
        assert browser._worker_status_base == wording


def test_open_folder_in_system_browser_uses_windows_startfile(monkeypatch, tmp_path: Path):
    opened = []
    target = tmp_path / "sample.tif"
    target.write_text("", encoding="utf-8")

    monkeypatch.setattr(file_browser_gui.os, "startfile", lambda path: opened.append(path), raising=False)

    open_folder_in_system_browser(target)

    assert opened == [str(tmp_path)]
