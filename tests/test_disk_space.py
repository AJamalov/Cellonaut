from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PySide6.QtWidgets import QMessageBox

from cellonaut.gui import validation as validation_gui
from cellonaut.gui.validation import CellonautGuiValidationMixin
from cellonaut.system import disk_space


class DummyValidation(CellonautGuiValidationMixin):
    def __init__(self):
        self.messages = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def test_estimate_disk_space_uses_existing_destination_ancestor(monkeypatch, tmp_path: Path):
    checked = []
    monkeypatch.setattr(
        disk_space.shutil,
        "disk_usage",
        lambda path: checked.append(path) or SimpleNamespace(total=10_000, used=2_000, free=8_000),
    )

    estimate = disk_space.estimate_disk_space(
        tmp_path / "new" / "results",
        source_bytes=1_000,
        output_multiplier=3.0,
        reserve_bytes=2_000,
    )

    assert checked == [tmp_path]
    assert estimate.free_bytes == 8_000
    assert estimate.recommended_bytes == 5_000
    assert not estimate.below_recommended


def test_unique_file_bytes_and_readiness_paths_ignore_duplicates(tmp_path: Path):
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    first.write_bytes(b"1234")
    second.write_bytes(b"123456")
    readiness = {
        "samples": [
            {"ready": True, "files": {"A": str(first), "B": str(first)}},
            {"ready": True, "files": {"A": str(second)}},
        ]
    }

    all_paths = disk_space.readiness_source_files(readiness)
    preview_paths = disk_space.readiness_source_files(readiness, first_ready_only=True)

    assert disk_space.unique_file_bytes(all_paths) == 10
    assert disk_space.unique_file_bytes(preview_paths) == 4


def test_disk_space_estimate_marks_critical_and_recommended_levels():
    estimate = disk_space.DiskSpaceEstimate(
        destination=Path("results"),
        free_bytes=disk_space.HARD_MINIMUM_FREE_BYTES - 1,
        recommended_bytes=disk_space.GIB,
        source_bytes=0,
    )

    assert estimate.critically_low
    assert estimate.below_recommended


def test_gui_preflight_blocks_critically_low_destination(monkeypatch, tmp_path: Path):
    critical_messages = []
    monkeypatch.setattr(
        validation_gui,
        "estimate_disk_space",
        lambda *_args, **_kwargs: disk_space.DiskSpaceEstimate(
            destination=tmp_path,
            free_bytes=disk_space.HARD_MINIMUM_FREE_BYTES - 1,
            recommended_bytes=disk_space.GIB,
            source_bytes=0,
        ),
    )
    monkeypatch.setattr(
        validation_gui.QMessageBox,
        "critical",
        lambda _parent, title, message: critical_messages.append((title, message)),
    )

    allowed = DummyValidation().confirm_disk_space_for_files(
        tmp_path,
        [],
        operation="Full pipeline run",
        output_multiplier=4.0,
        reserve_bytes=2 * disk_space.GIB,
    )

    assert not allowed
    assert critical_messages[0][0] == "Not enough disk space"
    assert "critically low free space" in critical_messages[0][1]


def test_gui_preflight_requires_confirmation_below_recommendation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        validation_gui,
        "estimate_disk_space",
        lambda *_args, **_kwargs: disk_space.DiskSpaceEstimate(
            destination=tmp_path,
            free_bytes=disk_space.GIB,
            recommended_bytes=2 * disk_space.GIB,
            source_bytes=0,
        ),
    )
    monkeypatch.setattr(
        validation_gui.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )

    allowed = DummyValidation().confirm_disk_space_for_files(
        tmp_path,
        [],
        operation="ND2 conversion",
        output_multiplier=3.0,
        reserve_bytes=2 * disk_space.GIB,
    )

    assert not allowed
