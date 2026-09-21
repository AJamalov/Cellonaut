from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHeaderView, QLabel, QTableWidget

from cellonaut.gui.dialogs import (
    ConfigurationSummaryDialog,
    PresetInfoDialog,
    TextReportDialog,
)
from cellonaut.masks.cell_qc import cell_qc_rules_to_text


pytestmark = pytest.mark.gui


def qt_app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


def setup_summary(*, warnings: list[str] | None = None) -> dict:
    return {
        "folders": {
            "input": "C:/data",
            "output": "C:/results",
            "fiji": "C:/Fiji.app",
            "layout": "TIFF stacks",
        },
        "setup_checks": [{"status": "OK", "check": "Input folder", "detail": "4 samples found."}],
        "images": [{"type": "Channel", "name": "Channel 1", "folder": "Channel 1"}],
        "relationships": [{"source": "Channel 1", "target": "Channel 1 mask", "self": False}],
        "cell_segmentation": [],
        "filters": [],
        "measurements": [{"source": "All measured channels", "enabled": ["Area", "Mean gray value"]}],
        "dry_run": {"available": True, "total": 4, "ready": 4, "incomplete": 0, "warning_count": 0},
        "warnings": list(warnings or []),
    }


def test_cell_qc_rules_to_text_normalizes_invalid_values():
    text = cell_qc_rules_to_text(
        {
            "Area": {"min": "50", "max": "nan"},
            "Mean intensity": {"max": 7000},
            "Broken": {"min": "not a number"},
        }
    )

    assert text == "Area:50-; Mean intensity:-7000"


def test_text_report_copy_uses_complete_report():
    app = qt_app()
    dialog = TextReportDialog("Setup report", "line one\nline two")
    try:
        dialog.text_box.selectAll()
        dialog.copy_text()
        assert app.clipboard().text() == "line one\nline two"
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_configuration_summary_uses_clear_status_wording_and_theme_cards():
    app = qt_app()
    dialog = ConfigurationSummaryDialog(setup_summary(warnings=["Cellpose masks are disabled."]), "text report")
    try:
        labels = dialog.findChildren(QLabel)
        texts = [label.text() for label in labels]
        assert "1 channel/mask item" in texts
        assert "1 mask relationship" in texts
        assert "1 warning" in texts
        warning_pill = next(label for label in labels if label.text() == "1 warning")
        assert warning_pill.property("statusKind") == "warning"
        assert dialog.copy_button.text() == "Copy summary"
        assert dialog.raw_button.text() == "Open text report"

        warning_card = dialog.build_warnings_card()
        warning_texts = [label.text() for label in warning_card.findChildren(QLabel)]
        assert "Warnings" in warning_texts
        assert "Cellpose masks are disabled." in warning_texts
        assert all(not text.startswith("WARNING:") for text in warning_texts)
        assert warning_card.styleSheet() == ""

        sample_card = dialog.build_dry_run_card()
        sample_texts = [label.text() for label in sample_card.findChildren(QLabel)]
        assert "Sample detection" in sample_texts
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_configuration_summary_key_value_tables_share_read_only_builder():
    app = qt_app()
    dialog = ConfigurationSummaryDialog(setup_summary())
    try:
        table = dialog.make_kv_table([("Detected samples", "4")])
        assert table.columnCount() == 2
        assert table.rowCount() == 1
        key_item = table.item(0, 0)
        value_item = table.item(0, 1)
        assert key_item is not None
        assert value_item is not None
        assert key_item.text() == "Detected samples"
        assert value_item.text() == "4"
        assert table.editTriggers().value == 0
        assert table.wordWrap() is True
        assert table.textElideMode() == Qt.TextElideMode.ElideNone
        assert table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert table.verticalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.ResizeToContents
        assert value_item.toolTip() == "4"
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_sample_problem_table_shows_mask_reuse_warning():
    app = qt_app()
    summary = setup_summary()
    summary["dry_run"]["problems"] = [{"sample": "sample-1", "reuse_mask_warnings": ["Saved mask is missing"]}]
    dialog = ConfigurationSummaryDialog(summary)
    try:
        card = dialog.build_dry_run_card()
        problem_table = next(
            table for table in card.findChildren(QTableWidget)
            if table.columnCount() == 5
        )
        warning_item = problem_table.item(0, 4)
        assert warning_item is not None
        assert warning_item.text() == "Saved mask is missing"
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_new_preset_requires_a_name_before_saving():
    app = qt_app()
    dialog = PresetInfoDialog()
    try:
        assert dialog.windowTitle() == "Save Preset"
        assert dialog.ok_button.text() == "Save Preset"
        assert dialog.ok_button.isEnabled() is False

        dialog.name_edit.setText("  Golgi analysis  ")
        assert dialog.ok_button.isEnabled() is True
        assert dialog.get_name() == "Golgi analysis"

        dialog.name_edit.setText("   ")
        assert dialog.ok_button.isEnabled() is False
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_existing_preset_uses_save_wording():
    app = qt_app()
    dialog = PresetInfoDialog("Golgi analysis")
    try:
        assert dialog.windowTitle() == "Save Preset"
        assert dialog.ok_button.text() == "Save"
        assert dialog.ok_button.property("role") == "primary"
        assert dialog.get_name() == "Golgi analysis"
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
