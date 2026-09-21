from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QMimeData, Qt
from PySide6.QtGui import QTextCursor

from cellonaut.gui.main_window import CellonautMainWindow
from cellonaut.gui.panel_notes import (
    COLLAPSED_NOTE_LINES,
    MAX_NOTE_LINES,
    PANEL_NOTE_KEYS,
    LimitedNoteEdit,
    PanelNotesWidget,
    limit_note_lines,
    normalize_panel_notes,
)

pytestmark = pytest.mark.gui


def test_note_text_helpers_bound_and_normalize_preset_values():
    long_note = "\r\n".join(f"Line {index}" for index in range(25))

    assert limit_note_lines(long_note).splitlines() == [f"Line {index}" for index in range(MAX_NOTE_LINES)]
    assert normalize_panel_notes(None) == {key: "" for key in PANEL_NOTE_KEYS}

    normalized = normalize_panel_notes(
        {
            PANEL_NOTE_KEYS[0]: long_note,
            PANEL_NOTE_KEYS[1]: 42,
            "unknown_panel": "not retained",
        }
    )
    assert len(normalized[PANEL_NOTE_KEYS[0]].splitlines()) == MAX_NOTE_LINES
    assert normalized[PANEL_NOTE_KEYS[1]] == "42"
    assert "unknown_panel" not in normalized


def test_note_editor_preserves_existing_lines_at_the_limit(qt_application):
    editor = LimitedNoteEdit()
    original = "\n".join(f"Existing {index}" for index in range(MAX_NOTE_LINES))
    editor.set_limited_text(original)
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    editor.setTextCursor(cursor)

    mime = QMimeData()
    mime.setText("\nPasted line\nAnother pasted line")
    editor.insertFromMimeData(mime)

    assert editor.toPlainText() == original
    assert editor.document().blockCount() == MAX_NOTE_LINES
    editor.deleteLater()
    qt_application.processEvents()


def test_notes_widget_expands_from_three_to_twenty_visible_lines(qt_application):
    widget = PanelNotesWidget()
    collapsed_height = widget.editor.height()

    assert widget.is_expanded() is False
    assert widget.toggle_button.arrowType() == Qt.ArrowType.DownArrow
    assert widget.toggle_button.toolTip() == f"Expand notes to {MAX_NOTE_LINES} lines"
    assert widget.toggle_button.size().width() == 18
    assert widget.toggle_button.size().height() == 16

    widget.toggle_button.click()

    assert widget.is_expanded() is True
    assert widget.editor.height() > collapsed_height
    assert widget.toggle_button.arrowType() == Qt.ArrowType.UpArrow
    assert widget.toggle_button.toolTip() == f"Reduce notes to {COLLAPSED_NOTE_LINES} lines"
    widget.deleteLater()
    qt_application.processEvents()


@pytest.fixture
def notes_window(monkeypatch, qt_application):
    monkeypatch.setattr(CellonautMainWindow, "load_last_settings_if_available", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "load_default_preset_on_startup", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "save_last_settings", lambda self: None)
    monkeypatch.setattr(CellonautMainWindow, "validate_all_fields", lambda self: True)
    monkeypatch.setattr(CellonautMainWindow, "start_pending_fiji_component_scan", lambda self: None)
    window = CellonautMainWindow()
    yield window
    window.log_flush_timer.stop()
    window.close()
    window.deleteLater()
    qt_application.processEvents()


def test_each_opened_panel_has_independent_preset_backed_notes(notes_window, qt_application):
    window = notes_window
    header_layout = window.pipeline_editor_header.layout()
    assert header_layout is not None
    assert header_layout.itemAt(3).widget() is window.pipeline_notes_widget

    expected_notes = {}
    for index, key in enumerate(PANEL_NOTE_KEYS):
        window.show_pipeline_section(index)
        note = f"Protocol note for {key}"
        window.pipeline_notes_widget.editor.setPlainText(note)
        expected_notes[key] = note

    saved = window.get_preset_dict()
    assert saved["panel_notes"] == expected_notes

    window.apply_preset_dict({**saved, "panel_notes": {key: f"Loaded {key}" for key in PANEL_NOTE_KEYS}})
    for index, key in enumerate(PANEL_NOTE_KEYS):
        window.show_pipeline_section(index)
        assert window.pipeline_notes_widget.text() == f"Loaded {key}"

    window.show_pipeline_section(0)
    window.pipeline_notes_widget.toggle_button.click()
    assert window.pipeline_notes_widget.is_expanded() is True
    window.show_pipeline_section(1)
    assert window.pipeline_notes_widget.is_expanded() is False
    window.show_pipeline_section(0)
    assert window.pipeline_notes_widget.is_expanded() is True
    qt_application.processEvents()


def test_note_edits_mark_and_save_the_selected_preset(notes_window, tmp_path):
    window = notes_window
    window.show_pipeline_section(2)
    window.remember_preset_state("Protocol", window.get_preset_dict())

    window.pipeline_notes_widget.editor.insertPlainText("Use the custom Cellpose model for this assay.")

    assert window.preset_has_unsaved_changes() is True
    destination = tmp_path / "Protocol.json"
    window._save_preset_to_path(
        destination,
        "Protocol",
        created=True,
        notify=False,
        refresh=False,
    )
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored["panel_notes"]["cellpose"] == "Use the custom Cellpose model for this assay."
    assert window.preset_has_unsaved_changes() is False


def test_expanded_notes_keep_every_panel_scrollable(notes_window, qt_application):
    window = notes_window
    window.show()
    window.pipeline_scroll.setMaximumHeight(260)

    for index in range(len(PANEL_NOTE_KEYS)):
        window.show_pipeline_section(index)
        window.pipeline_notes_widget.set_expanded(True)
        window.pipeline_tab_content.adjustSize()
        qt_application.processEvents()
        assert window.pipeline_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
        assert window.pipeline_scroll.verticalScrollBar().maximum() > 0
