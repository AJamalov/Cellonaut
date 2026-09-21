from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QTextEdit

from cellonaut.gui.logging import (
    LIGHT_LOG_LINE_STYLES,
    LOG_LINE_STYLES,
    MAX_LOG_BLOCKS,
    CellonautGuiLoggingMixin,
    current_log_line_styles,
)


pytestmark = pytest.mark.gui


class LoggingHarness(CellonautGuiLoggingMixin):
    def __init__(self):
        self._log_buffer: list[str] = []
        self.log_box = QTextEdit()


# Narrow PySide's application type once so palette-specific test calls remain typed.
def get_application() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


# Hold one application for the module because Qt widgets cannot outlive their application object.
@pytest.fixture(scope="module", autouse=True)
def application():
    app = get_application()
    previous_theme = app.property("cellonautTheme")
    app.setProperty("cellonautTheme", "dark_blue")
    yield app
    app.setProperty("cellonautTheme", previous_theme)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("[PREVIEW][ERROR] failed", "error"),
        ("[ND2][WARN] incomplete export", "warning"),
        ("Could not save report", "warning"),
        ("Classifier failed", "error"),
        ("[PREVIEW] Trying sample", "preview"),
        ("[ND2] Reading metadata", "nd2"),
        ("[QC] Applying filters", "qc"),
        ("Running classifier", "weka"),
        ("[CELLPOSE] Segmenting cells", "cellpose"),
        ("Saved measurement table", "export"),
        ("Sample finished successfully", "success"),
        ("[RUN] Starting pipeline", "progress"),
        ("Ordinary detail", "default"),
    ],
)
def test_log_line_kind_prioritizes_severity(message, expected):
    assert LoggingHarness().log_line_kind(message) == expected


def test_colorized_log_line_escapes_content_and_uses_inline_color():
    rendered = LoggingHarness().colorize_log_line("[WARN] a < b & c")

    assert 'class="log-warning"' in rendered
    assert 'style="color:#ffd166; font-weight:700;' in rendered
    assert "[WARN] a &lt; b &amp; c" in rendered


def test_configure_log_box_applies_log_palette_and_history_limit():
    app = get_application()
    harness = LoggingHarness()
    try:
        harness.configure_log_box()

        stylesheet = harness.log_box.document().defaultStyleSheet()
        assert LOG_LINE_STYLES["default"][0] in stylesheet
        assert LOG_LINE_STYLES["error"][0] in stylesheet
        assert LOG_LINE_STYLES["weka"][0] in stylesheet
        assert harness.log_box.document().maximumBlockCount() == MAX_LOG_BLOCKS
    finally:
        harness.log_box.deleteLater()
        app.processEvents()


def test_light_theme_uses_high_contrast_log_colors():
    app = get_application()
    previous_theme = app.property("cellonautTheme")
    try:
        app.setProperty("cellonautTheme", "light_blue")
        assert current_log_line_styles() is LIGHT_LOG_LINE_STYLES
        assert LIGHT_LOG_LINE_STYLES["default"][0] != LOG_LINE_STYLES["default"][0]
        assert LIGHT_LOG_LINE_STYLES["error"][0] != LOG_LINE_STYLES["error"][0]
    finally:
        app.setProperty("cellonautTheme", previous_theme)


def test_live_log_fragments_keep_inline_colors():
    app = get_application()
    original_stylesheet = app.styleSheet()
    harness = LoggingHarness()
    try:
        app.setStyleSheet("QTextEdit { color: #ffffff; }")
        harness.configure_log_box()
        harness.append_colored_log_lines(["[WARN] warning", "Running classifier", "Saved output"])

        colors = []
        block = harness.log_box.document().firstBlock()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid() and fragment.text().strip():
                    colors.append(fragment.charFormat().foreground().color().name())
                iterator += 1
            block = block.next()

        assert colors == ["#ffd166", "#f4a261", "#a7c957"]
    finally:
        app.setStyleSheet(original_stylesheet)
        harness.log_box.deleteLater()
        app.processEvents()


def test_flush_log_buffer_appends_messages_and_clears_snapshot():
    harness = LoggingHarness()
    harness.log("first")
    harness.log("[WARN] second")

    harness.flush_log_buffer()

    assert harness._log_buffer == []
    plain_text = harness.log_box.toPlainText()
    assert "first" in plain_text
    assert "[WARN] second" in plain_text


def test_flush_log_buffer_reports_display_failure_to_stderr(monkeypatch, capsys):
    harness = LoggingHarness()
    harness.log("message that must remain diagnosable")

    def fail_append(_lines):
        raise RuntimeError("document unavailable")

    monkeypatch.setattr(harness, "append_colored_log_lines", fail_append)
    harness.flush_log_buffer()

    captured = capsys.readouterr()
    assert "[GUI LOG ERROR]" in captured.err
    assert "document unavailable" in captured.err
    assert "message that must remain diagnosable" in captured.err
    assert harness._log_buffer == []


def test_flush_before_log_widget_exists_keeps_messages_queued():
    harness = LoggingHarness()
    harness.log_box.deleteLater()
    del harness.log_box
    harness.log("early startup message")

    harness.flush_log_buffer()

    assert harness._log_buffer == ["early startup message"]
