from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QColor, QPalette, QTextTable
from PySide6.QtWidgets import QApplication

from cellonaut.gui.help_browser import HelpBrowser


pytestmark = pytest.mark.gui


@pytest.mark.parametrize(
    ("base", "text", "box"), [("#ffffff", "#182230", "#e2e3e5"), ("#1e1e1e", "#f3f3f3", "#383a3d")]
)
def test_example_background_has_rounded_corners_and_preserves_text(base, text, box):
    instance = QApplication.instance()
    app = instance if isinstance(instance, QApplication) else QApplication([])
    previous_stylesheet = app.styleSheet()
    app.setStyleSheet("")
    browser = HelpBrowser()
    palette = browser.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor(base))
    palette.setColor(QPalette.ColorRole.Text, QColor(text))
    browser.setPalette(palette)
    browser.resize(420, 300)
    browser.setHtml(
        '<style>body { margin: 10px; }</style><p>Introduction</p>'
        '<table width="100%" cellspacing="0" cellpadding="12">'
        "<tr><td><h3>Example</h3><p>Selectable instructions.</p></td></tr></table><p>More help.</p>"
    )
    browser.show()
    try:
        app.processEvents()
        table = next(f for f in browser.document().rootFrame().childFrames() if isinstance(f, QTextTable))
        frame = browser.document().documentLayout().frameBoundingRect(table)
        x = round(frame.left() - table.format().cellPadding())
        y = round(frame.top() - table.format().cellPadding())
        image = browser.viewport().grab().toImage()
        # The upper edge is grey, but the corner keeps the page background.
        assert image.pixelColor(x + 15, y + 2).name() == box
        assert image.pixelColor(x + 1, y + 1).name() == base
        right = round(x + frame.width() - table.format().leftMargin() - table.format().rightMargin())
        assert image.pixelColor(right - 15, y + 2).name() == box
        assert image.pixelColor(right - 2, y + 1).name() == base
        assert browser.find("Selectable instructions.")
        assert browser.textCursor().selectedText() == "Selectable instructions."
    finally:
        browser.close()
        app.setStyleSheet(previous_stylesheet)
