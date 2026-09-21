import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QPushButton

from cellonaut.gui.icons import ICON_ALIASES, LUCIDE_ELEMENTS, monochrome_icon
from cellonaut.gui.theme import THEMES


pytestmark = pytest.mark.gui


def test_monochrome_icon_exposes_normal_checked_and_disabled_states():
    app = QApplication.instance() or QApplication([])
    icon = monochrome_icon("pipeline", THEMES["light_blue"])

    assert icon.isNull() is False
    assert icon.pixmap(18, 18, QIcon.Mode.Normal, QIcon.State.Off).isNull() is False
    assert icon.pixmap(18, 18, QIcon.Mode.Normal, QIcon.State.On).isNull() is False
    assert icon.pixmap(18, 18, QIcon.Mode.Disabled, QIcon.State.Off).isNull() is False
    app.processEvents()


def test_all_shell_icon_names_render():
    app = QApplication.instance() or QApplication([])
    names = {
        "pipeline",
        "files",
        "log",
        "settings",
        "help",
        "book-open",
        "info",
        "lock",
        "pan",
        "fit",
        "zoom_out",
        "zoom_in",
        "previous",
        "next",
        "select",
        "snapshot",
        "fullscreen",
        "reset",
        "arrow-left",
        "arrow-right",
        "arrow-up",
        "folder-open",
        "hard-drive",
        "file-image",
        "monitor",
        "columns",
        "play",
        "image",
        "circle-check",
        "save",
        "refresh-cw",
        "triangle-alert",
        "trash-2",
    }
    assert all(name in LUCIDE_ELEMENTS or name in ICON_ALIASES for name in names)
    assert all(not monochrome_icon(name, THEMES["dark_blue"]).isNull() for name in names)
    app.processEvents()


def test_preview_icon_setup_preserves_detailed_tooltip():
    from cellonaut.gui.build import CellonautGuiBuildMixin

    app = QApplication.instance() or QApplication([])
    builder = CellonautGuiBuildMixin()
    button = QPushButton()
    button.setToolTip("Save a PNG from the snapshot square.")
    builder.configure_preview_icon_button(button, "snapshot", "Save snapshot")
    assert button.toolTip() == "Save a PNG from the snapshot square."
    assert button.accessibleName() == "Save snapshot"

    default_button = QPushButton()
    builder.configure_preview_icon_button(default_button, "zoom_in", "Zoom in")
    assert default_button.toolTip() == "Zoom in"
    app.processEvents()
