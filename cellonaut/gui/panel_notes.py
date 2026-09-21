"""Preset-backed notes editor shared by pipeline and preview-tool panels."""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from cellonaut.gui.ui_tokens import SPACING

PANEL_NOTE_KEYS = (
    "input_nd2",
    "channels_masks",
    "cellpose",
    "processing",
    "measurements",
    "image_preview_tools",
)
COLLAPSED_NOTE_LINES = 3
MAX_NOTE_LINES = 20


def limit_note_lines(value: object, *, maximum: int = MAX_NOTE_LINES) -> str:
    """Normalize newline styles and retain at most the requested number of lines."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(text.split("\n")[: max(1, int(maximum))])


def normalize_panel_notes(value: object) -> dict[str, str]:
    """Return the complete, bounded panel-note mapping stored in presets."""
    source = value if isinstance(value, Mapping) else {}
    return {key: limit_note_lines(source.get(key, "")) for key in PANEL_NOTE_KEYS}


class LimitedNoteEdit(QPlainTextEdit):
    """Plain-text editor that preserves existing text when enforcing a line cap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._enforcing_limit = False
        self.setAcceptDrops(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTabChangesFocus(True)
        self.textChanged.connect(self._enforce_line_limit)

    def set_limited_text(self, value: object) -> None:
        """Replace the editor contents without emitting an intermediate over-limit value."""
        previous = self.blockSignals(True)
        try:
            self.setPlainText(limit_note_lines(value))
        finally:
            self.blockSignals(previous)

    def _insert_limited_text(self, value: str) -> None:
        """Insert as much text as fits while leaving all existing unselected lines intact."""
        normalized = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        cursor = self.textCursor()
        current = self.toPlainText()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        remaining = current[:start] + current[end:]
        available_newlines = max(0, (MAX_NOTE_LINES - 1) - remaining.count("\n"))
        if normalized.count("\n") > available_newlines:
            normalized = "\n".join(normalized.split("\n")[: available_newlines + 1])
        cursor.insertText(normalized)
        self.setTextCursor(cursor)

    def insertFromMimeData(self, source) -> None:  # noqa: N802 - Qt override
        if source.hasText():
            self._insert_limited_text(source.text())
            return
        super().insertFromMimeData(source)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self._insert_limited_text("\n")
            return
        super().keyPressEvent(event)

    def _enforce_line_limit(self) -> None:
        """Bound text from input methods that bypass key and paste insertion hooks."""
        if self._enforcing_limit:
            return
        text = self.toPlainText()
        limited = limit_note_lines(text)
        if text == limited:
            return
        self._enforcing_limit = True
        try:
            cursor = self.textCursor()
            position = min(cursor.position(), len(limited))
            self.setPlainText(limited)
            cursor = self.textCursor()
            cursor.setPosition(position, QTextCursor.MoveMode.MoveAnchor)
            self.setTextCursor(cursor)
        finally:
            self._enforcing_limit = False


class PanelNotesWidget(QWidget):
    """Three-line notes editor that expands to the full twenty-line allowance."""

    noteChanged = Signal(str)
    expandedChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PanelNotesWidget")
        self.setProperty("uiRole", "panelNotes")
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACING.xs, 0, 0)
        layout.setSpacing(SPACING.xs)

        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("Notes")
        self.label.setProperty("uiRole", "panelNotesLabel")
        self.limit_label = QLabel("Up to 20 lines")
        self.limit_label.setProperty("uiRole", "mutedLabel")
        label_row.addWidget(self.label)
        label_row.addStretch(1)
        label_row.addWidget(self.limit_label)
        layout.addLayout(label_row)

        self.editor = LimitedNoteEdit()
        self.editor.setObjectName("PanelNotesEdit")
        self.editor.setAccessibleName("Panel notes")
        self.editor.setPlaceholderText("Add workflow notes for this preset and panel.")
        self.editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.editor.textChanged.connect(lambda: self.noteChanged.emit(self.editor.toPlainText()))
        layout.addWidget(self.editor)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("PanelNotesToggleButton")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setFixedSize(18, 16)
        self.toggle_button.clicked.connect(self.toggle_expanded)
        button_row.addWidget(self.toggle_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.set_expanded(False)

    def text(self) -> str:
        return self.editor.toPlainText()

    def set_text(self, value: object) -> None:
        self.editor.set_limited_text(value)

    def is_expanded(self) -> bool:
        return self._expanded

    def toggle_expanded(self) -> None:
        self.set_expanded(not self._expanded, notify=True)

    def set_expanded(self, expanded: bool, *, notify: bool = False) -> None:
        self._expanded = bool(expanded)
        visible_lines = MAX_NOTE_LINES if self._expanded else COLLAPSED_NOTE_LINES
        self.editor.setFixedHeight(self._height_for_lines(visible_lines))
        if self._expanded:
            self.toggle_button.setArrowType(Qt.ArrowType.UpArrow)
            self.toggle_button.setAccessibleName("Reduce panel notes")
            self.toggle_button.setToolTip("Reduce notes to 3 lines")
        else:
            self.toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.toggle_button.setAccessibleName("Expand panel notes")
            self.toggle_button.setToolTip("Expand notes to 20 lines")
        if notify:
            self.expandedChanged.emit(self._expanded)

    def _height_for_lines(self, lines: int) -> int:
        document_margin = int(round(self.editor.document().documentMargin()))
        frame = self.editor.frameWidth()
        return (self.editor.fontMetrics().lineSpacing() * lines) + (2 * document_margin) + (2 * frame) + SPACING.xs

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() in {QEvent.Type.FontChange, QEvent.Type.StyleChange} and hasattr(self, "editor"):
            self.set_expanded(self._expanded)
