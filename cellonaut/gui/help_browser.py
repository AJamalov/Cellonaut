"""Help rich text with rounded example backgrounds supported by Qt painting."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QTextTable
from PySide6.QtWidgets import QTextBrowser


class HelpBrowser(QTextBrowser):
    """Paint example tables behind the text; Qt HTML does not support border-radius."""

    def paintEvent(self, event):
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        dark = self.palette().color(QPalette.ColorRole.Base).lightness() < 128
        painter.setBrush(QColor("#383a3d" if dark else "#e2e3e5"))
        layout = self.document().documentLayout()
        # Help tables are reserved for examples. Text remains native and selectable.
        for frame in self.document().rootFrame().childFrames():
            if isinstance(frame, QTextTable):
                padding = frame.format().cellPadding()
                rect = layout.frameBoundingRect(frame).translated(
                    -self.horizontalScrollBar().value() - padding,
                    -self.verticalScrollBar().value() - padding,
                )
                rect.setWidth(rect.width() - frame.format().leftMargin() - frame.format().rightMargin())
                if rect.intersects(event.rect().toRectF()):
                    painter.drawRoundedRect(rect, 9, 9)
        painter.end()
        super().paintEvent(event)
