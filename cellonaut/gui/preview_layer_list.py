"""Layer-list widget used by the interactive preview workspace."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QDrag, QPixmap
from PySide6.QtWidgets import QBoxLayout, QLabel, QListWidget


class PreviewLayerList(QListWidget):
    """Reorder layers with only the drop indicator, not Qt's drag ghost."""

    @staticmethod
    def drop_categories_match(source_category: str, target_category: str) -> bool:
        return bool(source_category and source_category == target_category)

    def mousePressEvent(self, event):
        """Let a second click on the current layer clear its selection."""
        item = self.itemAt(event.position().toPoint())
        if item is not None and item is self.currentItem() and item.isSelected():
            self.clearSelection()
            self.setCurrentRow(-1)
            event.accept()
            return
        super().mousePressEvent(event)

    # Hide Qt's drag image because it obscures the small list and precise drop position.
    def startDrag(self, _supported_actions):
        indexes = self.selectedIndexes()
        if not indexes:
            return

        self._drag_category = str(self.currentItem().data(Qt.ItemDataRole.UserRole + 1) or "")
        try:
            drag = QDrag(self)
            drag.setMimeData(self.model().mimeData(indexes))
            transparent = QPixmap(1, 1)
            transparent.fill(Qt.GlobalColor.transparent)
            drag.setPixmap(transparent)
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            self._drag_category = ""

    def dropEvent(self, event):
        """Reject drops into a different layer category."""
        target = self.itemAt(event.position().toPoint())
        target_category = str(target.data(Qt.ItemDataRole.UserRole + 1) or "") if target is not None else ""
        if not self.drop_categories_match(self._drag_category, target_category):
            event.ignore()
            return
        super().dropEvent(event)

    def refresh_category_headers(self) -> None:
        """Keep each category heading attached to its current first layer."""
        previous_category = ""
        for row in range(self.count()):
            item = self.item(row)
            category = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
            row_widget = self.itemWidget(item)
            if row_widget is None:
                previous_category = category
                continue

            layout = row_widget.layout()
            if not isinstance(layout, QBoxLayout):
                previous_category = category
                continue
            old_headers = [
                label for label in row_widget.findChildren(QLabel) if label.property("uiRole") == "previewLayerGroup"
            ]
            had_header = bool(old_headers)
            for label in old_headers:
                layout.removeWidget(label)
                label.setParent(None)
                label.deleteLater()

            needs_header = bool(category and category != previous_category)
            if needs_header:
                group_label = QLabel(category.upper())
                group_label.setProperty("uiRole", "previewLayerGroup")
                layout.insertWidget(0, group_label)

            if had_header != needs_header:
                hint = item.sizeHint()
                height = max(1, hint.height() + (16 if needs_header else -16))
                item.setSizeHint(QSize(hint.width(), height))
                row_widget.setMinimumHeight(height)
            previous_category = category
