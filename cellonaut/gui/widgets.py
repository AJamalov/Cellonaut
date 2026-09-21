"""Reusable Qt widgets used across Cellonaut's setup and preview panels.

The widgets in this module wrap common path rows, validated inputs, drag/drop
behavior, and preview controls so the larger GUI mixins stay focused on layout
and state flow.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QBrush, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGraphicsRectItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTabBar,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from cellonaut.gui.ui_tokens import SPACING, WORKSPACE


def processing_value_from_widget(widget, fallback: str = "") -> str:
    """Serialize a recipe control to the text stored in presets."""
    value_getter = getattr(widget, "processing_value", None)
    if callable(value_getter):
        return str(value_getter()).strip()
    if isinstance(widget, QComboBox):
        data = widget.currentData()
        return str(widget.currentText() if data is None else data).strip()
    if isinstance(widget, QLineEdit):
        return widget.text().strip()
    if isinstance(widget, QPushButton) and widget.isCheckable():
        return "true" if widget.isChecked() else "false"
    return str(fallback or "").strip()


def make_blocked_processing_value_cell(tooltip: str) -> QLineEdit:
    """Create the disabled placeholder used by unselected recipe rows."""
    blocked = QLineEdit("")
    blocked.setProperty("processingRole", "blocked_value")
    blocked.setEnabled(False)
    blocked.setToolTip(tooltip)
    return blocked


def fit_table_height(
    table: QTableWidget,
    *,
    minimum: int = 112,
    maximum: int = 320,
    reserve_horizontal_scrollbar: bool = True,
) -> None:
    """Fit short scientific tables to their rows and cap tall ones for scrolling."""
    row_height = table.verticalHeader().defaultSectionSize()
    rows_height = sum(max(row_height, table.rowHeight(row)) for row in range(table.rowCount()))
    header_height = max(30, table.horizontalHeader().height()) if not table.horizontalHeader().isHidden() else 0
    scrollbar_height = max(12, table.horizontalScrollBar().sizeHint().height()) if reserve_horizontal_scrollbar else 0
    desired = header_height + rows_height + scrollbar_height + (2 * table.frameWidth()) + SPACING.xs
    fitted = max(minimum, min(maximum, desired))
    table.setMinimumHeight(fitted)
    table.setMaximumHeight(fitted)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def fit_processing_table_columns(table: QTableWidget, *, control_width: int) -> None:
    """Prioritize recipe controls while keeping several channel values visible."""
    if table.columnCount() <= 0:
        return
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    table.horizontalHeader().setSectionsMovable(False)
    table.setColumnWidth(0, control_width)
    if table.columnCount() > 1:
        table.setColumnWidth(1, 64)
    for column in range(2, table.columnCount()):
        table.setColumnWidth(column, max(88, min(108, table.columnWidth(column))))


# Image and mask recipes share one hierarchical selector implementation while
# retaining their own vocabularies, labels, roles, and styling callbacks.
def build_processing_step_menu_button(
    *,
    role: str,
    step_type: str,
    definitions: dict[str, dict],
    options: list[tuple[str, str]],
    categories: list[str],
    category_for_type: Callable[[str], str],
    label_for_definition: Callable[[dict], str],
    selector_width: Callable[[QWidget], int],
    activate: Callable,
    refresh_style: Callable[[QWidget], None],
) -> QToolButton:
    button = QToolButton()
    button.setProperty("processingRole", role)
    button.setProperty("stepType", str(step_type or ""))
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setArrowType(Qt.ArrowType.NoArrow)
    button.setFixedHeight(28)
    button.setFixedWidth(selector_width(button))
    selected = definitions.get(str(step_type or ""), {})
    button.setText(label_for_definition(selected) if selected else "Choose step")

    root_menu = QMenu(button)
    menus: dict[str, QMenu] = {"": root_menu}
    option_categories = {category_for_type(option_type) for option_type, _label in options}
    used_paths = {
        path
        for category in option_categories
        for path in [" > ".join(category.split(" > ")[:index]) for index in range(1, len(category.split(" > ")) + 1)]
    }
    for category in categories:
        if category == "Other":
            continue
        if category not in used_paths:
            continue
        parent_path = ""
        for part in category.split(" > "):
            current_path = part if not parent_path else f"{parent_path} > {part}"
            if current_path not in menus:
                menus[current_path] = menus[parent_path].addMenu(part)
            parent_path = current_path

    for option_type, option_label in options:
        target_menu = menus.get(category_for_type(option_type), root_menu)
        action = QAction(option_label, target_menu)
        action.setData(option_type)
        target_menu.addAction(action)

    def choose(action: QAction) -> None:
        chosen_type = str(action.data() or "")
        if chosen_type:
            activate(button, chosen_type, label_for_definition(definitions.get(chosen_type, {})), refresh_style)

    root_menu.triggered.connect(choose)
    button.setMenu(root_menu)
    refresh_style(button)
    return button


class DropPathLineEdit(QLineEdit):
    """Line edit that accepts dragged local files/folders."""

    pathDropped = Signal(str)

    # Validate the mode at construction so a misspelled mode cannot silently
    # weaken a file-only or folder-only drop target.
    def __init__(self, default: object = "", mode: str = "any", parent=None):
        super().__init__("" if default is None else str(default), parent)
        if mode not in {"any", "dir", "file"}:
            raise ValueError(f"Unsupported path drop mode: {mode}")
        self.drop_mode = mode
        self.setAcceptDrops(True)

    def _path_is_allowed(self, path: Path) -> bool:
        if self.drop_mode == "dir":
            return path.is_dir()
        if self.drop_mode == "file":
            return path.is_file()
        return path.exists()

    # Prefer the first usable local URL so mixed clipboard payloads cannot put
    # a web URL or an invalid path into a filesystem setting.
    def _first_allowed_path(self, event):
        if not event.mimeData().hasUrls():
            return None

        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue

            path = Path(url.toLocalFile())
            if self._path_is_allowed(path):
                return path

        return None

    def dragEnterEvent(self, event):
        if self._first_allowed_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._first_allowed_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    # Emit a dedicated signal in addition to changing the text so callers can
    # distinguish a deliberate drop from ordinary typing when needed.
    def dropEvent(self, event):
        path = self._first_allowed_path(event)
        if path is None:
            event.ignore()
            return

        text = str(path)
        self.setText(text)
        self.pathDropped.emit(text)
        event.acceptProposedAction()


class BaseRow(QWidget):
    """Base widget for labeled form rows used throughout the GUI."""

    LABEL_WIDTH = 150

    def make_label(self, text: str, tooltip: str = "", width: int | None = None) -> QWidget:
        container = QWidget()
        container.setProperty("uiRole", "formLabelPanel")
        label_width = self.LABEL_WIDTH if width is None else width
        container.setMinimumWidth(label_width)
        container.setMaximumWidth(label_width)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        text_label = QLabel(text)
        text_label.setProperty("uiRole", "formLabel")
        text_label.setToolTip(tooltip if tooltip else "")

        layout.addWidget(text_label)
        layout.addStretch(1)

        if tooltip:
            help_label = QLabel("?")
            help_label.setObjectName("HelpMarker")
            help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            help_label.setFixedSize(18, 18)
            help_label.setToolTip(tooltip)
            help_label.setCursor(Qt.CursorShape.WhatsThisCursor)

            layout.addWidget(help_label)

        return container

    def make_row_layout(self) -> QHBoxLayout:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        return layout


class PathRow(BaseRow):
    """Reusable labeled path input with a browse button."""

    def __init__(
        self,
        label: str,
        default: object = "",
        mode: str = "dir",
        file_filter: str = "All Files (*.*)",
        tooltip: str = "",
        label_width: int | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.file_filter = file_filter

        layout = self.make_row_layout()

        self.label = self.make_label(label, tooltip)
        self.edit = DropPathLineEdit(default, mode=mode)
        self.button = QPushButton("Browse")
        self.button.setProperty("uiRole", "compactFormButton")
        self.button.setFixedWidth(72)
        self.button.clicked.connect(self.browse)

        if tooltip:
            self.edit.setToolTip(tooltip)
            self.button.setToolTip(tooltip)

        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def browse(self):
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select folder", self.get())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", self.get(), self.file_filter)

        if path:
            self.edit.setText(path)

    def get(self) -> str:
        return self.edit.text().strip()

    def set(self, value):
        self.edit.setText("" if value is None else str(value))


class EntryRow(BaseRow):
    """Reusable labeled single-line text entry row."""

    def __init__(
        self,
        label: str,
        default: object = "",
        tooltip: str = "",
        label_width: int | None = None,
    ):
        super().__init__()

        layout = self.make_row_layout()

        self.label = self.make_label(label, tooltip, label_width)
        self.edit = QLineEdit("" if default is None else str(default))

        if tooltip:
            self.edit.setToolTip(tooltip)

        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)

    def get(self) -> str:
        return self.edit.text().strip()

    def set(self, value):
        self.edit.setText("" if value is None else str(value))


class ComboRow(BaseRow):
    """Reusable labeled combo-box row."""

    # Selection controls align with other form rows and suppress wheel
    # changes that could alter a setting while the page is being scrolled.
    def __init__(
        self,
        label: str,
        values,
        default=None,
        tooltip: str = "",
        on_change=None,
        label_width: int | None = None,
    ):
        super().__init__()

        layout = self.make_row_layout()

        self.label = self.make_label(label, tooltip, label_width)
        self.combo = NoWheelComboBox()
        for value in values:
            if isinstance(value, tuple) and len(value) == 2:
                display, stored = value
                self.combo.addItem(str(display), str(stored))
            else:
                self.combo.addItem(str(value), str(value))

        if default is not None:
            idx = self.combo.findData(str(default))
            if idx < 0:
                idx = self.combo.findText(str(default))
            if idx >= 0:
                self.combo.setCurrentIndex(idx)

        if tooltip:
            self.combo.setToolTip(tooltip)
        if on_change is not None:
            self.combo.currentIndexChanged.connect(lambda _index: on_change(self.get()))

        layout.addWidget(self.label)
        layout.addWidget(self.combo, 1)

    def get(self) -> str:
        value = self.combo.currentData()
        return self.combo.currentText() if value is None else str(value)

    # Ignore unknown values so loading a bad setting cannot silently select an
    # unrelated first option.
    def set(self, value):
        idx = self.combo.findData(str(value))
        if idx < 0:
            idx = self.combo.findText(str(value))
        if idx >= 0:
            self.combo.setCurrentIndex(idx)


class MatrixToggleButton(QPushButton):
    """Green/red state toggle used in matrix tables."""

    # Store state in Qt properties so one theme rule can style the same compact
    # toggle in measurement and processing matrices.
    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("matrixToggle", "true")
        self.setMinimumWidth(64)
        self.setFixedHeight(24)
        self.setChecked(bool(checked))
        self.toggled.connect(self.update_visual_state)
        self.update_visual_state()

    # Re-polish after a dynamic property change because Qt stylesheets do not
    # always notice property updates on an existing widget.
    def refresh_theme_state(self):
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.update()

    # Make unavailable relationships non-interactive as well as visually muted
    # so their checked state cannot diverge from the underlying configuration.
    def set_unavailable(self):
        self.setProperty("matrixToggleState", "unavailable")
        self.setChecked(False)
        self.setEnabled(False)
        self.setText("N/A")
        self.setToolTip(
            "Unavailable because this column has no complete mask source.\n\n"
            "Add a classifier under an image in Masks, or configure a valid combined mask."
        )
        self.refresh_theme_state()

    # Synchronize text and style for both mouse clicks and programmatic changes
    # made while rebuilding a table from configuration.
    def update_visual_state(self, _checked: bool | None = None):
        if self.property("matrixToggleState") == "unavailable":
            return
        if self.isChecked():
            self.setText("ON")
            self.setProperty("matrixToggleState", "on")
        else:
            self.setText("OFF")
            self.setProperty("matrixToggleState", "off")
        self.refresh_theme_state()


class CollapsibleSection(QWidget):
    """Accordion card with a numbered header, summary, and persistent content."""

    OVERVIEW_HEIGHT = WORKSPACE.pipeline_row_height

    # Section contents remain alive while hidden so values and signal connections
    # survive repeated expansion without rebuilding expensive tables.
    def __init__(
        self,
        title: str,
        expanded: bool = False,
        *,
        summary: str = "",
        step_number: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self.setProperty("pipelineSection", "true")

        self.toggle_button = QPushButton()
        self.toggle_button.setProperty("uiRole", "pipelineSectionHeader")
        self.toggle_button.setAccessibleName(title)
        self.toggle_button.setFixedHeight(self.OVERVIEW_HEIGHT)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.toggled.connect(self.on_toggled)

        header_layout = QHBoxLayout(self.toggle_button)
        header_layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        header_layout.setSpacing(SPACING.md)

        # Assign the parent before making the badge visible. A parentless visible
        # QLabel is a native top-level window and otherwise flashes briefly for
        # every pipeline step during application startup.
        self.step_badge = QLabel(str(step_number or ""), self.toggle_button)
        self.step_badge.setProperty("uiRole", "pipelineStepBadge")
        self.step_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_badge.setFixedSize(24, 24)
        self.step_badge.setVisible(step_number is not None)

        self.title_label = QLabel(title)
        self.title_label.setProperty("uiRole", "pipelineSectionTitle")
        self.summary_label = QLabel(str(summary or ""))
        self.summary_label.setProperty("uiRole", "pipelineSectionSummary")
        self.summary_label.setMinimumWidth(0)
        self.summary_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.chevron_label = QLabel()
        self.chevron_label.setProperty("uiRole", "pipelineSectionChevron")
        self.chevron_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chevron_label.setFixedWidth(20)

        for label in (self.step_badge, self.title_label, self.summary_label, self.chevron_label):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.header_text_widget = QWidget()
        self.header_text_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.header_text_layout = QVBoxLayout(self.header_text_widget)
        self.header_text_layout.setContentsMargins(0, 0, 0, 0)
        self.header_text_layout.setSpacing(SPACING.xs)
        self.header_text_layout.addWidget(self.title_label)
        self.header_text_layout.addWidget(self.summary_label)

        header_layout.addWidget(self.step_badge)
        header_layout.addWidget(self.header_text_widget, 1)
        header_layout.addWidget(self.chevron_label)

        self.content = QWidget()
        self.content.setProperty("uiRole", "pipelineSectionContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.sm, SPACING.sm)
        self.content_layout.setSpacing(SPACING.sm)
        self.content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)
        self.on_toggled(expanded)
        self.set_summary(summary)

    # Listen to toggled rather than clicked so code-driven expansion updates the
    # label and content just like a user click.
    def on_toggled(self, expanded: bool | None = None):
        expanded = self.toggle_button.isChecked() if expanded is None else bool(expanded)
        self.chevron_label.setText("▼" if expanded else "▶")
        self.toggle_button.setProperty("expanded", "true" if expanded else "false")
        self.toggle_button.setToolTip(("Collapse " if expanded else "Expand ") + self._title)
        self.toggle_button.style().unpolish(self.toggle_button)
        self.toggle_button.style().polish(self.toggle_button)
        self.content.setVisible(expanded)

    def set_summary(self, summary: str) -> None:
        """Update the compact workflow status shown in the card header."""
        text = str(summary or "")
        self.summary_label.setText(text)
        self.summary_label.setToolTip(text)

    def set_overview_mode(self) -> None:
        """Constrain the navigation row so its two text lines cannot overlap."""
        self.toggle_button.show()
        self.content.hide()
        self.setMinimumHeight(self.OVERVIEW_HEIGHT)
        self.setMaximumHeight(self.OVERVIEW_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_editor_mode(self) -> None:
        """Release overview constraints so the selected editor can use its full height."""
        self.toggle_button.hide()
        self.content.show()
        self.setMinimumHeight(0)
        self.setMaximumHeight(16_777_215)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


class _ProcessingDragOverlay(QWidget):
    """Paint drag handles and drop indicators above table-cell widgets."""

    # Parent the overlay to the viewport so its coordinates always match table
    # rows even while the surrounding panel is resized.
    def __init__(self, table):
        super().__init__(table.viewport())
        self._table = table
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    # Paint outside cell widgets because those widgets otherwise hide the row
    # and insertion feedback supplied by QTableWidget.
    def paintEvent(self, event):
        super().paintEvent(event)
        table = self._table
        if table._drag_start_row <= 0 or table._drag_start_row >= table.rowCount() - 1:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if 0 <= table._drop_indicator_row < table.rowCount():
            source_rect = table.visualRect(table.model().index(table._drop_indicator_row, 0))
            ghost_rect = source_rect.adjusted(0, 1, table.viewport().width() - source_rect.width() - 1, -1)
            painter.fillRect(ghost_rect, QBrush(QColor(61, 139, 253, 70)))
            painter.setPen(QPen(QColor(61, 139, 253, 175), 1))
            painter.drawRect(ghost_rect)

        if 0 <= table._drop_indicator_row < table.rowCount():
            target_rect = table.visualRect(table.model().index(table._drop_indicator_row, 0))
            y = target_rect.top()
            if table._drop_indicator_row > table._drag_start_row:
                y = target_rect.bottom()
            painter.setPen(QPen(QColor("#ffd24a"), 3))
            painter.drawLine(0, y, table.viewport().width(), y)


class ProcessingStepsTable(QTableWidget):
    """Table used by the image-processing recipe editor."""

    # Drag state is separate from recipe data so this generic table can report
    # a reorder without knowing how image definitions are stored.
    def __init__(self, parent=None):
        super().__init__(parent)
        self.step_column_drop_callback: Callable[[int, int], None] | None = None
        self._drag_start_pos: QPoint | None = None
        self._drag_start_row = -1
        self._drop_indicator_row = -1
        self._drag_active = False
        self._drag_overlay = _ProcessingDragOverlay(self)
        self._drag_overlay.hide()

    # Convert Qt's floating-point event coordinates once so row and column
    # lookup use consistent integer viewport positions.
    @staticmethod
    def _event_position(event) -> QPoint:
        return event.position().toPoint()

    # Clamp drops to recipe rows; the final row is reserved for adding a step.
    def _editable_row_at(self, y: int) -> int:
        last_row = self.rowCount() - 2
        if last_row < 0:
            return -1

        target_row = self.rowAt(y)
        if target_row < 0:
            return 0 if y <= self.rowViewportPosition(0) else last_row
        return max(0, min(target_row, last_row))

    # Clear visual and pointer state in one place so cancelled and completed
    # drags cannot leak into the next mouse interaction.
    def _reset_drag_state(self):
        self._drag_start_pos = None
        self._drag_start_row = -1
        self._drop_indicator_row = -1
        self._drag_active = False
        self._drag_overlay.hide()
        self.viewport().update()

    # Only arm dragging from the handle column and an editable row, leaving
    # normal clicks in value cells untouched.
    def mousePressEvent(self, event):
        self._reset_drag_state()
        if event.button() == Qt.MouseButton.LeftButton:
            start_pos = self._event_position(event)
            source_row = self.rowAt(start_pos.y()) if self.columnAt(start_pos.x()) == 0 else -1
            if 0 <= source_row < self.rowCount() - 1:
                self._drag_start_pos = start_pos
                self._drag_start_row = source_row
                self._drop_indicator_row = source_row
        super().mousePressEvent(event)

    # Wait for Qt's normal drag distance before showing feedback so a slightly
    # unsteady click does not unexpectedly reorder scientific processing steps.
    def mouseMoveEvent(self, event):
        move_pos = self._event_position(event)
        if (
            self._drag_start_pos is not None
            and 0 <= self._drag_start_row < self.rowCount() - 1
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            if not self._drag_active:
                distance = (move_pos - self._drag_start_pos).manhattanLength()
                if distance < QApplication.startDragDistance():
                    super().mouseMoveEvent(event)
                    return
                self._drag_active = True

            target_row = self._editable_row_at(move_pos.y())
            self._drop_indicator_row = target_row
            self._drag_overlay.setGeometry(self.viewport().rect())
            self._drag_overlay.show()
            self._drag_overlay.raise_()
            self._drag_overlay.update()
            self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    # Report indices relative to recipe data, not the table's structural rows,
    # and only after a drag actually crossed the movement threshold.
    def mouseReleaseEvent(self, event):
        release_pos = self._event_position(event)
        target_row = self._editable_row_at(release_pos.y())
        source_row = self._drag_start_row
        was_dragging = self._drag_active
        self._reset_drag_state()

        if (
            was_dragging
            and self.step_column_drop_callback is not None
            and self.columnAt(release_pos.x()) == 0
            and source_row >= 0
            and target_row >= 0
            and source_row != target_row
            and source_row < self.rowCount() - 1
            and target_row < self.rowCount() - 1
        ):
            self.step_column_drop_callback(source_row, target_row)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    # Resize the transparent overlay with the viewport so drag feedback remains
    # aligned after column or window size changes.
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._drag_overlay.setGeometry(self.viewport().rect())


class FormSection(QGroupBox):
    """Standard form container used for most left-panel sections."""

    def __init__(self, title: str):
        super().__init__(title)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(6)

    def add_row(self, widget: QWidget):
        self._layout.addWidget(widget)

    def add_rows(self, widgets):
        for widget in widgets:
            self.add_row(widget)

    def add_widget(self, widget: QWidget):
        self._layout.addWidget(widget)

    def add_layout(self, layout):
        self._layout.addLayout(layout)


class SnapshotSquareItem(QGraphicsRectItem):
    """Movable, resizable square selection used for preview snapshots."""

    HANDLE_SIZE = 12.0
    MIN_SIZE = 8.0

    # The selection uses scene coordinates so captures remain accurate at
    # every preview zoom level.
    def __init__(self, scene_rect: QRectF, parent=None):
        super().__init__(parent)
        self._mode = ""
        self._press_scene_pos = QPointF()
        self._press_rect = QRectF()
        self._bounds = QRectF(scene_rect)
        self.setAcceptHoverEvents(True)
        self.setZValue(10_000)
        self.setPen(QPen(QColor("#ffd24a"), 2.0, Qt.PenStyle.SolidLine))
        self.setBrush(QBrush(QColor(255, 210, 74, 36)))
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setRect(self.initial_rect(scene_rect))

    # Start with a useful central crop while scaling naturally for both small
    # test images and large microscopy fields.
    def initial_rect(self, bounds: QRectF) -> QRectF:
        if not bounds.isValid() or bounds.isNull():
            return QRectF(0, 0, 128, 128)
        size = max(self.MIN_SIZE, min(bounds.width(), bounds.height()) * 0.35)
        x = bounds.left() + (bounds.width() - size) / 2.0
        y = bounds.top() + (bounds.height() - size) / 2.0
        return QRectF(x, y, size, size)

    # Re-clamp an existing selection when a new preview has different bounds so
    # stale coordinates cannot produce an empty crop.
    def set_bounds(self, bounds: QRectF):
        if not bounds.isValid() or bounds.isNull():
            return
        self._bounds = QRectF(bounds)
        self.setRect(self.clamp_rect(self.rect()))

    def handle_rect(self) -> QRectF:
        rect = self.rect()
        size = min(self.HANDLE_SIZE, rect.width(), rect.height())
        return QRectF(rect.right() - size, rect.bottom() - size, size, size)

    # Enforce a square fully inside the image because snapshot exports assume a
    # valid, equal-width crop and do not pad outside pixels.
    def clamp_rect(self, rect: QRectF) -> QRectF:
        bounds = self._bounds
        if not bounds.isValid() or bounds.isNull():
            return rect
        size = max(self.MIN_SIZE, min(rect.width(), rect.height()))
        size = min(size, bounds.width(), bounds.height())
        x = max(bounds.left(), min(rect.left(), bounds.right() - size))
        y = max(bounds.top(), min(rect.top(), bounds.bottom() - size))
        return QRectF(x, y, size, size)

    def hoverMoveEvent(self, event):
        if self.handle_rect().contains(event.pos()):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    # Capture both the starting pointer and rectangle so each move is calculated
    # from stable coordinates instead of accumulating rounding error.
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._mode = "resize" if self.handle_rect().contains(event.pos()) else "move"
        self._press_scene_pos = event.scenePos()
        self._press_rect = QRectF(self.rect())
        event.accept()

    # Clamp every intermediate position so the selection never disappears past
    # an image edge during a drag.
    def mouseMoveEvent(self, event):
        if self._mode not in {"move", "resize"}:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._press_scene_pos
        if self._mode == "move":
            rect = self._press_rect.translated(delta)
        else:
            grow = max(delta.x(), delta.y())
            size = max(self.MIN_SIZE, self._press_rect.width() + grow)
            rect = QRectF(self._press_rect.left(), self._press_rect.top(), size, size)
        self.setRect(self.clamp_rect(rect))
        event.accept()

    # Consume the release after an item drag so the graphics view does not also
    # treat it as the end of a pan operation.
    def mouseReleaseEvent(self, event):
        self._mode = ""
        event.accept()

    # Paint the handle after the standard item so it remains visible above the
    # translucent selection brush.
    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.fillRect(self.handle_rect(), QColor("#ffd24a"))


class ImageGraphicsView(QGraphicsView):
    """Graphics view with mouse-wheel zoom and fit/reset helpers for image previews."""

    MAX_ZOOM_STEPS = 20
    zoom_changed = Signal(int)

    # Anchor transformations beneath the pointer so users can inspect a region
    # without repeatedly panning after each zoom step.
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Scrollbars and preview controls can resize the viewport during file switches.
        # Keep its position instead of shifting the image toward the mouse each time.
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setBackgroundBrush(QColor("#202020"))
        self._zoom = 0
        self._fit_scale = 1.0

    # Ignore horizontal scrolling and bound both directions so trackpads cannot
    # accidentally invert or enlarge a preview beyond a usable scale.
    def wheelEvent(self, event):
        scene = self.scene()
        delta = event.angleDelta().y()
        if scene is None or not scene.items() or delta == 0:
            event.ignore()
            return

        direction = 1 if delta > 0 else -1
        next_zoom = self._zoom + direction
        if not -self.MAX_ZOOM_STEPS <= next_zoom <= self.MAX_ZOOM_STEPS:
            event.accept()
            return

        factor = 1.15 if direction > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom = next_zoom
        self.zoom_changed.emit(self.zoom_percent())
        event.accept()

    def zoom_percent(self) -> int:
        """Return zoom relative to the fitted image scale, where fit is 100%."""
        fit_scale = max(abs(self._fit_scale), 1e-9)
        return max(1, round(abs(self.transform().m11()) / fit_scale * 100))

    def zoom_by_steps(self, direction: int) -> None:
        """Apply one bounded zoom step for toolbar buttons."""
        scene = self.scene()
        if scene is None or not scene.items() or direction == 0:
            return
        step = 1 if direction > 0 else -1
        next_zoom = self._zoom + step
        if not -self.MAX_ZOOM_STEPS <= next_zoom <= self.MAX_ZOOM_STEPS:
            return
        self.scale(1.15 if step > 0 else 1 / 1.15, 1.15 if step > 0 else 1 / 1.15)
        self._zoom = next_zoom
        self.zoom_changed.emit(self.zoom_percent())

    def reset_zoom(self):
        self.resetTransform()
        self._fit_scale = 1.0
        self._zoom = 0
        self.zoom_changed.emit(self.zoom_percent())

    # Fit the complete scene rather than a particular item because overlays may
    # contain several aligned graphics objects.
    def fit_image(self):
        scene = self.scene()
        if scene is None:
            return
        items = scene.items()
        if not items:
            return
        rect = scene.itemsBoundingRect()
        if rect.isNull() or not rect.isValid():
            return
        scene.setSceneRect(rect)
        if getattr(self, "_preserve_view_during_load", False):
            return
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_scale = max(abs(self.transform().m11()), 1e-9)
        self._zoom = 0
        self.zoom_changed.emit(self.zoom_percent())


class NoWheelComboBox(QComboBox):
    """Combo box that ignores mouse-wheel changes unless the popup is open."""

    # Require an open popup before accepting wheel input so scrolling a settings
    # page cannot silently change the selected scientific option.
    def wheelEvent(self, event):
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()


class ElidedLabel(QLabel):
    """Single-line label that retains its complete value in a tooltip."""

    def __init__(self, text: str = "", parent=None, mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle):
        self._full_text = str(text or "")
        self._elide_mode = mode
        self._detail_tooltip = ""
        super().__init__("", parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._refresh_elision()

    def fullText(self) -> str:
        return self._full_text

    def setText(self, text: str) -> None:
        self._full_text = str(text or "")
        self._detail_tooltip = ""
        self._refresh_elision()

    def setDetailToolTip(self, text: str) -> None:
        self._detail_tooltip = str(text or "")
        self._refresh_elision()

    def _refresh_elision(self) -> None:
        width = max(0, self.contentsRect().width())
        displayed = self.fontMetrics().elidedText(self._full_text, self._elide_mode, width)
        super().setText(displayed)
        self.setToolTip(self._detail_tooltip or (self._full_text if displayed != self._full_text else ""))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_elision()


class NoWheelTabBar(QTabBar):
    """Tab bar that ignores mouse-wheel tab switching."""

    # Ignore wheel events so scrolling the page does not switch tabs.
    def wheelEvent(self, event):
        event.ignore()
