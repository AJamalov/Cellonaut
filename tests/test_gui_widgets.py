from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPointingDevice, QWheelEvent
from PySide6.QtWidgets import QApplication, QGraphicsScene

from cellonaut.gui.widgets import (
    CollapsibleSection,
    EntryRow,
    ImageGraphicsView,
    MatrixToggleButton,
    PathRow,
    ProcessingStepsTable,
)


pytestmark = pytest.mark.gui


def qt_app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


def mouse_event(
    event_type: QMouseEvent.Type,
    position: QPoint,
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
) -> QMouseEvent:
    point = QPointF(position)
    return QMouseEvent(
        event_type,
        point,
        point,
        point,
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
        QPointingDevice.primaryPointingDevice(),
    )


def wheel_event(delta: QPoint) -> QWheelEvent:
    return QWheelEvent(
        QPointF(20, 20),
        QPointF(20, 20),
        QPoint(),
        delta,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )


def test_form_rows_treat_none_as_blank_and_inherit_disabled_state():
    app = qt_app()
    path_row = PathRow("Folder", default=None)
    entry_row = EntryRow("Tag", default=None)
    try:
        path_row.set(None)
        entry_row.set(None)
        assert path_row.get() == ""
        assert entry_row.get() == ""

        path_row.setEnabled(False)
        entry_row.setEnabled(False)
        assert path_row.edit.isEnabled() is False
        assert path_row.button.isEnabled() is False
        assert entry_row.edit.isEnabled() is False
    finally:
        path_row.deleteLater()
        entry_row.deleteLater()
        app.processEvents()


def test_path_row_rejects_unknown_drop_mode():
    app = qt_app()
    with pytest.raises(ValueError, match="Unsupported path drop mode"):
        PathRow("Folder", mode="directory")
    app.processEvents()


def test_matrix_toggle_tracks_programmatic_changes_and_locks_unavailable_state():
    app = qt_app()
    button = MatrixToggleButton(False)
    try:
        button.setToolTip("Apply this processing step to Channel 1.")
        button.setChecked(True)
        assert button.text() == "ON"
        assert button.property("matrixToggleState") == "on"
        assert button.toolTip() == "Apply this processing step to Channel 1."

        button.set_unavailable()
        assert button.text() == "N/A"
        assert button.property("matrixToggleState") == "unavailable"
        assert button.isChecked() is False
        assert button.isEnabled() is False
    finally:
        button.deleteLater()
        app.processEvents()


def test_collapsible_section_tracks_programmatic_toggle_and_summary():
    app = qt_app()
    section = CollapsibleSection("Measurements", expanded=False, summary="4 outputs", step_number=6)
    try:
        assert section.step_badge.text() == "6"
        assert section.step_badge.parentWidget() is section.toggle_button
        assert section.step_badge.isWindow() is False
        assert section.summary_label.text() == "4 outputs"
        assert section.chevron_label.text() == "▶"
        section.toggle_button.setChecked(True)
        assert section.content.isHidden() is False
        assert section.chevron_label.text() == "▼"
        assert section.toggle_button.property("expanded") == "true"

        section.set_summary("5 outputs")
        section.toggle_button.setChecked(False)
        assert section.content.isHidden() is True
        assert section.chevron_label.text() == "▶"
        assert section.summary_label.text() == "5 outputs"

        section.set_overview_mode()
        assert section.toggle_button.isHidden() is False
        assert section.minimumHeight() == 72
        assert section.maximumHeight() == 72

        section.set_editor_mode()
        assert section.toggle_button.isHidden() is True
        assert section.content.isHidden() is False
        assert section.minimumHeight() == 0
        assert section.maximumHeight() == 16_777_215
    finally:
        section.deleteLater()
        app.processEvents()


def test_processing_drag_targets_only_editable_rows():
    app = qt_app()
    table = ProcessingStepsTable()
    try:
        table.setColumnCount(2)
        table.setRowCount(5)
        for row in range(table.rowCount()):
            table.setRowHeight(row, 24)
        table.resize(320, 180)
        table.show()
        app.processEvents()

        assert table._editable_row_at(-20) == 0
        assert table._editable_row_at(table.rowViewportPosition(2) + 5) == 2
        assert table._editable_row_at(table.rowViewportPosition(4) + 5) == 3
        assert table._editable_row_at(table.viewport().height() + 20) == 3
    finally:
        table.close()
        table.deleteLater()
        app.processEvents()


def test_processing_click_without_drag_does_not_reorder():
    app = qt_app()
    table = ProcessingStepsTable()
    reorder_calls: list[tuple[int, int]] = []
    try:
        table.setColumnCount(2)
        table.setRowCount(5)
        table.resize(320, 180)
        table.show()
        app.processEvents()
        table.step_column_drop_callback = lambda source, target: reorder_calls.append((source, target))

        x = table.columnViewportPosition(0) + 5
        source = QPoint(x, table.rowViewportPosition(1) + 5)
        target = QPoint(x, table.rowViewportPosition(3) + 5)
        table.mousePressEvent(
            mouse_event(QMouseEvent.Type.MouseButtonPress, source, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
        )
        table.mouseReleaseEvent(
            mouse_event(QMouseEvent.Type.MouseButtonRelease, target, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
        )

        assert reorder_calls == []
    finally:
        table.close()
        table.deleteLater()
        app.processEvents()


def test_processing_drag_reorders_using_recipe_indices():
    app = qt_app()
    table = ProcessingStepsTable()
    reorder_calls: list[tuple[int, int]] = []
    try:
        table.setColumnCount(2)
        table.setRowCount(5)
        table.resize(320, 180)
        table.show()
        app.processEvents()
        table.step_column_drop_callback = lambda source, target: reorder_calls.append((source, target))

        x = table.columnViewportPosition(0) + 5
        source = QPoint(x, table.rowViewportPosition(1) + 5)
        target = QPoint(x, table.rowViewportPosition(3) + 5)
        table.mousePressEvent(
            mouse_event(QMouseEvent.Type.MouseButtonPress, source, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
        )
        table.mouseMoveEvent(
            mouse_event(QMouseEvent.Type.MouseMove, target, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
        )
        table.mouseReleaseEvent(
            mouse_event(QMouseEvent.Type.MouseButtonRelease, target, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
        )

        assert reorder_calls == [(1, 3)]
    finally:
        table.close()
        table.deleteLater()
        app.processEvents()


def test_preview_zoom_ignores_horizontal_wheel_and_stays_bounded():
    app = qt_app()
    view = ImageGraphicsView()
    scene = QGraphicsScene(view)
    scene.addRect(0, 0, 100, 100)
    view.setScene(scene)
    zoom_updates = []
    view.zoom_changed.connect(zoom_updates.append)
    try:
        view.wheelEvent(wheel_event(QPoint(120, 0)))
        assert view._zoom == 0

        for _ in range(ImageGraphicsView.MAX_ZOOM_STEPS + 5):
            view.wheelEvent(wheel_event(QPoint(0, 120)))
        assert view._zoom == ImageGraphicsView.MAX_ZOOM_STEPS

        for _ in range(ImageGraphicsView.MAX_ZOOM_STEPS * 2 + 10):
            view.wheelEvent(wheel_event(QPoint(0, -120)))
        assert view._zoom == -ImageGraphicsView.MAX_ZOOM_STEPS
        assert zoom_updates

        view.reset_zoom()
        view.zoom_by_steps(1)
        assert view._zoom == 1
        assert view.zoom_percent() == 115

        view.resize(400, 300)
        view.fit_image()
        assert view.zoom_percent() == 100
        view.zoom_by_steps(1)
        assert view.zoom_percent() == 115
    finally:
        view.deleteLater()
        app.processEvents()
