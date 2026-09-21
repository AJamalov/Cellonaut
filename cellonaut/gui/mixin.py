"""Typing bridge for GUI mixins without adding another runtime Qt widget base.

Static analyzers need to know that each mixin ultimately operates on the main
window. At runtime, however, only ``CellonautMainWindow`` should inherit the Qt
widget class, so this module deliberately exposes different lightweight views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from cellonaut.config.state import EditableGuiConfiguration
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QGraphicsPixmapItem
    from PySide6.QtWidgets import QWidget
    from cellonaut.gui.state import GuiTaskState, ImageDefinitionRowWidgets, PreviewState

    class GuiMixin(QWidget):
        """Static view of behavior mixed into ``CellonautMainWindow``."""

        running: bool
        task_state: GuiTaskState
        configuration_state: EditableGuiConfiguration
        preview_state: PreviewState
        worker: Any
        worker_thread: Any
        preview_worker: Any
        preview_worker_thread: Any
        nd2_worker: Any
        nd2_worker_thread: Any
        _preview_was_cancelled: bool
        _close_retry_scheduled: bool
        @property
        def image_definitions(self) -> list[dict[str, Any]]: ...

        @image_definitions.setter
        def image_definitions(self, value: list[dict[str, Any]]) -> None: ...

        image_rows: list[ImageDefinitionRowWidgets | None]
        _rebuilding_image_tabs: bool
        _adding_image_tab: bool
        _inline_analysis_settings_row: int | None
        _inline_analysis_settings_panel: str
        preview_pixmap_item: QGraphicsPixmapItem | None
        _last_auto_stack_key: str
        nd2_detected_channel_names: list[str]

        # The main window creates many widgets dynamically during build_ui.
        # Keep undeclared attributes permissive while explicitly typing shared
        # state whose container or optional value shape matters to mixin code.
        def __getattr__(self, name: str) -> Any: ...

        # Centralize the cast needed by Qt parent arguments because a checker
        # cannot reconstruct the final multiple-inheritance window at this point.
        def as_qobject(self) -> QObject:
            return cast("QObject", self)

else:
    from cellonaut.gui.state import GuiTaskState, PreviewState

    class GuiMixin:
        """Runtime-neutral base for behavior mixed into the main window."""

        task_state: GuiTaskState
        preview_state: PreviewState

        def _get_task_state(self) -> GuiTaskState:
            state = getattr(self, "task_state", None)
            if state is None:
                state = GuiTaskState()
                self.task_state = state
            return state

        @property
        def running(self):
            return self._get_task_state().running

        @running.setter
        def running(self, value):
            self._get_task_state().running = bool(value)

        @property
        def worker(self):
            return self._get_task_state().pipeline.worker

        @worker.setter
        def worker(self, value):
            self._get_task_state().pipeline.worker = value

        @property
        def worker_thread(self):
            return self._get_task_state().pipeline.thread

        @worker_thread.setter
        def worker_thread(self, value):
            self._get_task_state().pipeline.thread = value

        @property
        def preview_worker(self):
            return self._get_task_state().preview.worker

        @preview_worker.setter
        def preview_worker(self, value):
            self._get_task_state().preview.worker = value

        @property
        def preview_worker_thread(self):
            return self._get_task_state().preview.thread

        @preview_worker_thread.setter
        def preview_worker_thread(self, value):
            self._get_task_state().preview.thread = value

        @property
        def nd2_worker(self):
            return self._get_task_state().nd2.worker

        @nd2_worker.setter
        def nd2_worker(self, value):
            self._get_task_state().nd2.worker = value

        @property
        def nd2_worker_thread(self):
            return self._get_task_state().nd2.thread

        @nd2_worker_thread.setter
        def nd2_worker_thread(self, value):
            self._get_task_state().nd2.thread = value

        @property
        def _preview_was_cancelled(self):
            return self._get_task_state().preview_was_cancelled

        @_preview_was_cancelled.setter
        def _preview_was_cancelled(self, value):
            self._get_task_state().preview_was_cancelled = bool(value)

        @property
        def _close_retry_scheduled(self):
            return self._get_task_state().close_retry_scheduled

        @_close_retry_scheduled.setter
        def _close_retry_scheduled(self, value):
            self._get_task_state().close_retry_scheduled = bool(value)

        # Return the composed window unchanged; the real QObject inheritance is
        # supplied once by QMainWindow later in the runtime method-resolution order.
        def as_qobject(self):
            return self
