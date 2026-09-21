"""Main Qt window assembly and startup flow."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QChildEvent, QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QSlider,
    QTextEdit,
    QWidget,
)

from cellonaut.config.defaults import (
    APP_WINDOW_DEFAULT_HEIGHT,
    APP_WINDOW_DEFAULT_WIDTH,
    APP_WINDOW_MIN_HEIGHT,
    APP_WINDOW_MIN_WIDTH,
)
from cellonaut.config.state import EditableGuiConfiguration
from cellonaut.gui.build import CellonautGuiBuildMixin
from cellonaut.gui.config import CellonautGuiConfigMixin
from cellonaut.gui.dataset import CellonautGuiDatasetMixin
from cellonaut.gui.guided_tutorial import GuidedTutorialMixin
from cellonaut.gui.file_browser import CellonautGuiFileBrowserMixin
from cellonaut.gui.logging import CellonautGuiLoggingMixin
from cellonaut.gui.nd2 import CellonautGuiNd2Mixin
from cellonaut.gui.preview import CellonautGuiPreviewMixin
from cellonaut.gui.results import CellonautGuiResultsMixin
from cellonaut.gui.state import GuiTaskState, PreviewState
from cellonaut.gui.run_progress import CellonautGuiRunProgressMixin
from cellonaut.gui.top_status import CellonautGuiTopStatusMixin
from cellonaut.gui.validation import CellonautGuiValidationMixin
from cellonaut.gui.workers import CellonautGuiWorkersMixin
from cellonaut.resources import APP_ICON_FILE, get_resource_path
from cellonaut.version import format_app_version


class CellonautMainWindow(
    CellonautGuiResultsMixin,
    CellonautGuiValidationMixin,
    CellonautGuiPreviewMixin,
    CellonautGuiDatasetMixin,
    CellonautGuiTopStatusMixin,
    CellonautGuiRunProgressMixin,
    CellonautGuiBuildMixin,
    GuidedTutorialMixin,
    CellonautGuiLoggingMixin,
    CellonautGuiWorkersMixin,
    CellonautGuiFileBrowserMixin,
    CellonautGuiNd2Mixin,
    CellonautGuiConfigMixin,
    QMainWindow,
):
    """Main Cellonaut window.

    The GUI is built in stages:
    - construct default state
    - build widgets/layout
    - restore saved settings and presets
    - keep dependent selectors synchronized with dynamic channel and mask rows
    """

    nd2_worker_finished_on_gui = Signal(dict)
    pipeline_worker_finished_on_gui = Signal(dict)
    pipeline_status_on_gui = Signal(dict)
    preview_worker_finished_on_gui = Signal(dict)
    preview_status_on_gui = Signal(dict)
    input_scan_result_on_gui = Signal(dict)
    input_scan_finished_on_gui = Signal()
    setup_check_result_on_gui = Signal(dict)

    # These existing editor names are views of one owner, never stored copies.
    @property
    def image_definitions(self) -> list[dict[str, Any]]:
        return self.configuration_state.image_definitions

    @image_definitions.setter
    def image_definitions(self, value: list[dict[str, Any]]) -> None:
        self.configuration_state.image_definitions = value

    @property
    def measurement_options(self) -> dict[str, bool]:
        return self.configuration_state.measurement_options

    @measurement_options.setter
    def measurement_options(self, value: dict[str, bool]) -> None:
        self.configuration_state.measurement_options = value

    @property
    def pipeline_panel_notes(self) -> dict[str, str]:
        return self.configuration_state.panel_notes

    @pipeline_panel_notes.setter
    def pipeline_panel_notes(self, value: dict[str, str]) -> None:
        self.configuration_state.panel_notes = value

    # Mixins expect their shared state to exist before any widgets connect signals during build_ui.
    def __init__(self):
        super().__init__()

        self.setWindowTitle(format_app_version())

        icon_path = get_resource_path(APP_ICON_FILE)
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.resize(APP_WINDOW_DEFAULT_WIDTH, APP_WINDOW_DEFAULT_HEIGHT)
        self.setMinimumSize(QSize(APP_WINDOW_MIN_WIDTH, APP_WINDOW_MIN_HEIGHT))
        self.setAcceptDrops(True)

        # Keep running-task state together for the GUI mixins.
        self.task_state = GuiTaskState()
        self._preview_was_cancelled = False

        self._close_retry_scheduled = False
        self._closing_requested = False

        # UI/session state
        self._log_buffer = []
        self._browser_history = []
        self._browser_forward_history = []
        self._browser_root_path = None
        self._active_preset_name = ""
        self._active_preset_snapshot = None
        self._handling_preset_selection = False
        self._preset_dirty = False
        self._primary_task_active = False
        self._last_auto_stack_key = ""
        self._last_input_scan_result = {}
        self._suppress_input_path_scan = False

        self.configuration_state = EditableGuiConfiguration()
        self.preview_state = PreviewState()
        # Scene references and presentation controls are owned by Qt, not the data model.
        self.preview_pixmap_item = None
        self._preview_layer_items = []
        self._preview_mask_selection_item = None
        self._preview_filter_item = None
        self._preview_population_items = {}
        self._preview_snapshot_item = None
        self._preview_focus_mode = False
        self._preview_inspector_auto_collapsed = False
        self._preview_tools_switching = False


        self._fiji_scan_debounce = QTimer(self)
        self._fiji_scan_debounce.setSingleShot(True)
        self._fiji_scan_debounce.setInterval(450)
        self._fiji_scan_debounce.timeout.connect(self.update_fiji_component_checklist)
        self._fiji_scan_thread: Any | None = None
        self._fiji_scan_worker: Any | None = None
        self._pending_fiji_scan_path = ""
        self._last_fiji_scan_result = {}
        self._setup_check_thread: Any | None = None
        self._setup_check_worker: Any | None = None
        self._pending_setup_check_result: dict | None = None

        self._rebuilding_image_tabs = False
        self._adding_image_tab = False

        self.nd2_detected_channel_names: list[str] = []
        self.nd2_detected_sizes: dict[str, int] = {}
        self.nd2_z_summary: dict[str, int | bool] = {}
        self.nd2_dataset_summary: dict[str, object] = {}
        self.nd2_detected_file_count = 0
        self.nd2_detected_first_display = ""
        self.nd2_channel_folder_state: dict[str, str] = {}
        self.nd2_channel_rows = []

        self.image_definitions = self.normalize_image_definitions(self.image_definitions)
        self.build_ui()
        self.commit_gui_edits()
        self.initialize_guided_tutorial()
        queued = Qt.ConnectionType.QueuedConnection
        self.nd2_worker_finished_on_gui.connect(self.on_nd2_worker_finished, queued)
        self.pipeline_worker_finished_on_gui.connect(self.on_pipeline_worker_finished, queued)
        self.pipeline_status_on_gui.connect(self.set_status_style, queued)
        self.preview_worker_finished_on_gui.connect(self.on_preview_worker_finished, queued)
        self.preview_status_on_gui.connect(self.set_status_style, queued)
        self.input_scan_result_on_gui.connect(self.apply_input_path_scan_result, queued)
        self.input_scan_finished_on_gui.connect(self.cleanup_input_path_scan, queued)
        self.setup_check_result_on_gui.connect(self.apply_setup_check_result, queued)
        self.apply_no_wheel_policy(self)
        self.configure_main_focus_order()

        self._preset_dirty_refresh_timer = QTimer(self)
        self._preset_dirty_refresh_timer.setSingleShot(True)
        self._preset_dirty_refresh_timer.setInterval(80)
        self._preset_dirty_refresh_timer.timeout.connect(self.refresh_preset_dirty_indicator)
        self.install_preset_edit_watchers(self.pipeline_tab_content)

        self.log_flush_timer = QTimer(self)
        self.log_flush_timer.timeout.connect(self.flush_log_buffer)
        self.log_flush_timer.start(120)
        self._startup_initialization_complete = False

    # Install the same lightweight watcher on the current editor tree; child-add
    # events extend coverage when channel and mask rows are rebuilt later.
    def install_preset_edit_watchers(self, root: QWidget) -> None:
        for widget in (root, *root.findChildren(QWidget)):
            widget.installEventFilter(self)
            if widget.property("cellonautPresetEditWatcher"):
                continue
            widget.setProperty("cellonautPresetEditWatcher", True)
            if isinstance(widget, QLineEdit):
                widget.textEdited.connect(self.queue_preset_dirty_refresh)
            elif isinstance(widget, QComboBox):
                widget.activated.connect(self.queue_preset_dirty_refresh)
            elif isinstance(widget, QAbstractButton):
                widget.clicked.connect(self.queue_preset_dirty_refresh)
            elif isinstance(widget, QAbstractSpinBox):
                widget.editingFinished.connect(self.queue_preset_dirty_refresh)
            elif isinstance(widget, QSlider):
                widget.sliderMoved.connect(self.queue_preset_dirty_refresh)
            elif isinstance(widget, (QPlainTextEdit, QTextEdit)):
                widget.textChanged.connect(self.queue_preset_dirty_refresh)

    # Wait for related edits to settle before checking for unsaved preset changes.
    def queue_preset_dirty_refresh(self, *_args) -> None:
        self._preset_dirty_refresh_timer.start()

    # Dynamic editors are rebuilt throughout the session, so detect new child
    # widgets as well as keyboard, mouse, wheel, drop, and focus-out edits.
    def eventFilter(self, watched, event):
        pipeline = getattr(self, "pipeline_tab_content", None)
        if isinstance(watched, QWidget) and isinstance(pipeline, QWidget):
            inside_pipeline = watched is pipeline or pipeline.isAncestorOf(watched)
            if inside_pipeline and isinstance(event, QChildEvent) and event.type() == QEvent.Type.ChildAdded:
                child = event.child()
                if isinstance(child, QWidget):
                    QTimer.singleShot(0, lambda child=child: self.install_preset_edit_watchers(child))
            user_event = event.type() in {
                QEvent.Type.KeyRelease,
                QEvent.Type.MouseButtonRelease,
                QEvent.Type.Wheel,
                QEvent.Type.Drop,
                QEvent.Type.FocusOut,
            }
            if inside_pipeline and user_event:
                self.queue_preset_dirty_refresh()
        return super().eventFilter(watched, event)

    # Reflow only the two compact command areas; scientific tables retain their
    # stable column geometry and continue to scroll normally.
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "top_status_frame"):
            self.update_pipeline_command_layout(self.width())
        if hasattr(self, "preview_inspector"):
            self.update_preview_responsive_layout()

    def set_preview_focus_mode(self, enabled: bool) -> None:
        """Let the image workspace temporarily occupy the complete application window."""
        enabled = bool(enabled)
        self._preview_focus_mode = enabled
        self.app_header_frame.setVisible(not enabled)
        self.left_container.setVisible(not enabled)
        self.bottom_status_frame.setVisible(not enabled)
        self.preview_fullscreen_button.setToolTip(
            "Restore normal workspace (Esc)" if enabled else "Expand preview workspace"
        )
        if self.preview_fullscreen_button.isChecked() != enabled:
            self.preview_fullscreen_button.blockSignals(True)
            self.preview_fullscreen_button.setChecked(enabled)
            self.preview_fullscreen_button.blockSignals(False)
        self.update_preview_responsive_layout()

    def keyPressEvent(self, event):
        if self._preview_focus_mode and event.key() == Qt.Key.Key_Escape:
            self.set_preview_focus_mode(False)
            event.accept()
            return
        super().keyPressEvent(event)

    # Explicit tab order keeps keyboard navigation predictable even though the
    # visible layout can reflow between one and two rows. QWidgetAction controls
    # belong to the overflow menu's native window and must not be chained to
    # main-window widgets; Qt handles keyboard navigation inside that menu.
    def configure_main_focus_order(self) -> None:
        ordered = [
            *self.primary_navigation_buttons,
            self.run_button,
            self.preview_pipeline_button,
            self.check_setup_button,
            self.nd2_import_button,
            self.cancel_button,
            self.reuse_existing_masks_checkbox,
            self.mask_source_report_button,
            self.preset_combo,
            self.save_to_preset_button,
            self.save_preset_as_button,
            self.revert_preset_button,
            self.preview_artifact_selector,
            self.preview_prev_sample_button,
            self.preview_next_sample_button,
            self.preview_inspector_toggle_button,
            self.preview_snapshot_toggle_button,
            self.preview_snapshot_capture_button,
            self.preview_fullscreen_button,
            self.preview_prev_artifact_button,
            self.preview_next_artifact_button,
            self.preview_zoom_in_button,
            self.preview_zoom_out_button,
            self.preview_metadata_button,
            self.preview_reset_view_button,
            self.browser_back_button,
            self.browser_forward_button,
            self.browser_up_button,
            self.browser_computer_button,
            self.browser_input_button,
            self.browser_output_button,
            self.browser_compare_runs_button,
            self.browser_open_location_button,
            self.browser_path_entry,
            self.browser_go_button,
            self.browser_browse_button,
            self.file_tree,
        ]
        for current, following in zip(ordered, ordered[1:]):
            if current.window() is self and following.window() is self:
                self.setTabOrder(current, following)

    # Load saved settings after styling is applied and all widgets exist.
    def finish_startup_initialization(self):
        if self._startup_initialization_complete:
            return
        self._startup_initialization_complete = True
        self.load_last_settings_if_available()
        self.load_default_preset_on_startup()
        if hasattr(self, "update_fiji_component_checklist"):
            self.update_fiji_component_checklist()
        self.validate_all_fields()

    # Qt destroys active QThreads unsafely, so shutdown waits for every worker family owned by the window.
    def _running_background_threads(self) -> list:
        threads = []
        for name in (
            "worker_thread",
            "preview_worker_thread",
            "nd2_worker_thread",
            "_nd2_scan_thread",
            "_input_scan_thread",
            "_fiji_scan_thread",
            "_setup_check_thread",
        ):
            thread = getattr(self, name, None)
            if thread is not None:
                try:
                    if thread.isRunning():
                        threads.append(thread)
                except RuntimeError:
                    continue
        return threads

    # Cooperative cancellation lets pipeline workers release Fiji, files, and child processes themselves.
    def _request_shutdown(self):
        for worker_name in (
            "worker",
            "preview_worker",
            "nd2_worker",
            "_nd2_scan_worker",
            "_input_scan_worker",
            "_fiji_scan_worker",
            "_setup_check_worker",
        ):
            worker = getattr(self, worker_name, None)
            request_cancel = getattr(worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()

    # Retry through Qt's event loop so thread-finished signals can run between close attempts.
    def _retry_close_when_idle(self):
        self._close_retry_scheduled = False
        self.close()

    # Stop timers and persist machine-level preferences only after background work has fully ended.
    def closeEvent(self, event):
        self._closing_requested = True
        self._pending_input_scan_path = ""
        self._pending_fiji_scan_path = ""
        for timer_name in ("_input_scan_debounce", "_input_scan_status_timeout", "_fiji_scan_debounce"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        self._request_shutdown()

        if self._running_background_threads():
            event.ignore()
            if not self._close_retry_scheduled:
                self._close_retry_scheduled = True
                QTimer.singleShot(100, self._retry_close_when_idle)
            return

        for timer_name in (
            "log_flush_timer",
            "_preset_dirty_refresh_timer",
            "_preset_feedback_timer",
            "_fiji_scan_debounce",
            "_input_scan_debounce",
            "_worker_status_heartbeat",
            "run_elapsed_timer",
        ):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        if self._startup_initialization_complete:
            self.save_last_settings()
        self.reset_preview_display(clear_title=True)
        event.accept()

    # Use one local-file selector for all drag events so hover feedback matches the eventual drop behavior.
    @staticmethod
    def _first_local_drop_path(event) -> str:
        if not event.mimeData().hasUrls():
            return ""
        for url in event.mimeData().urls():
            if url.isLocalFile() and url.toLocalFile():
                return str(url.toLocalFile())
        return ""

    # Accept drag entry only when dropEvent can actually open one of the supplied paths.
    def dragEnterEvent(self, event):
        self._update_local_drag_acceptance(event)

    # Recheck during movement because Qt can replace MIME data while a drag crosses widgets.
    def dragMoveEvent(self, event):
        self._update_local_drag_acceptance(event)

    def _update_local_drag_acceptance(self, event) -> None:
        """Keep drag-entry and drag-movement acceptance rules identical."""
        if self._first_local_drop_path(event):
            event.acceptProposedAction()
            return
        event.ignore()

    # Open only the first usable local path because the right panel displays one selected artifact at a time.
    def dropEvent(self, event):
        path = self._first_local_drop_path(event)
        if not path:
            event.ignore()
            return
        self.open_file_in_right_panel(path)
        event.acceptProposedAction()
