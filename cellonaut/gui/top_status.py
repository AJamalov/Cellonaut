"""Build and coordinate the pipeline command, preset, and progress controls."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
    QWidgetAction,
)
from cellonaut.gui.mixin import GuiMixin
from cellonaut.config.preset_store import is_default_preset
from cellonaut.gui.ui_tokens import SPACING, WORKSPACE
from cellonaut.resources import APP_ICON_FILE, get_resource_path
from cellonaut.version import __version__


# Centralize primary command state so run, preview, cancellation, and preset
# controls cannot disagree about whether a background operation is active.
class CellonautGuiTopStatusMixin(GuiMixin):
    """Manage persistent run controls, progress state, and preset commands."""

    # Global commands belong to the application shell, not to one navigation
    # page. Progress is built as a separate footer so it also remains visible.
    def build_top_status(self):
        self.app_header_frame = QFrame()
        self.app_header_frame.setObjectName("ApplicationHeader")
        self.app_header_frame.setMinimumHeight(WORKSPACE.header_height)
        self.top_status_frame = self.app_header_frame
        self.app_header_layout = QGridLayout(self.app_header_frame)
        self.app_header_layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        self.app_header_layout.setHorizontalSpacing(SPACING.md)
        self.app_header_layout.setVerticalSpacing(SPACING.xs)

        self.brand_widget = QWidget()
        brand_layout = QHBoxLayout(self.brand_widget)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(SPACING.sm)
        self.brand_icon_label = QLabel()
        self.brand_icon_label.setObjectName("ApplicationBrandIcon")
        self.brand_icon_label.setAccessibleName("Cellonaut application icon")
        self.brand_icon_label.setFixedSize(36, 36)
        self.brand_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_path = get_resource_path(APP_ICON_FILE)
        if icon_path.exists():
            self.brand_icon_label.setPixmap(QIcon(str(icon_path)).pixmap(32, 32))
        else:
            self.brand_icon_label.hide()
        self.brand_name_label = QLabel("Cellonaut")
        self.brand_name_label.setObjectName("ApplicationBrandName")
        self.brand_version_label = QLabel(f"Version {__version__}")
        self.brand_version_label.setObjectName("ApplicationBrandVersion")
        brand_layout.addWidget(self.brand_icon_label)
        brand_layout.addWidget(self.brand_name_label)
        self.brand_version_label.hide()

        self.pipeline_commands_widget = QWidget()
        command_row = QHBoxLayout(self.pipeline_commands_widget)
        command_row.setContentsMargins(0, 0, 0, 0)
        command_row.setSpacing(SPACING.sm)

        self.run_button = QPushButton("Run Pipeline")
        self.run_button.setToolTip("Run the complete pipeline with the current settings")
        self.run_button.setAccessibleName("Run pipeline")
        self.run_button.setProperty("cellonautIconName", "play")
        self.run_button.setIcon(self.shell_icon("play"))
        self.run_button.clicked.connect(self.run_clicked)
        self.set_button_role(self.run_button, "primary")
        self.run_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.run_button.setMinimumWidth(132)

        self.preview_pipeline_button = QPushButton("Preview One Sample")
        self.preview_pipeline_button.setToolTip(
            "Run the pipeline on the first valid sample and show the generated preview result."
        )
        self.preview_pipeline_button.setAccessibleName("Preview one sample")
        self.preview_pipeline_button.setProperty("cellonautIconName", "image")
        self.preview_pipeline_button.setIcon(self.shell_icon("image"))
        self.preview_pipeline_button.clicked.connect(self.preview_pipeline_clicked)
        self.preview_pipeline_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.preview_pipeline_button.setMinimumWidth(164)

        self.check_setup_button = QPushButton("Check Setup")
        self.check_setup_button.setToolTip(
            "Run the setup checks, sample detection, and measurement setup checks in one report."
        )
        self.check_setup_button.setAccessibleName("Check setup")
        self.check_setup_button.setProperty("cellonautIconName", "circle-check")
        self.check_setup_button.setIcon(self.shell_icon("circle-check"))
        self.check_setup_button.clicked.connect(self.show_setup_check_report)
        self.check_setup_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.check_setup_button.setMinimumWidth(118)

        self.reuse_existing_masks_checkbox = QCheckBox("Reuse existing masks")
        self.reuse_existing_masks_checkbox.setToolTip(
            "Skip mask generation when matching masks already exist in the selected mask source folder. "
            "Missing masks are generated with the current settings. Reused masks are not regenerated after detection settings change."
        )
        self.reuse_existing_masks_checkbox.setAccessibleName("Reuse existing masks")
        self.mask_source_report_button = QPushButton("Check existing masks")
        self.mask_source_report_button.setToolTip(
            "Report matching, missing, and shape-mismatched reusable masks."
        )
        self.mask_source_report_button.setAccessibleName("Check existing masks")
        self.mask_source_report_button.clicked.connect(self.show_mask_source_report)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setToolTip("Stop the active pipeline, preview, or ND2 conversion")
        self.cancel_button.setAccessibleName("Cancel active task")
        self.cancel_button.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self.cancel_button.clicked.connect(self.cancel_current_task)
        self.set_button_role(self.cancel_button, "danger")

        command_row.addWidget(self.run_button)
        command_row.addWidget(self.preview_pipeline_button)
        command_row.addWidget(self.check_setup_button)
        command_row.addWidget(self.cancel_button)
        command_row.addStretch(1)

        self.preset_commands_widget = QWidget()
        preset_row = QHBoxLayout(self.preset_commands_widget)
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(SPACING.sm)

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(0)
        self.preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.preset_combo.setAccessibleName("Preset")
        self.preset_combo.currentIndexChanged.connect(self.on_preset_selection_changed)

        self.preset_dirty_label = QLabel("*")
        self.preset_dirty_label.setObjectName("PresetDirtyLabel")
        self.preset_dirty_label.setProperty("messageState", "warning")
        self.preset_dirty_label.setToolTip("The selected preset has unsaved changes")
        self.preset_dirty_label.setAccessibleName("Unsaved preset changes")
        self.preset_dirty_label.setVisible(False)

        self.save_to_preset_button = QPushButton("Save")
        self.save_to_preset_button.setProperty("cellonautIconName", "save")
        self.save_to_preset_button.setIcon(self.shell_icon("save"))
        self.save_to_preset_button.setIconSize(QSize(18, 18))
        self.save_to_preset_button.setToolTip("Save changes to the selected preset")
        self.save_to_preset_button.setAccessibleName("Save selected preset")
        self.save_to_preset_button.setShortcut(QKeySequence.StandardKey.Save)
        self.save_to_preset_button.clicked.connect(self.save_to_selected_preset)

        self.save_preset_as_button = QPushButton("Save As")
        self.save_preset_as_button.setToolTip("Create a new preset from the current settings")
        self.save_preset_as_button.setAccessibleName("Save preset as")
        self.save_preset_as_button.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_preset_as_button.clicked.connect(self.save_preset_as)

        self.revert_preset_button = QPushButton("Revert")
        self.revert_preset_button.setToolTip("Discard unsaved changes and restore the selected preset")
        self.revert_preset_button.setAccessibleName("Revert selected preset changes")
        self.revert_preset_button.clicked.connect(self.revert_selected_preset)

        self.delete_preset_button = QPushButton("Delete preset")
        self.delete_preset_button.setProperty("cellonautIconName", "trash-2")
        self.delete_preset_button.setIcon(self.shell_icon("trash-2"))
        self.delete_preset_button.setToolTip("Delete the selected preset")
        self.delete_preset_button.setAccessibleName("Delete selected preset")
        self.delete_preset_button.clicked.connect(self.delete_selected_preset)
        self.set_button_role(self.delete_preset_button, "danger")

        preset_label = QLabel("Preset")
        preset_label.setObjectName("PresetLabel")
        preset_label.setProperty("uiRole", "formLabel")
        preset_row.addWidget(preset_label)
        self.preset_combo.setMinimumWidth(150)
        self.preset_combo.setMaximumWidth(220)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.preset_dirty_label)
        preset_row.addWidget(self.save_to_preset_button)
        preset_row.addWidget(self.save_preset_as_button)
        preset_row.addWidget(self.revert_preset_button)

        self.import_preset_button = QPushButton("Import Preset...")
        self.import_preset_button.setAccessibleName("Import Cellonaut preset")
        self.import_preset_button.setProperty("cellonautIconName", "folder-open")
        self.import_preset_button.setIcon(self.shell_icon("folder-open"))
        self.import_preset_button.setToolTip("Load shared settings, then select the files and folders on this computer.")
        self.import_preset_button.clicked.connect(self.import_preset)

        self.export_preset_button = QPushButton("Export Selected Preset...")
        self.export_preset_button.setAccessibleName("Export selected Cellonaut preset")
        self.export_preset_button.setProperty("cellonautIconName", "save")
        self.export_preset_button.setIcon(self.shell_icon("save"))
        self.export_preset_button.setToolTip("Export the current settings, including unsaved edits. File locations must be selected again after import.")
        self.export_preset_button.clicked.connect(self.export_selected_preset)

        self.preset_overflow_button = QToolButton()
        self.preset_overflow_button.setText("⋮")
        self.preset_overflow_button.setAccessibleName("More application actions")
        self.preset_overflow_button.setToolTip("More actions")
        self.preset_overflow_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.preset_overflow_menu = QMenu(self.preset_overflow_button)
        for button in (
            self.import_preset_button,
            self.export_preset_button,
            self.delete_preset_button,
        ):
            action = QWidgetAction(self.preset_overflow_menu)
            action.setDefaultWidget(button)
            self.preset_overflow_menu.addAction(action)
        self.preset_overflow_button.setMenu(self.preset_overflow_menu)
        preset_row.addWidget(self.preset_overflow_button)

        self.root_layout.addWidget(self.app_header_frame)
        self.build_run_progress_panel()
        self.update_pipeline_command_layout(2000)

        # Keep task feedback visible in one compact status bar below the workspace.
        self.bottom_status_frame = QFrame(self)
        self.bottom_status_frame.setObjectName("ApplicationStatusBar")
        progress_row = QHBoxLayout(self.bottom_status_frame)
        progress_row.setContentsMargins(SPACING.md, 2, SPACING.md, 2)
        progress_row.setSpacing(SPACING.sm)

        self.pipeline_spinner_label = QLabel("●")
        self.pipeline_spinner_label.setObjectName("PipelineSpinnerLabel")
        self.pipeline_spinner_label.setProperty("uiRole", "statusText")
        self.pipeline_spinner_label.setProperty("status", "idle")
        self.pipeline_spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pipeline_spinner_label.setFixedWidth(12)
        self.pipeline_spinner_label.setToolTip("Application is idle")

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("PipelineStatusLabel")
        self.status_label.setProperty("uiRole", "statusText")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setMinimumWidth(72)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("PipelineProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setFixedWidth(240)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()

        self.sample_counter_label = QLabel("0 / 0")
        self.sample_counter_label.setObjectName("SampleCounterLabel")
        self.sample_counter_label.setProperty("uiRole", "statusText")
        self.sample_counter_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sample_counter_label.setFixedWidth(70)

        self.current_sample_label = QLabel("No task running")
        self.current_sample_label.setObjectName("CurrentTaskLabel")
        self.current_sample_label.setProperty("uiRole", "taskText")
        self.current_sample_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_sample_label.setWordWrap(False)

        progress_row.addWidget(self.status_label)
        progress_row.addWidget(self.current_sample_label, 1)
        progress_row.addWidget(self.progress_bar)
        progress_row.addWidget(self.sample_counter_label)

        self.app_header_frame.setMinimumWidth(0)
        self.app_header_frame.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.bottom_status_frame.setFixedHeight(WORKSPACE.status_bar_height)
        self.bottom_status_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.run_stage_row.insertWidget(0, self.pipeline_spinner_label)
        self.refresh_preset_control_states()

    # Keep the header usable at the supported minimum window width by moving
    # preset controls beneath the brand and run commands when necessary.
    def update_pipeline_command_layout(self, available_width: int | None = None) -> None:
        layout = self.app_header_layout
        for widget in (self.brand_widget, self.pipeline_commands_widget, self.preset_commands_widget):
            layout.removeWidget(widget)

        compact = int(available_width or self.app_header_frame.width()) < 1180
        if compact:
            layout.addWidget(self.brand_widget, 0, 0)
            layout.addWidget(self.pipeline_commands_widget, 0, 1)
            layout.addWidget(self.preset_commands_widget, 1, 0, 1, 2)
            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 1)
        else:
            layout.addWidget(self.brand_widget, 0, 0)
            layout.addWidget(self.pipeline_commands_widget, 0, 1)
            layout.addWidget(self.preset_commands_widget, 0, 2)
            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 0)

    # Disable settings changes and new runs while a primary task is active.
    def set_primary_task_active(self, active: bool) -> None:
        self._primary_task_active = bool(active)
        for widget in (
            self.run_button,
            self.preview_pipeline_button,
            self.check_setup_button,
            self.reuse_existing_masks_checkbox,
            self.mask_source_report_button,
            self.preset_combo,
            self.save_preset_as_button,
            self.revert_preset_button,
            self.import_preset_button,
            self.export_preset_button,
        ):
            widget.setEnabled(not active)
        for name in (
            "nd2_import_button",
            "nd2_source_dir",
            "nd2_output_dir",
            "nd2_group",
            "nd2_convert_button",
            "mask_adjust_preview_button",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not active)
        if not active and hasattr(self, "update_nd2_channel_summary"):
            self.update_nd2_channel_summary()
        self.set_cancel_button_active(active)
        self.refresh_preset_control_states()

    # Save and Delete depend on both task state and preset state, so recompute
    # them together whenever a preset loads, changes, or finishes saving.
    def refresh_preset_control_states(self) -> None:
        busy = bool(getattr(self, "_primary_task_active", False))
        name = self.preset_combo.currentText().strip()
        dirty = bool(getattr(self, "_preset_dirty", False))
        default_selected = is_default_preset(name)
        self.preset_combo.setEnabled(not busy and self.preset_combo.count() > 0)
        self.save_to_preset_button.setEnabled(not busy and bool(name) and dirty)
        self.save_to_preset_button.setToolTip(
            "Default is locked; save these changes as a new preset"
            if default_selected
            else "Save changes to the selected preset"
        )
        self.save_to_preset_button.setAccessibleName(
            "Save Default changes as a new preset" if default_selected else "Save selected preset"
        )
        self.save_preset_as_button.setEnabled(not busy)
        self.revert_preset_button.setEnabled(not busy and bool(name) and dirty)
        self.delete_preset_button.setEnabled(not busy and bool(name) and not default_selected)
        self.import_preset_button.setEnabled(not busy)
        self.export_preset_button.setEnabled(not busy and bool(name))

    # Hide cancellation completely while idle so the command row only presents
    # actions that are currently useful.
    def set_cancel_button_active(self, active: bool) -> None:
        self.cancel_button.setVisible(active)
        self.cancel_button.setEnabled(active)
