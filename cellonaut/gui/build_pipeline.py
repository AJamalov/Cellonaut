"""Construction of the main pipeline configuration panel."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from cellonaut.config.defaults import QC_FILTER_MODE_LABELS, MASK_INTENSITY_SOURCE_LABELS
from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.panel_notes import PANEL_NOTE_KEYS, PanelNotesWidget, limit_note_lines, normalize_panel_notes
from cellonaut.gui.ui_tokens import SPACING
from cellonaut.gui.widgets import NoWheelTabBar, PathRow, ProcessingStepsTable


class CellonautGuiPipelineBuildMixin(GuiMixin):
    """Build the channel, processing, segmentation, and measurement panels."""

    SCIENTIFIC_TABLE_ROW_HEIGHT = 32

    def configure_scientific_table(self, table: QTableWidget) -> None:
        """Apply shared table spacing and appearance."""
        table.setProperty("scientificTable", "true")
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setCornerButtonEnabled(False)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        vertical_header = table.verticalHeader()
        vertical_header.setMinimumSectionSize(28)
        vertical_header.setDefaultSectionSize(self.SCIENTIFIC_TABLE_ROW_HEIGHT)
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        table.horizontalHeader().setFixedHeight(30)

    def build_pipeline_content(self):
        # Build setup controls first because later panels use them.
        self._build_pipeline_setup_controls()
        analysis_inner, analysis_layout = self._build_analysis_matrix_panel()
        self._build_measurement_settings_panel()
        self._build_filter_settings_panel()
        self._build_mask_adjustment_panel()
        self._assemble_pipeline_sections(analysis_inner, analysis_layout)

    def _build_pipeline_setup_controls(self) -> None:
        self.io_group = self.make_section("Setup")

        self.input_dir = PathRow(
            "Input directory",
            mode="dir",
            tooltip=(
                "Folder containing TIFF files, multi-channel TIFF stacks, or supported channel/sample subfolders. "
                "Cellonaut detects the layout and channel names after selection."
            ),
        )

        self.output_dir = PathRow(
            "Output directory",
            mode="dir",
            tooltip="All measurements, review files, logs, and overlays will be saved here.",
        )

        self.mask_source_dir = PathRow(
            "Mask reuse folder",
            mode="dir",
            tooltip=(
                "Previous Cellonaut output folder used when Reuse existing masks is enabled. "
                "Leave blank to reuse masks from the current output folder."
            ),
        )

        self.io_group.add_rows(
            [
                self.input_dir,
                self.output_dir,
                self.mask_source_dir,
            ]
        )

        self.nd2_import_button = QPushButton("Import ND2 files...")
        self.nd2_import_button.setToolTip(
            "Open the dedicated ND2 workflow to inspect channels and convert ND2 files to TIFF."
        )
        self.nd2_import_button.clicked.connect(self.open_nd2_import_dialog)
        self.io_group.add_widget(self.nd2_import_button)

        mask_reuse_row = QHBoxLayout()
        mask_reuse_row.setContentsMargins(0, 0, 0, 0)
        mask_reuse_row.setSpacing(8)
        mask_reuse_row.addWidget(self.reuse_existing_masks_checkbox)
        mask_reuse_row.addStretch(1)
        mask_reuse_row.addWidget(self.mask_source_report_button)
        self.io_group.add_layout(mask_reuse_row)

        self.images_group = self.make_section("Channels")

        channel_heading = QLabel("Channels")
        channel_heading.setProperty("uiRole", "sectionTitle")

        self.image_tabs = QTabWidget()
        self.image_tabs.setDocumentMode(True)
        self.image_tabs.setMovable(False)
        self.image_tabs.setTabBar(NoWheelTabBar())
        self.image_tabs.currentChanged.connect(self.on_image_tab_changed)

        self.mask_tabs = QTabWidget()
        self.mask_tabs.setDocumentMode(True)
        self.mask_tabs.setMovable(False)
        self.mask_tabs.setTabBar(NoWheelTabBar())
        self.mask_tabs.setMinimumHeight(210)
        self.mask_tabs.currentChanged.connect(self.on_mask_tab_changed)
        self.mask_tabs.tabBarClicked.connect(self.on_mask_tab_clicked)

        image_note = QLabel("Detected image channels appear here after you select an input folder.\n" \
        "Use + to add one manually.")
        image_note.setWordWrap(True)
        image_note.setProperty("muted", "true")
        image_note.setProperty("uiRole", "mutedLabel")

        self.images_group.add_widget(image_note)
        self.images_group.add_widget(channel_heading)
        self.images_group.add_widget(self.image_tabs)
        self.io_group.add_widget(self.images_group)

        self.masks_group = self.make_section("Masks")
        masks_note = QLabel(
            "Select an image, then use 'Add mask' to create a 2D Weka mask from its chosen Z slice or projection.\n"
            "Tabs are red with no mask, amber while setup is incomplete, and green when the classifier file is found.\n"
            "Disconnected regions in one mask are measured together; use Cellpose for separate cell results."
        )
        masks_note.setWordWrap(True)
        masks_note.setProperty("muted", "true")
        masks_note.setProperty("uiRole", "mutedLabel")
        self.mask_source_tabs = QTabWidget()
        self.mask_source_tabs.setDocumentMode(True)
        self.mask_source_tabs.setMovable(False)
        self.mask_source_tabs.setTabBar(NoWheelTabBar())
        self.mask_source_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mask_source_tabs.setFixedHeight(self.mask_source_tabs.fontMetrics().height() + 18)
        self.mask_source_tabs.currentChanged.connect(self.on_mask_source_tab_changed)
        self.masks_group.add_widget(masks_note)
        self.masks_group.add_widget(self.mask_source_tabs)
        self.masks_group.add_widget(self.mask_tabs)

        self.processing_group = self.make_section("Processing")
        self.image_processing_group = self.make_section("Before mask creation")

        image_processing_note = QLabel(
            "Add processing steps to the source images before creating the masks.\n" \
            "Train Weka classifiers on images processed in the same way.\n" \
            "Steps run from top to bottom; blank values or OFF skip an image channel.\n" \
        )
        image_processing_note.setWordWrap(True)
        image_processing_note.setProperty("muted", "true")
        image_processing_note.setProperty("uiRole", "mutedLabel")

        self.image_processing_table = ProcessingStepsTable()
        self.image_processing_table.setColumnCount(0)
        self.image_processing_table.setRowCount(0)
        self.image_processing_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.image_processing_table.horizontalHeader().setVisible(False)
        self.image_processing_table.verticalHeader().setVisible(False)
        self.configure_scientific_table(self.image_processing_table)
        self.image_processing_table.step_column_drop_callback = self.reorder_image_processing_recipe

        self.image_processing_group.add_widget(image_processing_note)
        self.image_processing_group.add_widget(self.image_processing_table)
        self.mask_processing_group = self.make_section("After mask creation")

        mask_processing_note = QLabel(
            "Add processing steps to clean up or adjust the masks after creation, including combined masks."
        )
        mask_processing_note.setWordWrap(True)
        mask_processing_note.setProperty("muted", "true")
        mask_processing_note.setProperty("uiRole", "mutedLabel")

        self.mask_processing_table = ProcessingStepsTable()
        self.mask_processing_table.setColumnCount(0)
        self.mask_processing_table.setRowCount(0)
        self.mask_processing_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.configure_scientific_table(self.mask_processing_table)
        self.mask_processing_table.step_column_drop_callback = self.reorder_mask_processing_recipe

        self.mask_processing_group.add_widget(mask_processing_note)
        self.mask_processing_group.add_widget(self.mask_processing_table)
        self.measurement_processing_group = self.make_section("Before measurement")
        measurement_processing_note = QLabel(
            "Add processing steps to the source images used for measurement.\n" \
            "Subtract Background is kept last: each radius creates a separate corrected result from the image after the other steps."
        )
        measurement_processing_note.setWordWrap(True)
        measurement_processing_note.setProperty("muted", "true")
        measurement_processing_note.setProperty("uiRole", "mutedLabel")
        self.measurement_processing_table = ProcessingStepsTable()
        self.measurement_processing_table.setColumnCount(0)
        self.measurement_processing_table.setRowCount(0)
        self.measurement_processing_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.measurement_processing_table.horizontalHeader().setVisible(False)
        self.measurement_processing_table.verticalHeader().setVisible(False)
        self.configure_scientific_table(self.measurement_processing_table)
        self.measurement_processing_table.step_column_drop_callback = self.reorder_measurement_processing_recipe
        self.measurement_processing_group.add_widget(measurement_processing_note)
        self.measurement_processing_group.add_widget(self.measurement_processing_table)

        self.processing_group.add_widget(self.image_processing_group)
        self.processing_group.add_widget(self.mask_processing_group)
        self.processing_group.add_widget(self.measurement_processing_group)
        self.cellpose_settings_group = self.make_section("Cellpose")

        cellpose_note = QLabel(
            "Each enabled row creates one reusable Channel_Cellpose mask column in Measurements.\n"
            "Choose the image channel Cellpose should use to find cells; any measured channel can reuse that mask."
        )
        cellpose_note.setWordWrap(True)
        cellpose_note.setProperty("muted", "true")
        cellpose_note.setProperty("uiRole", "mutedLabel")

        self.cellpose_settings_table = QTableWidget()
        self.cellpose_settings_table.setColumnCount(0)
        self.cellpose_settings_table.setRowCount(0)
        self.cellpose_settings_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.cellpose_settings_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cellpose_settings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.cellpose_settings_table.horizontalHeader().setSectionsMovable(False)
        self.configure_scientific_table(self.cellpose_settings_table)

        self.cellpose_settings_group.add_widget(cellpose_note)
        self.cellpose_settings_group.add_widget(self.cellpose_settings_table)
        self.masks_group.add_widget(self.cellpose_settings_group)

        self.rebuild_image_rows()

    def _build_analysis_matrix_panel(self) -> tuple[QWidget, QVBoxLayout]:

        self.analysis_mask_relationships_label = QLabel("Measured channels and regions")
        self.analysis_mask_relationships_label.setProperty("uiRole", "sectionTitle")

        self.analysis_matrix_table = QTableWidget()
        self.analysis_matrix_table.setColumnCount(0)
        self.analysis_matrix_table.setRowCount(0)
        self.analysis_matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.configure_scientific_table(self.analysis_matrix_table)
        self.analysis_matrix_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.analysis_matrix_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.analysis_matrix_warning_label = QLabel("Measurement setup: OK. No warnings found.")
        self.analysis_matrix_warning_label.setWordWrap(True)
        self.analysis_matrix_warning_label.setProperty("messageState", "ok")
        self.analysis_matrix_warning_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        analysis_matrix_inner = QWidget()
        analysis_matrix_inner_layout = QVBoxLayout(analysis_matrix_inner)
        analysis_matrix_inner_layout.setContentsMargins(0, 0, 0, 0)
        analysis_matrix_inner_layout.setSpacing(6)

        analysis_matrix_inner_layout.addWidget(self.analysis_mask_relationships_label)
        self.analysis_matrix_hint_label = QLabel(
            "Measured channels are in rows.\n"
            "Configured masks and reusable Channel_Cellpose masks are in columns.\n"
            "Turn ON any masks you want to measure. Cellpose columns can be shared across rows or combined on one row."
        )
        self.analysis_matrix_hint_label.setWordWrap(True)
        self.analysis_matrix_hint_label.setProperty("muted", "true")
        self.analysis_matrix_hint_label.setProperty("uiRole", "mutedLabel")
        analysis_matrix_inner_layout.addWidget(self.analysis_matrix_hint_label)
        analysis_matrix_inner_layout.addWidget(self.analysis_matrix_table)

        analysis_actions_row = QHBoxLayout()
        analysis_actions_row.setContentsMargins(0, 0, 0, 0)
        analysis_actions_row.setSpacing(8)
        analysis_actions_row.addWidget(self.analysis_matrix_warning_label, 1)
        analysis_actions_row.addStretch(0)
        analysis_matrix_inner_layout.addLayout(analysis_actions_row)
        return analysis_matrix_inner, analysis_matrix_inner_layout

    def _build_measurement_settings_panel(self) -> None:
        self.analysis_measurements_panel = QFrame()
        self.analysis_measurements_panel.setProperty("card", "true")
        measurements_panel_layout = QVBoxLayout(self.analysis_measurements_panel)
        measurements_panel_layout.setContentsMargins(8, 6, 8, 6)
        measurements_panel_layout.setSpacing(6)

        self.analysis_measurements_title = QLabel("Measurement settings")
        self.analysis_measurements_title.setProperty("muted", "true")
        self.analysis_measurements_title.setProperty("uiRole", "mutedLabel")
        measurements_panel_layout.addWidget(self.analysis_measurements_title)

        self.analysis_measurements_hint = QLabel("")
        self.analysis_measurements_hint.setWordWrap(True)
        self.analysis_measurements_hint.setProperty("muted", "true")
        self.analysis_measurements_hint.setProperty("uiRole", "mutedLabel")
        measurements_panel_layout.addWidget(self.analysis_measurements_hint)

        self.analysis_basic_measurement_checks_host = QGroupBox("Configured-mask measurements")
        self.analysis_basic_measurement_checks_layout = QGridLayout(self.analysis_basic_measurement_checks_host)
        self.analysis_basic_measurement_checks_layout.setContentsMargins(10, 8, 10, 10)
        self.analysis_basic_measurement_checks_layout.setHorizontalSpacing(14)
        self.analysis_basic_measurement_checks_layout.setVerticalSpacing(4)
        measurements_panel_layout.addWidget(self.analysis_basic_measurement_checks_host)

        self.analysis_whole_cell_measurement_checks_host = QGroupBox("Cellpose whole-cell measurements")
        self.analysis_whole_cell_measurement_checks_layout = QGridLayout(
            self.analysis_whole_cell_measurement_checks_host
        )
        self.analysis_whole_cell_measurement_checks_layout.setContentsMargins(10, 8, 10, 10)
        self.analysis_whole_cell_measurement_checks_layout.setHorizontalSpacing(14)
        self.analysis_whole_cell_measurement_checks_layout.setVerticalSpacing(4)
        measurements_panel_layout.addWidget(self.analysis_whole_cell_measurement_checks_host)

        self.analysis_cell_measurement_checks_host = QGroupBox("Configured mask within Cellpose cells")
        self.analysis_cell_measurement_checks_layout = QGridLayout(self.analysis_cell_measurement_checks_host)
        self.analysis_cell_measurement_checks_layout.setContentsMargins(10, 8, 10, 10)
        self.analysis_cell_measurement_checks_layout.setHorizontalSpacing(14)
        self.analysis_cell_measurement_checks_layout.setVerticalSpacing(4)
        measurements_panel_layout.addWidget(self.analysis_cell_measurement_checks_host)

    def _build_filter_settings_panel(self) -> None:
        self.analysis_filters_panel = QFrame()
        self.analysis_filters_panel.setProperty("previewToolPage", "true")
        self.analysis_filters_panel.setVisible(False)
        filters_panel_layout = QVBoxLayout(self.analysis_filters_panel)
        filters_panel_layout.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        filters_panel_layout.setSpacing(SPACING.md)

        self.analysis_filters_title = QLabel("Cell group settings")
        self.analysis_filters_title.setProperty("uiRole", "sectionTitle")
        filters_panel_layout.addWidget(self.analysis_filters_title)

        self.analysis_filters_hint = QLabel("")
        self.analysis_filters_hint.setWordWrap(True)
        self.analysis_filters_hint.setProperty("muted", "true")
        self.analysis_filters_hint.setProperty("uiRole", "mutedLabel")
        filters_panel_layout.addWidget(self.analysis_filters_hint)

        self.analysis_filter_source_label = QLabel("")
        self.analysis_filter_source_label.setWordWrap(True)
        self.analysis_filter_source_label.setProperty("muted", "true")
        self.analysis_filter_source_label.setProperty("uiRole", "previewToolContext")
        filters_panel_layout.addWidget(self.analysis_filter_source_label)

        self.analysis_population_controls = QWidget()
        population_controls = QVBoxLayout(self.analysis_population_controls)
        population_controls.setContentsMargins(0, 0, 0, 0)
        population_controls.setSpacing(SPACING.sm)
        population_select_row = QHBoxLayout()
        population_select_row.setContentsMargins(0, 0, 0, 0)
        population_select_row.setSpacing(SPACING.sm)
        population_select_row.addWidget(QLabel("Group"))
        self.analysis_population_combo = QComboBox()
        self.analysis_population_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.analysis_population_combo.currentIndexChanged.connect(self.on_analysis_population_changed)
        population_select_row.addWidget(self.analysis_population_combo, 1)
        population_controls.addLayout(population_select_row)
        population_edit_row = QHBoxLayout()
        population_edit_row.setContentsMargins(0, 0, 0, 0)
        population_edit_row.setSpacing(SPACING.sm)
        population_edit_row.addWidget(QLabel("Name"))
        self.analysis_population_name_edit = QLineEdit()
        self.analysis_population_name_edit.setPlaceholderText("Group name")
        self.analysis_population_name_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.analysis_population_name_edit.editingFinished.connect(self.on_analysis_population_name_changed)
        population_edit_row.addWidget(self.analysis_population_name_edit, 1)
        population_controls.addLayout(population_edit_row)
        population_action_row = QHBoxLayout()
        population_action_row.setContentsMargins(0, 0, 0, 0)
        population_action_row.setSpacing(SPACING.sm)
        population_action_row.addStretch(1)
        self.analysis_add_population_button = QPushButton("Add group")
        self.analysis_add_population_button.clicked.connect(self.add_analysis_population)
        population_action_row.addWidget(self.analysis_add_population_button)
        self.analysis_remove_population_button = QPushButton("Remove")
        self.analysis_remove_population_button.clicked.connect(self.remove_analysis_population)
        population_action_row.addWidget(self.analysis_remove_population_button)
        population_controls.addLayout(population_action_row)
        filters_panel_layout.addWidget(self.analysis_population_controls)
        self.analysis_active_filter_count_label = QLabel("Active conditions: 0 · Cell: 0 · Target mask: 0")
        self.analysis_active_filter_count_label.setProperty("uiRole", "mutedLabel")
        filters_panel_layout.addWidget(self.analysis_active_filter_count_label)

        self.analysis_filter_controls_group = QGroupBox("Preview behavior")
        self.analysis_filter_controls_group.setProperty("previewToolSection", "true")
        filter_controls_layout = QVBoxLayout(self.analysis_filter_controls_group)
        filter_controls_layout.setContentsMargins(SPACING.sm, SPACING.md, SPACING.sm, SPACING.sm)
        filter_controls_layout.setSpacing(SPACING.sm)

        filter_options_grid = QGridLayout()
        filter_options_grid.setContentsMargins(0, 0, 0, 0)
        filter_options_grid.setHorizontalSpacing(12)
        filter_options_grid.setVerticalSpacing(8)

        self.analysis_exclude_filtered_checkbox = QCheckBox("Exclude this group from CSV files")
        self.analysis_exclude_filtered_checkbox.setToolTip(
            "Omit matching cells from filtered CSV exports. A group with no conditions excludes no cells. Original measurements are unchanged."
        )
        self.analysis_exclude_filtered_checkbox.toggled.connect(self.on_inline_analysis_settings_changed)

        self.preview_filter_visible_checkbox = QCheckBox("Show cell groups")
        self.preview_filter_visible_checkbox.setChecked(True)
        self.preview_filter_visible_checkbox.setToolTip(
            "Show or hide cell-group overlays computed from saved measurements and Cellpose labels."
        )
        self.preview_filter_visible_checkbox.toggled.connect(self.on_preview_filter_visibility_changed)

        self.analysis_filter_mode_combo = QComboBox()
        self.analysis_filter_mode_combo.setMinimumWidth(170)
        self.analysis_filter_mode_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        for stored, display in QC_FILTER_MODE_LABELS.items():
            self.analysis_filter_mode_combo.addItem(display, stored)
        self.analysis_filter_mode_combo.setToolTip(
            "Choose whether to match all conditions, either category, only cell conditions, or only target-mask conditions. "
            "For Match either category, categories with no conditions are ignored."
        )
        self.analysis_filter_mode_combo.currentTextChanged.connect(self.on_inline_analysis_settings_changed)

        filter_options_grid.addWidget(self.analysis_exclude_filtered_checkbox, 0, 0, 1, 2)
        filter_options_grid.addWidget(self.preview_filter_visible_checkbox, 1, 0, 1, 2)
        filter_options_grid.addWidget(QLabel("Match rule"), 2, 0, 1, 2)
        filter_options_grid.addWidget(self.analysis_filter_mode_combo, 3, 0, 1, 2)
        filter_options_grid.addWidget(QLabel("Target mask for cell groups"), 4, 0, 1, 2)
        self.analysis_filter_target_mask_combo = QComboBox()
        self.analysis_filter_target_mask_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.analysis_filter_target_mask_combo.setToolTip(
            "Choose which enabled measurement mask supplies target-mask conditions for this channel."
        )
        self.analysis_filter_target_mask_combo.currentIndexChanged.connect(self.on_inline_analysis_settings_changed)
        filter_options_grid.addWidget(self.analysis_filter_target_mask_combo, 5, 0, 1, 2)

        filter_options_grid.addWidget(QLabel("Measure target intensity from"), 6, 0, 1, 2)
        self.analysis_mask_intensity_source_combo = QComboBox()
        self.analysis_mask_intensity_source_combo.setMinimumWidth(150)
        self.analysis_mask_intensity_source_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.analysis_mask_intensity_source_combo.addItems(
            list(MASK_INTENSITY_SOURCE_LABELS.values())
        )
        self.analysis_mask_intensity_source_combo.setToolTip(
            "Choose the image used for target-mask intensity conditions: the measured channel or the Cellpose source channel."
        )
        self.analysis_mask_intensity_source_combo.currentTextChanged.connect(self.on_inline_analysis_settings_changed)
        filter_options_grid.addWidget(self.analysis_mask_intensity_source_combo, 7, 0, 1, 2)
        filter_options_grid.setColumnStretch(1, 1)
        filter_controls_layout.addLayout(filter_options_grid)

        self.preview_filter_refresh_button = QPushButton("Update groups")
        self.preview_filter_refresh_button.setMinimumWidth(160)
        self.preview_filter_refresh_button.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.Fixed,
        )
        self.preview_filter_refresh_button.setProperty("cellonautIconName", "refresh-cw")
        self.preview_filter_refresh_button.setProperty("role", "primary")
        self.preview_filter_refresh_button.setIcon(self.shell_icon("refresh-cw"))
        self.preview_filter_refresh_button.setToolTip(
            "Recalculate cell-group membership without rerunning the pipeline."
        )
        self.preview_filter_refresh_button.clicked.connect(self.refresh_preview_filter_overlay)
        filters_panel_layout.addWidget(self.analysis_filter_controls_group)

        self.analysis_filter_export_group = QGroupBox("Export")
        self.analysis_filter_export_group.setProperty("previewToolFooter", "true")
        export_filter_layout = QVBoxLayout(self.analysis_filter_export_group)
        export_filter_layout.setContentsMargins(SPACING.sm, SPACING.md, SPACING.sm, SPACING.sm)
        export_filter_layout.setSpacing(SPACING.sm)

        export_filter_row = QHBoxLayout()
        export_filter_row.setContentsMargins(0, 0, 0, 0)
        export_filter_row.setSpacing(8)

        self.analysis_export_filtered_csv_button = QPushButton("Export filtered CSV for this sample")
        self.analysis_export_filtered_csv_button.setMinimumWidth(200)
        self.analysis_export_filtered_csv_button.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.Fixed,
        )
        self.analysis_export_filtered_csv_button.setEnabled(False)
        self.analysis_export_filtered_csv_button.setToolTip(
            "Open a generated overlay with linked cell measurements first."
        )
        self.analysis_export_filtered_csv_button.clicked.connect(self.export_current_filter_preview_csv)
        export_filter_row.addWidget(self.analysis_export_filtered_csv_button)

        self.analysis_export_all_filtered_csv_button = QPushButton(
            "Export filtered CSV for all samples in the pipeline"
        )
        self.analysis_export_all_filtered_csv_button.setMinimumWidth(230)
        self.analysis_export_all_filtered_csv_button.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.Fixed,
        )
        self.analysis_export_all_filtered_csv_button.setToolTip(
            "Apply the current cell groups to compatible samples in the open image's Results folder, "
            "or the configured output folder when no Results folder is linked."
        )
        self.analysis_export_all_filtered_csv_button.clicked.connect(self.export_all_filter_preview_csvs)
        export_filter_row.addWidget(self.analysis_export_all_filtered_csv_button)
        export_filter_row.addStretch(1)
        export_filter_layout.addLayout(export_filter_row)

        preview_filter_row = QHBoxLayout()
        preview_filter_row.setContentsMargins(0, 0, 0, 0)
        preview_filter_row.addStretch(1)
        preview_filter_row.addWidget(self.preview_filter_refresh_button)
        export_filter_layout.addLayout(preview_filter_row)
        self.analysis_filter_empty_label = QLabel("")
        self.analysis_filter_empty_label.setWordWrap(True)
        self.analysis_filter_empty_label.setProperty("muted", "true")
        self.analysis_filter_empty_label.setProperty("uiRole", "previewToolEmptyState")
        self.analysis_filter_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_filter_empty_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.analysis_filter_empty_label.setMinimumHeight(72)
        self.analysis_filter_empty_label.setVisible(False)
        filters_panel_layout.addWidget(self.analysis_filter_empty_label)

        filter_tables_row = QVBoxLayout()
        filter_tables_row.setContentsMargins(0, 0, 0, 0)
        filter_tables_row.setSpacing(12)

        self.analysis_cell_filter_host = QGroupBox("Cellpose cell measurements")
        self.analysis_cell_filter_host.setProperty("previewToolSection", "true")
        cell_filter_host_layout = QVBoxLayout(self.analysis_cell_filter_host)
        cell_filter_host_layout.setContentsMargins(10, 8, 10, 10)
        cell_filter_host_layout.setSpacing(SPACING.sm)
        cell_shape_group = QGroupBox("Cell shape and size")
        self.analysis_cell_shape_filter_layout = QGridLayout(cell_shape_group)
        cell_filter_host_layout.addWidget(cell_shape_group)
        cell_intensity_group = QGroupBox("Cellpose whole-cell intensity")
        cell_intensity_layout = QVBoxLayout(cell_intensity_group)
        self.analysis_cell_intensity_filter_layout = QGridLayout()
        cell_intensity_layout.addLayout(self.analysis_cell_intensity_filter_layout)
        self.analysis_cell_advanced_filter_group = QGroupBox("Advanced intensity filters")
        self.analysis_cell_advanced_filter_group.setCheckable(True)
        self.analysis_cell_advanced_filter_group.setChecked(False)
        self.analysis_cell_advanced_filter_group.toggled.connect(
            lambda expanded: self.set_advanced_filter_group_expanded(self.analysis_cell_advanced_filter_group, expanded)
        )
        self.analysis_cell_advanced_filter_layout = QGridLayout(self.analysis_cell_advanced_filter_group)
        cell_intensity_layout.addWidget(self.analysis_cell_advanced_filter_group)
        cell_filter_host_layout.addWidget(cell_intensity_group)

        self.analysis_mask_filter_host = QGroupBox("Target mask within cell")
        self.analysis_mask_filter_host.setProperty("previewToolSection", "true")
        mask_filter_host_layout = QVBoxLayout(self.analysis_mask_filter_host)
        mask_filter_host_layout.setContentsMargins(10, 8, 10, 10)
        mask_filter_host_layout.setSpacing(SPACING.sm)
        self.analysis_mask_filter_layout = QGridLayout()
        mask_filter_host_layout.addLayout(self.analysis_mask_filter_layout)
        self.analysis_mask_advanced_filter_group = QGroupBox("Advanced target-mask filters")
        self.analysis_mask_advanced_filter_group.setCheckable(True)
        self.analysis_mask_advanced_filter_group.setChecked(False)
        self.analysis_mask_advanced_filter_group.toggled.connect(
            lambda expanded: self.set_advanced_filter_group_expanded(self.analysis_mask_advanced_filter_group, expanded)
        )
        self.analysis_mask_advanced_filter_layout = QGridLayout(self.analysis_mask_advanced_filter_group)
        mask_filter_host_layout.addWidget(self.analysis_mask_advanced_filter_group)

        filter_tables_row.addWidget(self.analysis_cell_filter_host)
        filter_tables_row.addWidget(self.analysis_mask_filter_host)
        filters_panel_layout.addLayout(filter_tables_row)
        filters_panel_layout.addWidget(self.analysis_filter_export_group)
        filters_panel_layout.addStretch(1)

    def _build_mask_adjustment_panel(self) -> None:
        self.analysis_mask_adjust_panel = QFrame()
        self.analysis_mask_adjust_panel.setProperty("previewToolPage", "true")
        self.analysis_mask_adjust_panel.setVisible(False)
        mask_adjust_layout = QVBoxLayout(self.analysis_mask_adjust_panel)
        mask_adjust_layout.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        mask_adjust_layout.setSpacing(SPACING.md)

        self.analysis_mask_adjust_title = QLabel("Mask adjustments")
        self.analysis_mask_adjust_title.setProperty("uiRole", "sectionTitle")
        mask_adjust_layout.addWidget(self.analysis_mask_adjust_title)

        self.analysis_mask_adjust_hint = QLabel(
            "Adjust one mask target for the open preview without rerunning segmentation. "
            "Use Preview adjustment to compare a temporary layer; these controls do not change pipeline settings."
        )
        self.analysis_mask_adjust_hint.setWordWrap(True)
        self.analysis_mask_adjust_hint.setProperty("muted", "true")
        self.analysis_mask_adjust_hint.setProperty("uiRole", "mutedLabel")
        mask_adjust_layout.addWidget(self.analysis_mask_adjust_hint)

        self.mask_adjust_controls_group = QGroupBox("Adjustment controls")
        self.mask_adjust_controls_group.setProperty("previewToolSection", "true")
        mask_adjust_controls_layout = QVBoxLayout(self.mask_adjust_controls_group)
        mask_adjust_controls_layout.setContentsMargins(SPACING.sm, SPACING.md, SPACING.sm, SPACING.sm)
        mask_adjust_controls_layout.setSpacing(SPACING.sm)

        mask_adjust_grid = QGridLayout()
        mask_adjust_grid.setContentsMargins(0, 0, 0, 0)
        mask_adjust_grid.setHorizontalSpacing(SPACING.md)
        mask_adjust_grid.setVerticalSpacing(SPACING.sm)

        self.mask_adjust_target_combo = QComboBox()
        self.mask_adjust_target_combo.currentIndexChanged.connect(self.on_mask_adjust_target_changed)

        self.mask_adjust_x_spin = QSpinBox()
        self.mask_adjust_x_spin.setRange(-500, 500)
        self.mask_adjust_x_spin.setSuffix(" px")
        self.mask_adjust_x_spin.valueChanged.connect(self.on_mask_adjustment_changed)

        self.mask_adjust_y_spin = QSpinBox()
        self.mask_adjust_y_spin.setRange(-500, 500)
        self.mask_adjust_y_spin.setSuffix(" px")
        self.mask_adjust_y_spin.valueChanged.connect(self.on_mask_adjustment_changed)

        self.mask_adjust_grow_spin = QSpinBox()
        self.mask_adjust_grow_spin.setRange(-50, 50)
        self.mask_adjust_grow_spin.setSuffix(" px")
        self.mask_adjust_grow_spin.setToolTip("Positive = expand. Negative = shrink.")
        self.mask_adjust_grow_spin.valueChanged.connect(self.on_mask_adjustment_changed)

        self.mask_adjust_min_size_spin = QSpinBox()
        self.mask_adjust_min_size_spin.setRange(0, 10000000)
        self.mask_adjust_min_size_spin.setSingleStep(10)
        self.mask_adjust_min_size_spin.setToolTip(
            "After hole filling and growth or shrinkage, remove connected mask regions smaller than this area in pixels squared. 0 disables it."
        )
        self.mask_adjust_min_size_spin.valueChanged.connect(self.on_mask_adjustment_changed)

        self.mask_adjust_fill_holes_spin = QSpinBox()
        self.mask_adjust_fill_holes_spin.setRange(0, 1000)
        self.mask_adjust_fill_holes_spin.setSuffix(" px²")
        self.mask_adjust_fill_holes_spin.setToolTip("Fill enclosed holes up to this area. 0 disables it.")
        self.mask_adjust_fill_holes_spin.valueChanged.connect(self.on_mask_adjustment_changed)

        mask_target_label = QLabel("Mask target")
        mask_target_label.setProperty("uiRole", "formLabel")
        mask_adjust_grid.addWidget(mask_target_label, 0, 0)
        mask_adjust_grid.addWidget(self.mask_adjust_target_combo, 0, 1, 1, 5)

        for column, title in ((0, "Position"), (2, "Shape"), (4, "Cleanup")):
            heading = QLabel(title)
            heading.setProperty("uiRole", "previewToolSubheading")
            mask_adjust_grid.addWidget(heading, 1, column, 1, 2)

        mask_adjust_grid.addWidget(QLabel("X shift"), 2, 0)
        mask_adjust_grid.addWidget(self.mask_adjust_x_spin, 2, 1)
        mask_adjust_grid.addWidget(QLabel("Expand / shrink (px radius)"), 2, 2)
        mask_adjust_grid.addWidget(self.mask_adjust_grow_spin, 2, 3)
        mask_adjust_grid.addWidget(QLabel("Minimum area"), 2, 4)
        mask_adjust_grid.addWidget(self.mask_adjust_min_size_spin, 2, 5)
        mask_adjust_grid.addWidget(QLabel("Y shift"), 3, 0)
        mask_adjust_grid.addWidget(self.mask_adjust_y_spin, 3, 1)
        mask_adjust_grid.addWidget(QLabel("Fill holes"), 3, 4)
        mask_adjust_grid.addWidget(self.mask_adjust_fill_holes_spin, 3, 5)
        for column in (1, 3, 5):
            mask_adjust_grid.setColumnStretch(column, 1)
        mask_adjust_controls_layout.addLayout(mask_adjust_grid)
        mask_adjust_layout.addWidget(self.mask_adjust_controls_group)

        self.mask_adjust_export_group = QGroupBox("Actions")
        self.mask_adjust_export_group.setProperty("previewToolFooter", "true")
        mask_adjust_export_layout = QVBoxLayout(self.mask_adjust_export_group)
        mask_adjust_export_layout.setContentsMargins(SPACING.sm, SPACING.md, SPACING.sm, SPACING.sm)
        mask_adjust_export_layout.setSpacing(SPACING.sm)

        mask_adjust_buttons = QHBoxLayout()
        mask_adjust_buttons.setContentsMargins(0, 0, 0, 0)
        mask_adjust_buttons.setSpacing(8)

        self.mask_adjust_reset_button = QPushButton("Reset adjustments")
        self.mask_adjust_reset_button.setToolTip("Clear adjustments for the selected mask only; other masks retain their adjustments.")
        self.mask_adjust_reset_button.clicked.connect(self.reset_mask_adjustments)
        self.mask_adjust_preview_button = QPushButton("Preview adjustment")
        self.mask_adjust_preview_button.setProperty("role", "primary")
        self.mask_adjust_preview_button.setToolTip("Add or update a temporary adjusted layer for the selected Weka mask. Pipeline settings and original files are unchanged.")
        self.mask_adjust_preview_button.clicked.connect(self.preview_current_mask_adjustments)
        self.mask_adjust_status_label = QLabel("No preview mask linked")
        self.mask_adjust_status_label.setWordWrap(True)
        self.mask_adjust_status_label.setProperty("muted", "true")
        self.mask_adjust_status_label.setProperty("uiRole", "mutedLabel")

        mask_adjust_export_layout.addWidget(self.mask_adjust_status_label)
        mask_adjust_buttons.addWidget(self.mask_adjust_reset_button)
        mask_adjust_buttons.addStretch(1)
        mask_adjust_buttons.addWidget(self.mask_adjust_preview_button)
        mask_adjust_export_layout.addLayout(mask_adjust_buttons)
        mask_adjust_layout.addWidget(self.mask_adjust_export_group)
        mask_adjust_layout.addStretch(1)

    def _assemble_pipeline_sections(
        self,
        analysis_matrix_inner: QWidget,
        analysis_matrix_inner_layout: QVBoxLayout,
    ) -> None:
        analysis_matrix_inner_layout.addWidget(self.analysis_measurements_panel)
        self._populate_inline_measurement_settings()
        self.analysis_measurements_panel.setVisible(True)

        self.analysis_section = self._build_collapsible_section(
            "Measurements",
            analysis_matrix_inner,
            expanded=False,
            summary="Measurement regions and outputs",
            step_number=5,
        )

        self.project_section = self._build_collapsible_section(
            "Setup",
            self.io_group,
            expanded=False,
            summary="Choose folders and configure image channels",
            step_number=1,
        )

        self.images_section = self._build_collapsible_section(
            "Masks", self.masks_group, expanded=False, summary="Create and combine image masks", step_number=2
        )
        self.cellpose_settings_section = self._build_collapsible_section(
            "Cellpose",
            self.cellpose_settings_group,
            expanded=False,
            summary="Configure cell segmentation",
            step_number=3,
        )
        self.image_processing_section = self._build_collapsible_section(
            "Processing",
            self.processing_group,
            expanded=False,
            summary="Prepare images and refine masks",
            step_number=4,
        )
        self.preview_tools_content = QWidget()
        preview_tools_layout = QVBoxLayout(self.preview_tools_content)
        preview_tools_layout.setContentsMargins(0, 0, 0, 0)
        preview_tools_layout.setSpacing(SPACING.sm)

        preview_tools_context = QFrame()
        preview_tools_context.setObjectName("PreviewToolsContext")
        preview_tools_context_layout = QVBoxLayout(preview_tools_context)
        preview_tools_context_layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        preview_tools_context_layout.setSpacing(SPACING.sm)
        preview_tools_file_row = QHBoxLayout()
        preview_tools_file_row.setContentsMargins(0, 0, 0, 0)
        preview_tools_file_row.setSpacing(SPACING.sm)
        self.preview_tools_context_label = QLabel()
        self.preview_tools_context_label.setWordWrap(True)
        self.preview_tools_context_label.setProperty("uiRole", "mutedLabel")
        self.preview_tools_open_image_button = QPushButton("Open image…")
        self.preview_tools_open_image_button.setProperty("cellonautIconName", "folder-open")
        self.preview_tools_open_image_button.setIcon(self.shell_icon("folder-open"))
        self.preview_tools_open_image_button.clicked.connect(self.browse_for_preview_image)
        preview_tools_file_row.addWidget(self.preview_tools_context_label, 1)
        preview_tools_file_row.addWidget(self.preview_tools_open_image_button)
        preview_tools_context_layout.addLayout(preview_tools_file_row)

        preview_tools_source_row = QHBoxLayout()
        preview_tools_source_row.setContentsMargins(0, 0, 0, 0)
        preview_tools_source_row.setSpacing(SPACING.sm)
        preview_tools_source_label = QLabel("Measured channel")
        preview_tools_source_label.setProperty("uiRole", "formLabel")
        self.preview_tools_source_combo = QComboBox()
        self.preview_tools_source_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.preview_tools_source_combo.currentIndexChanged.connect(self.on_preview_tools_source_changed)
        preview_tools_source_row.addWidget(preview_tools_source_label)
        preview_tools_source_row.addWidget(self.preview_tools_source_combo, 1)
        preview_tools_context_layout.addLayout(preview_tools_source_row)
        preview_tools_layout.addWidget(preview_tools_context)

        self.preview_tools_tabs = QTabWidget()
        self.preview_tools_tabs.setObjectName("PreviewToolsTabs")
        self.preview_tools_tabs.setTabBar(NoWheelTabBar())
        self.preview_tools_tabs.addTab(self.analysis_mask_adjust_panel, "Mask Adjustments")
        self.preview_tools_tabs.addTab(self.analysis_filters_panel, "Cell Groups")
        self.preview_tools_tabs.currentChanged.connect(self.on_preview_tools_tab_changed)
        preview_tools_layout.addWidget(self.preview_tools_tabs)

        self.preview_tools_section = self._build_collapsible_section(
            "Image Preview Tools",
            self.preview_tools_content,
            expanded=False,
            summary="Review cell groups and temporary mask adjustments for the open overlay",
        )

        self.pipeline_sections = [
            self.project_section,
            self.images_section,
            self.cellpose_settings_section,
            self.image_processing_section,
            self.analysis_section,
            self.preview_tools_section,
        ]
        self.pipeline_section_descriptions = [
            "Choose the source and destination folders, import ND2 data, and configure image channels.",
            "Create masks for individual images and combine masks from several sources.",
            "Configure Cellpose segmentation used to create whole-cell masks.",
            "Prepare images before mask creation or measurement, and refine masks after creation.",
            "Choose measurement regions and metrics. Results are saved automatically.",
            "Define cell groups and preview temporary mask adjustments against the image currently open in Image Preview.",
        ]
        self.pipeline_section_help_topics = [
            "Input & ND2",
            "Channels & Masks",
            "Cellpose",
            "Processing",
            "Measurements",
            "Image Preview Tools",
        ]
        self.pipeline_panel_notes = normalize_panel_notes({})
        self._pipeline_note_expanded = {key: False for key in PANEL_NOTE_KEYS}

        self.pipeline_overview_header = QFrame()
        self.pipeline_overview_header.setObjectName("PipelineOverviewHeader")
        pipeline_overview_header_layout = QHBoxLayout(self.pipeline_overview_header)
        pipeline_overview_header_layout.setContentsMargins(SPACING.xs, SPACING.xs, SPACING.xs, SPACING.xs)
        pipeline_overview_header_layout.setSpacing(SPACING.sm)
        self.pipeline_overview_title = QLabel("Pipeline")
        self.pipeline_overview_title.setProperty("uiRole", "pipelineOverviewTitle")
        self.pipeline_overview_step_count = QLabel("5 steps")
        self.pipeline_overview_step_count.setProperty("uiRole", "mutedLabel")
        pipeline_overview_header_layout.addWidget(self.pipeline_overview_title)
        pipeline_overview_header_layout.addStretch(1)
        pipeline_overview_header_layout.addWidget(self.pipeline_overview_step_count)

        self.pipeline_editor_header = QFrame()
        self.pipeline_editor_header.setObjectName("PipelineEditorHeader")
        editor_header_layout = QVBoxLayout(self.pipeline_editor_header)
        editor_header_layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        editor_header_layout.setSpacing(SPACING.xs)
        self.pipeline_back_button = QPushButton("← Back to Pipeline")
        self.pipeline_back_button.setAccessibleName("Back to pipeline overview")
        self.pipeline_back_button.clicked.connect(self.show_pipeline_overview)
        self.pipeline_editor_title = QLabel()
        self.pipeline_editor_title.setProperty("uiRole", "pipelineEditorTitle")
        self.pipeline_editor_description = QLabel()
        self.pipeline_editor_description.setProperty("uiRole", "pipelineEditorDescription")
        self.pipeline_editor_description.setWordWrap(True)
        editor_header_layout.addWidget(self.pipeline_back_button, 0, Qt.AlignmentFlag.AlignLeft)
        editor_header_layout.addWidget(self.pipeline_editor_title)
        editor_description_row = QHBoxLayout()
        editor_description_row.addWidget(self.pipeline_editor_description, 1)
        self.pipeline_editor_help_link = self.make_help_link("Introduction")
        editor_description_row.addWidget(self.pipeline_editor_help_link, 0, Qt.AlignmentFlag.AlignBottom)
        editor_header_layout.addLayout(editor_description_row)
        self.pipeline_notes_widget = PanelNotesWidget()
        self.pipeline_notes_widget.noteChanged.connect(self.on_pipeline_panel_note_changed)
        self.pipeline_notes_widget.expandedChanged.connect(self.on_pipeline_panel_note_expanded_changed)
        editor_header_layout.addWidget(self.pipeline_notes_widget)
        self.pipeline_editor_header.hide()

        self.pipeline_editor_navigation = QWidget()
        editor_navigation_layout = QHBoxLayout(self.pipeline_editor_navigation)
        editor_navigation_layout.setContentsMargins(0, 4, 0, 0)
        self.pipeline_previous_section_button = QPushButton("← Previous")
        self.pipeline_previous_section_button.clicked.connect(lambda: self.show_adjacent_pipeline_section(-1))
        self.pipeline_next_section_button = QPushButton("Next →")
        self.pipeline_next_section_button.clicked.connect(lambda: self.show_adjacent_pipeline_section(1))
        editor_navigation_layout.addWidget(self.pipeline_previous_section_button)
        editor_navigation_layout.addStretch(1)
        editor_navigation_layout.addWidget(self.pipeline_next_section_button)
        self.pipeline_editor_navigation.hide()

        self.pipeline_tab_layout.addWidget(self.pipeline_overview_header)
        self.pipeline_tab_layout.addWidget(self.pipeline_editor_header)

        for index, section in enumerate(self.pipeline_sections):
            section.toggle_button.setChecked(False)
            section.toggle_button.setCheckable(False)
            section.chevron_label.setText("›")
            section.toggle_button.setToolTip(f"Open {section._title}")
            section.toggle_button.clicked.connect(
                lambda _checked=False, target=index: self.show_pipeline_section(target)
            )
            self.pipeline_tab_layout.addWidget(section)
        self.pipeline_tab_layout.addWidget(self.pipeline_editor_navigation)
        self.pipeline_tab_layout.addStretch(1)

        self._active_pipeline_section_index: int | None = None
        self.show_pipeline_overview()
        self.update_preview_tools_context()

        self.refresh_channel_name_dependent_ui()
        self.build_analysis_matrix_for_current_source()
        self.update_analysis_matrix_warning_label()
        self.refresh_pipeline_section_summaries()

    def show_pipeline_overview(self) -> None:
        """Show the six workflow cards without expanding their editor contents."""
        self._active_pipeline_section_index = None
        self.pipeline_overview_header.show()
        self.pipeline_editor_header.hide()
        self.pipeline_editor_navigation.hide()
        for section in self.pipeline_sections:
            section.show()
            section.set_overview_mode()
            section.content.setProperty("editorVisible", "false")
            section.content.style().unpolish(section.content)
            section.content.style().polish(section.content)
            section.content.hide()
        self.pipeline_scroll.verticalScrollBar().setValue(0)

    def show_pipeline_section(self, index: int) -> None:
        """Open one workflow editor while keeping its controls and state alive."""
        if not 0 <= index < len(self.pipeline_sections):
            return
        if self.pipeline_sections[index] is self.preview_tools_section and str(
            getattr(self, "_inline_analysis_settings_panel", "") or ""
        ) not in {"filters", "mask_adjust"}:
            active_defs = self.get_active_image_definitions()
            if active_defs:
                self.show_analysis_settings_panel(0, "mask_adjust")
                return
        self._active_pipeline_section_index = index
        selected = self.pipeline_sections[index]
        self.pipeline_overview_header.hide()
        for section_index, section in enumerate(self.pipeline_sections):
            section.setVisible(section_index == index)
            if section_index == index:
                section.set_editor_mode()
            section.content.setProperty("editorVisible", "true" if section_index == index else "false")
            section.content.style().unpolish(section.content)
            section.content.style().polish(section.content)
            section.content.setVisible(section_index == index)

        self.pipeline_editor_title.setText(
            selected._title if selected is self.preview_tools_section else f"{index + 1}. {selected._title}"
        )
        self.pipeline_editor_description.setText(self.pipeline_section_descriptions[index])
        self.set_help_link_topic(self.pipeline_editor_help_link, self.pipeline_section_help_topics[index])
        self.show_pipeline_panel_note(index)
        self.pipeline_previous_section_button.setEnabled(index > 0)
        self.pipeline_next_section_button.setEnabled(index < len(self.pipeline_sections) - 1)
        self.pipeline_previous_section_button.setText(
            f"← {self.pipeline_sections[index - 1]._title}" if index > 0 else "← Previous"
        )
        self.pipeline_next_section_button.setText(
            f"{self.pipeline_sections[index + 1]._title} →" if index < len(self.pipeline_sections) - 1 else "Next →"
        )
        self.pipeline_editor_header.show()
        self.pipeline_editor_navigation.show()
        self.pipeline_scroll.verticalScrollBar().setValue(0)

    def pipeline_panel_note_key(self, index: int) -> str:
        """Return the stable preset key for one opened pipeline editor."""
        return PANEL_NOTE_KEYS[index]

    def sync_pipeline_panel_note_from_editor(self) -> None:
        """Copy the visible note into preset state before serialization."""
        index = getattr(self, "_active_pipeline_section_index", None)
        if index is None or not 0 <= index < len(PANEL_NOTE_KEYS) or not hasattr(self, "pipeline_notes_widget"):
            return
        self.pipeline_panel_notes[self.pipeline_panel_note_key(index)] = limit_note_lines(self.pipeline_notes_widget.text())

    def show_pipeline_panel_note(self, index: int) -> None:
        """Display the active panel's independent note and expansion state."""
        key = self.pipeline_panel_note_key(index)
        self.pipeline_notes_widget.set_text(self.pipeline_panel_notes.get(key, ""))
        self.pipeline_notes_widget.set_expanded(self._pipeline_note_expanded.get(key, False))

    def on_pipeline_panel_note_changed(self, text: str) -> None:
        """Keep note edits synchronized with the active preset state."""
        index = getattr(self, "_active_pipeline_section_index", None)
        if index is None or not 0 <= index < len(PANEL_NOTE_KEYS):
            return
        self.pipeline_panel_notes[self.pipeline_panel_note_key(index)] = limit_note_lines(text)

    def on_pipeline_panel_note_expanded_changed(self, expanded: bool) -> None:
        """Remember each panel's editor size for the current app session."""
        index = getattr(self, "_active_pipeline_section_index", None)
        if index is None or not 0 <= index < len(PANEL_NOTE_KEYS):
            return
        self._pipeline_note_expanded[self.pipeline_panel_note_key(index)] = bool(expanded)

    def on_preview_tools_tab_changed(self, index: int) -> None:
        """Load the existing row context when users switch preview editors."""
        if bool(getattr(self, "_preview_tools_switching", False)):
            return
        active_defs = self.get_active_image_definitions()
        if not active_defs:
            return
        row = getattr(self, "_inline_analysis_settings_row", 0)
        row = int(row) if row is not None and 0 <= int(row) < len(active_defs) else 0
        self.show_analysis_settings_panel(row, "mask_adjust" if index == 0 else "filters")

    def on_preview_tools_source_changed(self, index: int) -> None:
        """Switch both tool tabs to the selected measured channel."""
        if bool(getattr(self, "_preview_tools_switching", False)) or index < 0:
            return
        source = self.preview_tools_source_combo.itemData(index)
        if not isinstance(source, dict):
            return
        row = source.get("row")
        if row is None:
            return
        self.preview_state.tools_source_label_override = str(source.get("label", "") or "")
        self.preview_state.tools_source_matches_pipeline = bool(source.get("matched", False))
        focus = "mask_adjust" if self.preview_tools_tabs.currentIndex() == 0 else "filters"
        self.show_analysis_settings_panel(int(row), focus)

    def update_preview_tools_source_options(self, active_defs: list[dict], current_row: int) -> None:
        """Keep the shared channel selector aligned with the active editor row."""
        combo = getattr(self, "preview_tools_source_combo", None)
        if combo is None:
            return
        previous_label = self.preview_state.tools_source_label_override
        physical_rows = [
            row for row, image_def in enumerate(active_defs) if self.is_physical_channel_definition(image_def)
        ]
        entries: list[dict] = []
        labels = list(self.preview_state.current_labels or [])
        roles = list(self.preview_state.current_layer_roles or [])
        if labels and len(labels) == len(roles):
            for preview_index, (label, role) in enumerate(zip(labels, roles, strict=True)):
                if str(role or "").strip().lower() != "image":
                    continue
                label_text = str(label or f"Image {preview_index + 1}")
                matched_row = next(
                    (
                        row
                        for row in physical_rows
                        if str(active_defs[row].get("name", "") or "").strip().casefold()
                        == label_text.strip().casefold()
                    ),
                    None,
                )
                fallback_row = matched_row if matched_row is not None else (physical_rows[0] if physical_rows else None)
                entries.append(
                    {
                        "label": label_text,
                        "row": fallback_row,
                        "preview_index": preview_index,
                        "matched": matched_row is not None,
                    }
                )
        else:
            entries = [
                {
                    "label": str(active_defs[row].get("name", "") or f"Image {row + 1}"),
                    "row": row,
                    "preview_index": None,
                    "matched": True,
                }
                for row in physical_rows
            ]

        self._preview_tools_switching = True
        combo.clear()
        for entry in entries:
            combo.addItem(entry["label"], entry)
        selected = combo.findText(previous_label)
        if selected < 0:
            selected = next(
                (index for index, entry in enumerate(entries) if entry.get("row") == int(current_row)),
                0,
            )
        combo.setCurrentIndex(selected if entries else -1)
        self.preview_state.tools_source_label_override = ""
        self.preview_state.tools_source_matches_pipeline = True
        if entries:
            selected_entry = entries[max(0, selected)]
            self.preview_state.tools_source_label_override = selected_entry["label"]
            self.preview_state.tools_source_matches_pipeline = bool(selected_entry["matched"])
        self._preview_tools_switching = False

    def update_preview_tools_context(self) -> None:
        """Explain whether the open image can drive the interactive tools."""
        label = getattr(self, "preview_tools_context_label", None)
        if label is None:
            return
        path_text = str(self.preview_state.file_path or "").strip()
        model = self.preview_state.tiff_model
        if not path_text or model is None:
            label.setText("Open a TIFF overlay to adjust masks or preview cell groups using linked cell measurements.")
            self.preview_tools_open_image_button.setVisible(True)
            self.preview_tools_section.set_summary("Open a TIFF overlay to use interactive preview tools")
            if hasattr(self, "preview_filter_refresh_button"):
                self.preview_filter_refresh_button.setEnabled(False)
            return
        roles = [str(role or "").lower() for role in self.preview_state.current_layer_roles or []]
        mask_count = sum("mask" in role and not role.startswith("adjusted_") for role in roles)
        label.setText(
            f"Editing {Path(path_text).name} · {mask_count} mask layer(s) available. "
            "Preview changes appear on the image at right."
        )
        self.preview_tools_open_image_button.setVisible(False)
        self.preview_tools_section.set_summary(f"Editing {Path(path_text).name}")
        if hasattr(self, "preview_filter_refresh_button"):
            self.preview_filter_refresh_button.setEnabled(True)
        active_defs = self.get_active_image_definitions()
        current_row = getattr(self, "_inline_analysis_settings_row", 0)
        if current_row is None or not 0 <= int(current_row) < len(active_defs):
            current_row = 0
        self.update_preview_tools_source_options(active_defs, int(current_row))
        if (
            str(getattr(self, "_inline_analysis_settings_panel", "") or "") == "filters"
            and active_defs
            and 0 <= int(current_row) < len(active_defs)
        ):
            selected_source = self.preview_tools_source_combo.currentData()
            selected_row = selected_source.get("row") if isinstance(selected_source, dict) else current_row
            if selected_row is not None and hasattr(self, "refresh_analysis_filter_panel_for_open_preview"):
                self.refresh_analysis_filter_panel_for_open_preview(int(selected_row))

    def show_adjacent_pipeline_section(self, offset: int) -> None:
        """Move between neighboring workflow editors."""
        if self._active_pipeline_section_index is None:
            return
        self.show_pipeline_section(self._active_pipeline_section_index + int(offset))

    def refresh_pipeline_section_summaries(self) -> None:
        """Refresh compact accordion summaries from the current workflow state."""
        required_paths = [self.input_dir.get().strip(), self.output_dir.get().strip()]
        selected_paths = sum(bool(path) for path in required_paths)
        setup_summary = (
            "Input and output folders selected"
            if selected_paths == 2
            else "Choose input and output folders"
            if selected_paths == 0
            else "1 of 2 required folders selected"
        )

        definitions = self.get_active_image_definitions()
        channel_count = sum(not bool(item.get("is_mask_only", False)) for item in definitions)
        mask_count = sum(
            bool(item.get("mask_slot_enabled", False)) or bool(item.get("is_mask_only", False)) for item in definitions
        )
        image_step_count = len(getattr(self, "_image_processing_step_refs", []) or [])
        measurement_step_count = len(getattr(self, "_measurement_processing_step_refs", []) or [])
        mask_step_count = len(getattr(self, "_mask_settings_step_refs", []) or [])
        cellpose_count = sum(bool(item.get("analysis_cell_segmentation_enabled", False)) for item in definitions)
        measurement_count = sum(bool(value) for value in dict(self.measurement_options or {}).values())

        self.project_section.set_summary(f"{setup_summary} · {channel_count} channel(s)")
        self.images_section.set_summary(f"{mask_count} mask source(s)")
        self.cellpose_settings_section.set_summary(
            f"{cellpose_count} reusable mask(s) configured" if cellpose_count else "Cell segmentation disabled"
        )
        self.image_processing_section.set_summary(
            f"{image_step_count} before masks · {mask_step_count} after masks · "
            f"{measurement_step_count} before measurement"
        )
        self.analysis_section.set_summary(f"{measurement_count} measurement option(s) enabled")
