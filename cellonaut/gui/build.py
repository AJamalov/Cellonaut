"""Main-window layout construction for the Cellonaut desktop GUI."""

from __future__ import annotations

import html as html_lib

from PySide6.QtCore import QSize, QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsView,
    QGraphicsScene,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QSpinBox,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from cellonaut.config.defaults import DEFAULT_THEME
from cellonaut.gui.help_content import HELP_PAGES, IMAGESCIENCE_HELP
from cellonaut.gui.help_browser import HelpBrowser
from cellonaut.gui.icons import colored_icon, monochrome_icon
from cellonaut.gui.preview_layer_list import PreviewLayerList
from cellonaut.gui.setup_workers import FijiInstallationScanWorker
from cellonaut.gui.ui_tokens import SPACING, WORKSPACE
from cellonaut.gui.preview_artifacts import PREVIEW_ARTIFACT_SELECTOR_LABELS
from cellonaut.gui.build_pipeline import CellonautGuiPipelineBuildMixin
from cellonaut.gui.theme import THEMES, apply_theme
from cellonaut.gui.widgets import ComboRow, ElidedLabel, ImageGraphicsView, PathRow
from cellonaut.resources import USER_APP_DIR, ensure_user_app_dirs, get_internal_fiji_path
from cellonaut.dependency_versions import collect_key_package_versions
from cellonaut.version import __version__


class CellonautGuiBuildMixin(CellonautGuiPipelineBuildMixin):
    """Build the main window and connect its shared panel widgets."""

    def shell_icon(self, name: str, size: int = 18):
        app = QApplication.instance()
        theme_name = str(app.property("cellonautTheme") or DEFAULT_THEME) if isinstance(app, QApplication) else DEFAULT_THEME
        return monochrome_icon(name, THEMES.get(theme_name, THEMES[DEFAULT_THEME]), size=size)

    def configure_preview_icon_button(self, button: QPushButton, icon_name: str, tooltip: str):
        button.setProperty("cellonautIconName", icon_name)
        button.setIcon(self.shell_icon(icon_name))
        button.setText("")
        if not button.toolTip():
            button.setToolTip(tooltip)
        # Give icon-only buttons labels that screen readers can read.
        button.setAccessibleName(tooltip)
        button.setFixedSize(WORKSPACE.preview_tool_button_size, WORKSPACE.preview_tool_button_size)

    def refresh_shell_icons(self) -> None:
        """Refresh theme-aware navigation and preview icons after a palette change."""
        for button in getattr(self, "primary_navigation_buttons", []):
            name = str(button.property("cellonautIconName") or "")
            if name:
                button.setIcon(self.shell_icon(name))
        if hasattr(self, "settings_nav_button"):
            self.settings_tab_default_icon = self.shell_icon("settings")
            self.settings_nav_button.setIcon(self.settings_tab_default_icon)
            if hasattr(self, "settings_warning_icon_label"):
                self.settings_warning_icon_label.setPixmap(
                    colored_icon("triangle-alert", "#FFD43B").pixmap(18, 18)
                )
        for button in self.findChildren(QPushButton):
            name = str(button.property("cellonautIconName") or "")
            if name:
                button.setIcon(self.shell_icon(name))
        if hasattr(self, "preview_empty_icon_label"):
            self.preview_empty_icon_label.setPixmap(self.shell_icon("image", 40).pixmap(40, 40))
        if hasattr(self, "preview_work_busy_icon"):
            self._preview_work_busy_base_pixmap = self.shell_icon("hourglass", 16).pixmap(16, 16)
            self.preview_work_busy_icon.setPixmap(self._preview_work_busy_base_pixmap)
        if hasattr(self, "preset_combo"):
            default_index = self.preset_combo.findText("Default")
            if default_index >= 0:
                self.preset_combo.setItemIcon(default_index, self.shell_icon("lock"))

    def update_preview_interaction_mode(self, snapshot_enabled: bool) -> None:
        """Use no-drag canvas input while the snapshot rectangle is active."""
        snapshot_enabled = bool(snapshot_enabled and self.preview_snapshot_toggle_button.isChecked())
        self.preview_view.setDragMode(
            QGraphicsView.DragMode.NoDrag if snapshot_enabled else QGraphicsView.DragMode.ScrollHandDrag
        )

    # The construction sequence is explicit because several later panels
    # connect to widgets created by earlier builders.
    def build_ui(self):
        self.build_main_layout()
        self.build_top_status()
        self.build_main_panes()
        self.build_left_pane()
        self.build_right_pane()
        self.build_pipeline_content()
        self.build_nd2_content()
        self.build_settings_content()
        self.build_workspace_content()
        self.finalize_ui()

    def build_main_layout(self):
        self.central = QWidget()
        self.setCentralWidget(self.central)

        self.root_layout = QVBoxLayout(self.central)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

    # Keep the workflow and image areas at a stable 50/50 proportion. A fixed
    # layout prevents accidental pane collapse and inconsistent saved geometry.
    def build_main_panes(self):
        self.main_workspace = QWidget()
        self.main_workspace.setObjectName("MainWorkspace")
        self.main_workspace_layout = QHBoxLayout(self.main_workspace)
        self.main_workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.main_workspace_layout.setSpacing(0)

        self.root_layout.addWidget(self.main_workspace, 1)
        self.root_layout.addWidget(self.bottom_status_frame)

    # Keep each task-focused help topic independent so users retain their place
    # while moving between the application and its guidance.
    def build_help_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        layout.setSpacing(SPACING.sm)

        help_topic_row = QHBoxLayout()
        help_topic_row.setContentsMargins(0, 0, 0, 0)
        help_topic_row.setSpacing(SPACING.sm)
        help_topic_label = QLabel("Help topic")
        help_topic_label.setProperty("uiRole", "formLabel")
        self.help_topic_selector = QComboBox()
        self.help_topic_selector.setAccessibleName("Help topic")
        self.help_topic_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        help_topic_row.addWidget(help_topic_label)
        help_topic_row.addWidget(self.help_topic_selector, 1)
        layout.addLayout(help_topic_row)

        self.help_stack = QStackedWidget()
        self.help_stack.setObjectName("HelpContentStack")
        self.help_browsers: list[QTextBrowser] = []
        for title, html in self.help_pages():
            if title == "About":
                continue
            browser = HelpBrowser()
            browser.setOpenExternalLinks(True)
            browser.setHtml(self.help_page_html(title, html))
            self.help_topic_selector.addItem(title, len(self.help_browsers))
            self.help_stack.addWidget(browser)
            self.help_browsers.append(browser)

        self.help_topic_selector.currentIndexChanged.connect(self.select_help_topic)
        layout.addWidget(self.help_stack, 1)

        return widget

    def select_help_topic(self, index: int) -> None:
        page_index = self.help_topic_selector.itemData(index)
        if page_index is None:
            return
        self.help_stack.setCurrentIndex(page_index)

    def open_help_topic(self, title: str) -> None:
        index = self.help_topic_selector.findText(title)
        if index >= 0:
            self.help_topic_selector.setCurrentIndex(index)
            self.show_left_page(self.help_tab)

    def make_help_link(self, topic: str) -> QLabel:
        link = QLabel('<a href="help">Help</a>')
        link.setAlignment(Qt.AlignmentFlag.AlignRight)
        link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.set_help_link_topic(link, topic)
        link.linkActivated.connect(lambda _url: self.open_help_topic(str(link.property("helpTopic"))))
        return link

    def set_help_link_topic(self, link: QLabel, topic: str) -> None:
        link.setProperty("helpTopic", topic)
        link.setAccessibleName(f"Help: {topic}")
        link.setToolTip(f"Open help: {topic}")

    def build_about_tab(self) -> QWidget:
        """Present application identity, citation, and licensing separately from task help."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        about_html = next((body for title, body in self.help_pages() if title == "About"), "")
        self.about_browser = QTextBrowser()
        self.about_browser.setObjectName("AboutBrowser")
        self.about_browser.setOpenExternalLinks(True)
        self.about_browser.setAccessibleName("About Cellonaut")
        self.about_browser.setHtml(self.help_page_html("About", about_html))
        layout.addWidget(self.about_browser)
        return widget

    # Wrap help fragments in one stylesheet so authored pages stay focused on
    # their content and remain visually consistent across platforms.
    def help_page_html(self, title: str, body: str) -> str:
        return f"""
        <html>
        <head>
        <style>
            body {{
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
                line-height: 1.45;
                margin: 10px;
            }}
            h2 {{
                margin-top: 0;
                margin-bottom: 8px;
            }}
            h3 {{
                margin-top: 16px;
                margin-bottom: 6px;
            }}
            ul, ol {{
                margin-top: 4px;
                margin-bottom: 10px;
            }}
            li {{
                margin-bottom: 4px;
            }}
            code {{
                font-family: Consolas, monospace;
            }}
            a {{
                color: #5aa9ff;
            }}
        </style>
        </head>
        <body>
        <h2>{title}</h2>
        {body}
        </body>
        </html>
        """

    def help_pages(self):
        return HELP_PAGES

    def configure_responsive_scroll_tab(self, scroll: QScrollArea, content: QWidget):
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setMinimumWidth(0)

        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def make_primary_navigation_button(self, label: str, icon_name: str) -> QPushButton:
        """Create one accessible destination in the persistent navigation rail."""
        button = QPushButton(label)
        button.setCheckable(True)
        button.setProperty("cellonautIconName", icon_name)
        button.setIcon(self.shell_icon(icon_name))
        button.setIconSize(QSize(18, 18))
        button.setProperty("uiRole", "primaryNavigation")
        button.setAccessibleName(label)
        button.setToolTip(label)
        button.setMinimumHeight(42)
        return button

    def show_left_page(self, page: QWidget) -> None:
        """Select a stacked page and keep its navigation button synchronized."""
        index = self.left_stack.indexOf(page)
        if index < 0:
            return
        self.left_stack.setCurrentIndex(index)
        button = self.left_page_buttons.get(page)
        if button is not None:
            button.setChecked(True)

    def configure_primary_navigation(self) -> None:
        """Register left-side pages in workflow order after every page exists."""
        page_specs = [
            ("pipeline", "Pipeline", self.pipeline_tab, "pipeline"),
            ("files", "Files", self.files_left_tab, "files"),
            ("log", "Log", self.log_tab, "log"),
            ("settings", "Settings", self.settings_tab, "settings"),
            ("help", "Help", self.help_tab, "help"),
            ("about", "About", self.about_tab, "info"),
        ]

        self.left_page_buttons: dict[QWidget, QPushButton] = {}
        self.primary_navigation_buttons: list[QPushButton] = []
        self.primary_navigation_button_group = QButtonGroup(self.as_qobject())
        self.primary_navigation_button_group.setExclusive(True)

        for position, (key, label, page, icon_name) in enumerate(page_specs):
            self.left_stack.addWidget(page)
            button = self.make_primary_navigation_button(label, icon_name)
            setattr(self, f"{key}_nav_button", button)
            self.left_page_buttons[page] = button
            self.primary_navigation_buttons.append(button)
            self.primary_navigation_button_group.addButton(button, position)
            button.clicked.connect(lambda checked=False, target=page: self.show_left_page(target) if checked else None)

            if key == "settings":
                self.settings_tab_default_icon = button.icon()
                self.settings_warning_icon_label = QLabel(button)
                self.settings_warning_icon_label.setObjectName("SettingsWarningIcon")
                self.settings_warning_icon_label.setAccessibleName("ImageScience warning")
                self.settings_warning_icon_label.setToolTip("ImageScience is not installed")
                self.settings_warning_icon_label.setPixmap(
                    colored_icon("triangle-alert", "#FFD43B").pixmap(18, 18)
                )
                self.settings_warning_icon_label.setFixedSize(18, 18)
                self.settings_warning_icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                warning_layout = QHBoxLayout(button)
                warning_layout.setContentsMargins(0, 0, SPACING.sm, 0)
                warning_layout.addStretch(1)
                warning_layout.addWidget(self.settings_warning_icon_label)
                self.settings_warning_icon_label.hide()
                self.primary_navigation_layout.addStretch(1)
            self.primary_navigation_layout.addWidget(button)
            if key == "settings":
                self.guided_tutorial_button = QPushButton("Tutorial")
                self.guided_tutorial_button.setObjectName("GuidedTutorialButton")
                self.guided_tutorial_button.setProperty("cellonautIconName", "book-open")
                self.guided_tutorial_button.setIcon(self.shell_icon("book-open"))
                self.guided_tutorial_button.setIconSize(QSize(18, 18))
                self.guided_tutorial_button.setProperty("uiRole", "primaryNavigation")
                self.guided_tutorial_button.setMinimumHeight(42)
                self.guided_tutorial_button.clicked.connect(self.toggle_guided_tutorial)
                self.primary_navigation_layout.addWidget(self.guided_tutorial_button)

        self.show_left_page(self.pipeline_tab)

    # Switch the left panel while keeping the image preview visible.
    def build_left_pane(self):
        self.left_container = QWidget()
        self.left_container.setMinimumWidth(470)
        self.left_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        left_layout = QHBoxLayout(self.left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.primary_navigation_rail = QFrame()
        self.primary_navigation_rail.setObjectName("PrimaryNavigationRail")
        self.primary_navigation_rail.setFixedWidth(WORKSPACE.sidebar_width)
        self.primary_navigation_layout = QVBoxLayout(self.primary_navigation_rail)
        self.primary_navigation_layout.setContentsMargins(SPACING.xs, SPACING.sm, SPACING.xs, SPACING.sm)
        self.primary_navigation_layout.setSpacing(SPACING.xs)

        self.left_stack = QStackedWidget()
        self.left_stack.setObjectName("LeftContentStack")
        self.left_stack.setMinimumWidth(0)
        self.left_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.primary_navigation_rail)
        left_layout.addWidget(self.left_stack, 1)

        self.pipeline_tab = QWidget()
        self.settings_tab = QWidget()

        self.build_files_content()

        self.pipeline_tab_outer_layout = QVBoxLayout(self.pipeline_tab)
        self.pipeline_tab_outer_layout.setContentsMargins(0, 0, 0, 0)
        self.pipeline_tab_outer_layout.setSpacing(0)

        self.pipeline_scroll = QScrollArea()

        self.pipeline_tab_content = QWidget()
        self.pipeline_tab_layout = QVBoxLayout(self.pipeline_tab_content)
        self.pipeline_tab_layout.setContentsMargins(8, 8, 8, 8)
        self.pipeline_tab_layout.setSpacing(SPACING.sm)

        self.configure_responsive_scroll_tab(self.pipeline_scroll, self.pipeline_tab_content)
        self.pipeline_scroll.setWidget(self.pipeline_tab_content)
        self.pipeline_tab_outer_layout.addWidget(self.pipeline_scroll, 1)

        self.settings_tab_outer_layout = QVBoxLayout(self.settings_tab)
        self.settings_tab_outer_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_tab_outer_layout.setSpacing(0)

        self.settings_scroll = QScrollArea()

        self.settings_tab_content = QWidget()
        self.settings_tab_layout = QVBoxLayout(self.settings_tab_content)
        self.settings_tab_layout.setContentsMargins(8, 8, 8, 8)
        self.settings_tab_layout.setSpacing(SPACING.sm)

        self.configure_responsive_scroll_tab(self.settings_scroll, self.settings_tab_content)
        self.settings_scroll.setWidget(self.settings_tab_content)
        self.settings_tab_outer_layout.addWidget(self.settings_scroll, 1)

        self.main_workspace_layout.addWidget(self.left_container, 1)

    def build_right_pane(self):
        self.right_container = QWidget()
        self.right_container.setMinimumWidth(390)
        self.right_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(8)

        self.main_workspace_layout.addWidget(self.right_container, 1)

    # Keep ND2 conversion in its own persistent workflow so Setup remains
    # compact and the importer is available before a source folder is chosen.
    def build_nd2_content(self):
        self.nd2_dialog = QDialog(self)
        self.nd2_dialog.setWindowTitle("Import ND2")
        self.nd2_dialog.setModal(False)
        self.nd2_dialog.setMinimumWidth(700)

        dialog_layout = QVBoxLayout(self.nd2_dialog)
        dialog_layout.setContentsMargins(12, 12, 12, 12)
        dialog_layout.setSpacing(10)

        intro = QLabel("Choose a folder containing ND2 files, then convert the files to TIFF.")
        intro.setWordWrap(True)
        dialog_layout.addWidget(intro)

        self.nd2_source_dir = PathRow(
            "ND2 source folder",
            mode="dir",
            tooltip="Folder containing ND2 files, including ND2 files in nested folders.",
        )
        self.nd2_source_dir.edit.textChanged.connect(self.on_nd2_source_path_changed)
        dialog_layout.addWidget(self.nd2_source_dir)

        self.nd2_source_scan_timer = QTimer(self)
        self.nd2_source_scan_timer.setSingleShot(True)
        self.nd2_source_scan_timer.setInterval(350)
        self.nd2_source_scan_timer.timeout.connect(self.inspect_nd2_source)

        self.nd2_output_dir = PathRow(
            "Converted TIFF folder",
            mode="dir",
            tooltip="Folder for converted OME-TIFF stacks. Existing filenames receive a numbered suffix; existing files are kept.",
        )
        dialog_layout.addWidget(self.nd2_output_dir)

        self.nd2_group = self.make_section("Conversion settings")

        self.nd2_detected_channels_label = QLabel("Waiting for files...")
        self.nd2_detected_channels_label.setWordWrap(True)

        self.nd2_z_mode = ComboRow(
            "Z handling",
            ["Max projection", "Single Z slice"],
            default="Max projection",
            tooltip="Choose how ND2 Z stacks are reduced during TIFF conversion.",
            on_change=self.on_nd2_z_mode_changed,
        )

        self.nd2_z_index_row = QWidget()
        nd2_z_index_layout = QHBoxLayout(self.nd2_z_index_row)
        nd2_z_index_layout.setContentsMargins(0, 0, 0, 0)
        nd2_z_index_layout.setSpacing(8)
        self.nd2_z_index_label = QLabel("Z slice")
        self.nd2_z_index_label.setMinimumWidth(150)
        self.nd2_z_index_label.setMaximumWidth(150)
        self.nd2_z_index_spin = QSpinBox()
        self.nd2_z_index_spin.setRange(1, 10000)
        self.nd2_z_index_spin.setValue(1)
        self.nd2_z_index_spin.setToolTip("Slice to keep, starting at 1.")
        self.nd2_z_index_spin.valueChanged.connect(self.on_nd2_z_index_changed)
        nd2_z_index_layout.addWidget(self.nd2_z_index_label)
        nd2_z_index_layout.addWidget(self.nd2_z_index_spin, 1)

        self.nd2_channel_rows_container = QWidget()
        self.nd2_channel_rows_container_layout = QVBoxLayout(self.nd2_channel_rows_container)
        self.nd2_channel_rows_container_layout.setContentsMargins(0, 0, 0, 0)
        self.nd2_channel_rows_container_layout.setSpacing(6)

        self.nd2_mapping_help = QLabel("ND2 channel mapping will appear after channel inspection.")
        self.nd2_mapping_help.setWordWrap(True)

        self.nd2_convert_button = QPushButton("Convert ND2 to TIFF")
        self.nd2_convert_button.setToolTip("Convert detected ND2 files to OME-TIFF using the channel mapping and Z settings. Channels with blank output names are skipped.")
        self.nd2_convert_button.clicked.connect(self.convert_nd2_folder_clicked)
        self.set_button_role(self.nd2_convert_button, "primary")

        self.nd2_group.add_widget(self.nd2_detected_channels_label)
        self.nd2_group.add_widget(self.nd2_z_mode)
        self.nd2_group.add_widget(self.nd2_z_index_row)
        self.nd2_group.add_widget(self.nd2_channel_rows_container)
        self.nd2_group.add_widget(self.nd2_mapping_help)
        self.nd2_group.add_widget(self.nd2_convert_button)

        dialog_layout.addWidget(self.nd2_group)

        self.nd2_progress_panel = QWidget()
        nd2_progress_layout = QVBoxLayout(self.nd2_progress_panel)
        nd2_progress_layout.setContentsMargins(0, 0, 0, 0)
        nd2_progress_layout.setSpacing(8)
        self.nd2_progress_label = QLabel("")
        self.nd2_progress_label.setWordWrap(True)
        self.nd2_progress_bar = QProgressBar()
        self.nd2_progress_bar.setRange(0, 100)
        self.nd2_progress_bar.setValue(0)
        nd2_progress_layout.addWidget(self.nd2_progress_label)
        nd2_progress_layout.addWidget(self.nd2_progress_bar)

        nd2_progress_buttons = QHBoxLayout()
        self.nd2_cancel_button = QPushButton("Cancel conversion")
        self.nd2_cancel_button.clicked.connect(self.cancel_nd2_conversion_clicked)
        self.nd2_open_output_button = QPushButton("Open converted folder")
        self.nd2_open_output_button.clicked.connect(self.open_nd2_output_folder)
        nd2_progress_buttons.addWidget(self.nd2_cancel_button)
        nd2_progress_buttons.addWidget(self.nd2_open_output_button)
        nd2_progress_layout.addLayout(nd2_progress_buttons)
        self.nd2_progress_panel.setVisible(False)
        dialog_layout.addWidget(self.nd2_progress_panel)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.nd2_dialog.close)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_row.addWidget(close_button)
        dialog_layout.addLayout(close_row)

        self.rebuild_nd2_channel_rows()
        self.update_nd2_z_controls()
        self.set_nd2_conversion_controls_visible(False)

    # Version rows make the bundled runtime inventory useful without exposing jar paths.
    def fiji_component_checklist_html(
        self, path_text: str, status=None, *, checking: bool = False, scan_error: str = ""
    ) -> str:
        path_text = str(path_text or "").strip()
        app = QApplication.instance()
        theme_name = str(app.property("cellonautTheme") or DEFAULT_THEME) if isinstance(app, QApplication) else DEFAULT_THEME
        theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
        ok_color = theme["success"]
        bad_color = theme["danger"]
        muted_color = theme["muted"]
        warn_color = theme["warning"]

        # Escape paths and scanner messages because QLabel renders this string as rich text.
        def line(
            ok: bool,
            label: str,
            detail: str = "",
            *,
            version: str | None = None,
            show_detail: bool = False,
        ) -> str:
            icon = "&#10003;" if ok else "&#10007;"
            color = ok_color if ok else bad_color
            text = html_lib.escape(label)
            state = html_lib.escape(version or "version unknown") if ok else "unavailable"
            suffix = (
                f" <span style='color:{muted_color};'>- {html_lib.escape(detail)}</span>"
                if detail and show_detail
                else ""
            )
            return f"<div><span style='color:{color}; font-weight:700;'>{icon}</span> {text} {state}{suffix}</div>"

        if not path_text:
            return (
                f"<div style='color:{bad_color};'>Bundled Fiji runtime not found. "
                "Reinstall Cellonaut from the complete official offline package.</div>"
            )
        if scan_error:
            return line(False, "Fiji component check", scan_error, show_detail=True)
        if checking or status is None:
            return (
                f"<div><span style='color:{warn_color}; font-weight:700;'>...</span> "
                "Checking Fiji components...</div>"
            )

        if not status.path_exists:
            return line(False, "Fiji folder")
        if not status.path_is_dir:
            return line(False, "Fiji folder")
        if not status.looks_like_fiji:
            return line(False, "Fiji installation")

        lines = [line(True, "Fiji", version=status.version)]
        for component in status.components:
            if component.key == "launcher" and component.ok:
                continue
            if component.key == "imagescience" and not component.ok:
                lines.append(
                    f"<div style='font-size:15px; margin-top:6px;'><span style='color:{warn_color}; "
                    "font-weight:800;'>! ImageScience is not installed.</span><br>"
                    f"{IMAGESCIENCE_HELP}<br><br>"
                    "<b>How to install ImageScience:</b><br>"
                    "1. Click <b>Open Fiji Folder</b>.<br>"
                    "2. Run the Fiji launcher from that folder.<br>"
                    "3. In Fiji, choose <b>Help &gt; Update...</b>.<br>"
                    "4. Click <b>Manage update sites</b> and enable <b>ImageScience</b>.<br>"
                    "5. Click <b>Apply changes</b>, restart Fiji to finish the update, then restart Cellonaut.</div>"
                )
                continue
            lines.append(line(component.ok, component.label, component.detail, version=component.version))
        return "".join(lines)

    def runtime_installation_summary_html(self, fiji_path: str) -> str:
        """Summarize the Python-side application versions."""
        versions = collect_key_package_versions()
        cellpose_version = html_lib.escape(str(versions.get("cellpose", "unknown")))
        torch_version = html_lib.escape(str(versions.get("torch", "unknown")))
        return (
            f"<div style='font-size:14px; font-weight:700;'>Cellonaut {html_lib.escape(__version__)}</div>"
            f"<div><b>Cellpose:</b> {cellpose_version}</div>"
            f"<div><b>PyTorch:</b> {torch_version}</div>"
        )

    # Reuse a matching result and otherwise schedule scanning away from the GUI thread.
    def update_fiji_component_checklist(self):
        label = getattr(self, "fiji_component_checklist", None)
        if label is None:
            return

        path_text = self.fiji_app.get() if hasattr(self, "fiji_app") else ""
        open_button = getattr(self, "open_fiji_folder_button", None)
        if open_button is not None:
            open_button.setVisible(False)
        summary = getattr(self, "runtime_installation_summary", None)
        if summary is not None:
            summary.setText(self.runtime_installation_summary_html(path_text))
        self._pending_fiji_scan_path = path_text
        if not str(path_text or "").strip():
            label.setText(self.fiji_component_checklist_html(path_text))
            self.update_settings_tab_imagescience_warning(None)
            return

        cached = getattr(self, "_last_fiji_scan_result", {}) or {}
        if str(cached.get("path", "") or "") == str(path_text or "") and cached.get("status") is not None:
            self.apply_fiji_component_scan_result(cached)
            return

        label.setText(self.fiji_component_checklist_html(path_text, checking=True))
        self.update_settings_tab_imagescience_warning(None)
        self.start_pending_fiji_component_scan()

    def update_settings_tab_imagescience_warning(self, status) -> None:
        """Show the Settings warning icon only when ImageScience is confirmed missing."""
        button = getattr(self, "settings_nav_button", None)
        if button is None:
            return

        image_science_missing = bool(
            status is not None
            and status.looks_like_fiji
            and any(component.key == "imagescience" and not component.ok for component in status.components)
        )
        warning_label = getattr(self, "settings_warning_icon_label", None)
        button.setIcon(self.settings_tab_default_icon)
        if image_science_missing:
            if warning_label is not None:
                warning_label.show()
            button.setToolTip("ImageScience is not installed. Open Settings for installation instructions.")
        else:
            if warning_label is not None:
                warning_label.hide()
            button.setToolTip("Settings")

    # Allow only one scan at a time; the latest edited path remains pending for the next worker.
    def start_pending_fiji_component_scan(self):
        if getattr(self, "_closing_requested", False):
            return
        path_text = str(getattr(self, "_pending_fiji_scan_path", "") or "").strip()
        thread = getattr(self, "_fiji_scan_thread", None)
        if thread is not None and thread.isRunning():
            return
        if not path_text:
            return

        self._pending_fiji_scan_path = ""
        worker = FijiInstallationScanWorker(path_text)
        thread, worker = self.prepare_worker_thread(
            worker,
            terminal_signal=worker.done_signal,
            result_callback=self.apply_fiji_component_scan_result,
            finished_callback=self.cleanup_fiji_component_scan,
            thread_factory=QThread,
        )
        self._fiji_scan_worker = worker
        self._fiji_scan_thread = thread
        thread.start()

    # Ignore this result if the Fiji location changed during the scan.
    def apply_fiji_component_scan_result(self, result: dict):
        result = dict(result or {})
        if result.get("scan_cancelled") or getattr(self, "_closing_requested", False):
            return
        scanned_path = str(result.get("path", "") or "")
        current_path = self.fiji_app.get() if hasattr(self, "fiji_app") else ""
        if scanned_path != str(current_path or "").strip():
            return

        self._last_fiji_scan_result = result
        label = getattr(self, "fiji_component_checklist", None)
        status = result.get("status")
        self.update_settings_tab_imagescience_warning(status)
        if label is not None:
            label.setText(
                self.fiji_component_checklist_html(
                    scanned_path,
                    status=status,
                    scan_error=str(result.get("scan_error", "") or ""),
                )
            )
        open_button = getattr(self, "open_fiji_folder_button", None)
        if open_button is not None:
            image_science_missing = bool(
                status is not None
                and status.looks_like_fiji
                and any(component.key == "imagescience" and not component.ok for component in status.components)
            )
            open_button.setVisible(image_science_missing)
            open_button.setEnabled(image_science_missing)

    def open_bundled_fiji_folder(self):
        """Open the packaged Fiji directory so optional plugins can be installed quickly."""
        path_text = self.fiji_app.get() if hasattr(self, "fiji_app") else ""
        if str(path_text or "").strip():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path_text)))

    def open_user_data_folder(self):
        """Open the writable folder containing presets, settings, and crash logs."""
        try:
            ensure_user_app_dirs()
        except OSError as exc:
            QMessageBox.warning(self, "Cellonaut data folder", f"Could not create the Cellonaut data folder:\n{exc}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(USER_APP_DIR))):
            QMessageBox.warning(self, "Cellonaut data folder", f"Could not open:\n{USER_APP_DIR}")

    # Start a deferred rescan only after Qt has fully released the previous thread.
    def cleanup_fiji_component_scan(self):
        self._fiji_scan_worker = None
        self._fiji_scan_thread = None
        pending = str(getattr(self, "_pending_fiji_scan_path", "") or "").strip()
        current = self.fiji_app.get() if hasattr(self, "fiji_app") else ""
        completed = str((getattr(self, "_last_fiji_scan_result", {}) or {}).get("path", "") or "")
        if pending and pending == str(current or "").strip() and pending != completed:
            QTimer.singleShot(0, self.start_pending_fiji_component_scan)

    # Appearance and external-tool settings stay outside pipeline presets
    # because they belong to the installation rather than an experiment.
    def build_settings_content(self):
        self.appearance_group = self.make_section("Appearance")
        self.appearance_group.add_widget(self.make_help_link("Presets & Settings"))
        self.appearance_mode = ComboRow(
            "Theme",
            [
                ("Light Blue", "light_blue"),
                ("Dark Blue", "dark_blue"),
                ("Dark Teal", "dark_teal"),
                ("Dark Purple", "dark_purple"),
            ],
            default=DEFAULT_THEME,
            tooltip="Choose the visual theme and accent color for the application.",
            on_change=self.on_theme_changed,
        )
        self.appearance_group.add_rows([self.appearance_mode])

        self.external_tools_group = self.make_section("Runtime status")
        bundled_fiji = get_internal_fiji_path()
        # Keep the Fiji path available to configuration code without showing it as a user setting.
        self.fiji_app = PathRow(
            "Bundled Fiji runtime",
            mode="dir",
            default=str(bundled_fiji or ""),
        )

        self.runtime_installation_summary = QLabel()
        self.runtime_installation_summary.setWordWrap(True)
        self.runtime_installation_summary.setTextFormat(Qt.TextFormat.RichText)
        self.runtime_installation_summary.setProperty("uiRole", "statusText")
        self.runtime_installation_summary.setText(
            self.runtime_installation_summary_html(self.fiji_app.get())
        )
        self.external_tools_group.add_widget(self.runtime_installation_summary)

        fiji_component_row = QWidget()
        fiji_component_layout = QHBoxLayout(fiji_component_row)
        fiji_component_layout.setContentsMargins(0, 0, 0, 0)
        fiji_component_layout.setSpacing(10)

        self.fiji_component_checklist = QLabel()
        self.fiji_component_checklist.setWordWrap(True)
        self.fiji_component_checklist.setTextFormat(Qt.TextFormat.RichText)
        self.fiji_component_checklist.setProperty("uiRole", "mutedLabel")
        fiji_component_layout.addWidget(self.fiji_component_checklist, 1)

        self.open_fiji_folder_button = QPushButton("Open Fiji Folder")
        self.open_fiji_folder_button.setVisible(False)
        self.open_fiji_folder_button.clicked.connect(self.open_bundled_fiji_folder)
        fiji_component_layout.addWidget(self.open_fiji_folder_button, 0, Qt.AlignmentFlag.AlignTop)
        self.external_tools_group.add_widget(fiji_component_row)

        self.update_fiji_component_checklist()

        self.data_folder_group = self.make_section("Cellonaut data folder")
        self.data_folder_help_label = QLabel(
            "Open Cellonaut's data folder to access crash logs, saved settings, and preset files."
        )
        self.data_folder_help_label.setWordWrap(True)
        self.data_folder_help_label.setProperty("uiRole", "mutedLabel")
        self.data_folder_group.add_widget(self.data_folder_help_label)

        self.open_data_folder_button = QPushButton("Open Cellonaut Data Folder")
        self.open_data_folder_button.clicked.connect(self.open_user_data_folder)
        self.data_folder_group.add_widget(self.open_data_folder_button)

        self.settings_tab_layout.addWidget(self.appearance_group)
        self.settings_tab_layout.addWidget(self.data_folder_group)
        self.settings_tab_layout.addWidget(self.external_tools_group)
        self.settings_tab_layout.addStretch(1)

    def build_workspace_content(self):
        self._build_preview_toolbar()
        self._build_preview_canvas_controls()
        self._build_preview_inspector()
        self._assemble_preview_workspace()
        self._build_workspace_support_tabs()

    def _build_preview_toolbar(self):
        self.preview_tab = QWidget()
        self.preview_tab.setObjectName("MainImageWorkspace")
        self.preview_tab_layout = QVBoxLayout(self.preview_tab)
        self.preview_tab_layout.setContentsMargins(8, 8, 8, 8)
        self.preview_tab_layout.setSpacing(8)

        self.preview_workspace_title = QLabel("Image Preview")
        self.preview_workspace_title.setProperty("uiRole", "workspaceTitle")
        self.preview_workspace_title.setAccessibleName("Image preview workspace")

        self.preview_toolbar = QFrame()
        self.preview_toolbar.setObjectName("PreviewHeaderToolbar")
        self.preview_toolbar.setProperty("uiRole", "toolbar")
        preview_toolbar_layout = QHBoxLayout(self.preview_toolbar)
        preview_toolbar_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        preview_toolbar_layout.setSpacing(SPACING.sm)

        self.preview_snapshot_toggle_button = QPushButton("Snapshot")
        self.preview_snapshot_toggle_button.setCheckable(True)
        self.preview_snapshot_toggle_button.setToolTip(
            "Show or hide the square snapshot selection on the preview image."
        )
        self.preview_snapshot_toggle_button.toggled.connect(self.toggle_preview_snapshot_square)
        self.preview_snapshot_toggle_button.toggled.connect(self.update_preview_interaction_mode)
        self.configure_preview_icon_button(
            self.preview_snapshot_toggle_button,
            "snapshot",
            "Snapshot Selection",
        )

        self.preview_snapshot_capture_button = QPushButton("Capture")
        self.preview_snapshot_capture_button.setToolTip("Save the selected square as a visible PNG or a layer montage PNG.")
        snapshot_menu = QMenu(self.preview_snapshot_capture_button)
        self.preview_snapshot_visible_action = snapshot_menu.addAction("Visible snapshot PNG")
        self.preview_snapshot_visible_action.triggered.connect(self.save_visible_snapshot_from_selection)
        self.preview_snapshot_montage_action = snapshot_menu.addAction("Layer montage PNG")
        self.preview_snapshot_montage_action.setEnabled(False)
        self.preview_snapshot_montage_action.triggered.connect(self.save_layer_snapshot_montage_from_selection)
        self.preview_snapshot_capture_button.setMenu(snapshot_menu)
        self.preview_snapshot_capture_button.setEnabled(False)
        self.preview_export_image_button = QPushButton("Save image")
        self.configure_preview_icon_button(self.preview_export_image_button, "save", "Save preview image")
        self.preview_export_image_button.setToolTip(
            "Save the full current page as a flattened 8-bit RGB TIFF or PNG, including visible layers, edits, and cell groups. Zoom and the snapshot square do not crop this export."
        )
        self.preview_export_image_button.setEnabled(False)
        self.preview_export_image_button.clicked.connect(self.export_preview_image)
        self.configure_preview_icon_button(
            self.preview_snapshot_capture_button,
            "snapshot",
            "Save snapshot",
        )

        self.preview_artifact_selector = QComboBox()
        self.preview_artifact_selector.setToolTip("Choose which generated preview files to browse.")
        self.preview_artifact_selector.setMinimumWidth(190)
        self.preview_artifact_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for mode, label in PREVIEW_ARTIFACT_SELECTOR_LABELS.items():
            self.preview_artifact_selector.addItem(label, mode)
        self.preview_artifact_selector.currentIndexChanged.connect(
            lambda _index=0: self.select_preview_artifact_view(self.preview_artifact_selector.currentData())
        )
        self.refresh_snapshot_artifact_availability()
        self.preview_prev_sample_button = QPushButton("Previous sample")
        self.configure_preview_icon_button(
            self.preview_prev_sample_button,
            "previous",
            "Previous sample",
        )
        self.preview_prev_sample_button.clicked.connect(lambda: self.step_preview_artifact_sample(-1))

        self.preview_next_sample_button = QPushButton("Next sample")
        self.configure_preview_icon_button(
            self.preview_next_sample_button,
            "next",
            "Next sample",
        )
        self.preview_next_sample_button.clicked.connect(lambda: self.step_preview_artifact_sample(1))

        self.preview_sample_nav_label = QLabel("Sample: - / -")
        self.preview_sample_nav_label.setProperty("muted", "true")
        self.preview_sample_nav_label.setProperty("uiRole", "mutedLabel")

        self.preview_prev_artifact_button = QPushButton("Previous image")
        self.configure_preview_icon_button(
            self.preview_prev_artifact_button,
            "previous",
            "Previous image for this sample",
        )
        self.preview_prev_artifact_button.clicked.connect(lambda: self.step_preview_artifact_file(-1))

        self.preview_next_artifact_button = QPushButton("Next image")
        self.configure_preview_icon_button(
            self.preview_next_artifact_button,
            "next",
            "Next image for this sample",
        )
        self.preview_next_artifact_button.clicked.connect(lambda: self.step_preview_artifact_file(1))

        self.preview_artifact_nav_label = QLabel("Preview: - / -")
        self.preview_artifact_nav_label.setProperty("muted", "true")
        self.preview_artifact_nav_label.setProperty("uiRole", "mutedLabel")

        self.preview_metadata_button = QPushButton("Show metadata")
        self.preview_metadata_button.setCheckable(True)
        self.configure_preview_icon_button(
            self.preview_metadata_button,
            "info",
            "Show metadata",
        )
        self.preview_metadata_button.setToolTip("Show or hide preview image metadata.")
        self.preview_metadata_button.toggled.connect(self.toggle_preview_metadata_panel)

        self.preview_inspector_toggle_button = QPushButton("Layers")
        self.preview_inspector_toggle_button.setCheckable(True)
        self.preview_inspector_toggle_button.setAccessibleName("Show image layers and metadata")
        self.preview_inspector_toggle_button.setToolTip("Show overlay layers and image metadata.")
        self.preview_inspector_toggle_button.toggled.connect(self.toggle_preview_inspector)

        self._recent_preview_images: list[str] = []
        self.preview_recent_images_button = QPushButton("Recent images")
        self.preview_recent_images_button.setToolTip("Reopen one of the 10 most recently viewed images.")
        self.preview_recent_images_menu = QMenu(self.preview_recent_images_button)
        self.preview_recent_images_menu.setToolTipsVisible(True)
        self.preview_recent_images_menu.aboutToShow.connect(self.rebuild_recent_preview_images_menu)
        self.preview_recent_images_button.setMenu(self.preview_recent_images_menu)
        self.preview_recent_images_button.setEnabled(False)

        self.preview_fullscreen_button = QPushButton("Fullscreen")
        self.preview_fullscreen_button.setCheckable(True)
        self.configure_preview_icon_button(
            self.preview_fullscreen_button,
            "fullscreen",
            "Expand preview workspace",
        )
        self.preview_fullscreen_button.toggled.connect(self.set_preview_focus_mode)

        self.preview_sample_navigation = QWidget()
        self.preview_sample_navigation.setProperty("uiRole", "previewToolbarGroup")
        preview_sample_navigation_layout = QHBoxLayout(self.preview_sample_navigation)
        preview_sample_navigation_layout.setContentsMargins(0, 0, 0, 0)
        preview_sample_navigation_layout.setSpacing(SPACING.xs)
        preview_sample_navigation_layout.addWidget(self.preview_prev_sample_button)
        preview_sample_navigation_layout.addWidget(self.preview_next_sample_button)
        preview_sample_navigation_layout.addWidget(self.preview_sample_nav_label)

        self.preview_image_actions = QWidget()
        self.preview_image_actions.setProperty("uiRole", "previewToolbarGroup")
        preview_image_actions_layout = QHBoxLayout(self.preview_image_actions)
        preview_image_actions_layout.setContentsMargins(SPACING.sm, 0, 0, 0)
        preview_image_actions_layout.setSpacing(SPACING.xs)
        preview_image_actions_layout.addWidget(self.preview_recent_images_button)
        preview_image_actions_layout.addWidget(self.preview_inspector_toggle_button)

        preview_toolbar_layout.addWidget(self.preview_artifact_selector, 1)
        preview_toolbar_layout.addWidget(self.preview_sample_navigation)
        preview_toolbar_layout.addWidget(self.preview_image_actions)
        preview_toolbar_layout.addWidget(self.make_help_link("Image Preview Tools"))

    def _build_preview_canvas_controls(self):
        self.preview_page_label = QLabel("Page: - / -")
        self.preview_page_label.setProperty("muted", "true")
        self.preview_page_label.setProperty("uiRole", "mutedLabel")

        self.preview_filter_status_label = ElidedLabel("No linked filter results", mode=Qt.TextElideMode.ElideRight)
        self.preview_filter_status_label.setProperty("muted", "true")
        self.preview_filter_status_label.setProperty("uiRole", "mutedLabel")

        self.preview_file_title_label = ElidedLabel(
            "No preview loaded. Run Preview One Sample or open an image from the Files tab."
        )
        self.preview_file_title_label.setProperty("muted", "true")
        self.preview_file_title_label.setProperty("uiRole", "mutedLabel")
        self.preview_file_title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.preview_scene = QGraphicsScene(self.as_qobject())
        self.preview_view = ImageGraphicsView()
        self.preview_view.setAcceptDrops(False)
        self.preview_view.setScene(self.preview_scene)
        preview_viewport_layout = QVBoxLayout(self.preview_view.viewport())
        preview_viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_empty_state = QWidget()
        self.preview_empty_state.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        preview_empty_layout = QVBoxLayout(self.preview_empty_state)
        preview_empty_layout.setContentsMargins(0, 0, 0, 0)
        preview_empty_layout.setSpacing(SPACING.sm)
        self.preview_empty_icon_label = QLabel()
        self.preview_empty_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_empty_icon_label.setPixmap(self.shell_icon("image", 40).pixmap(40, 40))
        self.preview_empty_text_label = QLabel("Drag and drop image here")
        self.preview_empty_text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_empty_text_label.setProperty("uiRole", "mutedLabel")
        preview_empty_layout.addWidget(self.preview_empty_icon_label)
        preview_empty_layout.addWidget(self.preview_empty_text_label)
        preview_viewport_layout.addStretch(1)
        preview_viewport_layout.addWidget(self.preview_empty_state, 0, Qt.AlignmentFlag.AlignCenter)
        preview_viewport_layout.addStretch(1)
        self.preview_zoom_label = QLabel("100%")
        self.preview_zoom_label.setObjectName("PreviewZoomLabel")
        self.preview_zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_zoom_label.setMinimumWidth(52)
        self.preview_view.zoom_changed.connect(lambda percent: self.preview_zoom_label.setText(f"{percent}%"))

        self.preview_zoom_in_button = QPushButton("Zoom in")
        self.configure_preview_icon_button(
            self.preview_zoom_in_button,
            "zoom_in",
            "Zoom in",
        )
        self.preview_zoom_in_button.clicked.connect(lambda: self.preview_view.zoom_by_steps(1))

        self.preview_zoom_out_button = QPushButton("Zoom out")
        self.configure_preview_icon_button(
            self.preview_zoom_out_button,
            "zoom_out",
            "Zoom out",
        )
        self.preview_zoom_out_button.clicked.connect(lambda: self.preview_view.zoom_by_steps(-1))

        self.preview_reset_view_button = QPushButton("Reset View")
        self.preview_reset_view_button.setToolTip("Fit the complete image and overlays in the canvas.")
        self.preview_reset_view_button.clicked.connect(self.fit_preview_image)
        self.configure_preview_icon_button(
            self.preview_reset_view_button,
            "reset",
            "Reset image view",
        )

    def _build_preview_inspector(self):
        self.preview_layer_bar = QFrame()
        self.preview_layer_bar.setProperty("uiRole", "flatPanel")
        self.preview_layer_bar.setMinimumHeight(112)
        self.preview_layer_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_layer_bar_layout = QVBoxLayout(self.preview_layer_bar)
        self.preview_layer_bar_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        self.preview_layer_bar_layout.setSpacing(SPACING.sm)

        preview_layer_header = QHBoxLayout()
        preview_layer_header.setContentsMargins(0, 0, 0, 0)
        self.preview_layer_title = QLabel("Overlay layers")
        self.preview_layer_title.setProperty("muted", "true")
        self.preview_layer_title.setProperty("uiRole", "mutedLabel")
        self.preview_composite_checkbox = QCheckBox("Composite")
        self.preview_composite_checkbox.setChecked(False)
        self.preview_composite_checkbox.setToolTip(
            "Use Fiji-style additive color blending. Turn off for normal ordered layer stacking."
        )
        self.preview_composite_checkbox.toggled.connect(self.on_preview_composite_mode_changed)
        preview_layer_header.addWidget(self.preview_layer_title)
        preview_layer_header.addStretch(1)
        preview_layer_header.addWidget(self.preview_composite_checkbox)
        self.preview_layer_bar_layout.addLayout(preview_layer_header)

        self.preview_layer_list = PreviewLayerList()
        self.preview_layer_list.setObjectName("PreviewLayerList")
        self.preview_layer_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.preview_layer_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.preview_layer_list.setDropIndicatorShown(True)
        self.preview_layer_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.preview_layer_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview_layer_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_layer_list.model().rowsMoved.connect(self.preview_layer_list.refresh_category_headers)
        self.preview_layer_list.model().rowsMoved.connect(self.on_preview_layer_order_changed)
        self.preview_layer_list.currentRowChanged.connect(self.on_preview_layer_selection_changed)
        self.preview_layer_bar_layout.addWidget(self.preview_layer_list, 1)

        self.preview_layer_bar.setVisible(False)

        self.preview_info_panel = QFrame()
        self.preview_info_panel.setProperty("uiRole", "flatPanel")
        self.preview_info_panel.setMinimumHeight(112)
        self.preview_info_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_info_panel.setVisible(False)
        preview_info_layout = QVBoxLayout(self.preview_info_panel)
        preview_info_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        preview_info_layout.setSpacing(SPACING.sm)

        self.preview_info_title = QLabel("Metadata")
        self.preview_info_title.setProperty("muted", "true")
        self.preview_info_title.setProperty("uiRole", "mutedLabel")
        preview_info_layout.addWidget(self.preview_info_title)

        self.preview_info_label = QLabel(
            "No preview loaded yet. Preview a sample or open a TIFF/ND2 image to inspect layers and metadata."
        )
        self.preview_info_label.setWordWrap(True)
        self.preview_info_label.setProperty("muted", "true")
        self.preview_info_label.setProperty("uiRole", "mutedLabel")
        self.preview_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        preview_info_layout.addWidget(self.preview_info_label, 1)

        self.preview_inspector = QFrame()
        self.preview_inspector.setObjectName("PreviewInspector")
        self.preview_inspector.setMinimumWidth(WORKSPACE.inspector_min_width)
        self.preview_inspector.setMaximumWidth(WORKSPACE.inspector_max_width)
        self.preview_inspector.setVisible(False)
        preview_inspector_layout = QVBoxLayout(self.preview_inspector)
        preview_inspector_layout.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        preview_inspector_layout.setSpacing(SPACING.md)

        inspector_header = QHBoxLayout()
        inspector_header.setContentsMargins(0, 0, 0, 0)
        self.preview_inspector_title = QLabel("Layers and metadata")
        self.preview_inspector_title.setProperty("uiRole", "sectionTitle")
        inspector_header.addWidget(self.preview_inspector_title)
        inspector_header.addStretch(1)
        self.preview_work_busy_indicator = QWidget()
        self.preview_work_busy_indicator.setVisible(False)
        preview_work_busy_layout = QHBoxLayout(self.preview_work_busy_indicator)
        preview_work_busy_layout.setContentsMargins(0, 0, 0, 0)
        preview_work_busy_layout.setSpacing(SPACING.xs)
        self.preview_work_busy_icon = QLabel()
        self.preview_work_busy_icon.setFixedSize(20, 20)
        self.preview_work_busy_icon.setProperty("cellonautIconName", "hourglass")
        self._preview_work_busy_base_pixmap = self.shell_icon("hourglass", 16).pixmap(16, 16)
        self.preview_work_busy_icon.setPixmap(self._preview_work_busy_base_pixmap)
        self.preview_work_busy_text = QLabel("Loading…")
        self.preview_work_busy_text.setProperty("uiRole", "mutedLabel")
        preview_work_busy_layout.addWidget(self.preview_work_busy_icon)
        preview_work_busy_layout.addWidget(self.preview_work_busy_text)
        preview_inspector_layout.addLayout(inspector_header)

        self.preview_inspector_empty_label = QLabel("Load an image to inspect its layers and metadata.")
        self.preview_inspector_empty_label.setWordWrap(True)
        self.preview_inspector_empty_label.setProperty("uiRole", "mutedLabel")
        preview_inspector_layout.addWidget(self.preview_inspector_empty_label)
        preview_inspector_layout.addWidget(self.preview_layer_bar, 1)
        preview_inspector_layout.addWidget(self.preview_info_panel, 1)
        inspector_footer = QHBoxLayout()
        inspector_footer.setContentsMargins(0, 0, 0, 0)
        inspector_footer.addStretch(1)
        inspector_footer.addWidget(self.preview_metadata_button)
        preview_inspector_layout.addLayout(inspector_footer)

    def _assemble_preview_workspace(self):
        self.preview_workspace = QWidget()
        preview_workspace_layout = QHBoxLayout(self.preview_workspace)
        preview_workspace_layout.setContentsMargins(0, 0, 0, 0)
        preview_workspace_layout.setSpacing(8)

        self.preview_tool_strip = QFrame()
        self.preview_tool_strip.setObjectName("PreviewToolStrip")
        self.preview_tool_strip.setFixedWidth(WORKSPACE.preview_tool_strip_width)
        preview_tool_strip_layout = QVBoxLayout(self.preview_tool_strip)
        preview_tool_strip_layout.setContentsMargins(SPACING.xs + 2, SPACING.sm, SPACING.xs + 2, SPACING.sm)
        preview_tool_strip_layout.setSpacing(SPACING.md)

        def add_tool_group(title: str, *buttons: QPushButton) -> None:
            group = QWidget()
            group.setProperty("uiRole", "previewToolGroup")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(SPACING.xs)
            label = QLabel(title)
            label.setProperty("uiRole", "previewToolCategory")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            group_layout.addWidget(label)
            for button in buttons:
                group_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
            preview_tool_strip_layout.addWidget(group)

        add_tool_group("View", self.preview_reset_view_button, self.preview_fullscreen_button)
        add_tool_group("Zoom", self.preview_zoom_in_button, self.preview_zoom_out_button)
        add_tool_group("Snapshot", self.preview_snapshot_toggle_button, self.preview_snapshot_capture_button)
        add_tool_group("Export", self.preview_export_image_button)
        preview_tool_strip_layout.addStretch(1)

        self.preview_bottom_bar = QFrame()
        self.preview_bottom_bar.setObjectName("PreviewBottomToolbar")
        self.preview_bottom_bar.setFixedHeight(WORKSPACE.preview_bottom_bar_height)
        preview_bottom_layout = QHBoxLayout(self.preview_bottom_bar)
        preview_bottom_layout.setContentsMargins(SPACING.sm, SPACING.xs, SPACING.sm, SPACING.xs)
        preview_bottom_layout.setSpacing(SPACING.sm)
        preview_bottom_layout.addWidget(self.preview_prev_artifact_button)
        preview_bottom_layout.addWidget(self.preview_next_artifact_button)
        preview_bottom_layout.addWidget(self.preview_page_label)
        preview_bottom_layout.addWidget(self.preview_artifact_nav_label)
        preview_bottom_layout.addWidget(self.preview_work_busy_indicator)
        preview_bottom_layout.addWidget(self.preview_filter_status_label, 1)
        preview_bottom_layout.addWidget(self.preview_zoom_label)

        preview_workspace_layout.addWidget(self.preview_tool_strip)
        preview_workspace_layout.addWidget(self.preview_view, 1)
        preview_workspace_layout.addWidget(self.preview_inspector)

        self.preview_tab_layout.addWidget(self.preview_workspace_title)
        self.preview_tab_layout.addWidget(self.preview_toolbar)
        self.preview_tab_layout.addWidget(self.preview_file_title_label)
        self.preview_tab_layout.addWidget(self.preview_workspace, 1)
        self.preview_tab_layout.addWidget(self.preview_bottom_bar)

    def _build_workspace_support_tabs(self):
        self.log_tab = QWidget()
        self.log_tab_layout = QVBoxLayout(self.log_tab)
        self.log_tab_layout.setContentsMargins(8, 8, 8, 8)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setAcceptRichText(True)
        self.log_box.setUndoRedoEnabled(False)
        self.log_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_box.setFont(QFont("Consolas", 10))
        self.configure_log_box()
        self.log_tab_layout.addWidget(self.log_box)

        self.right_layout.addWidget(self.preview_tab, 1)

        self.help_tab = self.build_help_tab()
        self.about_tab = self.build_about_tab()
        self.configure_primary_navigation()

    # Delay cross-panel setup until every referenced widget exists.
    def finalize_ui(self):
        self.setup_file_browser()
        self.setup_path_drop_helpers()
        self.refresh_presets_list()
        self.update_dynamic_ui_states()
        if hasattr(self, "update_preview_artifact_nav_controls"):
            self.update_preview_artifact_nav_controls()
        self.set_status_style("Idle")

    # Reapply the standard spacing after dynamic channel and mask sections are
    # rebuilt so every part of the interface keeps the same compact layout.
    def apply_standard_layout_spacing(self):
        for group in self.findChildren(QGroupBox):
            layout = group.layout()
            if layout is not None:
                layout.setSpacing(4)
                layout.setContentsMargins(8, 8, 8, 8)

        for layout in [self.pipeline_tab_layout, self.settings_tab_layout]:
            layout.setSpacing(8)

    # Apply appearance changes at application scope so dialogs and future
    # windows use the same palette.
    def on_theme_changed(self, value: str):
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, value)
            self.refresh_shell_icons()
            self.refresh_log_theme()
            self.update_fiji_component_checklist()

    # Refresh only states that depend on several panels after loading settings
    # or completing initial construction.
    def update_dynamic_ui_states(self):
        self.apply_standard_layout_spacing()
        self.update_nd2_visibility()
