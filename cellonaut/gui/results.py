"""File-browser and preview helpers for generated Cellonaut outputs.

Generated CSV/text files open in the left Log view, while image-like artifacts
open in the persistent image workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.file_browser import open_folder_in_system_browser
from cellonaut.gui.run_comparison import RunComparisonDialog


# Files-tab actions and preview paging share one owner because
# both operate on artifacts selected outside the pipeline configuration panels.
class CellonautGuiResultsMixin(GuiMixin):
    """Open, compare, and navigate completed analysis results."""

    # Send every supported image through the shared preview dispatcher so scientific TIFF metadata is preserved.
    def open_image_file_in_preview(self, path: Path):
        if not path.exists():
            return
        self.preview_file(str(path))

    def open_image_in_preview_tab(self, path: Path):
        self.open_image_file_in_preview(path)

    # Tables use the text viewer because raw CSV is more useful here than a truncated grid preview.
    def open_table_in_log_tab(self, path: Path):
        self.open_text_in_log_tab(path)

    # Format JSON for readability while preserving other text files exactly as written.
    def open_text_in_log_tab(self, path: Path):
        try:
            suffix = path.suffix.lower()

            if suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = json.dumps(data, indent=2)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")

            self.log_box.setPlainText(text)
            self.show_left_page(self.log_tab)

        except Exception as exc:
            QMessageBox.warning(self, "Open text file", f"Could not open text file:\n{path.name}\n\nReason:\n{exc}")

    # Dispatch once by suffix so a single file-tree click cannot open competing right-panel views.
    def open_file_in_right_panel(self, path: str):
        path = str(path)

        if getattr(self, "_current_open_right_panel_path", None) == path:
            return

        self._current_open_right_panel_path = path

        p = Path(path)
        if not p.exists() or not p.is_file():
            return

        suffix = p.suffix.lower()

        image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".nd2"}
        table_suffixes = {".csv", ".tsv"}
        text_suffixes = {".txt", ".log", ".json", ".py", ".md", ".yaml", ".yml", ".ini", ".cfg"}

        if suffix in table_suffixes:
            self.open_table_in_log_tab(p)
            return

        if suffix in text_suffixes:
            self.open_text_in_log_tab(p)
            return

        if suffix in image_suffixes:
            self.open_image_in_preview_tab(p)
            return

        # Fallback: try preview first
        self.open_image_in_preview_tab(p)

    # Build controls without attaching filesystem signals; setup_file_browser owns those connections once.
    def build_files_content(self):
        self.files_left_tab = QWidget()
        self.files_left_tab_layout = QVBoxLayout(self.files_left_tab)
        self.files_left_tab_layout.setContentsMargins(8, 8, 8, 8)
        self.files_left_tab_layout.setSpacing(8)

        self.files_browser_toolbar = QWidget()
        self.files_browser_toolbar.setProperty("uiRole", "toolbar")
        files_browser_toolbar_layout = QVBoxLayout(self.files_browser_toolbar)
        files_browser_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        files_browser_toolbar_layout.setSpacing(8)

        browser_button_row = QHBoxLayout()
        browser_button_row.setSpacing(4)

        self.browser_back_button = QPushButton()
        self.configure_browser_button(
            self.browser_back_button,
            "arrow-left",
            "Back",
            icon_only=True,
        )
        self.browser_back_button.clicked.connect(self.browser_go_back)

        self.browser_forward_button = QPushButton()
        self.configure_browser_button(
            self.browser_forward_button,
            "arrow-right",
            "Forward",
            icon_only=True,
        )
        self.browser_forward_button.clicked.connect(self.browser_go_forward)

        self.browser_up_button = QPushButton()
        self.configure_browser_button(
            self.browser_up_button,
            "arrow-up",
            "Up one folder",
            icon_only=True,
        )
        self.browser_up_button.clicked.connect(self.browser_go_up)

        self.browser_input_button = QPushButton("Input")
        self.configure_browser_button(
            self.browser_input_button,
            "folder-open",
            "Open input folder",
        )
        self.browser_input_button.clicked.connect(self.browse_to_input_dir)

        self.browser_output_button = QPushButton("Output")
        self.configure_browser_button(
            self.browser_output_button,
            "hard-drive",
            "Open output folder",
        )
        self.browser_output_button.clicked.connect(self.browse_to_output_dir)

        self.browser_computer_button = QPushButton()
        self.configure_browser_button(
            self.browser_computer_button,
            "monitor",
            "Show computer locations",
        )
        self.browser_computer_button.setText("Home")
        self.browser_computer_button.clicked.connect(self.browser_go_to_computer)

        self.browser_compare_runs_button = QPushButton("Compare Runs...")
        self.configure_browser_button(
            self.browser_compare_runs_button,
            "columns",
            "Compare completed runs or previews",
        )
        self.browser_compare_runs_button.clicked.connect(self.open_run_comparison_dialog)

        self.browser_open_location_button = QPushButton("Open location")
        self.configure_browser_button(
            self.browser_open_location_button,
            "folder-open",
            "Open the current folder in the system file explorer",
        )
        self.browser_open_location_button.clicked.connect(self.open_browser_location)

        browser_button_row.addWidget(self.browser_back_button)
        browser_button_row.addWidget(self.browser_forward_button)
        browser_button_row.addWidget(self.browser_up_button)
        browser_button_row.addSpacing(6)
        browser_button_row.addWidget(self.browser_computer_button)
        browser_button_row.addWidget(self.browser_input_button)
        browser_button_row.addWidget(self.browser_output_button)
        browser_button_row.addWidget(self.browser_compare_runs_button)
        self.files_help_link = self.make_help_link("Files & Results")
        browser_button_row.addWidget(self.files_help_link)
        browser_button_row.addStretch(1)
        browser_button_row.addWidget(self.browser_open_location_button)

        self.browser_path_entry = QLineEdit()
        self.browser_path_entry.setAccessibleName("Folder path")
        self.browser_path_entry.setPlaceholderText("Paste a folder or image path")
        self.browser_path_entry.returnPressed.connect(self.browser_go_to_path)

        self.browser_go_button = QPushButton()
        self.configure_browser_button(
            self.browser_go_button,
            "arrow-right",
            "Go to path",
            icon_only=True,
        )
        self.browser_go_button.clicked.connect(self.browser_go_to_path)
        self.browser_path_entry.textChanged.connect(self.update_browser_navigation_buttons)

        self.browser_browse_button = QPushButton()
        self.configure_browser_button(
            self.browser_browse_button,
            "folder-open",
            "Browse for a folder or image",
            icon_only=True,
        )
        browse_menu = QMenu(self.browser_browse_button)
        browse_menu.addAction("Open image...", self.browse_for_preview_image)
        browse_menu.addAction("Open folder...", self.browse_for_folder)
        self.browser_browse_button.setMenu(browse_menu)

        browser_path_row = QHBoxLayout()
        browser_path_row.addWidget(self.browser_path_entry, 1)
        browser_path_row.addWidget(self.browser_go_button)
        browser_path_row.addWidget(self.browser_browse_button)

        self.file_tree = QTreeView()
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.setSortingEnabled(True)
        files_browser_toolbar_layout.addLayout(browser_button_row)
        files_browser_toolbar_layout.addLayout(browser_path_row)
        self.files_left_tab_layout.addWidget(self.files_browser_toolbar)
        self.files_left_tab_layout.addWidget(self.file_tree, 1)

    # All browser controls use the same Lucide family as the application shell.
    def configure_browser_button(
        self,
        button: QPushButton,
        icon_name: str,
        tooltip: str,
        *,
        icon_only: bool = False,
    ) -> None:
        button.setProperty("cellonautIconName", icon_name)
        button.setIcon(self.shell_icon(icon_name))
        button.setIconSize(QSize(18, 18))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if icon_only:
            button.setText("")
            button.setFixedSize(34, 30)

    def browse_for_preview_image(self) -> None:
        """Choose a supported image, show its folder, and open it in Image Preview."""
        start_path = str(getattr(self, "_browser_root_path", "") or Path.home())
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open image",
            start_path,
            "Images (*.tif *.tiff *.nd2 *.png *.jpg *.jpeg *.bmp *.gif *.webp);;All files (*)",
        )
        if path:
            self.browser_path_entry.setText(path)
            self.browser_go_to_path()

    def browse_for_folder(self) -> None:
        """Choose a folder and display it in the Files panel."""
        start_path = str(getattr(self, "_browser_root_path", "") or Path.home())
        path = QFileDialog.getExistingDirectory(self, "Open folder", start_path)
        if path:
            self.set_browser_root(path)

    def open_browser_location(self) -> None:
        """Open the displayed file or folder location in the system file explorer."""
        raw_path = self.browser_path_entry.text().strip()
        if not raw_path:
            raw_path = str(getattr(self, "_browser_root_path", "") or "").strip()
        if not raw_path:
            return

        path = Path(raw_path)
        if not path.exists():
            QMessageBox.information(self, "Open location", f"Path does not exist:\n{raw_path}")
            return
        try:
            open_folder_in_system_browser(path)
        except OSError as exc:
            QMessageBox.warning(self, "Open location", f"Could not open:\n{path}\n\nReason:\n{exc}")

    # Seed the comparison from the selected artifact or configured output folder
    # so sibling run folders are discovered without another navigation step.
    def open_run_comparison_dialog(self):
        selected = self.browser_path_entry.text().strip()
        if not selected:
            selected = str(getattr(self, "_browser_root_path", "") or "").strip()
        if not selected and hasattr(self, "output_dir"):
            selected = str(self.output_dir.get() or "").strip()

        dialog = RunComparisonDialog(self, start_path=selected)
        if hasattr(self, "center_dialog_on_window"):
            self.center_dialog_on_window(dialog)
        dialog.exec()

    # Navigate TIFF pages separately from result files.
    def update_preview_page_controls(self):
        total = len(self.preview_state.pages)

        if total <= 0:
            self.preview_page_label.setText("Page: - / -")
            self.preview_page_label.setVisible(False)
            return

        current = self.preview_state.page_index + 1
        self.preview_page_label.setText(f"Page: {current} / {total}")
        multi_page = total > 1
        self.preview_page_label.setVisible(multi_page)

    # Clear file-specific scene state together so controls cannot retain a page from the previous image.
    def clear_preview_pages(self):
        self.preview_state.pages = []
        self.preview_state.page_index = 0
        if hasattr(self, "clear_preview_layer_items"):
            self.clear_preview_layer_items()
        if hasattr(self, "set_preview_file_title"):
            self.set_preview_file_title(None)
        snapshot_item = getattr(self, "_preview_snapshot_item", None)
        if snapshot_item is not None:
            try:
                snapshot_item.setVisible(False)
            except RuntimeError:
                self._preview_snapshot_item = None
        if hasattr(self, "update_preview_snapshot_controls"):
            self.update_preview_snapshot_controls()
        self.update_preview_page_controls()

    # Filter invalid pixmaps at the boundary so page navigation never lands on an empty Qt image.
    def set_preview_pages(self, pixmaps: list[QPixmap], file_path: str | None = None):
        self.preview_state.pages = [pm for pm in pixmaps if pm is not None and not pm.isNull()]
        self.preview_state.page_index = 0
        self.preview_state.file_path = file_path
        if hasattr(self, "set_preview_file_title"):
            self.set_preview_file_title(file_path if self.preview_state.pages else None)
        self.update_preview_page_controls()

        if self.preview_state.pages:
            self.show_preview_page(0)
        else:
            self.preview_scene.clear()
            self.preview_pixmap_item = None

    # Try layered rendering first, then cached TIFF rendering, and finally the flat pixmap fallback.
    def show_preview_page(self, index: int, preserve_view: bool = False):
        if not self.preview_state.pages:
            self.preview_scene.clear()
            self.preview_pixmap_item = None
            self.update_preview_page_controls()
            return

        index = max(0, min(index, len(self.preview_state.pages) - 1))
        if index != self.preview_state.page_index and hasattr(self, "disable_preview_snapshot"):
            self.disable_preview_snapshot()
        self.preview_state.page_index = index
        if hasattr(self, "set_preview_layer_items_for_page"):
            try:
                if self.set_preview_layer_items_for_page(index, preserve_view=preserve_view):
                    return
            except Exception as exc:
                self.log(f"[PREVIEW][WARN] Layered page rendering failed: {type(exc).__name__}: {exc}")
        if hasattr(self, "refresh_tiff_preview_page"):
            try:
                self.refresh_tiff_preview_page(index, preserve_view=preserve_view)
            except Exception as exc:
                self.log(f"[PREVIEW][WARN] TIFF page rendering failed: {type(exc).__name__}: {exc}")

        pixmap = self.preview_state.pages[index]
        if hasattr(self, "set_preview_pixmap"):
            self.set_preview_pixmap(pixmap, preserve_view=preserve_view)
        else:
            self.preview_scene.clear()
            self.preview_pixmap_item = self.preview_scene.addPixmap(pixmap)
            self.preview_view.reset_zoom()
            self.preview_view.fit_image()
        self.update_preview_page_controls()

    # Fit is an explicit action because automatic fitting after every update would erase the user's inspection zoom.
    def fit_preview_image(self):
        if hasattr(self, "preview_view"):
            self.preview_view.fit_image()

    # Summarize only categories that produced files so optional exports do not
    # fill the preview panel with zero-count noise.
    def update_preview_artifact_info(self, preview_result: dict):
        sample_label = preview_result.get("sample_label", "(unknown sample)")
        artifacts = preview_result.get("artifacts", {}) or {}

        parts = [f"Preview sample: {sample_label}"]

        label_map = {
            "combined_overlays": "Overlays",
            "qc_overlay_pngs": "Overlay PNG",
            "processing_montages": "Processing montages",
            "mask_overlays": "Mask overlays",
            "final_binary_masks": "Final binary masks",
            "weka_probability_maps": "Weka probability maps",
            "weka_threshold_masks": "Weka threshold masks",
            "cell_outline_stacks": "Cell outline stacks",
            "cell_qc_overlay_pngs": "Cell overlay PNG",
            "cell_label_images": "Cell label images",
            "cell_outline_images": "Cell outline images",
        }

        for key in [
            "combined_overlays",
            "qc_overlay_pngs",
            "processing_montages",
            "mask_overlays",
            "final_binary_masks",
            "weka_probability_maps",
            "weka_threshold_masks",
            "cell_outline_stacks",
            "cell_qc_overlay_pngs",
            "cell_label_images",
            "cell_outline_images",
        ]:
            values = artifacts.get(key, []) or []
            if values:
                parts.append(f"{label_map.get(key, key)}: {len(values)}")

        self.preview_info_label.setText("\n".join(parts))
