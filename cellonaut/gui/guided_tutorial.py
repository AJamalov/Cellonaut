"""Interactive walkthrough using the bundled logo example and reference results."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from PySide6.QtCore import QPoint, QRect, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from cellonaut.config.defaults import default_measurement_options
from cellonaut.gui.mixin import GuiMixin
from cellonaut.resources import get_resource_path


@dataclass(frozen=True)
class TutorialStep:
    """Describe a target; actions normally pause once unless single_click advances immediately."""

    title: str
    body: str
    target: str
    section: int | None
    action: str = ""
    single_click: bool = False


class TutorialHighlight(QWidget):
    """Paint one transparent rounded outline around a tutorial target."""

    OUTSET = 4

    def __init__(self, parent: QWidget, radius: int):
        super().__init__(parent)
        self.radius = radius
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#FFD43B"), 3))
        outline = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        painter.drawRoundedRect(outline, self.radius, self.radius)


TUTORIAL_STEPS = (
    TutorialStep("Input folder", "Select the input folder containing the images to analyze.", "input_dir", 0, "input"),
    TutorialStep("Output folder", "Select where to save results. Keep the output folder outside the input folder so results are not detected as new samples.", "output_dir", 0, "output"),
    TutorialStep("Image channel", "Channels describe source images in the dataset. You can rename them here.", "image_tabs", 0, "channel"),
    TutorialStep("Create a mask", "Masks identify regions to measure. Click 'Add mask' to create a mask from the selected channel.", "mask_tabs", 1, "mask"),
    TutorialStep("Weka classifier", "Select a .model classifier trained with Trainable Weka Segmentation in Fiji.", "mask_classifier", 1, "classifier"),
    TutorialStep("Weka classes", "Class numbers follow the order saved in the Fiji model, starting at 1. Here, 1,2,3 selects three classes; each produces a separate mask.", "mask_classes", 1, "classes"),
    TutorialStep("Probability threshold", "Weka produces probability maps. This control uses a Fiji/ImageJ threshold method to turn them into masks. Preview a sample and check that the masks include the intended structures without including background.", "mask_threshold", 1),
    TutorialStep("Enable Cellpose", "Cellpose can create individual cell outlines for per-cell analysis. You must enable it to modify the settings.", "cellpose_enable", 2, "cellpose_on"),
    TutorialStep("Cellpose source and model", "Choose the image channel Cellpose should analyze and its model here.", "cellpose_settings_table", 2),
    TutorialStep("Cellpose diameter", "Diameter rescales the image using a cell size in pixels. Blank keeps the original scale. The other columns control border removal, minimum area, probability and flow thresholds.", "cellpose_diameter", 2, "cellpose_diameter"),
    TutorialStep("Disable Cellpose", "This example uses Weka masks only, so disable Cellpose before continuing.", "cellpose_enable", 2, "cellpose_off"),
    TutorialStep("Before mask creation", "Apply the same processing used on your Weka training images. Drag steps to change their order; they run from top to bottom.", "image_processing_table", 3, "processing_add_before_mask"),
    TutorialStep("After mask creation", "If you need to refine the masks after creation, you can add processing steps here.", "mask_processing_table", 3, "processing_add_after_mask"),
    TutorialStep("Before measurement", "Apply processing to the image used for measurement here. Subtract Background stays last: each radius creates a separate corrected result after the other steps. The radii are not applied to one another.", "measurement_processing_table", 3, "processing_add_before_measurement"),
    TutorialStep("Disable processing examples", "This example needs no image or mask processing, so disable the example steps.", "processing_group", 3, "processing_disable_examples"),
    TutorialStep("Measurement region", "Turn ON the channel/mask pairs to measure. Rows are image channels and columns are masks.", "analysis_matrix_table", 4, "relationship"),
    TutorialStep("Measurement choices", "Choose the measurements to include. Result tables and review overlays are saved automatically.", "analysis_measurements_panel", 4, "measurements"),
    TutorialStep("Image Preview Tools", "Image Preview Tools can adjust a displayed mask temporarily and create cell groups after using Cellpose. They do not alter the saved pipeline and are not needed here.", "preview_tools_tabs", 5),
    TutorialStep("Check Setup", "Check Setup checks folders, samples, Fiji, Weka, and measurement relationships.", "check_setup_button", None, "check", True),
    TutorialStep("Preview one sample", "Preview One Sample processes the first sample in your input folder.", "preview_pipeline_button", None, "preview", True),
    TutorialStep("Run Pipeline", "Run Pipeline processes the full dataset in your input folder.", "run_button", None, "run", True),
    TutorialStep("Files toolbar", "Use the Files toolbar to navigate folders. Compare Runs compares settings and summaries from previous runs or previews.", "files_browser_toolbar", None),
    TutorialStep("Open the output folder", "The Output button opens your output location in the file area.", "browser_output_button", None, "show_results"),
    TutorialStep("Open the reference preview", "The preview_1 folder appears below Output. Opening it expands the tree so you can still see the complete location.", "file_tree", None, "open_reference_preview"),
    TutorialStep("Open the Results folder", "Results contains folders for measurement tables, masks, probability maps, and review overlays.", "file_tree", None, "open_results"),
    TutorialStep("Result folders", "CSV Data contains measurement tables.\nMasks contains probability maps and masks.\nOverlays contains images with channels and masks overlaid for review.", "file_tree", None),
    TutorialStep("Open a Weka result", "Clicking on an image opens it in Image Preview.", "file_tree", None, "open_result"),
    TutorialStep("Open the TIFF overlay", "The TIFF overlay combines the original image and the Weka masks as separate layers.", "file_tree", None, "open_overlay"),
    TutorialStep("Image Preview toolbar", "The result menu lets you switch between overlays, masks, previews, montages, and snapshots. The arrow controls move between samples, while Recent images, Layers, and Help provide quick access to inspection features.", "preview_toolbar", None),
    TutorialStep("Image tools", "Here you can find tools for zooming, saving snapshots, resetting the view, and fitting the complete image in the available space.", "preview_tool_strip", None),
    TutorialStep("Layers", "Layers opens the panel for showing or hiding layers and changing their opacity and colors. The button at the bottom right shows image metadata.", "preview_inspector_toggle_button", None, "show_layers", True),
    TutorialStep("Change layer colors", "Each layer has a color swatch that you can click on to choose a color.", "tutorial_layer_color", None, "change_layer_color"),
    TutorialStep("Layers and metadata", "Inspect image and mask layers separately or as a composite. Use the drag handles to reorder layers within their group.", "preview_inspector", None),
    TutorialStep("Snapshot selection", "Snapshot places a square selection on the preview which you can drag over the area you want to save.", "preview_snapshot_toggle_button", None, "show_snapshot"),
    TutorialStep("Capture snapshot", "Save the selected view as a PNG, or choose a layer montage for the same area across layers. Normal snapshots go to Results/Image Preview Tools/Snapshots.", "preview_snapshot_capture_button", None, "capture_snapshot"),
    TutorialStep("Check the snapshot", "Here is the captured snapshot.", "preview_view", None, "inspect_snapshot"),
    TutorialStep("Tutorial complete", "You have completed the tutorial. Click Finish to restore your setup and leave.", "", None),
)


class GuidedTutorialMixin(GuiMixin):
    """Guide the main window while preserving the user's prior pipeline."""

    def initialize_guided_tutorial(self) -> None:
        self._tutorial_settings = QSettings("Cellonaut", "Cellonaut")
        self._tutorial_dialog = None
        self._tutorial_step = 0
        self._tutorial_phase = 0
        self._tutorial_ending = False
        self._tutorial_snapshot = None
        self._tutorial_output_dir = None
        self._tutorial_preview_state = "idle"
        self._tutorial_run_state = "idle"
        self._tutorial_check_state = "idle"
        self._tutorial_highlighted_widget = None
        self._tutorial_highlight_frame = None
        self._tutorial_visual_revision = 0
        self._tutorial_geometry_timer = None
        self._tutorial_previous_browser_root = None
        self._tutorial_previous_browser_history = []
        self._tutorial_previous_browser_forward_history = []
        self._tutorial_previous_preview_path = None
        self._tutorial_previous_recent_images = []
        self._tutorial_previous_inspector_checked = False
        self._tutorial_temporary_files = None
        self._tutorial_saved_snapshot = None
        self._tutorial_overlay_path = None
        self._tutorial_original_layer_colors = {}
        self._tutorial_action_definitions = {}
        self._update_tutorial_button()

    def _tutorial_assets(self) -> tuple[Path, Path, Path]:
        base = Path("cellonaut/data/assets")
        return (
            get_resource_path(base / "Cellonaut_32bit_input"),
            get_resource_path(base / "Cellonaut_32bit_classifier"),
            get_resource_path(base / "Cellonaut_32bit_output"),
        )

    def _prepare_tutorial_workspace(self) -> None:
        # The presentation reads the packaged reference run directly and never writes analysis output.
        input_dir, classifier_dir, output_dir = self._tutorial_assets()
        model = classifier_dir / "Cellonaut_classifier.model"
        reference_dir = output_dir / "preview_1"
        results_dir = reference_dir / "Results"
        preview_result = (
            results_dir
            / "Masks"
            / "ProbabilityMaps"
            / "Cellonaut_32bits_Cellonaut mask_prob_Cellonaut_classifier.tif"
        )
        overlay = results_dir / "Overlays" / "TIFF Overlays" / "Cellonaut_32bits_Cellonaut_combined_overlay.tif"
        required = (
            input_dir / "Cellonaut_32bits.tif",
            model,
            classifier_dir / "Cellonaut_classifier_data.arff",
            results_dir / "CSV Data" / "Measurements.csv",
            preview_result,
            overlay,
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Tutorial asset(s) missing: {', '.join(missing)}")

        self._tutorial_input_dir = input_dir
        self._tutorial_output_dir = output_dir
        self._tutorial_model = model
        self._tutorial_reference_dir = reference_dir
        self._tutorial_results_dir = results_dir
        self._tutorial_preview_result = preview_result
        self._tutorial_overlay_path = overlay

    def _show_tutorial_section(self, index: int) -> None:
        self.show_left_page(self.pipeline_tab)
        self.show_pipeline_section(index)

    def _tutorial_target(self, name: str) -> QWidget | None:
        if name == "cellpose_enable":
            table = getattr(self, "cellpose_settings_table", None)
            cell = table.cellWidget(0, 0) if table is not None else None
            return cell.findChild(QPushButton) if cell is not None else None
        if name == "cellpose_diameter":
            table = getattr(self, "cellpose_settings_table", None)
            return table.cellWidget(0, 5) if table is not None else None
        if name.endswith("_processing_enable"):
            table = getattr(self, name.removesuffix("_enable") + "_table", None)
            cell = table.cellWidget(0, 1) if table is not None else None
            return cell.findChild(QPushButton) if cell is not None else None
        if name == "tutorial_layer_color":
            buttons = getattr(self, "_preview_color_buttons", []) or []
            index = self._tutorial_mask_layer_indices().get(1)
            return buttons[index] if index is not None and index < len(buttons) else None
        if name in {"mask_classifier", "mask_classes", "mask_threshold"}:
            rows = getattr(self, "image_rows", []) or []
            row = next((row for row in reversed(rows) if row is not None and row.mask_tab is not None), None)
            if row is None:
                fallback = getattr(self, "mask_tabs", None)
                return fallback if isinstance(fallback, QWidget) else None
            targets = {
                "mask_classifier": row.classifier.edit if row.classifier is not None else None,
                "mask_classes": row.probability_class_index.edit if row.probability_class_index is not None else None,
                "mask_threshold": getattr(row.threshold_method_host, "combo", row.threshold_method_host),
            }
            return targets.get(name) or row.mask_tab
        target = getattr(self, name, None)
        return target if isinstance(target, QWidget) else None

    def _tutorial_mask_layer_indices(self) -> dict[int, int]:
        labels = self.preview_state.current_labels or []
        indices = {}
        for index, label in enumerate(labels):
            match = re.search(r"_class\s*([123])$", str(label), re.IGNORECASE)
            if match and "cellonaut" in str(label).casefold():
                indices[int(match.group(1))] = index
        return indices

    def _tutorial_expand_folder(self, folder: Path) -> None:
        index = self.fs_model.index(str(folder))
        if not index.isValid():
            raise FileNotFoundError(f"Folder is not available in Files: {folder}")
        self.file_tree.expand(index)
        self.file_tree.scrollTo(index)

    def _tutorial_select_image(self, image: Path) -> None:
        for folder in reversed(image.parents):
            if folder == self._tutorial_output_dir or self._tutorial_output_dir in folder.parents:
                self._tutorial_expand_folder(folder)
        index = self.fs_model.index(str(image))
        if not index.isValid():
            raise FileNotFoundError(f"Image is not available in Files: {image}")
        self.file_tree.setCurrentIndex(index)
        self.file_tree.scrollTo(index)
        if str(self.preview_state.file_path or "") != str(image):
            self.on_file_tree_current_changed(index, index)

    def _clear_tutorial_highlight(self) -> None:
        self._tutorial_visual_revision += 1
        frame = self._tutorial_highlight_frame
        self._tutorial_highlight_frame = None
        if frame is not None:
            try:
                frame.hide()
                frame.deleteLater()
            except RuntimeError:
                pass
        self._tutorial_highlighted_widget = None

    def _highlight_tutorial_target(self, target: QWidget | None) -> None:
        self._clear_tutorial_highlight()
        if target is None:
            return
        self._tutorial_highlighted_widget = target
        target_rect = self._tutorial_visible_global_rect(target)
        outside_rect = target_rect.adjusted(-TutorialHighlight.OUTSET, -TutorialHighlight.OUTSET, TutorialHighlight.OUTSET, TutorialHighlight.OUTSET)
        host = target.parentWidget()
        while host is not None and host.parentWidget() is not None:
            host_rect = QRect(host.mapToGlobal(QPoint(0, 0)), host.size())
            if host_rect.contains(outside_rect):
                break
            host = host.parentWidget()
        if host is None:
            return
        self._tutorial_highlight_frame = TutorialHighlight(
            host,
            5 if isinstance(target, QGroupBox) else 4,
        )
        self._layout_tutorial_highlight(target)
        self._tutorial_highlight_frame.show()
        self._tutorial_highlight_frame.raise_()

    @staticmethod
    def _tutorial_visible_global_rect(target: QWidget) -> QRect:
        rect = QRect(target.mapToGlobal(QPoint(0, 0)), target.size())
        ancestor = target.parentWidget()
        while ancestor is not None:
            ancestor_rect = QRect(ancestor.mapToGlobal(QPoint(0, 0)), ancestor.size())
            rect = rect.intersected(ancestor_rect)
            ancestor = ancestor.parentWidget()
        return rect

    def _layout_tutorial_highlight(self, target: QWidget) -> None:
        frame = self._tutorial_highlight_frame
        if frame is None:
            return
        host = frame.parentWidget()
        if host is None:
            return
        target_rect = self._tutorial_visible_global_rect(target)
        outside_rect = target_rect.adjusted(-TutorialHighlight.OUTSET, -TutorialHighlight.OUTSET, TutorialHighlight.OUTSET, TutorialHighlight.OUTSET)
        outer = QRect(host.mapFromGlobal(outside_rect.topLeft()), outside_rect.size())
        if outer.isEmpty():
            frame.hide()
            return
        frame.setGeometry(outer)

    def _position_tutorial_dialog(self, target: QWidget | None) -> None:
        dialog = self._tutorial_dialog
        if dialog is None or target is None or not target.isVisible():
            return
        dialog.adjustSize()
        target_rect = self._tutorial_visible_global_rect(target)
        if target_rect.isEmpty():
            return
        top_left = target_rect.topLeft()
        bottom_right = target_rect.bottomRight()
        gap = 14
        screen = QGuiApplication.screenAt(top_left) or self.screen()
        available = screen.availableGeometry()
        candidates = (
            QPoint(bottom_right.x() + gap, top_left.y()),
            QPoint(top_left.x() - dialog.width() - gap, top_left.y()),
            QPoint(top_left.x(), bottom_right.y() + gap),
            QPoint(top_left.x(), top_left.y() - dialog.height() - gap),
        )
        for point in candidates:
            far = QPoint(point.x() + dialog.width(), point.y() + dialog.height())
            if available.contains(point) and available.contains(far):
                dialog.move(point)
                return
        dialog.move(max(available.left(), min(top_left.x(), available.right() - dialog.width())), max(available.top(), min(bottom_right.y() + gap, available.bottom() - dialog.height())))

    def _place_tutorial_cursor_on_next(self) -> None:
        dialog = self._tutorial_dialog
        button = getattr(self, "_tutorial_next_button", None)
        if dialog is not None and dialog.isVisible() and button is not None:
            QCursor.setPos(button.mapToGlobal(button.rect().center()))

    def _make_tutorial_channel(self) -> dict:
        channel = self.create_default_image_definition(0)
        channel.update({"name": "Cellonaut", "folder": "Cellonaut", "mask_slot_enabled": False})
        return channel

    def _apply_tutorial_step(self, step_index: int) -> bool:
        action = TUTORIAL_STEPS[step_index].action
        if action.startswith("cellpose_") or action.startswith("processing_"):
            self._tutorial_action_definitions[step_index] = deepcopy(self.get_active_image_definitions())
        if action == "input":
            self.input_dir.set(str(self._tutorial_input_dir))
            self.reuse_existing_masks_checkbox.setChecked(False)
        elif action == "output":
            self.output_dir.set(str(self._tutorial_output_dir))
        elif action == "channel":
            self.image_definitions = self.normalize_image_definitions([self._make_tutorial_channel()])
            self.rebuild_image_rows(sync_from_ui=False)
        elif action == "mask":
            mask = self.create_default_image_definition(1)
            mask.update({"name": "Cellonaut mask", "folder": "Cellonaut", "is_mask_only": True, "mask_slot_enabled": True, "mask_source_channel": "Cellonaut", "classifier": "", "probability_class_index": "1"})
            self.image_definitions = self.normalize_image_definitions([self._make_tutorial_channel(), mask])
            self.rebuild_image_rows(sync_from_ui=False)
        elif action in {"classifier", "classes"}:
            row = next(
                row for row in reversed(self.image_rows)
                if row is not None and row.mask_tab is not None
            )
            editor = row.classifier if action == "classifier" else row.probability_class_index
            if editor is not None:
                editor.set(str(self._tutorial_model) if action == "classifier" else "1,2,3")
        elif action in {"cellpose_on", "cellpose_off"}:
            button = self._tutorial_target("cellpose_enable")
            if not isinstance(button, QPushButton):
                raise ValueError("Cellpose enable control is unavailable.")
            button.setChecked(action == "cellpose_on")
        elif action == "cellpose_diameter":
            editor = self._tutorial_target("cellpose_diameter")
            if not isinstance(editor, QLineEdit):
                raise ValueError("Cellpose diameter control is unavailable.")
            editor.setText("30")
        elif action.startswith("processing_add_"):
            examples = {
                "processing_add_before_mask": ("add_image_processing_step", "gaussian_blur", "image_processing_steps", "gaussian_sigma", "1"),
                "processing_add_after_mask": ("add_mask_processing_step", "binary_fill_holes", "mask_processing_steps", "", ""),
                "processing_add_before_measurement": ("add_measurement_processing_step", "rolling_ball_background", "image_processing_steps", "bg_radii", "40"),
            }
            method, step_type, storage_key, param_key, value = examples[action]
            getattr(self, method)(step_type)
            if param_key:
                definitions = self.get_active_image_definitions()
                for definition in definitions:
                    for recipe_step in definition.get(storage_key, []):
                        if recipe_step.get("type") == step_type:
                            recipe_step.setdefault("params", {})[param_key] = value
                            definition[param_key] = value
                self.image_definitions = self.normalize_image_definitions(definitions)
                self.rebuild_mask_processing_table()
        elif action == "processing_disable_examples":
            for target in (
                "image_processing_enable",
                "mask_processing_enable",
                "measurement_processing_enable",
            ):
                button = self._tutorial_target(target)
                if not isinstance(button, QPushButton):
                    raise ValueError("Processing step enable control is unavailable.")
                button.setChecked(False)
        elif action == "relationship":
            definitions = self.get_active_image_definitions()
            definitions[0]["mask_relationships"] = {"Cellonaut mask": True}
            self.image_definitions = self.normalize_image_definitions(definitions)
            self.rebuild_image_rows(sync_from_ui=False)
        elif action == "measurements":
            options = {key: False for key in default_measurement_options()}
            options.update({"area": True, "mean": True, "raw_intden": True})
            self.measurement_options = options
            self._populate_inline_measurement_settings()
            self.update_analysis_matrix_warning_label()
        elif action == "check":
            self._tutorial_check_state = "simulated"
        elif action == "preview":
            self._tutorial_preview_state = "complete"
        elif action == "run":
            if self._tutorial_preview_state != "complete":
                return False
            self._tutorial_run_state = "complete"
        elif action == "show_results":
            self.show_left_page(self.files_left_tab)
            self.set_browser_root(str(self._tutorial_output_dir))
        elif action == "open_reference_preview":
            self._tutorial_expand_folder(self._tutorial_reference_dir)
        elif action == "open_results":
            self._tutorial_expand_folder(self._tutorial_results_dir)
        elif action == "open_result":
            if self._tutorial_preview_result is None:
                return False
            self._tutorial_select_image(self._tutorial_preview_result)
        elif action == "open_overlay":
            if self._tutorial_overlay_path is None:
                return False
            self._tutorial_select_image(self._tutorial_overlay_path)
        elif action == "show_layers":
            layers_button = self.preview_inspector_toggle_button
            layers_button.blockSignals(True)
            layers_button.setChecked(True)
            layers_button.blockSignals(False)
            self.preview_inspector.setVisible(True)
            self.refresh_preview_inspector_content()
        elif action == "change_layer_color":
            colors = self.preview_state.overlay_colors or []
            indices = self._tutorial_mask_layer_indices()
            if set(indices) != {1, 2, 3} or any(index >= len(colors) for index in indices.values()):
                raise ValueError("The reference overlay is missing one or more Weka class layers.")
            self._tutorial_original_layer_colors = {index: colors[index] for index in indices.values()}
            for class_number, color in ((1, "#FFFFFF"), (2, "#B0B0B0"), (3, "#8000FF")):
                self.set_preview_channel_color(indices[class_number], color)
        elif action == "show_snapshot":
            self.preview_snapshot_toggle_button.setChecked(True)
        elif action == "capture_snapshot":
            if self._tutorial_temporary_files is None:
                self._tutorial_temporary_files = TemporaryDirectory(prefix="cellonaut-tutorial-")
            snapshot = Path(self._tutorial_temporary_files.name) / "Cellonaut_overlay_snapshot.png"
            saved = self.save_visible_snapshot(snapshot)
            if saved is None or not snapshot.is_file():
                raise ValueError("Could not capture the tutorial snapshot.")
            self._tutorial_saved_snapshot = snapshot
        elif action == "inspect_snapshot":
            if not self._tutorial_saved_snapshot or not self.preview_file(str(self._tutorial_saved_snapshot)):
                raise ValueError("Could not open the tutorial snapshot.")
        return True

    def _undo_tutorial_step(self, step_index: int) -> None:
        action = TUTORIAL_STEPS[step_index].action
        snapshot = self._tutorial_snapshot or {}
        if action.startswith("cellpose_") or action.startswith("processing_"):
            prior = self._tutorial_action_definitions.pop(step_index, None)
            if prior is not None:
                self.image_definitions = self.normalize_image_definitions(prior)
                self.rebuild_image_rows(sync_from_ui=False)
        elif action == "input":
            self.input_dir.set(str(snapshot.get("input_dir", "") or ""))
        elif action == "output":
            self.output_dir.set(str(snapshot.get("output_dir", "") or ""))
        elif action == "channel":
            self.image_definitions = self.normalize_image_definitions(
                deepcopy(snapshot.get("image_definitions", []))
            )
            self.rebuild_image_rows(sync_from_ui=False)
        elif action == "mask":
            self.image_definitions = self.normalize_image_definitions([self._make_tutorial_channel()])
            self.rebuild_image_rows(sync_from_ui=False)
        elif action in {"classifier", "classes"}:
            row = next(
                row for row in reversed(self.image_rows)
                if row is not None and row.mask_tab is not None
            )
            editor = row.classifier if action == "classifier" else row.probability_class_index
            if editor is not None:
                editor.set("" if action == "classifier" else "1")
        elif action == "relationship":
            definitions = self.get_active_image_definitions()
            definitions[0]["mask_relationships"] = {"Cellonaut mask": False}
            self.image_definitions = self.normalize_image_definitions(definitions)
            self.rebuild_image_rows(sync_from_ui=False)
        elif action == "measurements":
            saved = snapshot.get("measurement_options", {})
            self.measurement_options = dict(saved) if isinstance(saved, dict) else default_measurement_options()
            self._populate_inline_measurement_settings()
            self.update_analysis_matrix_warning_label()
        elif action == "check":
            self._tutorial_check_state = "idle"
        elif action == "preview":
            self._tutorial_preview_state = "idle"
        elif action == "run":
            self._tutorial_run_state = "idle"
        elif action == "show_results":
            self._show_browser_computer_view()
        elif action == "open_reference_preview":
            self.file_tree.collapse(self.fs_model.index(str(self._tutorial_reference_dir)))
        elif action == "open_results":
            self.file_tree.collapse(self.fs_model.index(str(self._tutorial_results_dir)))
        elif action == "open_result":
            self.reset_preview_display(clear_title=True)
        elif action == "open_overlay":
            self._tutorial_select_image(self._tutorial_preview_result)
        elif action == "show_layers":
            layers_button = self.preview_inspector_toggle_button
            layers_button.blockSignals(True)
            layers_button.setChecked(False)
            layers_button.blockSignals(False)
            self.preview_inspector.setVisible(False)
            self.refresh_preview_inspector_content()
        elif action == "change_layer_color":
            for index, color in self._tutorial_original_layer_colors.items():
                self.set_preview_channel_color(index, color)
            self._tutorial_original_layer_colors = {}
        elif action == "show_snapshot":
            self.preview_snapshot_toggle_button.setChecked(False)
        elif action == "capture_snapshot":
            self._tutorial_saved_snapshot = None
        elif action == "inspect_snapshot":
            self.preview_file(str(self._tutorial_overlay_path))

    def _update_tutorial_button(self) -> None:
        button = self.guided_tutorial_button
        active = self._tutorial_dialog is not None
        seen = self._tutorial_settings.value("guided_tutorial_seen", False, type=bool)
        label = "Skip tutorial" if active else "Tutorial"
        button.setText(label)
        button.setAccessibleName(label)
        button.setToolTip("Leave the tutorial" if active else "Run the guided logo workflow")
        button.setProperty("attention", not active and not seen)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def toggle_guided_tutorial(self) -> None:
        if self._tutorial_dialog is not None:
            self._end_guided_tutorial()
            return
        if self.has_active_processing_task():
            QMessageBox.information(self, "Tutorial", "Wait for the current task to finish before starting the tutorial.")
            return
        answer = QMessageBox.question(self, "Tutorial", "Start the tutorial? It temporarily replaces your setup and restores it when you finish or skip.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.commit_gui_edits()
            self._tutorial_snapshot = deepcopy(self.get_preset_dict())
            self._tutorial_previous_browser_root = getattr(self, "_browser_root_path", None)
            self._tutorial_previous_browser_history = list(
                getattr(self, "_browser_history", [])
            )
            self._tutorial_previous_browser_forward_history = list(
                getattr(self, "_browser_forward_history", [])
            )
            self._tutorial_previous_preview_path = str(
                self.preview_state.file_path or ""
            )
            self._tutorial_previous_recent_images = list(
                getattr(self, "_recent_preview_images", [])
            )
            self._tutorial_previous_inspector_checked = bool(
                getattr(self, "preview_inspector_toggle_button", None)
                and self.preview_inspector_toggle_button.isChecked()
            )
            self._prepare_tutorial_workspace()
        except Exception as exc:
            self._tutorial_snapshot = None
            QMessageBox.critical(self, "Tutorial", f"Could not start the example:\n{exc}")
            return
        self._suppress_input_path_scan = True
        self._tutorial_step = 0
        self._tutorial_phase = 0
        dialog = QDialog(self)
        dialog.setObjectName("GuidedTutorialDialog")
        dialog.setWindowTitle("Tutorial")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setModal(True)
        dialog.setMinimumWidth(420)
        dialog.setMaximumWidth(520)
        layout = QVBoxLayout(dialog)
        self._tutorial_heading = QLabel()
        self._tutorial_heading.setProperty("uiRole", "sectionTitle")
        self._tutorial_body = QLabel()
        self._tutorial_body.setWordWrap(True)
        layout.addWidget(self._tutorial_heading)
        layout.addWidget(self._tutorial_body)
        buttons = QHBoxLayout()
        self._tutorial_previous_button = QPushButton("Previous")
        buttons.addWidget(self._tutorial_previous_button)
        buttons.addStretch(1)
        self._tutorial_skip_button = QPushButton("Skip tutorial")
        self._tutorial_next_button = QPushButton("Next")
        self._tutorial_next_button.setDefault(True)
        buttons.addWidget(self._tutorial_skip_button)
        buttons.addWidget(self._tutorial_next_button)
        layout.addLayout(buttons)
        self._tutorial_previous_button.clicked.connect(self._previous_guided_tutorial_step)
        self._tutorial_skip_button.clicked.connect(self._end_guided_tutorial)
        self._tutorial_next_button.clicked.connect(self._advance_guided_tutorial)
        dialog.finished.connect(self._on_tutorial_dialog_closed)
        self._tutorial_dialog = dialog
        self._tutorial_geometry_timer = QTimer(dialog)
        self._tutorial_geometry_timer.setInterval(40)
        self._tutorial_geometry_timer.timeout.connect(self._refresh_tutorial_highlight_geometry)
        self._tutorial_geometry_timer.start()
        self._prepare_tutorial_step()
        self._update_tutorial_button()
        dialog.show()
        dialog.raise_()

    def _prepare_tutorial_step(self) -> None:
        dialog = self._tutorial_dialog
        if dialog is None:
            return
        step = TUTORIAL_STEPS[self._tutorial_step]
        if step.title == "Tutorial complete":
            self._clear_tutorial_highlight()
            self._tutorial_heading.setText("Tutorial complete")
            self._tutorial_body.setText(step.body)
            self._tutorial_next_button.setText("Finish")
            self._tutorial_previous_button.setEnabled(True)
            dialog.adjustSize()
            center = self.frameGeometry().center()
            dialog.move(
                center.x() - dialog.width() // 2,
                center.y() - dialog.height() // 2,
            )
            frame_center = dialog.frameGeometry().center()
            dialog.move(dialog.pos() + center - frame_center)
            self._place_tutorial_cursor_on_next()
            return
        if step.section is not None:
            self._show_tutorial_section(step.section)
        elif step.target in {"files_browser_toolbar", "browser_output_button", "file_tree"}:
            self.show_left_page(self.files_left_tab)
        self._tutorial_heading.setText(f"{self._tutorial_step + 1} of {len(TUTORIAL_STEPS)} — {step.title}")
        self._tutorial_body.setText(step.body)
        finishing = self._tutorial_step == len(TUTORIAL_STEPS) - 1 and (
            not step.action or self._tutorial_phase == 1
        )
        self._tutorial_next_button.setText("Finish" if finishing else "Next")
        self._tutorial_previous_button.setEnabled(self._tutorial_step > 0 or self._tutorial_phase > 0)
        target = self._tutorial_target(step.target)
        if target is not None and step.section is not None:
            self.pipeline_scroll.ensureWidgetVisible(target, 24, 24)
        self._highlight_tutorial_target(target)
        revision = self._tutorial_visual_revision
        def settle_layout() -> None:
            if revision != self._tutorial_visual_revision:
                return
            if target is not None and target is self._tutorial_highlighted_widget:
                try:
                    self._layout_tutorial_highlight(target)
                    self._position_tutorial_dialog(target)
                except RuntimeError:
                    return
            self._place_tutorial_cursor_on_next()

        QTimer.singleShot(0, settle_layout)

    def _refresh_tutorial_highlight_geometry(self) -> None:
        if self._tutorial_dialog is None:
            return
        # Validation can rebuild a Qt editor, so resolve the current target instead of retaining a stale widget.
        target = self._tutorial_target(TUTORIAL_STEPS[self._tutorial_step].target)
        if target is not self._tutorial_highlighted_widget:
            self._highlight_tutorial_target(target)
            self._position_tutorial_dialog(target)
            self._place_tutorial_cursor_on_next()
            return
        if target is None:
            return
        try:
            self._layout_tutorial_highlight(target)
            frame = self._tutorial_highlight_frame
            if frame is not None:
                frame.show()
                frame.raise_()
        except RuntimeError:
            self._clear_tutorial_highlight()

    def _advance_guided_tutorial(self) -> None:
        step = TUTORIAL_STEPS[self._tutorial_step]
        if not step.action:
            if self._tutorial_step == len(TUTORIAL_STEPS) - 1:
                self._end_guided_tutorial()
                return
            self._tutorial_step += 1
            self._tutorial_phase = 0
            self._prepare_tutorial_step()
            return
        if self._tutorial_phase == 0:
            self._clear_tutorial_highlight()
            try:
                if not self._apply_tutorial_step(self._tutorial_step):
                    return
            except Exception as exc:
                QMessageBox.critical(self, "Tutorial", f"Could not complete this step:\n{exc}")
                return
            if step.single_click:
                if self._tutorial_step == len(TUTORIAL_STEPS) - 1:
                    self._end_guided_tutorial()
                    return
                self._tutorial_step += 1
                self._tutorial_phase = 0
                self._prepare_tutorial_step()
                return
            self._tutorial_phase = 1
            self._prepare_tutorial_step()
            return
        if self._tutorial_step == len(TUTORIAL_STEPS) - 1:
            self._end_guided_tutorial()
            return
        self._tutorial_step += 1
        self._tutorial_phase = 0
        self._prepare_tutorial_step()

    def _previous_guided_tutorial_step(self) -> None:
        if self._tutorial_step <= 0 and self._tutorial_phase == 0:
            return
        self._clear_tutorial_highlight()
        if self._tutorial_phase == 1:
            try:
                self._undo_tutorial_step(self._tutorial_step)
            except Exception as exc:
                QMessageBox.critical(self, "Tutorial", f"Could not return to the previous step:\n{exc}")
                self._prepare_tutorial_step()
                return
            self._tutorial_phase = 0
        else:
            self._tutorial_step -= 1
            self._tutorial_phase = 1 if TUTORIAL_STEPS[self._tutorial_step].action else 0
        self._prepare_tutorial_step()

    def _on_tutorial_dialog_closed(self, _result: int) -> None:
        if not self._tutorial_ending:
            self._end_guided_tutorial()

    def _end_guided_tutorial(self) -> None:
        if self._tutorial_ending:
            return
        self._tutorial_ending = True
        timer = self._tutorial_geometry_timer
        self._tutorial_geometry_timer = None
        if timer is not None:
            timer.stop()
        self._clear_tutorial_highlight()
        dialog = self._tutorial_dialog
        self._tutorial_dialog = None
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        snapshot = self._tutorial_snapshot
        self._tutorial_snapshot = None
        self._suppress_input_path_scan = False
        # Restore pipeline and inspection state independently because presets do not contain browser or preview state.
        if snapshot is not None:
            try:
                self.apply_preset_dict(snapshot, schedule_scan=True)
            except Exception as exc:
                QMessageBox.critical(self, "Tutorial", f"Could not restore your previous setup:\n{exc}")
        previous_browser_root = self._tutorial_previous_browser_root
        self._tutorial_previous_browser_root = None
        if hasattr(self, "file_tree"):
            if previous_browser_root and Path(previous_browser_root).is_dir():
                self.set_browser_root(str(previous_browser_root))
            else:
                self._show_browser_computer_view()
            self._browser_history = self._tutorial_previous_browser_history
            self._browser_forward_history = self._tutorial_previous_browser_forward_history
            self.update_browser_navigation_buttons()
        self._tutorial_previous_browser_history = []
        self._tutorial_previous_browser_forward_history = []
        previous_preview_path = self._tutorial_previous_preview_path
        self._tutorial_previous_preview_path = None
        if hasattr(self, "preview_view"):
            if previous_preview_path and Path(previous_preview_path).is_file():
                self.preview_file(previous_preview_path)
            else:
                self.reset_preview_display(clear_title=True)
        self._recent_preview_images = self._tutorial_previous_recent_images
        self._tutorial_previous_recent_images = []
        if hasattr(self, "preview_recent_images_button"):
            self.preview_recent_images_button.setEnabled(bool(self._recent_preview_images))
        if hasattr(self, "preview_inspector_toggle_button"):
            self.preview_inspector_toggle_button.setChecked(
                self._tutorial_previous_inspector_checked
            )
        if self._tutorial_temporary_files is not None:
            self._tutorial_temporary_files.cleanup()
            self._tutorial_temporary_files = None
        self._tutorial_saved_snapshot = None
        self._show_tutorial_section(0)
        self._tutorial_settings.setValue("guided_tutorial_seen", True)
        self._tutorial_settings.sync()
        self._tutorial_ending = False
        self._update_tutorial_button()
