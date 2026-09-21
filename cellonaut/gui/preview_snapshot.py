"""Snapshot capture and montage helpers for the preview tab."""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox

from cellonaut.gui.widgets import SnapshotSquareItem
from cellonaut.io.writers import save_pil_image, save_qt_image
from cellonaut.pipeline.processing_montage import PROCESSING_MONTAGE_METADATA_KEY
from cellonaut.results.layout import build_results_layout, results_root
from cellonaut.results.artifacts import artifact_root


# Tie snapshot selection to the current file and page so a crop cannot be reused
# accidentally after navigation changes its coordinate system.
class CellonautGuiPreviewSnapshotMixin:
    # Prefer the displayed pixmap bounds because scene bounds may also include handles and annotations.
    def image_preview_scene_rect(self) -> QRectF:
        if hasattr(self, "preview_pixmap_item") and self.preview_pixmap_item is not None:
            rect = self.preview_pixmap_item.boundingRect()
            if rect.isValid() and not rect.isNull():
                return QRectF(rect)
        scene = getattr(self, "preview_scene", None)
        if scene is not None:
            rect = scene.sceneRect()
            if rect.isValid() and not rect.isNull():
                return QRectF(rect)
            rect = scene.itemsBoundingRect()
            if rect.isValid() and not rect.isNull():
                return QRectF(rect)
        return QRectF()

    # Reuse one movable selection item so repeated renders do not leave stale controls in the scene.
    def ensure_preview_snapshot_square(self):
        scene = getattr(self, "preview_scene", None)
        if scene is None:
            return None
        bounds = self.image_preview_scene_rect()
        if not bounds.isValid() or bounds.isNull():
            return None
        item = getattr(self, "_preview_snapshot_item", None)
        try:
            item_scene = item.scene() if item is not None else None
        except RuntimeError:
            item = None
            item_scene = None
        if item is None or item_scene is not scene:
            item = SnapshotSquareItem(bounds)
            scene.addItem(item)
            self._preview_snapshot_item = item
        else:
            item.set_bounds(bounds)
        item.setVisible(True)
        return item

    # Require an open image before enabling snapshot selection.
    def toggle_preview_snapshot_square(self, checked: bool):
        if checked:
            item = self.ensure_preview_snapshot_square()
            if item is None and hasattr(self, "preview_snapshot_toggle_button"):
                self.preview_snapshot_toggle_button.blockSignals(True)
                self.preview_snapshot_toggle_button.setChecked(False)
                self.preview_snapshot_toggle_button.blockSignals(False)
                QMessageBox.information(self, "Snapshot", "Open an image preview before enabling the snapshot square.")
        else:
            item = getattr(self, "_preview_snapshot_item", None)
            if item is not None:
                try:
                    item.setVisible(False)
                except RuntimeError:
                    self._preview_snapshot_item = None
        self.update_preview_snapshot_controls()

    def preview_snapshot_enabled(self) -> bool:
        button = getattr(self, "preview_snapshot_toggle_button", None)
        return bool(button is not None and button.isChecked())

    # File and page changes invalidate scene coordinates, so clear the selection instead of reusing it on another image.
    def disable_preview_snapshot(self):
        button = getattr(self, "preview_snapshot_toggle_button", None)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        item = getattr(self, "_preview_snapshot_item", None)
        if item is not None:
            try:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            except RuntimeError:
                pass
        self._preview_snapshot_item = None
        self.update_preview_snapshot_controls()

    # Embedded panel geometry distinguishes generated processing montages from ordinary flat PNG files.
    def processing_montage_snapshot_layout(self) -> dict:
        current = Path(self.preview_state.file_path or "")
        if current.suffix.lower() != ".png" or not current.is_file():
            return {}
        try:
            with Image.open(current) as image:
                raw = image.info.get(PROCESSING_MONTAGE_METADATA_KEY, "")
            layout = json.loads(raw) if raw else {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        tiles = layout.get("tiles") if isinstance(layout, dict) else None
        if not isinstance(tiles, list) or not tiles:
            return {}
        for tile in tiles:
            if not isinstance(tile, dict):
                return {}
            try:
                if int(tile["width"]) <= 0 or int(tile["height"]) <= 0:
                    return {}
            except (KeyError, TypeError, ValueError):
                return {}
        return layout

    # Layer capture is meaningful only for a true channel stack or a Cellonaut processing montage.
    def layer_snapshot_available(self) -> bool:
        model = self.preview_state.tiff_model
        if model:
            data = model.get("data")
            if data is not None and getattr(data, "ndim", 0) == 5 and int(data.shape[2]) > 1:
                return True
        return bool(self.processing_montage_snapshot_layout())

    def update_preview_snapshot_controls(self):
        # Programmatic resets block toggle signals, so restore canvas dragging here too.
        if hasattr(self, "update_preview_interaction_mode"):
            self.update_preview_interaction_mode(self.preview_snapshot_enabled())
        image_available = self.image_preview_scene_rect().isValid()
        export_button = getattr(self, "preview_export_image_button", None)
        if export_button is not None:
            export_button.setEnabled(bool(image_available))
        toggle = getattr(self, "preview_snapshot_toggle_button", None)
        if toggle is not None:
            toggle.setEnabled(bool(image_available))
        button = getattr(self, "preview_snapshot_capture_button", None)
        if button is not None:
            button.setEnabled(bool(self.preview_snapshot_enabled() and self.current_snapshot_rect().isValid()))
        montage_action = getattr(self, "preview_snapshot_montage_action", None)
        if montage_action is not None:
            montage_action.setEnabled(self.layer_snapshot_available())

    # Rendering rebuilds scene items, so restore the user's selection instead of making them re-enable it.
    def restore_preview_snapshot_square_after_render(self):
        if self.preview_snapshot_enabled():
            self.ensure_preview_snapshot_square()
        self.update_preview_snapshot_controls()

    # Clamp the selection to a square inside the image to keep every saved snapshot predictable.
    def current_snapshot_rect(self) -> QRectF:
        item = getattr(self, "_preview_snapshot_item", None)
        if item is None:
            return QRectF()
        try:
            if not item.isVisible():
                return QRectF()
            rect = item.rect()
        except RuntimeError:
            self._preview_snapshot_item = None
            return QRectF()
        bounds = self.image_preview_scene_rect()
        if bounds.isValid() and not bounds.isNull():
            rect = rect.intersected(bounds)
        size = int(max(1, min(round(rect.width()), round(rect.height()))))
        x = int(max(0, round(rect.left())))
        y = int(max(0, round(rect.top())))
        if bounds.isValid() and not bounds.isNull():
            x = int(max(bounds.left(), min(x, bounds.right() - size)))
            y = int(max(bounds.top(), min(y, bounds.bottom() - size)))
        return QRectF(x, y, size, size)

    # Save beside the run being viewed, even when the configured output directory has since changed.
    def current_preview_results_root(self) -> Path:
        current = Path(self.preview_state.file_path or "")
        if current.name:
            root = artifact_root(current)
            if root is not None:
                return root

        stored = self.preview_state.artifact_results_root
        if stored:
            return Path(stored)

        output_widget = getattr(self, "output_dir", None)
        output_text = output_widget.get().strip() if output_widget is not None else ""
        if output_text:
            return results_root(Path(output_text))
        return Path("Results")

    def snapshot_output_dir(self) -> Path:
        return build_results_layout(self.current_preview_results_root(), create_root=False)["snapshots"]

    # Never overwrite an earlier capture; snapshots often document successive visual adjustments.
    def next_snapshot_path(self, suffix: str) -> Path:
        current = Path(self.preview_state.file_path or "")
        stem = current.stem if current.name else "snapshot"
        base_dir = self.snapshot_output_dir()
        candidate = base_dir / f"{stem}_{suffix}.png"
        if not candidate.exists():
            return candidate
        index = 2
        while True:
            numbered = base_dir / f"{stem}_{suffix}_{index}.png"
            if not numbered.exists():
                return numbered
            index += 1

    def save_visible_snapshot_from_selection(self):
        self.save_visible_snapshot(self.next_snapshot_path("snapshot"))

    # Use different filename suffixes for layer montages and combined images.
    def save_layer_snapshot_montage_from_selection(self):
        self.save_layer_snapshot_montage(self.next_snapshot_path("layer_montage"))

    # Render the scene to preserve the exact colors, opacity, order, and visibility shown to the user.
    def visible_snapshot_image(self, rect: QRectF) -> QImage:
        width = int(max(1, round(rect.width())))
        height = int(max(1, round(rect.height())))
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        item = getattr(self, "_preview_snapshot_item", None)
        was_visible = False
        if item is not None:
            try:
                was_visible = item.isVisible()
                item.setVisible(False)
            except RuntimeError:
                item = None
        painter = QPainter(image)
        try:
            self.preview_scene.render(painter, QRectF(0, 0, width, height), rect)
        finally:
            painter.end()
            if item is not None:
                item.setVisible(was_visible)
        return image

    def export_preview_image(self):
        """Export the full displayed page, including scene-based cell-group layers."""
        if not self.image_preview_scene_rect().isValid():
            return
        source = Path(self.preview_state.file_path or "image")
        suggested = self.snapshot_output_dir() / f"{source.stem}_edited"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self.as_qobject(), "Save preview image", str(suggested),
            "PNG image (*.png);;TIFF image (*.tif *.tiff)",
        )
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".tif" if selected_filter.startswith("TIFF") else ".png")
            if path.exists() and QMessageBox.question(
                self.as_qobject(), "Replace image?", f"Replace the existing file?\n{path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
        self.save_preview_image(path)

    def save_preview_image(self, path: Path) -> Path | None:
        """Save a flattened RGB image at scene resolution using an atomic writer."""
        rect = self.image_preview_scene_rect()
        if not rect.isValid() or rect.isNull():
            return None
        try:
            formats = {".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF"}
            file_format = formats.get(path.suffix.lower())
            if file_format is None:
                raise ValueError("Choose a .png, .tif, or .tiff filename.")
            rendered = self.visible_snapshot_image(rect).convertToFormat(QImage.Format.Format_RGBA8888)
            if rendered.isNull():
                raise ValueError("The preview image could not be rendered.")
            pixels = np.frombuffer(rendered.constBits(), dtype=np.uint8).reshape(
                rendered.height(), rendered.bytesPerLine()
            )[:, :rendered.width() * 4].reshape(rendered.height(), rendered.width(), 4)
            # Flatten transparent scene pixels onto black, matching the image canvas.
            rgba = Image.fromarray(pixels.copy())
            rgb = Image.new("RGB", rgba.size, "black")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            save_pil_image(rgb, path, format=file_format)
        except Exception as exc:
            QMessageBox.warning(self.as_qobject(), "Save preview image", f"Could not save image:\n{path}\n\n{exc}")
            return None
        self.log(f"[PREVIEW] Saved edited image: {path}")
        self.refresh_snapshot_artifact_availability()
        return path

    # Convert the captured image to a pixmap for montage assembly.
    def visible_snapshot_pixmap(self, rect: QRectF) -> QPixmap:
        return QPixmap.fromImage(self.visible_snapshot_image(rect))

    def save_visible_snapshot(self, out_path: Path) -> Path | None:
        rect = self.current_snapshot_rect()
        if not rect.isValid() or rect.isNull():
            QMessageBox.information(self.as_qobject(), "Snapshot", "Enable and position the snapshot square first.")
            return None
        image = self.visible_snapshot_image(rect)
        out_path = Path(out_path)
        if not save_qt_image(out_path, image, "PNG"):
            QMessageBox.warning(self.as_qobject(), "Snapshot", f"Could not save snapshot PNG:\n{out_path}")
            return None
        self.log(f"[PREVIEW] Saved visible snapshot: {out_path}")
        self.refresh_snapshot_artifact_availability()
        return out_path

    # Preserve the preview's reordered layer stack, then append any layers missing from older state.
    def snapshot_layer_order(self, count: int) -> list[int]:
        order = list(self.preview_state.layer_order or [])
        ordered = [int(index) for index in order if 0 <= int(index) < count]
        ordered.extend(index for index in range(count) if index not in ordered)
        return ordered

    # Crop normalized source layers separately so the montage can reveal contributions hidden in the composite.
    def snapshot_tiff_layer_crops(self) -> list[tuple[str, QPixmap]]:
        model = self.preview_state.tiff_model
        if not model:
            return []
        data = model.get("data")
        if data is None:
            return []
        page_index = max(0, int(self.preview_state.page_index))
        z_count = int(data.shape[1])
        t = page_index // z_count
        z = page_index % z_count
        page = self.get_normalized_tiff_preview_page(page_index, t, z)
        rect = self.current_snapshot_rect()
        if not rect.isValid() or rect.isNull():
            return []
        x = max(0, int(round(rect.left())))
        y = max(0, int(round(rect.top())))
        size = int(max(1, round(rect.width())))
        x2 = min(page.shape[2], x + size)
        y2 = min(page.shape[1], y + size)
        if x >= x2 or y >= y2:
            return []
        labels = list(self.preview_state.current_labels or [])
        colors = list(self.preview_state.overlay_colors or [])
        colors = self.preview_layer_colors(page.shape[0], colors)
        crops: list[tuple[str, QPixmap]] = []
        for index in self.snapshot_layer_order(page.shape[0]):
            label = labels[index] if index < len(labels) else f"Layer {index + 1}"
            crop = page[index, y:y2, x:x2]
            pixmap = self.make_preview_layer_pixmap(
                crop, colors[index], is_base=self.preview_layer_is_grayscale_base(index, colors)
            )
            crops.append((str(label), pixmap))
        overlay = self.visible_snapshot_pixmap(rect)
        if not overlay.isNull():
            crops.append(("Overlay", overlay))
        return crops

    # Use the selected panel as the coordinate reference, then apply that relative crop to every montage panel.
    def processing_montage_snapshot_crops(self) -> list[tuple[str, QPixmap]]:
        layout = self.processing_montage_snapshot_layout()
        tiles = list(layout.get("tiles", []) or [])
        rect = self.current_snapshot_rect()
        if not tiles or not rect.isValid() or rect.isNull():
            return []

        source = QPixmap(str(self.preview_state.file_path or ""))
        if source.isNull():
            return []

        source_tile = None
        source_body = QRectF()
        largest_overlap = 0.0
        for tile in tiles:
            body = QRectF(float(tile["x"]), float(tile["y"]), float(tile["width"]), float(tile["height"]))
            overlap = rect.intersected(body)
            area = max(0.0, overlap.width()) * max(0.0, overlap.height())
            if area > largest_overlap:
                largest_overlap = area
                source_tile = tile
                source_body = body
        if source_tile is None or largest_overlap <= 0 or source_body.isNull():
            return []

        selected = rect.intersected(source_body)
        x_fraction = (selected.left() - source_body.left()) / source_body.width()
        y_fraction = (selected.top() - source_body.top()) / source_body.height()
        width_fraction = selected.width() / source_body.width()
        height_fraction = selected.height() / source_body.height()

        crops: list[tuple[str, QPixmap]] = []
        for index, tile in enumerate(tiles):
            body = QRectF(float(tile["x"]), float(tile["y"]), float(tile["width"]), float(tile["height"]))
            x = int(round(body.left() + x_fraction * body.width()))
            y = int(round(body.top() + y_fraction * body.height()))
            width = max(1, int(round(width_fraction * body.width())))
            height = max(1, int(round(height_fraction * body.height())))
            crop_rect = QRect(x, y, width, height).intersected(source.rect())
            if crop_rect.isNull():
                continue
            label = str(tile.get("label", "") or f"Image {index + 1}")
            crops.append((label, source.copy(crop_rect)))
        return crops

    # Dispatch by preview structure so TIFF layers and processing-montage panels share one capture command.
    def snapshot_layer_crops(self) -> list[tuple[str, QPixmap]]:
        if self.processing_montage_snapshot_layout():
            return self.processing_montage_snapshot_crops()
        return self.snapshot_tiff_layer_crops()

    # Use a compact four-column grid so montages remain readable without becoming excessively wide.
    def make_snapshot_montage_pixmap(self, crops: list[tuple[str, QPixmap]]) -> QPixmap:
        if not crops:
            return QPixmap()
        label_height = 24
        gap = 6
        tile_w = max(pixmap.width() for _label, pixmap in crops)
        tile_h = max(pixmap.height() for _label, pixmap in crops)
        columns = min(4, max(1, len(crops)))
        rows = int(np.ceil(len(crops) / columns))
        width = columns * tile_w + (columns - 1) * gap
        height = rows * (tile_h + label_height) + (rows - 1) * gap
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(QColor("#111111"))
        painter = QPainter(image)
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#f2f2f2"))
        try:
            for idx, (label, pixmap) in enumerate(crops):
                row = idx // columns
                col = idx % columns
                x = col * (tile_w + gap)
                y = row * (tile_h + label_height + gap)
                painter.drawText(
                    QRectF(x + 2, y, tile_w - 4, label_height),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    label,
                )
                painter.drawPixmap(x, y + label_height, pixmap)
        finally:
            painter.end()
        return QPixmap.fromImage(image)

    # The menu disables unsupported files; this guard only handles a selection outside every valid source panel.
    def save_layer_snapshot_montage(self, out_path: Path) -> Path | None:
        crops = self.snapshot_layer_crops()
        if not crops:
            if self.layer_snapshot_available():
                QMessageBox.information(
                    self.as_qobject(),
                    "Snapshot montage",
                    "Position the snapshot square over an image panel before capturing the montage.",
                )
            return None
        montage = self.make_snapshot_montage_pixmap(crops)
        if montage.isNull():
            return None
        out_path = Path(out_path)
        if not save_qt_image(out_path, montage, "PNG"):
            QMessageBox.warning(self.as_qobject(), "Snapshot montage", f"Could not save montage PNG:\n{out_path}")
            return None
        self.log(f"[PREVIEW] Saved layer snapshot montage: {out_path}")
        self.refresh_snapshot_artifact_availability()
        return out_path
