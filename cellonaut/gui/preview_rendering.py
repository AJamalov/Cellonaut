"""Preview canvas rendering, page navigation, and display normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem

from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.preview_metadata import parse_hex_color


_PREVIEW_PAGE_CACHE_LIMIT = 4


class CellonautGuiPreviewRenderingMixin(GuiMixin):
    """Own cached preview state and rendering without file or worker I/O."""

    def set_preview_empty_state_visible(self, visible: bool) -> None:
        if hasattr(self, "preview_empty_state"):
            self.preview_empty_state.setVisible(bool(visible))

    # Preview state is cached aggressively for navigation speed, so every file
    # switch resets the model and normalized-page cache together.
    def clear_preview_tiff_state(self):
        self.preview_state.tiff_model = None
        self.preview_state.force_additive_composite = False
        self.preview_state.preserve_layer_colors = False
        self.preview_state.normalized_page_cache = {}
        self.preview_state.rendered_page_order = []
        self.preview_state.page_revisions = []
        self.preview_state.page_labels = []

    # Clear scene objects separately from TIFF metadata because both flat and layered previews use this canvas.
    def clear_preview_canvas(self):
        self.clear_preview_pages()
        self.clear_preview_mask_selection_outline()
        self.preview_scene.clear()
        self.preview_pixmap_item = None
        self.clear_preview_layer_items()
        self.set_preview_empty_state_visible(True)

    # Reset all linked preview state together so controls cannot operate on data from the previous file.
    def reset_preview_display(self, *, clear_title: bool = False):
        timer = getattr(self, "_preview_render_timer", None)
        if timer is not None:
            timer.stop()
        self.clear_preview_tiff_state()
        self.clear_preview_canvas()
        self.clear_preview_layer_controls()
        self.clear_preview_filter_overlay()
        self.preview_state.mask_adjustments_by_target = {}
        self.preview_state.mask_adjustment_last_preview_values = {}
        if getattr(self, "_current_open_right_panel_path", None) == self.preview_state.file_path:
            self._current_open_right_panel_path = None
        self.preview_state.file_path = None
        self.preview_state.overlay_colors = []
        self.preview_state.overlay_opacities = []
        self.preview_state.mask_adjust_target = ""
        self.preview_state.tools_source_label_override = ""
        self.preview_state.tools_source_matches_pipeline = True
        self.preview_state.render_revision = 0
        self.update_preview_snapshot_controls()
        if clear_title:
            self.set_preview_file_title(None)
            self.clear_preview_artifact_navigation()
        if hasattr(self, "update_preview_tools_context"):
            self.update_preview_tools_context()

    # Build the display in one place so cached pages and live layer updates use identical compositing rules.
    def make_tiff_page_pixmap(
        self,
        page_2d_or_cyx: np.ndarray,
        labels: list[str],
        colors: list[str],
        visibility: list[bool],
        opacities: list[float],
        already_normalized: bool = False,
    ) -> QPixmap:
        arr = np.asarray(page_2d_or_cyx)

        # Single-channel pages can use Qt grayscale images directly, which avoids
        # building unnecessary RGB arrays for ordinary mask previews.
        if arr.ndim == 2:
            u8 = (
                arr.astype(np.uint8, copy=False)
                if already_normalized
                else self.normalize_for_preview(arr).astype(np.uint8)
            )
            h, w = u8.shape
            opacity = max(0.0, min(1.0, float(opacities[0] if opacities else 1.0)))
            if colors:
                color = parse_hex_color(colors[0], fallback=(255, 255, 255))
                if not np.allclose(color, np.array([255, 255, 255], dtype=np.float32)):
                    alpha = (u8.astype(np.float32) / 255.0) * opacity
                    rgb = np.zeros((h, w, 3), dtype=np.float32)
                    for c in range(3):
                        rgb[..., c] = color[c] * alpha
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                    return QPixmap.fromImage(qimg)
            if opacity < 1.0:
                u8 = np.clip(u8.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
            qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
            return QPixmap.fromImage(qimg)

        # Multi-layer overlays are stored channel-first so layer controls can map
        # directly to TIFF pages and ImageJ sidecar colors.
        if arr.ndim != 3:
            raise ValueError(f"Unsupported preview page shape: {arr.shape}")

        c_count, h, w = arr.shape
        labels = list(labels or [])
        colors = list(colors or [])
        visibility = list(visibility or [])

        while len(labels) < c_count:
            labels.append(f"Channel {len(labels) + 1}")
        colors = self.preview_layer_colors(c_count, colors)
        while len(visibility) < c_count:
            visibility.append(True)
        opacities = [max(0.0, min(1.0, float(value))) for value in list(opacities or [])]
        while len(opacities) < c_count:
            opacities.append(1.0)

        # Plain RGB files should look natural by default; Fiji-style additive
        # coloring is reserved for overlay files with explicit layer metadata.
        if c_count in (3, 4) and not self.preview_state.force_additive_composite:
            chans = []
            for i in range(min(3, c_count)):
                if visibility[i]:
                    layer_u8 = (
                        arr[i].astype(np.uint8, copy=False)
                        if already_normalized
                        else self.normalize_for_preview(arr[i]).astype(np.uint8)
                    )
                    layer = layer_u8.astype(np.float32) * opacities[i]
                    chans.append(np.clip(layer, 0, 255).astype(np.uint8))
                else:
                    chans.append(np.zeros((h, w), dtype=np.uint8))
            while len(chans) < 3:
                chans.append(np.zeros((h, w), dtype=np.uint8))
            rgb = np.stack(chans[:3], axis=-1)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)

        # Saved overlays need to resemble Fiji composites so users can compare
        # Cellonaut previews with the files opened in ImageJ.
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        start_index = 0
        if self.preview_layer_is_grayscale_base(0, colors):
            base_layer = (
                arr[0].astype(np.uint8, copy=False)
                if already_normalized
                else self.normalize_for_preview(arr[0]).astype(np.uint8)
            )
            base_u8 = base_layer.astype(np.float32) if visibility[0] else np.zeros((h, w), dtype=np.float32)
            base_u8 *= opacities[0]
            rgb = np.stack([base_u8, base_u8, base_u8], axis=-1).astype(np.float32)
            start_index = 1

        for i in range(start_index, c_count):
            if not visibility[i]:
                continue

            layer_u8 = (
                arr[i].astype(np.uint8, copy=False)
                if already_normalized
                else self.normalize_for_preview(arr[i]).astype(np.uint8)
            )
            alpha = (layer_u8.astype(np.float32) / 255.0) * opacities[i]
            color = parse_hex_color(colors[i], fallback=(255, 0, 255))

            for c in range(3):
                rgb[..., c] = np.clip(rgb[..., c] + color[c] * alpha * 0.9, 0, 255)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    # Layer items are only worth the extra scene objects when users need
    # independent visibility, opacity, or ordering controls.
    def should_use_preview_layer_items(self) -> bool:
        model = self.preview_state.tiff_model
        if not model:
            return False
        data = model.get("data")
        if data is None or int(data.shape[2]) <= 1:
            return False
        composite = self.preview_state.composite_mode
        return bool(
            not composite and (model.get("is_overlay") or self.preview_state.force_additive_composite)
        )

    # Render an individual layer as RGBA so Qt can change its opacity and order without recomposing every layer.
    def make_preview_layer_pixmap(self, layer_u8: np.ndarray, color_hex: str, is_base: bool) -> QPixmap:
        u8 = np.asarray(layer_u8).astype(np.uint8, copy=False)
        h, w = u8.shape

        if is_base:
            rgb = np.stack([u8, u8, u8], axis=-1)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)

        color = parse_hex_color(color_hex, fallback=(255, 0, 255)).astype(np.uint8)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = u8
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(qimg)

    # Drop references after clearing the scene because Qt owns and destroys the underlying graphics items.
    def clear_preview_layer_items(self):
        self._preview_layer_items = []

    # Use separate scene items only for editable overlays; a single pixmap is cheaper for ordinary images.
    def set_preview_layer_items_for_page(self, page_index: int, preserve_view: bool = True) -> bool:
        if not self.should_use_preview_layer_items():
            return False

        model = self.preview_state.tiff_model
        if not model:
            return False

        data = model["data"]
        t_count = int(data.shape[0])
        z_count = int(data.shape[1])
        total_pages = max(1, t_count * z_count)
        page_index = max(0, min(int(page_index), total_pages - 1))
        t = page_index // z_count
        z = page_index % z_count
        page = self.get_normalized_tiff_preview_page(page_index, t, z)

        colors = list(self.preview_state.overlay_colors or [])
        colors = self.preview_layer_colors(page.shape[0], colors)
        self.preview_state.overlay_colors = colors

        opacities = self.get_overlay_layer_opacities()
        while len(opacities) < page.shape[0]:
            opacities.append(1.0)

        visibility = self.get_overlay_layer_visibility()
        while len(visibility) < page.shape[0]:
            visibility.append(True)

        view_state = self.capture_preview_view_state() if preserve_view else None
        self.clear_preview_mask_selection_outline()
        self.preview_scene.clear()
        self.preview_pixmap_item = None
        self._preview_layer_items = []
        self._preview_filter_item = None
        self.preview_scene.setSceneRect(0, 0, int(page.shape[2]), int(page.shape[1]))

        layer_order = list(self.preview_state.layer_order or range(page.shape[0]))
        for index in range(page.shape[0]):
            layer = page[index]
            pixmap = self.make_preview_layer_pixmap(
                layer, colors[index], is_base=self.preview_layer_is_grayscale_base(index, colors)
            )
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(layer_order.index(index) if index in layer_order else index)
            item.setVisible(bool(visibility[index]))
            item.setOpacity(max(0.0, min(1.0, float(opacities[index]))))
            self.preview_scene.addItem(item)
            self._preview_layer_items.append(item)

        self.update_preview_filter_overlay_item()
        self.restore_preview_snapshot_square_after_render()

        if preserve_view and view_state is not None:
            self.restore_preview_view_state(view_state)
        else:
            self.preview_view.fit_image()

        self.preview_state.page_index = page_index
        self.update_preview_mask_selection_outline(self.preview_state.selected_layer_index)
        self.update_preview_page_controls()
        return True

    # Update the scene item directly so dragging an opacity slider stays responsive on large images.
    def update_preview_layer_item_opacity(self, index: int, opacity: float) -> bool:
        items = getattr(self, "_preview_layer_items", []) or []
        if not self.should_use_preview_layer_items() or not 0 <= index < len(items):
            return False
        items[index].setOpacity(max(0.0, min(1.0, float(opacity))))
        return True

    # Toggle the existing item directly because visibility does not require rebuilding pixel data.
    def update_preview_layer_item_visibility(self, index: int, visible: bool) -> bool:
        items = getattr(self, "_preview_layer_items", []) or []
        if not self.should_use_preview_layer_items() or not 0 <= index < len(items):
            return False
        items[index].setVisible(bool(visible))
        return True

    # Rebuild only the recolored layer so color changes do not disturb zoom or the other layers.
    def update_preview_layer_item_color(self, index: int) -> bool:
        items = getattr(self, "_preview_layer_items", []) or []
        if not self.should_use_preview_layer_items() or not 0 <= index < len(items):
            return False

        model = self.preview_state.tiff_model
        if not model:
            return False

        data = model["data"]
        z_count = int(data.shape[1])
        page_index = max(0, min(self.preview_state.page_index, max(0, len(self.preview_state.pages) - 1)))
        t = page_index // z_count
        z = page_index % z_count
        page = self.get_normalized_tiff_preview_page(page_index, t, z)
        if not 0 <= index < page.shape[0]:
            return False

        colors = list(self.preview_state.overlay_colors or [])
        color_hex = colors[index] if 0 <= index < len(colors) else "#FFFFFF"
        layer = page[index]
        items[index].setPixmap(
            self.make_preview_layer_pixmap(
                layer,
                color_hex,
                is_base=self.preview_layer_is_grayscale_base(index, colors),
            )
        )
        return True

    # Refresh adjusted previews without changing the current page, zoom, or scroll position.
    def rebuild_tiff_preview_pages(self):
        model = self.preview_state.tiff_model
        if not model:
            return

        old_index = self.preview_state.page_index
        preserve_view = bool(self.preview_state.pages)
        self.preview_state.normalized_page_cache = {}
        self.clear_preview_layer_items()

        pixmaps, page_labels = self.build_tiff_preview_pages()
        self.preview_state.page_labels = list(page_labels)
        self.preview_state.page_revisions = [-1] * len(pixmaps)
        self.preview_state.rendered_page_order = []

        self.preview_state.pages = list(pixmaps)
        self.preview_state.file_path = self.preview_state.file_path or ""

        if not self.preview_state.pages:
            self.clear_preview_mask_selection_outline()
            self.preview_scene.clear()
            self.preview_pixmap_item = None
            self.preview_page_label.setText("Page: - / -")
            return

        new_index = max(0, min(old_index, len(self.preview_state.pages) - 1))
        self.show_preview_page(new_index, preserve_view=preserve_view)

        p = Path(self.preview_state.file_path) if self.preview_state.file_path else None
        data = model["data"]

        self.preview_info_label.setText(
            f"File: {p.name if p else '-'}\n"
            f"Type: TIFF\n"
            f"Original shape: {model.get('source_shape')}\n"
            f"Normalized TZCYX: {tuple(int(v) for v in data.shape)}\n"
            f"Dtype: {model.get('dtype')}\n"
            f"Channels: {len(self.preview_state.current_labels)}"
        )

    # Describe only dimensions that vary so single-plane files are not labelled with meaningless counters.
    def tiff_preview_page_label(self, t: int, z: int, t_count: int, z_count: int) -> str:
        if t_count > 1 and z_count > 1:
            return f"T {t + 1} / {t_count} | Z {z + 1} / {z_count}"
        if t_count > 1:
            return f"T {t + 1} / {t_count}"
        if z_count > 1:
            return f"Z {z + 1} / {z_count}"
        return "Image"

    # Convert one normalized T/Z plane on demand so page refreshes share the same index mapping.
    def render_tiff_preview_page(self, page_index: int) -> tuple[QPixmap, str]:
        model = self.preview_state.tiff_model
        if not model:
            raise ValueError("No TIFF/ND2 preview model is loaded")

        data = model["data"]  # T, Z, C, Y, X
        labels = self.preview_state.current_labels or ["Image"]
        colors = self.preview_state.overlay_colors or []
        visibility = self.get_overlay_layer_visibility()
        opacities = self.get_overlay_layer_opacities()

        t_count = int(data.shape[0])
        z_count = int(data.shape[1])
        c_count = int(data.shape[2])
        total_pages = max(1, t_count * z_count)
        page_index = max(0, min(int(page_index), total_pages - 1))

        if not visibility:
            visibility = [True] * c_count

        t = page_index // z_count
        z = page_index % z_count
        page = self.get_normalized_tiff_preview_page(page_index, t, z)  # C, Y, X uint8

        if c_count == 1:
            pixmap = self.make_tiff_page_pixmap(
                page[0],
                labels,
                colors,
                visibility,
                opacities,
                already_normalized=True,
            )
        else:
            pixmap = self.make_tiff_page_pixmap(
                page,
                labels,
                colors,
                visibility,
                opacities,
                already_normalized=True,
            )

        return pixmap, self.tiff_preview_page_label(t, z, t_count, z_count)

    # Cache normalized uint8 planes because percentile scaling is much costlier than page navigation.
    def get_normalized_tiff_preview_page(self, page_index: int, t: int, z: int) -> np.ndarray:
        cache = self.preview_state.normalized_page_cache
        if page_index in cache:
            normalized = cache.pop(page_index)
            cache[page_index] = normalized
            return normalized

        model = self.preview_state.tiff_model
        if not model:
            raise ValueError("No TIFF/ND2 preview model is loaded")

        page = np.asarray(model["data"][t, z])  # C, Y, X
        if page.ndim != 3:
            raise ValueError(f"Unsupported normalized preview page shape: {page.shape}")

        normalized = np.empty(page.shape, dtype=np.uint8)
        for channel_index in range(page.shape[0]):
            normalized[channel_index] = self.normalize_for_preview(page[channel_index]).astype(np.uint8, copy=False)

        cache[page_index] = normalized
        while len(cache) > _PREVIEW_PAGE_CACHE_LIMIT:
            cache.pop(next(iter(cache)))
        return normalized

    # Keep recently used rendered pages and release older ones to limit memory use.
    def remember_rendered_preview_page(self, page_index: int) -> None:
        order = list(self.preview_state.rendered_page_order or [])
        if page_index in order:
            order.remove(page_index)
        order.append(page_index)

        revisions = list(self.preview_state.page_revisions or [])
        while len(order) > _PREVIEW_PAGE_CACHE_LIMIT:
            expired = order.pop(0)
            if 0 <= expired < len(self.preview_state.pages):
                self.preview_state.pages[expired] = QPixmap()
            if 0 <= expired < len(revisions):
                revisions[expired] = -1
        self.preview_state.rendered_page_order = order
        self.preview_state.page_revisions = revisions

    # Reuse rendered pages while their layer settings remain unchanged.
    def refresh_tiff_preview_page(self, page_index: int | None = None, preserve_view: bool = True):
        model = self.preview_state.tiff_model
        if not model:
            return

        if page_index is None:
            page_index = self.preview_state.page_index
        if not self.preview_state.pages:
            return

        page_index = 0 if page_index is None else page_index
        page_index = max(0, min(int(page_index), len(self.preview_state.pages) - 1))
        current_revision = self.preview_state.render_revision
        page_revisions = list(self.preview_state.page_revisions or [])
        if len(page_revisions) != len(self.preview_state.pages):
            page_revisions = [-1] * len(self.preview_state.pages)

        if page_revisions[page_index] == current_revision:
            return

        pixmap, page_label = self.render_tiff_preview_page(page_index)
        self.preview_state.pages[page_index] = pixmap
        page_revisions[page_index] = current_revision
        self.preview_state.page_revisions = page_revisions
        self.remember_rendered_preview_page(page_index)

        page_labels = list(self.preview_state.page_labels or [])
        if len(page_labels) != len(self.preview_state.pages):
            page_labels = ["Image"] * len(self.preview_state.pages)
        page_labels[page_index] = page_label
        self.preview_state.page_labels = page_labels

        if page_index == self.preview_state.page_index:
            if self.set_preview_layer_items_for_page(page_index, preserve_view=preserve_view):
                return
            self.set_preview_pixmap(pixmap, preserve_view=preserve_view)
            self.update_preview_page_controls()

    # Create page placeholders so navigation works before all images are rendered.
    def build_tiff_preview_pages(self):
        model = self.preview_state.tiff_model
        if not model:
            return [], []

        data = model["data"]  # T, Z, C, Y, X
        t_count = int(data.shape[0])
        z_count = int(data.shape[1])

        pixmaps: list[QPixmap] = []
        page_labels: list[str] = []

        for t in range(t_count):
            for z in range(z_count):
                pixmaps.append(QPixmap())
                page_labels.append(self.tiff_preview_page_label(t, z, t_count, z_count))

        return pixmaps, page_labels

    # Batch visibility changes into one redraw when layers cannot be updated separately.
    def on_preview_layer_visibility_changed(self):
        if self.preview_state.tiff_model is None:
            return
        if self.should_use_preview_layer_items():
            items = getattr(self, "_preview_layer_items", []) or []
            visibility = self.get_overlay_layer_visibility()
            if visibility and len(items) == len(visibility):
                for index, visible in enumerate(visibility):
                    self.update_preview_layer_item_visibility(index, visible)
                return

        self.preview_state.render_revision = self.preview_state.render_revision + 1

        timer = getattr(self, "_preview_render_timer", None)
        if timer is None:
            timer = QTimer(self.as_qobject())
            timer.setSingleShot(True)
            timer.timeout.connect(self.apply_pending_preview_layer_render)
            self._preview_render_timer = timer

        if not timer.isActive():
            timer.start(16)

    # Render failures remain inside the preview panel because they should not terminate the application.
    def apply_pending_preview_layer_render(self):
        if self.preview_state.tiff_model is None:
            return
        try:
            self.refresh_tiff_preview_page(preserve_view=True)
        except Exception as e:
            self.preview_info_label.setText(f"Could not update TIFF preview.\n\n{type(e).__name__}: {e}")

    # Match ImageJ's default min/max display range without altering saved pixels.
    def normalize_for_preview(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)

        if arr.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        if arr.dtype == np.uint8:
            return arr

        arr = arr.astype(np.float32, copy=False)

        finite = np.isfinite(arr)
        if not finite.any():
            return np.zeros(arr.shape, dtype=np.uint8)

        vals = arr[finite]
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi <= lo:
            return np.where(finite & (arr > 0), 255, 0).astype(np.uint8)

        shown = np.zeros(arr.shape, dtype=np.uint8)
        shown[finite] = np.clip(
            np.floor((arr[finite].astype(np.float64) - lo) * (255.0 / (hi - lo)) + 0.5),
            0,
            255,
        ).astype(np.uint8)
        return shown

    # Save scroll position with the transform because either one alone can make a refreshed image appear to jump.
    def capture_preview_view_state(self) -> dict:
        hbar = self.preview_view.horizontalScrollBar()
        vbar = self.preview_view.verticalScrollBar()

        return {
            "transform": self.preview_view.transform(),
            "hvalue": hbar.value(),
            "vvalue": vbar.value(),
            "zoom_steps": getattr(self.preview_view, "_zoom", 0),
            "fit_scale": getattr(self.preview_view, "_fit_scale", 1.0),
        }

    # Treat viewport restoration as best effort because scene replacement can invalidate transient Qt state.
    def restore_preview_view_state(self, state: dict | None):
        if not state:
            return

        try:
            transform = state.get("transform")
            if transform is not None:
                self.preview_view.setTransform(transform)

            hbar = self.preview_view.horizontalScrollBar()
            vbar = self.preview_view.verticalScrollBar()

            if "hvalue" in state:
                hbar.setValue(int(state["hvalue"]))
            if "vvalue" in state:
                vbar.setValue(int(state["vvalue"]))
            if "zoom_steps" in state and "fit_scale" in state:
                self.preview_view._zoom = state["zoom_steps"]
                self.preview_view._fit_scale = state["fit_scale"]
                self.preview_view.zoom_changed.emit(self.preview_view.zoom_percent())
        except Exception:
            pass

    # Replace the pixmap in place where possible so zoom, scroll, filters, and snapshot controls remain stable.
    def set_preview_pixmap(self, pixmap: QPixmap, preserve_view: bool = False):
        view_state = self.capture_preview_view_state() if preserve_view else None
        self.set_preview_empty_state_visible(False)

        self.clear_preview_layer_items()
        if self.preview_pixmap_item is not None:
            self.preview_pixmap_item.setPixmap(pixmap)
        else:
            self.clear_preview_mask_selection_outline()
            self.preview_scene.clear()
            self._preview_filter_item = None
            self.preview_pixmap_item = QGraphicsPixmapItem(pixmap)
            self.preview_scene.addItem(self.preview_pixmap_item)

        self.update_preview_filter_overlay_item()
        self.update_preview_mask_selection_outline(self.preview_state.selected_layer_index)
        self.restore_preview_snapshot_square_after_render()

        if preserve_view and view_state is not None:
            self.restore_preview_view_state(view_state)
        else:
            self.preview_view.fit_image()
