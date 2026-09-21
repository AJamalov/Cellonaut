"""Preview file loading and format dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PySide6.QtGui import QPixmap

from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.preview_metadata import is_overlay_preview_sidecar, load_tiff_imagej_display_sidecar
import cellonaut.io.nd2_import as ndi


class CellonautGuiPreviewLoadingMixin(GuiMixin):
    """Decode supported image formats into the shared preview model."""

    PREVIEW_INSPECTOR_COLLAPSE_WIDTH = 620
    PREVIEW_INSPECTOR_RESTORE_WIDTH = 760

    def refresh_preview_inspector_content(self) -> None:
        """Show an empty hint only when neither inspector panel is active."""
        layer_visible = hasattr(self, "preview_layer_bar") and not self.preview_layer_bar.isHidden()
        metadata_visible = hasattr(self, "preview_info_panel") and not self.preview_info_panel.isHidden()
        if hasattr(self, "preview_inspector_empty_label"):
            self.preview_inspector_empty_label.setVisible(not layer_visible and not metadata_visible)

    def toggle_preview_inspector(self, checked: bool) -> None:
        """Show or hide the side inspector without changing its selected content."""
        available_width = self.right_container.width() if hasattr(self, "right_container") else 0
        if checked and 0 < available_width < self.PREVIEW_INSPECTOR_COLLAPSE_WIDTH:
            self._preview_inspector_auto_collapsed = True
            button = getattr(self, "preview_inspector_toggle_button", None)
            if button is not None:
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
            checked = False
        if hasattr(self, "preview_inspector"):
            self.preview_inspector.setVisible(bool(checked))
        if hasattr(self, "preview_inspector_toggle_button"):
            self.preview_inspector_toggle_button.setToolTip(
                "Hide overlay layers and image metadata."
                if checked
                else "Show overlay layers and image metadata."
            )
        self.refresh_preview_inspector_content()

    def update_preview_responsive_layout(self, available_width: int | None = None) -> None:
        """Protect canvas width by collapsing and later restoring the optional inspector."""
        button = getattr(self, "preview_inspector_toggle_button", None)
        if button is None:
            return
        width = int(available_width if available_width is not None else self.right_container.width())
        if width < self.PREVIEW_INSPECTOR_COLLAPSE_WIDTH and button.isChecked():
            self._preview_inspector_auto_collapsed = True
            button.setChecked(False)
        elif width >= self.PREVIEW_INSPECTOR_RESTORE_WIDTH and bool(
            getattr(self, "_preview_inspector_auto_collapsed", False)
        ):
            self._preview_inspector_auto_collapsed = False
            button.setChecked(True)

    def set_preview_layer_panel_visible(self, visible: bool) -> None:
        """Keep multilayer controls and the collapsible inspector synchronized."""
        if hasattr(self, "preview_layer_bar"):
            metadata_open = bool(
                hasattr(self, "preview_metadata_button") and self.preview_metadata_button.isChecked()
            )
            self.preview_layer_bar.setVisible(bool(visible and not metadata_open))
        if visible and hasattr(self, "preview_inspector_toggle_button"):
            self.preview_inspector_toggle_button.setChecked(True)
        elif not visible and hasattr(self, "preview_inspector_toggle_button"):
            metadata_open = bool(
                hasattr(self, "preview_metadata_button") and self.preview_metadata_button.isChecked()
            )
            if not metadata_open:
                self.preview_inspector_toggle_button.setChecked(False)
        self.refresh_preview_inspector_content()

    # Metadata stays optional so the image remains the main focus on smaller screens.
    def toggle_preview_metadata_panel(self, checked: bool):
        if hasattr(self, "preview_info_panel"):
            self.preview_info_panel.setVisible(bool(checked))
        if hasattr(self, "preview_layer_bar"):
            has_layers = len(self.preview_state.current_labels or []) > 1
            self.preview_layer_bar.setVisible(bool(has_layers and not checked))
        if hasattr(self, "preview_inspector_title"):
            self.preview_inspector_title.setText("Metadata" if checked else "Layers and metadata")
        if checked and hasattr(self, "preview_inspector_toggle_button"):
            self.preview_inspector_toggle_button.setChecked(True)
        if hasattr(self, "preview_metadata_button"):
            self.preview_metadata_button.setText("")
            self.preview_metadata_button.setToolTip("Hide metadata" if checked else "Show metadata")
        if not checked and hasattr(self, "preview_inspector_toggle_button"):
            layers_open = bool(hasattr(self, "preview_layer_bar") and not self.preview_layer_bar.isHidden())
            if not layers_open:
                self.preview_inspector_toggle_button.setChecked(False)
        self.refresh_preview_inspector_content()

    # Prefer tifffile metadata and sidecars because Qt cannot preserve scientific TIFF axes or layer colors.
    def preview_tiff_image(self, file_path: str):
        p = Path(file_path)
        self.reset_preview_display()

        try:
            with tifffile.TiffFile(file_path) as tif:
                series = tif.series[0]
                arr = series.asarray()
                series_axes = str(getattr(series, "axes", "") or "")
        except Exception as e:
            raise ValueError(f"Could not read TIFF with tifffile:\n{e}")

        arr = np.asarray(arr)
        sidecar = self.load_overlay_sidecar(file_path)
        if not sidecar:
            sidecar = load_tiff_imagej_display_sidecar(file_path)
        if series_axes and "axes" not in sidecar:
            sidecar["axes"] = series_axes

        self.preview_state.preserve_layer_colors = bool(is_overlay_preview_sidecar(sidecar))
        model = self.normalize_tiff_to_tzcyx(arr, sidecar=sidecar)
        self.preview_state.tiff_model = model
        self.preview_state.file_path = file_path

        labels = self.get_tiff_channel_labels(model, sidecar)
        colors = self.get_tiff_channel_colors(model, sidecar)

        self.preview_state.overlay_colors = colors
        self.preview_state.overlay_opacities = [1.0] * len(labels)

        self.preview_state.force_additive_composite = bool(
            model.get("is_overlay")
            or model.get("metadata_channel_axis")
            or sidecar.get("layer_labels")
            or sidecar.get("layer_colors")
        )
        self.set_preview_layer_metadata(sidecar, len(labels))
        if len(labels) > 1:
            self.set_preview_layer_controls(labels)
        else:
            self.clear_preview_layer_controls()
            self.preview_state.current_labels = labels
        self.update_mask_adjust_target_options()

        self.clear_preview_pages()
        self.rebuild_tiff_preview_pages()
        self.refresh_preview_filter_overlay()

        if not self.preview_state.pages:
            self.clear_preview_canvas()

            raise ValueError("No TIFF preview pages could be generated")

        data = model["data"]
        self.preview_info_label.setText(
            f"File: {p.name}\n"
            f"Type: TIFF\n"
            f"Original shape: {model.get('source_shape')}\n"
            f"Normalized TZCYX: {tuple(int(v) for v in data.shape)}\n"
            f"Dtype: {model.get('dtype')}\n"
            f"Channels: {len(labels)}\n"
            f"Pages: {len(self.preview_state.pages)}"
        )

    # Normalize ND2 into the TIFF preview model so both formats share rendering and navigation behavior.
    def preview_nd2_image(self, file_path: str):
        p = Path(file_path)
        self.reset_preview_display()

        try:
            ndi.require_nd2()
            with ndi.nd2.ND2File(p) as f:
                sizes = dict(getattr(f, "sizes", {}))
                channel_names = ndi.get_nd2_channel_names(f)
                channel_colors = ndi.get_nd2_channel_colors(f, channel_names)
                arr = ndi.read_first_nd2_position(f)
        except Exception as e:
            raise ValueError(f"Could not read ND2 file:\n{e}")

        self.preview_state.preserve_layer_colors = False
        model = self.normalize_nd2_to_tzcyx(arr, sizes)
        self.preview_state.tiff_model = model
        self.preview_state.file_path = file_path

        c_count = int(model["data"].shape[2])
        if channel_names and len(channel_names) == c_count:
            labels = list(channel_names)
        elif c_count == 1:
            labels = ["Image"]
        else:
            labels = [f"Channel {i + 1}" for i in range(c_count)]

        colors = self.get_tiff_channel_colors(
            model,
            {"layer_colors": channel_colors or self.inferred_channel_colors_from_labels(labels)},
        )
        self.preview_state.overlay_colors = colors
        self.preview_state.overlay_opacities = [1.0] * len(labels)
        self.preview_state.force_additive_composite = c_count > 1
        self.set_preview_layer_metadata({}, len(labels))

        if len(labels) > 1:
            self.set_preview_layer_controls(labels)
        else:
            self.clear_preview_layer_controls()
            self.preview_state.current_labels = labels

        self.clear_preview_pages()
        self.rebuild_tiff_preview_pages()
        self.clear_preview_filter_overlay()

        if not self.preview_state.pages:
            self.clear_preview_canvas()

            raise ValueError("No ND2 preview pages could be generated")

        data = model["data"]
        self.preview_info_label.setText(
            f"File: {p.name}\n"
            f"Type: ND2\n"
            f"ND2 sizes: {sizes}\n"
            f"Raw shape: {model.get('source_shape')}\n"
            f"Normalized TZCYX: {tuple(int(v) for v in data.shape)}\n"
            f"Dtype: {model.get('dtype')}\n"
            f"Channels: {', '.join(labels)}\n"
            f"Pages: {len(self.preview_state.pages)}"
        )

    def record_recent_preview_image(self, file_path: str) -> None:
        """Keep a bounded, most-recent-first list of images that opened successfully."""
        path = str(Path(file_path).resolve())
        recent = [item for item in getattr(self, "_recent_preview_images", []) if item != path]
        self._recent_preview_images = [path, *recent][:10]
        if hasattr(self, "preview_recent_images_button"):
            self.preview_recent_images_button.setEnabled(True)

    def rebuild_recent_preview_images_menu(self) -> None:
        """Refresh the recent-image menu and discard paths that no longer exist."""
        if not hasattr(self, "preview_recent_images_menu"):
            return
        menu = self.preview_recent_images_menu
        menu.clear()
        existing = [path for path in getattr(self, "_recent_preview_images", []) if Path(path).is_file()]
        self._recent_preview_images = existing[:10]
        for path in self._recent_preview_images:
            action = menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, selected=path: self.preview_file(selected))
        if not self._recent_preview_images:
            empty_action = menu.addAction("No recent images")
            empty_action.setEnabled(False)
        if hasattr(self, "preview_recent_images_button"):
            self.preview_recent_images_button.setEnabled(bool(self._recent_preview_images))

    # Choose the image loader from the file extension.
    def preview_file(self, file_path: str):
        scene = getattr(self, "preview_scene", None)
        view_state = self.capture_preview_view_state() if scene is not None and scene.items() else None
        view = getattr(self, "preview_view", None)
        if view is not None:
            view._preserve_view_during_load = view_state is not None
        self.disable_preview_snapshot()
        p = Path(file_path)
        suffix = p.suffix.lower()

        try:
            if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
                self.preview_standard_image(file_path)
            elif suffix in {".tif", ".tiff"}:
                self.preview_tiff_image(file_path)
            elif suffix == ".nd2":
                self.preview_nd2_image(file_path)
            else:
                self.preview_info_label.setText(
                    f"Unsupported preview type:\n{p.name}\n\nSupported: PNG, JPG, TIFF, ND2"
                )
                self.reset_preview_display(clear_title=True)

                return

            if self.preview_state.artifact_entries:
                if self.sync_preview_artifact_selection_to_current_file():
                    self.update_preview_artifact_nav_controls()
                else:
                    self.clear_preview_artifact_navigation()
            root = self.preview_state.artifact_results_root
            if root is not None and not p.resolve().is_relative_to(Path(root).resolve()):
                self.preview_state.artifact_results_root = None
            self.set_preview_file_title(file_path)
            self.set_preview_empty_state_visible(False)
            self.record_recent_preview_image(file_path)
            self.refresh_snapshot_artifact_availability()
            if hasattr(self, "update_preview_tools_context"):
                self.update_preview_tools_context()
            self._current_open_right_panel_path = str(file_path)
            if view_state is not None:
                self.restore_preview_view_state(view_state)
            return True

        except Exception as e:
            # A failed attempt must remain retryable from the Files tree. The
            # selection handler records the path before dispatching the loader.
            if getattr(self, "_current_open_right_panel_path", None) == str(file_path):
                self._current_open_right_panel_path = None
            self.reset_preview_display(clear_title=True)
            message = f"Could not preview file: {p.name} ({type(e).__name__}: {e})"
            self.preview_info_label.setText(f"Could not preview file:\n{p.name}\n\n{type(e).__name__}: {e}")
            log_func = getattr(self, "log", None)
            if callable(log_func):
                log_func(f"[PREVIEW][ERROR] {message}")
            if hasattr(self, "update_preview_tools_context"):
                self.update_preview_tools_context()
            return False
        finally:
            if view is not None:
                view._preserve_view_during_load = False

    # Let Qt load common display formats directly because they have no scientific axes to preserve.
    def preview_standard_image(self, file_path: str):
        self.reset_preview_display()

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            raise ValueError("Image could not be loaded")

        self.set_preview_pages([pixmap], file_path)

        self.preview_info_label.setText(
            f"File: {Path(file_path).name}\n"
            f"Size: {pixmap.width()} x {pixmap.height()}\n"
            f"Type: standard image\n"
            f"Pages: 1"
        )

    # Check artifact categories in priority order and open the newest file in the first available category.
    def show_latest_qc_overlay(self):
        out_dir = Path(str(getattr(self, "_latest_pipeline_output_dir", "") or self.output_dir.get()))
        if not out_dir.exists():
            return

        from cellonaut.pipeline.preview_artifacts import collect_preview_artifacts
        from cellonaut.results.artifacts import ArtifactResolver

        if ArtifactResolver.load(out_dir) is not None:
            preview = collect_preview_artifacts(out_dir).get("preview_path")
            if preview:
                self.preview_file(preview)
            return
        # Historical output folders have no artifact inventory.
        categories = (
            list(out_dir.rglob("*_combined_overlay.tif")),
            list(out_dir.rglob("*_cellpose_outline_stack.tif")),
            list(out_dir.rglob("*_overlay.tif")),
            list(out_dir.rglob("*_qc.png"))
            + list(out_dir.rglob("*_qc_overlay.png"))
            + list(out_dir.rglob("*_QC_overlay_flat.png")),
            list(out_dir.rglob("*_montage.png")),
        )
        for candidates in categories:
            matches = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
            if matches:
                self.preview_file(str(matches[0]))
                return
