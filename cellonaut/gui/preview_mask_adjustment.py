"""Preview-side mask adjustment controls.

These helpers let users temporarily inspect and tune mask shifts, growth, hole
filling, and minimum-area filtering on the current preview.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from cellonaut.config.defaults import default_mask_adjustments
from cellonaut.gui.mixin import GuiMixin
from cellonaut.gui.preview_colors import ADJUSTED_MASK_COLOR_PRIORITIES, DEFAULT_PREVIEW_COLOR
from cellonaut.masks.adjustments import adjust_mask_image

MASK_ADJUSTMENT_KEYS = tuple(default_mask_adjustments())


# Isolate temporary mask tuning from saved pipeline settings so users can inspect
# an adjustment without silently changing the reproducible workflow.
class CellonautGuiPreviewMaskAdjustmentMixin(GuiMixin):
    # Merge over fresh defaults because GUI state can be temporarily incomplete
    # while channels are rebuilt or a preset is being applied.
    def _mask_recipe(self, value: dict | None, target_key: str) -> dict:
        recipe = {"target": target_key, **default_mask_adjustments()}
        if isinstance(value, dict):
            for key in MASK_ADJUSTMENT_KEYS:
                recipe[key] = int(value.get(key, 0) or 0)
        return recipe

    # Match preview layers to image definitions when loading existing adjustment settings.
    def _weka_definition_index_for_preview_layer(self, layer_index: int) -> int | None:
        keys = list(self.preview_state.current_layer_keys or [])
        if not 0 <= layer_index < len(keys):
            return None
        image_key = str(keys[layer_index] or "").split("__class", 1)[0]
        if image_key.startswith("image") and image_key[5:].isdigit():
            return int(image_key[5:]) - 1
        return None

    # Parse target keys in one place so malformed sidecar data cannot address an
    # unintended preview layer when loading settings or previewing adjustments.
    @staticmethod
    def _weka_layer_index(target_key: str) -> int | None:
        if not str(target_key or "").startswith("weka_layer:"):
            return None
        try:
            return int(str(target_key).split(":", 1)[1])
        except (TypeError, ValueError):
            return None

    # Use the same field list when reading, loading, and resetting controls.
    def _mask_adjustment_widgets(self):
        return (
            ("dx", self.mask_adjust_x_spin),
            ("dy", self.mask_adjust_y_spin),
            ("grow_px", self.mask_adjust_grow_spin),
            ("min_size", self.mask_adjust_min_size_spin),
            ("fill_holes_area", self.mask_adjust_fill_holes_spin),
        )

    # Load saved adjustments only when a preview layer can be matched to a configured mask.
    def load_persisted_mask_adjustments(self):
        if not hasattr(self, "mask_adjust_target_combo"):
            return
        active_defs = self.get_active_image_definitions()
        adjustments = self.preview_state.mask_adjustments_by_target

        for combo_index in range(self.mask_adjust_target_combo.count()):
            target_key = str(self.mask_adjust_target_combo.itemData(combo_index) or "")
            layer_index = self._weka_layer_index(target_key)
            if layer_index is None:
                continue
            image_index = self._weka_definition_index_for_preview_layer(layer_index)
            if image_index is None or not 0 <= image_index < len(active_defs):
                continue
            adjustments.setdefault(target_key, self._mask_recipe(
                active_defs[image_index].get("mask_adjustments", {}),
                target_key,
            ))
        self.load_mask_adjustment_values_for_target(self.mask_adjust_target_key())

    # Require explicit sidecar roles because guessing from layer names can offer
    # raw image channels as masks eligible for temporary adjustment layers.
    def preview_weka_mask_layer_indices(self) -> list[int]:
        labels = list(self.preview_state.current_labels or [])
        roles = list(self.preview_state.current_layer_roles or [])
        if len(roles) != len(labels):
            return []
        return [index for index, role in enumerate(roles) if str(role or "").strip().lower() == "weka_mask"]

    # Read the stable data key rather than visible text because labels can be
    # duplicated when an image and its mask share a display name.
    def mask_adjust_target_key(self) -> str:
        return self.preview_state.mask_adjust_target

    # Resolve labels lazily from the current preview so status text follows a
    # newly loaded sample without persisting display-only names.
    def mask_adjust_target_label(self, target_key: str | None = None) -> str:
        key = str(target_key or self.mask_adjust_target_key())
        index = self._weka_layer_index(key)
        if index is not None:
            labels = list(self.preview_state.current_labels or [])
            label = str(labels[index]) if 0 <= index < len(labels) else f"layer {index}"
            return f"Mask: {label}"
        return key or "No mask selected"

    # Include the target beside numeric defaults because the transient editor
    # stores temporary recipes by layer key.
    def default_mask_adjustment_values(self, target_key: str = "") -> dict:
        return {"target": target_key, **default_mask_adjustments()}

    # Cache unsaved edits per target so switching between masks does not discard
    # values during the current preview session.
    def save_current_mask_adjustment_values(self):
        if not hasattr(self, "mask_adjust_target_combo"):
            return
        target_key = self.mask_adjust_target_key()
        if not target_key:
            return
        values = self.mask_adjustment_values()
        adjustments = self.preview_state.mask_adjustments_by_target
        adjustments[target_key] = values

    # Block signals while changing all controls so loading one recipe causes a
    # single deliberate status refresh rather than five partial saves.
    def load_mask_adjustment_values_for_target(self, target_key: str):
        if not hasattr(self, "mask_adjust_target_combo"):
            return
        adjustments = self.preview_state.mask_adjustments_by_target or {}
        values = dict(adjustments.get(target_key) or self.default_mask_adjustment_values(target_key))
        for key, widget in self._mask_adjustment_widgets():
            widget.blockSignals(True)
            widget.setValue(int(values.get(key, 0) or 0))
            widget.blockSignals(False)

    # Preserve the selected layer while rebuilding options because preview layer
    # controls are recreated after color, order, and adjusted-layer changes.
    def update_mask_adjust_target_options(self):
        if not hasattr(self, "mask_adjust_target_combo"):
            return
        current_key = self.mask_adjust_target_key()
        self.save_current_mask_adjustment_values()
        combo = self.mask_adjust_target_combo
        combo.blockSignals(True)
        combo.clear()
        labels = list(self.preview_state.current_labels or [])
        for index in self.preview_weka_mask_layer_indices():
            label = str(labels[index]) if 0 <= index < len(labels) else f"layer {index}"
            combo.addItem(f"Mask: {label}", f"weka_layer:{index}")
        new_index = combo.findData(current_key)
        combo.setCurrentIndex(new_index if new_index >= 0 else 0)
        combo.blockSignals(False)
        self.preview_state.mask_adjust_target = str(combo.currentData() or "")
        self.load_mask_adjustment_values_for_target(self.preview_state.mask_adjust_target)

    # Read every control through the shared field map so cached edits and previews
    # use the same recipe shape.
    def mask_adjustment_values(self) -> dict:
        if not hasattr(self, "mask_adjust_target_combo"):
            return self.default_mask_adjustment_values()
        values = {key: int(widget.value()) for key, widget in self._mask_adjustment_widgets()}
        return {"target": self.mask_adjust_target_key(), **values}

    # Treat all-zero morphology and shift values as inactive so a saved no-op
    # recipe does not force unnecessary mask regeneration.
    def mask_adjustment_is_active(self) -> bool:
        values = self.mask_adjustment_values()
        return any(int(values.get(key, 0) or 0) != 0 for key in MASK_ADJUSTMENT_KEYS)

    # Availability and the concise recipe summary update together so controls
    # cannot appear actionable when the open artifact has no editable mask layer.
    def update_mask_adjustment_status(self):
        if not hasattr(self, "mask_adjust_status_label"):
            return

        has_target = bool(self.mask_adjust_target_key())
        if hasattr(self, "mask_adjust_target_combo"):
            self.mask_adjust_target_combo.setEnabled(has_target)
        for _key, widget in self._mask_adjustment_widgets():
            widget.setEnabled(has_target)

        if not has_target:
            self.mask_adjust_status_label.setText("Open a generated overlay with linked Weka mask metadata to select a mask.")
            if hasattr(self, "mask_adjust_preview_button"):
                self.mask_adjust_preview_button.setEnabled(False)
            if hasattr(self, "mask_adjust_reset_button"):
                self.mask_adjust_reset_button.setEnabled(False)
            return

        values = self.mask_adjustment_values()
        target_label = self.mask_adjust_target_label(values.get("target"))
        active = self.mask_adjustment_is_active()
        has_adjusted_layer = self.adjusted_preview_layer_key(values["target"]) in (
            self.preview_state.current_layer_keys or []
        )
        if hasattr(self, "mask_adjust_preview_button"):
            self.mask_adjust_preview_button.setEnabled(True)
        if hasattr(self, "mask_adjust_reset_button"):
            self.mask_adjust_reset_button.setEnabled(active or has_adjusted_layer)
        if not active:
            self.mask_adjust_status_label.setText(f"Target: {target_label}; no adjustments active")
            return
        self.mask_adjust_status_label.setText(
            f"Target: {target_label} | "
            f"shift X={values['dx']} Y={values['dy']} | grow={values['grow_px']} | "
            f"min mask area={values['min_size']} | fill holes up to={values['fill_holes_area']} px²"
        )

    # Load the newly selected recipe before the generic change handler caches it.
    def on_mask_adjust_target_changed(self, *_args):
        self.preview_state.mask_adjust_target = str(self.mask_adjust_target_combo.currentData() or "")
        self.load_mask_adjustment_values_for_target(self.mask_adjust_target_key())
        self.on_mask_adjustment_changed()
        layer_index = self._weka_layer_index(self.mask_adjust_target_key())
        if layer_index is None or not hasattr(self, "preview_layer_list"):
            return
        for row in range(self.preview_layer_list.count()):
            item = self.preview_layer_list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == layer_index:
                self.preview_layer_list.setCurrentRow(row)
                break

    # Keep edits without rebuilding the image until the user clicks Preview.
    def on_mask_adjustment_changed(self, *_args):
        self.save_current_mask_adjustment_values()
        self.update_mask_adjustment_status()
        values = self.mask_adjustment_values()
        last_preview = dict(
            self.preview_state.mask_adjustment_last_preview_values.get(self.mask_adjust_target_key(), {})
        )
        has_adjusted_layer = self.adjusted_preview_layer_key(self.mask_adjust_target_key()) in (
            self.preview_state.current_layer_keys or []
        )
        if (self.mask_adjustment_is_active() or has_adjusted_layer) and values != last_preview:
            self.mask_adjust_status_label.setText(
                f"Changes not previewed · {self.mask_adjust_status_label.text()}"
            )

    # Remove both the temporary recipe and its generated layer so Reset returns
    # the preview to the unadjusted source rather than merely zeroing controls.
    def reset_mask_adjustments(self):
        if not hasattr(self, "mask_adjust_target_combo"):
            return
        for _key, widget in self._mask_adjustment_widgets():
            widget.blockSignals(True)
            widget.setValue(0)
            widget.blockSignals(False)
        target_key = self.mask_adjust_target_key()
        adjustments = self.preview_state.mask_adjustments_by_target
        if adjustments is not None:
            adjustments[target_key] = self.default_mask_adjustment_values(target_key)
        self.preview_state.mask_adjustment_last_preview_values.pop(target_key, None)
        state = (
            self.capture_preview_layer_control_state() if hasattr(self, "capture_preview_layer_control_state") else {}
        )
        self.remove_adjusted_mask_preview_layer(target_key, preserved_state=state)
        self.update_mask_adjustment_status()

    # Reuse the adjusted layer for this mask instead of adding a new copy each time.
    def adjusted_preview_layer_key(self, target_key: str) -> str:
        return f"{target_key}__adjusted_preview"

    def adjusted_preview_layer_color(self, colors: list[str]) -> str:
        """Choose a distinct adjustment color, preferring red then purple."""
        normalize = getattr(self, "normalize_preview_color_hex", lambda value: str(value).upper())
        used = {normalize(color).upper() for color in colors if str(color or "").strip()}
        for color in ADJUSTED_MASK_COLOR_PRIORITIES:
            if color not in used:
                return color
        fallback = getattr(self, "fallback_preview_color_for_index", None)
        return str(fallback(len(colors), used)) if callable(fallback) else ADJUSTED_MASK_COLOR_PRIORITIES[0]

    # Clamp every metadata list to actual channel count before slicing model data;
    # interrupted or older preview loads can otherwise leave stale extra entries.
    def preview_layer_metadata_for_channel_count(
        self, channel_count: int
    ) -> tuple[
        list[str],
        list[str],
        list[str],
        list[str],
        list[float],
    ]:
        channel_count = max(0, int(channel_count))
        labels = list(self.preview_state.current_labels or [])[:channel_count]
        roles = list(self.preview_state.current_layer_roles or [])[:channel_count]
        keys = list(self.preview_state.current_layer_keys or [])[:channel_count]
        colors = list(self.preview_state.overlay_colors or [])[:channel_count]
        opacities = list(self.preview_state.overlay_opacities or [])[:channel_count]

        while len(labels) < channel_count:
            labels.append(f"Layer {len(labels) + 1}")
        while len(roles) < channel_count:
            roles.append("image")
        while len(keys) < channel_count:
            keys.append("")
        while len(colors) < channel_count:
            colors.append(DEFAULT_PREVIEW_COLOR)
        while len(opacities) < channel_count:
            opacities.append(1.0)

        return labels, roles, keys, colors, opacities

    # Invalidate normalized pages and render revisions together because adding or
    # removing a channel changes both cached pixels and scene-layer indexing.
    def invalidate_mask_adjustment_preview_cache(self):
        self.preview_state.normalized_page_cache = {}
        self.preview_state.render_revision = int(self.preview_state.render_revision or 0) + 1

    # Remove generated layers by stable key rather than label because source and
    # adjusted layers may intentionally share similar display names.
    def remove_adjusted_mask_preview_layer(
        self,
        target_key: str | None = None,
        *,
        rebuild: bool = True,
        preserved_state: dict | None = None,
    ) -> bool:
        model = self.preview_state.tiff_model
        if not model:
            return False
        labels, roles, keys, colors, opacities = self.preview_layer_metadata_for_channel_count(
            int(model["data"].shape[2])
        )

        if target_key:
            remove_keys = {self.adjusted_preview_layer_key(target_key)}
        else:
            remove_keys = {key for key in keys if str(key).endswith("__adjusted_preview")}
        remove_indices = [index for index, key in enumerate(keys) if key in remove_keys]
        if not remove_indices:
            return False

        data = np.asarray(model["data"])
        remove_index_set = set(remove_indices)
        keep_indices = [index for index in range(data.shape[2]) if index not in remove_index_set]
        model["data"] = data[:, :, keep_indices, :, :]
        self.preview_state.current_layer_keys = [keys[index] if index < len(keys) else "" for index in keep_indices]
        self.preview_state.current_layer_roles = [roles[index] if index < len(roles) else "image" for index in keep_indices]
        self.preview_state.current_labels = [
            labels[index] if index < len(labels) else f"Layer {index + 1}" for index in keep_indices
        ]
        self.preview_state.overlay_colors = [
            colors[index] if index < len(colors) else DEFAULT_PREVIEW_COLOR for index in keep_indices
        ]
        self.preview_state.overlay_opacities = [
            opacities[index] if index < len(opacities) else 1.0 for index in keep_indices
        ]
        self.invalidate_mask_adjustment_preview_cache()
        if rebuild:
            self.set_preview_layer_controls(
                list(self.preview_state.current_labels),
                preserved_state=preserved_state,
            )
            self.rebuild_tiff_preview_pages()
        return True

    # Show the adjusted mask in a separate layer for comparison before saving its settings.
    def preview_current_mask_adjustments(self):
        with self.preview_work_busy("Applying adjustments…"):
            self._preview_current_mask_adjustments()

    def _preview_current_mask_adjustments(self):
        target_key = self.mask_adjust_target_key()
        layer_index = self._weka_layer_index(target_key)
        if layer_index is None:
            QMessageBox.information(self, "Mask adjustments", "Select a mask layer first.")
            return
        model = self.preview_state.tiff_model
        if not model:
            QMessageBox.information(self, "Mask adjustments", "Open a preview image first.")
            return
        data = np.asarray(model["data"])
        if data.ndim != 5 or not 0 <= layer_index < data.shape[2]:
            QMessageBox.information(self, "Mask adjustments", "The selected mask layer is not available.")
            return

        self.save_current_mask_adjustment_values()
        values = self.mask_adjustment_values()
        if not self.mask_adjustment_is_active():
            state = (
                self.capture_preview_layer_control_state()
                if hasattr(self, "capture_preview_layer_control_state") else {}
            )
            self.remove_adjusted_mask_preview_layer(target_key, preserved_state=state)
            self.preview_state.mask_adjustment_last_preview_values.pop(target_key, None)
            self.update_mask_adjustment_status()
            return
        adjusted = np.zeros((data.shape[0], data.shape[1], data.shape[3], data.shape[4]), dtype=data.dtype)
        for t_index in range(data.shape[0]):
            for z_index in range(data.shape[1]):
                adjusted[t_index, z_index] = adjust_mask_image(
                    data[t_index, z_index, layer_index],
                    dx=values["dx"],
                    dy=values["dy"],
                    grow_px=values["grow_px"],
                    min_size=values["min_size"],
                    fill_holes_area=values["fill_holes_area"],
                )
                self.pulse_preview_work_busy_icon()

        state = (
            self.capture_preview_layer_control_state() if hasattr(self, "capture_preview_layer_control_state") else {}
        )
        self.remove_adjusted_mask_preview_layer(target_key, rebuild=False)
        model = self.preview_state.tiff_model
        if not model:
            return
        data = np.asarray(model["data"])
        model["data"] = np.concatenate([data, adjusted[:, :, np.newaxis, :, :]], axis=2)

        labels, roles, keys, colors, opacities = self.preview_layer_metadata_for_channel_count(int(data.shape[2]))
        source_label = labels[layer_index] if 0 <= layer_index < len(labels) else f"Layer {layer_index}"
        labels.append(source_label)
        roles.append("adjusted_weka_mask")
        adjusted_key = self.adjusted_preview_layer_key(target_key)
        keys.append(adjusted_key)
        saved_adjusted_color = str(state.get("colors", {}).get(adjusted_key, "")) if state else ""
        adjusted_color = saved_adjusted_color or self.adjusted_preview_layer_color(colors)
        colors.append(adjusted_color)
        opacities.append(1.0)
        if state:
            state.setdefault("visibility", {})[adjusted_key] = bool(state.get("visibility", {}).get(adjusted_key, True))
            state.setdefault("colors", {})[adjusted_key] = adjusted_color
            state.setdefault("opacities", {})[adjusted_key] = float(state.get("opacities", {}).get(adjusted_key, 1.0))
        self.preview_state.current_labels = labels
        self.preview_state.current_layer_roles = roles
        self.preview_state.current_layer_keys = keys
        self.preview_state.overlay_colors = colors
        self.preview_state.overlay_opacities = opacities
        self.invalidate_mask_adjustment_preview_cache()
        self.set_preview_layer_controls(labels, preserved_state=state)
        self.rebuild_tiff_preview_pages()
        preview_values = self.preview_state.mask_adjustment_last_preview_values
        if preview_values is None:
            preview_values = {}
            self.preview_state.mask_adjustment_last_preview_values = preview_values
        preview_values[target_key] = dict(values)
        self.update_mask_adjustment_status()
