"""Overlay TIFF/PNG export helpers for masks, channels, and QC artifacts.

This module creates lossless multi-layer overlay stacks and lightweight PNG
previews while writing sidecar metadata that the GUI can use later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from skimage.measure import find_contours

from cellonaut.artifact_naming import portable_component
from cellonaut.colors import normalize_rgb_hex
from cellonaut.io.fiji_tiff import write_fiji_channel_stack
from cellonaut.io.nd2_import import infer_channel_color_hex
from cellonaut.io.image_io import imageplus_pixels_to_numpy_2d, imageplus_to_numpy_2d
from cellonaut.io.imagej_runtime import get_java_classes
from cellonaut.io.writers import save_imagej_png, write_text, write_tiff
from cellonaut.results.artifacts import record_artifact
from cellonaut.masks.roi_rasterization import imagej_roi_to_mask_image
from cellonaut.pipeline.roi_defs import make_class_roi_def, parse_class_roi_key


def _artifact_file_id(cfg: Any, result_id: str) -> str:
    variant = str(getattr(cfg, "output_variant", "") or "").strip()
    return f"{result_id}_{portable_component(variant)}" if variant else result_id


# Look up class-specific masks through their parent image definition.
def _get_image_def(cfg: Any, key: str):
    class_info = parse_class_roi_key(key)
    if class_info is not None:
        base_key, class_index = class_info
        base_def = _get_image_def(cfg, base_key)
        return make_class_roi_def(base_def, class_index)

    for img in cfg.images:
        if img.key == key:
            return img
    raise ValueError(f"Unknown image key: {key}")


# Convert Cellpose masks and ImageJ ROIs to NumPy masks and check their dimensions.
def _mask_array_from_roi_or_extra(
    base_img, roi, extra_mask: np.ndarray | None, label_shape: tuple[int, int]
) -> np.ndarray:
    if extra_mask is not None:
        mask_arr = np.asarray(extra_mask) > 0
    else:
        mask_imp = imagej_roi_to_mask_image(base_img, roi, "mask")
        try:
            mask_arr = imageplus_to_numpy_2d(mask_imp) > 0
        finally:
            try:
                mask_imp.close()
            except Exception:
                pass

    if tuple(mask_arr.shape[:2]) != label_shape:
        raise ValueError(f"Mask shape {mask_arr.shape[:2]} does not match base image shape {label_shape}")
    return mask_arr


# Choose a mask intensity that remains visible without forcing source channels into an 8-bit range.
def _max_value_for_dtype(dtype: np.dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return np.array(1.0, dtype=dtype).item()
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    return np.iinfo(np.uint16).max


# Match mask brightness to the source data so lossless stacks display sensibly in Fiji by default.
def _visible_mask_value(source_arrays: list[np.ndarray], output_dtype: np.dtype):
    finite_maxes: list[float] = []
    for arr in source_arrays:
        values = np.asarray(arr)
        if values.size == 0:
            continue
        if np.issubdtype(values.dtype, np.floating):
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_maxes.append(float(np.max(finite)))
        else:
            finite_maxes.append(float(np.max(values)))

    if finite_maxes:
        source_max = max(finite_maxes)
        if source_max > 0:
            return np.array(source_max, dtype=output_dtype).item()

    return _max_value_for_dtype(output_dtype)


# Restrict generated names to portable characters because result folders are shared across three platforms.
def _safe_layer_name(text: str) -> str:
    return portable_component(text)


# Respect microscopy metadata first and infer familiar channel colors only when none were configured.
def _display_color_for_image_def(image_def: Any, label: str = "") -> str:
    explicit = normalize_rgb_hex(getattr(image_def, "display_color", ""))
    if explicit:
        return explicit
    channel_name = str(label or getattr(image_def, "label", "") or "")
    folder_name = str(getattr(image_def, "folder_name", "") or "")
    return infer_channel_color_hex(channel_name, folder_name)


# Start masks with high-contrast colors that remain distinct from common fluorescence channels.
def _mask_color_palette() -> list[str]:
    return [
        "#00FFFF",
        "#FFFF00",
        "#FFA500",
        "#FFC0CB",
        "#8000FF",
        "#B0B0B0",
        "#FF00FF",
        "#0000FF",
        "#00FF00",
        "#FF0000",
    ]


# Avoid duplicate layer colors because color is an important identity cue in dense overlays.
def _choose_mask_color(preferred: str, index: int, avoid_colors: set[str]) -> str:
    avoid = {normalize_rgb_hex(color) for color in avoid_colors}
    avoid.discard("")
    normalized_preferred = normalize_rgb_hex(preferred)
    if normalized_preferred and normalized_preferred not in avoid:
        return normalized_preferred

    palette = _mask_color_palette()
    for offset in range(len(palette)):
        candidate = palette[(int(index) + offset) % len(palette)]
        if candidate not in avoid:
            return candidate

    for offset in range(64):
        value = (int(index) * 53 + offset * 37) % 256
        candidate = "#{:02X}{:02X}{:02X}".format(
            255 - value,
            (80 + value * 3) % 256,
            (160 + value * 5) % 256,
        )
        if candidate not in avoid:
            return candidate
    return normalized_preferred or "#00FFFF"


# Resolve a mask back to its raw channel so overlays use a biological channel name rather than a processing name.
def _source_image_def_for_mask(cfg: Any, mask_def: Any):
    mask_folder = str(getattr(mask_def, "folder_name", "") or "").strip()
    if not mask_folder:
        return None

    for candidate in getattr(cfg, "images", []) or []:
        if getattr(candidate, "key", "") == getattr(mask_def, "key", ""):
            continue
        candidate_folder = str(getattr(candidate, "folder_name", "") or "").strip()
        if candidate_folder != mask_folder or not _same_stack_channel(candidate, mask_def):
            continue
        if getattr(candidate, "combined_mask_source_keys", None):
            continue
        return candidate
    return None


# Missing stack indices mean separate-file input, where matching folder names already identify the channel.
def _same_stack_channel(left: Any, right: Any) -> bool:
    left_layer = getattr(left, "stack_channel_index", None)
    right_layer = getattr(right, "stack_channel_index", None)
    if left_layer in (None, "") or right_layer in (None, ""):
        return True
    return str(left_layer) == str(right_layer)


# Infer mask rows from configured behavior because older runtime objects do not carry a dedicated kind field.
def _looks_like_mask_definition(image_def: Any) -> bool:
    if getattr(image_def, "model_path", None) is not None:
        return True
    if getattr(image_def, "combined_mask_source_keys", None):
        return True
    label = str(getattr(image_def, "label", "") or "").lower()
    return "mask" in label


# Label source layers consistently even when the required image was opened through a mask definition.
def _overlay_source_layer_label(cfg: Any, image_def: Any) -> str:
    if _looks_like_mask_definition(image_def):
        return _source_channel_label_for_mask(cfg, image_def)
    return str(getattr(image_def, "label", "") or getattr(image_def, "folder_name", "") or "")


# Prefer the raw source channel label so users can relate a mask directly to the image it came from.
def _source_channel_label_for_mask(cfg: Any, mask_def: Any) -> str:
    mask_folder = str(getattr(mask_def, "folder_name", "") or "").strip()
    if not mask_folder:
        return str(getattr(mask_def, "label", "") or "")

    source_def = _source_image_def_for_mask(cfg, mask_def)
    if source_def is not None:
        return str(getattr(source_def, "label", "") or mask_folder)

    return str(getattr(mask_def, "label", "") or mask_folder)


# Metadata stays beside the overlay collection rather than mixing JSON files into the TIFF browser folder.
def _sidecar_dir_for_media_dir(out_path: Path) -> Path:
    out_path = Path(out_path)
    if out_path.name in {"TIFF Overlays", "TIFF Outlines"}:
        return out_path.parent / "JSON"
    return out_path


# Preserve native source intensities in a layered TIFF; the preview can color layers without altering measurements.
def _save_lossless_overlay_stack(
    base_img,
    roi_map: dict[str, Any],
    roi_keys: list[str],
    out_path: Path,
    result_id: str,
    base_label: str,
    cfg: Any,
    log_func: Callable[[str], None],
    extra_mask_map: dict[str, np.ndarray] | None = None,
    binary_out_path: Path | None = None,
    image_map: dict[str, Any] | None = None,
    write_overlay: bool = True,
) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    binary_out_path = binary_out_path or out_path
    binary_out_path.mkdir(parents=True, exist_ok=True)
    extra_mask_map = extra_mask_map or {}

    roi_keys = [k for k in roi_keys if (roi_map.get(k) is not None or k in extra_mask_map)]
    if not roi_keys:
        log_func(f"[{result_id}] Overlay stack skipped: no valid mask keys")
        return

    base_arr = imageplus_pixels_to_numpy_2d(base_img)
    if base_arr is None:
        base_arr = imageplus_to_numpy_2d(base_img)
    base_arr = np.asarray(base_arr)
    base_shape = (int(base_arr.shape[0]), int(base_arr.shape[1]))

    used_labels: list[str] = []
    used_colors_hex: list[str] = []
    used_roles: list[str] = []
    used_keys: list[str] = []
    stack_layers: list[np.ndarray] = []

    source_arrays = [(base_label, "", "#FFFFFF", base_arr)]
    if image_map:
        source_arrays = []
        seen_labels = set()
        for img_def in cfg.images:
            img = image_map.get(img_def.key)
            if img is None:
                continue
            try:
                arr = imageplus_pixels_to_numpy_2d(img)
                if arr is None:
                    arr = imageplus_to_numpy_2d(img)
                arr = np.asarray(arr)
            except Exception as exc:
                log_func(f"[WARN] [{result_id}] Skipping {img_def.label} in combined overlay: {exc}")
                continue
            if tuple(arr.shape[:2]) != base_shape:
                log_func(
                    f"[WARN] [{result_id}] Skipping {img_def.label} in combined overlay: "
                    f"image size {arr.shape[:2][::-1]} does not match base {base_shape[::-1]}"
                )
                continue
            display_label = _overlay_source_layer_label(cfg, img_def)
            if img_def.label in seen_labels or any(
                existing_label == display_label and np.array_equal(existing_arr, arr)
                for existing_label, _key, _color, existing_arr in source_arrays
            ):
                continue
            seen_labels.add(img_def.label)
            source_arrays.append(
                (
                    display_label,
                    str(img_def.key),
                    _display_color_for_image_def(img_def, display_label),
                    arr,
                )
            )

        if not source_arrays:
            source_arrays = [(base_label, "", "#FFFFFF", base_arr)]

    output_dtype = np.dtype(np.result_type(*[np.asarray(arr).dtype for _label, _key, _color, arr in source_arrays]))
    if not (np.issubdtype(output_dtype, np.integer) or np.issubdtype(output_dtype, np.floating)):
        output_dtype = np.dtype(np.uint16)
    mask_value = _visible_mask_value(
        [np.asarray(arr) for _label, _key, _color, arr in source_arrays],
        output_dtype,
    )

    for label, key, color_hex, arr in source_arrays:
        stack_layers.append(np.asarray(arr, dtype=output_dtype))
        used_labels.append(label)
        used_colors_hex.append(color_hex)
        used_roles.append("image")
        used_keys.append(str(key))

    image_layer_colors = {normalized for color in used_colors_hex if (normalized := normalize_rgb_hex(color))}
    mask_layer_colors: set[str] = set()

    for idx, key in enumerate(roi_keys):
        if key in extra_mask_map:
            if key == "__whole_cell_mask__":
                label_name = "WholeCellMask"
                layer_role = "cellpose_mask"
            elif key == "__flagged_cell_mask__":
                label_name = "FlaggedCells"
                layer_role = "filtered_cells"
            else:
                label_name = str(key)
                layer_role = "mask"
            mask_arr = _mask_array_from_roi_or_extra(base_img, None, extra_mask_map[key], base_shape)
        else:
            img_def = _get_image_def(cfg, key)
            # The layer represents the configured mask, not its source image.
            # Keep this label identical to Channels and relationship controls so
            # an image named "Hmg2" and its mask named "Hmg2 mask" stay distinct.
            label_name = str(getattr(img_def, "label", "") or _source_channel_label_for_mask(cfg, img_def))
            layer_role = "weka_mask"
            mask_arr = _mask_array_from_roi_or_extra(base_img, roi_map.get(key), None, base_shape)
        layer = np.where(mask_arr, mask_value, 0).astype(output_dtype)
        stack_layers.append(layer)
        used_labels.append(label_name)
        used_roles.append(layer_role)
        used_keys.append(str(key))
        avoid_colors = set(image_layer_colors) | set(mask_layer_colors)
        if key == "__flagged_cell_mask__":
            color_hex = _choose_mask_color("#FF0000", idx, avoid_colors)
        elif key in extra_mask_map:
            color_hex = _choose_mask_color("", idx, avoid_colors)
        else:
            color_hex = _choose_mask_color("", idx, avoid_colors)
        mask_layer_colors.add(color_hex)
        used_colors_hex.append(color_hex)

        safe_mask = _safe_layer_name(label_name)
        binary_path = binary_out_path / f"{_artifact_file_id(cfg, result_id)}_{safe_mask}_binary.tif"
        write_tiff(
            binary_path,
            np.multiply(mask_arr.astype(np.uint8), np.uint8(255)),
            photometric="minisblack",
        )

        record_artifact(
            binary_path,
            sample=result_id,
            target=str(key),
            label=label_name,
            kind="binary_mask",
            cell_mask=(
                str(getattr(cfg, "cell_segmentation_mask_source", "") or "")
                if key in {"__whole_cell_mask__", "__flagged_cell_mask__"}
                else ""
            ),
        )

    if not write_overlay:
        log_func(f"[{result_id}] Saved final binary masks; review overlay export is disabled")
        return

    stack_arr = np.stack(stack_layers, axis=0)

    safe_base = _safe_layer_name(base_label)
    file_id = _artifact_file_id(cfg, result_id)
    out_file = out_path / f"{file_id}_{safe_base}_combined_overlay.tif"
    write_fiji_channel_stack(out_file, stack_arr, used_labels, used_colors_hex)

    sidecar = {
        "type": "cellonaut_overlay_preview",
        "data_policy": "lossless_source_channels",
        "base_label": base_label,
        "layer_labels": used_labels,
        "layer_colors": used_colors_hex,
        "layer_roles": used_roles,
        "layer_keys": used_keys,
        "dtype": str(stack_arr.dtype),
        "compression": "none",
        "mask_intensity_policy": "base_image_data_max",
    }
    sidecar_dir = _sidecar_dir_for_media_dir(out_path)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_dir / f"{file_id}_{safe_base}_combined_overlay.json"
    write_text(sidecar_path, json.dumps(sidecar, indent=2), encoding="utf-8")
    target_key = str(getattr(cfg, "source_image_key", "") or next((image.key for image in cfg.images if image.label == base_label), base_label))
    target_def = _get_image_def(cfg, target_key)
    record_artifact(
        out_file,
        sample=result_id,
        target=target_key,
        label=str(getattr(target_def, "label", base_label)),
        kind="combined_overlay",
        sidecar=sidecar_path,
        cell_mask=str(getattr(cfg, "cell_segmentation_mask_source", "") or ""),
    )

    log_func(f"[{result_id}] Saved overlay stack: {out_file.name}")


# The public wrapper accepts absent base images because overlays are optional run artifacts.
def save_general_overlay_stack(
    base_img,
    roi_map: dict[str, Any],
    roi_keys: list[str],
    out_path: Path,
    result_id: str,
    base_label: str,
    cfg: Any,
    log_func: Callable[[str], None],
    extra_mask_map: dict[str, np.ndarray] | None = None,
    binary_out_path: Path | None = None,
    image_map: dict[str, Any] | None = None,
    write_overlay: bool = True,
) -> None:
    if base_img is None:
        log_func(f"[{result_id}] Overlay stack skipped: base image is None")
        return

    if write_overlay:
        log_func(f"[{result_id}] Building overlay stack...")
    else:
        log_func(f"[{result_id}] Saving final binary masks without a review overlay...")
    _save_lossless_overlay_stack(
        base_img=base_img,
        roi_map=roi_map,
        roi_keys=roi_keys,
        out_path=out_path,
        result_id=result_id,
        base_label=base_label,
        cfg=cfg,
        log_func=log_func,
        extra_mask_map=extra_mask_map,
        binary_out_path=binary_out_path,
        image_map=image_map,
        write_overlay=write_overlay,
    )


# Flatten only the review PNG so annotations are easy to inspect while the TIFF remains quantitatively lossless.
def save_general_flat_qc_overlay(
    base_img,
    roi_map: dict[str, Any],
    roi_keys: list[str],
    out_path: Path,
    result_id: str,
    warning_text: str,
    base_label: str,
    cfg: Any,
    log_func: Callable[[str], None],
    extra_mask_map: dict[str, np.ndarray] | None = None,
):
    classes = get_java_classes()
    Duplicator = classes["Duplicator"]
    Overlay = classes["Overlay"]
    TextRoi = classes["TextRoi"]
    Font = classes["Font"]
    Color = classes["Color"]
    FileSaver = classes["FileSaver"]
    IJ = classes["IJ"]

    if base_img is None:
        log_func(f"[{result_id}] Flat review overlay skipped: base image is None")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    extra_mask_map = extra_mask_map or {}

    roi_keys = [k for k in roi_keys if (roi_map.get(k) is not None or k in extra_mask_map)]
    if not roi_keys:
        log_func(f"[{result_id}] Flat filter overlay skipped: no valid mask keys")
        return

    flat = Duplicator().run(base_img)
    flattened = None

    try:
        IJ.run(flat, "8-bit", "")
        overlay = Overlay()

        colors = [
            Color.magenta,
            Color.blue,
            Color.green,
            Color.red,
            Color.cyan,
            Color.yellow,
            Color.orange,
            Color.pink,
        ]

        legend_items = []

        for idx, key in enumerate(roi_keys):
            color = Color.red if key == "__flagged_cell_mask__" else colors[idx % len(colors)]

            if key in extra_mask_map:
                mask_arr = np.asarray(extra_mask_map[key]) > 0
                contours = find_contours(mask_arr.astype(np.uint8), level=0.5)

                for contour in contours:
                    xs = np.array([float(p[1]) for p in contour], dtype=np.float32)
                    ys = np.array([float(p[0]) for p in contour], dtype=np.float32)

                    if len(xs) >= 2:
                        roi_poly = classes["PolygonRoi"](
                            xs,
                            ys,
                            len(xs),
                            classes["PolygonRoi"].POLYLINE,
                        )
                        roi_poly.setStrokeColor(color)
                        roi_poly.setStrokeWidth(2)
                        overlay.add(roi_poly)

                if key == "__whole_cell_mask__":
                    legend_items.append("Cellpose whole-cell mask")
                elif key == "__flagged_cell_mask__":
                    legend_items.append("Flagged out-of-range cells")
                else:
                    legend_items.append(str(key))
                continue

            roi = roi_map.get(key)
            if roi is None:
                continue

            img_def = _get_image_def(cfg, key)

            r = roi.clone()
            r.setStrokeColor(color)
            r.setStrokeWidth(2)
            overlay.add(r)

            legend_items.append(img_def.label)

        title = TextRoi(10, 10, f"{result_id} | base={base_label}", Font("SansSerif", Font.BOLD, 18))
        title.setStrokeColor(Color.white)
        overlay.add(title)

        if legend_items:
            legend_text = " | ".join(legend_items)
            legend = TextRoi(10, 35, legend_text, Font("SansSerif", Font.PLAIN, 12))
            legend.setStrokeColor(Color.white)
            overlay.add(legend)

        if warning_text and warning_text != "OK":
            warn = TextRoi(10, 55, warning_text, Font("SansSerif", Font.PLAIN, 12))
            warn.setStrokeColor(Color.yellow)
            overlay.add(warn)

        flat.setOverlay(overlay)
        IJ.run(flat, "Flatten", "")
        flattened = IJ.getImage()

        safe_base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base_label)
        out_file = out_path / f"{_artifact_file_id(cfg, result_id)}_{safe_base}_qc.png"
        if not save_imagej_png(out_file, flattened, FileSaver):
            raise OSError(f"Could not save flat review overlay: {out_file}")

        target_key = str(getattr(cfg, "source_image_key", "") or next((image.key for image in cfg.images if image.label == base_label), base_label))
        target_def = _get_image_def(cfg, target_key)
        record_artifact(
            out_file,
            sample=result_id,
            target=target_key,
            label=str(getattr(target_def, "label", base_label)),
            kind="overlay_png",
            cell_mask=str(getattr(cfg, "cell_segmentation_mask_source", "") or ""),
        )
        log_func(f"[{result_id}] Saved generalized flat review overlay: {out_file.name}")

    finally:
        try:
            if flattened is not None:
                flattened.close()
        except Exception:
            pass

        try:
            flat.close()
        except Exception:
            pass
