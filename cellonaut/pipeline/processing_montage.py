"""PNG montage previews for configured image-processing recipes."""

from __future__ import annotations

import json
import math
from pathlib import Path

from cellonaut.results.artifacts import record_artifact, ArtifactResolver
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import tifffile

from cellonaut.artifact_naming import portable_component
from cellonaut.config.defaults import (
    IMAGE_PROCESSING_SCOPE_MEASUREMENT,
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
)
from cellonaut.io.writers import save_pil_image
from cellonaut.io.image_io import imageplus_to_numpy_2d, normalize_numpy_image
from cellonaut.io.imagej_runtime import get_java_classes
from cellonaut.masks.roi_rasterization import imagej_roi_to_mask_image
from cellonaut.masks.cell_qc import cell_label_outline_mask
from cellonaut.pipeline.image_processing import (
    MEASUREMENT_TRANSFORM_STEP_TYPES,
    apply_imagej_processing_recipe_copy,
    imagej_processing_recipe_tiles,
)
from cellonaut.pipeline.planning import measurement_background_requests
from cellonaut.pipeline.roi_defs import parse_class_roi_key

PROCESSING_MONTAGE_MAX_TILE_SIDE = 1024
PROCESSING_MONTAGE_METADATA_KEY = "CellonautMontage"


# Result identifiers become filenames on all three platforms, so replace unsafe
# characters without changing readable channel and mask labels more than needed.
def _safe_name(text: str) -> str:
    return portable_component(text, fallback="image", strip=True)


# Display scaling should compare only scalar image planes; RGB overlays and empty
# arrays would otherwise distort the shared intensity range.
def _finite_values(arrays: Iterable[np.ndarray]) -> np.ndarray:
    values: list[np.ndarray] = []
    for arr in arrays:
        data = np.asarray(arr)
        if data.ndim != 2 or data.size == 0:
            continue
        if np.issubdtype(data.dtype, np.floating):
            finite = data[np.isfinite(data)]
        else:
            finite = data.reshape(-1)
        if finite.size:
            values.append(finite.astype(np.float64, copy=False))
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(values)


# Percentile limits keep processing stages visually comparable while preventing a
# few hot pixels from making the rest of a montage unreadably dark.
def _display_limits(arrays: Sequence[np.ndarray]) -> tuple[float, float] | None:
    values = _finite_values(arrays)
    if values.size == 0:
        return None
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.8))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return None
    return low, high


# Montage conversion is display-only. Preserve existing RGB colors, but normalize
# scalar scientific data into a browser-friendly eight-bit representation.
def _to_uint8(arr: np.ndarray, limits: tuple[float, float] | None = None) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        if data.dtype == np.uint8:
            return data[..., :3]
        data = data[..., :3].astype(np.float32, copy=False)
        max_value = 255.0 if float(np.nanmax(data)) > 1.0 else 1.0
        return np.clip(data / max_value * 255.0, 0, 255).astype(np.uint8)

    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if limits is None:
        finite = data[np.isfinite(data)]
        if finite.size:
            low = float(np.min(finite))
            high = float(np.max(finite))
        else:
            low = 0.0
            high = 1.0
    else:
        low, high = limits
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = (data - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


# Raw and background-subtracted tiles should retain their native bit-depth scale;
# auto-stretching each one independently could exaggerate small differences.
def _dtype_display_limits(arr: np.ndarray) -> tuple[float, float] | None:
    data = np.asarray(arr)
    if data.dtype == np.bool_:
        return 0.0, 1.0
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(data.dtype, np.floating):
        return 0.0, 1.0
    return None


# Limit tile size to reduce memory and file size while preserving image proportions.
def _resize_for_montage(image: Image.Image, max_tile_side: int) -> Image.Image:
    width, height = image.size
    max_side = max(width, height)
    if max_side <= max_tile_side:
        return image
    scale = float(max_tile_side) / float(max_side)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


# Place every stage on a fixed canvas so different source dimensions cannot make
# the montage grid uneven or move labels between rows.
def _tile_image(
    label: str,
    arr: np.ndarray,
    *,
    limits: tuple[float, float] | None,
    max_tile_side: int,
    tile_size: tuple[int, int],
) -> Image.Image:
    tile_limits = _tile_display_limits(label, arr, limits)
    data = _to_uint8(arr, limits=tile_limits)
    if data.ndim == 2:
        body = Image.fromarray(data, mode="L").convert("RGB")
    else:
        body = Image.fromarray(data, mode="RGB")
    body = _resize_for_montage(body, max_tile_side)

    canvas_width, canvas_height = tile_size
    label_height = 34
    tile = Image.new("RGB", (canvas_width, canvas_height + label_height), (18, 18, 18))
    x = (canvas_width - body.width) // 2
    y = label_height + (canvas_height - body.height) // 2
    tile.paste(body, (x, y))

    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    clean_label = str(label or "").replace("\n", " ")
    if len(clean_label) > 72:
        clean_label = clean_label[:69] + "..."
    draw.rectangle((0, 0, canvas_width, label_height), fill=(34, 34, 34))
    draw.text((8, 10), clean_label, fill=(245, 245, 245), font=font)
    return tile


# Compute one grid and shared display context for the complete recipe so tiles can
# be compared directly rather than appearing as unrelated auto-scaled images.
def save_montage_png(
    tiles: Sequence[tuple[str, np.ndarray]],
    out_file: Path,
    *,
    max_tile_side: int = PROCESSING_MONTAGE_MAX_TILE_SIDE,
    columns: int | None = None,
) -> Path | None:
    valid_tiles = [(str(label), np.asarray(arr)) for label, arr in tiles if arr is not None]
    if not valid_tiles:
        return None

    limits = _display_limits([arr for _label, arr in valid_tiles])
    resized_sizes = []
    for _label, arr in valid_tiles:
        data = _to_uint8(arr, limits=_tile_display_limits(_label, arr, limits))
        pil = Image.fromarray(data, mode="L" if data.ndim == 2 else "RGB")
        pil = _resize_for_montage(pil, max_tile_side)
        resized_sizes.append(pil.size)

    tile_width = max(width for width, _height in resized_sizes)
    tile_height = max(height for _width, height in resized_sizes)
    columns = int(columns or math.ceil(math.sqrt(len(valid_tiles))))
    columns = max(1, min(columns, len(valid_tiles)))
    rows = int(math.ceil(len(valid_tiles) / columns))

    label_height = 34
    gap = 4
    montage = Image.new(
        "RGB",
        (
            columns * tile_width + (columns - 1) * gap,
            rows * (tile_height + label_height) + (rows - 1) * gap,
        ),
        (0, 0, 0),
    )

    for index, (label, arr) in enumerate(valid_tiles):
        tile = _tile_image(
            label,
            arr,
            limits=limits,
            max_tile_side=max_tile_side,
            tile_size=(tile_width, tile_height),
        )
        row = index // columns
        col = index % columns
        montage.paste(tile, (col * (tile_width + gap), row * (tile_height + label_height + gap)))

    tile_metadata = []
    for index, ((label, _arr), (body_width, body_height)) in enumerate(zip(valid_tiles, resized_sizes)):
        row = index // columns
        col = index % columns
        tile_x = col * (tile_width + gap)
        tile_y = row * (tile_height + label_height + gap)
        tile_metadata.append(
            {
                "label": label,
                "x": tile_x + (tile_width - body_width) // 2,
                "y": tile_y + label_height + (tile_height - body_height) // 2,
                "width": body_width,
                "height": body_height,
            }
        )

    png_info = PngInfo()
    png_info.add_text(
        PROCESSING_MONTAGE_METADATA_KEY,
        json.dumps(
            {
                "version": 1,
                "columns": columns,
                "rows": rows,
                "tiles": tile_metadata,
            },
            separators=(",", ":"),
        ),
    )
    save_pil_image(montage, out_file, format="PNG", optimize=True, pnginfo=png_info)
    return out_file


# Masks, labels, probabilities, and overlays carry categorical or already-rendered
# values; only comparable grayscale processing stages should share contrast limits.
def _uses_shared_display_limits(label: str, arr: np.ndarray) -> bool:
    data = np.asarray(arr)
    if data.ndim != 2:
        return False
    lower_label = str(label or "").lower()
    if any(
        token in lower_label
        for token in (
            "mask",
            "overlay",
            "outline",
            "probability",
            "raw",
            "threshold",
            "label",
        )
    ):
        return False
    if data.dtype == np.bool_:
        return False
    if np.issubdtype(data.dtype, np.integer) and data.size:
        try:
            unique = np.unique(data)
            if unique.size <= 4 and float(np.min(unique)) >= 0 and float(np.max(unique)) <= 255:
                return False
        except Exception:
            pass
    return True


# Choose scaling by artifact meaning: probabilities need their own range, raw data
# uses dtype limits, and intermediate grayscale stages share a common range.
def _tile_display_limits(
    label: str,
    arr: np.ndarray,
    shared_limits: tuple[float, float] | None,
) -> tuple[float, float] | None:
    lower_label = str(label or "").lower()
    if "probability" in lower_label:
        return _display_limits([np.asarray(arr)])
    if "raw" in lower_label or "subtract background" in lower_label or "measurement" in lower_label:
        return _dtype_display_limits(np.asarray(arr))
    if _uses_shared_display_limits(label, arr):
        return shared_limits
    return None


# Keep Java duplication behind one lookup so rolling-ball branches cannot
# accidentally operate on and alter the raw ImagePlus.
def _duplicate_image(img: Any):
    return get_java_classes()["Duplicator"]().run(img)


# Close temporary images when montage preparation fails before returning them to the caller.
def _close_imagej_images(images: Sequence[Any]) -> None:
    for image in images:
        try:
            image.close()
        except Exception:
            pass


# Segmentation applies repeated radii sequentially, whereas measurement radii are
# a sweep from the same base image. The flag makes that scientific distinction explicit.
def _rolling_ball_tiles(
    img: Any,
    radii: Sequence[float],
    *,
    sequential: bool,
    options: Sequence[str] = (),
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray | None, list[Any]]:
    IJ = get_java_classes()["IJ"]
    temp_images: list[Any] = []
    tiles: list[tuple[str, np.ndarray]] = []
    final_arr = None

    try:
        if sequential:
            current = _duplicate_image(img)
            temp_images.append(current)
            for index, radius in enumerate(radii, start=1):
                safe_radius = max(0.0, float(radius))
                suffix = options[index - 1] if index <= len(options) else ""
                IJ.run(current, "Subtract Background...", f"rolling={safe_radius}{suffix}")
                final_arr = np.asarray(imageplus_to_numpy_2d(current)).copy()
                tiles.append((f"{index:02d} Subtract Background radius={safe_radius:g}", final_arr))
            return tiles, final_arr, temp_images

        for index, radius in enumerate(radii, start=1):
            safe_radius = max(0.0, float(radius))
            current = _duplicate_image(img)
            temp_images.append(current)
            suffix = options[index - 1] if index <= len(options) else ""
            IJ.run(current, "Subtract Background...", f"rolling={safe_radius}{suffix}")
            arr = np.asarray(imageplus_to_numpy_2d(current)).copy()
            if final_arr is None:
                final_arr = arr
            label = "Primary measurement correction" if index == 1 else "Measurement sweep"
            tiles.append((f"{index:02d} {label} radius={safe_radius:g}", arr))
        return tiles, final_arr, temp_images
    except Exception:
        _close_imagej_images(temp_images)
        raise


# Use the shared ImageJ rasterizer so montage outlines match exported binary masks exactly.
def _mask_array_from_roi(base_img: Any, roi: Any) -> np.ndarray | None:
    if roi is None:
        return None
    mask_imp = imagej_roi_to_mask_image(base_img, roi, "montage mask")
    try:
        return np.asarray(imageplus_to_numpy_2d(mask_imp)) > 0
    finally:
        try:
            mask_imp.close()
        except Exception:
            pass


# Derive a one-pixel outline with NumPy to avoid introducing another image library
# dependency solely for montage decoration.
def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones(mask.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return mask & ~eroded


# Cellpose labels need boundaries between adjacent cells as well as the outer
# foreground edge; a binary erosion would merge those distinctions.
def _label_boundary(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 2:
        labels = np.squeeze(labels)
    if labels.ndim != 2:
        return np.zeros((1, 1), dtype=bool)
    return cell_label_outline_mask(labels).astype(bool)


# Combine a light fill with a solid boundary so users can inspect both mask extent
# and the underlying image intensity in the same tile.
def _overlay_mask(base_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    gray = _to_uint8(np.asarray(base_arr))
    rgb = np.repeat(gray[..., None], 3, axis=2)
    mask = np.asarray(mask_arr, dtype=bool)
    if mask.shape[:2] != rgb.shape[:2]:
        return rgb
    rgb[mask] = (0.75 * rgb[mask] + np.array([255, 0, 0], dtype=None) * 0.25).astype(np.uint8)
    rgb[_mask_boundary(mask)] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


# Use a distinct color for whole-cell labels so Cellpose output cannot be confused
# with red Weka-mask overlays.
def _overlay_labels(base_arr: np.ndarray, label_arr: np.ndarray) -> np.ndarray:
    gray = _to_uint8(np.asarray(base_arr))
    rgb = np.repeat(gray[..., None], 3, axis=2)
    labels = np.asarray(label_arr)
    mask = labels > 0
    if mask.shape[:2] != rgb.shape[:2]:
        return rgb
    rgb[mask] = (0.75 * rgb[mask] + np.array([0, 170, 255], dtype=None) * 0.25).astype(np.uint8)
    rgb[_label_boundary(labels)] = np.array([0, 220, 255], dtype=np.uint8)
    return rgb


# ROI maps are keyed by stable pipeline IDs rather than display labels, so resolve
# the owning definition before selecting its image and export name.
def _image_def_by_key(cfg: Any, key: str) -> Any | None:
    for image_def in getattr(cfg, "images", []) or []:
        if getattr(image_def, "key", "") == key:
            return image_def
    return None


# Probability and threshold files are optional diagnostic artifacts. Missing or
# unreadable files should omit a tile rather than fail an otherwise successful run.
def _artifact_arrays(
    *,
    probability_dir: Path | None,
    threshold_dir: Path | None,
    result_id: str,
    label: str,
    class_index: int | None = None,
    target_key: str = "",
    log_func: Callable[[str], None] | None = None,
) -> list[tuple[str, np.ndarray]]:
    resolver = ArtifactResolver.load(probability_dir or threshold_dir or Path())
    if resolver is not None:
        tiles: list[tuple[str, np.ndarray]] = []
        for kind, title in (("probability", "Weka probability map"), ("threshold", "Threshold mask")):
            records = [record for record in resolver.matching(sample=result_id, target=target_key or None, kind=kind) if (target_key or record.get("label") == label) and (kind != "threshold" or class_index is None or record.get("mask") == str(class_index))]
            if not records:
                continue
            try:
                arr = np.asarray(tifffile.imread(resolver.path(records[0])))
                arr = _select_probability_layer(arr, class_index) if kind == "probability" else normalize_numpy_image(arr, stack_mode="first")
                tiles.append((title, arr))
            except (OSError, ValueError, tifffile.TiffFileError) as exc:
                if log_func is not None:
                    log_func(f"[WARN] [{result_id}] Could not load recorded montage artifact: {exc}")
        return tiles
    # Legacy folders retain their original classifier/class filename lookup.
    tiles: list[tuple[str, np.ndarray]] = []
    safe_label = _safe_name(label)
    raw_label = str(label or "")

    if probability_dir is not None and probability_dir.exists():
        probability_matches = sorted(probability_dir.glob(f"{result_id}_{raw_label}_prob_*.tif"))
        if not probability_matches and safe_label != raw_label:
            probability_matches = sorted(probability_dir.glob(f"{result_id}_{safe_label}_prob_*.tif"))
        if probability_matches:
            try:
                arr = _select_probability_layer(
                    np.asarray(tifffile.imread(probability_matches[0])),
                    class_index=class_index,
                )
                tiles.append(("Weka probability map", arr))
            except Exception as exc:
                if log_func is not None:
                    log_func(
                        f"[WARN] [{result_id}] Could not read montage probability map "
                        f"{probability_matches[0].name}: {type(exc).__name__}: {exc}"
                    )

    if threshold_dir is not None and threshold_dir.exists():
        threshold_patterns = []
        if class_index is not None:
            threshold_patterns.append(f"{result_id}_{raw_label}_class{int(class_index)}_thr_*.tif")
            if safe_label != raw_label:
                threshold_patterns.append(f"{result_id}_{safe_label}_class{int(class_index)}_thr_*.tif")
        threshold_patterns.append(f"{result_id}_{raw_label}_class*_thr_*.tif")
        if safe_label != raw_label:
            threshold_patterns.append(f"{result_id}_{safe_label}_class*_thr_*.tif")
        threshold_matches = []
        for pattern in threshold_patterns:
            threshold_matches = sorted(threshold_dir.glob(pattern))
            if threshold_matches:
                break
        if threshold_matches:
            try:
                arr = normalize_numpy_image(np.asarray(tifffile.imread(threshold_matches[0])), stack_mode="first")
                tiles.append(("Threshold mask", arr))
            except Exception as exc:
                if log_func is not None:
                    log_func(
                        f"[WARN] [{result_id}] Could not read montage threshold mask "
                        f"{threshold_matches[0].name}: {type(exc).__name__}: {exc}"
                    )

    return tiles


# Weka probability stacks are class-first, but single-class and RGB-like files may
# already be displayable; select only when the array shape clearly represents a stack.
def _select_probability_layer(arr: np.ndarray, class_index: int | None) -> np.ndarray:
    data = np.asarray(arr)
    data = np.squeeze(data)
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        return data
    if data.ndim == 3 and data.shape[-1] not in (3, 4):
        index = max(0, int(class_index or 1) - 1)
        if index < data.shape[0]:
            return normalize_numpy_image(data[index], stack_mode="first")
    return normalize_numpy_image(data, stack_mode="first")


# Build segmentation and measurement montages from the exact recipe and artifacts
# used for this sample, closing all temporary ImagePlus copies after each target.
def save_processing_montages_for_sample(
    *,
    cfg: Any,
    image_map: dict[str, Any],
    roi_map: dict[str, Any],
    export_dir: Path,
    result_id: str,
    log_func: Callable[[str], None],
    probability_dir: Path | None = None,
    threshold_dir: Path | None = None,
) -> list[Path]:
    saved: list[Path] = []

    for roi_key, roi in sorted((roi_map or {}).items()):
        if roi_key.startswith("__"):
            continue
        class_info = parse_class_roi_key(roi_key)
        base_key = class_info[0] if class_info is not None else roi_key
        class_index = int(class_info[1]) if class_info is not None else None
        image_def = _image_def_by_key(cfg, base_key)
        if image_def is None:
            continue
        img = image_map.get(image_def.key)
        if img is None:
            continue
        raw_arr = np.asarray(imageplus_to_numpy_2d(img)).copy()
        final_arr = raw_arr
        temp_images: list[Any] = []
        try:
            tiles: list[tuple[str, np.ndarray]] = [("Raw", raw_arr)]
            step_tiles, processed_arr, temp_images = imagej_processing_recipe_tiles(
                img,
                image_def,
                IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            )
            tiles.extend(step_tiles)
            if processed_arr is not None:
                final_arr = processed_arr

            roi_label = f"{image_def.label}_Class{class_index}" if class_index is not None else str(image_def.label)
            artifact_tiles = _artifact_arrays(
                probability_dir=probability_dir,
                threshold_dir=threshold_dir,
                result_id=result_id,
                label=str(image_def.label),
                class_index=class_index,
                target_key=image_def.key,
                log_func=log_func,
            )
            tiles.extend(artifact_tiles)

            mask_arr = _mask_array_from_roi(img, roi) if roi is not None else None
            if mask_arr is not None:
                has_mask_tile = any(
                    any(token in label.lower() for token in ("mask", "threshold")) for label, _arr in artifact_tiles
                )
                if not has_mask_tile:
                    tiles.append(("Mask", mask_arr.astype(np.uint8) * 255))
                tiles.append(("Final overlay", _overlay_mask(final_arr, mask_arr)))

            out_file = (
                export_dir
                / _safe_name(result_id)
                / f"{_safe_name(result_id)}_{_safe_name(roi_label)}_segmentation_montage.png"
            )
            saved_path = save_montage_png(tiles, out_file)
            if saved_path is not None:
                record_artifact(saved_path, sample=result_id, target=image_def.key, label=image_def.label, kind="montage")
                saved.append(saved_path)
                log_func(f"[{result_id}] Saved segmentation montage: {saved_path.name}")
        finally:
            _close_imagej_images(temp_images)

    for image_def in getattr(cfg, "images", []) or []:
        img = image_map.get(image_def.key)
        background_requests = [
            (radius, options)
            for radius, options in measurement_background_requests(image_def)
            if float(radius) > 0
        ]
        radii = [radius for radius, _options in background_requests]
        background_options = [options for _radius, options in background_requests]
        if img is None:
            continue
        raw_arr = np.asarray(imageplus_to_numpy_2d(img)).copy()
        temp_images: list[Any] = []
        try:
            transform_tiles, transform_arr, transform_temp = imagej_processing_recipe_tiles(
                img,
                image_def,
                IMAGE_PROCESSING_SCOPE_MEASUREMENT,
                include_step_types=MEASUREMENT_TRANSFORM_STEP_TYPES,
            )
            temp_images.extend(transform_temp)
            if not radii and not transform_tiles:
                continue

            measurement_base_img = img
            if transform_tiles and radii:
                measurement_base_img, _labels = apply_imagej_processing_recipe_copy(
                    img,
                    image_def,
                    IMAGE_PROCESSING_SCOPE_MEASUREMENT,
                    include_step_types=MEASUREMENT_TRANSFORM_STEP_TYPES,
                )
                if measurement_base_img is not img:
                    temp_images.append(measurement_base_img)

            rolling_tiles, final_arr, rolling_temp = (
                _rolling_ball_tiles(
                    measurement_base_img,
                    radii,
                    sequential=False,
                    options=background_options,
                )
                if radii
                else ([], transform_arr, [])
            )
            temp_images.extend(rolling_temp)
            if final_arr is None:
                continue
            tiles = [("Raw", raw_arr), *transform_tiles, *rolling_tiles]
            out_file = (
                export_dir
                / _safe_name(result_id)
                / f"{_safe_name(result_id)}_{_safe_name(image_def.label)}_measurement_montage.png"
            )
            saved_path = save_montage_png(tiles, out_file)
            if saved_path is not None:
                record_artifact(saved_path, sample=result_id, target=image_def.key, label=image_def.label, kind="montage")
                saved.append(saved_path)
                log_func(f"[{result_id}] Saved measurement montage: {saved_path.name}")
        finally:
            _close_imagej_images(temp_images)

    return saved


# Cellpose review needs label, outline, and flagged-cell views that do not exist in
# the ordinary image-processing recipe montage, so export them as a focused set.
def save_cellpose_montage_for_sample(
    *,
    image_map: dict[str, Any],
    cell_source_def: Any,
    source_def: Any,
    extra_overlay_masks: dict[str, np.ndarray],
    export_dir: Path,
    result_id: str,
    output_variant: str = "",
    cell_mask_key: str = "",
    log_func: Callable[[str], None],
) -> Path | None:
    labels = extra_overlay_masks.get("__whole_cell_mask__")
    if labels is None:
        return None

    source_img = image_map.get(getattr(cell_source_def, "key", ""))
    if source_img is None:
        return None

    raw_arr = np.asarray(imageplus_to_numpy_2d(source_img)).copy()
    label_arr = np.asarray(labels)
    foreground = (label_arr > 0).astype(np.uint8) * 255
    outline = _label_boundary(label_arr).astype(np.uint8) * 255
    tiles: list[tuple[str, np.ndarray]] = [
        ("Cellpose input", raw_arr),
        ("Cellpose labels", foreground),
        ("Cellpose outlines", outline),
        ("Final overlay", _overlay_labels(raw_arr, label_arr)),
    ]

    flagged = extra_overlay_masks.get("__flagged_cell_mask__")
    if flagged is not None:
        flagged_arr = np.asarray(flagged, dtype=bool)
        tiles.append(("CSV-excluded cell groups", flagged_arr.astype(np.uint8) * 255))
        tiles.append(("Flagged overlay", _overlay_mask(raw_arr, flagged_arr)))

    variant_suffix = f"_{_safe_name(output_variant)}" if str(output_variant or "").strip() else ""
    out_file = (
        export_dir
        / _safe_name(result_id)
        / f"{_safe_name(result_id)}_{_safe_name(getattr(source_def, 'label', 'Cellpose'))}{variant_suffix}_cellpose_montage.png"
    )
    saved_path = save_montage_png(tiles, out_file)
    if saved_path is not None:
        record_artifact(
            saved_path,
            sample=result_id,
            target=str(getattr(source_def, "key", source_def.label)),
            label=source_def.label,
            kind="montage",
            cell_mask=cell_mask_key,
        )
        log_func(f"[{result_id}] Saved Cellpose montage: {saved_path.name}")
    return saved_path
