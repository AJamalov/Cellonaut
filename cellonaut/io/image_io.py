"""TIFF/OME-TIFF image loading helpers used by the pipeline and preview GUI.

Microscopy stacks vary in axis order and metadata quality. These helpers keep
channel selection, Z projection, OME metadata extraction, and ImageJ fallback
logic in one place so the rest of the pipeline can work with 2-D images.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
from xml.etree.ElementTree import ParseError
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile

from cellonaut.runtime import PipelineRuntime, PipelineStage
from cellonaut.io.imagej_runtime import get_ij
from cellonaut.pipeline.discovery import (
    STRUCTURE_FLAT_TIFFS,
    STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    find_first_image_file,
    find_matching_image_file_by_stem,
    get_subfolder,
    is_supported_tiff_file,
)


# Accept the labels used by saved settings while keeping one internal mode name.
def _normalized_stack_mode(stack_mode: str | None) -> str:
    mode = str(stack_mode or "first").strip().lower()
    if mode in {"max", "max_projection", "maximum", "maximum_projection"}:
        return "max_projection"
    if mode in {"first", "single", "single_z", "slice"}:
        return "single_z"
    raise ValueError(f"Unsupported stack mode: {stack_mode!r}")


# User-facing layer and Z fields are one-based, unlike NumPy indices.
def _one_based_index(value: int | None, *, axis_size: int, label: str) -> int:
    index = 1 if value is None else int(value)
    if index < 1:
        raise ValueError(f"{label} must be >= 1, got {value}")
    if index > axis_size:
        raise ValueError(f"Requested {label} {index}, but image only has {axis_size} slice(s).")
    return index - 1


# Projection and single-slice modes use identical bounds handling for every stack axis.
def _project_stack_axis(
    arr: np.ndarray,
    axis: int,
    *,
    stack_mode: str,
    stack_index: int | None = None,
    label: str = "Z slice",
) -> np.ndarray:
    mode = _normalized_stack_mode(stack_mode)
    if mode == "max_projection":
        return np.max(arr, axis=axis)
    index = _one_based_index(stack_index, axis_size=int(arr.shape[axis]), label=label)
    return np.take(arr, index, axis=axis)


# Update axis labels whenever an array dimension is removed.
def _remove_axis(axes: str | None, axis: int) -> str | None:
    if not axes:
        return axes
    if axis < 0 or axis >= len(axes):
        return axes
    return axes[:axis] + axes[axis + 1 :]


def normalize_numpy_image(
    arr: np.ndarray,
    stack_mode: str = "first",
    stack_index: int | None = None,
) -> np.ndarray:
    """Reduce an unlabeled image array to one two-dimensional analysis plane."""

    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        arr = np.squeeze(arr[..., 0])
        if arr.ndim == 2:
            return arr
    if arr.ndim < 2:
        return arr.reshape(1, -1)

    y_size, x_size = arr.shape[-2:]
    pages = arr.reshape((-1, y_size, x_size))
    return _project_stack_axis(
        pages,
        0,
        stack_mode=stack_mode,
        stack_index=stack_index,
        label="stack slice",
    )


def ome_color_to_hex(raw_color: str | int | None) -> str:
    """Convert signed or unsigned OME color metadata to an RGB hex string."""

    if raw_color is None:
        return ""
    value = int(raw_color)
    if value < 0:
        value &= 0xFFFFFFFF

    # tifffile writes simple OME colors as 24-bit RRGGBB when alpha is absent.
    if 0 <= value <= 0xFFFFFF:
        return f"#{value:06X}"

    # OME/ImageJ metadata may also store colors as 32-bit RGBA.
    red = (value >> 24) & 0xFF
    green = (value >> 16) & 0xFF
    blue = (value >> 8) & 0xFF
    if red == green == blue == 0:
        red = (value >> 16) & 0xFF
        green = (value >> 8) & 0xFF
        blue = value & 0xFF
    return f"#{red:02X}{green:02X}{blue:02X}"


# Trust a declared C axis before considering shape-based TIFF fallbacks.
def _channel_axis_from_axes(axes: str | None) -> int | None:
    axes = str(axes or "")
    if "C" in axes:
        return axes.index("C")
    return None


# Only unlabeled TIFF page axes are treated as channel layers; explicit Z/T axes keep their meaning.
def _fallback_tiff_stack_axis(shape: tuple[int, ...], axes: str | None) -> int | None:
    if len(shape) < 3 or shape[-1] in (3, 4):
        return None
    labels = str(axes or "").upper()
    if not labels or len(labels) != len(shape):
        return 0
    return 0 if labels[0] in {"I", "Q"} else None


def select_tiff_stack_channel(
    arr: np.ndarray,
    stack_channel_index: int | None = None,
    axes: str | None = None,
    stack_z_mode: str = "max_projection",
    stack_z_index: int | None = None,
) -> np.ndarray:
    """Select a 2-D analysis plane; channel and single-Z indices are one-based.

    Declared C/Z axes take precedence over unlabeled page-stack heuristics.
    An omitted channel selects the first; Z is projected or selected according
    to stack_z_mode. T and sample-component S axes use index zero. Out-of-range
    selections raise ValueError. This selects pixels without display scaling.
    """

    arr = np.asarray(arr)
    axes = str(axes or "")
    channel_axis = _channel_axis_from_axes(axes)
    if channel_axis is not None and channel_axis >= arr.ndim:
        raise ValueError(f"TIFF axes {axes!r} do not match shape {arr.shape}")

    if stack_channel_index is not None:
        channel_index = int(stack_channel_index) - 1
        if channel_index < 0:
            raise ValueError(f"Stack layer must be >= 1, got {stack_channel_index}")

        if channel_axis is None:
            channel_axis = _fallback_tiff_stack_axis(tuple(arr.shape), axes)

        if channel_axis is None:
            if channel_index != 0:
                raise ValueError(f"TIFF has no stack/channel axis for layer {stack_channel_index}")
        else:
            if channel_index >= arr.shape[channel_axis]:
                raise ValueError(
                    f"Requested stack layer {stack_channel_index}, but TIFF only has "
                    f"{arr.shape[channel_axis]} layer(s) on axis {channel_axis}"
                )

            arr = np.take(arr, channel_index, axis=channel_axis)
            axes = _remove_axis(axes, channel_axis) or ""
    elif channel_axis is not None:
        arr = np.take(arr, 0, axis=channel_axis)
        axes = _remove_axis(axes, channel_axis) or ""

    z_axis = axes.index("Z") if "Z" in axes else None
    used_explicit_z_axis = False
    if z_axis is not None and z_axis < arr.ndim:
        used_explicit_z_axis = True
        arr = _project_stack_axis(
            arr,
            z_axis,
            stack_mode=stack_z_mode,
            stack_index=stack_z_index,
            label="Z slice",
        )
        axes = _remove_axis(axes, z_axis) or ""

    for axis_label in ("T", "S"):
        axis = axes.index(axis_label) if axis_label in axes else None
        if axis is not None and axis < arr.ndim:
            arr = np.take(arr, 0, axis=axis)
            axes = _remove_axis(axes, axis) or ""

    return normalize_numpy_image(
        arr,
        stack_mode=stack_z_mode,
        stack_index=None if used_explicit_z_axis else stack_z_index,
    )


def get_tiff_stack_info(path: Path) -> dict[str, Any]:
    """Read TIFF shape, axes, channels, and calibration without loading pixels."""

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = str(getattr(series, "axes", "") or "")
        shape = tuple(int(x) for x in series.shape)
        ome_metadata = str(getattr(tif, "ome_metadata", "") or "")

    channel_axis = _channel_axis_from_axes(axes)
    if channel_axis is None:
        channel_axis = _fallback_tiff_stack_axis(shape, axes)

    channel_count = int(shape[channel_axis]) if channel_axis is not None else 1
    channel_names: list[str] = []
    channel_colors: list[str] = []
    physical_size_x = None
    physical_size_y = None
    physical_size_unit = ""
    if ome_metadata:
        # OME metadata is useful but not guaranteed. If parsing fails, callers
        # still get shape and channel-count information from tifffile.
        try:
            root = ET.fromstring(ome_metadata)
            pixels = next(
                (element for element in root.iter() if str(element.tag).split("}")[-1] == "Pixels"),
                None,
            )
            if pixels is not None:
                physical_size_x = pixels.attrib.get("PhysicalSizeX")
                physical_size_y = pixels.attrib.get("PhysicalSizeY")
                physical_size_unit = str(
                    pixels.attrib.get("PhysicalSizeXUnit") or pixels.attrib.get("PhysicalSizeYUnit") or ""
                )
                for index, channel in enumerate(
                    element for element in pixels if str(element.tag).split("}")[-1] == "Channel"
                ):
                    channel_names.append(str(channel.attrib.get("Name") or f"Channel {index + 1}"))
                    raw_color = channel.attrib.get("Color")
                    if raw_color is None:
                        channel_colors.append("")
                        continue
                    try:
                        channel_colors.append(ome_color_to_hex(raw_color))
                    except ValueError:
                        channel_colors.append("")
        except (ParseError, DefusedXmlException, TypeError, ValueError):
            channel_names = []
            channel_colors = []

    return {
        "axes": axes,
        "shape": shape,
        "channel_axis": channel_axis,
        "channel_count": channel_count,
        "channel_names": channel_names[:channel_count],
        "channel_colors": channel_colors[:channel_count],
        "physical_size_x": _optional_float(physical_size_x),
        "physical_size_y": _optional_float(physical_size_y),
        "physical_size_unit": physical_size_unit,
    }


# Invalid optional metadata should not prevent otherwise valid TIFFs from opening.
def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


# Reading the series and selecting its plane together prevents axes and pixels from drifting apart.
def _read_tiff_selected_layer(
    path: Path,
    stack_channel_index: int | None = None,
    stack_z_mode: str = "max_projection",
    stack_z_index: int | None = None,
) -> np.ndarray:
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = str(getattr(series, "axes", "") or "")
        arr = series.asarray()
    return select_tiff_stack_channel(
        arr,
        stack_channel_index=stack_channel_index,
        axes=axes,
        stack_z_mode=stack_z_mode,
        stack_z_index=stack_z_index,
    )


def open_image_with_tifffile(
    path: Path,
    stack_channel_index: int | None = None,
    stack_z_mode: str = "max_projection",
    stack_z_index: int | None = None,
) -> Any:
    """Decode a TIFF plane with tifffile and expose it as an ImageJ ImagePlus."""

    ij = get_ij()

    arr2d = _read_tiff_selected_layer(
        path,
        stack_channel_index=stack_channel_index,
        stack_z_mode=stack_z_mode,
        stack_z_index=stack_z_index,
    )
    arr2d = np.ascontiguousarray(arr2d)

    return ij.py.to_imageplus(arr2d)


def staged_image_process_directory(process_id: int | None = None) -> Path:
    """Return the process-owned directory used for locally staged inputs."""

    pid = int(process_id if process_id is not None else os.getpid())
    return Path(tempfile.gettempdir()) / "cellonaut_staged_images" / str(pid)


def cleanup_staged_image_process_directory(
    process_id: int,
    log_func: Callable[[str], None] | None = None,
) -> None:
    """Remove a process-owned staging directory without masking prior failures."""

    directory = staged_image_process_directory(process_id)
    try:
        shutil.rmtree(directory, ignore_errors=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        if log_func is not None:
            log_func(f"[WARN] Could not remove staged-image folder {directory}: {exc}")


def stage_image_to_local_temp(
    path: Path,
    log_func: Callable[[str], None],
    should_cancel: Callable[[], bool] | None = None,
    cancel_exception: type[Exception] = RuntimeError,
    runtime: PipelineRuntime | None = None,
) -> Path:
    """Copy an input image to local temporary storage with cancellation support."""

    if runtime is not None:
        runtime.stage(PipelineStage.STAGING, "Copying network image locally")

    suffix = path.suffix if path.suffix else ".tif"
    safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in path.stem)

    temp_dir = staged_image_process_directory()
    temp_dir.mkdir(parents=True, exist_ok=True)

    descriptor, staged_name = tempfile.mkstemp(prefix=f"{safe_name}_", suffix=suffix, dir=temp_dir)
    os.close(descriptor)
    dst = Path(staged_name)

    src_str = str(path)
    dst_str = str(dst)

    log_func(f"[NETWORK] Staging network image locally: {src_str}")

    try:
        with open(src_str, "rb") as fsrc, open(dst_str, "wb") as fdst:
            try:
                total_bytes = int(os.fstat(fsrc.fileno()).st_size)
            except OSError:
                total_bytes = 0
            copied_bytes = 0
            last_reported_percent = -10
            while True:
                if should_cancel is not None and should_cancel():
                    raise cancel_exception("Pipeline cancelled during network file staging.")

                chunk = fsrc.read(1024 * 1024)
                if not chunk:
                    break
                fdst.write(chunk)
                copied_bytes += len(chunk)
                if total_bytes > 0:
                    percent = min(100, int(copied_bytes * 100 / total_bytes))
                    if percent >= last_reported_percent + 10:
                        last_reported_percent = percent
                        log_func(
                            f"[NETWORK] Staging {path.name}: {percent}% "
                            f"({copied_bytes / (1024 * 1024):.1f} / {total_bytes / (1024 * 1024):.1f} MiB)"
                        )
    except Exception:
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        raise

    return dst


def remove_staged_image(path: Path, log_func: Callable[[str], None]) -> None:
    """Remove one staged image and report non-fatal cleanup failures."""

    try:
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
    except OSError as exc:
        log_func(f"[WARN] Could not remove staged image {path}: {exc}")


def imageplus_to_numpy_2d(img) -> np.ndarray:
    """Convert an ImageJ image to a two-dimensional NumPy analysis array."""

    if isinstance(img, np.ndarray):
        return normalize_numpy_image(img)

    raw = imageplus_pixels_to_numpy_2d(img)
    if raw is not None:
        return raw

    ij = get_ij()
    arr = ij.py.from_java(img)
    return normalize_numpy_image(arr)


# ImageJ exposes unsigned 8/16-bit pixels through signed Java primitive arrays.
def _unsigned_pixels_for_bit_depth(pixels: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 8:
        if np.size(pixels) and np.nanmin(pixels) < 0:
            return (pixels.astype(np.int16, copy=False) & 0xFF).astype(np.uint8)
        return pixels.astype(np.uint8, copy=False)
    if bit_depth == 16:
        if np.size(pixels) and np.nanmin(pixels) < 0:
            return (pixels.astype(np.int32, copy=False) & 0xFFFF).astype(np.uint16)
        return pixels.astype(np.uint16, copy=False)
    if bit_depth == 32:
        return pixels.astype(np.float32, copy=False)
    return pixels


def imageplus_pixels_to_numpy_2d(img) -> np.ndarray | None:
    """Read raw ImageJ pixels, or return None for unsupported processors."""

    if isinstance(img, np.ndarray):
        return normalize_numpy_image(img)

    try:
        processor = img.getProcessor()
        pixels = np.asarray(processor.getPixels())
        width = int(img.getWidth())
        height = int(img.getHeight())
        bit_depth = int(img.getBitDepth())
    except Exception:
        return None

    try:
        arr = _unsigned_pixels_for_bit_depth(pixels, bit_depth)
        if bit_depth == 24:
            # RGB ImageJ processors store packed int pixels. Keep the red channel
            # for parity with normalize_numpy_image's RGB handling.
            arr = ((arr.astype(np.uint32, copy=False) >> 16) & 0xFF).astype(np.uint8)
        return np.asarray(arr).reshape((height, width))
    except Exception:
        return None


def read_tiff_numpy_2d(
    path: Path,
    stack_channel_index: int | None = None,
    stack_z_mode: str = "max_projection",
    stack_z_index: int | None = None,
) -> np.ndarray:
    """Read one channel and Z selection from a TIFF as a two-dimensional array."""

    return _read_tiff_selected_layer(
        path,
        stack_channel_index=stack_channel_index,
        stack_z_mode=stack_z_mode,
        stack_z_index=stack_z_index,
    )


def resolve_channel_indices_for_file_map(
    *,
    image_defs: list[Any],
    file_map: dict[str, Path | None],
    input_structure: str,
    log_func: Callable[[str], None],
) -> dict[str, int | None]:
    """Resolve configured physical channels to one-based TIFF layer indices."""

    channel_indices: dict[str, int | None] = {
        img.key: img.stack_channel_index for img in image_defs if img.key in file_map
    }

    file_backed_defs = [img for img in image_defs if file_map.get(img.key) is not None]
    if input_structure != STRUCTURE_FLAT_TIFFS or not file_backed_defs:
        return channel_indices

    paths = [Path(path) for path in file_map.values() if path is not None]
    unique_paths = {str(path) for path in paths}
    if len(unique_paths) != 1:
        return channel_indices

    physical_defs = [img for img in file_backed_defs if not getattr(img, "stack_source_image_key", "")]
    missing = [img for img in physical_defs if channel_indices.get(img.key) is None]

    def propagate_source_layers() -> None:
        for img in file_backed_defs:
            source_key = str(getattr(img, "stack_source_image_key", "") or "")
            if source_key:
                channel_indices[img.key] = channel_indices.get(source_key)

    if not missing:
        propagate_source_layers()
        return channel_indices

    sample_path = paths[0]
    try:
        info = get_tiff_stack_info(sample_path)
    except Exception as exc:
        raise ValueError(f"Cannot automatically map TIFF channels for {sample_path.name}: {exc}") from exc

    channel_count = int(info.get("channel_count", 1) or 1)
    if channel_count < len(physical_defs):
        raise ValueError(
            f"TIFF {sample_path.name} has {channel_count} channel(s), but {len(physical_defs)} physical channels are configured. "
            "Correct the channel configuration before processing this sample."
        )

    used_layers: set[int] = set()
    for img in physical_defs:
        configured_layer = channel_indices.get(img.key)
        if configured_layer is not None:
            used_layers.add(int(configured_layer))
    inferred_labels = []
    for index, img in enumerate(physical_defs, start=1):
        if channel_indices.get(img.key) is not None:
            continue
        available_layers = [layer for layer in range(1, channel_count + 1) if layer not in used_layers]
        if not available_layers:
            break
        layer = index if index in available_layers else available_layers[0]
        channel_indices[img.key] = layer
        used_layers.add(layer)
        inferred_labels.append(f"{img.label}=layer {layer}")

    propagate_source_layers()

    if inferred_labels:
        log_func("[INFO] Inferred TIFF stack layers from channel row order: " + ", ".join(inferred_labels))

    return channel_indices


def find_channel_files(
    sample_folder: Path,
    cfg: Any,
    id_label: str,
    log_func: Callable[[str], None],
    required_image_keys: set[str] | None = None,
    ambiguity_func: Callable[[str], None] | None = None,
) -> dict[str, Path | None] | None:
    """Resolve every configured physical channel to its source image file."""

    files: dict[str, Path | None] = {}
    required_keys = set(
        required_image_keys
        if required_image_keys is not None
        else [img.key for img in cfg.images if not getattr(img, "combined_mask_source_keys", None)]
    )

    if cfg.input_structure == STRUCTURE_FLAT_TIFFS:
        if not is_supported_tiff_file(sample_folder):
            log_func(f"[WARN] Skipping {id_label}: not a supported image file")
            return None
        if not cfg.images:
            log_func(f"[WARN] Skipping {id_label}: no image definition configured")
            return None

        for img in cfg.images:
            files[img.key] = None if getattr(img, "combined_mask_source_keys", None) else sample_folder
        return files

    if cfg.input_structure == STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS:
        if not cfg.images:
            log_func(f"[WARN] Skipping {id_label}: no image definition configured")
            return None

        sample_stem = sample_folder.stem
        for img in cfg.images:
            if getattr(img, "combined_mask_source_keys", None):
                files[img.key] = None
                continue
            image_folder = cfg.input_dir / img.folder_name
            tif_file = find_matching_image_file_by_stem(image_folder, sample_stem)
            if tif_file is None:
                if img.key in required_keys:
                    log_func(
                        f"[WARN] Skipping {id_label}: no matching image named {sample_stem}.tif/.tiff "
                        f"in image folder {img.folder_name}"
                    )
                    return None
                files[img.key] = None
                continue
            files[img.key] = tif_file
        return files

    for img in cfg.images:
        if getattr(img, "combined_mask_source_keys", None):
            files[img.key] = None
            continue
        folder = get_subfolder(sample_folder, img.folder_name)
        if folder is None:
            if img.key in required_keys:
                log_func(f"[WARN] Skipping {id_label}: missing folder {img.folder_name}")
                return None
            files[img.key] = None
            continue

        tif_file = find_first_image_file(folder, log_func, ambiguity_func)
        if tif_file is None:
            if img.key in required_keys:
                log_func(f"[WARN] Skipping {id_label}: no TIFF in {img.folder_name}")
                return None
            files[img.key] = None
            continue

        files[img.key] = tif_file

    return files


# Try closing every opened image, even if one close fails.
def _close_opened_images(opened: list[Any]) -> None:
    for opened_image in opened:
        try:
            opened_image.close()
        except Exception:
            pass


def open_channel_images(
    file_map: dict[str, Path | None],
    id_label: str,
    log_func: Callable[[str], None],
    channel_indices: dict[str, int | None] | None = None,
    channel_z_modes: dict[str, str] | None = None,
    channel_z_indices: dict[str, int | None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    cancel_exception: type[Exception] = RuntimeError,
    runtime: PipelineRuntime | None = None,
) -> dict[str, Any] | None:
    """Open required channels as ImageJ images, closing partial results on failure.

    Keys selecting the same file/channel/Z plane share an ImagePlus. The caller
    owns successful results and must close each distinct image once, after all
    borrowers finish. Missing optional paths map to None; a failed open returns
    None for the entire operation. Cancellation and unexpected errors propagate.
    """

    images: dict[str, Any] = {}
    opened: list[Any] = []
    opened_sources: dict[tuple[str, int | None, str, int | None], Any] = {}

    try:
        for key, path in file_map.items():
            if should_cancel is not None and should_cancel():
                raise cancel_exception("Pipeline cancelled while opening images.")

            if path is None:
                images[key] = None
                continue

            imp = None
            path_str = str(path)

            stack_channel_index = (channel_indices or {}).get(key)
            stack_z_mode = (channel_z_modes or {}).get(key, "max_projection")
            stack_z_index = (channel_z_indices or {}).get(key)
            source_key = (
                os.path.normcase(os.path.abspath(path_str)),
                stack_channel_index,
                stack_z_mode,
                stack_z_index,
            )
            if source_key in opened_sources:
                images[key] = opened_sources[source_key]
                log_func(f"[{id_label}] Reusing opened TIFF plane for {key}: {path.name}")
                continue

            try:
                imp = open_image_with_tifffile(
                    path,
                    stack_channel_index=stack_channel_index,
                    stack_z_mode=stack_z_mode,
                    stack_z_index=stack_z_index,
                )
            except Exception as exc:
                log_func(f"[WARN] Direct TIFF open failed for {key}: {exc}")
                imp = None

            if imp is None and path_str.startswith("\\\\"):
                local_path = None
                try:
                    local_path = stage_image_to_local_temp(
                        path,
                        log_func,
                        should_cancel=should_cancel,
                        cancel_exception=cancel_exception,
                        runtime=runtime,
                    )
                    imp = open_image_with_tifffile(
                        local_path,
                        stack_channel_index=stack_channel_index,
                        stack_z_mode=stack_z_mode,
                        stack_z_index=stack_z_index,
                    )
                finally:
                    if local_path is not None:
                        remove_staged_image(local_path, log_func)

            if imp is None:
                log_func(f"[WARN] Could not open image for sample {id_label}: {path}")
                _close_opened_images(opened)
                return None

            images[key] = imp
            opened.append(imp)
            opened_sources[source_key] = imp

        return images

    except Exception:
        _close_opened_images(opened)
        raise


def open_channel_arrays(
    file_map: dict[str, Path | None],
    id_label: str,
    log_func: Callable[[str], None],
    channel_indices: dict[str, int | None] | None = None,
    channel_z_modes: dict[str, str] | None = None,
    channel_z_indices: dict[str, int | None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    cancel_exception: type[Exception] = RuntimeError,
    runtime: PipelineRuntime | None = None,
) -> dict[str, Any] | None:
    """Open required channels as native two-dimensional NumPy arrays."""

    images: dict[str, Any] = {}

    for key, path in file_map.items():
        if should_cancel is not None and should_cancel():
            raise cancel_exception("Pipeline cancelled while opening images.")

        if path is None:
            images[key] = None
            continue

        try:
            images[key] = read_tiff_numpy_2d(
                path,
                stack_channel_index=(channel_indices or {}).get(key),
                stack_z_mode=(channel_z_modes or {}).get(key, "max_projection"),
                stack_z_index=(channel_z_indices or {}).get(key),
            )
        except Exception as exc:
            log_func(f"[WARN] Could not open image for sample {id_label}: {path} ({exc})")
            return None

    return images
