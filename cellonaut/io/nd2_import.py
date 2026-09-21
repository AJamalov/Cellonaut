"""ND2 inspection and OME-TIFF export helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from cellonaut.io.fiji_tiff import write_ome_channel_stack
from cellonaut.exceptions import PipelineCancelled, SetupError, SetupErrorCode, SetupImportError
from cellonaut.pipeline.cancellation import check_cancel
from cellonaut.pipeline.discovery import relative_path_sort_key, safe_iterdir, scan_matching_files
from cellonaut.pipeline.operation_state import save_operation_state

try:
    nd2: Any = importlib.import_module("nd2")
except ImportError:
    nd2 = None


LogFunc = Callable[[str], None]
ProgressFunc = Callable[[int], None]
CurrentFileFunc = Callable[[str], None]

ChannelSelector = int | str


@dataclass(slots=True)
class ND2ImportConfig:
    """Settings for exporting ND2 files as channel-preserving OME-TIFF stacks."""

    source_dir: Path
    output_dir: Path
    channel_map: dict[str, ChannelSelector] = field(default_factory=dict)

    z_mode: str = "max_projection"

    z_index: int = 0

    strip_after_last_underscore: bool = False

    preserve_subfolders: bool = True

    def validate(self) -> None:
        """Reject invalid conversion settings before any large ND2 file is opened."""
        if self.z_mode not in {"single_z", "max_projection"}:
            raise ValueError(f"Unsupported z_mode: {self.z_mode!r}. " "Expected 'single_z' or 'max_projection'.")

        if self.z_index < 0:
            raise ValueError("z_index must be >= 0.")

        if not self.channel_map:
            raise SetupError(SetupErrorCode.ND2_CHANNELS, "channel_map cannot be empty.")

        for folder_name, selector in self.channel_map.items():
            if not folder_name or not isinstance(folder_name, str):
                raise SetupError(SetupErrorCode.ND2_CHANNELS, f"Invalid channel_map key: {folder_name!r}")

            if isinstance(selector, int):
                if selector < 0:
                    raise SetupError(SetupErrorCode.ND2_CHANNELS,
                        f"Invalid channel index for {folder_name!r}: {selector!r}. "
                        "Channel indices must be non-negative integers."
                    )
            elif isinstance(selector, str):
                if not selector.strip():
                    raise SetupError(SetupErrorCode.ND2_CHANNELS, f"Invalid channel name for {folder_name!r}: {selector!r}")
            else:
                raise SetupError(SetupErrorCode.ND2_CHANNELS,
                    f"Invalid selector type for {folder_name!r}: {type(selector).__name__}. " "Expected int or str."
                )


def require_nd2() -> None:
    """Raise an actionable error when the optional ND2 backend is unavailable."""
    if nd2 is None:
        raise SetupImportError(SetupErrorCode.ND2_BACKEND, "The 'nd2' package is not installed. " "Install it with: pip install nd2 tifffile")


def log_default(message: str) -> None:
    """Print a conversion message when no GUI worker logger was supplied."""
    print(message)


def normalize_sample_name(
    stem: str,
    strip_after_last_underscore: bool = False,
) -> str:
    """Return the ND2 stem, optionally removing its final underscore-delimited suffix."""
    if strip_after_last_underscore and "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def list_nd2_files(source_dir: Path) -> list[Path]:
    """List ND2 files recursively while excluding build, cache, and output folders."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    files = scan_matching_files(
        source_dir,
        recursive=True,
        predicate=lambda path: path.is_file() and path.suffix.lower() == ".nd2",
        iterdir_func=safe_iterdir,
    )
    return sorted(files, key=lambda path: relative_path_sort_key(source_dir, path))


def nd2_relative_parent(cfg: ND2ImportConfig, nd2_path: Path) -> Path:
    """Return the mirrored source parent used to keep equal sample names distinct."""
    if not cfg.preserve_subfolders:
        return Path()
    try:
        relative = nd2_path.resolve().parent.relative_to(cfg.source_dir.resolve())
    except (OSError, ValueError):
        return Path()
    return Path() if str(relative) == "." else relative


def nd2_output_path(cfg: ND2ImportConfig, nd2_path: Path, sample_name: str) -> Path:
    """Return the canonical OME-TIFF destination for one ND2 sample."""
    return cfg.output_dir / nd2_relative_parent(cfg, nd2_path) / f"{sample_name}.ome.tif"


def validate_nd2_output_plan(cfg: ND2ImportConfig, nd2_files: Sequence[Path]) -> None:
    """Reject ambiguous source-name mappings before conversion opens source files."""
    destinations: dict[str, tuple[Path, Path]] = {}
    collisions: list[tuple[Path, Path, Path]] = []

    for nd2_path in nd2_files:
        sample_name = normalize_sample_name(
            nd2_path.stem,
            strip_after_last_underscore=cfg.strip_after_last_underscore,
        )
        destination = nd2_output_path(cfg, nd2_path, sample_name)
        destination_key = destination.absolute().as_posix().casefold()
        previous = destinations.get(destination_key)
        if previous is not None:
            collisions.append((previous[0], nd2_path, destination))
        else:
            destinations[destination_key] = (nd2_path, destination)

    if collisions:
        first_source, second_source, destination = collisions[0]
        raise ValueError(
            "ND2 conversion would map multiple source files to the same TIFF output: "
            f"{first_source.name}, {second_source.name} -> {destination}. "
            "Keep the full source names or preserve their source subfolders."
        )


# Make metadata names safe as dictionary keys because the GUI stores channel mappings by name.
def _unique_channel_names(raw_names: Sequence[str], channel_count: int) -> list[str]:
    total = max(1, int(channel_count), len(raw_names))
    names: list[str] = []
    used: set[str] = set()
    for index in range(total):
        base = str(raw_names[index] if index < len(raw_names) else "").strip() or f"Channel {index + 1}"
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base} {suffix}"
            suffix += 1
        names.append(candidate)
        used.add(candidate.casefold())
    return names


def get_nd2_channel_names(f: Any) -> list[str]:
    """Read channel labels and assign unique names when labels are missing or duplicated."""
    raw_names: list[str] = []

    try:
        metadata = getattr(f, "metadata", None)
        channels = getattr(metadata, "channels", None)
        if channels is not None:
            for ch in channels:
                name = getattr(getattr(ch, "channel", None), "name", None)
                if name is None:
                    name = getattr(ch, "name", None)
                raw_names.append(str(name).strip() if name else "")
    except Exception:
        pass

    sizes = dict(getattr(f, "sizes", {}))
    channel_count = int(sizes.get("C", len(raw_names) or 1))
    return _unique_channel_names(raw_names, channel_count)


def resolve_channel_selector(
    selector: ChannelSelector,
    channel_names: Sequence[str],
) -> int:
    """Resolve an exact index or a case-tolerant channel name without guessing."""
    if isinstance(selector, int):
        if not 0 <= selector < len(channel_names):
            raise IndexError(
                f"Requested channel index {selector}, but file only has " f"{len(channel_names)} channel(s)."
            )
        return selector

    target = selector.strip()

    # Exact names win so channels with case-only differences stay selectable.
    for i, name in enumerate(channel_names):
        if name == target:
            return i

    # Case-insensitive matching keeps user-entered presets tolerant of small
    # metadata capitalization differences between microscopes or exports.
    target_lower = target.lower()
    for i, name in enumerate(channel_names):
        if name.lower() == target_lower:
            return i

    raise KeyError(f"Could not find channel named {selector!r}. " f"Detected channels: {list(channel_names)}")


def auto_build_channel_map_from_names(
    detected_channel_names: Sequence[str],
    preferred_output_folders: Sequence[str] | None = None,
) -> dict[str, str]:
    """Map detected channels one-to-one in the OME-TIFF output order."""
    if preferred_output_folders is None:
        preferred_output_folders = [str(name) for name in detected_channel_names]

    return dict(zip(preferred_output_folders, detected_channel_names))


def infer_channel_color_hex(channel_name: str, folder_name: str = "") -> str:
    """Infer a conventional display color when ND2 metadata has no usable color."""
    text = f"{channel_name} {folder_name}".lower()
    color_rules = [
        (("dapi", "hoechst", "blue", "405", "bfp", "cfp"), "#0000FF"),
        (("gfp", "fitc", "green", "488", "yfp"), "#00FF00"),
        (("rfp", "tritc", "mcherry", "scarlet", "red", "561", "568"), "#FF0000"),
        (("cy5", "farred", "far-red", "647", "640"), "#FF00FF"),
        (("cy3", "orange", "555"), "#FFA500"),
    ]
    for needles, color in color_rules:
        if any(needle in text for needle in needles):
            return color
    return "#FFFFFF"


def metadata_color_to_hex(value: Any) -> str:
    """Normalize supported ND2 metadata color representations to ``#RRGGBB``."""
    if value is None:
        return ""

    try:
        if hasattr(value, "as_hex"):
            text = str(value.as_hex()).strip().upper()
            if len(text) == 7 and text.startswith("#"):
                int(text[1:], 16)
                return text
    except Exception:
        pass

    if hasattr(value, "_asdict"):
        try:
            value = value._asdict()
        except Exception:
            pass

    if isinstance(value, dict):
        for keys in (("r", "g", "b"), ("red", "green", "blue")):
            if all(key in value for key in keys):
                return _rgb_triplet_to_hex([value[key] for key in keys])
        for key in ("rgb", "rgba", "color", "value"):
            if key in value:
                parsed = metadata_color_to_hex(value[key])
                if parsed:
                    return parsed

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return _rgb_triplet_to_hex(value[:3])

    for keys in (("r", "g", "b"), ("red", "green", "blue")):
        try:
            if all(hasattr(value, key) for key in keys):
                return _rgb_triplet_to_hex([getattr(value, key) for key in keys])
        except Exception:
            pass

    if isinstance(value, str):
        text = value.strip()
        if len(text) == 7 and text.startswith("#"):
            try:
                int(text[1:], 16)
                return text.upper()
            except ValueError:
                return ""
        if text.lower().startswith("0x"):
            try:
                return _int_color_to_hex(int(text, 16))
            except ValueError:
                return ""
        if "," in text:
            parts = [part.strip() for part in text.split(",")]
            if len(parts) >= 3:
                return _rgb_triplet_to_hex(parts[:3])

    if isinstance(value, (int, np.integer)):
        return _int_color_to_hex(int(value))

    return ""


# Normalize integer and fractional RGB metadata while rejecting malformed values without blocking file inspection.
def _rgb_triplet_to_hex(values: Sequence[Any]) -> str:
    rgb: list[int] = []
    for value in list(values)[:3]:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return ""
        if not np.isfinite(number):
            return ""
        if 0.0 <= number <= 1.0:
            number *= 255.0
        rgb.append(max(0, min(255, int(round(number)))))
    if len(rgb) != 3:
        return ""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


# Decode packed metadata colors separately because some ND2 variants include alpha in the high byte.
def _int_color_to_hex(value: int) -> str:
    value = int(value)
    if value < 0:
        value &= 0xFFFFFFFF
    if value > 0xFFFFFF:
        red = value & 0xFF
        green = (value >> 8) & 0xFF
        blue = (value >> 16) & 0xFF
        if red == green == blue == 0:
            red = (value >> 16) & 0xFF
            green = (value >> 8) & 0xFF
            blue = value & 0xFF
        return f"#{red:02X}{green:02X}{blue:02X}"
    return f"#{value & 0xFFFFFF:06X}"


def get_nd2_channel_colors(f: Any, channel_names: Sequence[str] | None = None) -> list[str]:
    """Return one display color per unique channel name, in conversion order."""
    colors: list[str] = []
    names = list(channel_names or [])

    try:
        metadata = getattr(f, "metadata", None)
        channels = getattr(metadata, "channels", None)
        if channels is not None:
            for i, ch in enumerate(channels):
                channel_meta = getattr(ch, "channel", ch)
                color = metadata_color_to_hex(getattr(channel_meta, "color", None))
                if not color:
                    color = metadata_color_to_hex(getattr(ch, "color", None))
                if not color and i < len(names):
                    color = infer_channel_color_hex(names[i], names[i])
                colors.append(color or "#FFFFFF")
    except Exception:
        colors = []

    if not colors:
        if not names:
            names = get_nd2_channel_names(f)
        colors = [infer_channel_color_hex(name, name) for name in names]

    while len(colors) < len(names):
        colors.append(infer_channel_color_hex(names[len(colors)], names[len(colors)]))
    return colors[: len(names)] if names else colors


def get_nd2_pixel_sizes_um(f: Any, channel_index: int = 0) -> tuple[float | None, float | None]:
    """Return calibrated X/Y pixel sizes in micrometres when ND2 exposes them."""
    try:
        voxel = f.voxel_size(channel=channel_index)
    except TypeError:
        try:
            voxel = f.voxel_size()
        except Exception:
            return None, None
    except Exception:
        return None, None

    x = getattr(voxel, "x", None)
    y = getattr(voxel, "y", None)
    try:
        x = float(x) if x is not None else None
        y = float(y) if y is not None else None
    except Exception:
        return None, None
    return x, y


def inspect_nd2_file(path: Path) -> dict[str, Any]:
    """Inspect ND2 axes and channels without loading the complete acquisition."""
    require_nd2()

    with nd2.ND2File(path) as f:
        sizes = dict(getattr(f, "sizes", {}))
        channel_names = get_nd2_channel_names(f)
        channel_colors = get_nd2_channel_colors(f, channel_names)

        raw_shape = tuple(int(v) for v in sizes.values())
        dtype = "unknown"

        try:
            frame0 = np.asarray(f.read_frame(0))
            dtype = str(frame0.dtype)
        except Exception:
            pass

        return {
            "path": str(path),
            "name": path.name,
            "raw_shape": raw_shape,
            "sizes": sizes,
            "dtype": dtype,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "suggested_channel_map": auto_build_channel_map_from_names(channel_names),
        }


# Remove unsupported acquisition axes explicitly so remaining dimension names stay aligned with NumPy axes.
def _take_first_axis(
    array: np.ndarray,
    dim_order: list[str],
    dim_name: str,
) -> np.ndarray:
    if dim_name not in dim_order:
        return array

    axis = dim_order.index(dim_name)
    array = np.take(array, indices=0, axis=axis)
    dim_order.pop(axis)
    return array


# Select after position/time reduction so channel indices continue to match ND2 metadata.
def _take_channel(
    array: np.ndarray,
    dim_order: list[str],
    channel_index: int,
) -> np.ndarray:
    if "C" not in dim_order:
        if channel_index != 0:
            raise IndexError(
                f"Requested channel index {channel_index}, but this ND2 file "
                "does not appear to contain a C dimension."
            )
        return array

    axis = dim_order.index("C")
    channel_count = array.shape[axis]

    if not 0 <= channel_index < channel_count:
        raise IndexError(f"Requested channel index {channel_index}, but file only has " f"{channel_count} channel(s).")

    array = np.take(array, indices=channel_index, axis=axis)
    dim_order.pop(axis)
    return array


# Apply one documented Z policy and remove the dimension immediately to prevent accidental second projection.
def _handle_z_dimension(
    array: np.ndarray,
    dim_order: list[str],
    z_mode: str,
    z_index: int,
) -> np.ndarray:
    if "Z" not in dim_order:
        return array

    axis = dim_order.index("Z")
    z_count = array.shape[axis]

    if z_mode == "single_z":
        if not 0 <= z_index < z_count:
            raise IndexError(f"Requested z_index {z_index}, but file only has {z_count} z-slice(s).")
        array = np.take(array, indices=z_index, axis=axis)
        dim_order.pop(axis)
        return array

    if z_mode == "max_projection":
        array = np.max(array, axis=axis)
        dim_order.pop(axis)
        return array

    raise ValueError(f"Unsupported z_mode: {z_mode!r}")


# Refuse ambiguous remaining dimensions rather than silently reshaping scientific data into the wrong plane.
def _ensure_2d_yx(
    array: np.ndarray,
    dim_order: Sequence[str],
) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(
            "Could not reduce ND2 data to a 2D image. "
            f"Remaining shape: {array.shape}, remaining dimensions: {list(dim_order)}"
        )
    return array


def read_first_nd2_position(nd2_file: Any) -> np.ndarray:
    """Load only the first XY position supported by the conversion workflow."""

    return np.asarray(nd2_file.asarray(position=0))


# Reduce one channel to YX in named-axis order; P is position and S is an RGB sample component in python-nd2.
def _select_channel_plane(
    arr: np.ndarray,
    sizes: dict[str, int],
    channel_index: int,
    z_mode: str,
    z_index: int,
) -> np.ndarray:
    dim_order = list(sizes.keys())
    work = arr

    work = _take_first_axis(work, dim_order, "P")
    work = _take_first_axis(work, dim_order, "T")
    work = _take_channel(work, dim_order, channel_index)
    work = _handle_z_dimension(work, dim_order, z_mode, z_index)
    work = _take_first_axis(work, dim_order, "S")
    work = _ensure_2d_yx(work, dim_order)

    return np.asarray(work)


def export_nd2_file(
    nd2_path: Path,
    cfg: ND2ImportConfig,
    log_func: LogFunc = log_default,
    should_cancel: Callable[[], bool] | None = None,
) -> list[Path]:
    """Write selected channels as one uncompressed CYX OME-TIFF; return its path in a list.

    Export the first position, timepoint and RGB component. cfg.z_mode selects
    a maximum projection or cfg.z_index (zero-based, unlike GUI TIFF indices).
    Channel order follows cfg.channel_map. Preserve pixel dtype and available
    spatial calibration; do not apply display normalization. Publication is
    exclusive, with numbered names on collision rather than overwriting a file.
    """

    require_nd2()
    cfg.validate()
    check_cancel(should_cancel)

    sample_name = normalize_sample_name(
        nd2_path.stem,
        strip_after_last_underscore=cfg.strip_after_last_underscore,
    )

    log_func(f"[ND2] Opening {nd2_path}")
    check_cancel(should_cancel)

    with nd2.ND2File(nd2_path) as f:
        sizes = dict(getattr(f, "sizes", {}))
        channel_names = get_nd2_channel_names(f)
        channel_colors = get_nd2_channel_colors(f, channel_names)

        log_func(f"[ND2] Reading pixel data from {nd2_path.name} ...")
        check_cancel(should_cancel)
        arr = read_first_nd2_position(f)
        check_cancel(should_cancel)

        log_func(f"[ND2] Detected channels: {channel_names}")
        log_func(f"[ND2] Export sample TIFF stack: {sample_name}")
        for key, label in (("P", "XY position"), ("T", "timepoint"), ("S", "RGB component")):
            count = int(sizes.get(key, 0) or 0)
            if count > 1:
                log_func(f"[ND2] {label.capitalize()} count={count}; exporting index 0")

        planes = []
        labels = []
        colors = []
        pixel_size_x = None
        pixel_size_y = None
        for channel_folder, selector in cfg.channel_map.items():
            check_cancel(should_cancel)
            channel_index = resolve_channel_selector(selector, channel_names)
            channel_name = channel_names[channel_index]
            img2d = _select_channel_plane(
                arr=arr,
                sizes=sizes,
                channel_index=channel_index,
                z_mode=cfg.z_mode,
                z_index=cfg.z_index,
            )
            planes.append(np.asarray(img2d))
            labels.append(str(channel_folder or channel_name))
            color_hex = channel_colors[channel_index] if channel_index < len(channel_colors) else ""
            colors.append(color_hex or infer_channel_color_hex(channel_name, channel_folder))
            if pixel_size_x is None or pixel_size_y is None:
                pixel_size_x, pixel_size_y = get_nd2_pixel_sizes_um(f, channel_index)

        out_path = nd2_output_path(cfg, nd2_path, sample_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reduction_notes = [f"z_mode={cfg.z_mode}", f"z_index={cfg.z_index}"]
        for key, note in (("P", "position_index=0"), ("T", "time_index=0"), ("S", "rgb_component_index=0")):
            if int(sizes.get(key, 0) or 0) > 1:
                reduction_notes.append(note)
        stack = np.stack(planes, axis=0)
        base_path = out_path
        number = 1
        while True:
            check_cancel(should_cancel)
            if out_path.exists():
                number += 1
                out_path = base_path.with_name(f"{sample_name}_{number}.ome.tif")
                continue
            try:
                write_ome_channel_stack(
                    out_path,
                    stack,
                    labels=labels,
                    colors_hex=colors,
                    image_name=sample_name,
                    physical_size_x=pixel_size_x,
                    physical_size_y=pixel_size_y,
                    description=f"Converted from ND2 file {nd2_path.name}; " + "; ".join(reduction_notes),
                    exclusive=True,
                )
                break
            except FileExistsError:
                # Another writer may claim the destination after the existence check.
                number += 1
                out_path = base_path.with_name(f"{sample_name}_{number}.ome.tif")
        if out_path != base_path:
            log_func(f"[ND2] Destination already exists; saved numbered copy: {out_path.name}")
        log_func(f"[ND2] Exported multi-channel TIFF stack -> {out_path}")

    return [out_path]


def convert_nd2_folder(
    cfg: ND2ImportConfig,
    log_func: LogFunc = log_default,
    progress_func: ProgressFunc | None = None,
    current_file_func: CurrentFileFunc | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, int]:
    """Convert all ND2 files and report successes and per-file failures."""

    require_nd2()
    cfg.validate()
    check_cancel(should_cancel)

    nd2_files = list_nd2_files(cfg.source_dir)

    if not nd2_files:
        raise SetupError(SetupErrorCode.ND2_INPUT, f"No .nd2 files found in: {cfg.source_dir}")

    validate_nd2_output_plan(cfg, nd2_files)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_operation_state(
        cfg.output_dir,
        operation_type="nd2_conversion",
        state="in_progress",
        reset_started=True,
    )

    total = len(nd2_files)
    processed = 0
    failed = 0

    if progress_func is not None:
        progress_func(0)

    log_func(f"[ND2] Found {total} file(s) to convert.")

    # A secondary marker-write failure must not replace the cancellation or
    # conversion error that explains why processing stopped.
    def record_terminal_state(state: str, error: str) -> None:
        try:
            save_operation_state(
                cfg.output_dir,
                operation_type="nd2_conversion",
                state=state,
                error=error,
            )
        except (OSError, UnicodeError) as state_exc:
            log_func(f"[ND2][WARN] Could not update ConversionState.json: {state_exc}")

    try:
        for i, nd2_path in enumerate(nd2_files, start=1):
            check_cancel(should_cancel)

            if current_file_func is not None:
                current_file_func(f"Converting {i} / {total}: {nd2_path.name} | Done {processed} / {total}")

            log_func(f"[ND2] ({i}/{total}) Starting {nd2_path.name}")

            try:
                export_nd2_file(
                    nd2_path,
                    cfg,
                    log_func=log_func,
                    should_cancel=should_cancel,
                )
                processed += 1
            except PipelineCancelled:
                log_func("[ND2] Import cancelled by user.")
                raise
            except Exception as exc:
                failed += 1
                log_func(f"[ND2][ERROR] Failed on {nd2_path.name}: {exc}")

            if progress_func is not None:
                progress_func(int(i * 100 / total))
            if current_file_func is not None:
                current_file_func(f"Done {processed} / {total} | Failed {failed} | Last: {nd2_path.name}")
    except PipelineCancelled as exc:
        record_terminal_state("cancelled", str(exc))
        raise
    except BaseException as exc:
        record_terminal_state("failed", f"{type(exc).__name__}: {exc}")
        raise

    summary = {
        "total": total,
        "processed": processed,
        "failed": failed,
    }
    save_operation_state(
        cfg.output_dir,
        operation_type="nd2_conversion",
        state="completed_with_errors" if failed else "completed",
    )
    return summary
