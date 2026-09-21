"""Write channel stacks with metadata understood by Fiji and OME readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cellonaut.colors import normalize_rgb_hex
from cellonaut.io.writers import write_tiff


# Validate channel counts before writing because metadata mismatches can produce
# a TIFF that opens but assigns labels or colors to the wrong planes.
def _validated_channel_stack(
    stack_arr: np.ndarray,
    labels: list[str],
    colors_hex: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    stack = np.asarray(stack_arr)
    if stack.ndim != 3:
        raise ValueError(f"Channel stack must have CYX dimensions; received shape {stack.shape}.")
    channel_count = int(stack.shape[0])
    if channel_count < 1:
        raise ValueError("Channel stack must contain at least one channel.")
    if len(labels) != channel_count:
        raise ValueError(f"Expected {channel_count} channel labels, received {len(labels)}.")
    if len(colors_hex) != channel_count:
        raise ValueError(f"Expected {channel_count} channel colors, received {len(colors_hex)}.")
    return (
        stack,
        [str(label) for label in labels],
        [normalize_rgb_hex(color, fallback="#FFFFFF") for color in colors_hex],
    )


# Build Fiji's three-row lookup table with integer arithmetic so the requested
# display color does not alter the stored channel intensities.
def hex_to_lut(color_hex: str) -> np.ndarray:
    color = normalize_rgb_hex(color_hex, fallback="#FFFFFF")
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    ramp = np.arange(256, dtype=np.uint8)
    lut = np.zeros((3, 256), dtype=np.uint8)
    for idx, channel_value in enumerate(rgb):
        scaled = np.multiply(ramp.astype(np.uint16), np.uint16(channel_value))
        lut[idx] = np.floor_divide(scaled, np.uint16(255)).astype(np.uint8)
    return lut


# OME stores color as a signed 32-bit RGBA integer, including an opaque alpha
# byte; converting values above INT32_MAX keeps the generated XML schema-valid.
def hex_to_ome_color(color_hex: str) -> int:
    color = normalize_rgb_hex(color_hex, fallback="#FFFFFF")
    rgba = (int(color[1:], 16) << 8) | 0xFF
    return rgba - (1 << 32) if rgba >= (1 << 31) else rgba


# Store labels and LUTs in ImageJ metadata so Fiji opens exported overlays as an
# inspectable composite rather than a single unlabeled image sequence.
def write_fiji_channel_stack(
    path: Path,
    stack_arr: np.ndarray,
    labels: list[str],
    colors_hex: list[str],
    mode: str = "composite",
) -> None:
    stack, normalized_labels, normalized_colors = _validated_channel_stack(stack_arr, labels, colors_hex)
    normalized_mode = str(mode or "composite").strip().lower()
    if normalized_mode not in {"composite", "color", "grayscale"}:
        raise ValueError(f"Unsupported Fiji display mode: {mode!r}.")
    metadata: dict[str, Any] = {
        "axes": "CYX",
        "hyperstack": True,
        "mode": normalized_mode,
        "Labels": normalized_labels,
        "LUTs": [hex_to_lut(color) for color in normalized_colors],
    }
    write_tiff(
        path,
        stack,
        imagej=True,
        photometric="minisblack",
        metadata=metadata,
    )


# Preserve channel names, opaque display colors, and available pixel calibration
# so converted ND2 data remains self-describing outside Cellonaut.
def write_ome_channel_stack(
    path: Path,
    stack_arr: np.ndarray,
    labels: list[str],
    colors_hex: list[str],
    image_name: str,
    physical_size_x: float | None = None,
    physical_size_y: float | None = None,
    description: str | None = None,
    exclusive: bool = False,
) -> None:
    stack, normalized_labels, normalized_colors = _validated_channel_stack(stack_arr, labels, colors_hex)
    metadata: dict[str, Any] = {
        "axes": "CYX",
        "Name": str(image_name),
    }
    metadata["Channel"] = {
        "Name": normalized_labels,
        "Color": [hex_to_ome_color(color) for color in normalized_colors],
    }
    if physical_size_x is not None and physical_size_x > 0:
        metadata["PhysicalSizeX"] = float(physical_size_x)
        metadata["PhysicalSizeXUnit"] = "um"
    if physical_size_y is not None and physical_size_y > 0:
        metadata["PhysicalSizeY"] = float(physical_size_y)
        metadata["PhysicalSizeYUnit"] = "um"
    if description:
        metadata["Description"] = description

    write_tiff(
        path,
        stack,
        exclusive=exclusive,
        ome=True,
        photometric="minisblack",
        metadata=metadata,
    )
