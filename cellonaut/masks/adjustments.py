"""Integer-pixel translation and morphology for labels and binary masks."""

from __future__ import annotations

from typing import Optional, cast

import numpy as np


# Shift by copying overlapping slices so interpolation cannot invent object identifiers.
def _shift_2d(arr: np.ndarray, dx: int = 0, dy: int = 0, fill_value=0) -> np.ndarray:
    src = np.asarray(arr)
    if src.ndim != 2:
        return src
    dx = int(dx or 0)
    dy = int(dy or 0)
    if dx == 0 and dy == 0:
        return src.copy()

    out = np.full(src.shape, fill_value, dtype=src.dtype)
    h, w = src.shape
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx) if dx < 0 else w
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy) if dy < 0 else h

    if src_x1 > src_x0 and src_y1 > src_y0 and dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = src[src_y0:src_y1, src_x0:src_x1]
    return out


def _disk_structure(radius: int) -> Optional[np.ndarray]:
    """Build a circular footprint so morphology follows pixel distance."""
    radius = int(radius or 0)
    if radius <= 0:
        return None
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _ndimage():
    """Load SciPy only when adjustment operations require it."""
    try:
        from scipy import ndimage as ndi

        return ndi
    except Exception:
        return None


def _fill_holes_up_to_area(mask: np.ndarray, max_area: int, ndi) -> np.ndarray:
    """Fill enclosed background components no larger than the requested area."""
    max_area = int(max_area or 0)
    if max_area <= 0:
        return np.asarray(mask, dtype=bool)

    foreground = np.asarray(mask, dtype=bool)
    all_holes_filled = np.asarray(ndi.binary_fill_holes(foreground), dtype=bool)
    holes = all_holes_filled & ~foreground
    if not np.any(holes):
        return foreground

    labeled_raw, count_raw = cast(tuple[np.ndarray, int], ndi.label(holes))
    labeled = np.asarray(labeled_raw)
    fill = np.zeros(foreground.shape, dtype=bool)
    for index in range(1, int(count_raw) + 1):
        component = labeled == index
        if int(component.sum()) <= max_area:
            fill |= component
    return foreground | fill


def adjust_label_image(
    label_img: np.ndarray,
    *,
    dx: int = 0,
    dy: int = 0,
    grow_px: int = 0,
    min_size: int = 0,
    fill_holes_area: int = 0,
) -> np.ndarray:
    """Translate, size-filter, fill holes, then grow or shrink each label.

    For YX images, positive dx moves right and positive dy moves down; uncovered
    pixels become zero. grow_px is a signed pixel radius, while min_size and
    fill_holes_area count pixels. Inputs are not modified.

    Retained labels keep their IDs. At overlaps, higher IDs overwrite lower
    IDs because labels are processed in ascending order. Missing SciPy leaves
    only translation applied; morphology requires SciPy.
    """
    labels = np.asarray(label_img)
    if labels.ndim > 2:
        labels = np.squeeze(labels)
    if labels.ndim != 2:
        return labels

    adjusted = _shift_2d(labels, dx=dx, dy=dy, fill_value=0)
    grow_px = int(grow_px or 0)
    min_size = int(min_size or 0)
    fill_holes_area = int(fill_holes_area or 0)
    if grow_px == 0 and min_size <= 0 and fill_holes_area <= 0:
        return adjusted

    ndi = _ndimage()
    if ndi is None:
        return adjusted

    out = np.zeros(adjusted.shape, dtype=adjusted.dtype)
    structure = _disk_structure(abs(grow_px))
    for label_value in np.unique(adjusted):
        if int(label_value) <= 0:
            continue
        mask = adjusted == label_value
        if min_size > 0 and int(mask.sum()) < min_size:
            continue
        if fill_holes_area > 0:
            mask = _fill_holes_up_to_area(mask, fill_holes_area, ndi)
        if grow_px > 0 and structure is not None:
            mask = np.asarray(ndi.binary_dilation(mask, structure=structure), dtype=bool)
        elif grow_px < 0 and structure is not None:
            mask = np.asarray(ndi.binary_erosion(mask, structure=structure), dtype=bool)
        if np.any(mask):
            out[mask] = label_value
    return out


def adjust_mask_image(
    mask_img: np.ndarray,
    *,
    dx: int = 0,
    dy: int = 0,
    grow_px: int = 0,
    min_size: int = 0,
    fill_holes_area: int = 0,
) -> np.ndarray:
    """Shift, fill holes, grow or shrink, then filter connected foreground regions.

    Positive dx/dy move right/down without wrapping. grow_px is a signed pixel
    radius; area limits count pixels. Return an adjusted copy for 2-D input.
    Growth uses a circular footprint; size filtering uses 8-connectivity.
    Morphology binarizes positive pixels and retains the input dtype. Unlike
    label adjustments, minimum-area filtering runs after morphology here.
    """
    arr = np.asarray(mask_img)
    if arr.ndim != 2:
        return arr
    shifted = _shift_2d(arr, dx=dx, dy=dy, fill_value=0)
    grow_px = int(grow_px or 0)
    min_size = int(min_size or 0)
    fill_holes_area = int(fill_holes_area or 0)
    if grow_px == 0 and min_size <= 0 and fill_holes_area <= 0:
        return shifted

    ndi = _ndimage()
    if ndi is None:
        raise RuntimeError("SciPy is required for mask expansion, erosion, hole filling, and size filtering.")

    mask = shifted > 0
    if fill_holes_area > 0:
        mask = _fill_holes_up_to_area(mask, fill_holes_area, ndi)
    structure = _disk_structure(abs(grow_px))
    if grow_px > 0 and structure is not None:
        mask = np.asarray(ndi.binary_dilation(mask, structure=structure), dtype=bool)
    elif grow_px < 0 and structure is not None:
        mask = np.asarray(ndi.binary_erosion(mask, structure=structure), dtype=bool)
    if min_size > 0:
        labeled_raw, count_raw = cast(
            tuple[np.ndarray, int], ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
        )
        labeled = np.asarray(labeled_raw)
        keep = np.zeros(mask.shape, dtype=bool)
        for index in range(1, int(count_raw) + 1):
            component = labeled == index
            if int(component.sum()) >= min_size:
                keep |= component
        mask = keep

    max_value = int(np.max(shifted)) if shifted.size else 255
    return np.where(mask, max(1, max_value), 0).astype(arr.dtype, copy=False)
