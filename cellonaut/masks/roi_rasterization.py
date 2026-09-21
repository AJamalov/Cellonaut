"""Shared conversion of ImageJ ROI geometry into pixel masks."""

from __future__ import annotations

from typing import Any

import numpy as np

from cellonaut.io.imagej_runtime import get_java_classes


# ImageJ coordinate rules live in one place so overlays and montages cannot disagree about ROI pixels.
def imagej_roi_to_mask_image(ref_img: Any, roi: Any, title: str):
    classes = get_java_classes()
    ImagePlus = classes["ImagePlus"]
    ByteProcessor = classes["ByteProcessor"]

    if isinstance(ref_img, np.ndarray):
        height, width = np.asarray(ref_img).shape[:2]
    else:
        width = int(ref_img.getWidth())
        height = int(ref_img.getHeight())

    processor = ByteProcessor(width, height)
    mask = roi.getMask()
    bounds = roi.getBounds()

    if mask is not None:
        for yy in range(mask.getHeight()):
            for xx in range(mask.getWidth()):
                if mask.get(xx, yy) == 0:
                    continue
                x = bounds.x + xx
                y = bounds.y + yy
                if 0 <= x < width and 0 <= y < height:
                    processor.set(x, y, 255)
    else:
        # Rectangular and some area ROIs have no separate mask. contains()
        # fills their interior instead of reducing them to polygon vertices.
        for y in range(max(0, bounds.y), min(height, bounds.y + bounds.height)):
            for x in range(max(0, bounds.x), min(width, bounds.x + bounds.width)):
                if roi.contains(x, y):
                    processor.set(x, y, 255)

    return ImagePlus(title, processor)
