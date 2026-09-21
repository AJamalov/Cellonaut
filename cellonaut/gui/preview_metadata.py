"""Pure metadata helpers used by the preview tab.

The preview mixin is intentionally UI-heavy. Keeping TIFF/OME sidecar parsing
here allows metadata behavior to be tested independently of widgets.
"""

from __future__ import annotations

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
from xml.etree.ElementTree import ParseError

import numpy as np
import tifffile

from cellonaut.io.image_io import ome_color_to_hex


# Return a numeric triplet because NumPy compositing uses the same helper for
# sidecar colors and user-selected layer colors.
def parse_hex_color(value: str, fallback=(255, 0, 255)) -> np.ndarray:
    try:
        text = str(value).strip()
        if text.startswith("#") and len(text) == 7:
            return np.array(
                [int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)],
                dtype=np.float32,
            )
    except (TypeError, ValueError):
        pass
    return np.array(fallback, dtype=np.float32)


# Trust only Cellonaut's explicit sidecar types before applying additive layer
# rendering; ordinary RGB TIFFs must retain their natural color interpretation.
def is_overlay_preview_sidecar(sidecar: dict) -> bool:
    sidecar_type = str(sidecar.get("type", "")).strip().lower()
    return sidecar_type in {
        "cellonaut_overlay_preview",
        "cellonaut_cellpose_outline_stack",
    }


# Read only channel presentation metadata from OME XML so preview loading does
# not depend on a full microscopy metadata model.
def parse_ome_display_sidecar(ome_xml: str) -> dict:
    try:
        root = ET.fromstring(ome_xml)
    except (ParseError, DefusedXmlException, TypeError):
        return {}

    labels: list[str] = []
    colors: list[str] = []
    for channel in root.findall(".//{*}Channel"):
        name = channel.attrib.get("Name") or f"Channel {len(labels) + 1}"
        labels.append(str(name))
        color_value = channel.attrib.get("Color")
        if color_value is not None:
            try:
                colors.append(ome_color_to_hex(color_value))
            except (TypeError, ValueError, OverflowError):
                colors.append("#FFFFFF")

    sidecar: dict = {}
    if labels:
        sidecar["layer_labels"] = labels
    if colors:
        sidecar["layer_colors"] = colors
    return sidecar


# Prefer embedded ImageJ labels/LUTs and fall back to OME metadata because files
# written by Fiji and microscope software store equivalent display data differently.
def load_tiff_imagej_display_sidecar(file_path: str) -> dict:
    sidecar: dict = {}
    try:
        with tifffile.TiffFile(file_path) as tif:
            series = tif.series[0]
            axes = str(getattr(series, "axes", "") or "")
            metadata = tif.imagej_metadata or {}
            ome_xml = tif.ome_metadata
    except Exception:
        return {}

    if axes:
        sidecar["axes"] = axes

    labels = metadata.get("Labels")
    if isinstance(labels, (list, tuple)):
        sidecar["layer_labels"] = [str(label) for label in labels]
    elif labels:
        sidecar["layer_labels"] = [str(labels)]

    luts = metadata.get("LUTs")
    if isinstance(luts, np.ndarray) and luts.ndim == 2:
        luts = [luts]
    if isinstance(luts, (list, tuple)):
        colors = []
        for lut in luts:
            lut_arr = np.asarray(lut)
            if lut_arr.shape[0] >= 3 and lut_arr.shape[-1] >= 256:
                colors.append(
                    "#{:02X}{:02X}{:02X}".format(
                        int(lut_arr[0, 255]),
                        int(lut_arr[1, 255]),
                        int(lut_arr[2, 255]),
                    )
                )
        if colors:
            sidecar["layer_colors"] = colors

    if str(metadata.get("mode", "")).lower() == "composite":
        sidecar["type"] = "cellonaut_overlay_preview"
    if ome_xml and not any(key in sidecar for key in ("layer_labels", "layer_colors")):
        ome_sidecar = parse_ome_display_sidecar(ome_xml)
        sidecar.update(ome_sidecar)
    return sidecar
