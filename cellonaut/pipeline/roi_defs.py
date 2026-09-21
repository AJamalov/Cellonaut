"""Expand configured masks into stable ROI definitions used during processing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
from typing import Any, Sequence


# Preserve the user's class order because it also controls result and overlay ordering.
def dedupe_preserve_order(values: Sequence[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


# Runtime parsing stays forgiving so an older in-memory value cannot crash mask generation.
def parse_probability_class_indices(value: Any, fallback: int = 1) -> list[int]:
    text = str(value if value is not None else "").strip()
    if not text:
        text = str(fallback)

    indices: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(float(left.strip()))
                end = int(float(right.strip()))
            except (TypeError, ValueError, OverflowError):
                continue
            if start > end:
                start, end = end, start
            indices.extend(range(start, end + 1))
            continue

        try:
            indices.append(int(float(part)))
        except (TypeError, ValueError, OverflowError):
            continue

    return dedupe_preserve_order([max(1, idx) for idx in indices]) or [max(1, int(fallback))]


# Validation is deliberately stricter than runtime parsing so values such as 1.5 are not silently changed to class 1.
def probability_class_indices_are_valid(value: Any) -> bool:
    text = str(value if value is not None else "").strip()
    if not text:
        return False

    parts = [part.strip() for part in re.split(r"[,;]", text)]
    if not parts or any(not part for part in parts):
        return False

    for part in parts:
        single_match = re.fullmatch(r"\d+", part)
        if single_match:
            if int(part) < 1:
                return False
            continue

        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if range_match is None or any(int(bound) < 1 for bound in range_match.groups()):
            return False

    return True


# Class-specific keys keep several Weka probability classes distinct without changing the base channel key.
def class_roi_key(base_key: str, class_index: int) -> str:
    return f"{base_key}__class{int(class_index)}"


# Only keys created by class_roi_key should be interpreted as expanded Weka masks.
def parse_class_roi_key(key: str) -> tuple[str, int] | None:
    marker = "__class"
    if marker not in str(key):
        return None
    base_key, class_text = str(key).rsplit(marker, 1)
    if not base_key or not class_text:
        return None
    try:
        return base_key, int(class_text)
    except ValueError:
        return None


# Copy the complete validated definition so class expansion cannot drop newer
# processing, source, or display settings when ImageDef evolves.
def make_class_roi_def(base_def: Any, class_index: int):
    return replace(
        deepcopy(base_def),
        key=class_roi_key(base_def.key, class_index),
        label=f"{base_def.label}_Class{int(class_index)}",
        probability_class_index=int(class_index),
    )


# A single class keeps the original object identity; only multi-class masks need synthetic definitions.
def expand_image_def_to_class_roi_defs(img_def: Any) -> list[Any]:
    if getattr(img_def, "model_path", None) is None:
        return [img_def]
    class_indices = parse_probability_class_indices(img_def.probability_class_index)
    if len(class_indices) <= 1:
        return [img_def]
    return [make_class_roi_def(img_def, class_index) for class_index in class_indices]


def mask_reference_image_keys(source_keys: Sequence[str], image_defs: Sequence[Any]) -> list[str]:
    """Resolve combined-mask dependencies to image-backed masks in source order."""
    definitions = {image.key: image for image in image_defs}
    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(key: str) -> None:
        class_info = parse_class_roi_key(key)
        key = class_info[0] if class_info is not None else key
        if key in visiting:
            raise ValueError(f"Combined mask dependency cycle at {key}")
        definition = definitions.get(key)
        sources = list(getattr(definition, "combined_mask_source_keys", []) or [])
        if not sources:
            if key not in resolved:
                resolved.append(key)
            return
        visiting.add(key)
        for source in sources:
            visit(source)
        visiting.remove(key)

    for source in source_keys:
        visit(source)
    return resolved
