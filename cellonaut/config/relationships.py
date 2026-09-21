"""Keep channel and mask relationships valid while names and rows change."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cellonaut.config.defaults import MASK_SOURCE_MODE_COMBINED


# Relationship tables use display names as keys, so even an unfinished channel
# needs a stable fallback name rather than an empty string.
def image_display_name(image_def: dict[str, Any], index: int) -> str:
    return str(image_def.get("name", "") or f"Channel {index + 1}").strip() or f"Channel {index + 1}"


# Derive every relationship name through the same fallback rule to keep matrix
# keys consistent across rebuilding, saving, and validation.
def image_display_names(image_defs: list[dict[str, Any]]) -> list[str]:
    return [image_display_name(image_def, index) for index, image_def in enumerate(image_defs)]


# Whitespace-only paths are treated as unconfigured because they cannot produce
# a usable Weka mask later in the pipeline.
def image_has_classifier(image_def: dict[str, Any]) -> bool:
    return bool(str(image_def.get("classifier", "") or "").strip())


# Use the same mask-mode names in the GUI and configuration checks.
def image_uses_combined_mask(image_def: dict[str, Any]) -> bool:
    return str(image_def.get("mask_source_mode", "") or "").strip().casefold() == MASK_SOURCE_MODE_COMBINED.casefold()


def image_uses_non_classifier_mask(image_def: dict[str, Any]) -> bool:
    return image_uses_combined_mask(image_def)


# Offer only configured mask sources, even if other mask slots are visible.
def image_produces_mask(image_def: dict[str, Any]) -> bool:
    if not bool(image_def.get("is_mask_only", False)) and not bool(image_def.get("mask_slot_enabled", True)):
        return False
    combined = image_uses_combined_mask(image_def) and len(list(image_def.get("combined_mask_sources", []) or [])) >= 2
    return image_has_classifier(image_def) or combined


def configured_mask_names(image_defs: list[dict[str, Any]]) -> set[str]:
    """Resolve runnable basic and nested combined masks without accepting cycles."""

    names = image_display_names(image_defs)
    ready = {
        name
        for name, image_def in zip(names, image_defs)
        if not image_uses_combined_mask(image_def)
        and image_has_classifier(image_def)
    }
    changed = True
    while changed:
        changed = False
        for name, image_def in zip(names, image_defs):
            if name in ready or not image_uses_combined_mask(image_def):
                continue
            sources = {
                str(source or "").strip()
                for source in list(image_def.get("combined_mask_sources", []) or [])
                if str(source or "").strip() in ready and str(source or "").strip() != name
            }
            if len(sources) >= 2:
                ready.add(name)
                changed = True
    return ready


# Analysis matrices should contain only targets that can produce a mask, while
# preserving the user's channel order for predictable columns.
def classifier_target_names(image_defs: list[dict[str, Any]]) -> list[str]:
    names = image_display_names(image_defs)
    configured = configured_mask_names(image_defs)
    return [name for name in names if name in configured]


# Channel names are persisted inside several related fields. Remap them as one
# operation so a rename cannot leave only part of the analysis configuration stale.
def remap_analysis_references(
    image_defs: list[dict[str, Any]],
    old_names: list[str],
    new_names: list[str],
) -> list[dict[str, Any]]:
    updated_defs = deepcopy(image_defs)
    name_map = {
        old.strip(): new.strip() for old, new in zip(old_names, new_names) if old and new and old.strip() != new.strip()
    }
    valid_new_names = {name.strip() for name in new_names if name.strip()}

    # Invalid references fall back to a context-appropriate value instead of
    # surviving as invisible selections in the rebuilt GUI.
    def remap_name(value: str, fallback: str) -> str:
        value = str(value or "").strip()
        return name_map.get(value, value if value in valid_new_names else fallback)

    # Renames can collapse two old keys into one. OR preserves an enabled
    # relationship if either original key was selected.
    def remap_matrix(matrix: dict, fallback_name: str) -> dict[str, bool]:
        if not isinstance(matrix, dict):
            matrix = {}
        remapped: dict[str, bool] = {}
        for key, checked in matrix.items():
            new_key = remap_name(str(key), fallback_name)
            if new_key:
                remapped[new_key] = bool(checked) or bool(remapped.get(new_key, False))
        return remapped

    for index, image_def in enumerate(updated_defs):
        own_name = image_display_name(image_def, index)
        image_def["analysis_cell_segmentation_source"] = remap_name(
            image_def.get("analysis_cell_segmentation_source", ""),
            own_name,
        )
        image_def["cell_group_mask_source"] = remap_name(
            image_def.get("cell_group_mask_source", ""),
            "",
        )
        image_def["mask_source_channel"] = remap_name(
            image_def.get("mask_source_channel", ""),
            "",
        )
        image_def["mask_relationships"] = remap_matrix(
            image_def.get("mask_relationships", {}),
            own_name,
        )
        image_def["combined_mask_sources"] = [
            remap_name(source, "")
            for source in list(image_def.get("combined_mask_sources", []) or [])
            if remap_name(source, "")
        ]

    return updated_defs


# A new configuration should immediately produce one meaningful analysis, but
# an existing set of user choices must never be overwritten by that default.
def seed_first_classifier_self_relationship(image_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_defs = deepcopy(image_defs)
    has_relationship = any(
        any(dict(image_def.get("mask_relationships", {}) or {}).values()) for image_def in updated_defs
    )
    if has_relationship:
        return updated_defs

    for index, image_def in enumerate(updated_defs):
        if image_produces_mask(image_def):
            own_name = image_display_name(image_def, index)
            image_def["mask_relationships"] = {own_name: True}
            break

    return updated_defs


# Channels and masks can be removed or change source type after relationships
# are saved. Normalize a copy so stale references cannot reach the pipeline.
def normalize_analysis_relationships(image_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_defs = deepcopy(image_defs)
    names = image_display_names(updated_defs)
    valid_target_names = set(classifier_target_names(updated_defs))
    valid_channel_names = [
        name
        for name, image_def in zip(names, updated_defs)
        if not bool(image_def.get("is_mask_only", False)) and not image_uses_combined_mask(image_def)
    ]
    valid_channel_name_set = set(valid_channel_names)
    available_mask_names = configured_mask_names(updated_defs)

    for index, image_def in enumerate(updated_defs):
        own_name = names[index]
        mask_source_channel = str(image_def.get("mask_source_channel", "") or "").strip()
        if bool(image_def.get("is_mask_only", False)) and mask_source_channel not in valid_channel_name_set:
            image_def["mask_source_channel"] = valid_channel_names[0] if valid_channel_names else ""

        image_def["combined_mask_sources"] = [
            source
            for source in list(image_def.get("combined_mask_sources", []) or [])
            if source in available_mask_names and source != own_name
        ]

        cell_source = str(image_def.get("analysis_cell_segmentation_source", "") or "").strip()
        if cell_source not in valid_channel_name_set:
            image_def["analysis_cell_segmentation_source"] = own_name

        relationships_in = dict(image_def.get("mask_relationships", {}) or {})
        relationships = {
            target_name: bool(relationships_in.get(target_name, False)) if target_name in valid_target_names else False
            for target_name in names
        }
        image_def["mask_relationships"] = relationships
        selected_masks = [name for name, checked in relationships.items() if checked]
        cell_group_mask_source = str(image_def.get("cell_group_mask_source", "") or "").strip()
        image_def["cell_group_mask_source"] = (
            cell_group_mask_source
            if cell_group_mask_source in selected_masks
            else (selected_masks[0] if selected_masks else "")
        )

    return updated_defs
