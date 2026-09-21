"""Helpers for applying detected physical image channels to the GUI."""

from __future__ import annotations


# TIFF and ND2 discovery both replace the same channel state. Apply the complete
# synchronization sequence here so either importer preserves analysis references.
def apply_detected_image_definitions(gui, image_defs: list[dict], old_names: list[str], new_names: list[str]) -> None:
    if hasattr(gui, "remap_analysis_references_for_detected_folders"):
        gui.remap_analysis_references_for_detected_folders(image_defs, old_names, new_names)

    gui.image_definitions = gui.normalize_image_definitions(image_defs)
    gui.rebuild_image_rows(sync_from_ui=False)
    gui.refresh_channel_name_dependent_ui()

    if hasattr(gui, "analysis_matrix_table"):
        gui.build_analysis_matrix_for_current_source()
    if hasattr(gui, "update_analysis_matrix_warning_label"):
        gui.update_analysis_matrix_warning_label()
