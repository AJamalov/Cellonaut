"""Canonical locations for files written below a Cellonaut output folder."""

from __future__ import annotations

from pathlib import Path


def results_root(output_dir: Path) -> Path:
    """Accept an output folder or its Results folder and return the Results path."""
    output_dir = Path(output_dir)
    if output_dir.name.lower() == "results" or (output_dir / "Logs" / "RunSummary.json").is_file():
        return output_dir
    return output_dir / "Results"


# Producers create subfolders as needed, keeping unused output folders absent.
def build_results_layout(output_dir: Path, *, create_root: bool = True) -> dict[str, Path]:
    root = results_root(output_dir)
    csv_data = root / "CSV Data"
    overlays = root / "Overlays"
    cells = root / "Cells"
    masks = root / "Masks"
    layout = {
        "root": root,
        "logs": root / "Logs",
        "summaries": csv_data,
        "channel_tables": csv_data / "Per-Channel Tables",
        "preview_tool_exports": root / "Image Preview Tools",
        "filtered_exports": root / "Image Preview Tools" / "Cell Groups",
        "cell_signal_tables": csv_data / "Signal",
        "cell_geometry_tables": csv_data / "Geometry",
        "mask_overlays": overlays / "TIFF Overlays",
        "qc_overlay_pngs": overlays / "PNG",
        "mask_overlay_metadata": overlays / "JSON",
        "cell_segmentation": cells,
        "cell_segmentation_labels": cells / "TIFF Labels",
        "cell_segmentation_outlines": cells / "TIFF Outlines",
        "cell_segmentation_qc_pngs": cells / "PNG",
        "cell_segmentation_metadata": cells / "JSON",
        "cell_segmentation_tables": csv_data / "Cell Measurements",
        "masks": masks,
        "weka_probability_maps": masks / "ProbabilityMaps",
        "weka_threshold_masks": masks / "MaskImages",
        "final_binary_masks": masks / "BinaryMasks",
        "mask_skeletons": masks / "Skeletons",
        "processing_montages": root / "Processing Montages",
        "snapshots": root / "Image Preview Tools" / "Snapshots",
        "run_manifest_json": root / "Logs" / "RunSummary.json",
        "run_manifest_txt": root / "Logs" / "RunSummary.txt",
        "run_state_json": root / "Logs" / "RunState.json",
    }
    if create_root:
        root.mkdir(parents=True, exist_ok=True)
    return layout
