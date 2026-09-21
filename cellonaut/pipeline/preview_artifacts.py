"""Discovery of generated files suitable for GUI previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cellonaut.results.layout import build_results_layout
from cellonaut.results.artifacts import ArtifactResolver


# Artifact collection runs after processing, when a partial export may still be
# useful; ignore paths that disappeared instead of failing the completed sample.
def _safe_path_strings(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for path in paths:
        try:
            if path.exists():
                out.append(str(path))
        except OSError:
            pass
    return out


# Return every artifact category as well as one preferred preview so the GUI can
# open a useful result immediately without losing the full Files-tab inventory.
def collect_preview_artifacts(out_dir: Path) -> dict[str, Any]:
    resolver = ArtifactResolver.load(out_dir)
    if resolver is not None:
        categories = {
            "combined_overlays": "combined_overlay",
            "qc_overlay_pngs": "overlay_png",
            "processing_montages": "montage",
            "mask_overlays": "mask_overlay",
            "final_binary_masks": "binary_mask",
            "weka_probability_maps": "probability",
            "weka_threshold_masks": "threshold",
            "cell_outline_stacks": "cell_outline_stack",
            "cell_qc_overlay_pngs": "cell_png",
            "cell_label_images": "cell_labels",
            "cell_outline_images": "cell_outline",
        }
        artifacts = {
            category: [str(resolver.path(record)) for record in resolver.matching(kind=kind)]
            for category, kind in categories.items()
        }
        preferred = next((
            paths[0] for category in (
                "combined_overlays", "cell_outline_stacks", "mask_overlays",
                "qc_overlay_pngs", "cell_qc_overlay_pngs", "processing_montages",
            ) if (paths := artifacts[category])
        ), None)
        return {"preview_path": preferred, **artifacts}
    return _collect_legacy_preview_artifacts(out_dir)


def _collect_legacy_preview_artifacts(out_dir: Path) -> dict[str, Any]:
    """Historical result folders have no writer-recorded relationships."""
    layout = build_results_layout(out_dir, create_root=False)
    qc_dir = layout["qc_overlay_pngs"]
    masks_dir = layout["masks"]
    overlays_dir = layout["mask_overlays"]
    montage_dir = layout["processing_montages"]

    qc_overlay_pngs = sorted(qc_dir.rglob("*.png")) if qc_dir.exists() else []
    combined_overlays = sorted(overlays_dir.glob("**/*_combined_overlay.tif")) if overlays_dir.exists() else []
    mask_overlays = (
        sorted(
            path
            for path in overlays_dir.glob("**/*_overlay.tif")
            if not path.name.endswith("_combined_overlay.tif")
        )
        if overlays_dir.exists()
        else []
    )

    if masks_dir.exists():
        final_binary_masks = sorted(masks_dir.glob("**/*_binary.tif"))
        weka_probability_maps = sorted(masks_dir.glob("**/*_prob_*.tif"))
        weka_threshold_masks = sorted(masks_dir.glob("**/*_thr_*.tif"))
    else:
        final_binary_masks = []
        weka_probability_maps = []
        weka_threshold_masks = []

    cell_outline_dir = layout["cell_segmentation_outlines"]
    cell_png_dir = layout["cell_segmentation_qc_pngs"]
    cell_labels_dir = layout["cell_segmentation_labels"]
    cell_outline_stacks = (
        sorted(cell_outline_dir.rglob("*_cellpose_outline_stack.tif")) if cell_outline_dir.exists() else []
    )
    cell_qc_overlay_pngs = sorted(cell_png_dir.rglob("*_qc_overlay.png")) if cell_png_dir.exists() else []
    cell_label_images = sorted(cell_labels_dir.rglob("*_cellpose_labels.tif")) if cell_labels_dir.exists() else []
    cell_outline_images = (
        sorted(cell_outline_dir.rglob("*_cellpose_outline.tif")) if cell_outline_dir.exists() else []
    )
    processing_montages = sorted(montage_dir.glob("**/*_montage.png")) if montage_dir.exists() else []

    preview_path = None
    for candidates in (
        combined_overlays,
        cell_outline_stacks,
        mask_overlays,
        qc_overlay_pngs,
        cell_qc_overlay_pngs,
        processing_montages,
    ):
        if candidates:
            preview_path = str(candidates[0])
            break

    return {
        "preview_path": preview_path,
        "combined_overlays": _safe_path_strings(combined_overlays),
        "qc_overlay_pngs": _safe_path_strings(qc_overlay_pngs),
        "processing_montages": _safe_path_strings(processing_montages),
        "mask_overlays": _safe_path_strings(mask_overlays),
        "final_binary_masks": _safe_path_strings(final_binary_masks),
        "weka_probability_maps": _safe_path_strings(weka_probability_maps),
        "weka_threshold_masks": _safe_path_strings(weka_threshold_masks),
        "cell_outline_stacks": _safe_path_strings(cell_outline_stacks),
        "cell_qc_overlay_pngs": _safe_path_strings(cell_qc_overlay_pngs),
        "cell_label_images": _safe_path_strings(cell_label_images),
        "cell_outline_images": _safe_path_strings(cell_outline_images),
    }
