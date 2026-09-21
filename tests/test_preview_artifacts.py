from __future__ import annotations

from cellonaut.pipeline.preview_artifacts import collect_preview_artifacts


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def test_collect_preview_artifacts_uses_common_results_folders_and_cell_segmentation_overlay(tmp_path):
    results_dir = tmp_path / "Results"
    cell_overlay = results_dir / "Cells" / "TIFF Outlines" / "sample_cellpose_outline_stack.tif"
    cell_qc = results_dir / "Cells" / "PNG" / "sample_04_qc_overlay.png"
    cell_labels = results_dir / "Cells" / "TIFF Labels" / "sample_01_cellpose_labels.tif"
    cell_outline = results_dir / "Cells" / "TIFF Outlines" / "sample_cellpose_outline.tif"
    qc_png = results_dir / "Overlays" / "PNG" / "sample_flat_overlay.png"
    touch(cell_overlay)
    touch(cell_qc)
    touch(cell_labels)
    touch(cell_outline)
    touch(qc_png)

    artifacts = collect_preview_artifacts(results_dir)

    assert artifacts["preview_path"] == str(cell_overlay)
    assert artifacts["cell_outline_stacks"] == [str(cell_overlay)]
    assert artifacts["cell_qc_overlay_pngs"] == [str(cell_qc)]
    assert artifacts["cell_label_images"] == [str(cell_labels)]
    assert artifacts["cell_outline_images"] == [str(cell_outline)]
    assert artifacts["qc_overlay_pngs"] == [str(qc_png)]


def test_collect_preview_artifacts_prefers_cell_segmentation_overlay_when_available(tmp_path):
    results_dir = tmp_path / "Results"
    cell_overlay = results_dir / "Cells" / "TIFF Outlines" / "sample_cellpose_outline_stack.tif"
    weka_overlay = results_dir / "Masks" / "sample_base_overlay.tif"
    touch(cell_overlay)
    touch(weka_overlay)

    artifacts = collect_preview_artifacts(results_dir)

    assert artifacts["preview_path"] == str(cell_overlay)


def test_collect_preview_artifacts_prefers_combined_overlay(tmp_path):
    results_dir = tmp_path / "Results"
    combined_overlay = results_dir / "Overlays" / "TIFF Overlays" / "sample_signal_combined_overlay.tif"
    cell_overlay = results_dir / "Cells" / "TIFF Outlines" / "sample_cellpose_outline_stack.tif"
    touch(combined_overlay)
    touch(cell_overlay)

    artifacts = collect_preview_artifacts(results_dir)

    assert artifacts["preview_path"] == str(combined_overlay)
    assert artifacts["combined_overlays"] == [str(combined_overlay)]


def test_collect_preview_artifacts_prefers_combined_overlay_over_processing_montage(tmp_path):
    results_dir = tmp_path / "Results"
    montage = results_dir / "Processing Montages" / "sample_01" / "sample_01_GFP_segmentation_montage.png"
    combined_overlay = results_dir / "Overlays" / "TIFF Overlays" / "sample_signal_combined_overlay.tif"
    touch(montage)
    touch(combined_overlay)

    artifacts = collect_preview_artifacts(results_dir)

    assert artifacts["preview_path"] == str(combined_overlay)
    assert artifacts["processing_montages"] == [str(montage)]


def test_collect_preview_artifacts_finds_mask_overlays_in_overlay_folder(tmp_path):
    results_dir = tmp_path / "Results"
    weka_overlay = results_dir / "Overlays" / "TIFF Overlays" / "sample_mask_overlay.tif"
    combined_overlay = results_dir / "Overlays" / "TIFF Overlays" / "sample_GFP_combined_overlay.tif"
    touch(weka_overlay)
    touch(combined_overlay)

    artifacts = collect_preview_artifacts(results_dir)

    assert artifacts["mask_overlays"] == [str(weka_overlay)]
    assert artifacts["combined_overlays"] == [str(combined_overlay)]


def test_collect_preview_artifacts_finds_outputs_nested_by_source_folder(tmp_path):
    results_dir = tmp_path / "Results"
    qc_png = results_dir / "Overlays" / "PNG" / "Batch1" / "sample_qc.png"
    cell_png = results_dir / "Cells" / "PNG" / "Batch1" / "sample_04_qc_overlay.png"
    cell_labels = results_dir / "Cells" / "TIFF Labels" / "Batch1" / "sample_cellpose_labels.tif"
    touch(qc_png)
    touch(cell_png)
    touch(cell_labels)

    artifacts = collect_preview_artifacts(results_dir)

    assert artifacts["qc_overlay_pngs"] == [str(qc_png)]
    assert artifacts["cell_qc_overlay_pngs"] == [str(cell_png)]
    assert artifacts["cell_label_images"] == [str(cell_labels)]
