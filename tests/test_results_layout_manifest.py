from __future__ import annotations

from cellonaut.results.layout import build_results_layout


def test_results_layout_is_lazy_and_does_not_create_empty_leaf_folders(tmp_path):
    layout = build_results_layout(tmp_path / "out")

    assert layout["root"] == tmp_path / "out" / "Results"
    assert layout["summaries"] == tmp_path / "out" / "Results" / "CSV Data"
    assert layout["run_manifest_txt"] == tmp_path / "out" / "Results" / "Logs" / "RunSummary.txt"
    assert layout["filtered_exports"] == tmp_path / "out" / "Results" / "Image Preview Tools" / "Cell Groups"
    assert layout["mask_overlays"] == tmp_path / "out" / "Results" / "Overlays" / "TIFF Overlays"
    assert layout["final_binary_masks"] == tmp_path / "out" / "Results" / "Masks" / "BinaryMasks"
    assert layout["cell_segmentation_qc_pngs"] == tmp_path / "out" / "Results" / "Cells" / "PNG"
    assert layout["snapshots"] == tmp_path / "out" / "Results" / "Image Preview Tools" / "Snapshots"
    assert layout["root"].is_dir()
    assert not layout["logs"].exists()
    assert not layout["summaries"].exists()
    assert not layout["filtered_exports"].exists()
    assert not layout["preview_tool_exports"].exists()
    assert not layout["qc_overlay_pngs"].exists()
    assert not layout["cell_segmentation_labels"].exists()
    assert not layout["cell_segmentation_outlines"].exists()
    assert not layout["weka_probability_maps"].exists()
    assert not layout["weka_threshold_masks"].exists()
    assert not layout["final_binary_masks"].exists()
    assert not layout["mask_skeletons"].exists()
    assert not layout["mask_overlays"].exists()
    assert not layout["processing_montages"].exists()


def test_results_layout_keeps_only_run_manifest_as_top_level_helper(tmp_path):
    layout = build_results_layout(tmp_path / "out")

    assert layout["run_manifest_txt"] == tmp_path / "out" / "Results" / "Logs" / "RunSummary.txt"
    assert layout["run_manifest_json"] == tmp_path / "out" / "Results" / "Logs" / "RunSummary.json"
    assert layout["run_state_json"] == tmp_path / "out" / "Results" / "Logs" / "RunState.json"
    assert "analysis_summary_csv" not in layout
    assert "data_dictionary_csv" not in layout
    assert "warnings_csv" not in layout
    assert "manifest_csv" not in layout
    assert "manifest_json" not in layout


def test_results_layout_accepts_existing_results_folder_without_creating_it(tmp_path):
    results_dir = tmp_path / "earlier_run" / "Results"

    layout = build_results_layout(results_dir, create_root=False)

    assert layout["root"] == results_dir
    assert layout["snapshots"] == results_dir / "Image Preview Tools" / "Snapshots"
    assert not results_dir.exists()


def test_results_layout_contains_only_canonical_keys(tmp_path):
    layout = build_results_layout(tmp_path / "out", create_root=False)

    assert not {
        "tables",
        "measurements",
        "per_cell",
        "weka",
        "weka_binary_masks",
        "weka_overlays",
        "flat_overlays",
        "cell_segmentation_overlays",
    }.intersection(layout)
