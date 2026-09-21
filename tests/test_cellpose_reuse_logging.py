from __future__ import annotations

from pathlib import Path

from cellonaut.masks.cellpose import load_existing_cellpose_labels
from cellonaut.pipeline.discovery import build_common_results_export_dirs


def test_load_existing_cellpose_labels_logs_unreadable_mask(tmp_path: Path):
    export_dirs = build_common_results_export_dirs(tmp_path)
    labels_path = (
        export_dirs["cell_segmentation_labels"]
        / "SampleA_Cell_01_cellpose_labels.tif"
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("not a TIFF", encoding="utf-8")
    logs = []

    labels = load_existing_cellpose_labels(
        tmp_path,
        "SampleA",
        "Cell",
        log_func=logs.append,
    )

    assert labels is None
    assert any(str(labels_path) in message for message in logs)
    assert any("Could not read existing Cellpose labels" in message for message in logs)
