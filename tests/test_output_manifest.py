from __future__ import annotations

import json
from pathlib import Path

from cellonaut.pipeline.output_manifest import save_run_manifest
from cellonaut.version import __version__


def test_run_manifest_records_app_version(tmp_path: Path):
    classifier = tmp_path / "classifier.model"
    classifier.write_text("fake classifier", encoding="utf-8")
    paths = save_run_manifest(
        tmp_path,
        run_type="test",
        pipeline_summary={
            "runtime_environment": {
                "python": {
                    "executable": "python.exe",
                    "version": "3.10.0",
                },
                "platform": {
                    "system": "Windows",
                    "release": "test",
                    "machine": "AMD64",
                },
                "dependency_versions": {
                    "numpy": "1.0",
                },
                "configured_file_fingerprints": [
                    {
                        "role": "classifier:Mask",
                        "path": str(classifier),
                        "exists": True,
                        "sha256": "abc123",
                    }
                ],
                "cellpose_model_settings": [
                    {
                        "scope": "default",
                        "model_type": "cpsam",
                        "custom_model_path": "",
                        "diameter": 70,
                        "minimum_cell_area": 1250,
                        "gpu_requested": True,
                        "gpu_passed_to_cellpose": False,
                    }
                ],
            },
            "configuration_snapshot": {"input_structure": "Flat TIFF files"},
            "output_directory": str(tmp_path),
            "images": [
                {
                    "key": "image1",
                    "label": "Signal",
                    "folder_name": "Signal",
                    "model_path": "",
                    "mask_source_mode": "Weka classifier",
                    "combined_mask_source_keys": [],
                    "probability_class_index": 1,
                },
                {
                    "key": "mask2",
                    "label": "Combined mask",
                    "folder_name": "",
                    "model_path": "",
                    "mask_source_mode": "Combined masks",
                    "combined_mask_source_keys": ["mask1", "mask3"],
                    "probability_class_index": 1,
                },
            ],
            "measurement_targets": [
                {
                    "source_image_key": "image1",
                    "enabled": True,
                    "overlay_roi_keys": ["mask1"],
                    "do_cell_segmentation": False,
                }
            ],
        },
        run_stats={"processed_targets": 1, "failed_targets": 0},
        sample_status_rows=[
            {
                "sample_id": "Sample_001",
                "target": "image1",
                "status": "PROCESSED",
                "reason": "",
            }
        ],
    )

    manifest = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    text = Path(paths["text"]).read_text(encoding="utf-8")

    assert manifest["app_version"] == __version__
    assert manifest["runtime_environment"]["dependency_versions"]["numpy"] == "1.0"
    assert manifest["configuration_snapshot"]["input_structure"] == "Flat TIFF files"
    assert manifest["output_directory"] == str(tmp_path)
    assert manifest["results_directory"] == str(tmp_path / "Results")
    assert manifest["cellpose_inference_devices"] == []
    assert "gpu_requested=True, gpu_passed_to_cellpose=False" in text
    assert "Cellpose inference device(s): (none recorded)" in text
    assert manifest["measurement_targets"][0]["source_image_key"] == "image1"
    assert "analysis_targets" not in manifest
    assert f"Cellonaut version: {__version__}" in text
    assert "Dependency versions" in text
    assert "mask_source=Weka classifier (not configured)" in text
    assert "mask_source=Combined masks (OR; sources=['mask1', 'mask3'])" in text
    assert "classifier:Mask" in text
    assert "Signal (image1)" in text
    assert "Sample_001 | Signal | PROCESSED" in text
    assert f"Results directory: {tmp_path / 'Results'}" in text


def test_run_manifest_records_inference_devices_from_completed_run(tmp_path: Path):
    logs = tmp_path / "Results" / "Logs"
    logs.mkdir(parents=True)
    (logs / "run_log.txt").write_text(
        "Cellpose backend: CPU fallback\nCellpose inference device: CUDA\n"
        "Cellpose inference device: CUDA\n",
        encoding="utf-8",
    )

    paths = save_run_manifest(
        tmp_path,
        run_type="full",
        inference_devices={"CPU"},
        pipeline_summary={"runtime_environment": {"python": {"version": "test"}}},
    )
    manifest = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    report = Path(paths["text"]).read_text(encoding="utf-8")

    assert manifest["cellpose_inference_devices"] == ["CPU"]
    assert "Cellpose inference device(s): CPU" in report
