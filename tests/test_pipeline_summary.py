from __future__ import annotations

from pathlib import Path

import cellonaut.pipeline.summary as summary_module
from cellonaut.pipeline.models import DiameterDefault, MeasurementTarget, Config, ImageDef
from cellonaut.config.defaults import INPUT_STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS
from cellonaut.pipeline.summary import (
    build_pipeline_summary_dict,
    build_runtime_metadata,
    describe_mask_source_for_summary,
    selected_measurements_summary,
)
from cellonaut.version import __version__


def make_config(tmp_path: Path) -> Config:
    return Config(
        fiji_app_path=tmp_path / "Fiji",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        input_structure=INPUT_STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
        images=[
            ImageDef(
                key="image1",
                label="Cell",
                folder_name="GFP",
                model_path=tmp_path / "cell.model",
                probability_class_index="1,3",
                threshold_method="Otsu",
                stack_channel_index=2,
                stack_z_mode="specific_slice",
                stack_z_index=4,
                mask_processing_steps=[
                    {"type": "binary_fill_holes", "enabled": True, "params": {}}
                ],
            ),
            ImageDef(
                key="image2",
                label="Tubules",
                folder_name="RFP",
                model_path=tmp_path / "tubules.model",
                probability_class_index=2,
            ),
        ],
        exclusion_tag="_untreated_",
        threshold_method="Default",
        probability_class_index=1,
        measurement_options={"area": True, "mean": False},
        measurement_targets=[
            MeasurementTarget(
                source_image_key="image1",
                overlay_base_image_key="image1",
                overlay_roi_keys=["image1_class1", "image1_class3", "image2"],
                do_cell_segmentation=True,
                cell_segmentation_source="image1",
                per_cell_mask_source="image1_class1",
                overlay_whole_cell_mask=True,
                measurement_options={"area": True},
                cell_diameter=42,
                cell_min_size=300,
                cell_use_gpu=False,
                cell_mask_adjustments={"dx": 2, "fill_holes_area": 6},
            )
        ],
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=["image1_class1", "image1_class3", "image2"],
        do_cell_segmentation=True,
        cell_segmentation_source="image1",
        per_cell_mask_source="image1_class1",
    )


def test_selected_measurements_summary_lists_enabled_and_disabled_options(tmp_path: Path):
    cfg = make_config(tmp_path)

    assert selected_measurements_summary(cfg) == "enabled=Area (px²) | disabled=Mean gray value (a.u.)"


def test_selected_measurements_summary_ignores_unknown_internal_keys(tmp_path: Path):
    cfg = make_config(tmp_path)
    cfg.measurement_options["save_roi_outlines"] = True

    assert "save_roi_outlines" not in selected_measurements_summary(cfg)


def test_selected_measurements_summary_empty_and_all_disabled_contract(tmp_path: Path):
    cfg = make_config(tmp_path)
    cfg.measurement_options = {}
    assert selected_measurements_summary(cfg) == "(none set; using defaults)"

    cfg.measurement_options = {"area": False, "mean": False, "internal": True}
    assert selected_measurements_summary(cfg) == (
        "enabled=(none) | disabled=Area (px²), Mean gray value (a.u.)"
    )


def test_describe_mask_source_for_summary_covers_every_source_mode():
    assert describe_mask_source_for_summary(
        {"mask_source_mode": "Weka classifier", "model_path": "cell.model"}
    ) == "Weka classifier (cell.model)"
    assert describe_mask_source_for_summary(
        {"mask_source_mode": "Weka classifier"}
    ) == "Weka classifier (not configured)"
    assert describe_mask_source_for_summary(
        {"mask_source_mode": "Combined masks", "combined_mask_source_keys": ["a", "b"]}
    ) == "Combined masks (OR; sources=['a', 'b'])"
    assert describe_mask_source_for_summary({"model_path": "classifier.model"}) == (
        "Weka classifier (classifier.model)"
    )
    assert describe_mask_source_for_summary({}) == "not configured"


def test_build_pipeline_summary_dict_preserves_probability_classes_and_relationships(tmp_path: Path):
    cfg = make_config(tmp_path)

    summary = build_pipeline_summary_dict(cfg)

    assert summary["app_version"] == __version__
    assert summary["input_structure"] == INPUT_STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS
    assert summary["images"][0]["probability_class_index"] == "1,3"
    assert summary["images"][1]["probability_class_index"] == 2
    assert summary["images"][0]["stack_channel_index"] == 2
    assert summary["images"][0]["stack_z_mode"] == "specific_slice"
    assert summary["images"][0]["mask_processing_steps"][0]["type"] == "binary_fill_holes"
    assert summary["measurement_targets"][0]["source_image_key"] == "image1"
    assert summary["measurement_targets"][0]["overlay_roi_keys"] == [
        "image1_class1",
        "image1_class3",
        "image2",
    ]
    assert summary["measurement_targets"][0]["per_cell_mask_source"] == "image1_class1"
    assert summary["measurement_targets"][0]["cell_diameter"] == 42
    assert summary["measurement_targets"][0]["cell_mask_adjustments"]["fill_holes_area"] == 6


def test_build_pipeline_summary_has_complete_top_level_and_image_schema(tmp_path: Path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg.reuse_existing_masks = True
    cfg.mask_source_dir = tmp_path / "saved-masks"
    cfg.images[0].display_color = "#12ab34"
    cfg.images[0].combined_mask_source_keys = ["image2"]
    cfg.images[0].bg_radii_csv = "10,20"
    cfg.images[0].image_processing_steps = [{"type": "blur", "enabled": True}]
    monkeypatch.setattr(summary_module, "build_runtime_metadata", lambda _cfg: {"runtime": "fixed"})

    summary = build_pipeline_summary_dict(cfg)

    assert set(summary) == {
        "app_version",
        "runtime_environment",
        "configuration_snapshot",
        "input_directory",
        "output_directory",
        "fiji_app_path",
        "input_structure",
        "threshold_method",
        "probability_class_index",
        "reuse_existing_masks",
        "mask_source_dir",
        "measurement_options_summary",
        "measurement_options",
        "cell_segmentation_defaults",
        "measurement_targets",
        "images",
    }
    assert summary["runtime_environment"] == {"runtime": "fixed"}
    assert summary["input_directory"] == str(tmp_path / "input")
    assert summary["output_directory"] == str(tmp_path / "output")
    assert summary["fiji_app_path"] == str(tmp_path / "Fiji")
    assert summary["threshold_method"] == "Default"
    assert summary["probability_class_index"] == 1
    assert summary["reuse_existing_masks"] is True
    assert summary["mask_source_dir"] == str(tmp_path / "saved-masks")
    assert summary["measurement_options"] == {"area": True, "mean": False}
    assert summary["images"][0] == {
        "key": "image1",
        "label": "Cell",
        "folder_name": "GFP",
        "model_path": str(tmp_path / "cell.model"),
        "mask_source_mode": cfg.images[0].mask_source_mode,
        "combined_mask_source_keys": ["image2"],
        "combined_mask_operation": "OR",
        "display_color": "#12ab34",
        "stack_channel_index": 2,
        "stack_z_mode": "specific_slice",
        "stack_z_index": 4,
        "bg_radii_csv": "10,20",
        "image_processing_steps": [{"type": "blur", "enabled": True}],
        "mask_processing_steps": [
            {"type": "binary_fill_holes", "enabled": True, "params": {}}
        ],
        "probability_class_index": "1,3",
        "threshold_method": "Otsu",
    }


def test_json_safe_sorts_sets_for_stable_manifests():
    assert summary_module._json_safe({"channels": {"TxRed", "DAPI", "GFP"}}) == {
        "channels": ["DAPI", "GFP", "TxRed"]
    }


def test_json_safe_recursively_converts_dataclasses_paths_tuples_and_dict_keys(tmp_path: Path):
    image = ImageDef("image1", "GFP", "GFP", tmp_path / "model.file")
    converted = summary_module._json_safe(
        {1: image, "paths": (tmp_path / "a", tmp_path / "b"), "nested": [{"z", "a"}]}
    )

    assert converted["1"]["key"] == "image1"
    assert converted["1"]["model_path"] == str(tmp_path / "model.file")
    assert converted["paths"] == [str(tmp_path / "a"), str(tmp_path / "b")]
    assert converted["nested"] == [["a", "z"]]


def test_file_fingerprint_exact_contract_for_missing_file_directory_and_file(tmp_path: Path, monkeypatch):
    folder = tmp_path / "models"
    folder.mkdir()
    model = folder / "cell.model"
    model.write_bytes(b"abc")
    monkeypatch.setattr(summary_module, "_path_mtime_utc", lambda _path: "2026-01-02T03:04:05+00:00")

    assert summary_module._file_fingerprint(tmp_path / "missing.model", "missing") == {
        "role": "missing",
        "path": str(tmp_path / "missing.model"),
        "exists": False,
    }
    assert summary_module._file_fingerprint(folder, "directory") == {
        "role": "directory",
        "path": str(folder),
        "exists": True,
        "is_file": False,
        "is_dir": True,
        "modified_utc": "2026-01-02T03:04:05+00:00",
    }
    assert summary_module._file_fingerprint(model, "classifier") == {
        "role": "classifier",
        "path": str(model),
        "exists": True,
        "is_file": True,
        "is_dir": False,
        "modified_utc": "2026-01-02T03:04:05+00:00",
        "size_bytes": 3,
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }


def test_runtime_metadata_hashes_shared_classifier_once(tmp_path: Path, monkeypatch):
    classifier = tmp_path / "shared.model"
    classifier.write_text("classifier", encoding="utf-8")
    cfg = make_config(tmp_path)
    cfg.images[0].model_path = classifier
    cfg.images[1].model_path = classifier
    calls = []
    original = summary_module._file_sha256

    def tracked_sha256(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(summary_module, "_file_sha256", tracked_sha256)

    metadata = summary_module.build_runtime_metadata(cfg)
    fingerprints = metadata["configured_file_fingerprints"]

    assert calls == [classifier]
    assert [item["role"] for item in fingerprints] == ["classifier:Cell", "classifier:Tubules"]
    assert fingerprints[0]["sha256"] == fingerprints[1]["sha256"]


def test_configured_fingerprints_preserve_roles_deduplicate_and_include_custom_models(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared.model"
    shared.write_text("shared", encoding="utf-8")
    cfg = make_config(tmp_path)
    cfg.images[0].model_path = shared
    cfg.images[1].model_path = shared
    cfg.cellpose_custom_model_path = str(shared)
    cfg.measurement_targets[0].cellpose_custom_model_path = str(shared)
    calls: list[tuple[str, str]] = []

    def fingerprint(path, role):
        calls.append((str(path), role))
        return {"role": role, "path": str(path), "exists": True, "token": "same-path"}

    monkeypatch.setattr(summary_module, "_file_fingerprint", fingerprint)

    result = summary_module._configured_file_fingerprints(cfg)

    assert calls == [(str(shared), "")]
    assert result == [
        {"role": "classifier:Cell", "path": str(shared), "exists": True, "token": "same-path"},
        {"role": "classifier:Tubules", "path": str(shared), "exists": True, "token": "same-path"},
        {"role": "cellpose_custom_model:default", "path": str(shared), "exists": True, "token": "same-path"},
        {"role": "cellpose_custom_model:image1", "path": str(shared), "exists": True, "token": "same-path"},
    ]


def test_cellpose_model_settings_exact_default_override_and_disabled_contract(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(summary_module, "cellpose_acceleration_enabled", lambda requested: requested)
    cfg = make_config(tmp_path)
    cfg.cellpose_model_type = "cpsam"
    cfg.cellpose_custom_model_path = "default.model"
    cfg.cell_diameter = 55
    cfg.cell_min_size = 600
    cfg.cellprob_threshold = -1.5
    cfg.flow_threshold = 0.7
    cfg.cell_use_gpu = True
    target = cfg.measurement_targets[0]
    target.cellpose_model_type = "custom"
    target.cellpose_custom_model_path = "target.model"
    target.cell_diameter = 31
    target.cell_min_size = 222
    target.cellprob_threshold = -0.5
    target.flow_threshold = 0.4
    target.cell_use_gpu = False
    cfg.measurement_targets.append(MeasurementTarget(source_image_key="image2", do_cell_segmentation=False))

    assert summary_module._cellpose_model_settings(cfg) == [
        {
            "scope": "default",
            "model_type": "cpsam",
            "custom_model_path": "default.model",
            "diameter": 55,
            "minimum_cell_area": 600,
            "cellprob_threshold": -1.5,
            "flow_threshold": 0.7,
            "gpu_requested": True,
            "gpu_passed_to_cellpose": True,
        },
        {
            "scope": "target:image1",
            "model_type": "custom",
            "custom_model_path": "target.model",
            "diameter": 31,
            "minimum_cell_area": 222,
            "cellprob_threshold": -0.5,
            "flow_threshold": 0.4,
            "gpu_requested": False,
            "gpu_passed_to_cellpose": False,
        },
    ]


def test_cellpose_summary_separates_gpu_request_from_cpu_install(tmp_path: Path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg.cell_use_gpu = True
    monkeypatch.setattr(summary_module, "cellpose_acceleration_enabled", lambda requested: False)

    summary = build_pipeline_summary_dict(cfg)

    assert summary["cell_segmentation_defaults"]["gpu_requested"] is True
    assert summary["cell_segmentation_defaults"]["gpu_passed_to_cellpose"] is False
    assert summary["runtime_environment"]["cellpose_model_settings"][0]["gpu_requested"] is True
    assert summary["runtime_environment"]["cellpose_model_settings"][0]["gpu_passed_to_cellpose"] is False


def test_effective_measurement_target_settings_complete_exact_contract(tmp_path: Path):
    cfg = make_config(tmp_path)
    target = cfg.measurement_targets[0]
    target.enabled = False
    target.cellprob_threshold = -0.25
    target.flow_threshold = 0.55
    target.cell_remove_border = True
    target.cellpose_model_type = "custom"
    target.cellpose_custom_model_path = "target.model"

    assert summary_module._effective_measurement_target_settings(cfg, target) == {
        "source_image_key": "image1",
        "enabled": False,
        "overlay_base_image_key": "image1",
        "overlay_roi_keys": ["image1_class1", "image1_class3", "image2"],
        "overlay_whole_cell_mask": True,
        "do_cell_segmentation": True,
        "cell_segmentation_source": "image1",
        "per_cell_mask_source": "image1_class1",
        "measurement_options": {"area": True},
        "cell_diameter": 42,
        "cell_min_size": 300,
        "cell_gpu_requested": False,
        "cell_gpu_passed_to_cellpose": False,
        "cellprob_threshold": -0.25,
        "flow_threshold": 0.55,
        "cell_remove_border": True,
        "cellpose_model_type": "custom",
        "cellpose_custom_model_path": "target.model",
        "cell_mask_adjustments": {"dx": 2, "fill_holes_area": 6},
    }


def test_runtime_metadata_has_complete_schema_with_deterministic_dependencies(tmp_path: Path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg.fiji_app_path.mkdir()
    monkeypatch.setattr(summary_module, "collect_key_package_versions", lambda: {"numpy": "2.0"})
    monkeypatch.setattr(summary_module, "_cellpose_model_settings", lambda _cfg: [{"scope": "fixed"}])
    monkeypatch.setattr(summary_module, "_configured_file_fingerprints", lambda _cfg: [{"role": "fixed"}])

    metadata = build_runtime_metadata(cfg)

    assert set(metadata) == {
        "python",
        "platform",
        "dependency_versions",
        "fiji",
        "cellpose_model_settings",
        "configured_file_fingerprints",
    }
    assert set(metadata["python"]) == {"version", "executable", "implementation"}
    assert set(metadata["platform"]) == {"system", "release", "version", "machine"}
    assert metadata["dependency_versions"] == {"numpy": "2.0"}
    assert metadata["fiji"] == {
        "app_path": str(tmp_path / "Fiji"),
        "path_exists": True,
        "imagej_version": "(selected Fiji runtime; not queried during manifest write)",
    }
    assert metadata["cellpose_model_settings"] == [{"scope": "fixed"}]
    assert metadata["configured_file_fingerprints"] == [{"role": "fixed"}]


def test_pipeline_summary_records_effective_inherited_target_settings(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(summary_module, "cellpose_acceleration_enabled", lambda requested: requested)
    cfg = make_config(tmp_path)
    target = cfg.measurement_targets[0]
    assert isinstance(target, MeasurementTarget)
    target.cell_diameter = DiameterDefault.INHERIT
    target.cell_min_size = None
    target.cell_use_gpu = None
    target.cellpose_model_type = None
    target.measurement_options = None
    cfg.cell_diameter = 64
    cfg.cell_min_size = 900
    cfg.cell_use_gpu = True
    cfg.cellpose_model_type = "cpsam_v2"
    cfg.measurement_options = {"area": True, "mean": True}

    summary = build_pipeline_summary_dict(cfg)
    effective_target = summary["measurement_targets"][0]
    runtime_target = summary["runtime_environment"]["cellpose_model_settings"][1]

    assert effective_target["cell_diameter"] == 64
    assert effective_target["cell_min_size"] == 900
    assert effective_target["cell_gpu_requested"] is True
    assert effective_target["cell_gpu_passed_to_cellpose"] is True
    assert effective_target["cellpose_model_type"] == "cpsam_v2"
    assert effective_target["measurement_options"] == {"area": True, "mean": True}
    assert runtime_target["diameter"] == 64
    assert runtime_target["model_type"] == "cpsam_v2"

    raw_target = summary["configuration_snapshot"]["measurement_targets"][0]
    assert raw_target["cell_diameter"] == "use_run_default"
    assert raw_target["cellpose_model_type"] is None
