from __future__ import annotations

from pathlib import Path

import pytest

from cellonaut.pipeline import discovery
from cellonaut.io.image_io import find_channel_files
from cellonaut.pipeline.models import Config, ImageDef, MeasurementTarget
from cellonaut.pipeline.readiness import analyze_sample_readiness
from cellonaut.pipeline.discovery import (
    STRUCTURE_GROUPED_BY_PROTEIN,
    STRUCTURE_FLAT_TIFFS,
    STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    build_common_results_export_dirs,
    build_result_id,
    validate_unique_result_ids,
    detect_image_folders_from_input,
    get_measurement_sample_paths,
    get_image_folder_flat_tiff_sample_stems,
    get_sample_paths_for_structure,
    nest_export_dirs_for_sample,
)


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real tiff, only a path fixture")


def make_config(tmp_path: Path, images: list[ImageDef], input_structure: str) -> Config:
    return Config(
        fiji_app_path=tmp_path / "Fiji",
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        input_structure=input_structure,
        images=images,
        exclusion_tag="_ut_",
        threshold_method="Default",
        probability_class_index=1,
    )


def test_image_folder_flat_tiff_sample_stems_are_union_of_channel_folders(tmp_path: Path):
    touch(tmp_path / "GFP" / "sample_001.tif")
    touch(tmp_path / "GFP" / "sample_002.tiff")
    touch(tmp_path / "DAPI" / "sample_001.tif")
    touch(tmp_path / "DAPI" / "notes.txt")
    touch(tmp_path / "DAPI" / "._sample_003.tif")

    stems = get_image_folder_flat_tiff_sample_stems(tmp_path, ["GFP", "DAPI", "Missing"])

    assert stems == ["sample_001", "sample_002"]


def test_analysis_sample_paths_for_image_folder_flat_tiffs_use_sample_stems(tmp_path: Path):
    touch(tmp_path / "GFP" / "sample_001.tif")
    touch(tmp_path / "DAPI" / "sample_002.tif")
    cfg = make_config(
        tmp_path,
        [
            ImageDef("image1", "GFP", "GFP", None),
            ImageDef("image2", "DAPI", "DAPI", None),
        ],
        STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    )

    sample_paths = get_measurement_sample_paths(cfg)

    assert [path.name for path in sample_paths] == ["sample_001.tif", "sample_002.tif"]
    assert all(path.parent == tmp_path for path in sample_paths)


def test_find_channel_files_pairs_matching_tiff_names_across_image_folders(tmp_path: Path):
    gfp = tmp_path / "GFP" / "sample_001.tif"
    dapi = tmp_path / "DAPI" / "sample_001.tiff"
    touch(gfp)
    touch(dapi)
    cfg = make_config(
        tmp_path,
        [
            ImageDef("image1", "GFP", "GFP", None),
            ImageDef("image2", "DAPI", "DAPI", None),
        ],
        STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    )

    files = find_channel_files(tmp_path / "sample_001.tif", cfg, "sample_001", lambda _msg: None)

    assert files == {"image1": gfp, "image2": dapi}


def test_find_channel_files_skips_missing_needed_image_folder_match(tmp_path: Path):
    touch(tmp_path / "GFP" / "sample_001.tif")
    cfg = make_config(
        tmp_path,
        [
            ImageDef("image1", "GFP", "GFP", None),
            ImageDef("image2", "DAPI", "DAPI", None),
        ],
        STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    )

    files = find_channel_files(
        tmp_path / "sample_001.tif",
        cfg,
        "sample_001",
        lambda _msg: None,
        required_image_keys={"image1", "image2"},
    )

    assert files is None


def test_sample_readiness_uses_same_needed_file_logic_as_pipeline(tmp_path: Path):
    touch(tmp_path / "GFP" / "sample_001.tif")
    cfg = make_config(
        tmp_path,
        [
            ImageDef("image1", "GFP", "GFP", None),
            ImageDef("image2", "DAPI", "DAPI", None),
        ],
        STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    )

    cfg.measurement_targets = [MeasurementTarget(source_image_key="image2")]

    readiness = analyze_sample_readiness(cfg)

    assert readiness["total"] == 1
    assert readiness["ready"] == 0
    assert readiness["blocked"] == 1
    assert readiness["problems"][0]["sample"] == "sample_001"
    assert "DAPI" in readiness["problems"][0]["missing_required"][0]


def test_sample_readiness_reports_unused_missing_channel_without_blocking(tmp_path: Path):
    touch(tmp_path / "GFP" / "sample_001.tif")
    cfg = make_config(
        tmp_path,
        [
            ImageDef("image1", "GFP", "GFP", None),
            ImageDef("image2", "DAPI", "DAPI", None),
        ],
        STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    )

    cfg.measurement_targets = [MeasurementTarget(source_image_key="image1")]

    readiness = analyze_sample_readiness(cfg)

    assert readiness["total"] == 1
    assert readiness["ready"] == 1
    assert readiness["blocked"] == 0
    assert readiness["problems"][0]["missing_optional"] == ["DAPI"]


def test_flat_tiff_mode_reuses_same_file_for_all_images(tmp_path: Path):
    sample = tmp_path / "sample_001.tif"
    touch(sample)
    cfg = make_config(
        tmp_path,
        [
            ImageDef("image1", "Image1", "Image1", None),
            ImageDef("image2", "Image2", "Image2", None),
        ],
        STRUCTURE_FLAT_TIFFS,
    )

    files = find_channel_files(sample, cfg, "sample_001", lambda _msg: None)

    assert files == {"image1": sample, "image2": sample}


def test_flat_tiff_mode_recurses_into_named_folders_and_preserves_result_identity(tmp_path: Path):
    sample_a = tmp_path / "ConditionA" / "sample_001.tif"
    sample_b = tmp_path / "ConditionB" / "nested" / "sample_002.tif"
    touch(sample_a)
    touch(sample_b)

    sample_paths = get_sample_paths_for_structure(tmp_path, STRUCTURE_FLAT_TIFFS)

    assert [path.relative_to(tmp_path).as_posix() for path in sample_paths] == [
        "ConditionA/sample_001.tif",
        "ConditionB/nested/sample_002.tif",
    ]
    assert (
        build_result_id(
            sample_b,
            STRUCTURE_FLAT_TIFFS,
            input_dir=tmp_path,
        )
        == "ConditionB__nested__sample_002"
    )


def test_recursive_tiff_scan_ignores_revisited_directory_aliases(tmp_path: Path, monkeypatch):
    child = tmp_path / "child"
    image = child / "sample.tif"
    touch(image)
    real_safe_iterdir = discovery.safe_iterdir

    def cyclic_iterdir(folder: Path) -> list[Path]:
        if folder == child:
            return [image, tmp_path]
        return real_safe_iterdir(folder)

    monkeypatch.setattr(discovery, "safe_iterdir", cyclic_iterdir)

    assert discovery.contains_supported_tiff(tmp_path, recursive=True)
    assert discovery.iter_supported_tiff_files(tmp_path, recursive=True) == [image]


def test_common_result_export_dirs_can_be_nested_for_recursive_flat_samples(tmp_path: Path):
    sample = tmp_path / "Batch1" / "sample_001.tif"
    touch(sample)
    export_dirs = build_common_results_export_dirs(tmp_path / "out")

    nested = nest_export_dirs_for_sample(export_dirs, tmp_path, sample, STRUCTURE_FLAT_TIFFS)

    assert nested["weka_threshold_masks"] == tmp_path / "out" / "Results" / "Masks" / "MaskImages" / "Batch1"
    assert nested["cell_segmentation_qc_pngs"] == tmp_path / "out" / "Results" / "Cells" / "PNG" / "Batch1"


def test_shared_discovery_detects_image_folders_with_unsorted_tiffs(tmp_path: Path):
    touch(tmp_path / "GFP" / "cell_001.tif")
    touch(tmp_path / "DAPI" / "cell_001.tif")

    detection = detect_image_folders_from_input(
        tmp_path,
        expected_folders=["GFP", "DAPI"],
    )
    sample_paths = get_sample_paths_for_structure(
        tmp_path,
        STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
        ["GFP", "DAPI"],
    )

    assert detection["structure"] == STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS
    assert detection["folders"] == ["GFP", "DAPI"]
    assert [path.name for path in sample_paths] == ["cell_001.tif"]


def test_detection_preserves_expected_image_folder_order(tmp_path: Path):
    touch(tmp_path / "Z_Channel" / "cell_001.tif")
    touch(tmp_path / "A_Channel" / "cell_001.tif")

    detection = detect_image_folders_from_input(
        tmp_path,
        expected_folders=["Z_Channel", "A_Channel"],
    )

    assert detection["structure"] == STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS
    assert detection["folders"] == ["Z_Channel", "A_Channel"]


def test_grouped_sample_enumeration_finds_nested_samples(tmp_path: Path):
    touch(tmp_path / "Protein_A" / "Sample_01" / "GFP" / "img.tif")
    touch(tmp_path / "Protein_A" / "Sample_01" / "DAPI" / "img.tif")
    touch(tmp_path / "Protein_B" / "Sample_02" / "GFP" / "img.tif")

    sample_paths = get_sample_paths_for_structure(
        tmp_path,
        STRUCTURE_GROUPED_BY_PROTEIN,
        ["GFP", "DAPI"],
    )

    assert [path.name for path in sample_paths] == ["Sample_01", "Sample_02"]


def test_common_results_export_dirs_are_run_level_subfolders(tmp_path: Path):
    export_dirs = build_common_results_export_dirs(tmp_path / "out")

    assert export_dirs["masks"] == tmp_path / "out" / "Results" / "Masks"
    assert export_dirs["mask_skeletons"] == tmp_path / "out" / "Results" / "Masks" / "Skeletons"
    assert export_dirs["mask_overlays"] == tmp_path / "out" / "Results" / "Overlays" / "TIFF Overlays"
    assert export_dirs["qc_overlay_pngs"] == tmp_path / "out" / "Results" / "Overlays" / "PNG"
    assert export_dirs["cell_segmentation_labels"] == tmp_path / "out" / "Results" / "Cells" / "TIFF Labels"
    assert export_dirs["cell_segmentation_outlines"] == tmp_path / "out" / "Results" / "Cells" / "TIFF Outlines"
    assert export_dirs["cell_segmentation_qc_pngs"] == tmp_path / "out" / "Results" / "Cells" / "PNG"
    assert export_dirs["cell_segmentation_tables"] == tmp_path / "out" / "Results" / "CSV Data" / "Cell Measurements"
    assert export_dirs["cell_signal_tables"] == tmp_path / "out" / "Results" / "CSV Data" / "Signal"
    assert export_dirs["cell_geometry_tables"] == tmp_path / "out" / "Results" / "CSV Data" / "Geometry"
    assert not export_dirs["masks"].exists()
    assert not export_dirs["qc_overlay_pngs"].exists()
    assert not export_dirs["cell_segmentation_qc_pngs"].exists()


def test_result_id_validation_rejects_names_that_sanitize_to_same_output(tmp_path: Path):
    samples = [tmp_path / "sample:a.tif", tmp_path / "sample?a.tif"]

    with pytest.raises(ValueError, match="same output name"):
        validate_unique_result_ids(
            samples,
            STRUCTURE_FLAT_TIFFS,
            input_dir=tmp_path,
        )
