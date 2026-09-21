from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cellonaut.pipeline.models import MeasurementTarget, Config, ImageDef
from cellonaut.config.defaults import (
    IMAGE_PROCESSING_SCOPE_MEASUREMENT,
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST,
)
import cellonaut.pipeline.planning as planning
from cellonaut.pipeline.planning import (
    get_required_roi_defs_for_targets,
    image_keys_needed_for_processing,
    pipeline_needs_imagej,
    reusable_mask_warnings,
)


def make_reuse_config(tmp_path: Path) -> Config:
    classifier = tmp_path / "mask.model"
    classifier.write_text("fake", encoding="utf-8")
    return Config(
        fiji_app_path=tmp_path / "Fiji",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "out",
        input_structure="Samples directly in input folder",
        images=[
            ImageDef("image1", "Signal", "Signal", None),
            ImageDef("image2", "Mask", "Mask", classifier),
            ImageDef("image3", "Brightfield", "BF", None),
            ImageDef("image4", "Unused", "Unused", None),
        ],
        exclusion_tag="_ut_",
        threshold_method="Default",
        probability_class_index=1,
        reuse_existing_masks=True,
        mask_source_dir=tmp_path / "previous_run",
    )


def test_reuse_warning_shape_checks_cache_shared_source_reads(tmp_path: Path, monkeypatch):
    cfg = make_reuse_config(tmp_path)
    second_classifier = tmp_path / "second.model"
    second_classifier.write_text("fake", encoding="utf-8")
    cfg.images.append(ImageDef("image5", "Second", "Second", second_classifier))
    target = MeasurementTarget(source_image_key="image1", overlay_roi_keys=["image2", "image5"])

    assert cfg.mask_source_dir is not None
    mask_dir = cfg.mask_source_dir / "Results" / "Masks" / "MaskImages"
    mask_dir.mkdir(parents=True)
    for label, classifier in (("Mask", "mask"), ("Second", "second")):
        (mask_dir / f"SampleA_{label}_class1_thr_{classifier}.tif").write_bytes(b"mask")

    image_reads = []
    mask_reads = []
    monkeypatch.setattr(
        planning,
        "read_tiff_numpy_2d",
        lambda path, **_kwargs: image_reads.append(path) or np.zeros((8, 8), dtype=np.uint8),
    )
    monkeypatch.setattr(
        planning.tifffile,
        "imread",
        lambda path: mask_reads.append(path) or np.zeros((8, 8), dtype=np.uint8),
    )

    warnings = reusable_mask_warnings(
        cfg,
        [target],
        "SampleA",
        file_map={"image2": tmp_path / "shared.tif", "image5": tmp_path / "shared.tif"},
    )

    assert warnings == []
    assert len(image_reads) == 1
    assert len(mask_reads) == 2


def test_reuse_mask_planning_skips_mask_generation_images_when_masks_exist(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    result_id = "SampleA"
    assert cfg.mask_source_dir is not None
    mask_source = cfg.mask_source_dir / "Results"
    (mask_source / "Masks" / "MaskImages").mkdir(parents=True)
    (mask_source / "Cells" / "TIFF Labels").mkdir(parents=True)
    (mask_source / "Masks" / "MaskImages" / "SampleA_Mask_class1_thr_mask.tif").write_text("", encoding="utf-8")
    (mask_source / "Cells" / "TIFF Labels" / "SampleA_Signal_01_cellpose_labels.tif").write_text("", encoding="utf-8")

    target = MeasurementTarget(
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=["image2"],
        do_cell_segmentation=True,
        cell_segmentation_source="image3",
    )

    needed = image_keys_needed_for_processing(cfg, [target], [cfg.images[1]], result_id)

    assert needed == {"image1", "image3"}


def test_combined_mask_cleanup_opens_a_source_reference_image(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    cfg.reuse_existing_masks = False
    combined = ImageDef("image5", "Combined", "", None)
    combined.combined_mask_source_keys = ["image2", "image3"]
    combined.mask_processing_steps = [
        {"type": "binary_fill_holes", "enabled": True, "params": {}}
    ]
    target = MeasurementTarget(source_image_key="image1", overlay_roi_keys=["image5"])

    needed = image_keys_needed_for_processing(cfg, [target], [combined], "SampleA")

    assert needed == {"image1", "image2", "image3"}


def test_combined_mask_planning_prepares_sources_before_union(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    second_classifier = tmp_path / "second.model"
    second_classifier.write_text("fake", encoding="utf-8")
    cfg.images.append(ImageDef("image5", "Second", "Second", second_classifier))
    cfg.images.append(
        ImageDef(
            "image6",
            "Either",
            "Either",
            None,
            mask_source_mode="Combined masks",
            combined_mask_source_keys=["image2", "image5"],
        )
    )
    target = MeasurementTarget(
        source_image_key="image1",
        overlay_roi_keys=["image6"],
    )

    required = get_required_roi_defs_for_targets(cfg, [target])

    assert [image.key for image in required] == ["image2", "image5", "image6"]


def test_combined_mask_planning_rejects_dependency_cycles(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    cfg.images.extend(
        [
            ImageDef(
                "image5",
                "Combined A",
                "Combined A",
                None,
                mask_source_mode="Combined masks",
                combined_mask_source_keys=["image6"],
            ),
            ImageDef(
                "image6",
                "Combined B",
                "Combined B",
                None,
                mask_source_mode="Combined masks",
                combined_mask_source_keys=["image5"],
            ),
        ]
    )
    target = MeasurementTarget(source_image_key="image1", overlay_roi_keys=["image5"])

    with pytest.raises(ValueError, match="Combined mask dependency cycle"):
        get_required_roi_defs_for_targets(cfg, [target])


def test_reuse_mask_planning_opens_reused_weka_source_for_analyze_particles(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    cfg.images[1].mask_processing_steps = [
        {"type": "analyze_particles", "enabled": True, "params": {"particle_settings": "3-Infinity,0-1,false,false"}}
    ]
    result_id = "SampleA"
    assert cfg.mask_source_dir is not None
    mask_source = cfg.mask_source_dir / "Results"
    (mask_source / "Masks" / "MaskImages").mkdir(parents=True)
    (mask_source / "Cells" / "TIFF Labels").mkdir(parents=True)
    (mask_source / "Masks" / "MaskImages" / "SampleA_Mask_class1_thr_mask.tif").write_text("", encoding="utf-8")
    (mask_source / "Cells" / "TIFF Labels" / "SampleA_Signal_01_cellpose_labels.tif").write_text("", encoding="utf-8")

    target = MeasurementTarget(
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=["image2"],
        do_cell_segmentation=True,
        cell_segmentation_source="image3",
    )

    needed = image_keys_needed_for_processing(cfg, [target], [cfg.images[1]], result_id)

    assert needed == {"image1", "image2", "image3"}


def test_reuse_mask_planning_ignores_segmentation_only_radii_for_reused_masks(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    cfg.images[1].bg_radii_csv = "50"
    cfg.images[1].image_processing_steps = [
        {
            "type": "rolling_ball_background",
            "enabled": True,
            "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            "params": {"bg_radii": "50"},
        }
    ]
    result_id = "SampleA"
    assert cfg.mask_source_dir is not None
    mask_source = cfg.mask_source_dir / "Results"
    (mask_source / "Masks" / "MaskImages").mkdir(parents=True)
    (mask_source / "Cells" / "TIFF Labels").mkdir(parents=True)
    (mask_source / "Masks" / "MaskImages" / "SampleA_Mask_class1_thr_mask.tif").write_text("", encoding="utf-8")
    (mask_source / "Cells" / "TIFF Labels" / "SampleA_Signal_01_cellpose_labels.tif").write_text("", encoding="utf-8")

    target = MeasurementTarget(
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=["image2"],
        do_cell_segmentation=True,
        cell_segmentation_source="image3",
    )

    needed = image_keys_needed_for_processing(cfg, [target], [cfg.images[1]], result_id)

    assert needed == {"image1", "image3"}


def test_reuse_mask_planning_opens_reused_weka_source_for_skeleton_analysis(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    cfg.images[1].mask_processing_steps = [{"type": "analyze_skeleton", "enabled": True, "params": {}}]
    result_id = "SampleA"
    assert cfg.mask_source_dir is not None
    mask_source = cfg.mask_source_dir / "Results"
    (mask_source / "Masks" / "MaskImages").mkdir(parents=True)
    (mask_source / "Cells" / "TIFF Labels").mkdir(parents=True)
    (mask_source / "Masks" / "MaskImages" / "SampleA_Mask_class1_thr_mask.tif").write_text("", encoding="utf-8")
    (mask_source / "Cells" / "TIFF Labels" / "SampleA_Signal_01_cellpose_labels.tif").write_text("", encoding="utf-8")

    target = MeasurementTarget(
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=["image2"],
        do_cell_segmentation=True,
        cell_segmentation_source="image3",
    )

    needed = image_keys_needed_for_processing(cfg, [target], [cfg.images[1]], result_id)

    assert needed == {"image1", "image2", "image3"}


def test_pipeline_needs_imagej_ignores_segmentation_only_measurement_radii(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    cfg.images[0].bg_radii_csv = "50"
    cfg.images[0].image_processing_steps = [
        {
            "type": "rolling_ball_background",
            "enabled": True,
            "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            "params": {"bg_radii": "50"},
        }
    ]
    target = MeasurementTarget(source_image_key="image1")

    assert pipeline_needs_imagej(cfg, [target]) is False

    cfg.images[0].image_processing_steps[0]["scope"] = IMAGE_PROCESSING_SCOPE_MEASUREMENT

    assert pipeline_needs_imagej(cfg, [target]) is True


def test_pipeline_needs_imagej_for_measurement_contrast_processing(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    cfg.images[0].image_processing_steps = [
        {
            "type": IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST,
            "enabled": True,
            "scope": IMAGE_PROCESSING_SCOPE_MEASUREMENT,
            "params": {"contrast_saturation": "15"},
        }
    ]
    target = MeasurementTarget(source_image_key="image1")

    assert pipeline_needs_imagej(cfg, [target]) is True

    cfg.images[0].image_processing_steps[0]["scope"] = IMAGE_PROCESSING_SCOPE_SEGMENTATION

    assert pipeline_needs_imagej(cfg, [target]) is False


def test_reuse_mask_planning_opens_generation_images_for_missing_masks(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    target = MeasurementTarget(
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=["image2"],
        do_cell_segmentation=True,
        cell_segmentation_source="image3",
    )

    needed = image_keys_needed_for_processing(cfg, [target], [cfg.images[1]], "SampleA")

    assert needed == {"image1", "image2", "image3"}


def test_reuse_mask_planning_opens_segmentation_source_for_cell_qc_intensity(tmp_path: Path):
    cfg = make_reuse_config(tmp_path)
    result_id = "SampleA"
    assert cfg.mask_source_dir is not None
    mask_source = cfg.mask_source_dir / "Results"
    (mask_source / "Masks" / "MaskImages").mkdir(parents=True)
    (mask_source / "Cells" / "TIFF Labels").mkdir(parents=True)
    (mask_source / "Masks" / "MaskImages" / "SampleA_Mask_class1_thr_mask.tif").write_text("", encoding="utf-8")
    (mask_source / "Cells" / "TIFF Labels" / "SampleA_Signal_01_cellpose_labels.tif").write_text("", encoding="utf-8")

    target = MeasurementTarget(
        source_image_key="image1",
        overlay_base_image_key="image1",
        overlay_roi_keys=["image2"],
        do_cell_segmentation=True,
        cell_segmentation_source="image3",
    )

    needed = image_keys_needed_for_processing(cfg, [target], [cfg.images[1]], result_id)

    assert needed == {"image1", "image3"}
