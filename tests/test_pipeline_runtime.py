from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cellonaut.cell_segmentation import core as cell_segmentation_core
from cellonaut.exceptions import PipelineCancelled
from cellonaut.pipeline.models import Config, ImageDef, MeasurementTarget
from cellonaut.pipeline.planning import pipeline_needs_imagej
from cellonaut.pipeline.run_logging import make_logger
from cellonaut.pipeline.validation import (
    _validate_cell_settings,
    output_is_inside_input,
    portable_output_stem,
    validate_config,
    validate_input_folder_name,
    validate_portable_output_label,
)
from cellonaut.pipeline.discovery import STRUCTURE_FLAT_TIFFS, STRUCTURE_SAMPLES_DIRECTLY


def make_native_config(tmp_path: Path) -> Config:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    return Config(
        fiji_app_path=tmp_path / "missing-fiji",
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        input_structure=STRUCTURE_FLAT_TIFFS,
        images=[ImageDef("image1", "Signal", "Signal", None)],
        exclusion_tag="_ut_",
        threshold_method="Default",
        probability_class_index=1,
        measurement_targets=[MeasurementTarget(source_image_key="image1")],
    )


def test_cell_segmentation_cancellation_uses_pipeline_exception():
    cfg = cell_segmentation_core.CellSegmentationConfig()

    with pytest.raises(PipelineCancelled):
        cell_segmentation_core.run_cell_segmentation_on_image(
            np.zeros((4, 4), dtype=np.uint8),
            cfg,
            should_cancel=lambda: True,
        )


def test_native_pipeline_does_not_require_fiji(tmp_path: Path):
    cfg = make_native_config(tmp_path)

    validate_config(cfg)

    assert not cfg.fiji_app_path.exists()
    assert pipeline_needs_imagej(cfg, cfg.measurement_targets) is False


def test_weka_pipeline_still_requires_fiji(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    classifier = tmp_path / "mask.model"
    classifier.write_text("fake", encoding="utf-8")
    cfg.images[0].model_path = classifier
    cfg.measurement_targets[0].overlay_roi_keys = ["image1"]

    with pytest.raises(FileNotFoundError, match="Fiji app path"):
        validate_config(cfg)


def test_weka_pipeline_rejects_existing_non_fiji_folder(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    classifier = tmp_path / "mask.model"
    classifier.write_text("fake", encoding="utf-8")
    cfg.images[0].model_path = classifier
    cfg.measurement_targets[0].overlay_roi_keys = ["image1"]
    cfg.fiji_app_path.mkdir()

    with pytest.raises(ValueError, match="does not look like a Fiji installation"):
        validate_config(cfg)


def test_pipeline_validation_rejects_no_enabled_measurement_targets(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.measurement_targets[0].enabled = False

    with pytest.raises(ValueError, match="At least one measured channel"):
        validate_config(cfg)


@pytest.mark.parametrize("relative_output", [Path(), Path("results") / "run_1"])
def test_pipeline_validation_rejects_output_inside_input(tmp_path: Path, relative_output: Path):
    cfg = make_native_config(tmp_path)
    cfg.output_dir = cfg.input_dir / relative_output

    with pytest.raises(ValueError, match="outside the input directory"):
        validate_config(cfg)


def test_pipeline_validation_rejects_unknown_input_layout(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.input_structure = "Flat TIFF files"

    with pytest.raises(ValueError, match="Unsupported input layout"):
        validate_config(cfg)


def test_pipeline_validation_rejects_fractional_weka_class(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    classifier = tmp_path / "mask.model"
    classifier.write_text("fake", encoding="utf-8")
    cfg.images[0].model_path = classifier
    cfg.images[0].probability_class_index = "1.5"

    with pytest.raises(ValueError, match="positive whole numbers"):
        validate_config(cfg)


def test_pipeline_validation_rejects_duplicate_image_labels(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.images.append(ImageDef("image2", "Signal", "OtherSignal", None))

    with pytest.raises(ValueError, match="Duplicate image label: Signal"):
        validate_config(cfg)


def test_pipeline_validation_rejects_case_only_label_duplicates(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.images.append(ImageDef("image2", "signal", "OtherSignal", None))

    with pytest.raises(ValueError, match="Duplicate image label: signal"):
        validate_config(cfg)


def test_pipeline_validation_rejects_path_characters_in_output_labels(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.images[0].label = "GFP/mask"

    with pytest.raises(ValueError, match="cannot be used in output filenames"):
        validate_config(cfg)


@pytest.mark.parametrize("folder_name", ["../outside", "nested/channel", r"C:\outside"])
def test_pipeline_validation_rejects_channel_folder_paths(tmp_path: Path, folder_name: str):
    cfg = make_native_config(tmp_path)
    cfg.input_structure = STRUCTURE_SAMPLES_DIRECTLY
    cfg.images[0].folder_name = folder_name

    with pytest.raises(ValueError, match="one folder inside the selected input directory"):
        validate_config(cfg)


def test_pipeline_validation_rejects_labels_that_share_an_artifact_stem(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.images[0].label = "A+B"
    cfg.images.append(ImageDef("image2", "A=B", "OtherSignal", None))

    with pytest.raises(ValueError, match="same output filename"):
        validate_config(cfg)


def test_pipeline_validation_rejects_duplicate_measured_channel_targets(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.measurement_targets.append(MeasurementTarget(source_image_key="image1"))

    with pytest.raises(ValueError, match="enabled more than once"):
        validate_config(cfg)


def test_pipeline_validation_defaults_blank_target_cell_source(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.measurement_targets[0].do_cell_segmentation = True
    cfg.measurement_targets[0].cell_segmentation_source = ""

    validate_config(cfg)

    assert cfg.measurement_targets[0].cell_segmentation_source == "image1"


def test_pipeline_validation_checks_target_cellpose_size_settings(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.measurement_targets[0].do_cell_segmentation = True
    cfg.measurement_targets[0].cell_segmentation_source = "image1"
    cfg.measurement_targets[0].cell_diameter = 0

    with pytest.raises(ValueError, match="Cell diameter must be > 0"):
        validate_config(cfg)

    cfg.measurement_targets[0].cell_diameter = 20
    cfg.measurement_targets[0].cell_min_size = -1
    with pytest.raises(ValueError, match="Minimum cell area must be >= 0"):
        validate_config(cfg)


@pytest.mark.parametrize("diameter", [0, -0.1])
def test_cell_settings_reject_nonpositive_diameter(diameter: float):
    with pytest.raises(ValueError, match="Cell diameter must be > 0"):
        _validate_cell_settings(diameter, 0)


@pytest.mark.parametrize("diameter", [None, 0.1, 20.0])
def test_cell_settings_accept_valid_boundaries(diameter: float | None):
    _validate_cell_settings(diameter, 0)


@pytest.mark.parametrize("label", ["valid label", "DIA-1_2.ome"])
def test_portable_output_labels_accept_safe_display_names(label: str):
    validate_portable_output_label(label)


@pytest.mark.parametrize("label", ["bad/name", "bad\\name", "bad:name", "bad\x00name"])
def test_portable_output_labels_reject_path_or_control_characters(label: str):
    with pytest.raises(ValueError, match="cannot be used in output filenames"):
        validate_portable_output_label(label)


def test_portable_output_stem_is_stable_for_colliding_punctuation():
    assert portable_output_stem("A+B") == portable_output_stem("A=B")


@pytest.mark.parametrize("folder", ["channel", "channel name", "DIA_01"])
def test_input_folder_name_accepts_one_relative_folder(folder: str):
    validate_input_folder_name(folder, image_key="image1")


@pytest.mark.parametrize("folder", [".", "..", "/absolute", "nested/folder", "nested\\folder", "C:", r"C:\channel"])
def test_input_folder_name_rejects_navigation_absolute_and_nested_paths(folder: str):
    with pytest.raises(ValueError, match="key: image1"):
        validate_input_folder_name(folder, image_key="image1")


def test_output_location_boundary_distinguishes_sibling_from_descendant(tmp_path: Path):
    input_dir = tmp_path / "data"
    assert output_is_inside_input(input_dir, input_dir)
    assert output_is_inside_input(input_dir, input_dir / "results")
    assert not output_is_inside_input(input_dir, tmp_path / "data-results")


@pytest.mark.parametrize(("attribute", "value", "message"), [
    ("key", "  ", "Image key cannot be empty"),
    ("label", "  ", "Image label cannot be empty"),
    ("folder_name", "  ", "Image folder name cannot be empty"),
])
def test_pipeline_validation_rejects_blank_image_identity_fields(tmp_path: Path, attribute: str, value: str, message: str):
    cfg = make_native_config(tmp_path)
    setattr(cfg.images[0], attribute, value)

    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_pipeline_validation_rejects_duplicate_image_keys(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.images.append(ImageDef("image1", "Other", "Other", None))

    with pytest.raises(ValueError, match="Duplicate image key: image1"):
        validate_config(cfg)


def test_pipeline_validation_rejects_missing_and_empty_image_definitions(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.images = []

    with pytest.raises(ValueError, match="At least one image definition"):
        validate_config(cfg)


def test_pipeline_validation_rejects_missing_input_directory(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.input_dir = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
        validate_config(cfg)


def test_pipeline_validation_rejects_unknown_measurement_and_overlay_sources(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.measurement_targets[0].source_image_key = "missing"
    with pytest.raises(ValueError, match="Measured channel must reference"):
        validate_config(cfg)

    cfg = make_native_config(tmp_path / "second")
    cfg.measurement_targets[0].overlay_base_image_key = "missing"
    with pytest.raises(ValueError, match="Overlay base must reference"):
        validate_config(cfg)


def test_pipeline_validation_rejects_invalid_cellpose_and_mask_sources(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    target = cfg.measurement_targets[0]
    target.do_cell_segmentation = True
    target.cell_segmentation_source = "missing"
    with pytest.raises(ValueError, match="Cellpose source channel"):
        validate_config(cfg)

    cfg = make_native_config(tmp_path / "second")
    cfg.measurement_targets[0].per_cell_mask_source = "missing"
    with pytest.raises(ValueError, match="Per-cell mask source"):
        validate_config(cfg)


def test_pipeline_validation_rejects_whole_cell_overlay_without_segmentation(tmp_path: Path):
    cfg = make_native_config(tmp_path)
    cfg.measurement_targets[0].overlay_whole_cell_mask = True

    with pytest.raises(ValueError, match="requires whole-cell segmentation"):
        validate_config(cfg)


def test_logger_reports_file_failure_only_once(tmp_path: Path):
    messages: list[str] = []
    logger = make_logger(messages.append, tmp_path / "missing" / "run.log")

    logger("first")
    logger("second")

    warnings = [message for message in messages if message.startswith("[WARN] Run log")]
    assert len(warnings) == 1


def test_logger_writes_each_message_on_its_own_line(tmp_path: Path):
    log_path = tmp_path / "run.log"
    logger = make_logger(lambda _message: None, log_path)

    logger("first")
    logger("second")

    assert log_path.read_text(encoding="utf-8") == "first\nsecond\n"

@pytest.mark.parametrize('label', ['A_Class1', 'a_class1', 'A Class1', 'A+Class1'])
@pytest.mark.parametrize('reverse', [False, True])
def test_generated_class_labels_cannot_collide_with_configured_outputs(tmp_path, label, reverse):
    cfg = make_native_config(tmp_path)
    classifier = tmp_path / 'mask.model'
    classifier.write_bytes(b'model')
    masks = [ImageDef('mask', 'A', 'A', classifier, probability_class_index='1,2'),
             ImageDef('other', label, 'Other', classifier)]
    cfg.images.extend(reversed(masks) if reverse else masks)
    with pytest.raises(ValueError, match='Generated Weka class label'):
        validate_config(cfg)
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize('classes', ['1', '1,2', '1-3'])
def test_noncolliding_weka_class_labels_remain_valid(tmp_path, classes):
    cfg = make_native_config(tmp_path)
    classifier = tmp_path / 'mask.model'
    classifier.write_bytes(b'model')
    cfg.images.extend([ImageDef('mask', 'A', 'A', classifier, probability_class_index=classes),
                       ImageDef('other', 'B', 'B', classifier)])
    validate_config(cfg)


def test_single_weka_class_does_not_reserve_unused_generated_name(tmp_path):
    cfg = make_native_config(tmp_path)
    classifier = tmp_path / 'mask.model'
    classifier.write_bytes(b'model')
    cfg.images.extend([ImageDef('mask', 'A', 'A', classifier, probability_class_index='1'),
                       ImageDef('other', 'A_Class1', 'Other', classifier)])
    validate_config(cfg)
