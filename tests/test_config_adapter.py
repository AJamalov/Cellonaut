from __future__ import annotations

import json
from pathlib import Path

import pytest

from cellonaut.config.adapter import build_pipeline_config_from_gui_state, parse_cell_qc_limits_text
from cellonaut.config.defaults import (
    DEFAULT_CELL_DIAMETER,
    CELLPOSE_MODEL_OPTIONS,
    DEFAULT_CELLPOSE_MODEL_TYPE,
    DEFAULT_CELL_MIN_SIZE,
    DEFAULT_CELLPROB_THRESHOLD,
    DEFAULT_FLOW_THRESHOLD,
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    INPUT_STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    MEASUREMENT_METADATA,
    default_image_definition,
    default_image_definitions,
    default_measurement_options,
    normalize_image_processing_steps,
    normalize_mask_processing_steps,
)
from cellonaut.config.state import ImageGuiState, CellonautGuiState
from cellonaut.pipeline.models import ImageDef


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Area:10-20;Mean:5", {"Area": {"min": 10.0, "max": 20.0}, "Mean": {"max": 5.0}}),
        (
            '{"Area": {"min": 10, "max": 20}, "Mean": {"max": 5}}',
            {
                "Area": {"min": 10.0, "max": 20.0},
                "Mean": {"max": 5.0},
            },
        ),
    ],
)
def test_filter_rule_text_formats_normalize_to_the_same_mapping(text: str, expected: dict):
    assert parse_cell_qc_limits_text(text, strict=True) == expected


def make_state(
    tmp_path: Path,
    image_definitions: list[dict],
    measurement_options: dict | None = None,
    reuse_existing_masks: bool = False,
    mask_source_dir: str = "",
):
    return CellonautGuiState.from_widget_values(
        fiji_app_path=str(tmp_path / "Fiji"),
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        input_structure=INPUT_STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
        exclusion_tag="_ut_",
        reuse_existing_masks=reuse_existing_masks,
        mask_source_dir=mask_source_dir,
        measurement_options=measurement_options or default_measurement_options(),
        image_definitions=image_definitions,
    )


def classifier_file(tmp_path: Path, name: str = "classifier.model") -> Path:
    path = tmp_path / name
    path.write_text("fake classifier", encoding="utf-8")
    return path


def test_image_definition_schema_matches_typed_state_and_excludes_derived_fields():
    default_keys = set(default_image_definition(0))
    typed_keys = set(ImageGuiState.__dataclass_fields__)

    assert default_keys == typed_keys
    assert "mask_relationships" in default_keys
    assert default_keys.isdisjoint(
        {
            "analysis_enabled",
            "analysis_overlay_matrix",
            "analysis_mask_matrix",
            "cell_use_gpu",
        }
    )


def test_default_measurements_match_minimal_fiji_selection():
    options = default_measurement_options()

    enabled = {key for key, value in options.items() if value}

    assert enabled == {"area", "mean", "raw_intden"}
    assert "std_dev" in options
    assert "cell_area" in options
    assert "std_dev_in_positive_area" in options
    assert set(options) == set(MEASUREMENT_METADATA)


def test_default_cellpose_settings_match_cellpose_api_defaults():
    assert DEFAULT_CELL_DIAMETER == ""
    assert DEFAULT_CELL_MIN_SIZE == "15"
    assert DEFAULT_CELLPROB_THRESHOLD == "0.0"
    assert DEFAULT_FLOW_THRESHOLD == "0.4"

    assert DEFAULT_CELLPOSE_MODEL_TYPE == "cpsam"
    assert CELLPOSE_MODEL_OPTIONS == ("cpsam", "cpsam_v2")


def test_pipeline_image_defaults_match_gui_mask_defaults():
    image = ImageDef(
        key="image1",
        label="Channel 1",
        folder_name="Channel 1",
        model_path=None,
    )

    assert image.mask_processing_steps == []
    assert image.threshold_method == "Default"
    assert "min_roi_area" not in default_image_definitions()[0]
    assert "shift_x" not in default_image_definitions()[0]
    assert "analyze_mask_skeleton" not in default_image_definitions()[0]


def test_bundled_default_preset_matches_runtime_defaults():
    preset_path = Path(__file__).parents[1] / "cellonaut" / "data" / "presets" / "Default.json"
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    normalized_images = [
        ImageGuiState.from_dict(image, index).to_dict() for index, image in enumerate(preset["image_definitions"])
    ]

    assert preset["measurement_options"] == default_measurement_options()
    assert normalized_images == default_image_definitions()
    assert {image["cellpose_model_type"] for image in normalized_images} == {DEFAULT_CELLPOSE_MODEL_TYPE}
    for removed_key in ("fiji_app_path", "nd2_output_dir", "nd2_z_mode", "nd2_z_index", "nd2_channel_folder_state"):
        assert removed_key not in preset


def test_edited_preset_boolean_strings_keep_their_meaning():
    image = ImageGuiState.from_dict(
        {
            "mask_slot_enabled": "false",
            "cell_remove_border": "off",
            "analysis_cell_segmentation_enabled": "yes",
            "mask_relationships": {"Mask A": "false", "Mask B": "true"},
        },
        0,
    )

    assert image.mask_slot_enabled is False
    assert image.cell_remove_border is False
    assert image.analysis_cell_segmentation_enabled is True
    assert image.mask_relationships == {"Mask A": False, "Mask B": True}

    image_steps = normalize_image_processing_steps(
        [{"type": "smooth", "enabled": "false", "params": {"smooth_iterations": "2"}}]
    )
    mask_steps = normalize_mask_processing_steps(
        [{"type": "binary_erode", "enabled": "false", "params": {"binary_settings": "1,1,false"}}]
    )
    assert image_steps[0]["enabled"] is False
    assert mask_steps[0]["enabled"] is False


def test_edited_measurement_boolean_strings_are_normalized(tmp_path: Path):
    state = make_state(
        tmp_path,
        [{"name": "Cell", "analysis_cell_segmentation_enabled": True}],
        measurement_options={"area": "false", "mean": "true"},
    )

    assert state.measurement_options["area"] is False
    assert state.measurement_options["mean"] is True


def test_config_conversion_drops_unknown_measurement_keys(tmp_path: Path):
    options = default_measurement_options()
    options["save_roi_outlines"] = True
    state = make_state(
        tmp_path,
        [{"name": "Cell", "analysis_cell_segmentation_enabled": True}],
        measurement_options=options,
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert "save_roi_outlines" not in cfg.measurement_options
    assert cfg.cell_diameter is None
    assert cfg.cell_min_size == 15


def test_config_conversion_preserves_reuse_existing_masks_flag(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "GFP",
                "analysis_cell_segmentation_enabled": True,
            },
        ],
        reuse_existing_masks=True,
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.reuse_existing_masks is True
    assert cfg.mask_source_dir == tmp_path / "output"


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("input_dir", "Input directory cannot be empty"),
        ("output_dir", "Output directory cannot be empty"),
    ],
)
def test_config_conversion_rejects_blank_required_folders(tmp_path: Path, field: str, message: str):
    state = make_state(
        tmp_path,
        [{"name": "Cell", "analysis_cell_segmentation_enabled": True}],
    )
    setattr(state, field, "")

    with pytest.raises(ValueError, match=message):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_discards_unsupported_mask_source(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Mask",
                "mask_source_mode": "Mystery source",
                "mask_relationships": {"Mask": True},
            }
        ],
    )

    with pytest.raises(ValueError, match="No analysis task is enabled"):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_rejects_unknown_cellpose_model(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "analysis_cell_segmentation_enabled": True,
                "cellpose_model_type": "unknown-model",
            }
        ],
    )

    with pytest.raises(ValueError, match="Cellpose model for channel or mask 'Cell'"):
        build_pipeline_config_from_gui_state(state)


@pytest.mark.parametrize("model_type", CELLPOSE_MODEL_OPTIONS)
def test_config_conversion_accepts_current_builtin_cellpose_models(tmp_path: Path, model_type: str):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "analysis_cell_segmentation_enabled": True,
                "cellpose_model_type": model_type,
            }
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.measurement_targets[0].cellpose_model_type == model_type


def test_config_conversion_rejects_missing_custom_cellpose_model(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "analysis_cell_segmentation_enabled": True,
                "cellpose_custom_model_path": str(tmp_path / "missing-model"),
            }
        ],
    )

    with pytest.raises(ValueError, match="Custom Cellpose model file does not exist"):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_allows_missing_custom_cellpose_model_during_mask_reuse(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "analysis_cell_segmentation_enabled": True,
                "cellpose_custom_model_path": str(tmp_path / "missing-model"),
            }
        ],
        reuse_existing_masks=True,
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.measurement_targets[0].cellpose_custom_model_path == str(tmp_path / "missing-model")


def test_default_mask_processing_steps_are_empty_until_user_adds_steps(tmp_path: Path):
    state = make_state(
        tmp_path,
        image_definitions=[
            {"name": "Signal", "folder": "Signal", "mask_slot_enabled": False},
            {
                "name": "Mask",
                "folder": "Mask",
                "classifier": "",
                "is_mask_only": True,
                "mask_source_channel": "Signal",
            },
        ],
    )

    mask_def = next(image.to_dict() for image in state.image_definitions if image.name == "Mask")
    assert mask_def["mask_processing_steps"] == []


def test_config_conversion_uses_selected_mask_source_folder(tmp_path: Path):
    source = tmp_path / "previous_run"
    source.mkdir()
    state = make_state(
        tmp_path,
        [{"name": "Cell", "folder": "GFP", "analysis_cell_segmentation_enabled": True}],
        reuse_existing_masks=True,
        mask_source_dir=str(source),
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.mask_source_dir == source


def test_config_conversion_builds_named_combined_mask_target(tmp_path: Path):
    green = classifier_file(tmp_path, "green.model")
    red = classifier_file(tmp_path, "red.model")
    state = make_state(
        tmp_path,
        [
            {
                "name": "Signal",
                "mask_relationships": {"Either": True},
            },
            {"name": "Green", "classifier": str(green)},
            {"name": "Red", "classifier": str(red)},
            {
                "name": "Either",
                "mask_source_mode": "Combined masks",
                "combined_mask_sources": ["Green", "Red"],
                "combined_mask_operation": "AND",
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.images[3].combined_mask_source_keys == ["image2", "image3"]
    assert cfg.images[3].combined_mask_operation == "AND"
    assert cfg.measurement_targets[0].overlay_roi_keys == ["image4"]


def test_config_conversion_supports_nested_combined_masks_and_rejects_cycles(tmp_path: Path):
    green = classifier_file(tmp_path, "green.model")
    red = classifier_file(tmp_path, "red.model")
    base_images = [
        {"name": "Signal", "mask_relationships": {"Final": True}},
        {"name": "Green", "classifier": str(green)},
        {"name": "Red", "classifier": str(red)},
        {
            "name": "Overlap",
            "mask_source_mode": "Combined masks",
            "combined_mask_sources": ["Green", "Red"],
            "combined_mask_operation": "AND",
        },
        {
            "name": "Final",
            "mask_source_mode": "Combined masks",
            "combined_mask_sources": ["Overlap", "Green"],
            "combined_mask_operation": "OR",
        },
    ]

    cfg = build_pipeline_config_from_gui_state(make_state(tmp_path, base_images))
    assert cfg.images[4].combined_mask_source_keys == ["image4", "image2"]

    base_images[3]["combined_mask_sources"] = ["Final", "Red"]
    with pytest.raises(ValueError, match="dependency cycle"):
        build_pipeline_config_from_gui_state(make_state(tmp_path, base_images))


def test_config_conversion_preserves_tiff_stack_layer(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Stack L2",
                "folder": "Stack",
                "stack_channel_index": "2",
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.images[0].stack_channel_index == 2


def test_config_conversion_preserves_tiff_stack_z_projection_settings(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Stack Z2",
                "folder": "Stack",
                "stack_z_mode": "single_z",
                "stack_z_index": "2",
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.images[0].stack_z_mode == "single_z"
    assert cfg.images[0].stack_z_index == 2


def test_config_conversion_rejects_invalid_tiff_stack_z_mode(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Stack",
                "folder": "Stack",
                "stack_z_mode": "median",
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    with pytest.raises(ValueError, match="Stack Z mode for channel or mask 'Stack'"):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_rejects_invalid_contrast_saturation_step(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "Cell",
                "image_processing_steps": [
                    {
                        "type": "enhance_contrast",
                        "enabled": True,
                        "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                        "params": {"contrast_saturation": "101"},
                    }
                ],
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    with pytest.raises(ValueError, match="Contrast saturation for channel or mask 'Cell' must be <= 100"):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_rejects_invalid_smooth_iterations_step(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "Cell",
                "image_processing_steps": [
                    {
                        "type": "smooth",
                        "enabled": True,
                        "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                        "params": {"smooth_iterations": "0"},
                    }
                ],
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    with pytest.raises(ValueError, match="Smooth iterations for channel or mask 'Cell' must be >= 1"):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_preserves_mask_skeleton_option(tmp_path: Path):
    classifier = classifier_file(tmp_path)
    state = make_state(
        tmp_path,
        [
            {
                "name": "Tubules",
                "folder": "RFP",
                "classifier": str(classifier),
                "mask_processing_steps": [{"type": "analyze_skeleton", "enabled": True, "params": {}}],
                "mask_relationships": {"Tubules": True},
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.images[0].mask_processing_steps[0]["type"] == "analyze_skeleton"
    assert cfg.images[0].mask_processing_steps[0]["enabled"] is True


def test_reuse_existing_masks_allows_missing_classifier_file_for_mask_lookup(tmp_path: Path):
    missing_classifier = tmp_path / "missing.model"
    state = make_state(
        tmp_path,
        [
            {
                "name": "Signal",
                "folder": "GFP",
                "mask_relationships": {"Mask": True},
            },
            {
                "name": "Mask",
                "folder": "MASK",
                "classifier": str(missing_classifier),
            },
        ],
        reuse_existing_masks=True,
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.images[1].model_path == missing_classifier
    assert cfg.reuse_existing_masks is True


def test_config_conversion_preserves_multi_class_probability_strings_and_relationships(tmp_path: Path):
    classifier = classifier_file(tmp_path)
    state = make_state(
        tmp_path,
        [
            {
                "name": "MammalianCell",
                "folder": "GFP",
                "mask_relationships": {"ER": True},
            },
            {
                "name": "ER",
                "folder": "ER",
                "classifier": str(classifier),
                "probability_class_index": "1,3",
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.images[1].probability_class_index == "1,3"
    assert len(cfg.measurement_targets) == 1
    assert cfg.measurement_targets[0].source_image_key == "image1"
    assert cfg.measurement_targets[0].overlay_roi_keys == ["image2"]
    assert cfg.measurement_targets[0].per_cell_mask_source == "image2"


def test_config_conversion_ignores_relationship_targets_without_classifiers(tmp_path: Path):
    classifier = classifier_file(tmp_path)
    state = make_state(
        tmp_path,
        [
            {
                "name": "Source",
                "folder": "Source",
                "mask_relationships": {"ValidMask": True, "NoMask": True},
            },
            {
                "name": "ValidMask",
                "folder": "ValidMask",
                "classifier": str(classifier),
            },
            {
                "name": "NoMask",
                "folder": "NoMask",
                "classifier": "",
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.measurement_targets[0].overlay_roi_keys == ["image2"]
    assert cfg.measurement_targets[0].per_cell_mask_source == "image2"


def test_config_conversion_uses_explicit_cell_group_mask_source(tmp_path: Path):
    classifier = classifier_file(tmp_path)
    state = make_state(
        tmp_path,
        [
            {
                "name": "Source",
                "folder": "Source",
                "mask_relationships": {"Mask A": True, "Mask B": True},
                "cell_group_mask_source": "Mask B",
            },
            {"name": "Mask A", "folder": "Mask A", "classifier": str(classifier)},
            {"name": "Mask B", "folder": "Mask B", "classifier": str(classifier)},
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.measurement_targets[0].overlay_roi_keys == ["image2", "image3"]
    assert cfg.measurement_targets[0].per_cell_mask_source == "image3"


def test_config_conversion_derives_enabled_analysis_rows_from_relationships(tmp_path: Path):
    classifier = classifier_file(tmp_path)
    state = make_state(
        tmp_path,
        [
            {
                "name": "DisabledSource",
                "folder": "DisabledSource",
            },
            {
                "name": "Mask",
                "folder": "Mask",
                "classifier": str(classifier),
                "mask_relationships": {"Mask": True},
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert [target.source_image_key for target in cfg.measurement_targets] == ["image2"]


def test_config_conversion_requires_at_least_one_enabled_analysis_task(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Source",
                "folder": "Source",
                "mask_relationships": {"NoMask": True},
            },
            {
                "name": "NoMask",
                "folder": "NoMask",
                "classifier": "",
            },
        ],
    )

    with pytest.raises(ValueError, match="No analysis task is enabled"):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_rejects_duplicate_image_names(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Signal",
                "folder": "GFP",
                "analysis_cell_segmentation_enabled": True,
            },
            {
                "name": "Signal",
                "folder": "RFP",
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate name: Signal"):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_preserves_cell_segmentation_settings(tmp_path: Path):
    custom_model = tmp_path / "custom-model.pt"
    custom_model.write_bytes(b"model")
    state = make_state(
        tmp_path,
        [
            {
                "name": "Brightfield",
                "folder": "Brightfield",
                "analysis_cell_segmentation_enabled": True,
                "analysis_cell_segmentation_source": "Brightfield",
                "cell_diameter": "42.5",
                "cell_min_size": "321",
                "cellprob_threshold": "-1.5",
                "flow_threshold": "0.9",
                "cell_remove_border": False,
                "cellpose_model_type": "cpsam",
                "cellpose_custom_model_path": str(custom_model),
                "cell_qc_limits": "Area:50-500; Mean intensity:10-",
                "cell_qc_exclude_flagged": True,
                "mask_qc_limits": "Mask area:20-; Mask fraction of cell area:-0.8",
                "qc_filter_mode": "Exclude only if both fail",
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)
    target = cfg.measurement_targets[0]

    assert target.source_image_key == "image1"
    assert target.do_cell_segmentation is True
    assert target.cell_segmentation_source == "image1"
    assert target.cell_diameter == 42.5
    assert target.cell_min_size == 321
    assert target.cell_use_gpu is True
    assert target.cellprob_threshold == -1.5
    assert target.flow_threshold == 0.9
    assert target.cell_remove_border is False
    assert target.cellpose_model_type == "cpsam"
    assert target.cellpose_custom_model_path == str(custom_model)
    group = state.image_definitions[0].cell_populations[0]
    assert group["cell_qc_limits"] == "Area:50-500; Mean intensity:10-"
    assert group["exclude_from_csv"] is True
    assert not hasattr(target, "cell_qc_rules")

    # Channel settings belong to the target, not to defaults for other channels.
    assert cfg.cell_diameter is None
    assert cfg.cell_min_size == int(DEFAULT_CELL_MIN_SIZE)
    assert cfg.cell_use_gpu is True
    assert cfg.cellprob_threshold == float(DEFAULT_CELLPROB_THRESHOLD)
    assert cfg.flow_threshold == float(DEFAULT_FLOW_THRESHOLD)
    assert cfg.cellpose_model_type == DEFAULT_CELLPOSE_MODEL_TYPE
    assert cfg.cellpose_custom_model_path == ""


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cell_diameter", "0", "Cell diameter for channel or mask 'Cell' must be >= 1"),
        ("cell_min_size", "12.5", "Minimum cell area for channel or mask 'Cell' must be an integer"),
        (
            "cellprob_threshold",
            "not-a-number",
            "Cellpose probability threshold for channel or mask 'Cell' must be a number",
        ),
        ("flow_threshold", "-0.1", "Cellpose flow threshold for channel or mask 'Cell' must be >= 0"),
    ],
)
def test_config_conversion_rejects_invalid_numeric_fields(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
):
    image = {
        "name": "Cell",
        "folder": "Cell",
        "analysis_cell_segmentation_enabled": True,
        field: value,
    }
    state = make_state(tmp_path, [image])

    with pytest.raises(ValueError, match=message):
        build_pipeline_config_from_gui_state(state)


@pytest.mark.parametrize("value", ["2.5", "zero", "0", "-1"])
def test_config_conversion_rejects_invalid_stack_layers(tmp_path: Path, value: str):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "Cell",
                "stack_channel_index": value,
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    with pytest.raises(ValueError, match="Stack layer for channel or mask 'Cell'"):
        build_pipeline_config_from_gui_state(state)


@pytest.mark.parametrize("value", ["1.5", "one", "1,,3", "0", "1-"])
def test_config_conversion_rejects_invalid_probability_class_selections(tmp_path: Path, value: str):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "Cell",
                "probability_class_index": value,
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    with pytest.raises(ValueError, match="Weka class selection for channel or mask 'Cell'"):
        build_pipeline_config_from_gui_state(state)


def test_config_conversion_allows_blank_optional_numeric_fields(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "Cell",
                "stack_channel_index": "",
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.images[0].stack_channel_index is None


def test_config_conversion_preserves_normalized_image_processing_scopes(tmp_path: Path):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "Cell",
                "bg_radii": "50",
                "image_processing_steps": [
                    {
                        "type": "rolling_ball_background",
                        "enabled": True,
                        "row_enabled": True,
                        "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                        "params": {"bg_radii": "50"},
                    },
                ],
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert [step["scope"] for step in cfg.images[0].image_processing_steps] == [
        IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    ]


def test_config_conversion_copies_source_processing_recipe_to_assigned_weka_mask(tmp_path: Path):
    classifier = classifier_file(tmp_path)
    state = make_state(
        tmp_path,
        [
            {
                "name": "Signal",
                "folder": "Signal",
                "bg_radii": "50",
                "image_processing_steps": [
                    {
                        "type": "rolling_ball_background",
                        "enabled": True,
                        "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
                        "params": {"bg_radii": "50"},
                    }
                ],
                "mask_relationships": {"Signal mask": True},
            },
            {
                "name": "Signal mask",
                "folder": "Signal",
                "classifier": str(classifier),
                "is_mask_only": True,
                "mask_source_channel": "Signal",
            },
        ],
    )

    cfg = build_pipeline_config_from_gui_state(state)

    assert cfg.images[1].folder_name == "Signal"
    assert cfg.images[1].stack_source_image_key == "image1"
    assert cfg.images[1].bg_radii_csv == "50"
    assert cfg.images[1].image_processing_steps == [
        {
            "type": "rolling_ball_background",
            "enabled": True,
            "row_enabled": True,
            "scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            "params": {"bg_radii": "50"},
        }
    ]


@pytest.mark.parametrize("value", ["20,bad", "-5", "NaN", "inf"])
def test_config_conversion_rejects_invalid_background_radii(tmp_path: Path, value: str):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "Cell",
                "bg_radii": value,
                "analysis_cell_segmentation_enabled": True,
            },
        ],
    )

    with pytest.raises(ValueError, match="Background radii for channel or mask 'Cell'"):
        build_pipeline_config_from_gui_state(state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cell_qc_limits", "Area:abc-500"),
        ("cell_qc_limits", "Area:500-50"),
        ("cell_qc_limits", "Area"),
        ("cell_qc_limits", '{"Area": {"minimum": 10}}'),
        ("mask_qc_limits", "Mask fraction of cell area:NaN"),
    ],
)
def test_config_conversion_rejects_invalid_filter_rules(tmp_path: Path, field: str, value: str):
    state = make_state(
        tmp_path,
        [
            {
                "name": "Cell",
                "folder": "Cell",
                "analysis_cell_segmentation_enabled": True,
                field: value,
            },
        ],
    )

    with pytest.raises(ValueError, match="channel or mask 'Cell'"):
        build_pipeline_config_from_gui_state(state)
