from __future__ import annotations

from pathlib import Path

from cellonaut.pipeline.models import ImageDef
from cellonaut.pipeline.roi_defs import (
    class_roi_key,
    expand_image_def_to_class_roi_defs,
    parse_class_roi_key,
    parse_probability_class_indices,
    probability_class_indices_are_valid,
)
from cellonaut.masks.roi_processing import threshold_mask_path


def test_parse_probability_class_indices_accepts_single_values_lists_and_ranges():
    assert parse_probability_class_indices("1") == [1]
    assert parse_probability_class_indices("1,3") == [1, 3]
    assert parse_probability_class_indices("1-3") == [1, 2, 3]
    assert parse_probability_class_indices("3-1") == [1, 2, 3]
    assert parse_probability_class_indices("1; 2, 2") == [1, 2]


def test_parse_probability_class_indices_uses_safe_fallbacks():
    assert parse_probability_class_indices("") == [1]
    assert parse_probability_class_indices("not-a-class", fallback=2) == [2]
    assert parse_probability_class_indices("0,-2,3") == [1, 3]


def test_probability_class_validation_rejects_values_runtime_parsing_would_coerce():
    assert probability_class_indices_are_valid("1")
    assert probability_class_indices_are_valid("1, 3; 5-7")
    assert probability_class_indices_are_valid("3-1")
    assert not probability_class_indices_are_valid("")
    assert not probability_class_indices_are_valid("1.5")
    assert not probability_class_indices_are_valid("0")
    assert not probability_class_indices_are_valid("1,,2")
    assert not probability_class_indices_are_valid("not-a-class")


def test_threshold_mask_path_matches_saved_weka_mask_name(tmp_path):
    path = threshold_mask_path(tmp_path, "SampleA", "Mask", "classifier", 2)

    assert path == tmp_path / "SampleA_Mask_class2_thr_classifier.tif"


def test_class_roi_keys_round_trip():
    key = class_roi_key("image1", 3)
    assert key == "image1__class3"
    assert parse_class_roi_key(key) == ("image1", 3)
    assert parse_class_roi_key("image1") is None


def test_expand_image_def_to_class_roi_defs_keeps_single_class_as_base_def():
    image_def = ImageDef(
        key="image1",
        label="MammalianCell",
        folder_name="GFP",
        model_path=Path("classifier.model"),
        probability_class_index="1",
    )

    assert expand_image_def_to_class_roi_defs(image_def) == [image_def]


def test_expand_image_def_to_class_roi_defs_splits_multi_class_defs():
    mask_steps = [
        {
            "type": "binary_dilate",
            "enabled": True,
            "params": {"binary_settings": "2,1,false"},
        }
    ]
    image_def = ImageDef(
        key="image1",
        label="MammalianCell",
        folder_name="GFP",
        model_path=Path("classifier.model"),
        probability_class_index="1,3",
        image_processing_steps=[
            {
                "type": "rolling_ball_background",
                "enabled": True,
                "scope": "Segmentation input",
                "params": {"bg_radii": "50"},
            }
        ],
        mask_processing_steps=mask_steps,
        display_color="#00ff00",
    )

    expanded = expand_image_def_to_class_roi_defs(image_def)

    assert [item.key for item in expanded] == ["image1__class1", "image1__class3"]
    assert [item.label for item in expanded] == ["MammalianCell_Class1", "MammalianCell_Class3"]
    assert [item.probability_class_index for item in expanded] == [1, 3]
    assert [item.image_processing_steps for item in expanded] == [
        image_def.image_processing_steps,
        image_def.image_processing_steps,
    ]
    assert [item.mask_processing_steps for item in expanded] == [mask_steps, mask_steps]
    assert all(item.display_color == "#00ff00" for item in expanded)
    assert expanded[0].mask_processing_steps is not image_def.mask_processing_steps
