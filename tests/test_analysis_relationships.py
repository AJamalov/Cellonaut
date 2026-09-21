from __future__ import annotations

from cellonaut.config.relationships import (
    classifier_target_names,
    configured_mask_names,
    image_display_names,
    image_uses_combined_mask,
    image_uses_non_classifier_mask,
    normalize_analysis_relationships,
    remap_analysis_references,
    seed_first_classifier_self_relationship,
)


def image_def(name: str, classifier: str = "", **extra):
    data = {
        "name": name,
        "folder": name,
        "classifier": classifier,
        "mask_relationships": {},
        "analysis_cell_segmentation_enabled": False,
        "analysis_cell_segmentation_source": name,
    }
    data.update(extra)
    return data


def test_image_display_and_classifier_target_names_are_ordered():
    image_defs = [
        image_def("Cell"),
        image_def("", classifier="mask.model"),
        image_def("Tubules", classifier="tubules.model"),
    ]

    assert image_display_names(image_defs) == ["Cell", "Channel 2", "Tubules"]
    assert classifier_target_names(image_defs) == ["Channel 2", "Tubules"]


def test_combined_mask_is_exposed_as_named_matrix_target():
    image_defs = [
        image_def("Green", classifier="green.model"),
        image_def("Red", classifier="red.model"),
        image_def(
            "Either signal",
            mask_source_mode="Combined masks",
            combined_mask_sources=["Green", "Red"],
        ),
    ]

    assert classifier_target_names(image_defs) == [
        "Green",
        "Red",
        "Either signal",
    ]


def test_nested_combined_masks_require_runnable_dependencies_and_reject_cycles():
    image_defs = [
        image_def("Green", classifier="green.model"),
        image_def("Red", classifier="red.model"),
        image_def(
            "Overlap",
            mask_source_mode="Combined masks",
            combined_mask_sources=["Green", "Red"],
        ),
        image_def(
            "Final",
            mask_source_mode="Combined masks",
            combined_mask_sources=["Overlap", "Green"],
        ),
    ]

    assert configured_mask_names(image_defs) == {"Green", "Red", "Overlap", "Final"}

    image_defs[2]["combined_mask_sources"] = ["Final", "Red"]
    assert configured_mask_names(image_defs) == {"Green", "Red"}


def test_mask_source_predicates_normalize_saved_labels():
    combined = {"mask_source_mode": " combined MASKS "}

    assert image_uses_combined_mask(combined)
    assert image_uses_non_classifier_mask(combined)


def test_remap_analysis_references_preserves_relationships_after_renaming():
    image_defs = [
        image_def(
            "Cell",
            analysis_cell_segmentation_source="Tubules",
            mask_relationships={"Tubules": True},
        ),
        image_def("Tubules", classifier="tubules.model"),
    ]

    remapped = remap_analysis_references(image_defs, ["Cell", "Tubules"], ["Nucleus", "ER"])

    assert remapped[0]["analysis_cell_segmentation_source"] == "ER"
    assert remapped[0]["mask_relationships"] == {"ER": True}


def test_normalize_analysis_relationships_prunes_targets_without_classifiers():
    image_defs = [
        image_def(
            "Cell",
            mask_relationships={"Cell": True, "NoMask": True, "Mask": True},
        ),
        image_def("NoMask"),
        image_def("Mask", classifier="mask.model"),
    ]

    normalized = normalize_analysis_relationships(image_defs)

    assert normalized[0]["mask_relationships"] == {
        "Cell": False,
        "NoMask": False,
        "Mask": True,
    }
    assert normalized[1]["mask_relationships"] == {
        "Cell": False,
        "NoMask": False,
        "Mask": False,
    }


def test_normalize_analysis_relationships_keeps_cell_segmentation_enabled_source_active():
    image_defs = [
        image_def(
            "Brightfield",
            analysis_cell_segmentation_enabled=True,
            analysis_cell_segmentation_source="Missing",
        )
    ]

    normalized = normalize_analysis_relationships(image_defs)

    assert normalized[0]["analysis_cell_segmentation_source"] == "Brightfield"
    assert normalized[0]["analysis_cell_segmentation_enabled"] is True


def test_seed_first_classifier_self_relationship_only_when_empty():
    image_defs = [
        image_def("Cell"),
        image_def("Tubules", classifier="tubules.model"),
    ]

    seeded = seed_first_classifier_self_relationship(image_defs)

    assert seeded[1]["mask_relationships"] == {"Tubules": True}

    already_selected = seed_first_classifier_self_relationship(
        [
            image_def(
                "Cell",
                classifier="cell.model",
                mask_relationships={"Cell": True},
            ),
            image_def("Tubules", classifier="tubules.model"),
        ]
    )
    assert already_selected[0]["mask_relationships"] == {"Cell": True}
    assert already_selected[1]["mask_relationships"] == {}
