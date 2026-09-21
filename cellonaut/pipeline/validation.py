"""Validation and normalization of pipeline configuration."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

from cellonaut.exceptions import SetupError, SetupErrorCode, SetupFileNotFoundError
from cellonaut.artifact_naming import portable_component
from cellonaut.io.fiji_installation import looks_like_fiji_installation
from cellonaut.measurement.math import clean_name_or_none
from cellonaut.pipeline.models import Config, resolve_measurement_target
from cellonaut.pipeline.discovery import (
    STRUCTURE_FLAT_TIFFS,
    STRUCTURE_GROUPED_BY_PROTEIN,
    STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    STRUCTURE_SAMPLES_DIRECTLY,
)
from cellonaut.pipeline.planning import pipeline_needs_imagej
from cellonaut.pipeline.roi_defs import expand_image_def_to_class_roi_defs, probability_class_indices_are_valid


PORTABLE_FILENAME_FORBIDDEN = frozenset('<>:"/\\|?*')


# Display labels are embedded in result filenames by several exporters. Reject
# path syntax here so no feature can accidentally create a nested or invalid path.
def validate_portable_output_label(label: str) -> None:
    invalid = sorted({char for char in label if char in PORTABLE_FILENAME_FORBIDDEN or ord(char) < 32})
    if invalid:
        shown = " ".join(repr(char) for char in invalid)
        raise ValueError(f"Image label contains characters that cannot be used in output filenames: {shown}")


# Several artifact types normalize punctuation for portability. Validate that
# distinct display labels cannot collapse onto the same on-disk stem.
def portable_output_stem(label: str) -> str:
    return portable_component(label)


# Physical channel folders are names below the selected dataset root, not arbitrary paths.
def validate_input_folder_name(folder_name: str, *, image_key: str) -> None:
    text = str(folder_name or "").strip()
    windows_path = PureWindowsPath(text)
    if (
        text in {".", ".."}
        or "/" in text
        or "\\" in text
        or Path(text).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise ValueError(
            f"Image folder name must be one folder inside the selected input directory for key: {image_key}"
        )


# Results must not sit below their source tree because later discovery would
# treat exported TIFF masks and overlays as fresh input data.
def output_is_inside_input(input_dir: Path, output_dir: Path) -> bool:
    try:
        resolved_input = input_dir.resolve()
        resolved_output = output_dir.resolve()
    except OSError:
        resolved_input = input_dir.absolute()
        resolved_output = output_dir.absolute()
    return resolved_output == resolved_input or resolved_input in resolved_output.parents


# Fail before creating output folders or starting Fiji so configuration mistakes remain cheap to fix.
def validate_config(cfg: Config) -> None:
    """Validate a run and replace target requests with resolved execution settings."""
    if not cfg.input_dir.exists():
        raise SetupFileNotFoundError(SetupErrorCode.INPUT_MISSING, f"Input directory does not exist: {cfg.input_dir}")
    supported_structures = {
        STRUCTURE_SAMPLES_DIRECTLY,
        STRUCTURE_GROUPED_BY_PROTEIN,
        STRUCTURE_FLAT_TIFFS,
        STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS,
    }
    if cfg.input_structure not in supported_structures:
        raise ValueError(
            f"Unsupported input layout: {cfg.input_structure!r}. "
            "Select the input folder again so Cellonaut can detect its layout."
        )
    if output_is_inside_input(cfg.input_dir, cfg.output_dir):
        raise ValueError(
            "Output directory must be outside the input directory. "
            "Choose a separate folder so generated results cannot be detected as new input samples."
        )
    if not cfg.images:
        raise ValueError("At least one image definition is required")

    keys = set()
    labels: set[str] = set()
    output_stems: dict[str, str] = {}
    roi_keys = set()
    for image in cfg.images:
        if clean_name_or_none(image.key) is None:
            raise ValueError("Image key cannot be empty")
        if image.key in keys:
            raise ValueError(f"Duplicate image key: {image.key}")
        keys.add(image.key)

        if clean_name_or_none(image.label) is None:
            raise ValueError(f"Image label cannot be empty for key: {image.key}")
        validate_portable_output_label(image.label)
        label_key = image.label.casefold()
        if label_key in labels:
            raise ValueError(f"Duplicate image label: {image.label}")
        labels.add(label_key)
        output_stem = portable_output_stem(image.label).casefold()
        if output_stem in output_stems:
            raise ValueError(
                f"Image labels produce the same output filename: "
                f"{output_stems[output_stem]} and {image.label}"
            )
        output_stems[output_stem] = image.label
        if clean_name_or_none(image.folder_name) is None:
            raise ValueError(f"Image folder name cannot be empty for key: {image.key}")
        if cfg.input_structure != STRUCTURE_FLAT_TIFFS:
            validate_input_folder_name(image.folder_name, image_key=image.key)

        if image.model_path is not None or image.combined_mask_source_keys:
            roi_keys.add(image.key)
        if image.combined_mask_source_keys:
            if image.combined_mask_operation not in {"OR", "AND", "XOR"}:
                raise ValueError(f"Combined mask {image.label} operation must be OR, AND, or XOR.")
            if len(image.combined_mask_source_keys) < 2:
                raise ValueError(f"Combined mask {image.label} needs at least two source masks.")
            if image.key in image.combined_mask_source_keys:
                raise ValueError(f"Combined mask {image.label} cannot include itself.")
        if image.model_path is not None:
            if not image.model_path.exists() and not getattr(cfg, "reuse_existing_masks", False):
                raise SetupFileNotFoundError(SetupErrorCode.CLASSIFIER_MISSING, f"Classifier file does not exist for " f"{image.label}: {image.model_path}")
            if not probability_class_indices_are_valid(image.probability_class_index):
                raise ValueError(
                    f"Probability-map class numbers must contain positive whole numbers or ranges for {image.label}"
                )

    # Class expansion creates additional public names used as measurement keys
    # and filenames. Reserve configured names as well as every generated name.
    for image in cfg.images:
        for expanded in expand_image_def_to_class_roi_defs(image):
            if expanded.key == image.key:
                continue
            output_stem = portable_output_stem(expanded.label).casefold()
            if output_stem in output_stems:
                raise ValueError(
                    f"Generated Weka class label {expanded.label!r} from {image.label!r} "
                    f"conflicts with output label {output_stems[output_stem]!r}. "
                    "Rename one of the masks or channels before processing."
                )
            output_stems[output_stem] = expanded.label

    if cfg.probability_class_index < 1:
        raise ValueError("Weka class index must be >= 1")
    for image in cfg.images:
        missing_sources = [source_key for source_key in image.combined_mask_source_keys if source_key not in roi_keys]
        if missing_sources:
            raise ValueError(
                f"Combined mask {image.label} references invalid source masks: " + ", ".join(missing_sources)
            )

    allowed_overlay_extra_keys = {"__whole_cell_mask__"}
    # Resolve before validation so the checked values are exactly the values
    # used by every sample. Replace requests after validation has succeeded.
    resolved_targets = [resolve_measurement_target(cfg, target) for target in cfg.measurement_targets]
    enabled_targets = [target for target in resolved_targets if target.enabled]
    if not enabled_targets:
        raise ValueError("At least one measured channel must be enabled")

    seen_target_sources: set[str] = set()
    for target in enabled_targets:
        if target.source_image_key not in keys:
            raise SetupError(SetupErrorCode.MEASURED_CHANNEL, "Measured channel must reference a configured channel")
        if target.source_image_key in seen_target_sources:
            raise ValueError(f"Measured channel is enabled more than once: {target.source_image_key}")
        seen_target_sources.add(target.source_image_key)
        if target.overlay_base_image_key and target.overlay_base_image_key not in keys:
            raise SetupError(SetupErrorCode.OVERLAY_BASE, "Overlay base must reference a configured channel")
        for key in target.overlay_roi_keys:
            if key not in allowed_overlay_extra_keys and key not in roi_keys:
                raise SetupError(SetupErrorCode.OVERLAY_MASK, f"Overlay mask must reference a configured mask: {key}")
        if target.overlay_whole_cell_mask and not target.do_cell_segmentation:
            raise SetupError(SetupErrorCode.WHOLE_CELL_OVERLAY, "Whole cell mask overlay requires whole-cell segmentation to be enabled.")
        if target.cell_segmentation_source and target.cell_segmentation_source not in keys:
            raise SetupError(SetupErrorCode.CELL_SOURCE, "Cellpose source channel must reference a configured channel")
        if target.per_cell_mask_source and target.per_cell_mask_source not in roi_keys:
            raise SetupError(SetupErrorCode.PER_CELL_MASK, "Per-cell mask source must reference a configured mask")
        if target.do_cell_segmentation:
            _validate_cell_settings(
                target.cell_diameter,
                target.cell_min_size,
            )

    _validate_fiji_path_if_needed(cfg, enabled_targets)
    cfg.measurement_targets = list(resolved_targets)


# Cellpose accepts broad numeric inputs, but these two values would otherwise produce unusable masks.
def _validate_cell_settings(diameter: float | None, min_size: int) -> None:
    if diameter is not None and diameter <= 0:
        raise ValueError("Cell diameter must be > 0")
    if min_size < 0:
        raise ValueError("Minimum cell area must be >= 0")


# Native TIFF and Cellpose-only runs should remain usable without Fiji, while ImageJ work needs a real installation.
def _validate_fiji_path_if_needed(cfg: Config, targets) -> None:
    if not pipeline_needs_imagej(cfg, targets):
        return
    if not cfg.fiji_app_path.exists():
        raise SetupFileNotFoundError(SetupErrorCode.FIJI_MISSING, f"Fiji app path does not exist: {cfg.fiji_app_path}")
    if not cfg.fiji_app_path.is_dir() or not looks_like_fiji_installation(cfg.fiji_app_path):
        raise ValueError(f"Fiji app path does not look like a Fiji installation: {cfg.fiji_app_path}")
