"""Typed configuration objects shared by pipeline planning and execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from cellonaut.config.defaults import (
    DEFAULT_CELL_MIN_SIZE,
    DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH,
    DEFAULT_CELLPOSE_MODEL_TYPE,
    DEFAULT_CELL_REMOVE_BORDER,
    DEFAULT_CELLPROB_THRESHOLD,
    DEFAULT_FLOW_THRESHOLD,
    DEFAULT_MASK_SOURCE_MODE,
    DEFAULT_COMBINED_MASK_OPERATION,
    DEFAULT_STACK_Z_MODE,
    DEFAULT_THRESHOLD_METHOD,
)


# Separate per-image acquisition and mask settings from run-wide configuration
# because each channel may use a different source, stack layer, and recipe.
@dataclass
class ImageDef:
    key: str
    label: str
    folder_name: str
    model_path: Optional[Path]
    display_color: str = ""
    mask_source_mode: str = DEFAULT_MASK_SOURCE_MODE
    combined_mask_source_keys: List[str] = field(default_factory=list)
    combined_mask_operation: str = DEFAULT_COMBINED_MASK_OPERATION
    stack_channel_index: Optional[int] = None
    stack_z_mode: str = DEFAULT_STACK_Z_MODE
    stack_z_index: Optional[int] = None

    bg_radii_csv: str = ""
    probability_class_index: Any = 1
    threshold_method: str = DEFAULT_THRESHOLD_METHOD
    image_processing_steps: List[Dict[str, Any]] = field(default_factory=list)
    mask_processing_steps: List[Dict[str, Any]] = field(default_factory=list)
    # Mask-only definitions reuse their assigned physical channel instead of
    # consuming another automatically inferred TIFF stack layer.
    stack_source_image_key: str = ""


class DiameterDefault(str, Enum):
    """Explicit request for the run's diameter; None means original scale."""

    INHERIT = "use_run_default"


@dataclass
class MeasurementTarget:
    """Target choices before resolution, independent of editable GUI strings.

    Omitted diameter uses DiameterDefault.INHERIT; an explicit None keeps the
    original scale. For non-nullable scalar settings and collections, None
    explicitly requests the run default. Empty containers, empty custom-model
    paths, False and zero are values, never inheritance requests.

    Source/overlay relationships belong only to this target. Cell Group
    conditions belong to GUI/preset state, never to pipeline execution.
    The GUI adapter supplies every setting, including blank/empty values.
    """
    source_image_key: str
    enabled: bool = True

    overlay_base_image_key: str = ""
    overlay_roi_keys: List[str] = field(default_factory=list)

    do_cell_segmentation: bool = False
    cell_segmentation_source: str = ""
    per_cell_mask_source: str = ""
    overlay_whole_cell_mask: bool = False

    measurement_options: Optional[Dict[str, bool]] = None

    cell_diameter: float | None | DiameterDefault = DiameterDefault.INHERIT
    cell_min_size: Optional[int] = None
    cell_use_gpu: Optional[bool] = None
    cellprob_threshold: Optional[float] = None
    flow_threshold: Optional[float] = None
    cell_remove_border: Optional[bool] = None
    cellpose_model_type: Optional[str] = None
    cellpose_custom_model_path: Optional[str] = None
    cell_mask_adjustments: Optional[Dict[str, int]] = None


@dataclass
class ResolvedMeasurementTarget:
    """Execution settings with no inheritance left to evaluate.

    Construct through resolve_measurement_target. Mutable collections are
    owned by this target, including nested group/rule dictionaries.
    """

    source_image_key: str
    enabled: bool = True
    overlay_base_image_key: str = ""
    overlay_roi_keys: List[str] = field(default_factory=list)
    do_cell_segmentation: bool = False
    cell_segmentation_source: str = ""
    per_cell_mask_source: str = ""
    overlay_whole_cell_mask: bool = False
    measurement_options: Dict[str, bool] = field(default_factory=dict)
    cell_diameter: Optional[float] = None
    cell_min_size: int = int(DEFAULT_CELL_MIN_SIZE)
    cell_use_gpu: bool = True
    cellprob_threshold: float = float(DEFAULT_CELLPROB_THRESHOLD)
    flow_threshold: float = float(DEFAULT_FLOW_THRESHOLD)
    cell_remove_border: bool = DEFAULT_CELL_REMOVE_BORDER
    cellpose_model_type: str = DEFAULT_CELLPOSE_MODEL_TYPE
    cellpose_custom_model_path: str = DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH
    cell_mask_adjustments: Dict[str, int] = field(default_factory=dict)


TargetConfig = MeasurementTarget | ResolvedMeasurementTarget


# Pass analysis settings to workers without including GUI objects.
@dataclass
class Config:
    """Run resources plus explicit defaults for programmatically built targets.

    Paths, images, input layout, exclusion tag and mask reuse are run-wide.
    Measurement options are a shared GUI choice and a default for targets.
    Cellpose fields here are defaults, never a copy of a GUI channel.
    ImageDef owns each image's Weka class, threshold and processing recipe;
    the run's threshold/class fields remain defaults for existing callers.

    The source/overlay and Cellpose fields also support the existing flat
    sample-execution API. config_for_measurement_target materializes a fresh
    Config for that API; execution never consults another target or a proxy.
    """
    fiji_app_path: Path
    input_dir: Path
    output_dir: Path
    input_structure: str
    images: List[ImageDef]

    exclusion_tag: str
    threshold_method: str
    probability_class_index: int
    measurement_options: Dict[str, bool] = field(default_factory=dict)
    measurement_targets: List[TargetConfig] = field(default_factory=list)

    source_image_key: str = ""
    overlay_base_image_key: str = ""
    overlay_roi_keys: List[str] = field(default_factory=list)
    do_cell_segmentation: bool = False
    cell_segmentation_source: str = ""
    per_cell_mask_source: str = ""
    overlay_whole_cell_mask: bool = False

    cell_diameter: Optional[float] = None
    cell_min_size: int = int(DEFAULT_CELL_MIN_SIZE)
    cell_use_gpu: bool = True
    cellprob_threshold: float = float(DEFAULT_CELLPROB_THRESHOLD)
    flow_threshold: float = float(DEFAULT_FLOW_THRESHOLD)
    cell_remove_border: bool = DEFAULT_CELL_REMOVE_BORDER
    cellpose_model_type: str = DEFAULT_CELLPOSE_MODEL_TYPE
    cellpose_custom_model_path: str = DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH
    cell_mask_adjustments: Dict[str, int] = field(default_factory=dict)

    reuse_existing_masks: bool = False
    mask_source_dir: Optional[Path] = None


@dataclass(frozen=True, slots=True)
class SampleTargetStatus:
    """One durable sample/measurement-target outcome written to run manifests."""

    run_type: str
    sample_id: str
    target: str
    status: str
    source_image_file: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "run_type": self.run_type,
            "sample_id": self.sample_id,
            "target": self.target,
            "status": self.status,
            "source_image_file": self.source_image_file,
            "reason": self.reason,
        }


class FullRunStats(TypedDict):
    total_samples: int
    processed: int
    skipped: int
    failed: int
    total_targets: int
    processed_targets: int
    skipped_targets: int
    failed_targets: int
    elapsed_seconds: float
    warning_count: int


def resolve_measurement_target(cfg: Config, target: TargetConfig) -> ResolvedMeasurementTarget:
    """Resolve explicit inheritance once, preserving valid empty/unset values."""
    if isinstance(target, ResolvedMeasurementTarget):
        return target

    return ResolvedMeasurementTarget(
        source_image_key=target.source_image_key,
        enabled=target.enabled,
        overlay_base_image_key=target.overlay_base_image_key,
        overlay_roi_keys=deepcopy(target.overlay_roi_keys),
        do_cell_segmentation=target.do_cell_segmentation,
        cell_segmentation_source=(
            target.cell_segmentation_source or target.source_image_key
            if target.do_cell_segmentation else target.cell_segmentation_source
        ),
        per_cell_mask_source=target.per_cell_mask_source,
        overlay_whole_cell_mask=target.overlay_whole_cell_mask,
        measurement_options=deepcopy(cfg.measurement_options if target.measurement_options is None else target.measurement_options),
        cell_diameter=cfg.cell_diameter if isinstance(target.cell_diameter, DiameterDefault) else target.cell_diameter,
        cell_min_size=cfg.cell_min_size if target.cell_min_size is None else target.cell_min_size,
        cell_use_gpu=cfg.cell_use_gpu if target.cell_use_gpu is None else target.cell_use_gpu,
        cellprob_threshold=cfg.cellprob_threshold if target.cellprob_threshold is None else target.cellprob_threshold,
        flow_threshold=cfg.flow_threshold if target.flow_threshold is None else target.flow_threshold,
        cell_remove_border=cfg.cell_remove_border if target.cell_remove_border is None else target.cell_remove_border,
        cellpose_model_type=cfg.cellpose_model_type if target.cellpose_model_type is None else target.cellpose_model_type,
        cellpose_custom_model_path=(
            cfg.cellpose_custom_model_path if target.cellpose_custom_model_path is None else target.cellpose_custom_model_path
        ),
        cell_mask_adjustments=deepcopy(cfg.cell_mask_adjustments if target.cell_mask_adjustments is None else target.cell_mask_adjustments),
    )


def config_for_measurement_target(cfg: Config, target: TargetConfig) -> Config:
    """Materialize the flat sample API from run resources and a resolved target.

    Adapter/validation boundaries resolve targets before the sample loops.
    Direct callers can still supply an unresolved target. Each execution copy
    owns its target containers, so measurements cannot mutate the run settings.
    Run fields automatically survive dataclass replacement; target fields
    replace them by value, without dynamic fallback during execution.
    """
    resolved = deepcopy(resolve_measurement_target(cfg, target))
    settings = {item.name: getattr(resolved, item.name) for item in fields(MeasurementTarget) if item.name != "enabled"}
    return replace(cfg, measurement_targets=[resolved], **settings)
