"""Typed configuration values collected from the Cellonaut GUI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from cellonaut.config.defaults import (
    DEFAULT_BACKGROUND_RADII,
    DEFAULT_CELL_DIAMETER,
    DEFAULT_CELL_MIN_SIZE,
    DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH,
    DEFAULT_CELLPOSE_MODEL_TYPE,
    DEFAULT_QC_FILTER_MODE,
    DEFAULT_CELL_REMOVE_BORDER,
    DEFAULT_CELLPROB_THRESHOLD,
    DEFAULT_FLOW_THRESHOLD,
    DEFAULT_INPUT_STRUCTURE,
    DEFAULT_PROBABILITY_CLASS_INDEX,
    DEFAULT_STACK_CHANNEL_INDEX,
    DEFAULT_STACK_Z_INDEX,
    DEFAULT_STACK_Z_MODE,
    DEFAULT_THRESHOLD_METHOD,
    DEFAULT_EXCLUSION_TAG,
    DEFAULT_MASK_QC_INTENSITY_SOURCE,
    DEFAULT_MASK_SOURCE_MODE,
    MASK_SOURCE_MODE_OPTIONS,
    COMBINED_MASK_OPERATION_OPTIONS,
    DEFAULT_COMBINED_MASK_OPERATION,
    MINIMUM_IMAGE_COUNT,
    coerce_bool,
    default_image_definition,
    normalize_cellpose_model_type,
    default_image_definitions,
    normalize_image_processing_steps,
    normalize_mask_processing_steps,
    default_mask_adjustments,
    default_measurement_options,
)


# Preserve editable strings and incomplete rows separately from validated
# pipeline types because users may save or repair a partly configured preset.
@dataclass
class ImageGuiState:
    name: str
    folder: str
    display_color: str = ""
    classifier: str = ""
    mask_source_mode: str = DEFAULT_MASK_SOURCE_MODE
    mask_slot_enabled: bool = True
    is_mask_only: bool = False
    mask_source_channel: str = ""
    combined_mask_sources: list[str] = field(default_factory=list)
    combined_mask_operation: str = DEFAULT_COMBINED_MASK_OPERATION
    mask_adjustments: dict[str, int] = field(default_factory=default_mask_adjustments)
    cell_mask_adjustments: dict[str, int] = field(default_factory=default_mask_adjustments)
    stack_channel_index: str = DEFAULT_STACK_CHANNEL_INDEX
    stack_z_mode: str = DEFAULT_STACK_Z_MODE
    stack_z_index: str = DEFAULT_STACK_Z_INDEX

    probability_class_index: str = DEFAULT_PROBABILITY_CLASS_INDEX
    threshold_method: str = DEFAULT_THRESHOLD_METHOD
    bg_radii: str = DEFAULT_BACKGROUND_RADII
    image_processing_steps: list[dict[str, Any]] = field(default_factory=list)
    mask_processing_steps: list[dict[str, Any]] = field(default_factory=list)

    mask_relationships: dict[str, bool] = field(default_factory=dict)
    cell_group_mask_source: str = ""
    analysis_cell_segmentation_enabled: bool = False
    analysis_cell_segmentation_source: str = ""
    analysis_cellpose_mask_sources: list[str] = field(default_factory=list)
    # First selected mask retained for old presets and preview fallbacks.
    analysis_cellpose_mask_source: str = ""

    cell_diameter: str = DEFAULT_CELL_DIAMETER
    cell_min_size: str = DEFAULT_CELL_MIN_SIZE
    cellprob_threshold: str = DEFAULT_CELLPROB_THRESHOLD
    flow_threshold: str = DEFAULT_FLOW_THRESHOLD
    cell_remove_border: bool = DEFAULT_CELL_REMOVE_BORDER
    cellpose_model_type: str = DEFAULT_CELLPOSE_MODEL_TYPE
    cellpose_custom_model_path: str = DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH
    cell_populations: list[dict[str, Any]] = field(default_factory=list)

    # Presets may be incomplete or contain loosely typed JSON values. Merge
    # them with current defaults here so the rest of the GUI sees one stable shape.
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        index: int,
    ) -> "ImageGuiState":
        defaults = default_image_definition(index)
        base = dict(defaults)
        if isinstance(data, dict):
            base.update(data)

        string_keys = (
            "name",
            "folder",
            "display_color",
            "classifier",
            "mask_source_mode",
            "mask_source_channel",
            "probability_class_index",
            "threshold_method",
            "bg_radii",
            "stack_channel_index",
            "stack_z_mode",
            "stack_z_index",
            "analysis_cell_segmentation_source",
            "analysis_cellpose_mask_source",
            "cell_group_mask_source",
            "cell_diameter",
            "cell_min_size",
            "cellprob_threshold",
            "flow_threshold",
            "cellpose_model_type",
            "cellpose_custom_model_path",
        )
        for key in string_keys:
            base[key] = "" if base.get(key) is None else str(base.get(key, ""))
        if base["mask_source_mode"] not in MASK_SOURCE_MODE_OPTIONS:
            base["mask_source_mode"] = DEFAULT_MASK_SOURCE_MODE
            base["classifier"] = ""

        base["cellpose_model_type"] = normalize_cellpose_model_type(base["cellpose_model_type"])
        bool_keys = (
            "analysis_cell_segmentation_enabled",
            "cell_remove_border",
            "mask_slot_enabled",
            "is_mask_only",
        )
        for key in bool_keys:
            base[key] = coerce_bool(base.get(key), bool(defaults[key]))

        mapping_keys = (
            "mask_relationships",
            "mask_adjustments",
            "cell_mask_adjustments",
        )
        for key in mapping_keys:
            value = base.get(key, {})
            base[key] = dict(value) if isinstance(value, dict) else {}
        base["mask_relationships"] = {str(key): coerce_bool(value) for key, value in base["mask_relationships"].items()}
        supplied_populations = isinstance(data, dict) and "cell_populations" in data
        populations = base.get("cell_populations", []) if supplied_populations else []
        base["cell_populations"] = (
            [dict(value) for value in populations if isinstance(value, dict)] if isinstance(populations, list) else []
        )
        # Compatibility boundary: pre-Cell-Group presets used channel-level
        # settings. Translate once; current groups take precedence over stale
        # mirrors, and serialization writes only the group representation.
        if not base["cell_populations"]:
            base["cell_populations"] = [
                {
                    "name": "Cell group 1",
                    "color": "#00d7ff",
                    "cell_qc_limits": str(base.get("cell_qc_limits") or ""),
                    "mask_qc_limits": str(base.get("mask_qc_limits") or ""),
                    "qc_filter_mode": base.get("qc_filter_mode", DEFAULT_QC_FILTER_MODE),
                    "mask_qc_intensity_source": base.get("mask_qc_intensity_source", DEFAULT_MASK_QC_INTENSITY_SOURCE),
                    "exclude_from_csv": coerce_bool(base.get("cell_qc_exclude_flagged"), False),
                }
            ]
        for population in base["cell_populations"]:
            # Some early group presets omitted values and relied on channel
            # mirrors. Materialize those defaults here, before dropping them.
            for key, fallback in (
                ("cell_qc_limits", ""),
                ("mask_qc_limits", ""),
                ("qc_filter_mode", DEFAULT_QC_FILTER_MODE),
                ("mask_qc_intensity_source", DEFAULT_MASK_QC_INTENSITY_SOURCE),
            ):
                population.setdefault(key, base.get(key, fallback))
            population["exclude_from_csv"] = coerce_bool(population.get("exclude_from_csv"), False)
        raw_steps = (
            data.get("image_processing_steps") if isinstance(data, dict) and "image_processing_steps" in data else None
        )
        base["image_processing_steps"] = normalize_image_processing_steps(
            raw_steps,
            {
                "bg_radii": base.get("bg_radii", ""),
            },
        )
        raw_mask_steps = (
            data.get("mask_processing_steps") if isinstance(data, dict) and "mask_processing_steps" in data else None
        )
        base["mask_processing_steps"] = normalize_mask_processing_steps(raw_mask_steps)
        sources = base.get("combined_mask_sources", [])
        base["combined_mask_sources"] = (
            [str(value).strip() for value in sources if str(value).strip()]
            if isinstance(sources, (list, tuple))
            else []
        )
        raw_cellpose_sources = base.get("analysis_cellpose_mask_sources", [])
        cellpose_sources = (
            [str(value).strip() for value in raw_cellpose_sources if str(value or "").strip()]
            if isinstance(raw_cellpose_sources, (list, tuple))
            else []
        )
        legacy_cellpose_source = str(base.get("analysis_cellpose_mask_source", "") or "").strip()
        if not cellpose_sources and legacy_cellpose_source:
            cellpose_sources = [legacy_cellpose_source]
        base["analysis_cellpose_mask_sources"] = list(dict.fromkeys(cellpose_sources))
        base["analysis_cellpose_mask_source"] = (
            base["analysis_cellpose_mask_sources"][0]
            if base["analysis_cellpose_mask_sources"]
            else ""
        )
        operation = str(base.get("combined_mask_operation", DEFAULT_COMBINED_MASK_OPERATION) or "").strip().upper()
        base["combined_mask_operation"] = (
            operation if operation in COMBINED_MASK_OPERATION_OPTIONS else DEFAULT_COMBINED_MASK_OPERATION
        )

        if not base.get("analysis_cell_segmentation_source"):
            base["analysis_cell_segmentation_source"] = base["name"]
        if (
            isinstance(data, dict)
            and "analysis_cellpose_mask_sources" not in data
            and "analysis_cellpose_mask_source" not in data
            and base["analysis_cell_segmentation_enabled"]
        ):
            # Before reusable Cellpose columns, enabling a row both created and
            # selected that row's cell mask. Preserve that behavior on import.
            base["analysis_cellpose_mask_source"] = base["name"]
            base["analysis_cellpose_mask_sources"] = [base["name"]]

        return cls(**{field_name: base[field_name] for field_name in cls.__dataclass_fields__})

    # Preset files must contain plain JSON-compatible values rather than
    # dataclass or Qt objects, so serialization happens at this boundary.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# A detached, typed snapshot for conversion into execution configuration.
@dataclass
class CellonautGuiState:
    fiji_app_path: str = ""
    input_dir: str = ""
    output_dir: str = ""
    input_structure: str = DEFAULT_INPUT_STRUCTURE
    exclusion_tag: str = DEFAULT_EXCLUSION_TAG
    reuse_existing_masks: bool = False
    mask_source_dir: str = ""
    measurement_options: dict[str, bool] = field(default_factory=default_measurement_options)
    image_definitions: list[ImageGuiState] = field(
        default_factory=lambda: [ImageGuiState.from_dict(item, i) for i, item in enumerate(default_image_definitions())]
    )

    # Widget values are strings and may represent a partly built interface.
    # Normalize them before validation so pipeline code stays independent of Qt.
    @classmethod
    def from_widget_values(
        cls,
        *,
        fiji_app_path: str,
        input_dir: str,
        output_dir: str,
        input_structure: str,
        exclusion_tag: str | None = None,
        measurement_options: dict | None = None,
        image_definitions: list[dict] | list[ImageGuiState] | None = None,
        reuse_existing_masks: bool = False,
        mask_source_dir: str = "",
    ) -> "CellonautGuiState":
        options = default_measurement_options()
        if isinstance(measurement_options, dict):
            for key in options:
                if key in measurement_options:
                    options[key] = coerce_bool(measurement_options[key], options[key])

        images: list[ImageGuiState] = []
        raw_defs = image_definitions if isinstance(image_definitions, list) else []
        for idx, item in enumerate(raw_defs):
            if isinstance(item, ImageGuiState):
                images.append(item)
            else:
                images.append(ImageGuiState.from_dict(item if isinstance(item, dict) else None, idx))

        while len(images) < MINIMUM_IMAGE_COUNT:
            images.append(ImageGuiState.from_dict(None, len(images)))

        return cls(
            fiji_app_path=str(fiji_app_path or "").strip(),
            input_dir=str(input_dir or "").strip(),
            output_dir=str(output_dir or "").strip(),
            input_structure=str(input_structure or DEFAULT_INPUT_STRUCTURE),
            exclusion_tag=str(DEFAULT_EXCLUSION_TAG if exclusion_tag is None else exclusion_tag).strip(),
            reuse_existing_masks=coerce_bool(reuse_existing_masks),
            mask_source_dir=str(mask_source_dir or "").strip(),
            measurement_options=options,
            image_definitions=images,
        )


@dataclass
class EditableGuiConfiguration:
    """Committed GUI data. Dictionaries retain the existing editor/preset schema.

    Widgets may hold pending edits. Only commit_gui_edits captures them; readers
    and snapshot conversion never consult widgets or repair this state.
    """

    fiji_app_path: str = ""
    input_dir: str = ""
    output_dir: str = ""
    input_structure: str = DEFAULT_INPUT_STRUCTURE
    mask_source_dir: str = ""
    reuse_existing_masks: bool = False
    image_definitions: list[dict[str, Any]] = field(default_factory=default_image_definitions)
    measurement_options: dict[str, bool] = field(default_factory=default_measurement_options)
    panel_notes: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> CellonautGuiState:
        return CellonautGuiState(
            fiji_app_path=self.fiji_app_path,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            input_structure=self.input_structure,
            mask_source_dir=self.mask_source_dir,
            reuse_existing_masks=self.reuse_existing_masks,
            measurement_options=dict(self.measurement_options),
            image_definitions=[ImageGuiState(**deepcopy(item)) for item in self.image_definitions],
        )
