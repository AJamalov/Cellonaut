"""Central defaults and constants for the Cellonaut GUI.

Keep user-facing defaults here instead of scattering them across the main window,
dataset helpers, dialogs, and config conversion code.
"""

from __future__ import annotations

from typing import Any

APP_WINDOW_DEFAULT_WIDTH = 1400
APP_WINDOW_DEFAULT_HEIGHT = 860
APP_WINDOW_MIN_WIDTH = 900
APP_WINDOW_MIN_HEIGHT = 560

DEFAULT_THEME = "dark_blue"


# Presets are editable JSON, and string values such as "false" must not become
# enabled merely because non-empty Python strings are truthy.
def coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return bool(default)


INPUT_STRUCTURE_SAMPLES_DIRECTLY = "Samples directly in input folder"
INPUT_STRUCTURE_GROUPED_BY_PROTEIN = "Protein folders containing sample folders"
INPUT_STRUCTURE_FLAT_TIFFS = "Unsorted TIFF images in input folder"
INPUT_STRUCTURE_IMAGE_FOLDERS_FLAT_TIFFS = "Image folders containing unsorted TIFF images"
DEFAULT_INPUT_STRUCTURE = INPUT_STRUCTURE_FLAT_TIFFS

DEFAULT_EXCLUSION_TAG = ""
DEFAULT_THRESHOLD_METHOD = "Default"
# Display the same global Auto Threshold methods and names used by Fiji while
# retaining ImageJ's API value for MinError (I).
THRESHOLD_METHOD_OPTIONS = [
    ("Default (ImageJ IsoData)", "Default"),
    "Huang",
    "Intermodes",
    "IsoData",
    "Li",
    "MaxEntropy",
    "Mean",
    ("MinError (I)", "MinError"),
    "Minimum",
    "Moments",
    "Otsu",
    "Percentile",
    "RenyiEntropy",
    "Shanbhag",
    "Triangle",
    "Yen",
]
DEFAULT_PROBABILITY_CLASS_INDEX = "1"
DEFAULT_BACKGROUND_RADII = ""
DEFAULT_STACK_CHANNEL_INDEX = ""
STACK_Z_MODE_MAX_PROJECTION = "max_projection"
STACK_Z_MODE_SINGLE_Z = "single_z"
STACK_Z_MODE_OPTIONS = [
    STACK_Z_MODE_MAX_PROJECTION,
    STACK_Z_MODE_SINGLE_Z,
]
DEFAULT_STACK_Z_MODE = STACK_Z_MODE_MAX_PROJECTION
DEFAULT_STACK_Z_INDEX = ""

IMAGE_PROCESSING_SCOPE_SEGMENTATION = "Segmentation input"
IMAGE_PROCESSING_SCOPE_MEASUREMENT = "Measurement image"
IMAGE_PROCESSING_SCOPE_OPTIONS = [
    IMAGE_PROCESSING_SCOPE_SEGMENTATION,
    IMAGE_PROCESSING_SCOPE_MEASUREMENT,
]

DEFAULT_CELL_DIAMETER = ""
DEFAULT_CELL_MIN_SIZE = "15"
DEFAULT_CELLPROB_THRESHOLD = "0.0"
DEFAULT_FLOW_THRESHOLD = "0.4"
DEFAULT_CELL_REMOVE_BORDER = True
DEFAULT_CELLPOSE_MODEL_TYPE = "cpsam"
CELLPOSE_MODEL_OPTIONS = ("cpsam", "cpsam_v2")


def normalize_cellpose_model_type(value: object) -> str:
    """Use the current default when no built-in Cellpose model was selected."""
    model_type = str(value or "").strip()
    return model_type or DEFAULT_CELLPOSE_MODEL_TYPE


DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH = ""
DEFAULT_QC_FILTER_MODE = "Exclude if any filter fails"
QC_FILTER_MODE_OPTIONS = (
    "Exclude if any filter fails",
    "Exclude only if both fail",
    "Use cell filters only",
    "Use mask filters only",
)
DEFAULT_MASK_QC_INTENSITY_SOURCE = "Measured image"

# Keep editor and report wording consistent while preserving stored preset values.
QC_FILTER_MODE_LABELS: dict[str, str] = dict(zip(QC_FILTER_MODE_OPTIONS, (
    "Match all conditions",
    "Match either category",
    "Match cell properties only",
    "Match target-mask properties only",
), strict=True))
MASK_INTENSITY_SOURCE_LABELS = {
    "Measured image": "Measured channel",
    "Cell mask image": "Cellpose source channel",
}


# Store one canonical label for each filter strategy so dialogs, presets, and
# the filtering engine cannot interpret equivalent selections differently.
def normalize_qc_filter_mode(value: str) -> str:
    text = str(value or "").strip().lower()
    mapping = {option.lower(): option for option in QC_FILTER_MODE_OPTIONS}
    return mapping.get(text, "Exclude if any filter fails")


MINIMUM_IMAGE_COUNT = 1
STARTUP_IMAGE_COUNT = 1

DEFAULT_MASK_SOURCE_MODE = "Weka classifier"
MASK_SOURCE_MODE_WEKA = "Weka classifier"
MASK_SOURCE_MODE_COMBINED = "Combined masks"
COMBINED_MASK_OPERATION_OPTIONS = ("OR", "AND", "XOR")
DEFAULT_COMBINED_MASK_OPERATION = "OR"
MASK_SOURCE_MODE_OPTIONS = [
    MASK_SOURCE_MODE_WEKA,
    MASK_SOURCE_MODE_COMBINED,
]

IMAGE_PROCESSING_STEP_ROLLING_BALL = "rolling_ball_background"
IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST = "enhance_contrast"
IMAGE_PROCESSING_STEP_APPLY_LUT = "apply_lut"
IMAGE_PROCESSING_STEP_SMOOTH = "smooth"
IMAGE_PROCESSING_STEP_GAUSSIAN_BLUR = "gaussian_blur"
IMAGE_PROCESSING_STEP_MEDIAN = "median"
IMAGE_PROCESSING_STEP_DESPECKLE = "despeckle"
IMAGE_PROCESSING_STEP_REMOVE_OUTLIERS = "remove_outliers"
IMAGE_PROCESSING_STEP_BIT_DEPTH = "bit_depth"
DEFAULT_CONTRAST_SATURATION = "0.35"
DEFAULT_SMOOTH_ITERATIONS = "5"
DEFAULT_GAUSSIAN_SIGMA = "1.0"
DEFAULT_MEDIAN_RADIUS = "1.0"
DEFAULT_BIT_DEPTH = "8-bit"
IMAGE_PROCESSING_BIT_DEPTH_OPTIONS = ["8-bit", "16-bit", "32-bit"]

IMAGE_PROCESSING_STEP_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": IMAGE_PROCESSING_STEP_ROLLING_BALL,
        "label": "Subtract Background",
        "param_key": "bg_radii",
        "default": DEFAULT_BACKGROUND_RADII,
        "default_scope": IMAGE_PROCESSING_SCOPE_MEASUREMENT,
        "allowed_scopes": [
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        ],
        "measurement_safe": True,
        "validation": {
            "kind": "csv_float",
            "field_name": "Subtract Background radius",
            "minimum": 0,
            "allow_blank": True,
        },
    },
    {
        "type": IMAGE_PROCESSING_STEP_GAUSSIAN_BLUR,
        "label": "Gaussian Blur",
        "param_key": "gaussian_sigma",
        "default": DEFAULT_GAUSSIAN_SIGMA,
        "default_scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        "allowed_scopes": [
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
        ],
        "measurement_safe": False,
        "validation": {
            "kind": "float",
            "field_name": "Gaussian sigma",
            "minimum": 0.000001,
            "allow_blank": True,
        },
    },
    {
        "type": IMAGE_PROCESSING_STEP_MEDIAN,
        "label": "Median",
        "param_key": "median_radius",
        "default": DEFAULT_MEDIAN_RADIUS,
        "default_scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        "allowed_scopes": [
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
        ],
        "measurement_safe": False,
        "validation": {
            "kind": "float",
            "field_name": "Median radius",
            "minimum": 0.000001,
            "allow_blank": True,
        },
    },
    {
        "type": IMAGE_PROCESSING_STEP_DESPECKLE,
        "label": "Despeckle",
        "param_key": "",
        "default": "",
        "default_scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        "allowed_scopes": [
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
        ],
        "measurement_safe": False,
        "validation": {"kind": "none", "field_name": "Despeckle"},
    },
    {
        "type": IMAGE_PROCESSING_STEP_REMOVE_OUTLIERS,
        "label": "Remove Outliers",
        "param_key": "outlier_settings",
        "default": "",
        "default_scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        "allowed_scopes": [IMAGE_PROCESSING_SCOPE_SEGMENTATION, IMAGE_PROCESSING_SCOPE_MEASUREMENT],
        "measurement_safe": False,
        "validation": {"kind": "outliers", "field_name": "Remove Outliers", "allow_blank": True},
    },
    {
        "type": IMAGE_PROCESSING_STEP_ENHANCE_CONTRAST,
        "label": "Enhance contrast",
        "param_key": "contrast_saturation",
        "default": DEFAULT_CONTRAST_SATURATION,
        "default_scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        "allowed_scopes": [
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
        ],
        "measurement_safe": False,
        "validation": {
            "kind": "float",
            "field_name": "Contrast saturation",
            "minimum": 0,
            "maximum": 100,
            "allow_blank": True,
        },
    },
    {
        "type": IMAGE_PROCESSING_STEP_APPLY_LUT,
        "label": "Apply LUT",
        "param_key": "",
        "default": "",
        "default_scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        "allowed_scopes": [
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
        ],
        "measurement_safe": False,
        "validation": {"kind": "none", "field_name": "Apply LUT"},
    },
    {
        "type": IMAGE_PROCESSING_STEP_SMOOTH,
        "label": "Smooth",
        "param_key": "smooth_iterations",
        "default": DEFAULT_SMOOTH_ITERATIONS,
        "default_scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        "allowed_scopes": [
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
        ],
        "measurement_safe": False,
        "validation": {
            "kind": "int",
            "field_name": "Smooth iterations",
            "minimum": 1,
            "maximum": 100,
            "allow_blank": True,
        },
    },
    {
        "type": IMAGE_PROCESSING_STEP_BIT_DEPTH,
        "label": "Convert Bit Depth",
        "param_key": "bit_depth",
        "default": DEFAULT_BIT_DEPTH,
        "default_scope": IMAGE_PROCESSING_SCOPE_SEGMENTATION,
        "allowed_scopes": [
            IMAGE_PROCESSING_SCOPE_SEGMENTATION,
            IMAGE_PROCESSING_SCOPE_MEASUREMENT,
        ],
        "measurement_safe": False,
        "validation": {
            "kind": "choice",
            "field_name": "Bit depth",
            "choices": IMAGE_PROCESSING_BIT_DEPTH_OPTIONS,
            "allow_blank": True,
        },
    },
]


# Each operation allows different uses in segmentation and measurement. Invalid
# combinations fall back here before they can alter quantitative results.
def normalize_image_processing_scope(step_type: str, scope: str | None) -> str:
    definitions = {str(step["type"]): step for step in IMAGE_PROCESSING_STEP_DEFINITIONS}
    definition = definitions.get(str(step_type or ""))
    if definition is None:
        return ""

    allowed = [str(value) for value in definition.get("allowed_scopes", IMAGE_PROCESSING_SCOPE_OPTIONS)]
    default_scope = str(definition.get("default_scope", allowed[0] if allowed else "") or "")
    requested = str(scope or "").strip()
    return requested if requested in allowed else default_scope


# New channels start without hidden processing. Explicit flat values are turned
# into rows only when a caller supplies them as part of an existing state.
def default_image_processing_steps(values: dict | None = None) -> list[dict]:
    values = values if isinstance(values, dict) else None
    if values is None:
        return []

    steps = []
    for step in IMAGE_PROCESSING_STEP_DEFINITIONS:
        param_key = str(step.get("param_key", "") or "")
        if not param_key:
            continue
        value = values.get(param_key, "")
        if str(value or "").strip():
            steps.append(
                {
                    "type": step["type"],
                    "enabled": True,
                    "scope": normalize_image_processing_scope(
                        str(step["type"]),
                        str(step.get("default_scope", "")),
                    ),
                    "params": {param_key: value},
                }
            )
    return steps


# Processing rows come from editable tables and JSON files. Rebuild their shape
# here, preserving blank UI rows while dropping unknown operations.
def normalize_image_processing_steps(
    steps: list | None,
    values: dict | None = None,
) -> list[dict]:
    values = values if isinstance(values, dict) else {}
    definitions = {str(step["type"]): step for step in IMAGE_PROCESSING_STEP_DEFINITIONS}
    if steps is None:
        return default_image_processing_steps(values)

    normalized: list[dict] = []

    for raw_step in steps if isinstance(steps, list) else []:
        if not isinstance(raw_step, dict):
            continue
        step_type = str(raw_step.get("type", "") or "")
        definition = definitions.get(step_type)
        if not step_type:
            raw_scope = str(raw_step.get("scope", "") or "")
            normalized.append(
                {
                    "type": "",
                    "enabled": coerce_bool(raw_step.get("enabled", True), True),
                    "row_enabled": coerce_bool(raw_step.get("row_enabled", False), False),
                    "scope": raw_scope if raw_scope in IMAGE_PROCESSING_SCOPE_OPTIONS else "",
                    "params": {},
                }
            )
            continue

        if definition is None:
            continue
        params = dict(raw_step.get("params", {}) or {})
        param_key = str(definition.get("param_key", "") or "")
        if param_key and param_key not in params:
            params[param_key] = values.get(param_key, definition["default"])
        normalized.append(
            {
                "type": step_type,
                "enabled": coerce_bool(raw_step.get("enabled", True), True),
                "row_enabled": coerce_bool(
                    raw_step.get("row_enabled", raw_step.get("enabled", True)), True
                ),
                "scope": normalize_image_processing_scope(
                    step_type,
                    str(raw_step.get("scope", "") or ""),
                ),
                "params": params,
            }
        )

    # Measurement background radii create parallel corrected outputs after all
    # image transforms; keep their rows last so the visible order matches execution.
    normalized.sort(
        key=lambda step: int(
            step["scope"] == IMAGE_PROCESSING_SCOPE_MEASUREMENT
            and step["type"] == IMAGE_PROCESSING_STEP_ROLLING_BALL
        )
    )
    return normalized


# Callers that have no explicit recipe retain the pipeline's original behavior;
# once rows exist, their selected scope becomes authoritative.
def image_processing_step_has_scope(
    image_def,
    step_type: str,
    scope: str,
) -> bool:
    steps = list(getattr(image_def, "image_processing_steps", []) or [])
    if not steps:
        return True
    return any(
        str(step.get("type", "") or "") == step_type and str(step.get("scope", "") or "") == scope
        for step in steps
        if isinstance(step, dict)
    )


# Repeated processing rows are valid and may create several measurement outputs,
# so return every enabled value in recipe order rather than only the first.
def image_processing_step_param_values(
    image_def,
    step_type: str,
    scope: str,
    param_key: str,
) -> list[str]:
    values = []
    for step in list(getattr(image_def, "image_processing_steps", []) or []):
        if not isinstance(step, dict) or not coerce_bool(step.get("enabled", True), True):
            continue
        if str(step.get("type", "") or "") != step_type:
            continue
        if str(step.get("scope", "") or "") != scope:
            continue
        params = dict(step.get("params", {}) or {})
        value = str(params.get(param_key, "") or "").strip()
        if value:
            values.append(value)
    return values


MASK_PROCESSING_STEP_ANALYZE_SKELETON = "analyze_skeleton"
MASK_PROCESSING_STEP_BINARY_ERODE = "binary_erode"
MASK_PROCESSING_STEP_BINARY_DILATE = "binary_dilate"
MASK_PROCESSING_STEP_BINARY_OPEN = "binary_open"
MASK_PROCESSING_STEP_BINARY_CLOSE = "binary_close"
MASK_PROCESSING_STEP_BINARY_FILL_HOLES = "binary_fill_holes"
MASK_PROCESSING_STEP_BINARY_WATERSHED = "binary_watershed"
MASK_PROCESSING_STEP_BINARY_OUTLINE = "binary_outline"
MASK_PROCESSING_STEP_BINARY_SKELETONIZE = "binary_skeletonize"
MASK_PROCESSING_STEP_ANALYZE_PARTICLES = "analyze_particles"
MASK_PROCESSING_STEP_TRANSLATE = "translate"

MASK_PROCESSING_STEP_DEFINITIONS = [
    {
        "type": MASK_PROCESSING_STEP_ANALYZE_SKELETON,
        "label": "Analyze skeleton",
        "param_key": "",
        "default": "",
        "validation": {"kind": "none", "field_name": "Analyze skeleton"},
    },
    *[
        {
            "type": step_type,
            "label": label,
            "param_key": "binary_settings",
            "default": "1,1,false",
            "validation": {"kind": "binary", "field_name": label, "allow_blank": False},
        }
        for step_type, label in (
            (MASK_PROCESSING_STEP_BINARY_ERODE, "Erode"),
            (MASK_PROCESSING_STEP_BINARY_DILATE, "Dilate"),
            (MASK_PROCESSING_STEP_BINARY_OPEN, "Open"),
            (MASK_PROCESSING_STEP_BINARY_CLOSE, "Close"),
        )
    ],
    *[
        {
            "type": step_type,
            "label": label,
            "param_key": "",
            "default": "",
            "validation": {"kind": "none", "field_name": label},
        }
        for step_type, label in (
            (MASK_PROCESSING_STEP_BINARY_FILL_HOLES, "Fill Holes"),
            (MASK_PROCESSING_STEP_BINARY_WATERSHED, "Watershed"),
            (MASK_PROCESSING_STEP_BINARY_OUTLINE, "Outline"),
            (MASK_PROCESSING_STEP_BINARY_SKELETONIZE, "Skeletonize"),
        )
    ],
    {
        "type": MASK_PROCESSING_STEP_ANALYZE_PARTICLES,
        "label": "Analyze Particles",
        "param_key": "particle_settings",
        "default": "0-Infinity,0-1,false,false",
        "validation": {"kind": "particles", "field_name": "Analyze Particles", "allow_blank": False},
    },
    {
        "type": MASK_PROCESSING_STEP_TRANSLATE,
        "label": "Translate",
        "param_key": "translate_offsets",
        "default": "0,0",
        "validation": {"kind": "translate", "field_name": "Translate", "allow_blank": False},
    },
]


# Mask processing is opt-in; silently adding morphology would change the shape
# and measurements of a newly configured mask.
def default_mask_processing_steps() -> list[dict]:
    return []


# Persisted table rows stay structurally safe while retaining an empty row that
# the GUI uses as the user's next insertion point.
def normalize_mask_processing_steps(
    steps: list | None,
) -> list[dict]:
    definitions = {str(step["type"]): step for step in MASK_PROCESSING_STEP_DEFINITIONS}
    if steps is None:
        return []

    normalized: list[dict] = []
    for raw_step in steps if isinstance(steps, list) else []:
        if not isinstance(raw_step, dict):
            continue
        step_type = str(raw_step.get("type", "") or "")
        if not step_type:
            normalized.append(
                {
                    "type": "",
                    "enabled": coerce_bool(raw_step.get("enabled", False), False),
                    "row_enabled": coerce_bool(raw_step.get("row_enabled", False), False),
                    "params": {},
                }
            )
            continue
        definition = definitions.get(step_type)
        if definition is None:
            continue
        params = dict(raw_step.get("params", {}) or {})
        param_key = str(definition.get("param_key", "") or "")
        if param_key and param_key not in params:
            params[param_key] = definition.get("default", "")
        normalized.append(
            {
                "type": step_type,
                "enabled": coerce_bool(raw_step.get("enabled", True), True),
                "row_enabled": coerce_bool(
                    raw_step.get("row_enabled", raw_step.get("enabled", True)), True
                ),
                "params": params if param_key else {},
            }
        )
    return normalized


# Give each mask its own defaults so editing one cannot change another.
def default_mask_adjustments() -> dict[str, int]:
    return {
        "dx": 0,
        "dy": 0,
        "grow_px": 0,
        "min_size": 0,
        "fill_holes_area": 0,
    }


SHARED_MEASUREMENT_TERMS = (
    ("area", "cell_area", "positive_area_in_cell"),
    ("mean", "cell_mean", "mean_in_positive_area"),
    ("std_dev", "cell_std_dev", "std_dev_in_positive_area"),
    ("mode", "cell_mode", "mode_in_positive_area"),
    ("min_max", "cell_min_max", "min_max_in_positive_area"),
    ("centroid", "cell_centroid", "centroid_in_positive_area"),
    ("center_of_mass", "cell_center_of_mass", "center_of_mass_in_positive_area"),
    ("perimeter", "cell_perimeter", "perimeter_in_positive_area"),
    ("bounding_rect", "cell_bounding_rect", "bounding_rect_in_positive_area"),
    ("fit_ellipse", "cell_fit_ellipse", "fit_ellipse_in_positive_area"),
    ("feret", "cell_feret", "feret_in_positive_area"),
    ("circularity", "cell_circularity", "circularity_in_positive_area"),
    ("solidity", "cell_solidity", "solidity_in_positive_area"),
    ("raw_intden", "cell_raw_intden", "raw_intden_in_cell"),
    ("median", "cell_median", "median_in_positive_area"),
    ("skewness", "cell_skewness", "skewness_in_positive_area"),
    ("kurtosis", "cell_kurtosis", "kurtosis_in_positive_area"),
)

CONFIGURED_MASK_MEASUREMENT_KEYS = tuple(keys[0] for keys in SHARED_MEASUREMENT_TERMS)
CELLPOSE_MEASUREMENT_KEYS = tuple(keys[1] for keys in SHARED_MEASUREMENT_TERMS)
CONFIGURED_MASK_WITHIN_CELLPOSE_KEYS = tuple(keys[2] for keys in SHARED_MEASUREMENT_TERMS)


DEFAULT_MEASUREMENT_OPTIONS = {
    "area": True,
    "mean": True,
    "std_dev": False,
    "mode": False,
    "min_max": False,
    "centroid": False,
    "center_of_mass": False,
    "perimeter": False,
    "bounding_rect": False,
    "fit_ellipse": False,
    "feret": False,
    "circularity": False,
    "solidity": False,
    "raw_intden": True,
    "median": False,
    "skewness": False,
    "kurtosis": False,
    "cell_area": False,
    "cell_mean": False,
    "cell_std_dev": False,
    "cell_mode": False,
    "cell_min_max": False,
    "cell_centroid": False,
    "cell_center_of_mass": False,
    "cell_perimeter": False,
    "cell_bounding_rect": False,
    "cell_fit_ellipse": False,
    "cell_feret": False,
    "cell_circularity": False,
    "cell_solidity": False,
    "cell_median": False,
    "cell_raw_intden": False,
    "cell_skewness": False,
    "cell_kurtosis": False,
    "positive_area_in_cell": False,
    "mean_in_positive_area": False,
    "std_dev_in_positive_area": False,
    "mode_in_positive_area": False,
    "min_max_in_positive_area": False,
    "centroid_in_positive_area": False,
    "center_of_mass_in_positive_area": False,
    "perimeter_in_positive_area": False,
    "bounding_rect_in_positive_area": False,
    "fit_ellipse_in_positive_area": False,
    "feret_in_positive_area": False,
    "circularity_in_positive_area": False,
    "solidity_in_positive_area": False,
    "median_in_positive_area": False,
    "raw_intden_in_cell": False,
    "skewness_in_positive_area": False,
    "kurtosis_in_positive_area": False,
}

MEASUREMENT_METADATA = {
    "area": {
        "label": "Area",
        "unit": "px²",
        "description": "Number of pixels in the selected measurement region.",
    },
    "mean": {
        "label": "Mean gray value",
        "unit": "a.u.",
        "description": "Average pixel intensity inside the selected measurement region.",
    },
    "std_dev": {
        "label": "Standard deviation",
        "unit": "a.u.",
        "description": "Standard deviation of pixel intensities inside the selected measurement region.",
    },
    "mode": {
        "label": "Modal gray value",
        "unit": "a.u.",
        "description": "Most frequent pixel intensity inside the selected measurement region.",
    },
    "min_max": {
        "label": "Min & max gray value",
        "unit": "a.u.",
        "description": "Lowest and highest pixel intensity inside the selected measurement region.",
    },
    "centroid": {
        "label": "Centroid",
        "unit": "px",
        "description": "Geometric center of the selected measurement region.",
    },
    "center_of_mass": {
        "label": "Center of mass",
        "unit": "px",
        "description": "Intensity-weighted center of the selected measurement region.",
    },
    "perimeter": {
        "label": "Perimeter",
        "unit": "px",
        "description": "Length of the selected measurement-region boundary.",
    },
    "bounding_rect": {
        "label": "Bounding rectangle",
        "unit": "px",
        "description": "Bounding box position and size for the selected measurement region.",
    },
    "fit_ellipse": {
        "label": "Fit ellipse",
        "unit": "px; degrees for angle",
        "description": "Major axis, minor axis, and angle fitted to the selected measurement region.",
    },
    "feret": {
        "label": "Feret's diameter",
        "unit": "px; degrees for angle",
        "description": "Caliper measurements for the selected measurement region.",
    },
    "circularity": {
        "label": "Circularity",
        "unit": "unitless",
        "description": "Four pi times area divided by perimeter squared for the selected measurement region.",
    },
    "solidity": {
        "label": "Solidity",
        "unit": "0-1",
        "description": "Region area divided by convex-hull area for the selected measurement region.",
    },
    "raw_intden": {
        "label": "Integrated density",
        "unit": "intensity × px²",
        "description": "Sum of pixel intensities inside the selected measurement region. Also known as RawIntDen in ImageJ/Fiji exports.",
    },
    "median": {
        "label": "Median",
        "unit": "a.u.",
        "description": "Median pixel intensity inside the selected measurement region.",
    },
    "skewness": {
        "label": "Skewness",
        "unit": "unitless",
        "description": "Asymmetry of the pixel-intensity distribution in the selected measurement region.",
    },
    "kurtosis": {
        "label": "Kurtosis",
        "unit": "unitless",
        "description": "Tail weight of the pixel-intensity distribution in the selected measurement region.",
    },
    "cell_area": {
        "label": "Area",
        "unit": "px²",
        "description": "Area enclosed by each Cellpose cell mask.",
    },
    "cell_mean": {
        "label": "Mean gray value",
        "unit": "a.u.",
        "description": "Average measured-channel intensity inside each Cellpose cell.",
    },
    "cell_std_dev": {
        "label": "Standard deviation",
        "unit": "a.u.",
        "description": "Sample standard deviation of measured-channel intensities inside each Cellpose cell; blank for cells containing fewer than two pixels.",
    },
    "cell_mode": {
        "label": "Modal gray value",
        "unit": "a.u.",
        "description": "Most frequent measured-channel pixel value inside each Cellpose cell; ties use the lowest value.",
    },
    "cell_min_max": {
        "label": "Min & max gray value",
        "unit": "a.u.",
        "description": "Lowest and highest measured-channel intensity inside each Cellpose cell.",
    },
    "cell_centroid": {
        "label": "Centroid",
        "unit": "px",
        "description": "Geometric center of each Cellpose cell in zero-based pixel coordinates.",
    },
    "cell_center_of_mass": {
        "label": "Center of mass",
        "unit": "px",
        "description": "Measured-channel-intensity-weighted center of each Cellpose cell; blank when the intensity sum is zero.",
    },
    "cell_perimeter": {
        "label": "Perimeter",
        "unit": "px",
        "description": "Boundary length of each Cellpose cell mask.",
    },
    "cell_bounding_rect": {
        "label": "Bounding rectangle",
        "unit": "px",
        "description": "Zero-based bounding-box position and size for each Cellpose cell.",
    },
    "cell_fit_ellipse": {
        "label": "Fit ellipse",
        "unit": "px; degrees for angle",
        "description": "Major axis, minor axis, and orientation of the equivalent ellipse for each Cellpose cell.",
    },
    "cell_feret": {
        "label": "Feret's diameter",
        "unit": "px; degrees for angle",
        "description": "Maximum caliper diameter of each Cellpose cell.",
    },
    "cell_circularity": {
        "label": "Circularity",
        "unit": "unitless",
        "description": "Four pi times area divided by perimeter squared for each Cellpose cell.",
    },
    "cell_solidity": {
        "label": "Solidity",
        "unit": "0-1",
        "description": "Cell area divided by convex-hull area for each Cellpose cell.",
    },
    "cell_median": {
        "label": "Median",
        "unit": "a.u.",
        "description": "Median measured-channel intensity inside each Cellpose cell.",
    },
    "cell_raw_intden": {
        "label": "Integrated density",
        "unit": "intensity × px²",
        "description": "Sum of measured-channel pixel intensities inside each Cellpose cell.",
    },
    "cell_skewness": {
        "label": "Skewness",
        "unit": "unitless",
        "description": "Bias-corrected skewness of measured-channel intensities inside each Cellpose cell; blank when undefined.",
    },
    "cell_kurtosis": {
        "label": "Kurtosis",
        "unit": "unitless",
        "description": "Bias-corrected excess kurtosis of measured-channel intensities inside each Cellpose cell; blank when undefined.",
    },
    "positive_area_in_cell": {
        "label": "Area",
        "unit": "px²",
        "description": "Area where the selected mask overlaps each Cellpose cell.",
    },
    "mean_in_positive_area": {
        "label": "Mean gray value",
        "unit": "a.u.",
        "description": "Average measured-channel intensity where the selected mask overlaps each cell.",
    },
    "std_dev_in_positive_area": {
        "label": "Standard deviation",
        "unit": "a.u.",
        "description": "Sample standard deviation of measured-channel intensities where the mask overlaps each cell; blank for fewer than two pixels.",
    },
    "mode_in_positive_area": {
        "label": "Modal gray value",
        "unit": "a.u.",
        "description": "Most frequent measured-channel pixel value where the configured mask overlaps each Cellpose cell; ties use the lowest value.",
    },
    "min_max_in_positive_area": {
        "label": "Min & max gray value",
        "unit": "a.u.",
        "description": "Lowest and highest measured-channel intensity where the selected mask overlaps each cell.",
    },
    "centroid_in_positive_area": {
        "label": "Centroid",
        "unit": "px",
        "description": "Geometric center of the configured-mask region inside each Cellpose cell in zero-based pixel coordinates.",
    },
    "center_of_mass_in_positive_area": {
        "label": "Center of mass",
        "unit": "px",
        "description": "Measured-channel-intensity-weighted center of the configured-mask region inside each Cellpose cell; blank when the intensity sum is zero.",
    },
    "perimeter_in_positive_area": {
        "label": "Perimeter",
        "unit": "px",
        "description": "Boundary length of the configured-mask region inside each Cellpose cell.",
    },
    "bounding_rect_in_positive_area": {
        "label": "Bounding rectangle",
        "unit": "px",
        "description": "Zero-based bounding-box position and size of the configured-mask region inside each Cellpose cell.",
    },
    "fit_ellipse_in_positive_area": {
        "label": "Fit ellipse",
        "unit": "px; degrees for angle",
        "description": "Major axis, minor axis, and orientation of the equivalent ellipse for the configured-mask region inside each Cellpose cell.",
    },
    "feret_in_positive_area": {
        "label": "Feret's diameter",
        "unit": "px; degrees for angle",
        "description": "Maximum caliper diameter of the configured-mask region inside each Cellpose cell.",
    },
    "circularity_in_positive_area": {
        "label": "Circularity",
        "unit": "unitless",
        "description": "Four pi times area divided by perimeter squared for the configured-mask region inside each Cellpose cell.",
    },
    "solidity_in_positive_area": {
        "label": "Solidity",
        "unit": "0-1",
        "description": "Region area divided by convex-hull area for the configured-mask region inside each Cellpose cell.",
    },
    "median_in_positive_area": {
        "label": "Median",
        "unit": "a.u.",
        "description": "Median measured-channel intensity where the selected mask overlaps each cell.",
    },
    "raw_intden_in_cell": {
        "label": "Integrated density",
        "unit": "intensity × px²",
        "description": "Sum of measured-channel pixel intensities inside the selected mask and cell.",
    },
    "skewness_in_positive_area": {
        "label": "Skewness",
        "unit": "unitless",
        "description": "Bias-corrected skewness of measured-channel intensities in the configured-mask region inside each Cellpose cell; blank when undefined.",
    },
    "kurtosis_in_positive_area": {
        "label": "Kurtosis",
        "unit": "unitless",
        "description": "Bias-corrected excess kurtosis of measured-channel intensities in the configured-mask region inside each Cellpose cell; blank when undefined.",
    },
}

MEASUREMENT_LABELS = {
    key: (f"{metadata['label']} ({metadata['unit']})" if metadata.get("unit") else metadata["label"])
    for key, metadata in MEASUREMENT_METADATA.items()
}

DEFAULT_IMAGE_BASE_NAMES = ["Channel 1", "Channel 2", "Channel 3"]


# Measurement settings are mutable in the GUI, so callers need an independent
# copy rather than the module-level baseline itself.
def default_measurement_options() -> dict:
    return dict(DEFAULT_MEASUREMENT_OPTIONS)


# Channels and masks share one complete definition shape. Creating it centrally
# keeps dynamic GUI rows and loaded presets aligned as settings evolve.
def default_image_definition(index: int) -> dict:
    idx = int(index) + 1
    name = DEFAULT_IMAGE_BASE_NAMES[index] if index < len(DEFAULT_IMAGE_BASE_NAMES) else f"Channel {idx}"
    folder = name

    return {
        "name": name,
        "folder": folder,
        "display_color": "",
        "classifier": "",
        "mask_source_mode": DEFAULT_MASK_SOURCE_MODE,
        "mask_slot_enabled": True,
        "is_mask_only": False,
        "mask_source_channel": "",
        "combined_mask_sources": [],
        "combined_mask_operation": DEFAULT_COMBINED_MASK_OPERATION,
        "mask_adjustments": default_mask_adjustments(),
        "cell_mask_adjustments": default_mask_adjustments(),
        "stack_channel_index": DEFAULT_STACK_CHANNEL_INDEX,
        "stack_z_mode": DEFAULT_STACK_Z_MODE,
        "stack_z_index": DEFAULT_STACK_Z_INDEX,
        "probability_class_index": DEFAULT_PROBABILITY_CLASS_INDEX,
        "threshold_method": DEFAULT_THRESHOLD_METHOD,
        "bg_radii": DEFAULT_BACKGROUND_RADII,
        "image_processing_steps": default_image_processing_steps(),
        "mask_processing_steps": default_mask_processing_steps(),
        "mask_relationships": {},
        "cell_group_mask_source": "",
        "analysis_cell_segmentation_enabled": False,
        "analysis_cell_segmentation_source": name,
        "analysis_cellpose_mask_sources": [],
        "analysis_cellpose_mask_source": "",
        "cell_diameter": DEFAULT_CELL_DIAMETER,
        "cell_min_size": DEFAULT_CELL_MIN_SIZE,
        "cellprob_threshold": DEFAULT_CELLPROB_THRESHOLD,
        "flow_threshold": DEFAULT_FLOW_THRESHOLD,
        "cell_remove_border": DEFAULT_CELL_REMOVE_BORDER,
        "cellpose_model_type": DEFAULT_CELLPOSE_MODEL_TYPE,
        "cellpose_custom_model_path": DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH,
        "cell_populations": [
            {
                "name": "Cell group 1",
                "color": "#00d7ff",
                "cell_qc_limits": "",
                "mask_qc_limits": "",
                "qc_filter_mode": DEFAULT_QC_FILTER_MODE,
                "mask_qc_intensity_source": DEFAULT_MASK_QC_INTENSITY_SOURCE,
                "exclude_from_csv": False,
            }
        ],
    }


# Build each channel separately so nested adjustment and processing collections
# are never shared between startup rows.
def default_image_definitions(count: int = STARTUP_IMAGE_COUNT) -> list[dict]:
    return [default_image_definition(i) for i in range(int(count))]
