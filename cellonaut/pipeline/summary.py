"""Run summary and runtime metadata builders.

Summaries make completed runs easier to audit by recording selected settings,
software versions, input/output paths, and configured external files alongside
the generated results.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys
from typing import Any

from cellonaut.dependency_versions import collect_key_package_versions
from cellonaut.config.defaults import (
    MASK_SOURCE_MODE_COMBINED,
    MASK_SOURCE_MODE_WEKA,
    MEASUREMENT_LABELS,
)
from cellonaut.version import __version__
from cellonaut.pipeline.models import config_for_measurement_target
from cellonaut.system.gpu_detection import cellpose_acceleration_enabled


def _gpu_selection(requested: bool) -> dict[str, bool]:
    return {
        "gpu_requested": requested,
        "gpu_passed_to_cellpose": cellpose_acceleration_enabled(requested),
    }

# Describe each source in one compact form so logs and saved manifests use the
# same terminology for Weka and combined masks.
def describe_mask_source_for_summary(image: dict[str, Any]) -> str:
    mode = str(image.get("mask_source_mode", "") or "").strip()
    model_path = str(image.get("model_path", "") or "").strip()
    combined_sources = [str(value) for value in image.get("combined_mask_source_keys", []) or []]

    if mode == MASK_SOURCE_MODE_WEKA:
        return f"Weka classifier ({model_path or 'not configured'})"
    if mode == MASK_SOURCE_MODE_COMBINED:
        operation = str(image.get("combined_mask_operation", "OR") or "OR")
        return f"Combined masks ({operation}; sources={combined_sources or []})"
    if model_path:
        return f"Weka classifier ({model_path})"
    return mode or "not configured"


# Execution logs group selected and unselected measurements
# while excluding internal switches that are not direct user choices.
def selected_measurements_summary(cfg: Any) -> str:
    if not cfg.measurement_options:
        return "(none set; using defaults)"
    enabled = [
        str(MEASUREMENT_LABELS[key])
        for key, value in cfg.measurement_options.items()
        if key in MEASUREMENT_LABELS and value
    ]
    disabled = [
        str(MEASUREMENT_LABELS[key])
        for key, value in cfg.measurement_options.items()
        if key in MEASUREMENT_LABELS and not value
    ]
    enabled_text = ", ".join(enabled) if enabled else "(none)"
    disabled_text = ", ".join(disabled) if disabled else "(none)"
    return f"enabled={enabled_text} | disabled={disabled_text}"


# Normalize dataclasses and filesystem values before JSON encoding, and sort
# sets because otherwise identical runs can produce different manifest order.
def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=lambda item: repr(item))]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


# Record UTC explicitly so manifests from different systems remain comparable.
def _path_mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


# Hash configured models in chunks to avoid loading large Cellpose or Weka files
# into memory merely to make a run auditable.
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Treat fingerprints as diagnostics rather than a reason to abort a run when an
# external model disappears or becomes inaccessible during summary creation.
def _file_fingerprint(path_value: Any, role: str) -> dict[str, Any]:
    path = Path(str(path_value or ""))
    entry: dict[str, Any] = {"role": role, "path": str(path), "exists": False}
    try:
        entry["exists"] = path.exists()
    except OSError as exc:
        entry["inspection_error"] = f"{type(exc).__name__}: {exc}"
        return entry
    if not entry["exists"]:
        return entry

    try:
        entry["is_file"] = path.is_file()
        entry["is_dir"] = path.is_dir()
        entry["modified_utc"] = _path_mtime_utc(path)
        if entry["is_file"]:
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = _file_sha256(path)
    except OSError as exc:
        entry["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return entry


# Retain one entry per configuration role but reuse path metadata so a classifier
# shared by several masks is not hashed repeatedly.
def _configured_file_fingerprints(cfg: Any) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    path_cache: dict[str, dict[str, Any]] = {}
    fingerprints: list[dict[str, Any]] = []

    # Roles remain separate for traceability even when they point to one file.
    def add(path_value: Any, role: str) -> None:
        text = str(path_value or "").strip()
        if not text:
            return
        key = (role, text)
        if key in seen:
            return
        seen.add(key)
        if text not in path_cache:
            path_cache[text] = _file_fingerprint(text, "")
        entry = dict(path_cache[text])
        entry["role"] = role
        fingerprints.append(entry)

    for image in getattr(cfg, "images", []) or []:
        add(getattr(image, "model_path", None), f"classifier:{getattr(image, 'label', '')}")

    add(getattr(cfg, "cellpose_custom_model_path", ""), "cellpose_custom_model:default")
    for target in getattr(cfg, "measurement_targets", []) or []:
        mask_key = (
            getattr(target, "cell_segmentation_mask_source", "")
            or getattr(target, "source_image_key", "")
            if bool(getattr(target, "do_cell_segmentation", False))
            else ""
        )
        add(
            getattr(target, "cellpose_custom_model_path", ""),
            "cellpose_custom_model:"
            f"{getattr(target, 'source_image_key', '')}:"
            f"{mask_key}",
        )

    return fingerprints


# Record effective default and per-target Cellpose settings separately because
# targets can override the global model without changing the visible default row.
def _cellpose_model_settings(cfg: Any) -> list[dict[str, Any]]:
    settings = [
        {
            "scope": "default",
            "model_type": getattr(cfg, "cellpose_model_type", ""),
            "custom_model_path": getattr(cfg, "cellpose_custom_model_path", ""),
            "diameter": getattr(cfg, "cell_diameter", None),
            "minimum_cell_area": getattr(cfg, "cell_min_size", None),
            "cellprob_threshold": getattr(cfg, "cellprob_threshold", None),
            "flow_threshold": getattr(cfg, "flow_threshold", None),
            **_gpu_selection(bool(getattr(cfg, "cell_use_gpu", False))),
        }
    ]
    for target in getattr(cfg, "measurement_targets", []) or []:
        if not bool(getattr(target, "do_cell_segmentation", False)):
            continue
        effective = config_for_measurement_target(cfg, target)
        settings.append(
            {
                "scope": (
                    f"target:{getattr(target, 'source_image_key', '')}:"
                    f"{effective.cell_segmentation_mask_source}"
                ),
                "model_type": effective.cellpose_model_type,
                "custom_model_path": effective.cellpose_custom_model_path,
                "diameter": effective.cell_diameter,
                "minimum_cell_area": effective.cell_min_size,
                "cellprob_threshold": effective.cellprob_threshold,
                "flow_threshold": effective.flow_threshold,
                **_gpu_selection(bool(effective.cell_use_gpu)),
            }
        )
    return settings


# The raw target remains in configuration_snapshot; this compact form records
# the values that were actually handed to processing after inheritance.
def _effective_measurement_target_settings(cfg: Any, target: Any) -> dict[str, Any]:
    effective = config_for_measurement_target(cfg, target)
    return {
        "source_image_key": target.source_image_key,
        "enabled": bool(target.enabled),
        "overlay_base_image_key": effective.overlay_base_image_key,
        "overlay_roi_keys": list(effective.overlay_roi_keys),
        "overlay_whole_cell_mask": bool(effective.overlay_whole_cell_mask),
        "do_cell_segmentation": bool(effective.do_cell_segmentation),
        "cell_segmentation_source": effective.cell_segmentation_source,
        "cell_segmentation_mask_source": effective.cell_segmentation_mask_source,
        "output_variant": effective.output_variant,
        "per_cell_mask_source": effective.per_cell_mask_source,
        "measurement_options": dict(effective.measurement_options or {}),
        "cell_diameter": effective.cell_diameter,
        "cell_min_size": effective.cell_min_size,
        "cell_gpu_requested": bool(effective.cell_use_gpu),
        "cell_gpu_passed_to_cellpose": cellpose_acceleration_enabled(bool(effective.cell_use_gpu)),
        "cellprob_threshold": effective.cellprob_threshold,
        "flow_threshold": effective.flow_threshold,
        "cell_remove_border": effective.cell_remove_border,
        "cellpose_model_type": effective.cellpose_model_type,
        "cellpose_custom_model_path": effective.cellpose_custom_model_path,
        "cell_mask_adjustments": dict(effective.cell_mask_adjustments or {}),
    }


# Capture runtime facts at execution time because source checkouts and packaged
# applications may use different Python, Fiji, and compute environments.
def build_runtime_metadata(cfg: Any) -> dict[str, Any]:
    fiji_path_text = str(getattr(cfg, "fiji_app_path", "") or "").strip()
    fiji_path = Path(fiji_path_text) if fiji_path_text else None
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "dependency_versions": collect_key_package_versions(),
        "fiji": {
            "app_path": fiji_path_text,
            "path_exists": bool(fiji_path and fiji_path.exists()),
            "imagej_version": "(selected Fiji runtime; not queried during manifest write)",
        },
        "cellpose_model_settings": _cellpose_model_settings(cfg),
        "configured_file_fingerprints": _configured_file_fingerprints(cfg),
    }


# Store structured run settings alongside the readable summary.
def build_pipeline_summary_dict(cfg: Any) -> dict[str, Any]:
    configuration_snapshot = _json_safe(cfg)
    if isinstance(configuration_snapshot, dict):
        configuration_snapshot.pop("exclusion_tag", None)
    summary = {
        "app_version": __version__,
        "runtime_environment": build_runtime_metadata(cfg),
        "configuration_snapshot": configuration_snapshot,
        "input_directory": str(cfg.input_dir),
        "output_directory": str(cfg.output_dir),
        "fiji_app_path": str(cfg.fiji_app_path),
        "input_structure": cfg.input_structure,
        "threshold_method": cfg.threshold_method,
        "probability_class_index": cfg.probability_class_index,
        "reuse_existing_masks": bool(getattr(cfg, "reuse_existing_masks", False)),
        "mask_source_dir": str(getattr(cfg, "mask_source_dir", "") or ""),
        "measurement_options_summary": selected_measurements_summary(cfg),
        "measurement_options": dict(cfg.measurement_options or {}),
        "cell_segmentation_defaults": {
            "diameter": cfg.cell_diameter,
            "min_size": cfg.cell_min_size,
            **_gpu_selection(bool(cfg.cell_use_gpu)),
            "cellprob_threshold": cfg.cellprob_threshold,
            "flow_threshold": cfg.flow_threshold,
            "remove_border": bool(cfg.cell_remove_border),
            "model_type": cfg.cellpose_model_type,
            "custom_model_path": cfg.cellpose_custom_model_path,
            "mask_adjustments": dict(getattr(cfg, "cell_mask_adjustments", {}) or {}),
        },
        "measurement_targets": [
            _effective_measurement_target_settings(cfg, target)
            for target in (cfg.measurement_targets or [])
        ],
        "images": [
            {
                "key": image.key,
                "label": image.label,
                "folder_name": image.folder_name,
                "model_path": str(image.model_path) if image.model_path is not None else "",
                "mask_source_mode": image.mask_source_mode,
                "combined_mask_source_keys": list(image.combined_mask_source_keys),
                "combined_mask_operation": image.combined_mask_operation,
                "display_color": image.display_color,
                "stack_channel_index": getattr(image, "stack_channel_index", None),
                "stack_z_mode": getattr(image, "stack_z_mode", "max_projection"),
                "stack_z_index": getattr(image, "stack_z_index", None),
                "bg_radii_csv": image.bg_radii_csv,
                "image_processing_steps": [dict(step) for step in getattr(image, "image_processing_steps", []) or []],
                "mask_processing_steps": [dict(step) for step in getattr(image, "mask_processing_steps", []) or []],
                "probability_class_index": image.probability_class_index,
                "threshold_method": image.threshold_method,
            }
            for image in cfg.images
        ],
    }
    return summary
