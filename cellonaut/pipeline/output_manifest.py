"""Write the human-readable and JSON records for a completed run or preview."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from cellonaut.io.writers import write_text
from cellonaut.pipeline.summary import describe_mask_source_for_summary
from cellonaut.results.layout import build_results_layout
from cellonaut.results.artifacts import read_manifest
from cellonaut.version import __version__


# Use an offset-aware timestamp because run reports may be compared across
# workstations without sharing their local timezone.
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Return a readable source label, including the Weka class number when present;
# fall back to the key.
def _display_source_key(key: Any, image_labels: dict[str, str]) -> str:
    text = str(key or "")
    if text in image_labels:
        return image_labels[text]
    match = re.fullmatch(r"(image\d+)(?:__class|_class)(\d+)", text)
    if match:
        base_key, class_index = match.groups()
        base_label = image_labels.get(base_key, base_key)
        return f"{base_label} (class {class_index})"
    return text


def save_run_manifest(
    output_dir: Path,
    *,
    run_type: str,
    inference_devices: set[str] | None = None,
    pipeline_summary: dict[str, Any],
    run_stats: dict[str, Any] | None = None,
    sample_status_rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write JSON/text run summaries and return their paths under json/text keys.

    Preserve the artifact inventory already published by writers. Inference
    devices are observed runtime facts, not values inferred from requested GPU
    settings or log text. Each report is written atomically, not as a two-file
    transaction; corrupt existing JSON is an error rather than an empty inventory.
    """
    output_dir = Path(output_dir)
    layout = build_results_layout(output_dir, create_root=False)
    layout["root"].mkdir(parents=True, exist_ok=True)
    layout["logs"].mkdir(parents=True, exist_ok=True)
    devices = sorted(inference_devices or set())

    manifest = {
        "manifest_type": "Cellonaut Run Summary",
        "app_version": __version__,
        "created_utc": _now_iso(),
        "run_type": run_type,
        "runtime_environment": pipeline_summary.get("runtime_environment", {}),
        "configuration_snapshot": pipeline_summary.get("configuration_snapshot", {}),
        "input_directory": pipeline_summary.get("input_directory", ""),
        "output_directory": pipeline_summary.get("output_directory", str(output_dir)),
        "results_directory": str(layout["root"]),
        "fiji_app_path": pipeline_summary.get("fiji_app_path", ""),
        "input_structure": pipeline_summary.get("input_structure", ""),
        "threshold_method": pipeline_summary.get("threshold_method", ""),
        "weka_class_index": pipeline_summary.get("probability_class_index", 1),
        "reuse_existing_masks": pipeline_summary.get("reuse_existing_masks", False),
        "mask_source_dir": pipeline_summary.get("mask_source_dir", ""),
        "measurement_options": pipeline_summary.get("measurement_options", {}),
        "measurement_options_summary": pipeline_summary.get("measurement_options_summary", ""),
        "cell_segmentation_defaults": pipeline_summary.get("cell_segmentation_defaults", {}),
        "cellpose_inference_devices": devices,
        "images": pipeline_summary.get("images", []),
        "measurement_targets": pipeline_summary.get("measurement_targets", []),
        "run_stats": run_stats or {},
        "sample_status": sample_status_rows or [],
    }

    existing = read_manifest(layout["root"])
    if "artifacts" in existing:
        manifest["artifacts"] = existing["artifacts"]

    write_text(
        layout["run_manifest_json"],
        json.dumps(manifest, indent=2),
    )

    lines = [
        "Cellonaut Run Summary",
        "=" * 60,
        f"Cellonaut version: {manifest['app_version']}",
        f"Created UTC: {manifest['created_utc']}",
        f"Run type: {manifest['run_type']}",
        f"Input directory: {manifest['input_directory']}",
        f"Output directory: {manifest['output_directory']}",
        f"Results directory: {manifest['results_directory']}",
        f"Input structure: {manifest['input_structure']}",
        f"Fiji app path: {manifest['fiji_app_path']}",
        f"Reuse existing masks: {manifest['reuse_existing_masks']}",
        "Mask reuse folder: "
        f"{manifest['mask_source_dir'] or ('(output directory)' if manifest['reuse_existing_masks'] else '(not used)')}",
        f"Measurement options: {manifest['measurement_options_summary'] or '(none)'}",
        "",
        "Runtime environment",
        "-" * 60,
    ]
    runtime = manifest.get("runtime_environment", {}) or {}
    python_info = runtime.get("python", {}) or {}
    platform_info = runtime.get("platform", {}) or {}
    lines.extend(
        [
            f"Python executable: {python_info.get('executable', '')}",
            f"Python version: {python_info.get('version', '').splitlines()[0]}",
            "Platform: "
            f"{platform_info.get('system', '')} "
            f"{platform_info.get('release', '')} "
            f"{platform_info.get('machine', '')}".strip(),
            "",
            "Dependency versions",
            "-" * 60,
        ]
    )
    for package_name, version in (runtime.get("dependency_versions", {}) or {}).items():
        lines.append(f"{package_name}: {version}")

    lines.extend(
        [
            "",
            "Configured file fingerprints",
            "-" * 60,
        ]
    )
    fingerprints = runtime.get("configured_file_fingerprints", []) or []
    if fingerprints:
        for entry in fingerprints:
            parts = [
                f"{entry.get('role', '')}: {entry.get('path', '')}",
                f"exists={entry.get('exists', False)}",
            ]
            if entry.get("sha256"):
                parts.append(f"sha256={entry.get('sha256')}")
            if entry.get("size_bytes") is not None:
                parts.append(f"size_bytes={entry.get('size_bytes')}")
            if entry.get("modified_utc"):
                parts.append(f"modified_utc={entry.get('modified_utc')}")
            if entry.get("inspection_error"):
                parts.append(f"inspection_error={entry.get('inspection_error')}")
            lines.append(" | ".join(parts))
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "Cellpose model settings",
            "-" * 60,
        ]
    )
    for setting in runtime.get("cellpose_model_settings", []) or []:
        lines.append(
            f"- {setting.get('scope', '')}: model={setting.get('model_type') or '(default)'}, "
            f"custom_model={setting.get('custom_model_path') or '(none)'}, "
            f"diameter={setting.get('diameter')}, min_area={setting.get('minimum_cell_area')}, "
            f"cell_probability={setting.get('cellprob_threshold')}, "
            f"flow={setting.get('flow_threshold')}, "
            f"gpu_requested={setting.get('gpu_requested')}, "
            f"gpu_passed_to_cellpose={setting.get('gpu_passed_to_cellpose')}"
        )
    lines.append(
        "Cellpose inference device(s): "
        + (", ".join(inference_devices) if inference_devices else "(none recorded)")
    )

    lines.extend(
        [
            "",
            "Configured channels and masks",
            "-" * 60,
        ]
    )
    image_labels = {
        str(image.get("key", "")): str(image.get("label", "") or image.get("key", ""))
        for image in manifest["images"]
    }
    for image in manifest["images"]:
        lines.append(
            f"- {image.get('label', '')} | key={image.get('key', '')} | "
            f"folder={image.get('folder_name', '')} | "
            f"mask_source={describe_mask_source_for_summary(image)} | "
            f"classes={image.get('probability_class_index', '')} | "
            f"stack_channel={image.get('stack_channel_index')} | "
            f"z_mode={image.get('stack_z_mode', '')} | "
            f"image_steps={image.get('image_processing_steps', [])} | "
            f"mask_steps={image.get('mask_processing_steps', [])}"
        )
    if not manifest["images"]:
        lines.append("(none)")

    lines.extend(["", "Measurements", "-" * 60])
    for target in manifest["measurement_targets"]:
        source_key = str(target.get("source_image_key", ""))
        mask_keys = [str(key) for key in target.get("overlay_roi_keys", []) or []]
        mask_labels = [_display_source_key(key, image_labels) for key in mask_keys]
        lines.append(
            f"- {_display_source_key(source_key, image_labels)} ({source_key}) | "
            f"enabled={target.get('enabled', '')} | masks={mask_labels} | "
            f"cell_masks={target.get('do_cell_segmentation', False)} | "
            f"cell_mask={_display_source_key(target.get('cell_segmentation_mask_source', ''), image_labels) or '(none)'}, "
            f"cell_source={_display_source_key(target.get('cell_segmentation_source', ''), image_labels) or '(none)'}, "
            f"output_variant={target.get('output_variant', '') or '(none)'}"
        )
    if not manifest["measurement_targets"]:
        lines.append("(none)")

    lines.extend(["", "Run stats", "-" * 60])
    for key, value in manifest["run_stats"].items():
        lines.append(f"{str(key).replace('_', ' ').capitalize()}: {value}")
    if not manifest["run_stats"]:
        lines.append("(none)")

    lines.extend(["", "Sample status", "-" * 60])
    for status in manifest["sample_status"]:
        sample = status.get("sample_id", "") or "(unknown sample)"
        target_key = str(status.get("target", "") or "")
        target_label = _display_source_key(target_key, image_labels) or "(all targets)"
        cell_mask_key = str(status.get("cell_mask", "") or "")
        if cell_mask_key:
            target_label += f" / {_display_source_key(cell_mask_key, image_labels)}_Cellpose"
        reason = str(status.get("reason", "") or "").strip()
        line = f"- {sample} | {target_label} | {status.get('status', '')}"
        if reason:
            line += f" | {reason}"
        lines.append(line)
    if not manifest["sample_status"]:
        lines.append("(none)")

    write_text(layout["run_manifest_txt"], "\n".join(lines))
    return {
        "json": str(layout["run_manifest_json"]),
        "text": str(layout["run_manifest_txt"]),
    }
