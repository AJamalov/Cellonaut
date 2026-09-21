"""Read-only sample readiness analysis shared by setup checks and reports."""

from __future__ import annotations

from typing import Any

from cellonaut.io.image_io import find_channel_files
from cellonaut.pipeline.discovery import (
    build_result_id,
    build_sample_label,
    get_measurement_sample_paths,
    validate_unique_result_ids,
)
from cellonaut.pipeline.models import Config
from cellonaut.pipeline.planning import (
    get_enabled_measurement_targets,
    get_required_roi_defs_for_targets,
    image_keys_needed_for_processing,
    reusable_mask_warnings,
)


def analyze_sample_readiness(cfg: Config) -> dict[str, Any]:
    """Report input availability using the same channel/mask planning as execution.

    Return sample rows, ready/blocked counts and a problems list containing both
    warnings and blocked samples. This checks discovery and reusable-mask
    availability, not inference success; it does not run segmentation or write
    results. Discovery and result-ID collision errors propagate to the caller.
    """
    samples = get_measurement_sample_paths(cfg)
    validate_unique_result_ids(samples, cfg.input_structure, input_dir=cfg.input_dir)
    rows: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    ready = 0
    blocked = 0

    enabled_targets = get_enabled_measurement_targets(cfg)
    required_roi_defs = get_required_roi_defs_for_targets(cfg, enabled_targets)

    for sample_index, sample_folder in enumerate(samples, start=1):
        sample_label = build_sample_label(sample_folder, cfg.input_structure)
        result_id = build_result_id(sample_folder, cfg.input_structure, input_dir=cfg.input_dir)
        required_image_keys = image_keys_needed_for_processing(
            cfg,
            enabled_targets,
            required_roi_defs,
            result_id,
        )
        messages: list[str] = []
        ambiguous: list[str] = []
        file_map = find_channel_files(
            sample_folder,
            cfg,
            sample_label,
            messages.append,
            required_image_keys=required_image_keys,
            ambiguity_func=ambiguous.append,
        )

        if file_map is None:
            blocked += 1
            entry = {
                "sample": sample_label,
                "path": str(sample_folder),
                "ready": False,
                "status": "BLOCKED",
                "missing_required": messages or ["Could not resolve required input files."],
                "missing_optional": [],
                "ambiguous": [],
                "messages": messages,
                "files": {},
                "index": sample_index,
            }
            rows.append(entry)
            problems.append(entry)
            continue

        missing_optional = [
            image.label
            for image in cfg.images
            if file_map.get(image.key) is None
            and image.key not in required_image_keys
            and not getattr(image, "combined_mask_source_keys", None)
        ]
        reuse_warnings = reusable_mask_warnings(
            cfg,
            enabled_targets,
            result_id,
            file_map=file_map,
        )
        messages.extend(reuse_warnings)
        resolved_files = {
            image.label: str(file_map[image.key]) for image in cfg.images if file_map.get(image.key) is not None
        }

        ready += 1
        entry = {
            "sample": sample_label,
            "path": str(sample_folder),
            "ready": True,
            "status": "READY_WITH_WARNINGS" if missing_optional or ambiguous or reuse_warnings else "READY",
            "missing_required": [],
            "missing_optional": missing_optional,
            "ambiguous": ambiguous,
            "reuse_mask_warnings": reuse_warnings,
            "messages": messages,
            "files": resolved_files,
            "index": sample_index,
        }
        rows.append(entry)
        if missing_optional or ambiguous or reuse_warnings:
            problems.append(entry)

    return {
        "available": True,
        "root": str(cfg.input_dir),
        "structure": cfg.input_structure,
        "expected_folders": [
            image.folder_name for image in cfg.images if not getattr(image, "combined_mask_source_keys", None)
        ],
        "total": len(samples),
        "ready": ready,
        "incomplete": blocked,
        "blocked": blocked,
        "warning_count": len(problems),
        "samples": rows,
        "problems": problems,
        "error": "",
    }
