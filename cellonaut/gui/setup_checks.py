"""Qt-free setup-check construction and report formatting.

The GUI collects current settings and cached background scans, then delegates
to this module.  Keeping the decision logic here makes setup reports usable in
tests and other front ends without constructing widgets.
"""

from __future__ import annotations

from cellonaut.gui.help_content import IMAGESCIENCE_HELP

from pathlib import Path
from typing import Any, Sequence

from cellonaut.config.relationships import image_uses_non_classifier_mask
from cellonaut.io.fiji_installation import scan_fiji_installation
from cellonaut.io.nd2_import import list_nd2_files
from cellonaut.io.path_keys import input_path_key
from cellonaut.system.gpu_detection import describe_cellpose_backend


def count_label(count: int, singular: str, plural: str | None = None) -> str:
    """Return a count with the appropriate singular or plural label."""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"


def compact_repeated_warnings(warnings: Sequence[str]) -> list[str]:
    """Combine warnings that share the same instruction."""
    grouped: dict[str, list[str]] = {}
    passthrough: list[str] = []
    ordered_details: list[str] = []

    for warning in warnings:
        text = str(warning or "").strip()
        if not text:
            continue
        if ": " not in text:
            passthrough.append(text)
            continue
        subject, detail = (part.strip() for part in text.split(": ", 1))
        if not subject or not detail:
            passthrough.append(text)
            continue
        if detail not in grouped:
            grouped[detail] = []
            ordered_details.append(detail)
        if subject not in grouped[detail]:
            grouped[detail].append(subject)

    compacted = list(passthrough)
    for detail in ordered_details:
        compacted.append(f"{', '.join(grouped[detail])}: {detail}")
    return compacted


def _setup_row(status: str, check: str, detail: str) -> dict[str, str]:
    return {"status": status, "check": check, "detail": detail}


def _folder_and_backend_check_items(cfg: Any) -> list[dict[str, str]]:
    """Check required folders and the selected Cellpose runtime."""
    checks: list[dict[str, str]] = []
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    if input_dir.exists() and input_dir.is_dir():
        checks.append(_setup_row("OK", "Input folder", str(input_dir)))
    else:
        checks.append(
            _setup_row(
                "BLOCKED",
                "Input folder",
                f"Folder does not exist: {input_dir}. Choose a reachable TIFF folder. For raw ND2 files, use Import ND2 first.",
            )
        )

    if output_dir.exists() and not output_dir.is_dir():
        checks.append(
            _setup_row(
                "BLOCKED",
                "Output folder",
                f"Path exists but is not a folder: {output_dir}. Choose a folder path for Cellonaut results.",
            )
        )
    elif output_dir.exists():
        checks.append(_setup_row("OK", "Output folder", str(output_dir)))
    elif output_dir.parent.exists():
        checks.append(_setup_row("OK", "Output folder", f"Will be created: {output_dir}"))
    else:
        checks.append(
            _setup_row(
                "BLOCKED",
                "Output folder",
                f"Parent folder does not exist: {output_dir.parent}. Choose an existing parent folder.",
            )
        )

    backend = describe_cellpose_backend()
    checks.append(_setup_row(backend.status, backend.label, backend.detail))
    return checks


def _fiji_check_items(fiji_path: Path, cached_fiji_scan: dict[str, Any] | None) -> list[dict[str, str]]:
    """Build Fiji installation and component checks, reusing a matching cached scan."""
    fiji_scan = cached_fiji_scan or {}
    cached_fiji_path = str(fiji_scan.get("path", "") or "").strip()
    fiji_status = fiji_scan.get("status") if cached_fiji_path == str(fiji_path) else None
    if fiji_status is None:
        fiji_status = scan_fiji_installation(fiji_path)
    if not fiji_status.path_exists:
        return [
            _setup_row(
                "BLOCKED",
                "Bundled Fiji runtime",
                f"Bundled runtime folder does not exist: {fiji_path}. Reinstall Cellonaut from the complete official offline package.",
            )
        ]
    if not fiji_status.path_is_dir:
        return [
            _setup_row(
                "BLOCKED",
                "Bundled Fiji runtime",
                f"Bundled runtime path is not a folder: {fiji_path}. Reinstall Cellonaut from the complete official offline package.",
            )
        ]
    if not fiji_status.looks_like_fiji:
        return [
            _setup_row("BLOCKED", "Bundled Fiji runtime", f"Folder exists but does not look like Fiji: {fiji_path}.")
        ]
    return [
        _setup_row("OK", "Bundled Fiji runtime", str(fiji_path)),
        *[
            _setup_row(
                "OK" if component.ok else "BLOCKED" if component.required else "WARNING",
                f"Fiji {component.label}",
                component.detail if component.ok or component.required else IMAGESCIENCE_HELP,
            )
            for component in fiji_status.components
        ],
    ]


def _nd2_check_items(
    input_dir: Path,
    cached_input_scan: dict[str, Any] | None,
    detected_channel_names: Sequence[str],
) -> list[dict[str, str]]:
    """Describe bounded ND2 discovery, preferring the matching background result."""
    input_scan = cached_input_scan or {}
    cached_input_matches = input_path_key(str(input_scan.get("path", "") or "")) == input_path_key(input_dir)
    nd2_count = 0
    nd2_first: Path | str = ""
    truncated = False
    if cached_input_matches:
        nd2_count = int(input_scan.get("nd2_count", 0) or 0)
        nd2_first = str(input_scan.get("nd2_first", "") or "")
        truncated = bool(input_scan.get("nd2_scan_truncated", False))
        if input_scan.get("nd2_error"):
            return [_setup_row("WARNING", "ND2 files", f"Could not inspect ND2 files: {input_scan['nd2_error']}")]
    else:
        try:
            nd2_files = list_nd2_files(input_dir) if input_dir.exists() and input_dir.is_dir() else []
        except Exception as exc:
            return [_setup_row("WARNING", "ND2 files", f"Could not scan for ND2 files: {exc}")]
        nd2_count = len(nd2_files)
        nd2_first = nd2_files[0] if nd2_files else ""

    if not nd2_count:
        return [_setup_row("OK", "ND2 files", "No ND2 files detected in the input folder or subfolders.")]
    try:
        first_nd2 = Path(nd2_first).relative_to(input_dir)
    except ValueError:
        first_nd2 = nd2_first
    count_text = count_label(nd2_count, "ND2 file")
    if truncated:
        count_text = f"at least {count_text}"
    channel_count = len(detected_channel_names)
    if channel_count:
        detail = (
            f"{count_text} detected, including {first_nd2}; {count_label(channel_count, 'channel')} detected. "
            "Raw ND2 cannot be analyzed directly. Use Import ND2, then select the converted TIFF folder as input."
        )
        return [_setup_row("WARNING", "ND2 files", detail)]
    return [
        _setup_row(
            "WARNING",
            "ND2 files",
            f"{count_text} detected, including {first_nd2}. Use Import ND2, then select the converted TIFF folder as input.",
        )
    ]


def _classifier_check_item(cfg: Any, active_defs: Sequence[dict[str, Any]]) -> dict[str, str]:
    missing = []
    for image in active_defs:
        if image_uses_non_classifier_mask(image):
            continue
        classifier = str(image.get("classifier", "") or "").strip()
        if classifier and not Path(classifier).exists():
            missing.append(f"{image.get('name', '(unnamed)')}: {classifier}")
    if not missing:
        return _setup_row("OK", "Classifier files", "All configured classifier paths are available.")
    if bool(getattr(cfg, "reuse_existing_masks", False)):
        detail = (
            "Missing classifier files are allowed while reusing masks. "
            "If a matching mask is absent, regeneration will fail: " + "; ".join(missing[:4])
        )
        return _setup_row("WARNING", "Classifier files", detail)
    detail = (
        "; ".join(missing[:4])
        + ". Choose the missing .model files or enable Reuse existing masks."
    )
    return _setup_row("BLOCKED", "Classifier files", detail)


def _measurement_check_items(
    cfg: Any,
    relationships: Sequence[dict[str, Any]],
    warnings: Sequence[str],
) -> list[dict[str, str]]:
    enabled_targets = [target for target in getattr(cfg, "measurement_targets", []) if target.enabled]
    if enabled_targets:
        relationship_detail = f" with {count_label(len(relationships), 'mask relationship')}" if relationships else ""
        measurement = _setup_row(
            "OK",
            "Measurements",
            f"{count_label(len(enabled_targets), 'measured channel')} enabled{relationship_detail}.",
        )
    else:
        measurement = _setup_row(
            "BLOCKED",
            "Measurements",
            "No measurements are selected. Turn ON a configured-mask or Cellpose-mask intersection in Measurements.",
        )
    warning_check = _setup_row(
        "WARNING" if warnings else "OK",
        "Measurement setup",
        f"{count_label(len(warnings), 'warning')} detected." if warnings else "No warnings found.",
    )
    return [measurement, warning_check]


def _sample_detection_check_item(dry_run: dict[str, Any]) -> dict[str, str]:
    if not dry_run.get("available", False):
        return _setup_row("WARNING", "Sample detection", dry_run.get("error", "Sample detection unavailable."))
    total = int(dry_run.get("total", 0) or 0)
    ready = int(dry_run.get("ready", 0) or 0)
    blocked = int(dry_run.get("blocked", dry_run.get("incomplete", 0)) or 0)
    if total <= 0:
        return _setup_row("BLOCKED", "Sample detection", "No TIFF samples were detected in the input folder.")
    if ready <= 0:
        return _setup_row(
            "BLOCKED", "Sample detection", f"{count_label(total, 'sample')} detected, but none are runnable."
        )
    sample_noun = "sample" if total == 1 else "samples"
    if blocked > 0:
        return _setup_row("WARNING", "Sample detection", f"{ready}/{total} {sample_noun} ready; {blocked} blocked.")
    return _setup_row("OK", "Sample detection", f"{ready}/{total} {sample_noun} ready.")


def build_setup_check_items(
    cfg: Any,
    active_defs: Sequence[dict[str, Any]],
    relationships: Sequence[dict[str, Any]],
    warnings: Sequence[str],
    dry_run: dict[str, Any],
    *,
    cached_fiji_scan: dict[str, Any] | None = None,
    cached_input_scan: dict[str, Any] | None = None,
    nd2_detected_channel_names: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build the common setup-check rows from configuration and scan state."""
    checks: list[dict[str, str]] = []

    input_dir = Path(cfg.input_dir)
    fiji_path = Path(cfg.fiji_app_path)
    checks.extend(_folder_and_backend_check_items(cfg))
    checks.extend(_fiji_check_items(fiji_path, cached_fiji_scan))

    if active_defs:
        checks.append(
            _setup_row(
                "OK",
                "Configured channels",
                f"{count_label(len(active_defs), 'channel or mask definition')} configured.",
            )
        )
    else:
        checks.append(_setup_row("BLOCKED", "Configured channels", "No channels or masks are configured."))

    checks.extend(_nd2_check_items(input_dir, cached_input_scan, nd2_detected_channel_names))
    checks.append(_classifier_check_item(cfg, active_defs))
    checks.extend(_measurement_check_items(cfg, relationships, warnings))
    checks.append(_sample_detection_check_item(dry_run))

    return checks


def format_configuration_summary_text(summary: dict[str, Any]) -> str:
    """Format setup summary data as a portable plain-text report."""
    lines = ["Cellonaut Setup Check", "=" * 72, "", "Setup checks", "-" * 72]
    for item in summary.get("setup_checks", []):
        lines.append(f"- [{item['status']}] {item['check']}: {item['detail']}")

    folders = summary.get("folders", {})
    lines.extend(
        [
            "",
            "Folders",
            "-" * 72,
            f"Input folder: {folders.get('input', '')}",
            f"Output folder: {folders.get('output', '')}",
            f"Bundled Fiji runtime: {folders.get('fiji', '')}",
            f"Input layout: {folders.get('layout', '')}",
            "",
            "Channels and masks",
            "-" * 72,
        ]
    )
    for image in summary.get("images", []):
        item_type = str(image.get("type", "Channel or mask") or "Channel or mask")
        lines.append(
            f"- {item_type} {image.get('name', '')}: source={image.get('folder', '')}, "
            f"classifier={image.get('classifier') or '(none)'}, classes={image.get('probability_class', '')}"
        )

    lines.extend(["", "Measured channels and masks", "-" * 72])
    relationships = summary.get("relationships", [])
    if relationships:
        for relationship in relationships:
            mask = "its own mask" if relationship.get("self") else f"{relationship.get('target', '')} mask"
            lines.append(f"- {relationship.get('source', '')} measured with {mask}")
    else:
        lines.append("(none)")

    lines.extend(["", "Cellpose", "-" * 72])
    cellpose_rows = summary.get("cell_segmentation", []) or []
    for item in cellpose_rows:
        lines.append(
            f"- {item.get('source', '')}_Cellpose: source={item.get('seg_source', '')}, "
            f"diameter={item.get('diameter', '')}, minimum cell area={item.get('min_size', '')}, "
            f"probability threshold={item.get('cellprob', '')}, flow threshold={item.get('flow', '')}, "
            f"remove border={'Yes' if item.get('remove_border', True) else 'No'}"
        )
    if not cellpose_rows:
        lines.append("No reusable Cellpose masks are configured.")

    lines.extend(["", "Cell Groups", "-" * 72])
    group_rows = summary.get("filters", []) or []
    for item in group_rows:
        lines.append(
            f"- {item.get('source', '')}: cell conditions={item.get('cell_limits', '') or '(none)'}, "
            f"target-mask conditions={item.get('mask_limits', '') or '(none)'}, "
            f"match rule={item.get('mode', '')}, intensity source={item.get('mask_intensity_source', '')}, "
            f"exclude from CSV={'Yes' if item.get('exclude', False) else 'No'}"
        )
    if not group_rows:
        lines.append("No cell-group conditions configured.")

    lines.extend(["", "Measurement settings", "-" * 72])
    measurement_sets = summary.get("measurements", [])
    for item in measurement_sets:
        enabled = list(item.get("enabled", []) or [])
        lines.append(f"- {item.get('source', '')}: {', '.join(enabled) if enabled else '(none)'}")
    if not measurement_sets:
        lines.append("(none)")

    dry_run = summary.get("dry_run", {})
    lines.extend(["", "Sample detection", "-" * 72])
    if dry_run.get("available", False):
        lines.append(
            f"Detected: {dry_run.get('total', 0)} | Ready: {dry_run.get('ready', 0)} | "
            f"Blocked: {dry_run.get('blocked', dry_run.get('incomplete', 0))}"
        )
    else:
        lines.append(f"Unavailable: {dry_run.get('error', 'unknown error')}")

    warnings = summary.get("warnings", [])
    lines.extend(["", "Warnings", "-" * 72])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("(none)")
    return "\n".join(lines)
