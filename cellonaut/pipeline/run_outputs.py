"""Reserve numbered output folders without overwriting earlier analyses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cellonaut.pipeline.models import Config
from cellonaut.pipeline.operation_state import save_operation_state


# Return the first gap so deleted previews or runs can reuse their number without overwriting anything.
def next_numbered_child(parent: Path, prefix: str) -> Path:
    parent = Path(parent)
    index = 1

    while True:
        candidate = parent / f"{prefix}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


# Keep the original mask location when assigning a new output folder.
def config_for_numbered_child_output(cfg: Config, prefix: str) -> Config:
    mask_source_dir = cfg.mask_source_dir
    if getattr(cfg, "reuse_existing_masks", False) and mask_source_dir is None:
        mask_source_dir = cfg.output_dir
    output_dir = next_numbered_child(cfg.output_dir, prefix)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        save_operation_state(
            output_dir,
            operation_type="full" if prefix == "run" else prefix,
            state="in_progress",
            reset_started=True,
        )
    except Exception:
        # The state writer may have created these known directories before it
        # failed. Remove only empty paths so unrelated files are never touched.
        for path in (
            output_dir / "Results" / "Logs",
            output_dir / "Results",
            output_dir,
        ):
            try:
                path.rmdir()
            except OSError:
                pass
        raise
    return replace(cfg, output_dir=output_dir, mask_source_dir=mask_source_dir)
