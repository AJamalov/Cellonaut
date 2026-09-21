"""Validate optional model runtimes before Cellpose starts expensive work."""

from __future__ import annotations

import os
from pathlib import Path

from cellonaut.config.defaults import CELLPOSE_MODEL_OPTIONS
from cellonaut.resources import SOURCE_ROOT


def configure_local_cellpose_models() -> Path | None:
    """Point source runs at prepared offline weights before Cellpose is imported."""
    configured = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", "").strip()
    if configured:
        return Path(configured)

    candidates = (
        SOURCE_ROOT / ".local" / "release_assets" / "offline" / "cellpose_models",
        Path.cwd() / ".local" / "release_assets" / "offline" / "cellpose_models",
    )
    for candidate in candidates:
        if candidate.is_dir():
            resolved = candidate.resolve()
            os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(resolved)
            return resolved
    return None


def cellpose_model_runtime_error(model_type: str) -> str | None:
    """Return a repair message when the selected model runtime is unavailable."""
    model_type = str(model_type or "").strip()
    model_root = configure_local_cellpose_models()
    if model_root is not None and model_type in CELLPOSE_MODEL_OPTIONS:
        model_path = model_root / model_type
        if not model_path.is_file():
            if os.environ.get("CELLONAUT_OFFLINE", "").strip() == "1":
                repair = "Reinstall Cellonaut from the complete official offline package."
            else:
                repair = "Prepare the complete offline release assets again."
            return (
                f"The local Cellpose model '{model_type}' is missing from {model_root}. "
                f"{repair}"
            )
    return None


def require_cellpose_model_runtime(model_type: str) -> None:
    """Raise a clear error before model construction when a runtime is missing."""
    error = cellpose_model_runtime_error(model_type)
    if error is not None:
        raise RuntimeError(error)
