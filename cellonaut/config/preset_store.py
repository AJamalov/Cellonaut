"""Qt-free helpers for naming, comparing, and storing preset JSON files."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from cellonaut.io.writers import write_text

PRESET_EXCLUDED_SETTING_KEYS = frozenset({"fiji_app_path", "nd2_output_dir"})
DEFAULT_PRESET_NAME = "Default"
PRESET_PRIVATE_PATH_KEYS = frozenset(
    {
        "input_dir",
        "output_dir",
        "mask_source_dir",
        "classifier",
        "cellpose_custom_model_path",
    }
)


def private_path_tail(value: object) -> str:
    """Retain only a path's final file or folder name on either path syntax."""
    text = str(value or "").strip()
    if not text:
        return ""
    path = PureWindowsPath(text) if "\\" in text or (len(text) > 1 and text[1] == ":") else PurePosixPath(text)
    return path.name


def privacy_safe_preset(data: dict) -> dict:
    """Remove parent directories from known path fields; preserve other text."""
    cleaned = deepcopy(data)

    def clean_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: private_path_tail(item) if key in PRESET_PRIVATE_PATH_KEYS else clean_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [clean_value(item) for item in value]
        return value

    cleaned = clean_value(cleaned)
    if not isinstance(cleaned, dict):
        raise TypeError("Preset data must be a dictionary.")
    cleaned["_preset_privacy"] = "Parent folders were removed from known path fields by Cellonaut preset export. Filenames, panel notes, and other text remain. Reselect local paths."
    return cleaned


def sanitize_preset_name(name: str) -> str:
    """Return a cross-platform filename-safe preset name."""
    name = (name or "").strip()
    if not name:
        return ""
    return "".join(character if character.isalnum() or character in (" ", "_", "-") else "_" for character in name).strip()


def is_default_preset(name: object) -> bool:
    """Return whether a name refers to the protected recovery preset."""
    return str(name or "").strip().casefold() == DEFAULT_PRESET_NAME.casefold()


def preset_path(presets_dir: Path, name: str) -> Path:
    """Resolve a preset name inside the supplied preset directory."""
    return presets_dir / f"{sanitize_preset_name(name)}.json"


def preset_state_token(data: dict) -> str:
    """Build a stable comparison token for the complete workflow."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_object(
    path: Path,
    data: dict,
    *,
    writer: Callable[..., None] = write_text,
) -> None:
    """Write a human-readable UTF-8 JSON object atomically."""
    writer(path, json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json_object(path: Path) -> dict:
    """Read JSON and reject top-level values that are not objects."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return data
