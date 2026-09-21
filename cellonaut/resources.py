"""Resolve bundled read-only assets and platform-appropriate writable data paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cellonaut.version import APP_NAME

SOURCE_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DATA_ROOT = Path(__file__).resolve().parent / "data"


# Resource roots differ between source runs and PyInstaller bundles. Keeping
# the search order here lets the rest of the app use the same relative paths.
def resource_base_dirs() -> list[Path]:
    """Return packaged and source resource roots in lookup priority order."""
    bases: list[Path] = []

    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        bases.append(Path(frozen_base))

    if getattr(sys, "frozen", False):
        bases.append(Path(sys.executable).resolve().parent)

    bases.extend([SOURCE_ROOT, Path.cwd(), PACKAGE_DATA_ROOT, Path(__file__).resolve().parent])

    unique: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        key = str(base.resolve()) if base.exists() else str(base)
        if key not in seen:
            unique.append(base)
            seen.add(key)
    return unique


def platform_user_config_dir() -> Path:
    """Return the Windows per-user Cellonaut configuration folder."""
    return Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming")) / APP_NAME


APP_BASE_DIR = resource_base_dirs()[0]

# Per-user writable data directory: use this for presets/settings/logs.
USER_APP_DIR = platform_user_config_dir()

PRESETS_DIR = USER_APP_DIR / "presets"

SETTINGS_FILE = USER_APP_DIR / "last_settings.json"


# Directory creation is deferred until startup needs it, which keeps imports
# read-only and makes command-line and test use less intrusive.
def ensure_user_app_dirs() -> None:
    """Create writable application folders when startup first needs them."""
    USER_APP_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


# Read bundled assets from install/app folder, not from user settings.
APP_ICON_FILE = Path("assets") / "icon.ico"


# Resolve source and packaged Windows layouts in one place.
def get_resource_path(path_value: str | Path) -> Path:
    """Resolve a bundled resource from the active source or packaged layout."""
    path_value = Path(path_value)
    if path_value.is_absolute() and path_value.exists():
        return path_value

    for base_dir in resource_base_dirs():
        candidate = base_dir / path_value
        if candidate.exists():
            return candidate

    if path_value.is_absolute():
        return path_value

    return APP_BASE_DIR / path_value


# Packaged releases place Fiji beside the frozen Python resources. Source runs
# use the prepared release asset only through get_internal_fiji_path below.
def get_bundled_fiji_path() -> Path | None:
    """Return Fiji embedded in a packaged application, if present."""
    for base_dir in resource_base_dirs():
        candidate = base_dir / "offline" / "Fiji.app"
        if candidate.is_dir():
            return candidate
    return None


def get_internal_fiji_path() -> Path | None:
    """Resolve Fiji without exposing its location as an end-user setting."""
    bundled = get_bundled_fiji_path()
    if bundled is not None:
        return bundled

    configured = str(os.environ.get("CELLONAUT_FIJI_APP_PATH", "") or "").strip()
    if configured and Path(configured).is_dir():
        return Path(configured)

    source_asset = SOURCE_ROOT / ".local" / "release_assets" / "offline" / "Fiji.app"
    return source_asset if source_asset.is_dir() else None
