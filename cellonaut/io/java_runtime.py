"""Windows Java runtime discovery used before starting Fiji/PyImageJ."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from cellonaut.system.subprocesses import hidden_window_kwargs


# Provide a discard-only output stream for GUI builds without a console.
class _NullWriter:
    # Frozen GUI builds may have no console streams, but libraries still expect a file-like writer.
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


# Replace missing PyInstaller console streams before Java libraries try to write diagnostics.
def ensure_stdio() -> None:
    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()


def _java_executable_names() -> tuple[str, ...]:
    return ("java.exe",)


def _jvm_library_names() -> tuple[str, ...]:
    return ("jvm.dll",)


def _java_executable(path: Path) -> Path | None:
    for name in _java_executable_names():
        candidate = path / "bin" / name
        if candidate.exists():
            return candidate
    return None


# Java home must contain both the launcher and embeddable JVM used by PyImageJ.
def is_valid_java_home(path: Path) -> bool:
    if _java_executable(path) is None:
        return False
    return (path / "bin" / "server" / "jvm.dll").exists() or (
        path / "jre" / "bin" / "server" / "jvm.dll"
    ).exists()


# Query the executable instead of folder names because vendors use inconsistent versioned directories.
def java_major_version(path: Path) -> int | None:
    java_exe = _java_executable(path)
    if java_exe is None:
        return None
    try:
        result = subprocess.run(
            [str(java_exe), "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            **hidden_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    first_line = (result.stderr or result.stdout).splitlines()
    if not first_line:
        return None
    version_text = first_line[0].split('"')
    if len(version_text) < 2:
        return None
    version = version_text[1]
    try:
        return int(version.split(".")[1] if version.startswith("1.") else version.split(".")[0])
    except (IndexError, ValueError):
        return None


# Unknown versions remain usable because some packaged Java launchers omit a conventional version line.
def is_supported_java_home(path: Path, minimum_version: int = 11) -> bool:
    if not is_valid_java_home(path):
        return False
    major = java_major_version(path)
    return major is None or major >= minimum_version


# Downloaded Java runtime files belong outside read-only application bundles and source directories.
def configure_java_caches() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Cellonaut"
    else:
        base = Path(__file__).resolve().parents[2] / ".local"

    cache_root = base / "java_runtime"
    cjdk_cache = cache_root / "cjdk"
    cjdk_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CJDK_CACHE_DIR", str(cjdk_cache.resolve()))
    return cache_root


# Derive JAVA_HOME from known JVM layouts before falling back to a broader parent search.
def _java_home_from_jvm_path(jvm_path: Path) -> Path | None:
    parts_lower = [part.lower() for part in jvm_path.parts]

    if len(parts_lower) >= 3 and parts_lower[-3:] == ["bin", "server", "jvm.dll"]:
        return jvm_path.parents[2]

    if len(parts_lower) >= 4 and parts_lower[-4:] == ["jre", "bin", "server", "jvm.dll"]:
        return jvm_path.parents[3]

    return None


# Fiji can nest its runtime several levels deep, so locate the JVM first and walk back to Java home.
def find_java_home_from_jvm_dll(root: Path) -> Path | None:
    jvm_names = set(_jvm_library_names())
    try:
        for jvm_path in root.rglob("*"):
            if jvm_path.name.lower() not in jvm_names:
                continue
            jvm_path = jvm_path.resolve()
            java_home = _java_home_from_jvm_path(jvm_path)
            if java_home is not None and _java_executable(java_home) is not None:
                return java_home

            for parent in jvm_path.parents:
                if _java_executable(parent) is not None:
                    return parent
    except OSError:
        return None
    return None


# Resolve paths before comparing them so aliases do not cause repeated filesystem scans.
def _deduplicate_existing_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve()) if path.exists() else str(path)
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _activate_java_home(path: Path) -> Path:
    """Expose a selected JDK through JAVA_HOME and PATH for Java launchers."""
    path = path.resolve()
    os.environ["JAVA_HOME"] = str(path)
    java_bin = str(path / "bin")
    current_path = os.environ.get("PATH", "")
    entries = [entry for entry in current_path.split(os.pathsep) if entry]
    if java_bin.casefold() not in {entry.casefold() for entry in entries}:
        os.environ["PATH"] = os.pathsep.join([java_bin, *entries])
    return path


# Prefer an explicit or Fiji-bundled runtime, then check packaged, environment, and system locations.
def configure_java_home(
    fiji_app_path: Path | None = None,
    log_func: Callable[[str], None] = print,
) -> Path | None:
    existing = os.environ.get("JAVA_HOME")
    if existing:
        existing_path = Path(existing)
        if is_supported_java_home(existing_path):
            return existing_path
        log_func(f"[WARN] Ignoring unsupported JAVA_HOME: {existing_path}")

    preferred_candidates: list[Path] = []
    if fiji_app_path is not None:
        fiji_app_path = Path(fiji_app_path)
        preferred_candidates.extend(
            [
                fiji_app_path,
                fiji_app_path.parent,
                fiji_app_path / "java",
                fiji_app_path / "jdk",
                fiji_app_path / "jre",
                fiji_app_path.parent / "java",
                fiji_app_path.parent / "jdk",
                fiji_app_path.parent / "jre",
            ]
        )

        for candidate in _deduplicate_existing_paths(preferred_candidates):
            if candidate.exists() and is_supported_java_home(candidate):
                return _activate_java_home(candidate)

        for candidate in _deduplicate_existing_paths(preferred_candidates):
            if candidate.exists():
                java_home = find_java_home_from_jvm_dll(candidate)
                if java_home is not None and is_supported_java_home(java_home):
                    return _activate_java_home(java_home)

    bundle_candidates: list[Path] = []
    pyinstaller_root = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and pyinstaller_root:
        root = Path(pyinstaller_root)
        bundle_candidates.extend(
            [
                root / "Library",
                root / "Library" / "lib" / "jvm",
                root / "jdk",
                root / "openjdk",
                root / "java",
                root / "jre",
            ]
        )

    for candidate in _deduplicate_existing_paths(bundle_candidates):
        if candidate.exists() and is_supported_java_home(candidate):
            return _activate_java_home(candidate)

    exe_dir = Path(sys.executable).resolve().parent
    environment_roots = [exe_dir, exe_dir.parent, Path(sys.prefix), Path(sys.base_prefix)]

    environment_candidates: list[Path] = []
    for root in _deduplicate_existing_paths(environment_roots):
        environment_candidates.extend(
            [
                root / "Library" / "lib" / "jvm",
                root / "Library",
                root / "jdk",
                root / "openjdk",
                root / "jre",
                root,
            ]
        )
        if root.exists() and root.name.lower() in {"jvm", "java"}:
            try:
                environment_candidates.extend(child for child in root.iterdir() if child.is_dir())
            except OSError:
                pass
    for candidate in _deduplicate_existing_paths(environment_candidates):
        if candidate.exists() and is_supported_java_home(candidate):
            return _activate_java_home(candidate)
    return None
