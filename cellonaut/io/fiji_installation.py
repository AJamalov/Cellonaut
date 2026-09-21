"""Inspect Fiji folders and report the components Cellonaut needs."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable


BIOP_IMAGE_LOADER_SERVICE = "ch/epfl/biop/bdv/img/Services.class"
MASTODON_PLUGIN_API = "org/mastodon/mamut/plugin/MamutPlugin.class"
IMAGE_SCIENCE_CLASS = "imagescience/image/Image.class"
WEKA_SEGMENTATION_CLASS = "trainableSegmentation/WekaSegmentation.class"
BIO_FORMATS_READER_CLASS = "loci/formats/ImageReader.class"
BIO_FORMATS_PLUGIN_CLASS = "loci/plugins/BF.class"


# Represent each dependency check as data so background scans can cross the Qt
# thread boundary without passing Java, filesystem, or widget objects.
@dataclass(frozen=True)
class FijiComponentStatus:
    """Describe one required or optional component in a Fiji installation."""
    key: str
    label: str
    ok: bool
    detail: str
    required: bool = True
    version: str | None = None


# The requested path and all component outcomes stay together so stale
# asynchronous scan results can be rejected safely by the GUI.
@dataclass(frozen=True)
class FijiInstallationStatus:
    """Describe whether a Fiji folder contains Cellonaut's required runtime."""
    path: Path
    path_exists: bool
    path_is_dir: bool
    looks_like_fiji: bool
    launcher: Path | None
    components: tuple[FijiComponentStatus, ...]
    version: str | None = None

    # Callers usually need only actionable failures, not optional diagnostics.
    @property
    def missing_required(self) -> tuple[FijiComponentStatus, ...]:
        return tuple(component for component in self.components if component.required and not component.ok)

    # Require a recognizable Fiji folder and all required components.
    @property
    def ready(self) -> bool:
        return self.looks_like_fiji and not self.missing_required


def fiji_launcher_candidates(fiji_app_path: Path) -> list[Path]:
    """Return supported launcher paths for the bundled Windows Fiji runtime."""
    names = (
        "ImageJ-win64.exe",
        "fiji-windows-x64.exe",
        "fiji.bat",
        "ImageJ-win32.exe",
        "ImageJ.exe",
        "fiji-win64.exe",
    )
    return [fiji_app_path / name for name in names]


# Launcher checks are intentionally shallow because plugin validation handles the deeper inspection.
def find_fiji_launcher(fiji_app_path: Path) -> Path | None:
    """Return the first usable Fiji launcher under the supplied folder."""
    for candidate in fiji_launcher_candidates(Path(fiji_app_path)):
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


# Marker folders support Fiji distributions whose launcher has been renamed by a package manager.
def looks_like_fiji_installation(fiji_app_path: Path) -> bool:
    """Return whether a folder has the shallow structure of a Fiji installation."""
    path = Path(fiji_app_path)
    if not path.exists() or not path.is_dir():
        return False
    if find_fiji_launcher(path) is not None:
        return True
    marker_count = sum(1 for marker in ("jars", "plugins", "macros", "scripts") if (path / marker).is_dir())
    return marker_count >= 2 and "fiji" in path.name.lower()


# Prefer Fiji's bundled Java because it is the runtime tested with that Fiji installation.
def _java_runtime_status(fiji_app_path: Path) -> FijiComponentStatus:
    from cellonaut.io.java_runtime import find_java_home_from_jvm_dll, is_valid_java_home

    java_home = find_java_home_from_jvm_dll(fiji_app_path)
    if java_home is not None:
        return FijiComponentStatus(
            "java",
            "Java runtime",
            True,
            f"Found local Java runtime: {java_home}",
        )

    env_java_home = os.environ.get("JAVA_HOME", "").strip()
    if env_java_home and is_valid_java_home(Path(env_java_home)):
        return FijiComponentStatus(
            "java",
            "Java runtime",
            True,
            f"Using JAVA_HOME: {env_java_home}",
        )

    return FijiComponentStatus(
        "java",
        "Java runtime",
        False,
        "No local Fiji Java runtime or valid JAVA_HOME was found.",
    )


# Scan jar files once because large Fiji installations can contain hundreds of archives.
def _find_jars_and_versions(
    fiji_app_path: Path,
    class_files: tuple[str, ...],
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[dict[str, Path | None], dict[Path, str | None]]:
    requested = tuple(dict.fromkeys(str(class_file) for class_file in class_files if str(class_file)))
    found: dict[str, Path | None] = {class_file: None for class_file in requested}
    remaining = set(requested)
    versions: dict[Path, str | None] = {}
    if not remaining:
        return found, versions

    try:
        for jar_path in Path(fiji_app_path).rglob("*.jar"):
            if cancel_requested is not None and cancel_requested():
                raise InterruptedError("Fiji installation scan cancelled")
            try:
                with zipfile.ZipFile(jar_path) as archive:
                    archive_entries = set(archive.namelist())
                    try:
                        manifest = archive.read("META-INF/MANIFEST.MF").decode("utf-8", errors="replace")
                    except KeyError:
                        manifest = ""
            except (OSError, zipfile.BadZipFile):
                continue

            for class_file in tuple(remaining.intersection(archive_entries)):
                found[class_file] = jar_path
                versions[jar_path] = _implementation_version_from_manifest(manifest)
                remaining.remove(class_file)
            if not remaining:
                break
    except InterruptedError:
        raise
    except OSError:
        pass
    return found, versions


def find_jars_containing_classes(fiji_app_path: Path, class_files: tuple[str, ...]) -> dict[str, Path | None]:
    """Locate requested Java classes with one pass over the Fiji jar set."""
    return _find_jars_and_versions(fiji_app_path, class_files)[0]


def _implementation_version_from_manifest(manifest: str) -> str | None:
    for line in manifest.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if line.lower().startswith("implementation-version:"):
            return line.split(":", 1)[1].strip() or None
    return None


def _jar_implementation_version(jar_path: Path | None) -> str | None:
    if jar_path is None:
        return None
    try:
        with zipfile.ZipFile(jar_path) as archive:
            manifest = archive.read("META-INF/MANIFEST.MF").decode("utf-8", errors="replace")
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    return _implementation_version_from_manifest(manifest)


def _scanned_jar_version(versions: dict[Path, str | None], jar_path: Path | None) -> str | None:
    """Return scanned manifest metadata while accepting an absent class JAR."""
    return versions.get(jar_path) if jar_path is not None else None


def _fiji_version(fiji_app_path: Path) -> str | None:
    try:
        candidates = sorted((fiji_app_path / "jars").glob("fiji-*.jar"))
    except OSError:
        return None
    for candidate in candidates:
        version = _jar_implementation_version(candidate)
        if version:
            return version
        name = candidate.stem
        if name.startswith("fiji-") and name != "fiji-lib" and name != "fiji-links":
            return name.removeprefix("fiji-")
    return None


def _java_version(fiji_app_path: Path) -> str | None:
    try:
        release_files = list((fiji_app_path / "java").rglob("release"))
    except OSError:
        return None
    for release_file in release_files:
        try:
            lines = release_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("JAVA_VERSION="):
                return line.split("=", 1)[1].strip().strip('"') or None
    return None


def scan_fiji_installation(
    fiji_app_path: Path | str,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> FijiInstallationStatus:
    """Inspect Fiji, Java, Bio-Formats, Weka, and optional ImageScience availability."""
    if cancel_requested is not None and cancel_requested():
        raise InterruptedError("Fiji installation scan cancelled")
    path = Path(str(fiji_app_path or "")).expanduser()
    path_exists = path.exists()
    path_is_dir = path.is_dir() if path_exists else False
    launcher = find_fiji_launcher(path) if path_is_dir else None
    looks_like_fiji = looks_like_fiji_installation(path) if path_is_dir else False

    components: list[FijiComponentStatus] = []
    if looks_like_fiji:
        class_jars, jar_versions = _find_jars_and_versions(
            path,
            (
                BIO_FORMATS_READER_CLASS,
                BIO_FORMATS_PLUGIN_CLASS,
                WEKA_SEGMENTATION_CLASS,
                IMAGE_SCIENCE_CLASS,
            ),
            cancel_requested=cancel_requested,
        )
        bioformats_found = bool(class_jars[BIO_FORMATS_READER_CLASS] or class_jars[BIO_FORMATS_PLUGIN_CLASS])
        weka_found = class_jars[WEKA_SEGMENTATION_CLASS] is not None
        imagescience_found = class_jars[IMAGE_SCIENCE_CLASS] is not None
        java_version = _java_version(path)

        components.extend(
            [
                FijiComponentStatus(
                    "launcher",
                    "Fiji launcher",
                    launcher is not None,
                    f"Found launcher: {launcher}"
                    if launcher is not None
                    else "No platform launcher was found in this Fiji folder.",
                ),
                FijiComponentStatus(
                    "java",
                    "Java",
                    (java_status := _java_runtime_status(path)).ok,
                    java_status.detail,
                    version=java_version,
                ),
                FijiComponentStatus(
                    "bioformats",
                    "Bio-Formats / ND2",
                    bioformats_found,
                    "Bio-Formats classes found." if bioformats_found else "Bio-Formats classes were not found.",
                    version=_scanned_jar_version(
                        jar_versions,
                        class_jars[BIO_FORMATS_PLUGIN_CLASS] or class_jars[BIO_FORMATS_READER_CLASS]
                    ),
                ),
                FijiComponentStatus(
                    "weka",
                    "Trainable Weka Segmentation",
                    weka_found,
                    "Trainable Weka Segmentation found."
                    if weka_found
                    else "Trainable Weka Segmentation was not found.",
                    version=_scanned_jar_version(jar_versions, class_jars[WEKA_SEGMENTATION_CLASS]),
                ),
                FijiComponentStatus(
                    "imagescience",
                    "ImageScience",
                    imagescience_found,
                    "ImageScience found." if imagescience_found else "ImageScience was not found.",
                    required=False,
                    version=_scanned_jar_version(jar_versions, class_jars[IMAGE_SCIENCE_CLASS]),
                ),
            ]
        )

    return FijiInstallationStatus(
        path=path,
        path_exists=path_exists,
        path_is_dir=path_is_dir,
        looks_like_fiji=looks_like_fiji,
        launcher=launcher,
        components=tuple(components),
        version=_fiji_version(path) if looks_like_fiji else None,
    )


# Keep the single-class API for focused checks while sharing the one-pass scanner.
def find_jar_containing_class(fiji_app_path: Path, class_file: str) -> Path | None:
    """Return the jar containing one Java class, if installed."""
    return find_jars_containing_classes(fiji_app_path, (class_file,))[class_file]


def fiji_contains_class(fiji_app_path: Path, class_file: str) -> bool:
    """Return whether the Fiji classpath contains a requested Java class."""
    return find_jar_containing_class(fiji_app_path, class_file) is not None


# A partial BIOP installation can break local startup, so use the clean Maven runtime in that specific case.
def requires_clean_fiji_runtime(fiji_app_path: Path) -> bool:
    """Detect the partial BIOP combination that requires PyImageJ's clean fallback."""
    class_jars = find_jars_containing_classes(
        fiji_app_path,
        (BIOP_IMAGE_LOADER_SERVICE, MASTODON_PLUGIN_API),
    )
    return class_jars[BIOP_IMAGE_LOADER_SERVICE] is not None and class_jars[MASTODON_PLUGIN_API] is None


# The clean Maven fallback does not inherit Fiji's plugin classpath. Add an
# optional user-installed ImageScience jar explicitly when it is available.
def add_clean_runtime_dependencies(
    sj_mod,
    fiji_app_path: Path,
) -> None:
    """Add optional local jars that PyImageJ's clean Maven fallback cannot see."""
    image_science_jar = find_jar_containing_class(fiji_app_path, IMAGE_SCIENCE_CLASS)
    if image_science_jar is None:
        return

    sj_mod.config.add_classpath(str(image_science_jar))
