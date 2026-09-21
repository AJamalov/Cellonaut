"""Collect licenses for the exact transitive dependency set bundled in a release."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
import shutil
import sys
from typing import Iterable

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


LICENSE_PREFIXES = ("license", "licence", "copying", "notice", "copyright", "unlicense")
LICENSE_OVERRIDES = {
    "antlr4-python3-runtime": Path(__file__).resolve().parents[1] / "data" / "licenses" / "ANTLR4_LICENSE.txt",
    "unlicense": Path(__file__).resolve().parents[1] / "data" / "licenses" / "UNLICENSE.txt",
}
QT_OPEN_SOURCE_DISTRIBUTIONS = {
    "pyside6",
    "pyside6-addons",
    "pyside6-essentials",
    "shiboken6",
}
QT_COMMERCIAL_REFERENCE = "licenseref-qt-commercial.txt"
QT_OPEN_SOURCE_NOTICE = "QT_PYSIDE_OPEN_SOURCE_NOTICE.txt"
QT_GPL_LICENSE = "GPL-3.0.txt"
PYTHON_LICENSE_FALLBACK = Path(__file__).resolve().parents[1] / "data" / "licenses" / "PYTHON_3_12_LICENSE.txt"


# Retain both metadata and copied-file evidence so the generated report can
# distinguish declared licensing from notices actually bundled in the artifact.
@dataclass(frozen=True)
class LicenseInventoryEntry:
    """Record one dependency and the license evidence copied for it."""
    name: str
    version: str
    license_summary: str
    distribution_basis: str
    source_url: str
    copied_license_files: tuple[str, ...]


def _installed_distribution(name: str) -> importlib_metadata.Distribution:
    """Resolve one installed package, preferring modern metadata over stale egg-info."""
    canonical_name = str(canonicalize_name(name))
    matches = [
        distribution
        for distribution in importlib_metadata.distributions()
        if canonicalize_name(_metadata_value(distribution, "Name")) == canonical_name
    ]
    if not matches:
        raise importlib_metadata.PackageNotFoundError(name)
    return max(matches, key=lambda distribution: str(getattr(distribution, "_path", "")).endswith(".dist-info"))


# Evaluate environment markers before traversing dependencies so the inventory
# describes the current release artifact rather than every optional platform.
def _active_requirements(distribution: importlib_metadata.Distribution) -> list[str]:
    environment: dict[str, str] = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    names: list[str] = []
    for raw_requirement in distribution.requires or ():
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        names.append(str(canonicalize_name(requirement.name)))
    return names


# Read roots from the installed project metadata so editable and packaged build
# environments use the same dependency authority as pip.
def _runtime_requirement_names(project_name: str = "cellonaut") -> list[str]:
    try:
        project_distribution = _installed_distribution(project_name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{project_name} must be installed before its dependency licenses can be collected."
        ) from exc
    return _active_requirements(project_distribution)


# License obligations include transitive packages, so walk the complete active
# dependency closure and canonicalize names to avoid duplicate distributions.
def _resolve_dependency_closure(root_names: Iterable[str]) -> list[importlib_metadata.Distribution]:
    pending: list[str] = [str(canonicalize_name(name)) for name in root_names]
    resolved: dict[str, importlib_metadata.Distribution] = {}
    while pending:
        requested_name = pending.pop()
        if requested_name in resolved:
            continue
        try:
            distribution = _installed_distribution(requested_name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required package is not installed: {requested_name}") from exc
        installed_name = str(canonicalize_name(_metadata_value(distribution, "Name") or requested_name))
        resolved[installed_name] = distribution
        pending.extend(name for name in _active_requirements(distribution) if name not in resolved)
    return [resolved[name] for name in sorted(resolved)]


# Package metadata may repeat fields; use the first declared value consistently
# across summaries, folder names, and generated reports.
def _metadata_value(distribution: importlib_metadata.Distribution, key: str) -> str:
    values = distribution.metadata.get_all(key, [])
    return values[0].strip() if values else ""


# Prefer standardized license expressions, then classifiers, because free-form
# Ambiguous package metadata is harder for release reviewers to interpret reliably.
def _license_summary(distribution: importlib_metadata.Distribution) -> str:
    expression = _metadata_value(distribution, "License-Expression")
    if expression:
        return expression

    classifiers = [
        value.removeprefix("License :: ").strip()
        for value in distribution.metadata.get_all("Classifier", [])
        if value.startswith("License :: ")
    ]
    if classifiers:
        return "; ".join(classifiers)

    declared_license = _metadata_value(distribution, "License")
    if declared_license and declared_license.upper() != "UNKNOWN":
        return declared_license.splitlines()[0].strip()
    return "Not declared in package metadata"


# Installed wheels place notices at different package depths; inspect declared
# distribution files instead of assuming one top-level LICENSE filename.
def _distribution_license_files(
    distribution: importlib_metadata.Distribution,
) -> list[tuple[importlib_metadata.PackagePath, Path]]:
    matches: list[tuple[importlib_metadata.PackagePath, Path]] = []
    for package_path in distribution.files or ():
        if not package_path.name.lower().startswith(LICENSE_PREFIXES):
            continue
        source = Path(str(distribution.locate_file(package_path)))
        if source.is_file():
            matches.append((package_path, source))
    return matches


# PySide wheels advertise open-source and commercial choices but currently
# bundle only a commercial-reference file. Cellonaut is GPLv3 software, so
# release packages select PySide's GPLv3 option and include that license text.
def _project_gpl_license() -> Path:
    source_license = Path(__file__).resolve().parents[2] / "LICENSE"
    if source_license.is_file():
        return source_license

    try:
        project_distribution = importlib_metadata.distribution("cellonaut")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError("Cellonaut's GPL license file could not be located.") from exc
    for package_path, license_path in _distribution_license_files(project_distribution):
        if package_path.name.casefold() == "license":
            return license_path
    raise RuntimeError("Cellonaut's GPL license file could not be located.")


def _write_qt_open_source_files(
    package_dir: Path,
    *,
    package_name: str,
    package_version: str,
    source_url: str,
) -> list[str]:
    shutil.copyfile(_project_gpl_license(), package_dir / QT_GPL_LICENSE)
    source_text = source_url or "https://code.qt.io/cgit/pyside/pyside-setup.git/"
    notice = (
        f"{package_name} {package_version} is part of Qt for Python.\n\n"
        "Cellonaut uses the community edition and distributes this component "
        "under its GPL-3.0-only open-source option. No Qt commercial license is claimed.\n\n"
        f"The full GPLv3 terms are provided in {QT_GPL_LICENSE}. Matching corresponding "
        "source and Qt third-party notices are provided from the same Cellonaut release "
        "page as the binary. See SOURCE_AVAILABILITY.md in the application folder.\n\n"
        f"Upstream source: {source_text}\n"
        "Qt licensing information: https://www.qt.io/development/open-source-lgpl-obligations\n"
    )
    (package_dir / QT_OPEN_SOURCE_NOTICE).write_text(notice, encoding="utf-8")
    return [QT_GPL_LICENSE, QT_OPEN_SOURCE_NOTICE]


# Prefer a source repository over a marketing home page so corresponding source
# remains straightforward to locate from the generated inventory.
def _source_url(distribution: importlib_metadata.Distribution) -> str:
    fallback = ""
    for project_url in distribution.metadata.get_all("Project-URL", []):
        label, separator, url = project_url.partition(",")
        if not separator:
            continue
        normalized_label = label.strip().casefold()
        if normalized_label in {"source", "repository", "source code"}:
            return url.strip()
        if normalized_label in {"homepage", "home"}:
            fallback = url.strip()
    return fallback or _metadata_value(distribution, "Home-page")


# Python is part of every frozen application but is not a normal project
# dependency, so collect its license directly from common installation layouts.
def _python_license_files() -> list[Path]:
    """Find the interpreter license, with a pinned copy for minimal build images."""

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    roots = {
        Path(sys.base_prefix),
        Path(sys.prefix),
        Path(sys.executable).resolve().parent,
    }
    candidates = [
        *(root / name for root in roots for name in ("LICENSE", "LICENSE.txt", "LICENSE_PYTHON.txt")),
        Path("/usr/share/doc") / f"python{version}" / "copyright",
        Path("/usr/share/doc") / "python3" / "copyright",
        PYTHON_LICENSE_FALLBACK,
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() not in {path.resolve() for path in unique}:
            unique.append(candidate)
    return unique


# Metadata becomes a path component in the release bundle; replace unsafe
# characters here rather than relying on platform-specific filename behavior.
def _safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


# Preserve the upstream path when it is short, but hash unusually deep paths
# so Windows installer tools do not exceed their path-length limits.
def _license_destination_name(package_path: importlib_metadata.PackagePath) -> str:
    raw_path = str(package_path).replace("\\", "/")
    flattened = _safe_component(raw_path.replace("/", "__"))
    if len(flattened) <= 96:
        return flattened
    digest = sha256(raw_path.encode("utf-8")).hexdigest()[:12]
    return f"{digest}__{_safe_component(package_path.name)[-80:]}"


# Require the exact destination name before clearing it because regeneration
# replaces the directory and must never remove an arbitrary caller-supplied path.
def write_license_inventory(
    output_dir: Path,
    *,
    root_names: Iterable[str] | None = None,
) -> list[LicenseInventoryEntry]:
    """Generate the dependency-license inventory used by a release bundle."""
    output_dir = Path(output_dir)
    if output_dir.name != "THIRD_PARTY_LICENSES":
        raise ValueError("The license inventory output folder must be named THIRD_PARTY_LICENSES.")

    distributions = _resolve_dependency_closure(root_names or _runtime_requirement_names())
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    entries: list[LicenseInventoryEntry] = []
    for distribution in distributions:
        name = _metadata_value(distribution, "Name") or "unknown-package"
        canonical_name = str(canonicalize_name(name))
        version = distribution.version
        package_dir = output_dir / f"{_safe_component(name)}-{_safe_component(version)}"
        package_dir.mkdir()

        copied_files: list[str] = []
        for package_path, source in _distribution_license_files(distribution):
            if (
                canonical_name in QT_OPEN_SOURCE_DISTRIBUTIONS
                and package_path.name.casefold() == QT_COMMERCIAL_REFERENCE
            ):
                continue
            destination_name = _license_destination_name(package_path)
            destination = package_dir / destination_name
            shutil.copyfile(source, destination)
            copied_files.append(destination_name)

        summary = _license_summary(distribution)
        source_url = _source_url(distribution)
        distribution_basis = summary
        if canonical_name in QT_OPEN_SOURCE_DISTRIBUTIONS:
            distribution_basis = "GPL-3.0-only (open-source option selected by Cellonaut)"
            copied_files.extend(
                _write_qt_open_source_files(
                    package_dir,
                    package_name=name,
                    package_version=version,
                    source_url=source_url,
                )
            )
        if not copied_files:
            override = LICENSE_OVERRIDES.get(canonical_name, LICENSE_OVERRIDES.get(summary.casefold()))
            if override is not None and override.is_file():
                destination_name = override.name
                shutil.copyfile(override, package_dir / destination_name)
                copied_files.append(destination_name)
        metadata_text = (
            f"Package: {name}\n"
            f"Version: {version}\n"
            f"License metadata: {summary}\n"
            f"Cellonaut distribution basis: {distribution_basis}\n"
            f"Source URL: {source_url}\n"
            f"Project URL: {_metadata_value(distribution, 'Project-URL')}\n"
            f"Home page: {_metadata_value(distribution, 'Home-page')}\n"
        )
        (package_dir / "PACKAGE_METADATA.txt").write_text(metadata_text, encoding="utf-8")
        entries.append(
            LicenseInventoryEntry(
                name=name,
                version=version,
                license_summary=summary,
                distribution_basis=distribution_basis,
                source_url=source_url,
                copied_license_files=tuple(sorted(copied_files)),
            )
        )

    python_licenses = _python_license_files()
    if not python_licenses:
        raise RuntimeError("The build environment does not expose a Python license file.")
    python_dir = output_dir / "Python"
    python_dir.mkdir()
    for source in python_licenses:
        shutil.copyfile(source, python_dir / _safe_component(source.name))

    lines = [
        "# Third-Party License Inventory",
        "",
        "This inventory was generated from the installed runtime dependency closure used to build Cellonaut.",
        "Package license files are copied without modification when they are present in the installed distribution.",
        "",
        "| Package | Version | Declared license | Cellonaut distribution basis | Source | Packaged license files |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        license_files = ", ".join(entry.copied_license_files) or "None supplied by package"
        summary = entry.license_summary.replace("|", "\\|").replace("\n", " ")
        basis = entry.distribution_basis.replace("|", "\\|").replace("\n", " ")
        source_link = entry.source_url.replace("|", "%7C")
        source_cell = f"[project source]({source_link})" if source_link else "Not declared"
        lines.append(f"| {entry.name} | {entry.version} | {summary} | {basis} | {source_cell} | {license_files} |")
    (output_dir / "DEPENDENCIES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return entries


# Make the output path explicit at the command line so build scripts cannot
# silently place legal files in a temporary or unrelated working directory.
def main(argv: list[str] | None = None) -> int:
    """Generate an inventory and fail when a package lacks license text."""
    parser = argparse.ArgumentParser(description="Collect licenses for Cellonaut's installed runtime dependencies.")
    parser.add_argument("--output", required=True, type=Path, help="Destination named THIRD_PARTY_LICENSES.")
    args = parser.parse_args(argv)

    entries = write_license_inventory(args.output)
    missing = [entry.name for entry in entries if not entry.copied_license_files]
    print(f"Collected license information for {len(entries)} runtime packages.")
    if missing:
        print("Packages without a bundled license file: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
