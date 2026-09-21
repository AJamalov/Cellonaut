"""Validate that release documents agree with code and prepared offline assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from cellonaut.release_checks.offline_assets import (
    CELLPOSE_MODEL_NAMES,
    CELLPOSE_MODEL_REVISION,
    MODEL_NOTICE,
)
from cellonaut.version import __version__


REQUIRED_DOCUMENTS = (
    "README.md",
    "CITATION.cff",
    "CITATIONS.md",
    "THIRD_PARTY_NOTICES.md",
    "BUNDLED_COMPONENTS.md",
    "SOURCE_AVAILABILITY.md",
)
REQUIRED_CITATION_DOIS = (
    "10.1101/2025.04.28.651001",
    "10.1038/s41592-020-01018-x",
    "10.1038/s41592-022-01663-4",
    "10.1038/s41592-025-02595-5",
    "10.1038/nmeth.2019",
    "10.1038/nmeth.2089",
    "10.1038/s41592-022-01655-4",
    "10.1186/s12859-017-1934-z",
    "10.1083/jcb.201004104",
    "10.1093/bioinformatics/btx180",
)
STALE_MODEL_PHRASES = (
    "all four built-in Cellpose",
    "four built-in Cellpose",
    "four built-in model weights",
    "every built-in Cellpose model",
)
MODEL_DOCUMENTS = (
    "README.md",
    "CITATIONS.md",
    "THIRD_PARTY_NOTICES.md",
    "BUNDLED_COMPONENTS.md",
)
RELEASE_ARTIFACT_NAMES = (
    f"Cellonaut-{__version__}-windows.exe",
)
RELEASE_SUPPORT_FILE_NAMES = (
    f"Cellonaut-{__version__}-windows.sha256",
)
DEFERRED_PLATFORM_ARTIFACT_MARKERS = ("-macos-", "-linux-")
DEFERRED_PLATFORM_NAMES = ("macOS", "Linux")


def _read_documents(project_root: Path) -> tuple[dict[str, str], list[str]]:
    """Read the required release documents and report missing files uniformly."""
    documents: dict[str, str] = {}
    problems: list[str] = []
    for relative_path in REQUIRED_DOCUMENTS:
        path = project_root / relative_path
        try:
            documents[relative_path] = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"could not read {relative_path}: {exc}")
    return documents, problems


def _cff_scalar(citation: str, key: str) -> str | None:
    """Read a top-level scalar needed for synchronization without a YAML dependency."""
    match = re.search(rf"(?m)^{re.escape(key)}:\s*[\"']?([^\"'\r\n]+)", citation)
    return match.group(1).strip() if match else None


def validate_release_documents(project_root: Path, offline_root: Path | None = None) -> list[str]:
    """Return every metadata, attribution, or offline-asset mismatch found."""
    project_root = Path(project_root)
    documents, problems = _read_documents(project_root)
    if problems:
        return problems

    readme = documents["README.md"]
    citation_cff = documents["CITATION.cff"]
    citations = documents["CITATIONS.md"]
    bundled = documents["BUNDLED_COMPONENTS.md"]
    combined = "\n".join(documents.values())

    if f"Current release: **Cellonaut {__version__}**" not in readme:
        problems.append(f"README.md does not identify version {__version__}")
    if _cff_scalar(citation_cff, "version") != __version__:
        problems.append(f"CITATION.cff version does not match {__version__}")
    if _cff_scalar(citation_cff, "license") != "GPL-3.0-or-later":
        problems.append("CITATION.cff license does not match GPL-3.0-or-later")
    if _cff_scalar(citation_cff, "repository-code") != "https://github.com/AJamalov/Cellonaut":
        problems.append("CITATION.cff repository-code does not match the release repository")
    for document_name, document in (("README.md", readme),):
        for artifact_name in RELEASE_ARTIFACT_NAMES:
            if artifact_name not in document:
                problems.append(f"{document_name} does not name release artifact {artifact_name}")
        for support_name in RELEASE_SUPPORT_FILE_NAMES:
            if support_name not in document:
                problems.append(f"{document_name} does not name release support file {support_name}")
        normalized_document = " ".join(document.split())
        if "Source code" not in normalized_document or "not runnable" not in normalized_document:
            problems.append(f"{document_name} does not distinguish source archives from runnable packages")
        if "Windows-only release" not in normalized_document and "Windows only" not in normalized_document:
            problems.append(f"{document_name} does not identify the release as Windows-only")
        for marker in DEFERRED_PLATFORM_ARTIFACT_MARKERS:
            if marker in document:
                problems.append(f"{document_name} advertises deferred platform artifact marker {marker!r}")
        for platform_name in DEFERRED_PLATFORM_NAMES:
            if platform_name.casefold() in document.casefold():
                problems.append(f"{document_name} mentions deferred platform {platform_name!r}")
    for document_name in MODEL_DOCUMENTS:
        document = documents[document_name]
        for model_name in CELLPOSE_MODEL_NAMES:
            if f"`{model_name}`" not in document:
                problems.append(f"{document_name} does not name supported Cellpose model {model_name}")
    for phrase in STALE_MODEL_PHRASES:
        if phrase.casefold() in combined.casefold():
            problems.append(f"release documents contain stale model wording: {phrase!r}")
    if CELLPOSE_MODEL_REVISION not in bundled:
        problems.append("BUNDLED_COMPONENTS.md does not contain the pinned Cellpose revision")
    for doi in REQUIRED_CITATION_DOIS:
        if doi not in citations:
            problems.append(f"CITATIONS.md is missing DOI {doi}")

    if offline_root is not None:
        offline_root = Path(offline_root)
        manifest_path = offline_root / "OFFLINE_ASSETS.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"could not read valid offline asset manifest {manifest_path}: {exc}")
        else:
            models = manifest.get("cellpose_models")
            if not isinstance(models, dict) or set(models) != set(CELLPOSE_MODEL_NAMES):
                problems.append("offline asset manifest model names do not match supported models")
            else:
                for model_name in CELLPOSE_MODEL_NAMES:
                    metadata = models.get(model_name)
                    if not isinstance(metadata, dict):
                        problems.append(f"offline asset manifest has invalid metadata for {model_name}")
                        continue
                    size = metadata.get("bytes")
                    digest = str(metadata.get("sha256", ""))
                    if not isinstance(size, int) or f"{size:,}" not in bundled:
                        problems.append(f"BUNDLED_COMPONENTS.md size does not match {model_name}")
                    if not digest or digest not in bundled:
                        problems.append(f"BUNDLED_COMPONENTS.md SHA-256 does not match {model_name}")

        notice_path = offline_root / "cellpose_models" / "MODEL_NOTICE.txt"
        try:
            notice = notice_path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"could not read {notice_path}: {exc}")
        else:
            if notice != MODEL_NOTICE:
                problems.append("prepared MODEL_NOTICE.txt does not match the release source")

    return problems


def main(argv: list[str] | None = None) -> int:
    """Run release-document validation as a build-friendly command-line gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--offline-assets", type=Path)
    args = parser.parse_args(argv)

    problems = validate_release_documents(args.project_root, args.offline_assets)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print("Release documents and offline asset metadata are synchronized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
