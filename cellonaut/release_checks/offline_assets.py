"""Prepare and verify the large assets required by offline releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from cellonaut.io.fiji_installation import (
    IMAGE_SCIENCE_CLASS,
    find_fiji_launcher,
    find_jar_containing_class,
    scan_fiji_installation,
)


CELLPOSE_MODEL_NAMES = ("cpsam", "cpsam_v2")
REMOVED_CELLPOSE_MODEL_NAMES = ("cpdino", "cpdino-vitb")
CELLPOSE_MODEL_REVISION = "7c61431b5fbb078f3296754bd15d9f51b320f837"
CELLPOSE_REPOSITORY_URL = "https://huggingface.co/mouseland/cellpose-sam"
CELLPOSE_MODEL_BASE_URL = f"{CELLPOSE_REPOSITORY_URL}/resolve/{CELLPOSE_MODEL_REVISION}"
FIJI_ARCHIVE_ROOT = "https://downloads.imagej.net/fiji/archive/latest/20260718-0417"
FIJI_ARCHIVE_NAME = "fiji-latest-win64-jdk.zip"
PYIMAGEJ_BRIDGE_ARTIFACTS = {
    "imglib2-imglyb-1.1.0.jar": (
        "https://maven.imagej.net/content/repositories/releases/net/imglib2/"
        "imglib2-imglyb/1.1.0/imglib2-imglyb-1.1.0.jar",
        "010e547ba4d3d4dc6d1c2cf94247ca8c6190344e1a829435e4ff48c36508e789",
    ),
    "imglib2-unsafe-1.0.0.jar": (
        "https://repo.maven.apache.org/maven2/net/imglib2/"
        "imglib2-unsafe/1.0.0/imglib2-unsafe-1.0.0.jar",
        "2077441d94f7198545e9d939e89a0358d9587bdf5c517a156fb8efe09349f62b",
    ),
}
PYIMAGEJ_BRIDGE_MAVEN_PATHS = {
    "imglib2-imglyb-1.1.0.jar": Path("net/imglib2/imglib2-imglyb/1.1.0/imglib2-imglyb-1.1.0.jar"),
    "imglib2-unsafe-1.0.0.jar": Path("net/imglib2/imglib2-unsafe/1.0.0/imglib2-unsafe-1.0.0.jar"),
}
PYIMAGEJ_BRIDGE_NOTICE = """ImgLib2 Python bridge JARs

Components: imglib2-imglyb 1.1.0 and imglib2-unsafe 1.0.0
Source: https://github.com/imglib/imglib2-imglyb and
        https://github.com/imglib/imglib2-unsafe
Copyright (c) Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
MINIMUM_MODEL_BYTES = 100_000_000
MODEL_NOTICE = """Cellpose built-in model weights

Source: https://huggingface.co/mouseland/cellpose-sam
Revision: 7c61431b5fbb078f3296754bd15d9f51b320f837
Repository license designation: BSD-3-Clause

The Cellpose project separately documents licensing terms for its training and
annotated datasets. Review the current upstream terms for the intended use.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fiji_archive_url() -> str:
    """Return the pinned Fiji archive URL used by the Windows release."""
    return f"{FIJI_ARCHIVE_ROOT}/{FIJI_ARCHIVE_NAME}"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            copied = 0
            next_report = 256 * 1024 * 1024
            print(f"Downloading {destination.name} ({total / (1024**2):.1f} MiB expected)...", flush=True)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                copied += len(chunk)
                if copied >= next_report:
                    print(f"  {destination.name}: {copied / (1024**2):.0f} MiB", flush=True)
                    next_report += 256 * 1024 * 1024
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _download_verified_archive(url: str, destination: Path) -> None:
    checksum_path = destination.with_suffix(destination.suffix + ".sha256")
    _download(f"{url}.sha256", checksum_path)
    expected = checksum_path.read_text(encoding="ascii").strip().split()[0].lower()
    if destination.exists() and _sha256(destination) != expected:
        destination.unlink()
    _download(url, destination)
    actual = _sha256(destination)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {destination.name}: expected {expected}, found {actual}")


def _download_verified_file(url: str, destination: Path, expected_sha256: str) -> None:
    """Download one pinned file and reject stale or corrupted cached copies."""
    if destination.exists() and _sha256(destination) != expected_sha256:
        destination.unlink()
    _download(url, destination)
    actual = _sha256(destination)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: expected {expected_sha256}, found {actual}"
        )


def _model_pointer(name: str, downloads_root: Path) -> tuple[str, int]:
    pointer_url = f"{CELLPOSE_REPOSITORY_URL}/raw/{CELLPOSE_MODEL_REVISION}/{name}"
    pointer_path = downloads_root / f"{name}.lfs-pointer"
    _download(pointer_url, pointer_path)
    pointer = pointer_path.read_text(encoding="ascii")
    sha_match = re.search(r"oid sha256:([0-9a-f]{64})", pointer)
    size_match = re.search(r"^size (\d+)$", pointer, re.MULTILINE)
    if sha_match is None or size_match is None:
        raise RuntimeError(f"Could not read the pinned model checksum pointer for {name}")
    return sha_match.group(1), int(size_match.group(1))


def _download_verified_model(name: str, destination: Path, downloads_root: Path) -> None:
    expected_sha, expected_size = _model_pointer(name, downloads_root)
    if destination.exists() and (
        destination.stat().st_size != expected_size or _sha256(destination) != expected_sha
    ):
        destination.unlink()
    _download(f"{CELLPOSE_MODEL_BASE_URL}/{name}", destination)
    if destination.stat().st_size != expected_size or _sha256(destination) != expected_sha:
        raise RuntimeError(f"Checksum or size mismatch for the pinned Cellpose model: {name}")


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        source.extractall(destination)


def _run_fiji_updater(fiji_path: Path) -> None:
    launcher = find_fiji_launcher(fiji_path)
    if launcher is None:
        raise RuntimeError(f"The downloaded Fiji archive has no usable launcher: {fiji_path}")
    launcher_command = [str(launcher)]
    if launcher.suffix.lower() in {".bat", ".cmd"}:
        launcher_command = ["cmd.exe", "/d", "/c", str(launcher)]
    commands = ([*launcher_command, "--headless", "--update", "update"],)
    for command in commands:
        result = subprocess.run(command, cwd=fiji_path, check=False, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(f"Fiji updater failed with exit code {result.returncode}: {' '.join(command)}")


def _remove_imagescience_from_release_bundle(fiji_path: Path) -> None:
    """Keep separately licensed ImageScience out of distributable Fiji assets."""
    jar_path = find_jar_containing_class(fiji_path, IMAGE_SCIENCE_CLASS)
    if jar_path is None:
        return
    if "imagescience" not in jar_path.name.casefold():
        raise RuntimeError(f"ImageScience is embedded in an unexpected archive and cannot be removed safely: {jar_path}")
    jar_path.unlink()


def _install_pyimagej_bridge(fiji_path: Path, downloads_root: Path) -> None:
    """Bundle the two Java bridge jars PyImageJ otherwise resolves with Maven."""
    jars_root = fiji_path / "jars"
    jars_root.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_sha256) in PYIMAGEJ_BRIDGE_ARTIFACTS.items():
        download = downloads_root / filename
        destination = jars_root / filename
        if not download.exists():
            candidates = [destination]
            maven_path = PYIMAGEJ_BRIDGE_MAVEN_PATHS.get(filename)
            if maven_path is not None:
                candidates.append(Path.home() / ".m2" / "repository" / maven_path)
            for candidate in candidates:
                if candidate.is_file() and _sha256(candidate) == expected_sha256:
                    download.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, download)
                    print(f"Using verified cached {filename}: {candidate}", flush=True)
                    break
        _download_verified_file(url, download, expected_sha256)
        if not destination.exists() or _sha256(destination) != expected_sha256:
            shutil.copy2(download, destination)
    licenses_root = fiji_path / "licenses"
    licenses_root.mkdir(parents=True, exist_ok=True)
    (licenses_root / "imglib2-python-bridge-BSD-2-Clause.txt").write_text(
        PYIMAGEJ_BRIDGE_NOTICE,
        encoding="utf-8",
    )


def _find_extracted_fiji_bundles(extraction_root: Path) -> list[Path]:
    """Return top-level Fiji folders from the Windows archive."""
    candidates = [
        path
        for path in extraction_root.rglob("*")
        if path.is_dir()
        and path.name.casefold() in {"fiji.app", "fiji"}
    ]
    return [
        path
        for path in candidates
        if not any(other != path and other in path.parents for other in candidates)
    ]


def prepare_fiji(offline_root: Path, downloads_root: Path) -> Path:
    """Prepare and validate the pinned Fiji bundle for offline redistribution."""
    destination = offline_root / "Fiji.app"
    if destination.exists():
        _remove_imagescience_from_release_bundle(destination)
        _install_pyimagej_bridge(destination, downloads_root)
        if scan_fiji_installation(destination).ready:
            return destination
        raise RuntimeError(f"An incomplete Fiji bundle already exists; remove it and rebuild: {destination}")

    archive_url = fiji_archive_url()
    archive = downloads_root / archive_url.rsplit("/", 1)[-1]
    _download_verified_archive(archive_url, archive)
    with tempfile.TemporaryDirectory(prefix="cellonaut-fiji-", dir=offline_root.parent) as temp_name:
        extraction_root = Path(temp_name)
        _safe_extract_zip(archive, extraction_root)
        candidates = _find_extracted_fiji_bundles(extraction_root)
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one Fiji or Fiji.app folder in {archive}, found {len(candidates)}"
            )
        staged = candidates[0]
        _run_fiji_updater(staged)
        _remove_imagescience_from_release_bundle(staged)
        _install_pyimagej_bridge(staged, downloads_root)
        status = scan_fiji_installation(staged)
        if not status.ready:
            missing = ", ".join(item.label for item in status.missing_required)
            raise RuntimeError(f"Prepared Fiji is incomplete; missing: {missing}")
        if destination.exists():
            raise RuntimeError(f"An incomplete Fiji bundle already exists; remove it and rebuild: {destination}")
        shutil.move(str(staged), str(destination))
    return destination


def prepare_cellpose_models(offline_root: Path, downloads_root: Path) -> Path:
    """Prepare supported Cellpose weights and remove unsupported DINO weights."""
    model_root = offline_root / "cellpose_models"
    model_root.mkdir(parents=True, exist_ok=True)
    for name in REMOVED_CELLPOSE_MODEL_NAMES:
        removed_model = model_root / name
        if removed_model.is_file():
            removed_model.unlink()
    for name in CELLPOSE_MODEL_NAMES:
        model_path = model_root / name
        _download_verified_model(name, model_path, downloads_root)
        if model_path.stat().st_size < MINIMUM_MODEL_BYTES:
            raise RuntimeError(f"Downloaded Cellpose model is unexpectedly small: {model_path}")
    (model_root / "MODEL_NOTICE.txt").write_text(MODEL_NOTICE, encoding="utf-8")
    return model_root


def verify_offline_assets(offline_root: Path) -> dict[str, object]:
    """Validate Fiji and model assets and return their release manifest."""
    fiji_path = offline_root / "Fiji.app"
    if find_jar_containing_class(fiji_path, IMAGE_SCIENCE_CLASS) is not None:
        raise RuntimeError("Offline Fiji bundle must not contain separately distributed ImageScience.")
    fiji_status = scan_fiji_installation(fiji_path)
    if not fiji_status.ready:
        missing = ", ".join(item.label for item in fiji_status.missing_required) or "Fiji installation"
        raise RuntimeError(f"Offline Fiji bundle is incomplete; missing: {missing}")
    for filename, (_url, expected_sha256) in PYIMAGEJ_BRIDGE_ARTIFACTS.items():
        path = fiji_path / "jars" / filename
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise RuntimeError(f"Offline Fiji bundle is missing its pinned PyImageJ bridge: {path}")
    if PYIMAGEJ_BRIDGE_ARTIFACTS:
        bridge_notice = fiji_path / "licenses" / "imglib2-python-bridge-BSD-2-Clause.txt"
        if not bridge_notice.is_file() or bridge_notice.read_text(encoding="utf-8") != PYIMAGEJ_BRIDGE_NOTICE:
            raise RuntimeError("Offline Fiji bundle is missing its ImgLib2 bridge license notice.")

    model_root = offline_root / "cellpose_models"
    for name in REMOVED_CELLPOSE_MODEL_NAMES:
        removed_model = model_root / name
        if removed_model.exists():
            raise RuntimeError(f"Unsupported DINO model must not be included in offline assets: {removed_model}")
    models: dict[str, dict[str, object]] = {}
    for name in CELLPOSE_MODEL_NAMES:
        path = model_root / name
        if not path.is_file() or path.stat().st_size < MINIMUM_MODEL_BYTES:
            raise RuntimeError(f"Offline Cellpose model is missing or incomplete: {path}")
        models[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}

    return {
        "offline": True,
        "fiji": str(fiji_path),
        "cellpose_models": models,
    }


def prepare_offline_assets(output: Path) -> dict[str, object]:
    """Prepare every offline asset and write the authoritative asset manifest."""
    offline_root = output / "offline"
    offline_root.mkdir(parents=True, exist_ok=True)
    downloads_root = output / "downloads"
    prepare_fiji(offline_root, downloads_root)
    prepare_cellpose_models(offline_root, downloads_root)
    manifest = verify_offline_assets(offline_root)
    (offline_root / "OFFLINE_ASSETS.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    """Run offline asset preparation or verification from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        manifest = prepare_offline_assets(args.output)
    else:
        manifest = verify_offline_assets(args.output / "offline")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
