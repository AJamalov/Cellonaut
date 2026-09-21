"""Build the corresponding-source companion archive for an official release."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
import zipfile

from cellonaut.version import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_REPOSITORY = "https://github.com/AJamalov/Cellonaut"


@dataclass(frozen=True, kw_only=True)
class SourceArchive:
    """One immutable upstream source archive included in the companion bundle."""

    component: str
    filename: str
    url: str
    expected_sha256: str


@dataclass(frozen=True, kw_only=True)
class GitSource:
    """One upstream repository archived from a verified full commit ID."""

    component: str
    filename: str
    repository: str
    commit: str


DIRECT_SOURCES = (
    SourceArchive(
        component="Qt",
        filename="qt-everywhere-src-6.10.3.tar.xz",
        url="https://download.qt.io/official_releases/qt/6.10/6.10.3/single/qt-everywhere-src-6.10.3.tar.xz",
        expected_sha256="cbc81e726b0ff3c0cdb0219bf74545e91cec013c4a8503c20f93f83d73dff5d2",
    ),
    SourceArchive(
        component="Qt for Python / PySide",
        filename="pyside-setup-everywhere-src-6.10.3.tar.xz",
        url=(
            "https://download.qt.io/official_releases/QtForPython/pyside6/"
            "PySide6-6.10.3-src/pyside-setup-everywhere-src-6.10.3.tar.xz"
        ),
        expected_sha256="2c7462fe0cecb5b8ac0a3d92014b8d0b88bd4d9f8646709dab5286d9416f45bc",
    ),
    SourceArchive(
        component="Bio-Formats",
        filename="bioformats-8.5.0.tar.xz",
        url="https://downloads.openmicroscopy.org/bio-formats/8.5.0/artifacts/bioformats-8.5.0.tar.xz",
        expected_sha256="0bb858f9517d9e8e9b48b5b22592387b6b72963a2cd95e2bf844980e23f8f349",
    ),
)
GIT_SOURCES = (
    GitSource(
        component="Fiji",
        filename="fiji-c691dc761719086b49a9f927e77d744ae4e5a816.tar.gz",
        repository="https://github.com/fiji/fiji.git",
        commit="c691dc761719086b49a9f927e77d744ae4e5a816",
    ),
    GitSource(
        component="Trainable Weka Segmentation",
        filename="trainable-segmentation-a55f593c08ea3fd94e860a7cd1878f58ed1bff1b.tar.gz",
        repository="https://github.com/fiji/Trainable_Segmentation.git",
        commit="a55f593c08ea3fd94e860a7cd1878f58ed1bff1b",
    ),
)
PYPI_SOURCES = (("fastremap", "1.20.0"), ("fill-voids", "2.1.2"))


def _digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    if not expected_sha256:
        raise ValueError(f"A known SHA-256 digest is required for {url}")
    if destination.is_file() and _digest(destination) == expected_sha256:
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    with urlopen(url) as response, partial.open("wb") as output:  # noqa: S310 - pinned release sources
        shutil.copyfileobj(response, output, length=1024 * 1024)
    digest = _digest(partial)
    if digest != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {url}: {digest}")
    partial.replace(destination)


def _run_git(repository_dir: Path, *args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository_dir,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _archive_git_source(source: GitSource, destination: Path, repositories_dir: Path) -> dict[str, object]:
    """Create an archive only after Git validates the pinned commit object."""

    repository_dir = repositories_dir / source.component.casefold().replace(" ", "-").replace("/", "-")
    repository_dir.mkdir(parents=True, exist_ok=True)
    if not (repository_dir / ".git").is_dir():
        _run_git(repository_dir, "init")

    remotes = _run_git(repository_dir, "remote", capture_output=True).stdout.splitlines()
    if "origin" in remotes:
        _run_git(repository_dir, "remote", "set-url", "origin", source.repository)
    else:
        _run_git(repository_dir, "remote", "add", "origin", source.repository)

    present = subprocess.run(
        ["git", "cat-file", "-e", f"{source.commit}^{{commit}}"],
        cwd=repository_dir,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    if not present:
        _run_git(repository_dir, "fetch", "--depth=1", "origin", source.commit)

    resolved = _run_git(
        repository_dir, "rev-parse", f"{source.commit}^{{commit}}", capture_output=True
    ).stdout.strip()
    if resolved != source.commit:
        raise RuntimeError(
            f"Git source mismatch for {source.component}: expected {source.commit}, found {resolved}"
        )
    _run_git(repository_dir, "fsck", "--full", "--no-dangling", source.commit)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    prefix = destination.name.removesuffix(".tar.gz") + "/"
    _run_git(
        repository_dir,
        "archive",
        "--format=tar.gz",
        f"--prefix={prefix}",
        f"--output={temporary.resolve()}",
        source.commit,
    )
    temporary.replace(destination)
    return {
        "component": source.component,
        "filename": source.filename,
        "source": source.repository,
        "commit": source.commit,
        "bytes": destination.stat().st_size,
        "sha256": _digest(destination),
    }


def _pypi_source(package: str, version: str) -> SourceArchive:
    metadata_url = f"https://pypi.org/pypi/{package}/{version}/json"
    with urlopen(metadata_url) as response:  # noqa: S310 - official PyPI metadata API
        metadata = json.load(response)
    candidates = [item for item in metadata.get("urls", []) if item.get("packagetype") == "sdist"]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one source distribution for {package} {version}, found {len(candidates)}")
    candidate = candidates[0]
    digest = str(candidate.get("digests", {}).get("sha256", ""))
    if not digest:
        raise RuntimeError(f"PyPI did not provide a SHA-256 digest for {package} {version}")
    return SourceArchive(
        component=package,
        filename=str(candidate["filename"]),
        url=str(candidate["url"]),
        expected_sha256=digest,
    )


def _git_source(destination: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status:
        raise RuntimeError("Create the corresponding-source bundle from a clean, committed checkout.")
    release_tag = f"v{__version__}"
    tags = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if release_tag not in tags:
        raise RuntimeError(f"Create and check out release tag {release_tag} before building corresponding sources.")
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            f"--prefix=Cellonaut-{__version__}/",
            f"--output={destination}",
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return {
        "component": "Cellonaut",
        "filename": destination.name,
        "source": f"{PUBLIC_REPOSITORY}/tree/{release_tag}",
        "tag": release_tag,
        "commit": commit,
        "bytes": destination.stat().st_size,
        "sha256": _digest(destination),
    }


def prepare_corresponding_sources(output_dir: Path) -> Path:
    """Download pinned sources and create the release-page companion archive."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / ".local" / "corresponding_sources" / __version__
    cache_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    project_archive = cache_dir / f"Cellonaut-{__version__}-source.zip"
    records.append(_git_source(project_archive))

    sources = [*DIRECT_SOURCES, *(_pypi_source(package, version) for package, version in PYPI_SOURCES)]
    for source in sources:
        destination = cache_dir / source.filename
        _download(source.url, destination, source.expected_sha256)
        records.append(
            {
                "component": source.component,
                "filename": source.filename,
                "source": source.url,
                "bytes": destination.stat().st_size,
                "sha256": _digest(destination),
            }
        )

    repositories_dir = PROJECT_ROOT / ".local" / "corresponding_sources_git"
    for git_source in GIT_SOURCES:
        records.append(
            _archive_git_source(git_source, cache_dir / git_source.filename, repositories_dir)
        )

    manifest = {"cellonaut_version": __version__, "release_repository": PUBLIC_REPOSITORY, "files": records}
    manifest_path = cache_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(PROJECT_ROOT / "SOURCE_AVAILABILITY.md", cache_dir / "README.md")

    archive_path = output_dir / f"Cellonaut-{__version__}-corresponding-sources.zip"
    archive_path.unlink(missing_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(cache_dir.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() and not path.name.endswith(".part"):
                archive.write(path, path.name)
    return archive_path


def main(argv: list[str] | None = None) -> int:
    """Build the corresponding-source archive requested by the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    archive = prepare_corresponding_sources(args.output)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
