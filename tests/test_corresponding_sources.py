from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import subprocess

import pytest

from cellonaut.release_checks import corresponding_sources


def test_corresponding_source_manifest_pins_required_components():
    sources = {source.component: source for source in corresponding_sources.DIRECT_SOURCES}

    assert set(sources) == {
        "Qt",
        "Qt for Python / PySide",
        "Bio-Formats",
    }
    assert "6.10.3" in sources["Qt"].url
    assert sources["Qt"].expected_sha256 == (
        "cbc81e726b0ff3c0cdb0219bf74545e91cec013c4a8503c20f93f83d73dff5d2"
    )
    assert sources["Qt for Python / PySide"].expected_sha256 == (
        "2c7462fe0cecb5b8ac0a3d92014b8d0b88bd4d9f8646709dab5286d9416f45bc"
    )
    assert sources["Bio-Formats"].expected_sha256 == (
        "0bb858f9517d9e8e9b48b5b22592387b6b72963a2cd95e2bf844980e23f8f349"
    )
    assert all(source.expected_sha256 for source in corresponding_sources.DIRECT_SOURCES)
    git_sources = {source.component: source for source in corresponding_sources.GIT_SOURCES}
    assert git_sources["Fiji"].commit == "c691dc761719086b49a9f927e77d744ae4e5a816"
    assert git_sources["Trainable Weka Segmentation"].commit == (
        "a55f593c08ea3fd94e860a7cd1878f58ed1bff1b"
    )
    assert corresponding_sources.PYPI_SOURCES == (("fastremap", "1.20.0"), ("fill-voids", "2.1.2"))


def test_source_digest_reads_binary_content(tmp_path: Path):
    source = tmp_path / "source.tar.gz"
    content = b"pinned source"
    source.write_bytes(content)

    assert corresponding_sources._digest(source) == sha256(content).hexdigest()


def test_download_reuses_a_verified_cached_archive(tmp_path: Path, monkeypatch):
    destination = tmp_path / "source.tar.gz"
    content = b"cached source"
    destination.write_bytes(content)
    monkeypatch.setattr(
        corresponding_sources,
        "urlopen",
        lambda _url: pytest.fail("a valid cached source should not be downloaded"),
    )

    corresponding_sources._download(
        "https://example.invalid/source.tar.gz",
        destination,
        sha256(content).hexdigest(),
    )

    assert destination.read_bytes() == content


def test_download_requires_a_known_digest(tmp_path: Path):
    with pytest.raises(ValueError, match="known SHA-256"):
        corresponding_sources._download(
            "https://example.invalid/source.tar.gz",
            tmp_path / "source.tar.gz",
            "",
        )


def test_git_source_archive_is_created_from_verified_commit(tmp_path: Path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    (upstream / "source.txt").write_text("verified source\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.txt"], cwd=upstream, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Cellonaut Tests", "-c", "user.email=tests@example.invalid",
         "commit", "-m", "source"],
        cwd=upstream,
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=upstream, check=True, capture_output=True, text=True
    ).stdout.strip()
    source = corresponding_sources.GitSource(
        component="Example",
        filename=f"example-{commit}.tar.gz",
        repository=str(upstream),
        commit=commit,
    )
    destination = tmp_path / source.filename

    record = corresponding_sources._archive_git_source(
        source,
        destination,
        tmp_path / "repositories",
    )

    assert destination.is_file()
    assert record["commit"] == commit
    assert record["sha256"] == corresponding_sources._digest(destination)


def test_download_replaces_invalid_cache_atomically(tmp_path: Path, monkeypatch):
    destination = tmp_path / "source.tar.gz"
    destination.write_bytes(b"stale")
    content = b"verified replacement"
    monkeypatch.setattr(corresponding_sources, "urlopen", lambda _url: io.BytesIO(content))

    corresponding_sources._download(
        "https://example.invalid/source.tar.gz",
        destination,
        sha256(content).hexdigest(),
    )

    assert destination.read_bytes() == content
    assert not destination.with_suffix(".gz.part").exists()


def test_download_removes_partial_file_after_checksum_mismatch(tmp_path: Path, monkeypatch):
    destination = tmp_path / "source.tar.gz"
    monkeypatch.setattr(corresponding_sources, "urlopen", lambda _url: io.BytesIO(b"tampered"))

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        corresponding_sources._download(
            "https://example.invalid/source.tar.gz",
            destination,
            "0" * 64,
        )

    assert not destination.exists()
    assert not destination.with_suffix(".gz.part").exists()


def test_pypi_source_uses_the_only_sdist_and_its_digest(monkeypatch):
    metadata = {
        "urls": [
            {"packagetype": "bdist_wheel", "filename": "package.whl", "url": "https://example.invalid/wheel"},
            {
                "packagetype": "sdist",
                "filename": "package-1.2.3.tar.gz",
                "url": "https://example.invalid/source",
                "digests": {"sha256": "a" * 64},
            },
        ]
    }
    monkeypatch.setattr(
        corresponding_sources,
        "urlopen",
        lambda _url: io.BytesIO(json.dumps(metadata).encode("utf-8")),
    )

    source = corresponding_sources._pypi_source("package", "1.2.3")

    assert source == corresponding_sources.SourceArchive(
        component="package",
        filename="package-1.2.3.tar.gz",
        url="https://example.invalid/source",
        expected_sha256="a" * 64,
    )


def test_pypi_source_rejects_missing_or_ambiguous_sdists(monkeypatch):
    for urls, count in (([], 0), ([{"packagetype": "sdist"}, {"packagetype": "sdist"}], 2)):
        monkeypatch.setattr(
            corresponding_sources,
            "urlopen",
            lambda _url, payload={"urls": urls}: io.BytesIO(json.dumps(payload).encode("utf-8")),
        )
        with pytest.raises(RuntimeError, match=rf"found {count}"):
            corresponding_sources._pypi_source("package", "1.2.3")

    metadata = {
        "urls": [
            {
                "packagetype": "sdist",
                "filename": "package-1.2.3.tar.gz",
                "url": "https://example.invalid/source",
                "digests": {},
            }
        ]
    }
    monkeypatch.setattr(
        corresponding_sources,
        "urlopen",
        lambda _url: io.BytesIO(json.dumps(metadata).encode("utf-8")),
    )
    with pytest.raises(RuntimeError, match="did not provide a SHA-256 digest"):
        corresponding_sources._pypi_source("package", "1.2.3")


def test_git_source_requires_clean_tagged_revision(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(corresponding_sources, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(corresponding_sources, "__version__", "1.2.3")
    responses = iter(
        [
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout="v1.2.3\n"),
        ]
    )

    def fake_run(command, **_kwargs):
        if command[:2] == ["git", "archive"]:
            output_arg = next(part for part in command if part.startswith("--output="))
            Path(output_arg.removeprefix("--output=")).write_bytes(b"project source")
            return SimpleNamespace(stdout="")
        return next(responses)

    monkeypatch.setattr(corresponding_sources.subprocess, "run", fake_run)
    destination = tmp_path / "Cellonaut-1.2.3-source.zip"

    record = corresponding_sources._git_source(destination)

    assert record["tag"] == "v1.2.3"
    assert record["commit"] == "abc123"
    assert record["sha256"] == sha256(b"project source").hexdigest()
    assert record["bytes"] == len(b"project source")


@pytest.mark.parametrize(
    ("status", "tags", "message"),
    [
        (" M README.md\n", "v1.2.3\n", "clean, committed checkout"),
        ("", "v1.2.2\n", "release tag v1.2.3"),
    ],
)
def test_git_source_rejects_dirty_or_untagged_checkout(tmp_path: Path, monkeypatch, status, tags, message):
    monkeypatch.setattr(corresponding_sources, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(corresponding_sources, "__version__", "1.2.3")
    responses = iter([SimpleNamespace(stdout="abc123\n"), SimpleNamespace(stdout=status), SimpleNamespace(stdout=tags)])
    monkeypatch.setattr(corresponding_sources.subprocess, "run", lambda *_args, **_kwargs: next(responses))

    with pytest.raises(RuntimeError, match=message):
        corresponding_sources._git_source(tmp_path / "source.zip")


def test_prepare_writes_manifest_readme_and_all_sources_to_archive(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "SOURCE_AVAILABILITY.md").write_text("source instructions\n", encoding="utf-8")
    monkeypatch.setattr(corresponding_sources, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(corresponding_sources, "__version__", "1.2.3")
    monkeypatch.setattr(
        corresponding_sources,
        "DIRECT_SOURCES",
        (
            corresponding_sources.SourceArchive(
                component="Example",
                filename="example-source.tar.gz",
                url="https://example.invalid/source",
                expected_sha256="f" * 64,
            ),
        ),
    )
    monkeypatch.setattr(corresponding_sources, "PYPI_SOURCES", ())
    monkeypatch.setattr(corresponding_sources, "GIT_SOURCES", ())

    def fake_git_source(destination: Path):
        destination.write_bytes(b"cellonaut")
        return {"component": "Cellonaut", "filename": destination.name, "sha256": "1" * 64}

    def fake_download(_url: str, destination: Path, _expected: str):
        destination.write_bytes(b"dependency")
        destination.with_suffix(destination.suffix + ".part").write_bytes(b"ignored partial")

    monkeypatch.setattr(corresponding_sources, "_git_source", fake_git_source)
    monkeypatch.setattr(corresponding_sources, "_download", fake_download)

    archive_path = corresponding_sources.prepare_corresponding_sources(tmp_path / "output")

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == [
            "Cellonaut-1.2.3-source.zip",
            "example-source.tar.gz",
            "MANIFEST.json",
            "README.md",
        ]
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["cellonaut_version"] == "1.2.3"
        assert [record["component"] for record in manifest["files"]] == ["Cellonaut", "Example"]
        assert archive.read("README.md").decode("utf-8").splitlines() == ["source instructions"]


def test_corresponding_source_cli_prints_created_archive(tmp_path: Path, monkeypatch, capsys):
    expected = tmp_path / "Cellonaut-corresponding-sources.zip"
    monkeypatch.setattr(corresponding_sources, "prepare_corresponding_sources", lambda output: output / expected.name)

    assert corresponding_sources.main(["prepare", "--output", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == str(expected)
