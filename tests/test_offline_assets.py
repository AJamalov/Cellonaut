from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cellonaut.release_checks import offline_assets
from cellonaut.cell_segmentation.runtime import cellpose_model_runtime_error, configure_local_cellpose_models


def test_fiji_archive_url_uses_pinned_windows_archive():
    assert offline_assets.fiji_archive_url().endswith("fiji-latest-win64-jdk.zip")


def test_verify_offline_assets_requires_every_model(tmp_path: Path, monkeypatch):
    offline_root = tmp_path / "offline"
    (offline_root / "Fiji.app").mkdir(parents=True)
    models = offline_root / "cellpose_models"
    models.mkdir()
    monkeypatch.setattr(
        offline_assets,
        "scan_fiji_installation",
        lambda _path: SimpleNamespace(ready=True, missing_required=()),
    )
    monkeypatch.setattr(offline_assets, "MINIMUM_MODEL_BYTES", 1)
    monkeypatch.setattr(offline_assets, "PYIMAGEJ_BRIDGE_ARTIFACTS", {})
    for name in offline_assets.CELLPOSE_MODEL_NAMES[:-1]:
        (models / name).write_bytes(name.encode("ascii"))

    with pytest.raises(RuntimeError, match=offline_assets.CELLPOSE_MODEL_NAMES[-1]):
        offline_assets.verify_offline_assets(offline_root)

    (models / offline_assets.CELLPOSE_MODEL_NAMES[-1]).write_bytes(b"model")
    manifest = offline_assets.verify_offline_assets(offline_root)

    assert manifest["offline"] is True
    models_manifest = manifest["cellpose_models"]
    assert isinstance(models_manifest, dict)
    assert set(models_manifest) == set(offline_assets.CELLPOSE_MODEL_NAMES)


def test_verify_offline_assets_rejects_removed_dino_models(tmp_path: Path, monkeypatch):
    offline_root = tmp_path / "offline"
    (offline_root / "Fiji.app").mkdir(parents=True)
    models = offline_root / "cellpose_models"
    models.mkdir()
    for name in offline_assets.CELLPOSE_MODEL_NAMES:
        (models / name).write_bytes(b"model")
    (models / "cpdino").write_bytes(b"removed model")
    monkeypatch.setattr(
        offline_assets,
        "scan_fiji_installation",
        lambda _path: SimpleNamespace(ready=True, missing_required=()),
    )
    monkeypatch.setattr(offline_assets, "MINIMUM_MODEL_BYTES", 1)
    monkeypatch.setattr(offline_assets, "PYIMAGEJ_BRIDGE_ARTIFACTS", {})

    with pytest.raises(RuntimeError, match="must not be included"):
        offline_assets.verify_offline_assets(offline_root)


def test_prepare_cellpose_models_removes_stale_dino_models(tmp_path: Path, monkeypatch):
    offline_root = tmp_path / "offline"
    models = offline_root / "cellpose_models"
    models.mkdir(parents=True)
    for name in offline_assets.REMOVED_CELLPOSE_MODEL_NAMES:
        (models / name).write_bytes(b"stale model")
    monkeypatch.setattr(offline_assets, "MINIMUM_MODEL_BYTES", 1)
    monkeypatch.setattr(
        offline_assets,
        "_download_verified_model",
        lambda name, destination, _downloads: destination.write_bytes(name.encode("ascii")),
    )

    offline_assets.prepare_cellpose_models(offline_root, tmp_path / "downloads")

    assert all(not (models / name).exists() for name in offline_assets.REMOVED_CELLPOSE_MODEL_NAMES)
    assert all((models / name).is_file() for name in offline_assets.CELLPOSE_MODEL_NAMES)


def test_safe_extract_zip_rejects_parent_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.txt", "unsafe")

    with pytest.raises(RuntimeError, match="Unsafe archive member"):
        offline_assets._safe_extract_zip(archive, tmp_path / "extract")


def test_release_bundle_removes_separately_installed_imagescience(tmp_path: Path):
    fiji_path = tmp_path / "Fiji.app"
    image_science_jar = fiji_path / "jars" / "imagescience.jar"
    image_science_jar.parent.mkdir(parents=True)
    with zipfile.ZipFile(image_science_jar, "w") as output:
        output.writestr(offline_assets.IMAGE_SCIENCE_CLASS, b"class")

    offline_assets._remove_imagescience_from_release_bundle(fiji_path)

    assert not image_science_jar.exists()


def test_pyimagej_bridge_uses_artifact_authoritative_repositories():
    imglyb_url, _imglyb_sha = offline_assets.PYIMAGEJ_BRIDGE_ARTIFACTS[
        "imglib2-imglyb-1.1.0.jar"
    ]
    unsafe_url, _unsafe_sha = offline_assets.PYIMAGEJ_BRIDGE_ARTIFACTS[
        "imglib2-unsafe-1.0.0.jar"
    ]

    assert imglyb_url.startswith("https://maven.imagej.net/content/repositories/releases/")
    assert unsafe_url.startswith("https://repo.maven.apache.org/maven2/")
    assert imglyb_url.endswith("/imglib2-imglyb-1.1.0.jar")
    assert unsafe_url.endswith("/imglib2-unsafe-1.0.0.jar")


def test_install_pyimagej_bridge_copies_verified_jars(tmp_path: Path, monkeypatch):
    fiji_path = tmp_path / "Fiji.app"
    downloads = tmp_path / "downloads"
    payload = b"pinned bridge"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        offline_assets,
        "PYIMAGEJ_BRIDGE_ARTIFACTS",
        {"bridge.jar": ("https://example.invalid/bridge.jar", digest)},
    )
    monkeypatch.setattr(
        offline_assets,
        "_download",
        lambda _url, destination: (destination.parent.mkdir(parents=True, exist_ok=True), destination.write_bytes(payload)),
    )

    offline_assets._install_pyimagej_bridge(fiji_path, downloads)

    assert (fiji_path / "jars" / "bridge.jar").read_bytes() == payload
    assert (fiji_path / "licenses" / "imglib2-python-bridge-BSD-2-Clause.txt").is_file()


def test_install_pyimagej_bridge_reuses_verified_local_maven_cache(tmp_path: Path, monkeypatch):
    fiji_path = tmp_path / "Fiji.app"
    downloads = tmp_path / "downloads"
    fake_home = tmp_path / "home"
    filename = "bridge.jar"
    relative_cache_path = Path("example/bridge/1.0/bridge.jar")
    payload = b"verified cached bridge"
    digest = hashlib.sha256(payload).hexdigest()
    cached = fake_home / ".m2" / "repository" / relative_cache_path
    cached.parent.mkdir(parents=True)
    cached.write_bytes(payload)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(
        offline_assets,
        "PYIMAGEJ_BRIDGE_ARTIFACTS",
        {filename: ("https://expired.example.invalid/bridge.jar", digest)},
    )
    monkeypatch.setattr(
        offline_assets,
        "PYIMAGEJ_BRIDGE_MAVEN_PATHS",
        {filename: relative_cache_path},
    )
    download_calls = []

    def cached_download(_url, destination):
        download_calls.append(destination)
        assert destination.is_file()

    monkeypatch.setattr(offline_assets, "_download", cached_download)

    offline_assets._install_pyimagej_bridge(fiji_path, downloads)

    assert (downloads / filename).read_bytes() == payload
    assert (fiji_path / "jars" / filename).read_bytes() == payload
    assert download_calls == [downloads / filename]


@pytest.mark.parametrize("folder_name", ["Fiji", "Fiji.app"])
def test_prepare_fiji_accepts_official_archive_folder_names(
    tmp_path: Path, monkeypatch, folder_name: str
):
    offline_root = tmp_path / "offline"
    downloads_root = tmp_path / "downloads"
    archive = downloads_root / "fiji-latest-win64-jdk.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"{folder_name}/fiji-windows-x64.exe", b"")

    monkeypatch.setattr(offline_assets, "_download_verified_archive", lambda *_args: None)
    monkeypatch.setattr(
        offline_assets,
        "fiji_archive_url",
        lambda: "https://example.invalid/fiji-latest-win64-jdk.zip",
    )
    monkeypatch.setattr(offline_assets, "_run_fiji_updater", lambda _path: None)
    monkeypatch.setattr(offline_assets, "_install_pyimagej_bridge", lambda *_args: None)
    monkeypatch.setattr(
        offline_assets,
        "scan_fiji_installation",
        lambda path: SimpleNamespace(ready=path.name in {folder_name, "Fiji.app"}),
    )

    prepared = offline_assets.prepare_fiji(offline_root, downloads_root)

    assert prepared == offline_root / "Fiji.app"
    assert (prepared / "fiji-windows-x64.exe").is_file()


def test_offline_cellpose_runtime_refuses_missing_weight(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CELLONAUT_OFFLINE", "1")
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path))

    error = cellpose_model_runtime_error("cpsam_v2")

    assert error is not None
    assert "complete official offline package" in error


def test_source_run_uses_prepared_offline_cellpose_models(tmp_path: Path, monkeypatch):
    model_root = tmp_path / ".local" / "release_assets" / "offline" / "cellpose_models"
    model_root.mkdir(parents=True)
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", "")
    monkeypatch.setattr("cellonaut.cell_segmentation.runtime.SOURCE_ROOT", tmp_path)

    configured = configure_local_cellpose_models()

    assert configured == model_root.resolve()
    assert os.environ["CELLPOSE_LOCAL_MODELS_PATH"] == str(model_root.resolve())
