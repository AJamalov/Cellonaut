from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest

from cellonaut import app
from cellonaut.release_checks import build_provenance, checksum, smoke
from cellonaut.version import __version__


def test_write_and_verify_clean_build_provenance(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(build_provenance, "git_state", lambda _root: ("abc123", False))
    output = tmp_path / "BUILD_INFO.json"

    written = build_provenance.write_build_info(output, tmp_path, "standard")
    verified = build_provenance.verify_build_info(output, tmp_path)

    assert verified == written
    assert written["app_version"] == __version__
    assert written["build_profile"] == "standard"
    assert written["source_revision"] == "abc123"
    assert written["source_dirty"] is False
    assert written["python_version"] == platform.python_version()
    assert written["built_utc"].endswith("+00:00")


def test_build_provenance_rejects_dirty_or_stale_release(tmp_path: Path, monkeypatch):
    output = tmp_path / "BUILD_INFO.json"
    monkeypatch.setattr(build_provenance, "git_state", lambda _root: ("abc123", True))

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        build_provenance.write_build_info(output, tmp_path, "standard")

    build_provenance.write_build_info(output, tmp_path, "standard", allow_dirty=True)
    monkeypatch.setattr(build_provenance, "git_state", lambda _root: ("different", False))
    with pytest.raises(RuntimeError, match="different Git revision.*uncommitted source"):
        build_provenance.verify_build_info(output, tmp_path)


def test_build_provenance_rejects_a_different_python_runtime(tmp_path: Path, monkeypatch):
    output = tmp_path / "BUILD_INFO.json"
    monkeypatch.setattr(build_provenance, "git_state", lambda _root: ("abc123", False))
    build_provenance.write_build_info(output, tmp_path, "standard")
    info = json.loads(output.read_text(encoding="utf-8"))
    info["python_version"] = "0.0.0"
    output.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(RuntimeError, match="different Python version"):
        build_provenance.verify_build_info(output, tmp_path)


def test_checksum_writes_standard_sha256_sidecar(tmp_path: Path):
    artifact = tmp_path / "Cellonaut.zip"
    artifact.write_bytes(b"release artifact")

    sidecar = checksum.write_sha256(artifact)

    expected = hashlib.sha256(b"release artifact").hexdigest()
    assert sidecar == tmp_path / "Cellonaut.zip.sha256"
    assert sidecar.read_text(encoding="ascii") == f"{expected}  Cellonaut.zip\n"
    assert sidecar.read_bytes() == f"{expected}  Cellonaut.zip\n".encode("ascii")


def test_checksum_writes_sorted_multipart_manifest(tmp_path: Path):
    exe = tmp_path / "Cellonaut-installer.exe"
    bin_part = tmp_path / "Cellonaut-installer-1.bin"
    exe.write_bytes(b"installer")
    bin_part.write_bytes(b"payload")
    manifest = tmp_path / "Cellonaut-installer.sha256"

    output = checksum.write_sha256_manifest([exe, bin_part], manifest)

    assert output == manifest
    assert manifest.read_text(encoding="ascii").splitlines() == [
        f"{hashlib.sha256(b'payload').hexdigest()}  Cellonaut-installer-1.bin",
        f"{hashlib.sha256(b'installer').hexdigest()}  Cellonaut-installer.exe",
    ]
    assert b"\r" not in manifest.read_bytes()
    assert manifest.read_bytes().endswith(b"\n")


def test_checksum_cli_expands_multipart_pattern_without_hashing_manifest(tmp_path: Path):
    (tmp_path / "Cellonaut-installer.exe").write_bytes(b"installer")
    (tmp_path / "Cellonaut-installer-1.bin").write_bytes(b"payload")
    manifest = tmp_path / "Cellonaut-installer.sha256"
    manifest.write_text("stale", encoding="ascii")

    exit_code = checksum.main(
        [
            "--input-pattern",
            str(tmp_path / "Cellonaut-installer*"),
            "--output",
            str(manifest),
        ]
    )

    assert exit_code == 0
    text = manifest.read_text(encoding="ascii")
    assert "Cellonaut-installer.exe" in text
    assert "Cellonaut-installer-1.bin" in text
    assert "Cellonaut-installer.sha256" not in text


def test_packaged_release_smoke_writes_version_and_failures(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def fake_import_modules(names: list[str]) -> list[str]:
        calls.append(names)
        return ["broken: unavailable"] if names is smoke.HEAVY_IMPORTS else []

    monkeypatch.setattr(smoke, "import_modules", fake_import_modules)
    monkeypatch.setattr(smoke, "validate_packaged_offline_assets", lambda: [])
    output = tmp_path / "smoke.json"

    exit_code = app._run_release_smoke(output)

    assert exit_code == 1
    assert calls == [smoke.BASE_IMPORTS, smoke.HEAVY_IMPORTS]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "app_version": __version__,
        "failures": ["broken: unavailable"],
    }


def test_packaged_offline_asset_check_validates_runtime_paths(tmp_path: Path, monkeypatch):
    offline_root = tmp_path / "offline"
    fiji_path = offline_root / "Fiji.app"
    model_root = offline_root / "cellpose_models"
    fiji_path.mkdir(parents=True)
    model_root.mkdir()
    models = {}
    for name in ("cpsam", "cpsam_v2"):
        model_path = model_root / name
        model_path.write_bytes(name.encode("ascii"))
        models[name] = {
            "bytes": model_path.stat().st_size,
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
    (offline_root / "OFFLINE_ASSETS.json").write_text(
        json.dumps({"offline": True, "cellpose_models": models}),
        encoding="utf-8",
    )

    ready_status = type("Status", (), {"ready": True, "missing_required": ()})()
    monkeypatch.setenv("CELLONAUT_OFFLINE", "1")
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(model_root))
    monkeypatch.setattr("cellonaut.resources.get_bundled_fiji_path", lambda: fiji_path)
    monkeypatch.setattr(
        "cellonaut.resources.get_resource_path",
        lambda _path: offline_root / "OFFLINE_ASSETS.json",
    )
    monkeypatch.setattr("cellonaut.io.fiji_installation.scan_fiji_installation", lambda _path: ready_status)

    assert smoke.validate_packaged_offline_assets() == []

    model_path = model_root / "cpsam"
    model_path.write_bytes(b"X" * model_path.stat().st_size)
    assert any("SHA-256 does not match manifest (cpsam)" in failure for failure in smoke.validate_packaged_offline_assets())


def test_packaged_offline_asset_check_reports_missing_payloads(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CELLONAUT_OFFLINE", "1")
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path / "models"))
    monkeypatch.setattr("cellonaut.resources.get_bundled_fiji_path", lambda: None)
    monkeypatch.setattr("cellonaut.resources.get_resource_path", lambda _path: tmp_path / "missing.json")

    failures = smoke.validate_packaged_offline_assets()

    assert any("Fiji.app could not be resolved" in failure for failure in failures)
    assert any("missing manifest" in failure for failure in failures)
    assert any("cpsam" in failure for failure in failures)


def test_packaged_runtime_check_initializes_fiji_and_each_cellpose_model(tmp_path: Path, monkeypatch):
    calls: list[object] = []
    fake_fiji = tmp_path / "Fiji.app"
    fake_fiji.mkdir()

    class FakeImageJ:
        pass

    monkeypatch.setattr("cellonaut.resources.get_bundled_fiji_path", lambda: fake_fiji)
    monkeypatch.setattr(
        "cellonaut.io.imagej_runtime.get_ij",
        lambda path, log_func: calls.append(path) or FakeImageJ(),
    )
    monkeypatch.setattr("cellonaut.config.defaults.CELLPOSE_MODEL_OPTIONS", ("cpsam", "cpsam_v2"))
    monkeypatch.setattr(
        "cellonaut.cell_segmentation.core.get_cellpose_model",
        lambda config: calls.append(config.model_type),
    )
    monkeypatch.setattr(
        "cellonaut.cell_segmentation.core.clear_cellpose_model_cache",
        lambda: calls.append("clear"),
    )
    monkeypatch.setattr(smoke, "validate_windows_cuda_inference", lambda: ("no_compatible_gpu", []))
    output = tmp_path / "runtime.json"

    exit_code = app._run_release_runtime_check(output)

    assert exit_code == 0
    assert calls == [fake_fiji, "cpsam", "clear", "cpsam_v2", "clear"]
    assert json.loads(output.read_text(encoding="utf-8"))["failures"] == []
    assert json.loads(output.read_text(encoding="utf-8"))["cuda_inference"] == "no_compatible_gpu"
